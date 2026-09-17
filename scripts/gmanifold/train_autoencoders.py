

import argparse
import json
import os
import time

import torch

from execute.remote_storage import push
from execute.sharded_writer import _scratch_root, discover_layer_names, load_shard_rows
from gmanifold import GlobalManifold, fit, save_weights
import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _weights_path(out_dir: str, layer: str, sample_idx: int, pos: int) -> str:
    return os.path.join(out_dir, f"{layer}.{sample_idx}.{pos}.pt")


def load_layer_vecs(
    activations_dir: str, layer: str, pull_dir: str | None
) -> torch.Tensor:
    """Pulls every row of `layer`'s "vec" dataset (ensure_local under the
    hood, from local scratch or the remote store) and reshapes it to
    (n_samples, seq, n_vocab, d_model) -- row order is unravelled from
    (sample_idx, pos, vocab_idx) at generation time (see
    generate_perturbed_sweep.build_chunk_rows), and n_samples/seq/d_model
    come from the shard manifest's own attrs rather than a config file, so
    this matches whatever was actually written even if the originating
    config has since changed. n_vocab is derived rather than read from attrs
    since it isn't stored directly (it may be cfg.vocab_size or a
    --n-vocab-sample subsample -- either way n_rows / (n_samples * seq)
    recovers it).

    `pull_dir`: pull missing shards here (e.g. node-local scratch) instead
    of onto NFS -- only a small subsample is actually used for training, so
    landing the full multi-GB shard permanently on shared NFS storage would
    be wasted space; see load_shard_rows."""
    layer_key = os.path.join(activations_dir, layer)
    with open(layer_key + ".manifest.json") as f:
        manifest = json.load(f)
    n_samples = manifest["attrs"]["n_samples"]
    seq = manifest["attrs"]["seq"]
    d_model = manifest["attrs"]["d_model"]
    n_rows = manifest["n_rows"]
    n_vocab = n_rows // (n_samples * seq)

    vecs = load_shard_rows(
        ".", layer_key, dataset_names=["vec"], pull_dir=pull_dir,
    )["vec"]

    return torch.from_numpy(vecs).reshape(n_samples, seq, n_vocab, d_model)


def train_and_save_cell(
    vecs_cell: torch.Tensor,
    out_dir: str,
    layer: str,
    sample_idx: int,
    pos: int,
    latent_dim: int,
    max_train_rows: int,
    epochs: int,
    seed: int,
    no_push: bool,
) -> str:
    """Fits one GlobalManifold on a single (sample_idx, pos) cell's rows
    (shape (n_vocab, d_model) -- never mixed with any other cell) and saves
    its weights. Returns the val_recon_over_spacing for logging."""
    X = vecs_cell
    if len(X) > max_train_rows:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(X), generator=g)[:max_train_rows]
        X = X[idx]

    # Save 10% of data for validation
    n_val = max(len(X) // 10, 1)

    # permutation of all vectors
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))

    # training and validation split
    tr_i, va_i = perm[:-n_val], perm[-n_val:]

    # Move data to GPU
    Xtr, Xva = X[tr_i].to(DEVICE), X[va_i].to(DEVICE)

    # Create model
    model = fit(
        GlobalManifold(latent_dim=latent_dim, device=DEVICE),
        Xtr, epochs=epochs, X_val=Xva, seed=seed,
    )

    os.makedirs(out_dir, exist_ok=True)
    path = _weights_path(out_dir, layer, sample_idx, pos)
    save_weights(model, path)
    if not no_push:
        push(path)

    val_recon = model.history[-1]["val_recon_over_spacing"]
    print(
        f"{layer}[{sample_idx},{pos}]: n_train={len(Xtr)} "
        f"val_recon_over_spacing={val_recon:.4f} -> {path}",
        flush=True,
    )
    return path


def train_and_save_layer(
    activations_dir: str,
    out_dir: str,
    layer: str,
    latent_dim: int,
    max_train_rows: int,
    epochs: int,
    seed: int,
    no_push: bool,
    force: bool,
    pull_dir: str | None,
) -> None:
    """Fits and saves one GlobalManifold per (sample_idx, pos) cell in
    `layer`, skipping cells whose weights file already exists (unless
    `force`) -- so a run that dies partway through a layer resumes at the
    next untrained cell instead of redoing the whole layer."""
    print("Generating samples", flush=True)
    vecs = load_layer_vecs(activations_dir, layer, pull_dir)
    print("Generated samples", flush=True)
    
    n_samples, seq, _, _ = vecs.shape

    for sample_idx in tqdm.tqdm(range(n_samples)):
        for pos in range(seq):
            path = _weights_path(out_dir, layer, sample_idx, pos)
            if os.path.exists(path) and not force:
                continue
            train_and_save_cell(
                vecs[sample_idx, pos],
                out_dir,
                layer,
                sample_idx,
                pos,
                latent_dim=latent_dim,
                max_train_rows=max_train_rows,
                epochs=epochs,
                seed=seed,
                no_push=no_push,
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--activations-dir",
        required=True,
        help="perturbed_sweep output dir (contains one *.manifest.json per layer)",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="One weights .pt file is written here per (layer, sample_idx, pos) cell",
    )
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=50_000,
        help="Subsample cap -- fit()'s kNN scan is only tractable at this scale",
    )
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--layers",
        nargs="+",
        default=None,
        help="Train only these layers (default: every complete layer found)",
    )
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Retrain a (sample_idx, pos) cell even if its weights file already exists",
    )
    ap.add_argument(
        "--pull-dir",
        default="__scratch__",
        help="Directory to pull missing shards into (default: node-local scratch, via "
        "execute.sharded_writer's _scratch_root/$NEW_JAC_SCRATCH -- only a small subsample "
        "of each layer is actually used for training, so pulling full multi-GB shards onto "
        "shared NFS would be wasted space). Pass '' to pull onto NFS under the project root instead.",
    )
    args = ap.parse_args()

    pull_dir = (
        _scratch_root() if args.pull_dir == "__scratch__" else (args.pull_dir or None)
    )
    print("Getting data", flush=True)
    all_layers = discover_layer_names(args.activations_dir)
    complete_layers = []
    for layer in all_layers:
        manifest_path = os.path.join(args.activations_dir, f"{layer}.manifest.json")
        with open(manifest_path) as f:
            if json.load(f)["complete"]:
                complete_layers.append(layer)
            else:
                print(f"skipping {layer}: manifest not complete", flush=True)

    layers = args.layers if args.layers is not None else complete_layers
    unknown = set(layers) - set(all_layers)
    if unknown:
        raise ValueError(
            f"--layers {sorted(unknown)} not found under {args.activations_dir} (available: {all_layers})"
        )
    print("Running Training", flush=True)
    T0 = time.time()
    for layer in layers:
        train_and_save_layer(
            args.activations_dir,
            args.out_dir,
            layer,
            latent_dim=args.latent_dim,
            max_train_rows=args.max_train_rows,
            epochs=args.epochs,
            seed=args.seed,
            no_push=args.no_push,
            force=args.force,
            pull_dir=pull_dir,
        )
        print(f"[{time.time() - T0:6.0f}s] {layer}: done", flush=True)

    print(f"done in {time.time() - T0:.1f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
