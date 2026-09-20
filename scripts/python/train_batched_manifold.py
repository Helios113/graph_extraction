"""Trains one batched_autoencoder across every (sublayer, pos) cell of a
single sample from a scripts.python.generate_perturbed_sweep output directory.

Takes a toml ModelConfig (the same one used to produce the sweep via
scripts.python.generate_perturbed_sweep); sweep_dir is derived from it as
onnx_models/{cfg.name}/perturbed_sweep (see autoencoder.data.sweep_dir_for).
For each sublayer, --sample-idx selects one sample's (seq, n_vocab, d_model)
slice of that sublayer's vec dataset (row order is (sample_idx, pos,
vocab_idx), so this is a plain reshape+index, no filtering -- see
scripts/generate_perturbed_sweep.build_chunk_rows). Sublayers and positions
are then stacked into one (n_sublayers * seq, n_vocab, d_model) tensor and
trained as a single batched_autoencoder, one model per (sublayer, pos) pair,
data_size = n_vocab (~50k for GPT-2).

Adapted directly from autoencoder/test_script.py -- same knn/mem helpers and
training loop.
"""

import argparse
import math
import os
import time

import torch
import tqdm

from autoencoder.batched_autoencoder import batched_autoencoder, save_all
from autoencoder.data import load_sweep_data, sweep_dir_for
from config import load_config
from execute.sharded_writer import discover_layer_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mem(label):
    if device.type != "cuda":
        return
    alloc = torch.cuda.memory_allocated(device) / 2**30
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    free, total = (x / 2**30 for x in torch.cuda.mem_get_info(device))
    print(
        f"[mem] {label}: allocated={alloc:.2f}GiB peak={peak:.2f}GiB "
        f"reserved={reserved:.2f}GiB free={free:.2f}GiB total={total:.2f}GiB"
    )


@torch.no_grad()
def chunked_recon_loss(bmnfld: batched_autoencoder, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Mean squared reconstruction error per model row, computed in
    chunk_size-sized pieces along the point dimension instead of one
    unchunked bmnfld.project(x) call over all of x at once -- project()'s
    output (and the encoder's hidden activation) scale with x's full point
    count, which for a whole train/val set is far bigger than any tensor the
    minibatch training loop ever allocates (this is what was OOMing:
    verified against the crash logs -- e.g. a (100, 45231, 768) float32
    x_recon is exactly the reported "Tried to allocate 12.94 GiB")."""
    n_models, n_points = x.shape[0], x.shape[1]
    sq_err_sum = torch.zeros(n_models, device=x.device)
    for s in range(0, n_points, chunk_size):
        chunk = x[:, s:s + chunk_size]
        recon = bmnfld.project(chunk)
        sq_err_sum += ((recon - chunk) ** 2).sum(-1).sum(-1)
    return sq_err_sum / n_points


@torch.no_grad()
def knn(x, k, query_set=None, chunk=4096, model_chunk=None, device=None, progress=False):
    """x, query_set: CPU tensors. Each (model_chunk, chunk) slice is moved to
    `device` only for the duration of its cdist/topk call."""

    self_q = query_set is None
    query_set = x if self_q else query_set
    data_size = x.shape[1]

    kk = min(k + int(self_q), data_size)
    n_models, q_size = query_set.shape[0], query_set.shape[1]
    model_chunk = model_chunk or n_models
    ds = torch.empty(n_models, q_size, k, dtype=x.dtype)
    js = torch.empty(n_models, q_size, k, dtype=torch.long)
    model_steps = range(0, n_models, model_chunk)
    if progress:
        model_steps = tqdm.tqdm(model_steps)
    for m in model_steps:
        x_m = x[m:m + model_chunk].to(device)
        for s in range(0, q_size, chunk):
            d = torch.cdist(query_set[m:m + model_chunk, s:s + chunk].to(device), x_m)
            if self_q:
                b = torch.arange(d.shape[0], device=d.device)[:, None]
                r = torch.arange(d.shape[1], device=d.device)[None, :]
                d[b, r, s + r] = float("inf")
            dd, jj = d.topk(kk, largest=False, dim=-1)
            ds[m:m + model_chunk, s:s + chunk] = (dd[..., :k] if self_q else dd).cpu()
            js[m:m + model_chunk, s:s + chunk] = (jj[..., :k] if self_q else jj).cpu()
    return ds, js


def train(
    data: torch.Tensor,
    network_dims: list[int],
    k: int = 32,
    knn_chunk: int = 512,
    knn_model_chunk: int = 8,
    epochs: int = 300,
    batch_size: int = 96,
    lr: float = 2e-3,
    lam_geom: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 0,
    model_index_offset: int = 0,
    checkpoint_every: int = 0,
    out_dir: str | None = None,
    verbose: bool = True,
) -> tuple[batched_autoencoder, dict]:
    """Core training logic (data already loaded): splits `data` (n_models,
    n_points, d_model) into train/val, fits a batched_autoencoder, and
    returns (model, metrics) with metrics holding final train/val recon and
    geom loss (each (n_models,), per model row) -- used directly by both
    __main__ and scripts/python/hparam_search.py.

    `model_index_offset`: this call's row 0 is model_index_offset globally --
    nonzero when `data` is one chunk of a larger run (see main()'s
    --n-models-per-chunk). Each row's train/val split uses its own
    independent seed (seed + model_index_offset + row), matching
    batched_autoencoder.reattach exactly, so reattach is correct later
    regardless of how training was chunked.

    `checkpoint_every`: if > 0 (and `out_dir` given), save_all() the model to
    `out_dir` every this many epochs, overwriting the previous checkpoint --
    so a crash mid-training still leaves the most recently completed
    checkpoint's weights on disk instead of losing the whole run."""
    d_model = data.shape[-1]
    n_models = data.shape[0]
    n_points = data.shape[1]

    # ------------------------------------------------------------ train/val split
    # Same per-row-seeded recipe batched_autoencoder.reattach uses to re-derive
    # `states` later: each model row gets its own independent Generator, not a
    # shared stream -- see reattach's docstring for why.
    n_train_rows = round(n_points * (1 - val_frac))
    perms = torch.stack([
        torch.randperm(n_points, generator=torch.Generator().manual_seed(seed + model_index_offset + row))
        for row in range(n_models)
    ])
    train_idx, val_idx = perms[:, :n_train_rows], perms[:, n_train_rows:]

    model_arange_full = torch.arange(n_models)[:, None]
    train_data = data[model_arange_full, train_idx]  # (n_models, n_train_rows, d_model)
    val_data = data[model_arange_full, val_idx]  # (n_models, n_val_rows, d_model)
    data_size = train_data.shape[1]
    if verbose:
        print(f"train/val split: {train_data.shape[1]} / {val_data.shape[1]} points per model")

    center = torch.mean(train_data, (1), keepdim=True)
    scale = ((train_data - center) ** 2).sum(2, keepdim=True).mean(1, keepdim=True).sqrt()

    if verbose:
        mem("before knn")
    dist, neighbor_idx = knn(
        train_data, k, progress=verbose, chunk=knn_chunk, model_chunk=knn_model_chunk, device=device,
    )
    if verbose:
        mem("after knn")

    bmnfld = batched_autoencoder(
        center.to(device), scale.to(device), n_models, network_dims=[d_model, *network_dims], n_neighbors=k,
    ).to(device)
    if verbose:
        mem("after bmnfld construction")

    # train_data/neighbor_idx are small enough to keep resident on GPU (e.g. ~12GiB
    # for 96 GPT-2 models x 45k train rows x 768 dims) -- moved once here so every
    # minibatch step below indexes/gathers directly on GPU instead of doing a CPU
    # gather + host-to-device copy on every one of the (data_size/batch_size)*epochs steps.
    train_data = train_data.to(device)
    neighbor_idx = neighbor_idx.to(device)
    val_data = val_data.to(device)
    if verbose:
        mem("after moving train_data/neighbor_idx/val_data to device")

    # ---------------------------------------------------------------- training

    opt = torch.optim.Adam(bmnfld.parameters(), lr=lr)

    model_arange = torch.arange(n_models, device=device)[:, None, None]
    epoch_iter = tqdm.tqdm(range(epochs), desc="fit") if verbose else range(epochs)

    for ep in epoch_iter:
        a = time.time()

        perm = torch.stack([torch.randperm(data_size, device=device) for _ in range(n_models)])
        step_iter = tqdm.tqdm(range(0, data_size, batch_size)) if verbose else range(0, data_size, batch_size)

        for s in step_iter:
            idx = perm[:, s:s + batch_size]  # (n_models, B)
            nbr_idx = neighbor_idx[model_arange[:, :, 0], idx]  # (n_models, B, k)

            x = train_data[model_arange[:, :, 0], idx]  # (n_models, B, d_model)
            neighbors = train_data[model_arange, nbr_idx]  # (n_models, B, k, d_model)

            z = bmnfld.encode(x)  # (n_models, B, latent)
            z_neighbors = bmnfld.encode(
                neighbors.reshape(n_models, -1, neighbors.shape[-1])
            ).reshape(n_models, x.shape[1], bmnfld.n_neighbors, bmnfld.latent_dim)

            x_recon = bmnfld.decode(z)

            ambient_dist = (neighbors - x[:, :, None]).norm(dim=-1)
            latent_dist = (z_neighbors - z[:, :, None]).norm(dim=-1)

            recon_loss = ((x_recon - x) ** 2).sum(-1).mean()
            geom_loss = ((latent_dist - ambient_dist) ** 2).mean() / (ambient_dist ** 2).mean()

            loss = recon_loss + lam_geom * geom_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        epoch_val_recon_loss = chunked_recon_loss(bmnfld, val_data, batch_size).mean().item()

        if verbose:
            print(
                f"epoch {ep} time: {time.time() - a:.1f}s  recon: {recon_loss.item():.4f}  "
                f"geom: {geom_loss.item():.4f}  val_recon: {epoch_val_recon_loss:.4f}"
            )
            mem(f"end of epoch {ep}")

        if checkpoint_every and out_dir and (ep + 1) % checkpoint_every == 0:
            bmnfld.states = train_data
            bmnfld.seed = seed
            bmnfld.n_train_rows = n_train_rows
            bmnfld.model_index_offset = model_index_offset
            save_all(bmnfld, out_dir)
            print(f"checkpoint: saved epoch {ep + 1} to {out_dir}", flush=True)

    # ---------------------------------------------------------------- train/val metrics

    val_dist, val_neighbor_idx = knn(
        val_data, k, chunk=knn_chunk, model_chunk=knn_model_chunk, device=device,
    )
    val_neighbor_idx = val_neighbor_idx.to(device)
    with torch.no_grad():
        train_recon_loss = chunked_recon_loss(bmnfld, train_data, batch_size)  # (n_models,)
        val_recon_loss = chunked_recon_loss(bmnfld, val_data, batch_size)  # (n_models,)

        # Chunked geom-loss pass, same reason as chunked_recon_loss: val_z_neighbors'
        # unchunked shape (n_models, n_val, n_neighbors, latent_dim) scales with the
        # full val set, not a training-loop-sized minibatch.
        n_val = val_data.shape[1]
        val_sq_diff_sum = torch.zeros(n_models, device=device)
        val_ambient_sq_sum = torch.zeros(n_models, device=device)
        for vs in range(0, n_val, batch_size):
            v_x = val_data[:, vs:vs + batch_size]
            v_nbr_idx = val_neighbor_idx[model_arange[:, :, 0], vs + torch.arange(v_x.shape[1], device=device)]
            v_neighbors = val_data[model_arange, v_nbr_idx]  # (n_models, b, k, d_model)
            v_z = bmnfld.encode(v_x)
            v_z_neighbors = bmnfld.encode(
                v_neighbors.reshape(n_models, -1, v_neighbors.shape[-1])
            ).reshape(n_models, v_x.shape[1], bmnfld.n_neighbors, bmnfld.latent_dim)
            v_ambient_dist = (v_neighbors - v_x[:, :, None]).norm(dim=-1)
            v_latent_dist = (v_z_neighbors - v_z[:, :, None]).norm(dim=-1)
            val_sq_diff_sum += ((v_latent_dist - v_ambient_dist) ** 2).sum((-1, -2))
            val_ambient_sq_sum += (v_ambient_dist ** 2).sum((-1, -2))
        val_geom_loss = val_sq_diff_sum / val_ambient_sq_sum  # (n_models,)
    if verbose:
        print(f"val recon (mean over models): {val_recon_loss.mean().item():.4f}  val geom: {val_geom_loss.mean().item():.4f}")

    # states = the train subset actually fit on; not saved directly (see save_all),
    # only the seed/n_train_rows recipe needed to re-derive it later via reattach()
    bmnfld.states = train_data
    bmnfld.seed = seed
    bmnfld.n_train_rows = n_train_rows
    bmnfld.model_index_offset = model_index_offset

    metrics = dict(
        train_recon_loss=train_recon_loss.detach().cpu(),
        val_recon_loss=val_recon_loss.detach().cpu(),
        val_geom_loss=val_geom_loss.detach().cpu(),
    )
    return bmnfld, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to a toml ModelConfig -- sweep_dir is derived as "
                   "onnx_models/{cfg.name}/perturbed_sweep")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--layers", nargs="*", default=None, help="defaults to every layer in sweep_dir")
    p.add_argument("--network-dims", type=int, nargs="*", default=[512, 32])
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--knn-chunk", type=int, default=512)
    p.add_argument("--knn-model-chunk", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lam-geom", type=float, default=0.1)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-models-per-chunk", type=int, default=100,
                   help="train at most this many (layer, pos) models at once, in separate sequential "
                        "train() calls covering the full data -- caps GPU memory regardless of how many "
                        "layers/positions the sweep has, while still covering every one of them")
    p.add_argument("--out-dir", default=None, help="if set, save_all() checkpoints here after training")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="save_all() checkpoints to --out-dir every this many epochs during training "
                        "(overwriting the previous periodic checkpoint), so a crash mid-training still "
                        "leaves usable weights on disk. 0 disables periodic checkpointing (only the "
                        "final checkpoint after training completes is saved). No-op without --out-dir")
    args = p.parse_args()

    sweep_dir = sweep_dir_for(load_config(args.config))
    print(f"sweep_dir: {sweep_dir}")
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}")

    print("loading sweep data", flush=True)
    a = time.time()
    data = load_sweep_data(sweep_dir, layers, args.sample_idx)  # (n_layers * seq, n_vocab, d_model)
    print(f"loaded {data.shape} in {time.time() - a:.1f}s", flush=True)

    n_models = data.shape[0]
    chunk_size = args.n_models_per_chunk
    n_chunks = math.ceil(n_models / chunk_size)
    print(f"training {n_models} models in {n_chunks} chunk(s) of <= {chunk_size}", flush=True)

    for c in range(n_chunks):
        offset = c * chunk_size
        chunk_data = data[offset:offset + chunk_size]
        n_chunk_models = chunk_data.shape[0]
        print(f"\n=== chunk {c + 1}/{n_chunks}: models {offset}..{offset + n_chunk_models - 1} ===", flush=True)

        if args.out_dir and all(
            os.path.exists(os.path.join(args.out_dir, f"model_{offset + i}.pt")) for i in range(n_chunk_models)
        ):
            print(f"chunk {c + 1}/{n_chunks}: all {n_chunk_models} checkpoints already in {args.out_dir}, skipping", flush=True)
            continue

        bmnfld, metrics = train(
            chunk_data,
            network_dims=args.network_dims,
            k=args.k,
            knn_chunk=args.knn_chunk,
            knn_model_chunk=args.knn_model_chunk,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lam_geom=args.lam_geom,
            val_frac=args.val_frac,
            seed=args.seed,
            model_index_offset=offset,
            checkpoint_every=args.checkpoint_every,
            out_dir=args.out_dir,
        )

        if args.out_dir:
            save_all(bmnfld, args.out_dir)
            print(f"saved chunk {c + 1}/{n_chunks} ({bmnfld.encoder.n_models} checkpoints) to {args.out_dir}")

        # bmnfld.states and train()'s local train_data/val_data/neighbor_idx are GPU-resident
        # (see train()'s docstring) and otherwise stay referenced via bmnfld into the next
        # chunk's iteration -- drop them and empty the allocator's cache now, or the next
        # chunk's own train_data/val_data/neighbor_idx allocation OOMs against memory this
        # chunk never released (verified: this is what was OOMing on chunk 2/2 above).
        del bmnfld, metrics
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
