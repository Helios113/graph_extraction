"""Trains one batched_autoencoder across every (sample, sublayer, pos) cell
of one or more samples from a scripts.python.generate_perturbed_sweep output
directory.

Takes a toml ModelConfig (the same one used to produce the sweep via
scripts.python.generate_perturbed_sweep); sweep_dir is derived from it as
onnx_models/{cfg.name}/perturbed_sweep (see autoencoder.data.sweep_dir_for).
For each sublayer, each --sample-idx value selects one sample's
(seq, n_vocab, d_model) slice of that sublayer's vec dataset (row order is
(sample_idx, pos, vocab_idx), so this is a plain reshape+index, no filtering
-- see scripts/generate_perturbed_sweep.build_chunk_rows). Sublayers and
positions are stacked into one (n_sublayers * seq, n_vocab, d_model) tensor
per sample, and every sample's tensor is concatenated into one pool before
chunking (--n-models-per-chunk), so a single batched GPU training call can
span several samples' models at once -- each row is still checkpointed to
its own sample's directory (autoencoder.data.ae_checkpoints_dir_for) with
its own local (sublayer, pos) numbering, so results are identical to
training each sample separately, just processed together on GPU. Passing a
single --sample-idx value (the default) reduces to training one sample,
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
from autoencoder.data import ae_checkpoints_dir_for, load_sweep_data, sweep_dir_for
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
    model_indices: torch.Tensor | None = None,
    checkpoint_every: int = 0,
    out_dir: str | list[str] | None = None,
    sample_idx: int | list[int] | None = None,
    verbose: bool = True,
) -> tuple[batched_autoencoder, dict]:
    """Core training logic (data already loaded): splits `data` (n_models,
    n_points, d_model) into train/val, fits a batched_autoencoder, and
    returns (model, metrics) with metrics holding final train/val recon and
    geom loss (each (n_models,), per model row) -- used directly by both
    __main__ and scripts/python/hparam_search.py.

    `model_indices`: row i's own (local, per-sample) index -- defaults to
    plain `range(n_models)` (i.e. this call's row 0 is global index 0) when
    None, the single-sample case. Pass an explicit (n_models,) tensor when
    `data` mixes rows from more than one sample (see main()'s multi-sample
    --sample-idx handling): each row's train/val split uses its own
    independent seed (seed + model_indices[row]), matching
    batched_autoencoder.reattach exactly, so reattach is correct later
    regardless of how training was chunked or how many samples were mixed
    into one chunk.

    `checkpoint_every`: if > 0 (and `out_dir` given), save_all() the model to
    `out_dir` every this many epochs, overwriting the previous checkpoint --
    so a crash mid-training still leaves the most recently completed
    checkpoint's weights on disk instead of losing the whole run.

    `out_dir`/`sample_idx`: forwarded straight to save_all() (see its
    docstring) for the periodic checkpoints written during training -- may
    be per-row lists when `data` mixes samples, exactly as save_all accepts;
    has no effect on training itself."""
    d_model = data.shape[-1]
    n_models = data.shape[0]
    n_points = data.shape[1]
    if model_indices is None:
        model_indices = torch.arange(n_models)

    # ------------------------------------------------------------ train/val split
    # Same per-row-seeded recipe batched_autoencoder.reattach uses to re-derive
    # `states` later: each model row gets its own independent Generator, not a
    # shared stream -- see reattach's docstring for why.
    n_train_rows = round(n_points * (1 - val_frac))
    perms = torch.stack([
        torch.randperm(n_points, generator=torch.Generator().manual_seed(seed + int(model_indices[row])))
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
            save_all(bmnfld, out_dir, index_offset=model_indices.tolist(), sample_idx=sample_idx)
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
    bmnfld.model_indices = model_indices

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
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="one or more samples to train on. Multiple values are concatenated into one pool "
                        "before chunking (see --n-models-per-chunk), so a single batched GPU training call "
                        "can span several samples' (layer, pos) models at once -- each row still gets "
                        "checkpointed to its own sample's directory with its own local numbering, so results "
                        "are identical to running each sample separately, just processed together on GPU")
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
                   help="train at most this many (sample, layer, pos) models at once, in separate sequential "
                        "train() calls covering the full multi-sample pool -- caps GPU memory regardless of "
                        "how many layers/positions/samples are combined, while still covering every one of "
                        "them; a chunk may span more than one sample's models when --sample-idx has multiple "
                        "values and n-models-per-chunk doesn't divide evenly along sample boundaries")
    p.add_argument("--out-dir", default=None,
                   help="root dir for save_all() checkpoints; each sample's models are written under "
                        "{out-dir}/sample{idx} (default: autoencoder.data.ae_checkpoints_dir_for(cfg, idx) "
                        "per sample, i.e. $NEW_JAC_SCRATCH/onnx_models/{cfg.name}/ae_checkpoints/sample{idx})")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="save_all() checkpoints to each sample's out-dir every this many epochs during "
                        "training (overwriting the previous periodic checkpoint), so a crash mid-training "
                        "still leaves usable weights on disk. 0 disables periodic checkpointing (only the "
                        "final checkpoint after each chunk completes is saved)")
    p.add_argument("--backend", choices=["hetzner", "nfs"], default="hetzner",
                   help="transport load_sweep_data pulls any not-yet-local sweep shard through -- 'hetzner' "
                        "(default) via execute.remote_storage, 'nfs' via execute.node_relay's scratch_staging/ "
                        "relay (e.g. when this run is training on ruapehu against sweep data relayed in from "
                        "mauao via scripts/helper/sync_node_scratch.sh + node_relay, rather than data pushed "
                        "to the Hetzner store). Must match how the sweep data being read was actually pushed.")
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    print(f"sweep_dir: {sweep_dir}")
    full_layers = discover_layer_names(sweep_dir)
    layers = args.layers or full_layers
    print(f"layers ({len(layers)} of {len(full_layers)}): {layers}")

    def sample_out_dir(sample_idx: int) -> str:
        return os.path.join(args.out_dir, f"sample{sample_idx}") if args.out_dir else ae_checkpoints_dir_for(cfg, sample_idx)

    # Each row's checkpoint is keyed by a GLOBAL index (layer_idx * seq + pos,
    # layer_idx = this layer's position in the FULL sweep_dir layer list, not
    # just in `layers`) rather than a position local to this run's own
    # `--layers` subset -- otherwise two runs with disjoint --layers into the
    # same --out-dir both produce model_0.pt, model_1.pt, ... and silently
    # clobber each other. This is the write-side of the scheme
    # run_autoencoder_sampled_sweep.py already assumes on the read side (see
    # its model_indices = model_row_offset + layer_idx * seq + pos).
    seq = cfg.seq
    print(f"loading sweep data for samples {args.sample_idx}", flush=True)
    a = time.time()
    pools, row_sample_idx, row_local_index = [], [], []
    for s in args.sample_idx:
        d = load_sweep_data(sweep_dir, layers, s, backend=args.backend)  # (n_layers * seq, n_vocab, d_model)
        pools.append(d)
        row_sample_idx += [s] * d.shape[0]
        row_local_index += [
            full_layers.index(layer) * seq + pos for layer in layers for pos in range(seq)
        ]
    data = torch.cat(pools, dim=0)
    row_sample_idx = torch.tensor(row_sample_idx)
    row_local_index = torch.tensor(row_local_index)
    print(f"loaded {data.shape} ({len(args.sample_idx)} sample(s)) in {time.time() - a:.1f}s", flush=True)

    n_models = data.shape[0]
    chunk_size = args.n_models_per_chunk
    n_chunks = math.ceil(n_models / chunk_size)
    print(f"training {n_models} models in {n_chunks} chunk(s) of <= {chunk_size}", flush=True)

    for c in range(n_chunks):
        offset = c * chunk_size
        chunk_data = data[offset:offset + chunk_size]
        chunk_sample_idx = row_sample_idx[offset:offset + chunk_size]
        chunk_local_index = row_local_index[offset:offset + chunk_size]
        n_chunk_models = chunk_data.shape[0]
        print(f"\n=== chunk {c + 1}/{n_chunks}: models {offset}..{offset + n_chunk_models - 1} "
              f"(samples {sorted(set(chunk_sample_idx.tolist()))}) ===", flush=True)

        chunk_out_dirs = [sample_out_dir(s) for s in chunk_sample_idx.tolist()]
        done = [
            os.path.exists(os.path.join(chunk_out_dirs[i], f"model_{chunk_local_index[i].item()}.pt"))
            for i in range(n_chunk_models)
        ]
        if all(done):
            print(f"chunk {c + 1}/{n_chunks}: all {n_chunk_models} checkpoints already present, skipping", flush=True)
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
            model_indices=chunk_local_index,
            checkpoint_every=args.checkpoint_every,
            out_dir=chunk_out_dirs,
            sample_idx=chunk_sample_idx.tolist(),
        )

        save_all(
            bmnfld, chunk_out_dirs,
            index_offset=chunk_local_index.tolist(), sample_idx=chunk_sample_idx.tolist(),
        )
        print(f"saved chunk {c + 1}/{n_chunks} ({bmnfld.encoder.n_models} checkpoints) across "
              f"{len(set(chunk_out_dirs))} sample dir(s)")

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
