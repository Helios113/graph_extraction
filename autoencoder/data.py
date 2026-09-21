"""Loads point clouds for autoencoder/batched_autoencoder training and
reattach() from a scripts.python.generate_perturbed_sweep output directory. Shared
by scripts/python/train_batched_manifold.py, sample_autoencoder.py,
validate_autoencoder.py, and hparam_search.py so all agree on exactly which activations a given
(sweep_dir, layers, sample_idx) selects."""

import json
import os
from typing import Literal

import h5py
import numpy as np
import torch

from config import ModelConfig
from execute.sharded_writer import _scratch_root

Backend = Literal["hetzner", "nfs"]


def sweep_dir_for(cfg: ModelConfig) -> str:
    """The scripts.python.generate_perturbed_sweep output dir for `cfg`, following
    the same onnx_models/{cfg.name}/perturbed_sweep convention its --out-dir
    default uses. This is the virtual/NFS-relative layer_key namespace;
    load_sample_cloud's pull_dir=_scratch_root() is what makes the actual
    bytes get checked on node-local scratch first."""
    return os.path.join("onnx_models", cfg.name, "perturbed_sweep")


def ae_checkpoints_dir_for(cfg: ModelConfig, sample_idx: int) -> str:
    """The train_batched_manifold.py save_all() output dir for `cfg`/`sample_idx`,
    following the {_scratch_root()}/onnx_models/{cfg.name}/ae_checkpoints/sample{sample_idx}
    convention scripts/batch_scripts/train_batched_manifold_mauao.sh passes as
    --out-dir (train_batched_manifold.py's own --out-dir has no default; the
    launcher supplies this path). The sample_idx suffix lives on the directory
    rather than the model_{i}.pt filenames themselves, since save_all/load_batched
    key off those filenames (model index, not sample) to reassemble a
    batched_autoencoder. Unlike sweep_dir_for, this is scratch-only
    (CLAUDE.md's "local scratch, not NFS" -- checkpoints are never pushed to
    NFS or the remote store), so the scratch root is baked in rather than left
    as a virtual/NFS-relative path."""
    return os.path.join(_scratch_root(), "onnx_models", cfg.name, "ae_checkpoints", f"sample{sample_idx}")


def n_samples_for(sweep_dir: str, layer: str) -> int:
    """n_samples the perturbed sweep was generated with, per that layer's
    manifest (every layer shares the same n_samples -- see
    generate_perturbed_sweep.py's attrs)."""
    with open(os.path.join(sweep_dir, layer) + ".manifest.json") as f:
        return json.load(f)["attrs"]["n_samples"]


def _load_row_range(
    layer_key: str, row_start: int, row_end: int, dataset_name: str, backend: Backend = "nfs",
) -> np.ndarray:
    """Reads only rows [row_start, row_end) of `dataset_name` for `layer_key`,
    touching only the shards that actually overlap that range (pulling each
    from local scratch first, then NFS staging -- see
    execute.node_relay.ensure_local) instead of load_shard_rows'
    whole-layer read. Rows are contiguous across shards in shard_idx order,
    same as load_shard_rows produces.

    `backend`: must match whatever the writer that produced this layer used
    (see execute.sharded_writer.ShardedLayerWriter.backend) -- "nfs"
    (default) pulls from NFS staging via execute.node_relay, "hetzner" pulls
    from the SFTP store (only relevant for older layers pushed there before
    this project moved to the NFS relay). The caller must already be
    running on the node it wants shards pulled onto."""
    if backend == "hetzner":
        from execute.remote_storage import ensure_local
    elif backend == "nfs":
        from execute.node_relay import ensure_local
    else:
        raise ValueError(f"unknown backend {backend!r} -- expected 'hetzner' or 'nfs'")

    with open(layer_key + ".manifest.json") as f:
        manifest = json.load(f)

    out = np.empty((row_end - row_start, *manifest["schema"][dataset_name][0]), dtype=manifest["schema"][dataset_name][1])
    for shard in sorted(manifest["shards"], key=lambda s: s["shard_idx"]):
        s_start, s_end = shard["row_start"], shard["row_end"]
        if s_end <= row_start or s_start >= row_end:
            continue  # shard doesn't overlap the wanted range
        if shard["pushed"]:
            local_path = ensure_local(shard["virtual_path"], dest_dir=_scratch_root())
        else:
            local_path = os.path.join(_scratch_root(), layer_key, f"shard_{shard['shard_idx']:05d}.h5")
        lo, hi = max(row_start, s_start), min(row_end, s_end)
        with h5py.File(local_path, "r") as h5f:
            out[lo - row_start:hi - row_start] = h5f[dataset_name][lo - s_start:hi - s_start]
    return out


def load_sample_cloud(sweep_dir: str, layer: str, sample_idx: int, backend: Backend = "nfs") -> torch.Tensor:
    """One sample's (seq, n_vocab, d_model) point cloud for `layer`, reading
    only the rows belonging to `sample_idx` (row order is (sample_idx, pos,
    vocab_idx)-ravelled, so that's one contiguous row range -- see
    scripts/python/generate_perturbed_sweep.build_chunk_rows) instead of pulling and
    reading the whole layer (every sample, e.g. ~40GB/layer for GPT-2) just
    to keep 1/n_samples of it."""
    layer_key = os.path.join(sweep_dir, layer)
    with open(layer_key + ".manifest.json") as f:
        manifest = json.load(f)
    n_samples = manifest["attrs"]["n_samples"]
    seq = manifest["attrs"]["seq"]
    d_model = manifest["attrs"]["d_model"]
    n_rows = manifest["n_rows"]
    n_vocab = n_rows // (n_samples * seq)

    row_start = sample_idx * seq * n_vocab
    row_end = row_start + seq * n_vocab
    vec = _load_row_range(layer_key, row_start, row_end, "vec", backend=backend)
    vec = torch.from_numpy(vec).reshape(seq, n_vocab, d_model)
    return vec


def load_sweep_data(
    sweep_dir: str, layers: list[str], sample_idx: int, backend: Backend = "nfs",
) -> torch.Tensor:
    """Stacks load_sample_cloud across `layers` into one (n_layers * seq,
    n_vocab, d_model) tensor -- one model per (layer, pos) pair, matching
    train_batched_manifold.py's batch layout exactly (so it's the correct
    full_activations pool to pass to batched_autoencoder.reattach)."""
    clouds = [load_sample_cloud(sweep_dir, layer, sample_idx, backend=backend) for layer in layers]
    return torch.cat(clouds, dim=0)
