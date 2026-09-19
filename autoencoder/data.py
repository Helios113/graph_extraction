"""Loads point clouds for autoencoder/batched_autoencoder training and
reattach() from a scripts.generate_perturbed_sweep output directory. Shared
by scripts/autoencoder/train_batched_manifold.py, sample.py, validate.py, and
hparam_search.py so all agree on exactly which activations a given
(sweep_dir, layers, sample_idx) selects."""

import json
import os

import h5py
import numpy as np
import torch

from config import ModelConfig
from execute.remote_storage import ensure_local
from execute.sharded_writer import _scratch_root


def sweep_dir_for(cfg: ModelConfig) -> str:
    """The scripts.generate_perturbed_sweep output dir for `cfg`, following
    the same onnx_models/{cfg.name}/perturbed_sweep convention its --out-dir
    default uses. This is the virtual/NFS-relative layer_key namespace;
    load_sample_cloud's pull_dir=_scratch_root() is what makes the actual
    bytes get checked on node-local scratch first."""
    return os.path.join("onnx_models", cfg.name, "perturbed_sweep")


def _load_row_range(layer_key: str, row_start: int, row_end: int, dataset_name: str) -> np.ndarray:
    """Reads only rows [row_start, row_end) of `dataset_name` for `layer_key`,
    touching only the shards that actually overlap that range (pulling each
    from local scratch first, then NFS, then remote -- see
    execute.remote_storage.ensure_local) instead of load_shard_rows'
    whole-layer read. Rows are contiguous across shards in shard_idx order,
    same as load_shard_rows produces."""
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


def load_sample_cloud(sweep_dir: str, layer: str, sample_idx: int) -> torch.Tensor:
    """One sample's (seq, n_vocab, d_model) point cloud for `layer`, reading
    only the rows belonging to `sample_idx` (row order is (sample_idx, pos,
    vocab_idx)-ravelled, so that's one contiguous row range -- see
    scripts/generate_perturbed_sweep.build_chunk_rows) instead of pulling and
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
    vec = _load_row_range(layer_key, row_start, row_end, "vec")
    vec = torch.from_numpy(vec).reshape(seq, n_vocab, d_model)
    return vec


def load_sweep_data(sweep_dir: str, layers: list[str], sample_idx: int) -> torch.Tensor:
    """Stacks load_sample_cloud across `layers` into one (n_layers * seq,
    n_vocab, d_model) tensor -- one model per (layer, pos) pair, matching
    train_batched_manifold.py's batch layout exactly (so it's the correct
    full_activations pool to pass to batched_autoencoder.reattach)."""
    clouds = [load_sample_cloud(sweep_dir, layer, sample_idx) for layer in layers]
    return torch.cat(clouds, dim=0)
