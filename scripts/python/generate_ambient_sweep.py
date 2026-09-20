"""For each sublayer (input=src, output=dst) in a toml ModelConfig, samples
points by adding isotropic Gaussian noise to real ambient-space activations
and runs them through the real ONNX sublayer, saving (sampled input, real
output) pairs -- the ambient-space analogue of
scripts.python.run_autoencoder_sampled_sweep, which instead samples near a fitted
autoencoder manifold. Unlike that script, this one has no dependency on a
trained autoencoder.

Mechanism (per sublayer, per position `pos`):
  1. Anchor: draw n indices uniformly (with replacement) from src's real
     (n_vocab, d_model) activation cloud at position `pos` (same
     autoencoder.data.load_sample_cloud pull used to fit the manifold
     sampler, but consumed directly -- no autoencoder/fit_stats involved).
  2. Sample: x = anchor_point + N(0, tau^2 I) in ambient (d_model) space.
  3. Context: base_seqs[--sample-idx] (the same fixed sequence used
     elsewhere in this sweep family) is run once through
     ensure_extra_outputs to get REAL src-activations at every position.
     Cached per sublayer since it doesn't depend on the sample.
  4. Splice: each sampled point replaces position `pos`'s value in that
     real src tensor -- every other position keeps its real activation.
  5. Run the full (1, seq, d_model) spliced tensor through the {src}.{dst}
     subgraph (ensure_subgraphs) and read dst at position `pos`.

Output: one h5 file per sublayer (see execute.sharded_writer.ShardedLayerWriter
and CLAUDE.md's per-sublayer resilience pattern), pushed as soon as that
sublayer finishes. Each row holds the sampled src vector, the real dst
vector, the position, and the anchor index (which real vocab-cloud point the
noise was added to); multiplicity is always 1 (no ball-membership concept
applies here) and is kept only for schema parity with run_sampled_sweep's
h5 files.
"""

import argparse
import os
import time

import h5py
import numpy as np
import onnxruntime as ort
import torch

from autoencoder.data import load_sample_cloud, sweep_dir_for
from config import ModelConfig, load_config
from construct.construct import ensure_base_model, ensure_extra_outputs, ensure_subgraphs
from execute.remote_storage import AsyncPusher
from execute.sharded_writer import ShardedLayerWriter, discover_layer_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SESSION_PROVIDERS = ["CUDAExecutionProvider"]


def _session_options() -> ort.SessionOptions:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.intra_op_num_threads = 8
    return opts


def _make_session(path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), sess_options=_session_options(), providers=_SESSION_PROVIDERS)


def load_base_seq(sweep_dir: str, sample_idx: int) -> np.ndarray:
    """base_seqs.h5 is a small, non-sharded NFS file written directly by
    scripts.python.generate_perturbed_sweep (out_dir/base_seqs.h5) -- unlike the
    layers' own vec data, it's never sharded/pushed, so this is a plain
    h5py read, no load_shard_rows/manifest resolution needed."""
    with h5py.File(os.path.join(sweep_dir, "base_seqs.h5"), "r") as f:
        return f["base_seqs"][sample_idx]  # (seq,) token ids


def context_activations(cfg: ModelConfig, src: str, token_ids: np.ndarray) -> np.ndarray:
    """Real src-activations at every position for one sequence, via
    ensure_extra_outputs (full-model forward, padded to the fixed exported
    batch dim). Returns (seq, d_model)."""
    ensure_base_model(cfg, force=False, verbose=False)
    extra_outputs_path = ensure_extra_outputs(cfg, [src], force=False)
    session = _make_session(extra_outputs_path)
    input_name = session.get_inputs()[0].name

    fixed_batch = cfg.batch_for_node()
    padded = np.zeros((fixed_batch, cfg.seq), dtype=np.int64)
    padded[0] = token_ids
    (arr,) = session.run([src], {input_name: padded})
    return np.asarray(arr[0])  # (seq, d_model)


def make_subgraph_map(cfg: ModelConfig, src: str, dst: str, batch_size: int | None = None):
    """map_fn(x) = the real sublayer src -> dst applied to a batch of full
    (B, seq, d_model) spliced state tensors (all positions real except the
    one replaced by a sample) -- unlike gmanifold.onnx_transformer.make_map,
    this feeds every position's real value through, not just position 0
    with the rest zeroed, so it's correct across attention too.

    The extracted subgraph's input has a fixed static batch dim
    (cfg.batch_for_node(), e.g. 32000, baked in at export time -- same as
    every other ONNX session in this repo), so map_fn pads each chunk up to
    that size rather than feeding it a smaller batch directly, which
    ONNXRuntime rejects with a dimension-mismatch error."""
    paths = ensure_subgraphs(cfg, force=False)
    [(path, _)] = [p for p in paths if p[1].input == src and p[1].output == dst]
    session = _make_session(path)
    input_name = session.get_inputs()[0].name

    fixed_batch = cfg.batch_for_node()
    chunk = min(batch_size or fixed_batch, fixed_batch)

    @torch.no_grad()
    def map_fn(x: np.ndarray) -> np.ndarray:
        """x: (B, seq, d_model) -> (B, seq, d_model) at dst."""
        B, seq, d_model = x.shape
        out = np.empty((B, seq, d_model), dtype=np.float32)
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            n_rows = e - s
            padded = np.zeros((fixed_batch, seq, d_model), dtype=np.float32)
            padded[:n_rows] = x[s:e]
            (arr,) = session.run([dst], {input_name: padded})
            out[s:e] = np.asarray(arr[:n_rows])
        return out

    return map_fn


def ambient_sample(
    cloud: np.ndarray, n: int, tau: float, generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """cloud: (n_vocab, d_model) real activations at one (layer, pos).
    Draws n anchor indices uniformly with replacement and adds isotropic
    N(0, tau^2 I) noise in ambient space. Returns (points (n, d_model),
    anchor indices (n,))."""
    n_vocab = cloud.shape[0]
    anchor = generator.integers(0, n_vocab, size=n)
    noise = generator.normal(scale=tau, size=(n, cloud.shape[1])).astype(np.float32)
    return cloud[anchor].astype(np.float32) + noise, anchor.astype(np.int64)


def run_sublayer(
    cfg: ModelConfig,
    sweep_dir: str,
    src: str,
    dst: str,
    sample_idx: int,
    base_ids: np.ndarray,
    n: int,
    tau: float,
    seed: int,
    out_dir: str,
    flush_every: int,
    no_push: bool,
    pusher: AsyncPusher | None,
) -> None:
    seq = cfg.seq
    d_model = cfg.d_model
    generator = np.random.default_rng(seed)

    print(f"[{src}->{dst}] loading real ambient cloud for {src}", flush=True)
    cloud = load_sample_cloud(sweep_dir, src, sample_idx).numpy()  # (seq, n_vocab, d_model)

    print(f"[{src}->{dst}] getting real context activations for sample_idx={sample_idx}", flush=True)
    ctx = context_activations(cfg, src, base_ids)  # (seq, d_model), real values at every position

    map_fn = make_subgraph_map(cfg, src, dst)

    layer_key = os.path.join(out_dir, f"{src}.{dst}")
    writer = ShardedLayerWriter(
        layer_key=layer_key,
        n_rows=seq * n,
        schema={
            "src_vec": ((d_model,), "float32"),
            "dst_vec": ((d_model,), "float32"),
            "pos": ((), "int64"),
            "anchor": ((), "int64"),
            "multiplicity": ((), "int64"),
        },
        shard_rows=max(seq * n, 1),
        flush_every=flush_every,
        no_push=no_push,
        attrs=dict(sample_idx=sample_idx, n_samples=n, seq=seq, d_model=d_model, tau=tau),
        pusher=pusher,
    )

    row = writer.next_row
    for pos in range(seq):
        pts, anchor = ambient_sample(cloud[pos], n, tau, generator)  # (n, d_model), (n,)

        spliced = np.broadcast_to(ctx, (n, seq, d_model)).copy()  # (n, seq, d_model)
        spliced[:, pos] = pts
        out = map_fn(spliced.astype(np.float32))  # (n, seq, d_model)
        dst_vecs = np.ascontiguousarray(out[:, pos])  # (n, d_model)

        writer.write_rows(
            row, row + n,
            src_vec=pts, dst_vec=dst_vecs,
            pos=np.full(n, pos, dtype=np.int64),
            anchor=anchor, multiplicity=np.ones(n, dtype=np.int64),
        )
        row += n
        print(f"[{src}->{dst}] pos={pos}: sampled {n} ambient points", flush=True)

    writer.close()
    print(f"[{src}->{dst}] done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, default=0, help="which base sequence's real activations to anchor/splice into")
    p.add_argument("--layers", nargs="*", default=None,
                   help="default: every layer in sweep_dir")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="only run these src.dst sublayers (default: every cfg.sublayers entry whose input is in --layers)")
    p.add_argument("--n", type=int, default=1000, help="samples per (sublayer, position)")
    p.add_argument("--tau", type=float, default=1.0, help="isotropic Gaussian noise stddev added to each anchor point")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--upload-workers", type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}", flush=True)

    base_ids = load_base_seq(sweep_dir, args.sample_idx)

    sublayers = [(s.input, s.output) for s in cfg.sublayers if s.input in layers]
    if args.sublayers is not None:
        wanted = set(args.sublayers)
        sublayers = [(src, dst) for src, dst in sublayers if f"{src}.{dst}" in wanted]
    print(f"{len(sublayers)} sublayer(s) to run: {sublayers}", flush=True)

    pusher = None if args.no_push else AsyncPusher(max_workers=args.upload_workers)

    for src, dst in sublayers:
        t0 = time.time()
        run_sublayer(
            cfg, sweep_dir, src, dst, args.sample_idx, base_ids,
            n=args.n, tau=args.tau, seed=args.seed,
            out_dir=args.out_dir, flush_every=args.flush_every,
            no_push=args.no_push, pusher=pusher,
        )
        print(f"[{src}->{dst}] took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
