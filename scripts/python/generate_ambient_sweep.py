"""For each sublayer (input=src, output=dst) in a toml ModelConfig, samples
points by adding isotropic Gaussian noise to real ambient-space activations
and runs them through the real ONNX sublayer, saving (sampled input, real
output) pairs -- the ambient-space analogue of
scripts.python.run_autoencoder_sampled_sweep, which instead samples near a fitted
autoencoder manifold. Unlike that script, this one has no dependency on a
trained autoencoder.

--sample-idx takes one or more samples; --samples-per-chunk groups them
purely for progress-reporting/organization (default 1), same as
scripts.python.generate_tangent_sweep. Each sample's ambient cloud and
context activations are its own (they depend on sample_idx), but the
{src}.{dst} subgraph session (make_subgraph_map) is built once per sublayer
and reused across every sample and tau, since it doesn't depend on either.

--tau also takes one or more values: a (sample, sublayer)'s already-loaded
ambient cloud and context activations are reused across all requested taus
(only the noise-sampling step and the ONNX forward pass re-run per tau), and
each tau gets its own output layer_key so runs at different taus can
complete/resume/push independently, same as --alpha in
scripts.python.run_autoencoder_sampled_sweep (and --radius in
scripts.python.generate_tangent_sweep, this script's tangent-space analogue).

Mechanism (per sublayer, per sample, per tau, per position `pos`):
  1. Anchor: draw n indices uniformly (with replacement) from src's real
     (n_vocab, d_model) activation cloud at position `pos` (same
     autoencoder.data.load_sample_cloud pull used to fit the manifold
     sampler, but consumed directly -- no autoencoder/fit_stats involved).
  2. Sample: x = anchor_point + N(0, tau^2 I) in ambient (d_model) space.
  3. Context: base_seqs[sample_idx] (the same fixed sequence used elsewhere
     in this sweep family) is run once per (sample, sublayer) through
     ensure_extra_outputs to get REAL src-activations at every position.
  4. Splice: each sampled point replaces position `pos`'s value in that
     real src tensor -- every other position keeps its real activation.
  5. Run the full (1, seq, d_model) spliced tensor through the {src}.{dst}
     subgraph (ensure_subgraphs) and read dst at position `pos`.

Output: one h5 file per (sublayer, sample, tau) --
{src}.{dst}.tau{t}.sample{idx}.h5 (see execute.sharded_writer.ShardedLayerWriter
and CLAUDE.md's per-sublayer resilience pattern), pushed as soon as that
(sublayer, sample, tau) finishes. Each row holds the sampled src vector, the
real dst vector, the position, and the anchor index (which real vocab-cloud
point the noise was added to); multiplicity is always 1 (no ball-membership
concept applies here) and is kept only for schema parity with
run_sampled_sweep's h5 files.
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
from execute.remote_storage import AsyncPusher as HetznerAsyncPusher
from execute.sharded_writer import Backend, ShardedLayerWriter, discover_layer_names

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


def run_sample(
    cfg: ModelConfig,
    sweep_dir: str,
    src: str,
    dst: str,
    sample_idx: int,
    base_ids: np.ndarray,
    map_fn,
    n: int,
    taus: list[float],
    seed: int,
    out_dir: str,
    flush_every: int,
    no_push: bool,
    backend: Backend,
    pusher: object | None,
) -> None:
    """One (sample, sublayer)'s worth of work across every requested tau --
    `map_fn` (the {src}.{dst} subgraph session) is built once per sublayer
    by the caller and passed in, since it doesn't depend on sample_idx or
    tau; only the ambient cloud/context activations below are sample-
    specific and get reloaded per call."""
    seq = cfg.seq
    d_model = cfg.d_model

    print(f"[{src}->{dst}] sample {sample_idx}: loading real ambient cloud for {src}", flush=True)
    cloud = load_sample_cloud(sweep_dir, src, sample_idx).numpy()  # (seq, n_vocab, d_model)

    print(f"[{src}->{dst}] sample {sample_idx}: getting real context activations", flush=True)
    ctx = context_activations(cfg, src, base_ids)  # (seq, d_model), real values at every position

    for tau in taus:
        generator = np.random.default_rng(seed)

        layer_key = os.path.join(out_dir, f"{src}.{dst}.tau{tau:g}.sample{sample_idx}")
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
            backend=backend,
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
            print(f"[{src}->{dst}] sample {sample_idx} tau={tau:g} pos={pos}: sampled {n} ambient points",
                  flush=True)

        writer.close()
        print(f"[{src}->{dst}] sample {sample_idx} tau={tau:g} done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="one or more base sequences' real activations to anchor/splice into")
    p.add_argument("--samples-per-chunk", type=int, default=1,
                   help="how many samples to process per progress-reporting group -- purely an "
                        "overhead/organization knob, same as generate_tangent_sweep.py's flag: each "
                        "sample's ambient-cloud/context-activation loading is always independent, so this "
                        "never changes what gets computed, only how the run is grouped/logged")
    p.add_argument("--layers", nargs="*", default=None,
                   help="default: every layer in sweep_dir")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="only run these src.dst sublayers (default: every cfg.sublayers entry whose input is in --layers)")
    p.add_argument("--n", type=int, default=1000, help="samples per (sublayer, position)")
    p.add_argument("--tau", type=float, nargs="+", default=[1.0],
                   help="one or more isotropic Gaussian noise stddevs added to each anchor point -- each "
                        "gets its own output layer_key ({src}.{dst}.tau{t}.sample{idx}), sharing this "
                        "(sample, sublayer)'s already-loaded ambient cloud/context activations across taus "
                        "instead of reloading them per tau")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--upload-workers", type=int, default=4)
    p.add_argument("--backend", choices=["hetzner", "nfs"], default="nfs",
                   help="transport ShardedLayerWriter pushes each finished sublayer's shards through -- "
                        "'nfs' (default) via execute.node_relay's NFS-staging relay (e.g. when this run's "
                        "compute node isn't where the data should durably land, and a later step relays "
                        "it in via scripts/helper/sync_node_scratch.sh), 'hetzner' via "
                        "execute.remote_storage (not used by this project's own launchers anymore). "
                        "Ignored if --no-push is also set.")
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}", flush=True)

    sublayers = [(s.input, s.output) for s in cfg.sublayers if s.input in layers]
    if args.sublayers is not None:
        wanted = set(args.sublayers)
        sublayers = [(src, dst) for src, dst in sublayers if f"{src}.{dst}" in wanted]
    print(f"{len(sublayers)} sublayer(s) to run: {sublayers}", flush=True)

    pusher_cls = HetznerAsyncPusher
    if args.backend == "nfs":
        from execute.node_relay import AsyncPusher as NfsAsyncPusher
        pusher_cls = NfsAsyncPusher
    pusher = None if args.no_push else pusher_cls(max_workers=args.upload_workers)

    groups = [
        args.sample_idx[i:i + args.samples_per_chunk]
        for i in range(0, len(args.sample_idx), args.samples_per_chunk)
    ]

    for src, dst in sublayers:
        # map_fn (the {src}.{dst} subgraph session) doesn't depend on sample_idx or
        # tau, so it's built once per sublayer and reused across every sample/tau below.
        map_fn = make_subgraph_map(cfg, src, dst)

        for group in groups:
            print(f"\n=== [{src}->{dst}] samples {group} ===", flush=True)
            for sample_idx in group:
                t0 = time.time()
                base_ids = load_base_seq(sweep_dir, sample_idx)
                run_sample(
                    cfg, sweep_dir, src, dst, sample_idx, base_ids, map_fn,
                    n=args.n, taus=args.tau, seed=args.seed,
                    out_dir=args.out_dir, flush_every=args.flush_every,
                    no_push=args.no_push, backend=args.backend, pusher=pusher,
                )
                print(f"[{src}->{dst}] sample {sample_idx} took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
