"""For each sublayer (input=src, output=dst) in a toml ModelConfig, samples
points from the local tangent ball at each real point of src's activation
cloud (as saved by scripts.python.generate_tangent_bases) and runs them through the
real ONNX sublayer, saving (sampled input, real output) pairs -- the
tangent-space analogue of scripts.python.generate_ambient_sweep, which instead adds
isotropic noise directly in ambient space with no local structure.

--sample-idx takes one or more samples; --samples-per-chunk groups them
purely for progress-reporting/organization (default 1). Each sample's
tangent-space sampling is fully independent -- its own tangent bases (see
scripts.python.generate_tangent_bases), own context activations
(base_seqs[sample_idx]) -- so grouping never changes what gets computed.

--radius also takes one or more values: every (sample, sublayer)'s already-
loaded tangent bases and context activations are reused across all requested
radii (only the ball-sampling step and the ONNX forward pass re-run per
radius), and each radius gets its own output layer_key so runs at different
radii can complete/resume/push independently, same as --alpha in
scripts.python.run_autoencoder_sampled_sweep.

Mechanism (per sample, per sublayer, per radius, per position `pos`):
  1. Anchor: draw n indices uniformly (with replacement) from src's
     (basis, center) rows at position `pos`, as saved by
     scripts.python.generate_tangent_bases to local scratch for that sample.
  2. Sample: draw a point uniformly in the d_model-dimensional ball of
     radius `radius` in that anchor's tangent basis, then map it back to
     ambient space and add it to the anchor's center (same
     sample_ball_in_tangent_basis_batch_gpu used by
     scripts.python.generate_tangent_manifold_distances).
  3. Context: base_seqs[sample_idx] (that sample's fixed sequence, used
     elsewhere in this sweep family too) is run once per (sample, sublayer)
     through ensure_extra_outputs to get REAL src-activations at every
     position.
  4. Splice: each sampled point replaces position `pos`'s value in that
     real src tensor -- every other position keeps its real activation.
  5. Run the full (1, seq, d_model) spliced tensor through the {src}.{dst}
     subgraph (ensure_subgraphs) and read dst at position `pos`.

Output: one h5 file per (sample, sublayer, radius) --
{src}.{dst}.radius{r}.sample{idx}.h5 (see execute.sharded_writer.ShardedLayerWriter
and CLAUDE.md's per-sublayer resilience pattern), pushed as soon as that
(sample, sublayer, radius) finishes. Each row holds the sampled src vector,
the real dst vector, the position, and the anchor index (which real point's
tangent ball the sample was drawn from); multiplicity is always 1 (no
ball-membership concept applies here) and is kept only for schema parity
with run_sampled_sweep's/generate_ambient_sweep's h5 files.
"""

import argparse
import os
import time

import h5py
import numpy as np
import onnxruntime as ort
import torch

from autoencoder.data import sweep_dir_for
from config import ModelConfig, load_config
from construct.construct import ensure_base_model, ensure_extra_outputs, ensure_subgraphs
from execute.remote_storage import AsyncPusher as HetznerAsyncPusher
from execute.sharded_writer import Backend, ShardedLayerWriter, discover_layer_names, load_shard_rows
from scripts.python.generate_tangent_manifold_distances import sample_ball_in_tangent_basis_batch_gpu

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


def load_tangent_bases(tangent_bases_dir: str, layer: str, sample_idx: int) -> dict[str, np.ndarray]:
    """Reads scripts.python.generate_tangent_bases' saved (basis, center, pos,
    vocab_id) rows for one layer. Written with no_push=True, so
    load_shard_rows resolves every shard straight off local scratch (no
    remote pull) -- this must run on the same node the bases were computed
    on. `sample_idx` must match the --sample-idx generate_tangent_bases was
    run with (it's baked into the saved layer_key)."""
    layer_key = os.path.join(tangent_bases_dir, f"{layer}.sample{sample_idx}")
    return load_shard_rows(".", layer_key, dataset_names=["basis", "center", "pos", "vocab_id"])


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


def tangent_sample(
    basis: np.ndarray, center: np.ndarray, n: int, radius: float, generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """basis: (n_points, D, D), center: (n_points, D) -- src's saved tangent
    bases/centers at one position. Draws n anchor indices uniformly with
    replacement and samples one point in each anchor's tangent ball.
    Returns (points (n, D), anchor indices (n,))."""
    n_points = basis.shape[0]
    anchor = torch.randint(0, n_points, (n,), generator=generator, device=device)
    basis_t = torch.from_numpy(basis).to(device)[anchor]  # (n, D, D)
    center_t = torch.from_numpy(center).to(device)[anchor]  # (n, D)
    sampled = sample_ball_in_tangent_basis_batch_gpu(
        basis_t, center_t, radius, n_samples=1, generator=generator,
    )  # (n, 1, D)
    return sampled.squeeze(1).detach().cpu().numpy().astype(np.float32), anchor.cpu().numpy().astype(np.int64)


def run_sublayer(
    cfg: ModelConfig,
    sweep_dir: str,
    src: str,
    dst: str,
    sample_idx: int,
    base_ids: np.ndarray,
    tangent_bases_dir: str,
    n: int,
    radii: list[float],
    seed: int,
    out_dir: str,
    flush_every: int,
    no_push: bool,
    backend: Backend,
    pusher: object | None,
) -> None:
    seq = cfg.seq
    d_model = cfg.d_model

    print(f"[{src}->{dst}] loading tangent bases for {src}", flush=True)
    bases = load_tangent_bases(tangent_bases_dir, src, sample_idx)  # basis: (n_rows, D, D), center/pos/vocab_id: (n_rows, ...)

    print(f"[{src}->{dst}] getting real context activations for sample_idx={sample_idx}", flush=True)
    ctx = context_activations(cfg, src, base_ids)  # (seq, d_model), real values at every position

    map_fn = make_subgraph_map(cfg, src, dst)

    for radius in radii:
        generator = torch.Generator(device=device).manual_seed(seed)

        layer_key = os.path.join(out_dir, f"{src}.{dst}.radius{radius:g}.sample{sample_idx}")
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
            attrs=dict(sample_idx=sample_idx, n_samples=n, seq=seq, d_model=d_model, radius=radius),
            pusher=pusher,
        )

        if writer.next_row >= writer.n_rows:  # already complete from an earlier run
            writer.close()
            continue

        row = writer.next_row
        for pos in range(seq):
            at_pos = bases["pos"] == pos
            pts, anchor = tangent_sample(
                bases["basis"][at_pos], bases["center"][at_pos], n, radius, generator,
            )  # (n, d_model), (n,) -- anchor indexes into this pos's rows

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
            print(f"[{src}->{dst}] radius={radius:g} pos={pos}: sampled {n} tangent-space points", flush=True)

        writer.close()
        print(f"[{src}->{dst}] radius={radius:g} done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--tangent-bases-dir", required=True,
                   help="--out-dir used by scripts.python.generate_tangent_bases (must be on this node's local scratch)")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="one or more base sequences' real activations to splice into")
    p.add_argument("--samples-per-chunk", type=int, default=1,
                   help="how many samples to process per progress-reporting group in one process -- purely "
                        "an overhead/organization knob, same as generate_tangent_bases.py's flag: each "
                        "sample's tangent-space sampling is always independent (its own tangent bases, own "
                        "context activations), so this never changes what gets computed")
    p.add_argument("--layers", nargs="*", default=None, help="default: every layer in sweep_dir")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="only run these src.dst sublayers (default: every cfg.sublayers entry whose input is in --layers)")
    p.add_argument("--n", type=int, default=1000, help="samples per (sublayer, position)")
    p.add_argument("--radius", type=float, nargs="+", default=[0.1],
                   help="one or more radii of the tangent-space ball to sample from -- each gets its own "
                        "output layer_key ({src}.{dst}.radius{r}.sample{idx}), sharing this (sample, "
                        "sublayer)'s already-loaded tangent bases/context activations across radii instead "
                        "of reloading them per radius (same grouping benefit --samples-per-chunk gives "
                        "across samples)")
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
    for group in groups:
        print(f"\n=== samples {group} ===", flush=True)
        for sample_idx in group:
            base_ids = load_base_seq(sweep_dir, sample_idx)
            for src, dst in sublayers:
                t0 = time.time()
                run_sublayer(
                    cfg, sweep_dir, src, dst, sample_idx, base_ids, args.tangent_bases_dir,
                    n=args.n, radii=args.radius, seed=args.seed,
                    out_dir=args.out_dir, flush_every=args.flush_every,
                    no_push=args.no_push, backend=args.backend, pusher=pusher,
                )
                print(f"[sample {sample_idx}] [{src}->{dst}] took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
