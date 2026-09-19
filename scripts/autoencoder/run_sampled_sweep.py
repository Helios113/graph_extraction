"""For each sublayer (input=src, output=dst) in a toml ModelConfig, samples
points from src's trained batched_autoencoder and runs them through the
real ONNX sublayer, saving (sampled input, real output) pairs -- the
autoencoder-sampling analogue of scripts.generate_perturbed_sweep, which
instead perturbs at the vocab/token level.

Mechanism (per sublayer, per position `pos`):
  1. Sample n ambient points at src, position `pos`, from the trained
     batched_autoencoder loaded from --ae-checkpoints-dir (model_index =
     layers.index(src) * seq + pos, same (layer, pos) -> model_index mapping
     train_batched_manifold.py used when saving that checkpoint dir).
  2. Context: base_seqs[--sample-idx] (the same fixed sequence the
     autoencoder was trained on -- see sweep_dir/base_seqs.h5) is run once
     through ensure_extra_outputs to get REAL src-activations at every
     position. This is cached per sublayer since it doesn't depend on the
     sample.
  3. Splice: each sampled point replaces position `pos`'s value in that
     real src tensor -- every other position keeps its real activation.
  4. Run the full (1, seq, d_model) spliced tensor through the {src}.{dst}
     subgraph (ensure_subgraphs) and read dst at position `pos`. Since every
     other position already holds its real value, this is correct even for
     sublayers that cross attention (unlike gmanifold.onnx_transformer's
     make_map, which zero-pads every other position and so is only correct
     for strictly tokenwise sublayers).

Output: one h5 file per sublayer (see execute.sharded_writer.ShardedLayerWriter
and CLAUDE.md's per-sublayer resilience pattern), pushed as soon as that
sublayer finishes. Each row holds the sampled src vector, the real dst
vector, the position, the autoencoder's anchor index (which fitted state's
latent ball the sample was drawn from), and its multiplicity -- ess/tau
(per (sublayer, pos), not per row) are stored as manifest attrs.
"""

import argparse
import os
import time

import h5py
import numpy as np
import onnxruntime as ort
import torch

from autoencoder import sampling
from autoencoder.batched_autoencoder import load_batched
from autoencoder.data import load_sweep_data, sweep_dir_for
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
    scripts.generate_perturbed_sweep (out_dir/base_seqs.h5) -- unlike the
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
    every other ONNX session in this repo, see gmanifold/onnx_transformer.py's
    collect_states/make_map), so map_fn pads each chunk up to that size
    rather than feeding it a smaller batch directly, which ONNXRuntime
    rejects with a dimension-mismatch error."""
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


def run_sublayer(
    cfg: ModelConfig,
    sweep_dir: str,
    src: str,
    dst: str,
    model,  # loaded batched_autoencoder
    layers: list[str],
    sample_idx: int,
    base_ids: np.ndarray,
    n: int,
    alpha: float,
    radius: str,
    tau: float,
    auto_tau: bool,
    seed: int,
    jacobian_chunk: int,
    jacobian_model_chunk: int | None,
    support_chunk: int,
    support_model_chunk: int | None,
    out_dir: str,
    flush_every: int,
    no_push: bool,
    pusher: AsyncPusher | None,
) -> None:
    layer_idx = layers.index(src)
    seq = cfg.seq
    d_model = cfg.d_model
    model_indices = [layer_idx * seq + pos for pos in range(seq)]

    print(f"[{src}->{dst}] sampling {n} points x {seq} positions", flush=True)
    x, info = sampling.batched_sample(
        model, n, alpha=alpha, radius=radius, tau=tau, auto_tau=auto_tau, seed=seed,
        jacobian_chunk=jacobian_chunk, jacobian_model_chunk=jacobian_model_chunk,
        support_chunk=support_chunk, support_model_chunk=support_model_chunk,
    )  # x: (n_models_total, n, d_model); only this sublayer's model_indices rows matter

    # PyTorch's CUDA caching allocator holds onto freed blocks instead of
    # releasing them back to the driver, so ONNXRuntime's own separate CUDA
    # allocator (used below) can fail to allocate even when PyTorch itself
    # has plenty of "free" (but cached, unreleased) memory -- release it
    # explicitly before handing off to any ONNXRuntime session.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        attrs=dict(
            sample_idx=sample_idx, n_samples=n, seq=seq, d_model=d_model,
            alpha=alpha, radius=radius,
        ),
        pusher=pusher,
    )

    row = writer.next_row
    for pos in range(seq):
        model_idx = model_indices[pos]
        pts = x[model_idx].detach().cpu().numpy().astype(np.float32)  # (n, d_model)
        anchor = info["anchor"][model_idx].detach().cpu().numpy().astype(np.int64)
        multiplicity = info["multiplicity"][model_idx].detach().cpu().numpy().astype(np.int64)
        ess = float(info["ess"][model_idx])
        tau_used = float(info["tau"][model_idx])
        print(f"[{src}->{dst}] pos={pos}: ess={ess:.0f}/{info['n_candidates']}, tau={tau_used:.2f}", flush=True)

        spliced = np.broadcast_to(ctx, (n, seq, d_model)).copy()  # (n, seq, d_model)
        spliced[:, pos] = pts
        out = map_fn(spliced.astype(np.float32))  # (n, seq, d_model)
        dst_vecs = np.ascontiguousarray(out[:, pos])  # (n, d_model)

        writer.write_rows(
            row, row + n,
            src_vec=pts, dst_vec=dst_vecs,
            pos=np.full(n, pos, dtype=np.int64),
            anchor=anchor, multiplicity=multiplicity,
        )
        row += n

    writer.close()
    print(f"[{src}->{dst}] done -> {layer_key}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--ae-checkpoints-dir", required=True, help="save_all() output dir from train_batched_manifold.py")
    p.add_argument("--knn-chunk", type=int, default=512, help="point-dim chunk size for fit_stats()'s knn passes")
    p.add_argument("--knn-model-chunk", type=int, default=8,
                   help="model-dim chunk size for fit_stats()'s knn passes -- unchunked (all models at once) "
                        "OOM'd at n_models=100 with cdist trying to allocate >8GiB in one shot")
    p.add_argument("--sample-idx", type=int, default=0, help="must match the --sample-idx used at training time")
    p.add_argument("--layers", nargs="*", default=None,
                   help="must match the --layers used at training time (default: every layer in sweep_dir)")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="only run these src.dst sublayers (default: every cfg.sublayers entry whose input "
                        "has a trained autoencoder)")
    p.add_argument("--n", type=int, default=1000, help="samples per (sublayer, position)")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--radius", choices=["global", "local"], default="global")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--auto-tau", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jacobian-chunk", type=int, default=256,
                   help="point-dim chunk size for the volume-reweighting Jacobian pass in sampling.batched_sample")
    p.add_argument("--jacobian-model-chunk", type=int, default=8,
                   help="model-dim chunk size for the same Jacobian pass -- forward-mode AD's memory scales with "
                        "points x models processed together, so a large n_models (many sublayer/position models "
                        "loaded at once) needs this bounded too, not just --jacobian-chunk (default: all models "
                        "at once, which OOM'd at n_models=104 with the old unchunked jacobian())")
    p.add_argument("--support-chunk", type=int, default=2048,
                   help="point-dim chunk size for the support-density cdist pass in sampling.batched_sample")
    p.add_argument("--support-model-chunk", type=int, default=4,
                   help="model-dim chunk size for the same cdist pass -- its (n_models, chunk, n_train_rows) "
                        "result scales with n_models too, so a large n_models needs this bounded (unchunked "
                        "OOM'd trying to allocate 69GiB at n_models=100; model_chunk=8/chunk=4096 still OOM'd "
                        "at 11GiB against a thin free margin, hence the smaller defaults here)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--flush-every", type=int, default=10)
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--upload-workers", type=int, default=4)
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}", flush=True)

    print(f"loading batched_autoencoder from {args.ae_checkpoints_dir}", flush=True)
    model = load_batched(args.ae_checkpoints_dir, device=device)
    model.eval()

    print("reattaching states and computing fit_stats (latent_codes/latent_scale) for sampling", flush=True)
    full_activations = load_sweep_data(sweep_dir, layers, args.sample_idx)
    model.reattach(full_activations.to(device))
    model.fit_stats(knn_chunk=args.knn_chunk, knn_model_chunk=args.knn_model_chunk)

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
            cfg, sweep_dir, src, dst, model, layers, args.sample_idx, base_ids,
            n=args.n, alpha=args.alpha, radius=args.radius, tau=args.tau, auto_tau=args.auto_tau,
            seed=args.seed, jacobian_chunk=args.jacobian_chunk, jacobian_model_chunk=args.jacobian_model_chunk,
            support_chunk=args.support_chunk, support_model_chunk=args.support_model_chunk,
            out_dir=args.out_dir, flush_every=args.flush_every,
            no_push=args.no_push, pusher=pusher,
        )
        print(f"[{src}->{dst}] took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
