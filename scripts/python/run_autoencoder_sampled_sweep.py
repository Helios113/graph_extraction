"""For each sublayer (input=src, output=dst) in a toml ModelConfig, samples
points from src's trained batched_autoencoder and runs them through the
real ONNX sublayer, saving (sampled input, real output) pairs -- the
autoencoder-sampling analogue of scripts.python.generate_perturbed_sweep, which
instead perturbs at the vocab/token level.

Mechanism (per sample, per sublayer, per position `pos`):
  1. --sample-idx takes one or more samples; --samples-per-chunk groups them
     (default 1 = fully sequential) and merges each group's checkpoint
     directories into one batched_autoencoder via
     autoencoder.batched_autoencoder.load_batched_multi, so
     reattach()/fit_stats() run once per group instead of once per sample --
     purely a throughput/memory knob (CLAUDE.md's model_chunk convention),
     output is identical at any --samples-per-chunk value.
  2. Sample n ambient points at src, position `pos`, from the (possibly
     multi-sample-merged) batched_autoencoder (model_index =
     model_row_offset + layers.index(src) * seq + pos, where
     model_row_offset is 0 for a single sample or that sample's cumulative
     row offset into the merged batch -- see run_sublayer's docstring).
  3. Context: base_seqs[sample_idx] (the same fixed sequence that sample's
     autoencoder was trained on -- see sweep_dir/base_seqs.h5) is run once
     per sample through ensure_extra_outputs to get REAL src-activations at
     every position.
  4. Splice: each sampled point replaces position `pos`'s value in that
     real src tensor -- every other position keeps its real activation.
  5. Run the full (1, seq, d_model) spliced tensor through the {src}.{dst}
     subgraph (ensure_subgraphs) and read dst at position `pos`. Since every
     other position already holds its real value, this is correct even for
     sublayers that cross attention (unlike gmanifold.onnx_transformer's
     make_map, which zero-pads every other position and so is only correct
     for strictly tokenwise sublayers).

--alpha takes one or more latent-ball alpha levels (default: [0.3]); each is
sampled and written independently. --noise optionally also writes a
held-out-real-plus-Gaussian-noise calibration variant per level given (a
multiple of each anchor's local ambient spacing, matching
plot_autoencoder_sample_diagnostics.py's 0.5x/1x bands) -- unlike the AE
samples, these don't depend on the autoencoder's decoder/sampling at all,
only on model.states/ambient_spacing from fit_stats().

Output: one h5 file per (sample, sublayer, variant) --
{src}.{dst}.alpha{a}.sample{idx}.h5 per alpha,
{src}.{dst}.noise{level}.sample{idx}.h5 per noise level (see
execute.sharded_writer.ShardedLayerWriter and CLAUDE.md's per-sublayer
resilience pattern) -- pushed as soon as that variant finishes, so a crash
partway through one sample/alpha/noise level doesn't touch the others'
already-completed files. Each row holds the sampled src vector, the real dst vector,
the position, the anchor index (which fitted state's latent ball the AE
sample was drawn from, or which fitted state the noise was added to), and
its multiplicity (AE samples: ball-membership count; noise: always 1) --
ess/tau (per (sublayer, alpha, pos), not per row) are printed, not stored;
alpha/noise_level are stored as manifest attrs.
"""

import argparse
import os
import time

import h5py
import glob
import re

import numpy as np
import onnxruntime as ort
import torch

from autoencoder import sampling
from autoencoder.batched_autoencoder import load_batched_multi
from autoencoder.data import ae_checkpoints_dir_for, load_sweep_data, sweep_dir_for
from config import ModelConfig, load_config
from construct.construct import ensure_base_model, ensure_extra_outputs, ensure_subgraphs
from execute.remote_storage import AsyncPusher as HetznerAsyncPusher
from execute.sharded_writer import Backend, ShardedLayerWriter, discover_layer_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_SESSION_PROVIDERS = ["CUDAExecutionProvider"]


def _assert_checkpoint_dir_complete(ckpt_dir: str, expected_indices: set[int]) -> None:
    """Checked independently of load_batched_multi's own state (which
    concatenates each source's row POSITION, not its global model_index --
    see autoencoder.batched_autoencoder.load_batched's docstring on sparse
    indices) since load_batched_multi doesn't expose a completeness check of
    its own. Every source dir load_batched_multi merges here must hold
    exactly `expected_indices` (== every (layer, pos) in `layers`), not a
    subset left behind by an in-progress multi-batch train_batched_manifold.py
    run -- otherwise this script would silently sample from however many
    rows happened to be on disk, with model_row_offset's contiguous-block
    arithmetic pointing at the wrong rows entirely."""
    by_index = set()
    for p in glob.glob(os.path.join(ckpt_dir, "model_*.pt")):
        m = re.match(r"model_(\d+)\.pt$", os.path.basename(p))
        if m:
            by_index.add(int(m.group(1)))
    assert by_index == expected_indices, (
        f"{ckpt_dir}: expected checkpoint indices {sorted(expected_indices)}, got {sorted(by_index)} -- "
        f"missing {sorted(expected_indices - by_index)}, unexpected {sorted(by_index - expected_indices)} "
        "(a multi-batch train_batched_manifold.py run may not have finished writing every layer batch yet)"
    )


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


def _write_variant(
    layer_key: str,
    seq: int,
    d_model: int,
    n: int,
    pts_by_pos: list[np.ndarray],  # (n, d_model) per position
    anchor_by_pos: list[np.ndarray],  # (n,) per position
    multiplicity_by_pos: list[np.ndarray],  # (n,) per position
    ctx: np.ndarray,
    map_fn,
    extra_attrs: dict,
    sample_idx: int,
    flush_every: int,
    no_push: bool,
    backend: Backend,
    pusher: object | None,
) -> None:
    """Shared write loop for both AE-manifold samples and noise variants --
    splices each position's already-drawn points into the real per-position
    context and writes (sampled src, real dst) pairs to one sharded h5 file.
    `layer_key` (including the out_dir/{src}.{dst}.<variant> path) is the
    ShardedLayerWriter resume key, so each (sublayer, alpha) or (sublayer,
    noise-level) combination gets its own file/progress -- a crash partway
    through one variant doesn't touch the others' already-completed files,
    per CLAUDE.md's per-sublayer resilience pattern."""
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
        attrs=dict(sample_idx=sample_idx, n_samples=n, seq=seq, d_model=d_model, **extra_attrs),
        pusher=pusher,
    )

    if writer.next_row >= writer.n_rows:  # already complete from an earlier run
        writer.close()
        return

    row = writer.next_row
    for pos in range(seq):
        pts = pts_by_pos[pos]
        spliced = np.broadcast_to(ctx, (n, seq, d_model)).copy()  # (n, seq, d_model)
        spliced[:, pos] = pts
        out = map_fn(spliced.astype(np.float32))  # (n, seq, d_model)
        dst_vecs = np.ascontiguousarray(out[:, pos])  # (n, d_model)

        writer.write_rows(
            row, row + n,
            src_vec=pts, dst_vec=dst_vecs,
            pos=np.full(n, pos, dtype=np.int64),
            anchor=anchor_by_pos[pos], multiplicity=multiplicity_by_pos[pos],
        )
        row += n

    writer.close()
    print(f"[{layer_key}] done", flush=True)


def run_sublayer(
    cfg: ModelConfig,
    sweep_dir: str,
    src: str,
    dst: str,
    model,  # loaded batched_autoencoder, possibly merged across several samples (see load_batched_multi)
    layers: list[str],
    model_row_offset: int,
    sample_idx: int,
    base_ids: np.ndarray,
    n: int,
    alphas: list[float],
    radius: str,
    tau: float,
    auto_tau: bool,
    noise_levels: list[float],
    seed: int,
    jacobian_chunk: int,
    jacobian_model_chunk: int | None,
    support_chunk: int,
    support_model_chunk: int | None,
    out_dir: str,
    flush_every: int,
    no_push: bool,
    backend: Backend,
    pusher: object | None,
) -> None:
    """`model_row_offset`: row index in `model` where this sample_idx's own
    (layer, pos) block starts -- 0 when `model` holds a single sample
    (load_batched's plain contiguous 0..n-1), or that sample's cumulative
    row count into a merged multi-sample batch (load_batched_multi
    concatenates samples in order, each contributing its own full
    n_layers*seq rows)."""
    layer_idx = layers.index(src)
    seq = cfg.seq
    d_model = cfg.d_model
    model_indices = [model_row_offset + layer_idx * seq + pos for pos in range(seq)]

    print(f"[{src}->{dst}] getting real context activations for sample_idx={sample_idx}", flush=True)
    ctx = context_activations(cfg, src, base_ids)  # (seq, d_model), real values at every position
    map_fn = make_subgraph_map(cfg, src, dst)

    for alpha in alphas:
        print(f"[{src}->{dst}] alpha={alpha:g}: sampling {n} points x {seq} positions", flush=True)
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

        pts_by_pos, anchor_by_pos, mult_by_pos = [], [], []
        for pos in range(seq):
            model_idx = model_indices[pos]
            pts_by_pos.append(x[model_idx].detach().cpu().numpy().astype(np.float32))
            anchor_by_pos.append(info["anchor"][model_idx].detach().cpu().numpy().astype(np.int64))
            mult_by_pos.append(info["multiplicity"][model_idx].detach().cpu().numpy().astype(np.int64))
            ess = float(info["ess"][model_idx])
            tau_used = float(info["tau"][model_idx])
            print(f"[{src}->{dst}] alpha={alpha:g} pos={pos}: ess={ess:.0f}/{info['n_candidates']}, "
                  f"tau={tau_used:.2f}", flush=True)

        _write_variant(
            layer_key=os.path.join(out_dir, f"{src}.{dst}.alpha{alpha:g}.sample{sample_idx}"),
            seq=seq, d_model=d_model, n=n,
            pts_by_pos=pts_by_pos, anchor_by_pos=anchor_by_pos, multiplicity_by_pos=mult_by_pos,
            ctx=ctx, map_fn=map_fn, extra_attrs=dict(alpha=alpha, radius=radius, variant="alpha"),
            sample_idx=sample_idx, flush_every=flush_every, no_push=no_push, backend=backend, pusher=pusher,
        )

    if noise_levels:
        # Held-out-noise calibration variant, matching
        # plot_autoencoder_sample_diagnostics.py's "real + Nx noise" bands:
        # anchor = one of this (layer, pos)'s fitted real states (model.states,
        # the same subset fit_stats() built latent_codes/ambient_spacing from),
        # displaced by isotropic Gaussian noise scaled by that anchor's own
        # local ambient spacing (model.unit) rather than a flat stddev -- so a
        # given noise level means the same thing (a multiple of local point
        # density) at every position/sublayer, unlike generate_ambient_sweep.py's
        # flat --tau (that script has no autoencoder/fit_stats and so no
        # per-point spacing to scale by).
        n_train_rows = model.states.shape[1]
        g = torch.Generator(device=model.device).manual_seed(seed)
        for level in noise_levels:
            print(f"[{src}->{dst}] noise={level:g}x local spacing: sampling {n} points x {seq} positions",
                  flush=True)
            pts_by_pos, anchor_by_pos, mult_by_pos = [], [], []
            for pos in range(seq):
                model_idx = model_indices[pos]
                j = torch.randint(0, n_train_rows, (n,), generator=g, device=model.device)
                anchor_pts = model.states[model_idx, j]  # (n, d_model)
                # model.unit(j) gathers against the FULL (n_models_total, n_train_rows)
                # model.ambient_spacing, so it can't be called with a single row's j
                # (shape (n,)) -- index this model row's spacing directly instead,
                # replicating unit()'s anchor-vs-global logic for just model_idx.
                row_spacing = model.ambient_spacing[model_idx]  # (n_train_rows,)
                unit = (
                    row_spacing[j] if model.spacing_unit == "anchor"
                    else row_spacing.median().expand(n)
                )
                nz = torch.randn(n, d_model, device=model.device, generator=g)
                nz = nz / nz.norm(dim=1, keepdim=True) * (level * unit)[:, None]
                pts_by_pos.append((anchor_pts + nz).detach().cpu().numpy().astype(np.float32))
                anchor_by_pos.append(j.detach().cpu().numpy().astype(np.int64))
                mult_by_pos.append(np.ones(n, dtype=np.int64))

            _write_variant(
                layer_key=os.path.join(out_dir, f"{src}.{dst}.noise{level:g}.sample{sample_idx}"),
                seq=seq, d_model=d_model, n=n,
                pts_by_pos=pts_by_pos, anchor_by_pos=anchor_by_pos, multiplicity_by_pos=mult_by_pos,
                ctx=ctx, map_fn=map_fn, extra_attrs=dict(noise_level=level, variant="noise"),
                sample_idx=sample_idx, flush_every=flush_every, no_push=no_push, backend=backend, pusher=pusher,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--ae-checkpoints-dir", default=None,
                   help="save_all() output dir from train_batched_manifold.py -- if given, used directly and "
                        "must hold exactly one sample's checkpoints (no --samples-per-chunk grouping); default: "
                        "autoencoder.data.ae_checkpoints_dir_for(cfg, sample_idx) per sample, letting multiple "
                        "--sample-idx values each resolve to their own directory for grouping")
    p.add_argument("--knn-chunk", type=int, default=512, help="point-dim chunk size for fit_stats()'s knn passes")
    p.add_argument("--knn-model-chunk", type=int, default=8,
                   help="model-dim chunk size for fit_stats()'s knn passes -- unchunked (all models at once) "
                        "OOM'd at n_models=100 with cdist trying to allocate >8GiB in one shot")
    p.add_argument("--sample-idx", type=int, nargs="+", default=[0],
                   help="one or more samples, must match the --sample-idx values used at training time")
    p.add_argument("--samples-per-chunk", type=int, default=1,
                   help="how many samples' checkpoints to merge (via load_batched_multi) into one "
                        "batched_autoencoder and run reattach/fit_stats/sampling against jointly, per group -- "
                        "1 (default) processes samples fully sequentially and is always safe; raise it on "
                        "hardware with more GPU memory headroom to combine samples' states into fewer, larger "
                        "reattach/fit_stats/sampling passes. Purely a throughput/memory knob -- output is "
                        "identical at every value, see CLAUDE.md's model_chunk chunking convention")
    p.add_argument("--layers", nargs="*", default=None,
                   help="must match the --layers used at training time (default: every layer in sweep_dir)")
    p.add_argument("--sublayers", nargs="*", default=None,
                   help="only run these src.dst sublayers (default: every cfg.sublayers entry whose input "
                        "has a trained autoencoder)")
    p.add_argument("--n", type=int, default=1000, help="samples per (sublayer, position, variant)")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.3],
                   help="one or more latent-ball alpha levels to sweep; each gets its own "
                        "{src}.{dst}.alpha{a}.h5 (default: [0.3])")
    p.add_argument("--radius", choices=["global", "local"], default="global")
    p.add_argument("--tau", type=float, default=1.0, help="volume-reweighting tempering exponent (sampling.batched_sample's tau, not --noise)")
    p.add_argument("--auto-tau", action="store_true")
    p.add_argument("--noise", type=float, nargs="*", default=[],
                   help="also write held-out-real-plus-Gaussian-noise calibration variant(s), each a multiple "
                        "of local ambient spacing (matching plot_autoencoder_sample_diagnostics.py's 0.5x/1x "
                        "bands) -- each gets its own {src}.{dst}.noise{level}.h5. Default: none written.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jacobian-chunk", type=int, default=64,
                   help="point-dim chunk size for the volume-reweighting Jacobian pass in sampling.batched_sample "
                        "(default lowered from 256: at gpt2's d_model=768/latent_dim=128 -- larger than the "
                        "configs model_chunk=8/chunk=256 were originally tuned against -- that combination OOM'd "
                        "trying to allocate 48.83GiB on a 44GiB GPU at n_models=100; scale back up for configs "
                        "with a smaller decoder)")
    p.add_argument("--jacobian-model-chunk", type=int, default=1,
                   help="model-dim chunk size for the same Jacobian pass -- forward-mode AD's memory scales with "
                        "points x models x latent_dim tangent directions processed together, so a large n_models "
                        "(many sublayer/position models loaded at once) needs this bounded too, not just "
                        "--jacobian-chunk (default lowered from 8 to 1 for the same gpt2 OOM as --jacobian-chunk "
                        "above -- unchunked (all models at once) OOM'd at n_models=104 with the old unchunked "
                        "jacobian())")
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
    p.add_argument("--backend", choices=["hetzner", "nfs"], default="nfs",
                   help="transport ShardedLayerWriter pushes each finished (sublayer, variant)'s shards "
                        "through, and load_sweep_data pulls this sample's perturbed-sweep cloud through -- "
                        "'nfs' (default) via execute.node_relay's NFS-staging relay, 'hetzner' via "
                        "execute.remote_storage (not used by this project's own launchers anymore). Must "
                        "match how the sweep data being read was actually pushed; ignored on the write side "
                        "if --no-push is also set.")
    args = p.parse_args()

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}", flush=True)
    n_models_per_sample = len(layers) * cfg.seq

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

    if args.ae_checkpoints_dir is not None:
        assert len(args.sample_idx) == 1, (
            "--ae-checkpoints-dir is a single explicit directory and can only hold one sample's checkpoints "
            "-- pass multiple --sample-idx values without --ae-checkpoints-dir to use the per-sample default "
            "(autoencoder.data.ae_checkpoints_dir_for) instead"
        )
        sample_ckpt_dir = {args.sample_idx[0]: args.ae_checkpoints_dir}
    else:
        sample_ckpt_dir = {s: ae_checkpoints_dir_for(cfg, s) for s in args.sample_idx}

    groups = [
        args.sample_idx[i:i + args.samples_per_chunk]
        for i in range(0, len(args.sample_idx), args.samples_per_chunk)
    ]
    expected_indices = {layers.index(layer) * cfg.seq + pos for layer in layers for pos in range(cfg.seq)}
    for group in groups:
        print(f"\n=== samples {group}: loading + merging {len(group)} checkpoint dir(s) ===", flush=True)
        for s in group:
            _assert_checkpoint_dir_complete(sample_ckpt_dir[s], expected_indices)
        model = load_batched_multi([sample_ckpt_dir[s] for s in group], device=device)
        model.eval()

        print("reattaching states and computing fit_stats (latent_codes/latent_scale) for sampling", flush=True)
        full_activations = torch.cat(
            [load_sweep_data(sweep_dir, layers, s, backend=args.backend) for s in group], dim=0,
        )
        model.reattach(full_activations.to(device))
        model.fit_stats(knn_chunk=args.knn_chunk, knn_model_chunk=args.knn_model_chunk)

        for i, sample_idx in enumerate(group):
            model_row_offset = i * n_models_per_sample
            base_ids = load_base_seq(sweep_dir, sample_idx)
            for src, dst in sublayers:
                t0 = time.time()
                run_sublayer(
                    cfg, sweep_dir, src, dst, model, layers, model_row_offset, sample_idx, base_ids,
                    n=args.n, alphas=args.alpha, radius=args.radius, tau=args.tau, auto_tau=args.auto_tau,
                    noise_levels=args.noise,
                    seed=args.seed, jacobian_chunk=args.jacobian_chunk, jacobian_model_chunk=args.jacobian_model_chunk,
                    support_chunk=args.support_chunk, support_model_chunk=args.support_model_chunk,
                    out_dir=args.out_dir, flush_every=args.flush_every,
                    no_push=args.no_push, backend=args.backend, pusher=pusher,
                )
                print(f"[sample {sample_idx}] [{src}->{dst}] took {time.time() - t0:.1f}s", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
