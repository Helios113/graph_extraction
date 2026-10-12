"""Jacobian determinant of each sublayer at every perturbed point of an
existing `scripts/python/generate_perturbed_sweep.py` run.

A perturbed-sweep row is (sample_idx, pos, vocab_id): base sequence
`base_seqs[sample_idx]` with the token at `pos` replaced by `vocab_id`. For
each sublayer `{in}.{out}` in the config, this computes -- at that row's
perturbed sequence -- the d_model x d_model diagonal Jacobian block
d out[pos] / d in[pos] (the same block `compute_base_jacobian_stats.py`
computes at the unperturbed sequences), and saves its determinant as
(sign, log|det|) alongside the row's index.

Everything runs through the existing ONNX framework, not torch autograd:
  1. The sweep only stored the swept position's vector, but the Jacobian
     needs the full (seq, d_model) `in` activation of the perturbed
     sequence, so each block of rows is re-run through
     base_model_extra_outputs.onnx (the same graph the sweep used).
  2. Each sublayer's ONNX pullback graph (MaskedSumLoss, built by
     construct.ensure_subgraph_pullback) gives one Jacobian row per batch
     slot via a one-hot mask. The forward subgraphs have a fixed batch dim
     (cfg.batch_for_node(), e.g. 32000), so the pullback is built at that
     batch -- not the config's `input_shape` batch (e.g. 32), which
     wouldn't match -- and packs `--points-per-call` distinct perturbed
     points x d_model one-hot rows into every call.
  3. The pullback also returns the sublayer's forward output. Its value at
     `pos` is checked against the sweep's stored `vec` for that row, which
     confirms every Jacobian was taken at the stored perturbed point.
  4. torch.linalg.slogdet (float64) on the gathered Jacobians, on GPU, as
     compute_base_jacobian_stats does with svdvals. log|det| is saved
     instead of det: a 384x384 determinant easily overflows or underflows
     float64, and det = sign * exp(log_abs_det).

Output is one ShardedLayerWriter per sublayer at --out-dir/{in}.{out}, one
row per perturbed-sweep row, in the same row order:
  row_idx      global row index into the perturbed sweep's layer shards
  sample_idx, pos, vocab_id   copied from the sweep row
  sign         sign of det (-1, 0, +1)
  log_abs_det  log|det J|
Rows are written per block and flushed every --flush-every blocks; finished
shards are pushed to NFS staging (unless --no-push). Re-running the same
command resumes from the last completed shard.

The `input_ids -> embedding` sublayer is skipped: its input is integer
token ids, so it has no Jacobian.

Usage:
    uv run python -m scripts.python.compute_perturbed_jacobian_det configs/simple_stories.toml \\
        --perturbed-sweep-dir onnx_models/simple_stories/perturbed_sweep \\
        --out-dir onnx_models/simple_stories/perturbed_jacobian_det
"""
import argparse
import dataclasses
import os

import h5py
import numpy as np
import onnxruntime as ort
import torch
import tqdm

from config import load_config
from construct.construct import (
    ensure_base_model,
    ensure_extra_outputs,
    ensure_subgraph_pullback,
    ensure_subgraphs,
    exported_batch_size,
)
from execute.io_onnx import run_iobinding_cpu_input_gpu_output, run_iobinding_gpu_inputs
from execute.sharded_writer import ShardedLayerWriter, load_shard_rows

DEVICE = torch.device("cuda")


def _session(path) -> ort.InferenceSession:
    """CUDA session with TF32 matmuls OFF. ORT's CUDA EP enables TF32 by
    default on Ampere+ GPUs (A40, H100), which keeps only a 10-bit mantissa:
    outputs carry ~1e-3 relative error, i.e. ~5e-4 absolute noise on this
    model's residual stream. That swamps small output differences (a 2e-4
    input step reads as a ratio of ~4 instead of ~1.6) and puts ~5e-3 of
    error on Jacobian singular values -- so small singular values and det
    signs near a det = 0 crossing are only meaningful with it off."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_options.intra_op_num_threads = 8
    return ort.InferenceSession(
        str(path), sess_options=sess_options, providers=[("CUDAExecutionProvider", {"use_tf32": "0"})],
    )


def open_pullback(fwd_path, sublayer) -> tuple[ort.InferenceSession, ort.IOBinding, torch.Tensor, torch.Tensor]:
    """(session, io_binding, inp_buf, mask_buf) for `sublayer`'s pullback,
    built at the forward subgraph's own fixed batch: the MaskedSumLoss mask
    must match the output shape exactly (see module docstring). The two
    buffers are the preallocated pullback inputs _block_slogdet reuses."""
    fixed_batch = exported_batch_size(fwd_path)
    _, seq, d_model = sublayer.input_shape
    pb_conf = dataclasses.replace(sublayer, input_shape=[fixed_batch, seq, d_model])
    [pb_path] = ensure_subgraph_pullback([(fwd_path, pb_conf)], force=False)
    pb_sess = _session(pb_path)
    mask_shape = next(i.shape for i in pb_sess.get_inputs() if i.name == "mask")
    if list(mask_shape) != [fixed_batch, seq, d_model]:
        raise RuntimeError(
            f"cached pullback {pb_path} has mask shape {mask_shape}, expected "
            f"{[fixed_batch, seq, d_model]} -- it was built for a different batch; "
            f"delete it (and its .data file) so it's rebuilt"
        )
    inp_buf = torch.zeros((fixed_batch, seq, d_model), dtype=torch.float32, device=DEVICE)
    return pb_sess, pb_sess.io_binding(), inp_buf, torch.zeros_like(inp_buf)


def jacobian_chunk(
    pb_sess: ort.InferenceSession,
    pb_io: ort.IOBinding,
    in_act: torch.Tensor,
    pos: torch.Tensor,
    input_name: str,
    output_name: str,
    inp_buf: torch.Tensor,
    mask_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Jacobian blocks J_k = d out[pos_k] / d in[pos_k] and the forward
    output at pos_k for n points, in ONE pullback call -- n * d_model must
    fit the pullback's fixed batch.

    in_act: (n, seq, d_model) GPU `in` activations; pos: (n,) GPU long.
    inp_buf/mask_buf: preallocated (fixed_batch, seq, d_model) pullback
    inputs, reused across calls. Slots past n * d_model keep stale inputs,
    but their mask is all zero, so they contribute nothing to the Jacobian
    rows that are read.

    Returns (jac (n, d_model, d_model) with jac[k, i, j] = d out_i / d in_j,
    out_at_pos (n, d_model)), both GPU float32 and owned (safe to keep
    across later calls).
    """
    n_pts, _, d_model = in_act.shape
    n_slots = n_pts * d_model

    # slot k * d_model + i holds point k's full input and a one-hot mask
    # at (pos_k, i), so its input gradient is row i of point k's Jacobian:
    # grad[slot, pos_k, j] = d out[pos_k, i] / d in[pos_k, j].
    slots = torch.arange(n_slots, device=DEVICE)
    slot_pos = pos.repeat_interleave(d_model)
    slot_dim = torch.arange(d_model, device=DEVICE).repeat(n_pts)

    inp_buf[:n_slots] = in_act.repeat_interleave(d_model, dim=0)
    mask_buf.zero_()
    mask_buf[slots, slot_pos, slot_dim] = 1.0

    grad, out = run_iobinding_gpu_inputs(
        pb_sess,
        {input_name: inp_buf, "mask": mask_buf},
        [f"{input_name}_grad", output_name],
        io_binding=pb_io,
    )
    jac = grad[slots, slot_pos, :].reshape(n_pts, d_model, d_model)
    return jac, out[slots[::d_model], pos].clone()


def _block_slogdet(
    pb_sess: ort.InferenceSession,
    pb_io: ort.IOBinding,
    in_act: torch.Tensor,
    pos: torch.Tensor,
    input_name: str,
    output_name: str,
    points_per_call: int,
    inp_buf: torch.Tensor,
    mask_buf: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """slogdet of d out[pos_k] / d in[pos_k] for every point k in a block,
    via jacobian_chunk on points_per_call points at a time.

    in_act: (n, seq, d_model) GPU `in` activations of n perturbed sequences.
    pos: (n,) GPU long, each sequence's swept position.

    Returns (sign (n,), log_abs_det (n,), out_at_pos (n, d_model)) on GPU.
    out_at_pos is the pullback's own forward output at pos, used to check
    against the sweep's stored vec.
    """
    n = in_act.shape[0]
    signs, logdets, outs = [], [], []
    for s in range(0, n, points_per_call):
        e = min(s + points_per_call, n)
        jac, out = jacobian_chunk(
            pb_sess, pb_io, in_act[s:e], pos[s:e], input_name, output_name, inp_buf, mask_buf,
        )
        sign, logabsdet = torch.linalg.slogdet(jac.to(torch.float64))

        signs.append(sign)
        logdets.append(logabsdet)
        outs.append(out)
        del jac

    return torch.cat(signs), torch.cat(logdets), torch.cat(outs)


def compute_perturbed_jacobian_det(
    cfg,
    perturbed_sweep_dir: str,
    out_dir: str,
    sublayer_names: list[str] | None,
    samples: list[int] | None,
    points_per_call: int | None,
    max_rows: int | None,
    shard_rows: int,
    flush_every: int,
    no_push: bool,
    verify_atol: float,
) -> None:
    with h5py.File(os.path.join(perturbed_sweep_dir, "base_seqs.h5"), "r") as f:
        base_seqs = f["base_seqs"][:]
    print(f"base_seqs: {base_seqs.shape}")

    ensure_base_model(cfg, force=False, verbose=False)
    # Same extra outputs as generate_perturbed_sweep.py, so its cached
    # base_model_extra_outputs.onnx is reused instead of regenerated.
    extra_path = ensure_extra_outputs(cfg, [s.output for s in cfg.sublayers], force=False)
    seq_batch = exported_batch_size(extra_path)
    extra_sess = _session(extra_path)
    extra_io = extra_sess.io_binding()
    extra_input_name = extra_sess.get_inputs()[0].name

    sub_graphs = ensure_subgraphs(cfg, force=False)
    if sublayer_names is not None:
        known = {f"{c.input}.{c.output}" for _, c in sub_graphs}
        unknown = set(sublayer_names) - known
        if unknown:
            raise ValueError(f"unknown --sublayers {sorted(unknown)}; config has {sorted(known)}")

    for fwd_path, sublayer in sub_graphs:
        layer_name = f"{sublayer.input}.{sublayer.output}"
        if sublayer_names is not None and layer_name not in sublayer_names:
            continue
        if sublayer.input == cfg.input_name:
            print(f"{layer_name}: input is token ids (no Jacobian), skipping")
            continue

        fixed_batch = exported_batch_size(fwd_path)
        _, seq, d_model = sublayer.input_shape
        ppc = points_per_call or fixed_batch // d_model
        if ppc * d_model > fixed_batch:
            raise ValueError(
                f"--points-per-call {ppc} * d_model {d_model} exceeds the subgraph's "
                f"fixed batch {fixed_batch}"
            )

        rows = load_shard_rows(
            ".", os.path.join(perturbed_sweep_dir, sublayer.output),
            dataset_names=["sample_idx", "pos", "vocab_id", "vec"],
        )
        # global row index into the sweep, kept through the --samples filter
        row_idx_all = np.arange(len(rows["pos"]), dtype=np.int64)
        if samples is not None:
            keep = np.isin(rows["sample_idx"], samples)
            rows = {k: v[keep] for k, v in rows.items()}
            row_idx_all = row_idx_all[keep]
            print(f"{layer_name}: --samples {samples} -> {len(row_idx_all)} of {len(keep)} rows")
        n_rows = len(rows["pos"]) if max_rows is None else min(max_rows, len(rows["pos"]))

        writer = ShardedLayerWriter(
            layer_key=os.path.join(out_dir, layer_name),
            n_rows=n_rows,
            schema={
                "row_idx": ((), "int64"),
                "sample_idx": ((), "int64"),
                "pos": ((), "int64"),
                "vocab_id": ((), "int64"),
                "sign": ((), "int8"),
                "log_abs_det": ((), "float64"),
            },
            shard_rows=shard_rows,
            flush_every=flush_every,
            attrs={"seq": seq, "d_model": d_model, "sublayer_input": sublayer.input,
                   "sublayer_output": sublayer.output,
                   "perturbed_sweep_dir": perturbed_sweep_dir,
                   **({"samples": list(samples)} if samples is not None else {})},
            no_push=no_push,
            project_root=".",
        )
        if writer.next_row >= n_rows:
            continue

        pb_sess, pb_io, inp_buf, mask_buf = open_pullback(fwd_path, sublayer)

        max_err = 0.0
        ld_min, ld_max = np.inf, -np.inf
        n_nonpos = 0
        for start in tqdm.trange(writer.next_row, n_rows, seq_batch, desc=layer_name):
            end = min(start + seq_batch, n_rows)
            n = end - start
            sample_idx = rows["sample_idx"][start:end]
            pos = rows["pos"][start:end]
            vocab_id = rows["vocab_id"][start:end]

            # rebuild each row's perturbed sequence, padded to the extra
            # outputs graph's fixed batch
            token_ids = np.zeros((seq_batch, seq), dtype=np.int64)
            token_ids[:n] = base_seqs[sample_idx]
            token_ids[np.arange(n), pos] = vocab_id

            (in_act,) = run_iobinding_cpu_input_gpu_output(
                extra_sess, extra_input_name, token_ids, [sublayer.input], io_binding=extra_io,
            )
            in_act = in_act[:n].to(torch.float32).clone()
            pos_t = torch.from_numpy(pos).to(DEVICE)

            sign, logabsdet, out_at_pos = _block_slogdet(
                pb_sess, pb_io, in_act, pos_t, sublayer.input, sublayer.output,
                ppc, inp_buf, mask_buf,
            )
            del in_act

            stored = torch.from_numpy(rows["vec"][start:end]).to(DEVICE)
            err = (out_at_pos - stored).abs().max().item()
            max_err = max(max_err, err)
            if err > verify_atol:
                raise RuntimeError(
                    f"{layer_name} rows [{start}, {end}): recomputed {sublayer.output} at pos "
                    f"differs from the perturbed sweep's stored vec by {err:.3g} "
                    f"(> --verify-atol {verify_atol}) -- the Jacobian is not being taken at "
                    f"the stored perturbed point"
                )

            sign_np = sign.cpu().numpy().astype(np.int8)
            logabsdet_np = logabsdet.cpu().numpy()
            writer.write_rows(
                start, end,
                row_idx=row_idx_all[start:end],
                sample_idx=sample_idx,
                pos=pos,
                vocab_id=vocab_id,
                sign=sign_np,
                log_abs_det=logabsdet_np,
            )

            finite = logabsdet_np[np.isfinite(logabsdet_np)]
            if finite.size:
                ld_min, ld_max = min(ld_min, finite.min()), max(ld_max, finite.max())
            n_nonpos += int((sign_np <= 0).sum())

        writer.close()
        del pb_sess, pb_io, inp_buf, mask_buf, rows
        torch.cuda.empty_cache()

        print(
            f"{layer_name}: this run's log|det| range [{ld_min:.4g}, {ld_max:.4g}], "
            f"{n_nonpos} row(s) with sign <= 0, max |recomputed - stored vec| = {max_err:.3g} "
            f"-> {writer.layer_key}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--perturbed-sweep-dir",
        type=str,
        default="onnx_models/simple_stories/perturbed_sweep",
        help="--out-dir of the generate_perturbed_sweep.py run to read (project-relative; "
        "holds base_seqs.h5 and one {output}.manifest.json per layer)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/simple_stories/perturbed_jacobian_det",
        help="Project-relative output dir (ShardedLayerWriter layer_key prefix) -- one sharded "
        "output per sublayer, {in}.{out}",
    )
    parser.add_argument(
        "--sublayers",
        type=str,
        nargs="+",
        default=None,
        help="Only these sublayers, as {in}.{out} (e.g. embedding.add_4). Default: all "
        "sublayers except the token-id input one",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help="Only perturbed rows whose sample_idx is in this list (e.g. 0). row_idx stays the "
        "global sweep index. Default: all samples. Use a separate --out-dir, since progress "
        "is recorded against the filtered row count",
    )
    parser.add_argument(
        "--points-per-call",
        type=int,
        default=None,
        help="Perturbed points packed into one pullback call (each takes d_model batch slots). "
        "Default: as many as fit, fixed_batch // d_model (e.g. 32000 // 384 = 83)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Only process the first N perturbed rows per sublayer (smoke tests) -- use a "
        "separate --out-dir, since progress is recorded against this row count",
    )
    parser.add_argument(
        "--shard-rows",
        type=int,
        default=500_000,
        help="Rows per output shard -- each is pushed to NFS staging as soon as it's full",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=5,
        help="Flush the current shard's h5 file to local scratch every this many blocks "
        "(a block is one extra-outputs batch of perturbed rows, e.g. 32000)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Keep finished shards on local scratch instead of pushing to NFS staging",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-3,
        help="Max allowed |recomputed sublayer output at pos - perturbed sweep's stored vec| "
        "before aborting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    compute_perturbed_jacobian_det(
        cfg,
        perturbed_sweep_dir=args.perturbed_sweep_dir,
        out_dir=args.out_dir,
        sublayer_names=args.sublayers,
        samples=args.samples,
        points_per_call=args.points_per_call,
        max_rows=args.max_rows,
        shard_rows=args.shard_rows,
        flush_every=args.flush_every,
        no_push=args.no_push,
        verify_atol=args.verify_atol,
    )
    print(f"saved per-row Jacobian (sign, log|det|) to {args.out_dir}")


if __name__ == "__main__":
    main()
