"""Interpolate between two perturbed points of opposite Jacobian-determinant
sign, run the interpolated inputs forward through the sublayer, and save
(input, output) vectors for Lipschitz calculations.

Reads the per-row (sign, log|det|) written by
`scripts/python/compute_perturbed_jacobian_det.py` for one sublayer
`{in}.{out}` and picks two rows A, B:
  - by default at random (--seed) from one (sample_idx, pos) group that has
    both signs, A with det > 0 and B with det < 0. det is continuous in the
    input, so the straight path between them must cross a singular Jacobian
    (det = 0) somewhere.
  - or explicitly with --row-a / --row-b (perturbed-sweep row indices).
Both rows must share `pos`. Sharing `sample_idx` too is recommended but not
required: with the same sample, every position before `pos` is identical in
A and B, and later positions can't affect the output at `pos` (causal
attention), so only the `pos` vector effectively moves along the path. With
different samples the earlier-position context is interpolated too, and a
warning is printed.

Interpolation is in the sublayer's input space, all through the existing
ONNX graphs:
  1. X_A, X_B = full (seq, d_model) `in` activations of the two perturbed
     sequences, via base_model_extra_outputs.onnx (the same graph the
     perturbed sweep used).
  2. x(t) = (1 - t) X_A + t X_B for --n-points evenly spaced t in [0, 1].
  3. The sublayer's ONNX forward subgraph runs on each x(t).
  4. vec_in = x(t)[pos], vec_out = f(x(t))[pos], the same per-point (n, d)
     in/out pair arrays the Lipschitz scripts build from the perturbed
     sweep (see scripts/lipschitz/lipschitz_ratio_bootstrap_eqd_sweep_gpu.py's
     compute_lipschitz_ratios_gpu).
At t = 0 and t = 1, vec_out is checked against the perturbed sweep's stored
`out` vec for rows A and B (--verify-atol).

Output: one small h5 per pair at
  {--out-dir}/{in}.{out}/row{A}_row{B}.h5
with datasets t (n,), vec_in (n, d_model), vec_out (n, d_model), and attrs
for both endpoints (row, sample_idx, pos, vocab_id, sign, log_abs_det). The
file is written once and is tiny (n * 2 * d_model floats), so it goes
straight to NFS rather than through a ShardedLayerWriter.

Usage (on mauao -- the perturbed sweep's shards live on its local scratch):
    uv run python -m scripts.python.generate_det_sign_interpolation configs/simple_stories.toml \\
        --sublayer add_4.add_6 --n-points 1000
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch

from config import load_config
from construct.construct import ensure_extra_outputs, ensure_subgraphs, exported_batch_size
from execute.io_onnx import run_iobinding_cpu_input_gpu_output
from execute.node_relay import ensure_local
from execute.sharded_writer import _scratch_root, load_shard_rows
from scripts.python.compute_perturbed_jacobian_det import _session  # TF32 off


def _read_rows(layer_key: str, rows: list[int], names: list[str]) -> dict[str, np.ndarray]:
    """Read just `rows` of a ShardedLayerWriter layer, opening only the shards
    that hold them -- instead of load_shard_rows' full concatenation (a
    perturbed-sweep layer is ~3 GB, and only two rows are needed here)."""
    with open(layer_key + ".manifest.json") as f:
        manifest = json.load(f)
    out = {name: [] for name in names}
    for row in rows:
        shard = next(
            (s for s in manifest["shards"] if s["row_start"] <= row < s["row_end"]), None,
        )
        if shard is None:
            raise ValueError(f"row {row} not in any completed shard of {layer_key}")
        if shard["pushed"]:
            path = ensure_local(shard["virtual_path"], dest_dir=_scratch_root())
        else:
            path = os.path.join(_scratch_root(), layer_key, f"shard_{shard['shard_idx']:05d}.h5")
        with h5py.File(path, "r") as h5f:
            for name in names:
                out[name].append(h5f[name][row - shard["row_start"]])
    return {name: np.stack(v) for name, v in out.items()}


def _pick_pair(det: dict[str, np.ndarray], rng: np.random.Generator) -> tuple[int, int]:
    """Random (A, B) with det_A > 0, det_B < 0, sharing (sample_idx, pos),
    drawn uniformly from the (sample_idx, pos) groups that contain both
    signs."""
    key = det["sample_idx"] * (det["pos"].max() + 1) + det["pos"]
    pos_keys = np.unique(key[det["sign"] > 0])
    neg_keys = np.unique(key[det["sign"] < 0])
    both = np.intersect1d(pos_keys, neg_keys)
    if both.size == 0:
        raise RuntimeError(
            "no (sample_idx, pos) group has both det signs in the det rows available "
            "so far -- wait for more of compute_perturbed_jacobian_det to finish, or "
            "pass --row-a/--row-b"
        )
    k = rng.choice(both)
    in_group = key == k
    a = rng.choice(np.flatnonzero(in_group & (det["sign"] > 0)))
    b = rng.choice(np.flatnonzero(in_group & (det["sign"] < 0)))
    return int(det["row_idx"][a]), int(det["row_idx"][b])


def generate_det_sign_interpolation(
    cfg,
    sublayer_name: str,
    det_dir: str,
    perturbed_sweep_dir: str,
    out_dir: str,
    n_points: int,
    row_a: int | None,
    row_b: int | None,
    seed: int,
    verify_atol: float,
) -> str:
    sub_graphs = ensure_subgraphs(cfg, force=False)
    matches = [(p, s) for p, s in sub_graphs if f"{s.input}.{s.output}" == sublayer_name]
    if not matches:
        raise ValueError(
            f"unknown --sublayer {sublayer_name!r}; config has "
            f"{[f'{s.input}.{s.output}' for _, s in sub_graphs]}"
        )
    [(fwd_path, sublayer)] = matches
    if sublayer.input == cfg.input_name:
        raise ValueError(f"{sublayer_name}: input is token ids, can't interpolate")

    det = load_shard_rows(".", os.path.join(det_dir, sublayer_name))
    print(f"{sublayer_name}: {len(det['row_idx'])} det rows available, "
          f"{int((det['sign'] > 0).sum())} positive, {int((det['sign'] < 0).sum())} negative, "
          f"{int((det['sign'] == 0).sum())} zero")

    if (row_a is None) != (row_b is None):
        raise ValueError("pass both --row-a and --row-b, or neither")
    if row_a is None:
        row_a, row_b = _pick_pair(det, np.random.default_rng(seed))

    # det rows are in increasing global row_idx order, but not necessarily
    # contiguous from 0 (compute_perturbed_jacobian_det --samples filters them)
    ends = {}
    for name, row in (("a", row_a), ("b", row_b)):
        i = int(np.searchsorted(det["row_idx"], row))
        if i >= len(det["row_idx"]) or det["row_idx"][i] != row:
            raise ValueError(f"--row-{name} {row} has no det result (not computed yet, or filtered out)")
        ends[name] = {k: det[k][i].item() for k in ("sample_idx", "pos", "vocab_id", "sign", "log_abs_det")}
        ends[name]["row"] = row
        print(f"  {name.upper()}: row {row} {ends[name]}")
    if ends["a"]["pos"] != ends["b"]["pos"]:
        raise ValueError(f"rows {row_a} and {row_b} have different pos -- pick rows at the same position")
    if ends["a"]["sample_idx"] != ends["b"]["sample_idx"]:
        print("  warning: different sample_idx -- the context before pos is interpolated too, "
              "not just the pos vector")
    if ends["a"]["sign"] == ends["b"]["sign"]:
        print("  warning: both endpoints have the same det sign -- the path need not cross det = 0")
    pos = ends["a"]["pos"]

    # 1. full `in` activations of both perturbed sequences
    with h5py.File(os.path.join(perturbed_sweep_dir, "base_seqs.h5"), "r") as f:
        base_seqs = f["base_seqs"][:]
    extra_path = ensure_extra_outputs(cfg, [s.output for s in cfg.sublayers], force=False)
    seq_batch = exported_batch_size(extra_path)
    extra_sess = _session(extra_path)
    token_ids = np.zeros((seq_batch, cfg.seq), dtype=np.int64)
    for i, e in enumerate((ends["a"], ends["b"])):
        token_ids[i] = base_seqs[e["sample_idx"]]
        token_ids[i, pos] = e["vocab_id"]
    (in_act,) = run_iobinding_cpu_input_gpu_output(
        extra_sess, extra_sess.get_inputs()[0].name, token_ids, [sublayer.input],
    )
    x_a, x_b = in_act[:2].to(torch.float32).cpu().numpy()  # (seq, d_model) each
    del extra_sess, in_act

    # 2-3. x(t) through the sublayer's forward subgraph, in fixed-batch chunks
    t = np.linspace(0.0, 1.0, n_points)
    fwd_batch = exported_batch_size(fwd_path)
    fwd_sess = _session(fwd_path)
    vec_in = np.empty((n_points, cfg.d_model), dtype=np.float32)
    vec_out = np.empty((n_points, cfg.d_model), dtype=np.float32)
    for s in range(0, n_points, fwd_batch):
        e = min(s + fwd_batch, n_points)
        x = np.zeros((fwd_batch, *x_a.shape), dtype=np.float32)
        tt = t[s:e, None, None].astype(np.float32)
        x[: e - s] = (1 - tt) * x_a + tt * x_b
        (y,) = run_iobinding_cpu_input_gpu_output(
            fwd_sess, fwd_sess.get_inputs()[0].name, x, [sublayer.output],
        )
        vec_in[s:e] = x[: e - s, pos]
        vec_out[s:e] = y[: e - s, pos].to(torch.float32).cpu().numpy()

    # 4. endpoints must reproduce the perturbed sweep's stored output vecs
    stored = _read_rows(os.path.join(perturbed_sweep_dir, sublayer.output), [row_a, row_b], ["vec"])["vec"]
    err = max(np.abs(vec_out[0] - stored[0]).max(), np.abs(vec_out[-1] - stored[1]).max())
    if err > verify_atol:
        raise RuntimeError(
            f"endpoint outputs differ from the perturbed sweep's stored vecs by {err:.3g} "
            f"(> --verify-atol {verify_atol})"
        )
    print(f"  endpoint check: max |f(x(0|1))[pos] - stored vec| = {err:.3g}")

    out_path = os.path.join(out_dir, sublayer_name, f"row{row_a}_row{row_b}.h5")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as h5f:
        h5f.create_dataset("t", data=t)
        h5f.create_dataset("vec_in", data=vec_in)
        h5f.create_dataset("vec_out", data=vec_out)
        h5f.attrs["sublayer_input"] = sublayer.input
        h5f.attrs["sublayer_output"] = sublayer.output
        h5f.attrs["pos"] = pos
        h5f.attrs["perturbed_sweep_dir"] = perturbed_sweep_dir
        h5f.attrs["det_dir"] = det_dir
        for name, e in ends.items():
            for k, v in e.items():
                h5f.attrs[f"{name}_{k}"] = v
    os.replace(tmp_path, out_path)

    in_step = np.linalg.norm(np.diff(vec_in, axis=0), axis=1)
    out_step = np.linalg.norm(np.diff(vec_out, axis=0), axis=1)
    ratio = out_step / np.maximum(in_step, np.finfo(np.float32).tiny)
    print(f"  |x_B - x_A| at pos = {np.linalg.norm(x_b[pos] - x_a[pos]):.4g}; "
          f"consecutive-step |dy|/|dx| range [{ratio.min():.4g}, {ratio.max():.4g}]")
    print(f"saved {n_points} interpolated (vec_in, vec_out) points -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("--sublayer", type=str, required=True, help="Sublayer as {in}.{out}, e.g. add_4.add_6")
    parser.add_argument(
        "--det-dir",
        type=str,
        default="onnx_models/simple_stories/perturbed_jacobian_det",
        help="--out-dir of compute_perturbed_jacobian_det.py (project-relative)",
    )
    parser.add_argument(
        "--perturbed-sweep-dir",
        type=str,
        default="onnx_models/simple_stories/perturbed_sweep",
        help="--out-dir of generate_perturbed_sweep.py (project-relative; holds base_seqs.h5)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/simple_stories/det_sign_interpolation",
        help="Output dir -- writes {out-dir}/{sublayer}/row{A}_row{B}.h5",
    )
    parser.add_argument("--n-points", type=int, default=1000, help="Interpolated points, t evenly spaced in [0, 1]")
    parser.add_argument("--row-a", type=int, default=None, help="Perturbed-sweep row of endpoint A (t = 0)")
    parser.add_argument("--row-b", type=int, default=None, help="Perturbed-sweep row of endpoint B (t = 1)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the random pair pick")
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-3,
        help="Max allowed |endpoint output - perturbed sweep's stored vec| before aborting",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_det_sign_interpolation(
        cfg,
        sublayer_name=args.sublayer,
        det_dir=args.det_dir,
        perturbed_sweep_dir=args.perturbed_sweep_dir,
        out_dir=args.out_dir,
        n_points=args.n_points,
        row_a=args.row_a,
        row_b=args.row_b,
        seed=args.seed,
        verify_atol=args.verify_atol,
    )


if __name__ == "__main__":
    main()
