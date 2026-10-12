"""Sample a sphere around the det = 0 crossing between two opposite-sign
perturbed points, and save (input, output, det) of every sphere point for
Lipschitz analysis.

Input is one `scripts/python/generate_det_sign_interpolation.py` file: the
straight path x(t) = (1 - t) X_A + t X_B between perturbed points A
(det > 0) and B (det < 0) of sublayer `{in}.{out}`, at sequence position
`pos`. det is continuous in the input, so it crosses 0 somewhere on the
path. This script:
  1. computes (sign, log|det|) of the d_model x d_model Jacobian block
     d out[pos] / d in[pos] at every path point, finds where the sign
     changes, and refines that crossing t* by bisection (--bisect-iters);
     if the sign changes more than once, --crossing picks which one(s)
     (indices or 'all'); steps 2-8 run once per chosen crossing;
  2. centres a sphere on c = x(t*)[pos] with radius r = |c - x_A[pos]|
     (--radius-to a, default) or |c - x_B[pos]| (--radius-to b), or a fixed
     --radius R (overrides --radius-to; output file tagged _r{R});
  3. samples --n-points points uniformly on that sphere in the d_model-dim
     space of the `pos` vector (Gaussian direction, normalized; --ball
     samples the solid ball uniformly instead). Every other position keeps
     its x(t*) value, so only the `pos` vector moves, as on the path;
  4. runs every sphere point (and the centre) through the sublayer's ONNX
     pullback graph, which returns the forward output and the Jacobian
     rows in one call (the same machinery as compute_perturbed_jacobian_det.py),
     giving vec_out = f(x_k)[pos] and (sign, log|det|) at each point;
  5. takes the full Jacobian J at the centre, J = U S V^T, and its right
     singular vector v_min for the smallest singular value -- the input
     direction the sublayer collapses most at the det = 0 point;
  6. samples --n-line points c + s d, s in [-r, r], along d = v_min and,
     for comparison, along the A -> B path direction; a control set of
     random points in the ball around c (--n-line points uniform in the ball
     of radius r, each with its own random ambient direction, drawn from
     --seed); and 2 * --n-line-log points along v_min, and as many random
     ball points (fresh direction each), at |s| log-spaced from --line-min-s
     to r (v_min: to --line-max-s if given), to probe close to c and, along
     v_min, beyond the sphere. All run through the same pullback;
  7. samples the same log-spaced s along v_min with noise: c + s v_min +
     eps |s| u, u a fresh random unit vector orthogonal to v_min per point
     (a tilt of ~eps rad), for each eps in --noise-levels;
  8. samples an --n-plane x --n-plane grid over the plane of the two
     smallest singular directions, c + a v_min + b v_2, a, b in
     [-extent, extent] (--plane-extent, default r), with det at each point.

Output: {--out-dir}/{in}.{out}/row{A}_row{B}_sphere_c{crossing:02d}.h5, one per crossing, with
  vec_in (n, d), vec_out (n, d), sign (n,), log_abs_det (n,)  -- sphere points
  center_in (d,), center_out (d,), center_sign, center_log_abs_det (attrs)
  path_t (m,), path_sign (m,), path_log_abs_det (m,)           -- det along A -> B
  singular_values (d,), v_min (d,), path_dir (d,)
  line_s (n_line,), line_{singular,path}/{s, vec_in, vec_out, sign, log_abs_det}
  line_control/...     -- random points uniform in the ball (s = their radius)
  line_{singular,control}_log/{s, vec_in, vec_out, sign, log_abs_det}  -- log-spaced |s|
                         (control: a fresh random direction per point)
  v_2 (d,); noisy/{eps (L,), s (m,), vec_in (L, m, d), vec_out (L, m, d)}
  plane/{ab (k,), vec_out (k, k, d), sign (k, k), log_abs_det (k, k)}  -- inputs are c + a v_min + b v_2
  attrs: t_star, radius, radius_to, all crossings, plus the interpolation
         file's endpoint attrs
Analysis/plots: scripts/analysis/plot_det_zero_sphere_lipschitz.py.

Usage (on mauao):
    uv run python -m scripts.python.generate_det_zero_sphere configs/simple_stories.toml \\
        onnx_models/simple_stories/det_sign_interpolation/add_4.add_6/row26685_row26140.h5
"""
import argparse
import os

import h5py
import numpy as np
import torch

from config import load_config
from construct.construct import ensure_extra_outputs, ensure_subgraphs, exported_batch_size
from execute.io_onnx import run_iobinding_cpu_input_gpu_output
from scripts.python.compute_perturbed_jacobian_det import (
    DEVICE,
    _block_slogdet,
    _session,
    jacobian_chunk,
    open_pullback,
)


def _endpoint_inputs(cfg, attrs: dict, input_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Full (seq, d_model) `in` activations X_A, X_B of the two perturbed
    sequences, via base_model_extra_outputs.onnx -- the same recomputation
    generate_det_sign_interpolation.py does."""
    with h5py.File(os.path.join(attrs["perturbed_sweep_dir"], "base_seqs.h5"), "r") as f:
        base_seqs = f["base_seqs"][:]
    extra_path = ensure_extra_outputs(cfg, [s.output for s in cfg.sublayers], force=False)
    sess = _session(extra_path)
    token_ids = np.zeros((exported_batch_size(extra_path), cfg.seq), dtype=np.int64)
    for i, end in enumerate("ab"):
        token_ids[i] = base_seqs[attrs[f"{end}_sample_idx"]]
        token_ids[i, attrs["pos"]] = attrs[f"{end}_vocab_id"]
    (act,) = run_iobinding_cpu_input_gpu_output(sess, sess.get_inputs()[0].name, token_ids, [input_name])
    act = act[:2].to(torch.float32).clone()
    return act[0], act[1]


def generate_det_zero_sphere(
    cfg,
    interp_path: str,
    out_dir: str,
    n_points: int,
    radius_to: str,
    ball: bool,
    fixed_radius: float | None,
    crossings: list[int] | None,
    bisect_iters: int,
    seed: int,
    n_line: int,
    n_line_log: int,
    line_min_s: float,
    line_max_s: float | None,
    noise_levels: list[float],
    n_plane: int,
    plane_extent: float | None,
) -> list[str]:
    with h5py.File(interp_path, "r") as f:
        path_t = f["t"][:]
        path_vec_in = f["vec_in"][:]
        attrs = dict(f.attrs)
    pos = int(attrs["pos"])
    sublayer_name = f"{attrs['sublayer_input']}.{attrs['sublayer_output']}"
    [(fwd_path, sublayer)] = [
        (p, s) for p, s in ensure_subgraphs(cfg, force=False) if f"{s.input}.{s.output}" == sublayer_name
    ]

    x_a, x_b = _endpoint_inputs(cfg, attrs, sublayer.input)
    err = np.abs(x_a[pos].cpu().numpy() - path_vec_in[0]).max()
    if err > 1e-5:
        raise RuntimeError(f"recomputed X_A[pos] differs from {interp_path}'s vec_in[0] by {err:.3g}")

    pb = open_pullback(fwd_path, sublayer)
    ppc = exported_batch_size(fwd_path) // cfg.d_model

    def x_of(t: torch.Tensor) -> torch.Tensor:  # (n,) -> (n, seq, d)
        t = t.to(torch.float32)[:, None, None]
        return (1 - t) * x_a + t * x_b

    def slogdet_out(x: torch.Tensor):
        pos_t = torch.full((len(x),), pos, dtype=torch.long, device=DEVICE)
        return _block_slogdet(pb[0], pb[1], x, pos_t, sublayer.input, sublayer.output, ppc, pb[2], pb[3])

    # 1. det along the path, then bisect the chosen sign change
    t_dev = torch.from_numpy(path_t).to(DEVICE)
    path_sign, path_logdet, _ = slogdet_out(x_of(t_dev))
    path_sign = path_sign.cpu().numpy().astype(np.int8)
    path_logdet = path_logdet.cpu().numpy()
    flips = np.flatnonzero(path_sign[:-1] != path_sign[1:])
    print(f"{sublayer_name} pos {pos}: det sign at A {path_sign[0]:+d}, at B {path_sign[-1]:+d}; "
          f"{len(flips)} sign change(s) along the path at t ~ {np.round(path_t[flips], 4).tolist()}")
    if len(flips) == 0:
        raise RuntimeError("det never changes sign along the path -- nothing to centre on")
    bad = [c for c in (crossings or []) if not -len(flips) <= c < len(flips)]
    if bad:
        raise ValueError(f"--crossing {bad} out of range for {len(flips)} sign change(s)")
    # each crossing gets its own sphere / lines / plane, centred on its own t*;
    # the det-along-path above and the ONNX sessions are shared
    todo = list(range(len(flips))) if crossings is None else sorted({c % len(flips) for c in crossings})
    out_paths = []
    for crossing in todo:
        print(f"-- crossing {crossing} / {len(flips) - 1} (t ~ {path_t[flips[crossing]]:.4f})")
        kf = flips[crossing]
        lo, hi = float(path_t[kf]), float(path_t[kf + 1])
        sign_lo = path_sign[kf]
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            s, _, _ = slogdet_out(x_of(torch.tensor([mid], device=DEVICE)))
            if int(s.item()) == sign_lo:
                lo = mid
            else:
                hi = mid
        t_star = 0.5 * (lo + hi)

        # 2. sphere centre and radius
        center = x_of(torch.tensor([t_star], device=DEVICE))[0]  # (seq, d)
        if fixed_radius is not None:
            radius = fixed_radius
        else:
            end = x_a if radius_to == "a" else x_b
            radius = float(torch.linalg.norm(center[pos] - end[pos]))

        # 3. uniform points on the sphere (or in the ball) around center[pos]
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        u = torch.randn((n_points, cfg.d_model), generator=gen, device=DEVICE, dtype=torch.float64)
        u /= torch.linalg.norm(u, dim=1, keepdim=True)
        if ball:
            u *= torch.rand((n_points, 1), generator=gen, device=DEVICE, dtype=torch.float64) ** (1 / cfg.d_model)
        x = center.expand(n_points + 1, -1, -1).clone()
        x[1:, pos] += (radius * u).to(torch.float32)  # row 0 is the centre itself

        # 4. outputs + det at the centre and every sphere point
        sign, logdet, out = slogdet_out(x)
        sign = sign.cpu().numpy().astype(np.int8)
        logdet = logdet.cpu().numpy()
        vec_in = x[:, pos].cpu().numpy()
        vec_out = out.cpu().numpy()

        # 5. the centre's Jacobian and its most-collapsed input direction: J = U S V^T
        #    with J[i, j] = d out_i / d in_j, so V's last row (right singular vector of
        #    the smallest singular value) is the input direction J shrinks the most
        jac_c, _ = jacobian_chunk(
            pb[0], pb[1], center[None], torch.tensor([pos], device=DEVICE),
            sublayer.input, sublayer.output, pb[2], pb[3],
        )
        _, svals, vh = torch.linalg.svd(jac_c[0].to(torch.float64))
        v_min = vh[-1]
        path_dir = (x_b[pos] - x_a[pos]).to(torch.float64)
        path_dir /= torch.linalg.norm(path_dir)

        # 6. lines through the centre along v_min and the path direction (other
        #    positions fixed at x(t*)), and, for comparison with v_min, a control
        #    set of random points in the ball around c: each point has its own
        #    uniformly random direction (ambient, not projected off v_min)
        def random_dirs(n: int) -> torch.Tensor:
            u = torch.randn((n, cfg.d_model), generator=gen, device=DEVICE, dtype=torch.float64)
            return u / torch.linalg.norm(u, dim=1, keepdim=True)
        line_s = torch.linspace(-radius, radius, n_line, device=DEVICE, dtype=torch.float64)
        # log-spaced |s| from line_min_s to r (both signs; v_min out to line_max_s
        # if given, to follow the collapsed direction past the sphere), to see how the output
        # behaves close to the centre -- kept separate from the uniform line,
        # whose pair ratios would be swamped by output noise if it had
        # near-coincident points bunched at the centre
        def log_grid(s_max: float) -> torch.Tensor:
            log_abs = torch.logspace(np.log10(line_min_s), np.log10(s_max), n_line_log, device=DEVICE, dtype=torch.float64)
            return torch.cat([-log_abs.flip(0), log_abs])

        def sample_line(direction: torch.Tensor, s_grid: torch.Tensor) -> dict[str, np.ndarray]:
            xl = center.expand(len(s_grid), -1, -1).clone()
            xl[:, pos] += (s_grid[:, None] * direction).to(torch.float32)
            l_sign, l_logdet, l_out = slogdet_out(xl)
            return {
                "s": s_grid.cpu().numpy(), "vec_in": xl[:, pos].cpu().numpy(), "vec_out": l_out.cpu().numpy(),
                "sign": l_sign.cpu().numpy().astype(np.int8), "log_abs_det": l_logdet.cpu().numpy(),
            }

        lines = {}
        for name, direction in (("singular", v_min), ("path", path_dir)):
            lines[name] = sample_line(direction, line_s)
        lines["singular_log"] = sample_line(v_min, log_grid(line_max_s or radius))
        # control: n_line points uniform in the ball of radius r (radius
        # r U^(1/d) -- in d_model dims almost all of them sit near r), and the
        # log-spaced radii of the v_min line (up to r) each with a fresh random
        # direction, so the ball is probed across the same scales as v_min
        ball_s = radius * torch.rand(n_line, generator=gen, device=DEVICE, dtype=torch.float64) ** (1 / cfg.d_model)
        lines["control"] = sample_line(random_dirs(n_line), ball_s)
        ctrl_log_s = log_grid(radius).abs()
        lines["control_log"] = sample_line(random_dirs(len(ctrl_log_s)), ctrl_log_s)

        # 7. noisy kernel direction: c + s v_min + eps |s| u, u a fresh random unit
        #    vector orthogonal to v_min per point -- a tilt of ~eps rad off v_min --
        #    on the same log-spaced s grid as singular_log, one set per noise level
        s_noisy = log_grid(line_max_s or radius)
        noisy = []
        for eps in noise_levels:
            u = torch.randn((len(s_noisy), cfg.d_model), generator=gen, device=DEVICE, dtype=torch.float64)
            u -= (u @ v_min)[:, None] * v_min
            u /= torch.linalg.norm(u, dim=1, keepdim=True)
            xl = center.expand(len(s_noisy), -1, -1).clone()
            xl[:, pos] += (s_noisy[:, None] * v_min + eps * s_noisy.abs()[:, None] * u).to(torch.float32)
            _, _, l_out = slogdet_out(xl)
            noisy.append((xl[:, pos].cpu().numpy(), l_out.cpu().numpy()))

        # 8. the plane of the two smallest singular directions (v_min, v_2): an
        #    n_plane x n_plane grid of c + a v_min + b v_2, a, b in [-extent, extent]
        v_2 = vh[-2]
        extent = plane_extent or radius
        ab = torch.linspace(-extent, extent, n_plane, device=DEVICE, dtype=torch.float64)
        ga, gb = torch.meshgrid(ab, ab, indexing="ij")
        xp = center.expand(n_plane * n_plane, -1, -1).clone()
        xp[:, pos] += (ga.reshape(-1, 1) * v_min + gb.reshape(-1, 1) * v_2).to(torch.float32)
        p_sign, p_logdet, p_out = slogdet_out(xp)
        plane = {
            "ab": ab.cpu().numpy(),
            "vec_out": p_out.cpu().numpy().reshape(n_plane, n_plane, -1),
            "sign": p_sign.cpu().numpy().astype(np.int8).reshape(n_plane, n_plane),
            "log_abs_det": p_logdet.cpu().numpy().reshape(n_plane, n_plane),
        }
        svals_np = svals.cpu().numpy()
        print(f"  J at centre: sigma_min = {svals_np[-1]:.3g}, next {svals_np[-2]:.3g}, sigma_max = {svals_np[0]:.3g}; "
              f"|cos(v_min, path direction)| = {abs(float(v_min @ path_dir)):.3f}")

        print(f"  t* = {t_star:.6f} (bisected over [{path_t[kf]:.4f}, {path_t[kf + 1]:.4f}]), "
              f"centre log|det| = {logdet[0]:.3f}; radius ({'fixed' if fixed_radius is not None else 'to ' + radius_to.upper()}) = {radius:.4g}")
        print(f"  {n_points} {'ball' if ball else 'sphere'} points: {int((sign[1:] > 0).sum())} det > 0, "
              f"{int((sign[1:] < 0).sum())} det < 0; log|det| range [{logdet[1:].min():.3g}, {logdet[1:].max():.3g}]")

        out_path = os.path.join(out_dir, sublayer_name,
                                f"row{attrs['a_row']}_row{attrs['b_row']}_{'ball' if ball else 'sphere'}"
                                f"{'' if fixed_radius is None else f'_r{fixed_radius:g}'}_c{crossing:02d}.h5")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as h5f:
            h5f.create_dataset("vec_in", data=vec_in[1:])
            h5f.create_dataset("vec_out", data=vec_out[1:])
            h5f.create_dataset("sign", data=sign[1:])
            h5f.create_dataset("log_abs_det", data=logdet[1:])
            h5f.create_dataset("center_in", data=vec_in[0])
            h5f.create_dataset("center_out", data=vec_out[0])
            h5f.create_dataset("path_t", data=path_t)
            h5f.create_dataset("path_sign", data=path_sign)
            h5f.create_dataset("path_log_abs_det", data=path_logdet)
            h5f.create_dataset("singular_values", data=svals_np)
            h5f.create_dataset("v_min", data=v_min.cpu().numpy())
            h5f.create_dataset("path_dir", data=path_dir.cpu().numpy())
            h5f.create_dataset("line_s", data=line_s.cpu().numpy())
            h5f.attrs["control_kind"] = "ball"  # line_control*: random points in the ball, not one direction
            h5f.create_dataset("v_2", data=v_2.cpu().numpy())
            h5f.create_dataset("noisy/eps", data=np.asarray(noise_levels, dtype=np.float64))
            h5f.create_dataset("noisy/s", data=s_noisy.cpu().numpy())
            h5f.create_dataset("noisy/vec_in", data=np.stack([n[0] for n in noisy]))
            h5f.create_dataset("noisy/vec_out", data=np.stack([n[1] for n in noisy]))
            for k, v in plane.items():
                h5f.create_dataset(f"plane/{k}", data=v)
            for name, data in lines.items():
                for k, v in data.items():
                    h5f.create_dataset(f"line_{name}/{k}", data=v)
            for key, val in attrs.items():
                h5f.attrs[key] = val
            h5f.attrs["interpolation_file"] = interp_path
            h5f.attrs["t_star"] = t_star
            h5f.attrs["crossings_t"] = path_t[flips]
            h5f.attrs["crossing"] = crossing
            h5f.attrs["radius"] = radius
            h5f.attrs["radius_to"] = "fixed" if fixed_radius is not None else radius_to
            h5f.attrs["ball"] = ball
            h5f.attrs["seed"] = seed
            h5f.attrs["center_sign"] = int(sign[0])
            h5f.attrs["center_log_abs_det"] = float(logdet[0])
        os.replace(tmp_path, out_path)
        print(f"saved {n_points} sphere points -> {out_path}")
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("interpolation", type=str, help="generate_det_sign_interpolation.py output h5")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="onnx_models/simple_stories/det_zero_sphere",
        help="Output dir -- writes {out-dir}/{sublayer}/row{A}_row{B}_sphere_c{crossing:02d}.h5",
    )
    parser.add_argument("--n-points", type=int, default=1000, help="Points sampled on the sphere")
    parser.add_argument(
        "--radius-to",
        choices=["a", "b"],
        default="a",
        help="Radius = distance from the crossing to endpoint A (det > 0) or B (det < 0)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Fixed sphere/ball radius (overrides --radius-to); the lines, noise and plane grids "
        "scale with it too. Output file gets an _r{R} tag so it doesn't overwrite the default-radius run",
    )
    parser.add_argument("--ball", action="store_true", help="Sample the solid ball uniformly instead of the sphere")
    parser.add_argument(
        "--crossing",
        nargs="+",
        default=["0"],
        help="Which sign change(s) along A -> B to centre on: indices (0 = first, -1 = last) or 'all'. "
        "Each gets its own output file, ..._c{index:02d}.h5",
    )
    parser.add_argument("--bisect-iters", type=int, default=25, help="Bisection steps refining t*")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the sphere samples")
    parser.add_argument(
        "--n-line",
        type=int,
        default=401,
        help="Points on each line through the centre (s in [-r, r], along v_min and along the path)",
    )
    parser.add_argument(
        "--n-line-log",
        type=int,
        default=200,
        help="Log-spaced points per sign of s along v_min, |s| from --line-min-s to r",
    )
    parser.add_argument(
        "--line-min-s",
        type=float,
        default=1e-5,
        help="Smallest |s| (input distance from the centre) of the log-spaced v_min samples",
    )
    parser.add_argument(
        "--line-max-s",
        type=float,
        default=None,
        help="Largest |s| of the log-spaced v_min samples (default: the radius r). Can exceed r, "
        "to follow the collapsed direction beyond the sphere; the control line stays within r",
    )
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1],
        help="Relative noise eps for the noisy v_min samples: each point is c + s v_min + eps |s| u, "
        "u random unit orthogonal to v_min (a tilt of ~eps rad); same s grid as the log-spaced v_min line",
    )
    parser.add_argument("--n-plane", type=int, default=101, help="Grid size per axis in the (v_min, v_2) plane")
    parser.add_argument(
        "--plane-extent",
        type=float,
        default=None,
        help="Half-width of the (v_min, v_2) grid (default: the radius r)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    crossings = None if args.crossing == ["all"] else [int(c) for c in args.crossing]
    generate_det_zero_sphere(
        cfg,
        interp_path=args.interpolation,
        out_dir=args.out_dir,
        n_points=args.n_points,
        radius_to=args.radius_to,
        ball=args.ball,
        fixed_radius=args.radius,
        crossings=crossings,
        bisect_iters=args.bisect_iters,
        seed=args.seed,
        n_line=args.n_line,
        n_line_log=args.n_line_log,
        line_min_s=args.line_min_s,
        line_max_s=args.line_max_s,
        noise_levels=args.noise_levels,
        n_plane=args.n_plane,
        plane_extent=args.plane_extent,
    )


if __name__ == "__main__":
    main()
