"""Project the det = 0 crossing between two opposite-sign perturbed points
onto the real activation manifold, and sample around that manifold point:
A on the manifold, B in the ambient ball, C along the kernel direction.

Input is one `scripts/python/generate_det_sign_interpolation.py` file (the
straight path x(t) = (1 - t) X_A + t X_B, A det > 0, B det < 0, sublayer
`{in}.{out}`, position `pos`), exactly as for generate_det_zero_sphere.py.
The manifold is the real `in`-layer point cloud of A's (sample_idx, pos)
in the perturbed sweep (load_sample_cloud: one point per vocab substitution
at `pos`) -- the same cloud A and B themselves are points of. No autoencoder
is involved: the projection and the checks use the model-free validators
from gmanifold/validate.py's support_report (TangentCharts + KernelScore),
ported here since gmanifold itself is unimportable.

  1. det along the path, bisect the chosen sign change(s) to t*, c = x(t*)[pos]
     (same as generate_det_zero_sphere.py, --crossing / --bisect-iters);
  2. local tangent charts of the cloud (gmanifold TangentCharts): chart j is
     X_j + span(U_j), U_j the top-m PCA directions of X_j's K neighbours
     (m = --m, default the TwoNN intrinsic dimension; K = max(4m, 32));
  3. nearest manifold point p to c: project c onto the charts of the
     --n-cand nearest anchors, keep the one with the smallest ||c - p_j||,
     re-query the anchors around that foot point and repeat until the
     anchor is stable (--project-iters). The in-chart step from X_j is
     capped at --chart-reach x X_j's spacing, so a chart is never followed
     beyond where its linearisation is meaningful;
  4. p replaces c as the base: base = x(t*) with base[pos] = p. det and the
     full Jacobian are computed at p (and at c, for comparison);
  5. around p, radius rho (--radius, else --radius-units x p's anchor
     spacing), every point running through the sublayer pullback:
       A      --n-points on the manifold by convex neighbour sampling: step from
              p along sum_j w_j (x_j - p), x_j p's --n-nbrs nearest real points,
              w ~ Dirichlet(1), scaled to radius rho U^(1/k). (Local-PCA tangent
              directions score like ambient ones on these clouds -- at ~4k points in
              384-d the local spectrum is flat -- while directions into the hull
              of the nearby real points keep the kernel score at real-data levels.)
       B      --n-points uniform in the d_model-ball around p (ambient),
       C      --n-line points p + s v_min, s in [-rho, rho], v_min = J(p)'s
              smallest right singular vector (the kernel direction at p),
     and around the critical point c itself, same radius rho:
       cB     --n-points uniform in the d_model-ball around c (ambient),
       cC     --n-line points c + s v_min(c), the kernel direction at c,
     plus *_log versions of all five at |s| log-spaced from --line-min-s to
     rho (A, B, cB: a fresh random direction per point; C, cC: both signs);
  6. per point: chart residual (normal distance to the nearest anchor's
     chart, in spacing units -- 0 on the manifold) and KernelScore u
     (fitted on a 90% split of the cloud; ~1 on the manifold; --kernel, default
     gauss, or the paper's regularized Laplace kernel), with
     calibration bands from the held-out 10% (real, real + 0.5x / 1x
     spacing noise).

Output: {--out-dir}/{in}.{out}/row{A}_row{B}_manifold[_r{R}][_{tag}]_c{crossing:02d}.h5 -- existing files
are never replaced unless --overwrite is given (the run stops before any work instead)
  c (d,), p (d,), p_out (d,), tangent_basis (d, m)
  path_t, path_sign, path_log_abs_det
  singular_values_{p,c} (d,), v_min_{p,c} (d,)
  {A,B,C,cB,cC}{,_log}/{vec_in, vec_out, sign, log_abs_det, dist, residual, u}
      dist = ||x - p|| (A, B, C) or ||x - c|| (cB, cC); C*/s, cC*/s = signed offset along v_min
  c_out (d,)
  calib/{u_heldout, u_noise05, u_noise1, residual_noise05, residual_noise1}
  attrs: t_star, crossing, dist_c_p, dist_c_p_units, spacing, anchor, m, K,
         radius, p_sign, p_log_abs_det, c_sign, c_log_abs_det,
         tangent_overlap_{p,c} (= ||U^T v_min||^2), cos_vmin_pc, plus the
         interpolation file's endpoint attrs
The file is written once and is small, so it goes straight to NFS (like the
sphere script), not through a ShardedLayerWriter.

Usage (on mauao -- the perturbed sweep's shards live on its local scratch):
    uv run python -m scripts.python.generate_det_zero_manifold configs/simple_stories.toml \\
        onnx_models/simple_stories/det_sign_interpolation/add_4.add_6/row26685_row26140.h5
"""
import argparse
import os

import h5py
import numpy as np
import torch

from autoencoder.batched_autoencoder import knn_single
from autoencoder.data import load_sample_cloud
from config import load_config
from construct.construct import ensure_subgraphs, exported_batch_size
from scripts.python.compute_perturbed_jacobian_det import DEVICE, _block_slogdet, jacobian_chunk, open_pullback
from scripts.python.generate_det_zero_sphere import _endpoint_inputs
from scripts.python.plot_autoencoder_sample_diagnostics import KernelScore


@torch.no_grad()
def twonn_dimension(X: torch.Tensor) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017), duplicates removed --
    ported from gmanifold.geometry.intrinsic_dimension."""
    X = torch.unique(X, dim=0)
    dist, _ = knn_single(X, 2)
    mu = (dist[:, 1] / dist[:, 0].clamp_min(1e-12)).sort().values
    keep = int(len(mu) * 0.9)
    mu = mu[:keep]
    F = torch.arange(1, keep + 1, dtype=mu.dtype, device=mu.device) / len(X)
    x, y = mu.log(), -(1 - F).log()
    return float((x * y).sum() / (x * x).sum())


class TangentCharts:
    """Local PCA charts of a point cloud, ported from gmanifold.tangent.TangentCharts
    (residual as the validator), plus `project`: the nearest point on the
    charts to an arbitrary ambient point."""

    def __init__(self, X: torch.Tensor, m: int, K: int | None = None, K_s: int = 8) -> None:
        self.X, self.m = X.float(), m
        self.K = K or max(4 * m, 32)
        if self.K <= m:
            raise ValueError(f"K={self.K} neighbours cannot span an m={m} tangent space (need K > m)")
        dist, self.idx = knn_single(self.X, self.K)
        self.r = dist[:, K_s - 1]  # local spacing of each anchor
        hub = float(self.r[self.idx[:, 0]].median()) < 0.5 * float(self.r.median())
        self.r_unit = self.r.median().expand_as(self.r) if hub else self.r  # length unit for residuals

    @torch.no_grad()
    def bases(self, anchors: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """Tangent bases U (n, D, m) of the given anchors: centred SVD of each K-neighbourhood."""
        U = []
        for s in range(0, len(anchors), chunk):
            nb = self.X[self.idx[anchors[s:s + chunk]]]
            nb = nb - nb.mean(1, keepdim=True)
            U.append(torch.linalg.svd(nb, full_matrices=False)[2][:, :self.m].transpose(1, 2))
        return torch.cat(U)

    @torch.no_grad()
    def residual(self, Y: torch.Tensor) -> torch.Tensor:
        """Normal residual to the nearest anchor's tangent chart / that anchor's spacing (0 on the chart)."""
        Y = Y.to(self.X.device).float()
        j = knn_single(self.X, 1, Y)[1][:, 0]
        uniq, inv = torch.unique(j, return_inverse=True)
        U = self.bases(uniq)[inv]
        d = Y - self.X[j]
        u = (d[:, None] @ U)[:, 0]
        return (d - (U @ u[..., None])[..., 0]).norm(dim=1) / self.r_unit[j]

    @torch.no_grad()
    def project(
        self, Y: torch.Tensor, n_cand: int = 16, reach: float = 2.0, iters: int = 10, chunk: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nearest chart point P to each row of Y: (P (n, D), anchor (n,), step (n,)).

        Each pass projects Y onto the charts of the n_cand anchors nearest
        the current foot point (Y itself on the first pass), with the
        in-chart step ||U^T (Y - X_j)|| capped at reach * r_j, and keeps the
        chart giving the smallest ||Y - P||; stops once no row's anchor
        changes. step is the final in-chart step in units of r_anchor."""
        Y = Y.to(self.X.device).float()
        outs = []
        for s in range(0, len(Y), chunk):
            y = Y[s:s + chunk]
            P, anchor, step = y, None, None
            for _ in range(iters):
                cand = knn_single(self.X, n_cand, P)[1]  # (n, c)
                uniq, inv = torch.unique(cand, return_inverse=True)
                U = self.bases(uniq)[inv]  # (n, c, D, m)
                d = y[:, None] - self.X[cand]  # (n, c, D)
                u = torch.einsum("ncd,ncdm->ncm", d, U)
                cap = reach * self.r[cand]
                u = u * (cap / u.norm(dim=-1).clamp_min(1e-12)).clamp(max=1.0)[..., None]
                foot = self.X[cand] + torch.einsum("ncdm,ncm->ncd", U, u)
                best = (y[:, None] - foot).norm(dim=-1).argmin(1)
                ar = torch.arange(len(y), device=y.device)
                new_anchor = cand[ar, best]
                P = foot[ar, best]
                step = u[ar, best].norm(dim=-1) / self.r[new_anchor]
                if anchor is not None and torch.equal(new_anchor, anchor):
                    break
                anchor = new_anchor
            outs.append((P, new_anchor, step))
        return tuple(torch.cat(t) for t in zip(*outs))


def generate_det_zero_manifold(
    cfg,
    interp_path: str,
    out_dir: str,
    crossings: list[int] | None,
    bisect_iters: int,
    m: int | None,
    n_cand: int,
    chart_reach: float,
    project_iters: int,
    fixed_radius: float | None,
    radius_units: float,
    n_points: int,
    n_line: int,
    n_line_log: int,
    line_min_s: float,
    n_nbrs: int,
    seed: int,
    kernel: str = "gauss",
    tag: str = "",
    log_radii: tuple[float, float] | None = None,
    overwrite: bool = False,
    kernel_r: float = 1e-3,
) -> list[str]:
    with h5py.File(interp_path, "r") as f:
        path_t = f["t"][:]
        path_vec_in = f["vec_in"][:]
        attrs = dict(f.attrs)
    pos = int(attrs["pos"])
    sample_idx = int(attrs["a_sample_idx"])
    if int(attrs["b_sample_idx"]) != sample_idx:
        print(f"WARNING: A (sample {sample_idx}) and B (sample {attrs['b_sample_idx']}) differ -- "
              f"using sample {sample_idx}'s cloud, but the path's context positions are interpolated too")
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

    def svd_at(x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """J = d out[pos] / d in[pos] at one (seq, d) input: (singular values desc, v_min)."""
        jac, _ = jacobian_chunk(pb[0], pb[1], x_seq[None], torch.tensor([pos], device=DEVICE),
                                sublayer.input, sublayer.output, pb[2], pb[3])
        _, svals, vh = torch.linalg.svd(jac[0].to(torch.float64))
        return svals, vh[-1]

    # the manifold: A's (sample, pos) `in` cloud -- A and B are points of it
    cloud = load_sample_cloud(attrs["perturbed_sweep_dir"], sublayer.input, sample_idx)[pos].to(DEVICE).float()
    for end, x_end in (("A", x_a), ("B", x_b)):
        d_end = float(torch.cdist(x_end[pos][None].float(), cloud).min())
        print(f"{end} -> nearest cloud point: {d_end:.3g}")
        if d_end > 1e-3 * float(x_end[pos].norm()):
            raise RuntimeError(f"endpoint {end} is not a point of the (sample {sample_idx}, pos {pos}) "
                               f"{sublayer.input} cloud (distance {d_end:.3g}) -- wrong cloud?")
    m = m or max(1, round(twonn_dimension(cloud)))
    charts = TangentCharts(cloud, m)
    print(f"cloud {tuple(cloud.shape)}: m = {m}, K = {charts.K}, median spacing {float(charts.r.median()):.4g}")

    # KernelScore on a 90% split; the held-out 10% (+ spacing-scaled noise) gives the calibration bands
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    torch.manual_seed(seed)  # Dirichlet draws in set A (torch.distributions takes no generator)
    perm = torch.randperm(len(cloud), generator=gen, device=DEVICE)
    n_hold = max(1, len(cloud) // 10)
    held, fit = cloud[perm[:n_hold]], cloud[perm[n_hold:]]
    kern = KernelScore(fit, seed=seed, kernel=kernel, r=kernel_r)
    j_held = knn_single(fit, 1, held)[1][:, 0]
    unit_held = knn_single(fit, 8)[0][:, 7][j_held]
    nz = torch.randn(held.shape, generator=gen, device=DEVICE)
    nz = nz / nz.norm(dim=1, keepdim=True) * unit_held[:, None]
    calib = {
        "u_heldout": kern(held), "u_noise05": kern(held + 0.5 * nz), "u_noise1": kern(held + nz),
        "residual_noise05": charts.residual(held + 0.5 * nz), "residual_noise1": charts.residual(held + nz),
    }
    print("calibration (median): u held-out {:.3f}, +0.5x noise {:.3f}, +1x noise {:.3f}; "
          "residual +0.5x {:.3f}, +1x {:.3f}".format(*(float(v.median()) for v in calib.values())))

    # 1. det along the path, then bisect the chosen sign change(s)
    t_dev = torch.from_numpy(path_t).to(DEVICE)
    path_sign, path_logdet, _ = slogdet_out(x_of(t_dev))
    path_sign = path_sign.cpu().numpy().astype(np.int8)
    path_logdet = path_logdet.cpu().numpy()
    flips = np.flatnonzero(path_sign[:-1] != path_sign[1:])
    print(f"{sublayer_name} pos {pos}: {len(flips)} sign change(s) along the path at t ~ "
          f"{np.round(path_t[flips], 4).tolist()}")
    if len(flips) == 0:
        raise RuntimeError("det never changes sign along the path -- nothing to centre on")
    bad = [c for c in (crossings or []) if not -len(flips) <= c < len(flips)]
    if bad:
        raise ValueError(f"--crossing {bad} out of range for {len(flips)} sign change(s)")
    todo = list(range(len(flips))) if crossings is None else sorted({c % len(flips) for c in crossings})

    def path_for(crossing: int) -> str:
        return os.path.join(out_dir, sublayer_name,
                            f"row{attrs['a_row']}_row{attrs['b_row']}_manifold"
                            f"{'' if fixed_radius is None else f'_r{fixed_radius:g}'}{f'_{tag}' if tag else ''}"
                            f"_c{crossing:02d}.h5")

    # never silently replace an earlier run: refuse up front, before any of the expensive work
    existing = [path_for(c) for c in todo if os.path.exists(path_for(c))]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} output file(s) already exist, e.g. {existing[0]} -- pass a new "
                              f"--out-dir or --tag to keep both runs, or --overwrite to replace them")

    out_paths = []
    for crossing in todo:
        kf = flips[crossing]
        lo, hi = float(path_t[kf]), float(path_t[kf + 1])
        for _ in range(bisect_iters):
            mid = 0.5 * (lo + hi)
            s, _, _ = slogdet_out(x_of(torch.tensor([mid], device=DEVICE)))
            if int(s.item()) == path_sign[kf]:
                lo = mid
            else:
                hi = mid
        t_star = 0.5 * (lo + hi)
        center = x_of(torch.tensor([t_star], device=DEVICE))[0]  # (seq, d)
        c = center[pos]

        # 3. nearest manifold point to c, which becomes the base
        p, anchor, step = charts.project(c[None], n_cand=n_cand, reach=chart_reach, iters=project_iters)
        p, anchor, step = p[0], int(anchor[0]), float(step[0])
        spacing = float(charts.r[anchor])
        dist_cp = float((c - p).norm())
        U = charts.bases(torch.tensor([anchor], device=DEVICE))[0]  # (D, m)
        base = center.clone()
        base[pos] = p

        # 4. det / Jacobian at p and at c
        sign0, logdet0, out0 = slogdet_out(torch.stack([center, base]))
        svals_c, v_min_c = svd_at(center)
        svals_p, v_min_p = svd_at(base)
        U64 = U.to(torch.float64)
        overlap_p = float(((U64.T @ v_min_p) ** 2).sum())
        overlap_c = float(((U64.T @ v_min_c) ** 2).sum())
        radius = fixed_radius if fixed_radius is not None else radius_units * spacing
        print(f"-- crossing {crossing}: t* = {t_star:.6f}; |c - p| = {dist_cp:.4g} "
              f"({dist_cp / spacing:.2f} spacings; in-chart step {step:.2f}); "
              f"u(c) = {float(kern(c[None])):.3f}, u(p) = {float(kern(p[None])):.3f}")
        print(f"   det: c sign {int(sign0[0]):+d} log|det| {float(logdet0[0]):.3f}; "
              f"p sign {int(sign0[1]):+d} log|det| {float(logdet0[1]):.3f}")
        print(f"   sigma_min c {float(svals_c[-1]):.3g}, p {float(svals_p[-1]):.3g} (sigma_max p {float(svals_p[0]):.3g}); "
              f"|cos(v_min_p, v_min_c)| = {abs(float(v_min_p @ v_min_c)):.3f}; "
              f"||U^T v_min||^2 p {overlap_p:.3f}, c {overlap_c:.3f} (random: {m / cfg.d_model:.3f})")

        # 5. around p: A (tangent, re-projected), B (ambient), C (v_min(p) line);
        #    around c: cB (ambient), cC (v_min(c) line) -- dist is from each set's own centre
        def run(vecs: torch.Tensor, ref: torch.Tensor) -> dict[str, np.ndarray]:
            x = base.expand(len(vecs), -1, -1).clone()
            x[:, pos] = vecs.to(torch.float32)
            sgn, ld, out = slogdet_out(x)
            return {
                "vec_in": vecs.float().cpu().numpy(), "vec_out": out.cpu().numpy(),
                "sign": sgn.cpu().numpy().astype(np.int8), "log_abs_det": ld.cpu().numpy(),
                "dist": (vecs.float() - ref).norm(dim=1).cpu().numpy(),
                "residual": charts.residual(vecs).cpu().numpy(), "u": kern(vecs.float()).cpu().numpy(),
            }

        def unit_dirs(n: int, k: int) -> torch.Tensor:
            g = torch.randn((n, k), generator=gen, device=DEVICE)
            return g / g.norm(dim=1, keepdim=True)

        nbrs = cloud[knn_single(cloud, n_nbrs, p[None])[1][0]]  # p's nearest real points (k, D)
        dirichlet = torch.distributions.Dirichlet(torch.ones(n_nbrs, device=DEVICE))

        def convex_pts(radii: torch.Tensor) -> torch.Tensor:
            """p + radius * unit(sum_j w_j (x_j - p)), w ~ Dirichlet(1): directions into the hull of p's
            nearest real points (torch.distributions has no generator arg; seeded via torch.manual_seed)."""
            dvec = dirichlet.sample((len(radii),)) @ (nbrs - p)
            return p + radii[:, None] * dvec / dvec.norm(dim=1, keepdim=True)

        log_r = torch.logspace(np.log10(line_min_s), np.log10(radius), n_line_log, device=DEVICE)
        if log_radii is None:  # volume-uniform balls, evenly spaced lines
            ball_k = radius * torch.rand(n_points, generator=gen, device=DEVICE) ** (1 / n_nbrs)
            ball_d = radius * torch.rand(n_points, generator=gen, device=DEVICE) ** (1 / cfg.d_model)
            ball_dc = radius * torch.rand(n_points, generator=gen, device=DEVICE) ** (1 / cfg.d_model)
            line_s = torch.linspace(-radius, radius, n_line, device=DEVICE)
        else:  # radii (|s| on the lines, random sign) log-uniform in [lo, hi]: every scale equally sampled
            lo, hi = np.log10(log_radii[0]), np.log10(log_radii[1])

            def log_uniform(n: int) -> torch.Tensor:
                return 10 ** (lo + (hi - lo) * torch.rand(n, generator=gen, device=DEVICE))
            ball_k, ball_d, ball_dc = log_uniform(n_points), log_uniform(n_points), log_uniform(n_points)
            sign = torch.where(torch.rand(n_line, generator=gen, device=DEVICE) < 0.5, -1.0, 1.0)
            line_s = (sign * log_uniform(n_line)).sort().values
        log_s = torch.cat([-log_r.flip(0), log_r])
        v, vc = v_min_p.to(torch.float32), v_min_c.to(torch.float32)
        sets = {
            "A": run(convex_pts(ball_k), p),
            "A_log": run(convex_pts(log_r), p),
            "B": run(p + unit_dirs(n_points, cfg.d_model) * ball_d[:, None], p),
            "B_log": run(p + unit_dirs(n_line_log, cfg.d_model) * log_r[:, None], p),
            "C": run(p + line_s[:, None] * v, p) | {"s": line_s.cpu().numpy()},
            "C_log": run(p + log_s[:, None] * v, p) | {"s": log_s.cpu().numpy()},
            "cB": run(c + unit_dirs(n_points, cfg.d_model) * ball_dc[:, None], c),
            "cB_log": run(c + unit_dirs(n_line_log, cfg.d_model) * log_r[:, None], c),
            "cC": run(c + line_s[:, None] * vc, c) | {"s": line_s.cpu().numpy()},
            "cC_log": run(c + log_s[:, None] * vc, c) | {"s": log_s.cpu().numpy()},
        }
        for name, d in sets.items():
            print(f"   {name:6s} n={len(d['sign']):5d}: det>0 {int((d['sign'] > 0).sum()):5d}, "
                  f"det<0 {int((d['sign'] < 0).sum()):5d}; median residual {np.median(d['residual']):.3f}, "
                  f"median u {np.median(d['u']):.3f}")

        out_path = path_for(crossing)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_path = out_path + ".tmp"
        with h5py.File(tmp_path, "w") as h5f:
            h5f.create_dataset("c", data=c.cpu().numpy())
            h5f.create_dataset("p", data=p.cpu().numpy())
            h5f.create_dataset("p_out", data=out0[1].cpu().numpy())
            h5f.create_dataset("c_out", data=out0[0].cpu().numpy())
            h5f.create_dataset("tangent_basis", data=U.cpu().numpy())
            h5f.create_dataset("path_t", data=path_t)
            h5f.create_dataset("path_sign", data=path_sign)
            h5f.create_dataset("path_log_abs_det", data=path_logdet)
            h5f.create_dataset("singular_values_p", data=svals_p.cpu().numpy())
            h5f.create_dataset("singular_values_c", data=svals_c.cpu().numpy())
            h5f.create_dataset("v_min_p", data=v_min_p.cpu().numpy())
            h5f.create_dataset("v_min_c", data=v_min_c.cpu().numpy())
            for k, val in calib.items():
                h5f.create_dataset(f"calib/{k}", data=val.cpu().numpy())
            for name, d in sets.items():
                for k, val in d.items():
                    h5f.create_dataset(f"{name}/{k}", data=val)
            for key, val in attrs.items():
                h5f.attrs[key] = val
            h5f.attrs.update({
                "interpolation_file": interp_path, "t_star": t_star, "crossing": crossing,
                "crossings_t": path_t[flips], "dist_c_p": dist_cp, "dist_c_p_units": dist_cp / spacing,
                "spacing": spacing, "anchor": anchor, "chart_step_units": step, "m": m, "K": charts.K,
                "radius": radius, "log_radii": np.asarray(log_radii or [np.nan, np.nan]), "n_nbrs": n_nbrs, "a_sampler": "convex_nbrs", "seed": seed,
                "kernel": kernel, "kernel_r": kernel_r, "kernel_sigma": kern.sigma,
                "p_sign": int(sign0[1]), "p_log_abs_det": float(logdet0[1]),
                "c_sign": int(sign0[0]), "c_log_abs_det": float(logdet0[0]),
                "u_c": float(kern(c[None])), "u_p": float(kern(p[None])),
                "tangent_overlap_p": overlap_p, "tangent_overlap_c": overlap_c,
                "cos_vmin_pc": float(v_min_p @ v_min_c),
            })
        os.replace(tmp_path, out_path)
        print(f"   saved -> {out_path}")
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument("interpolation", type=str, help="generate_det_sign_interpolation.py output h5")
    parser.add_argument("--out-dir", type=str, default="onnx_models/simple_stories/det_zero_manifold",
                        help="Writes {out-dir}/{sublayer}/row{A}_row{B}_manifold_c{crossing:02d}.h5")
    parser.add_argument("--crossing", nargs="+", default=["0"],
                        help="Which sign change(s) along A -> B to use: indices (0 = first, -1 = last) or 'all'")
    parser.add_argument("--bisect-iters", type=int, default=25, help="Bisection steps refining t*")
    parser.add_argument("--m", type=int, default=None,
                        help="Chart (tangent) dimension; default: TwoNN intrinsic dimension of the cloud")
    parser.add_argument("--n-cand", type=int, default=16, help="Candidate anchors whose charts c is projected onto")
    parser.add_argument("--chart-reach", type=float, default=2.0,
                        help="Max in-chart step from an anchor, in units of that anchor's spacing")
    parser.add_argument("--project-iters", type=int, default=10, help="Max re-anchoring passes of the projection")
    parser.add_argument("--radius", type=float, default=None,
                        help="Fixed sampling radius around p (overrides --radius-units; tags the file _r{R})")
    parser.add_argument("--radius-units", type=float, default=1.0,
                        help="Sampling radius in units of p's anchor spacing (8th-NN distance)")
    parser.add_argument("--n-points", type=int, default=1000, help="Points in each of the A / B balls")
    parser.add_argument("--n-line", type=int, default=401, help="Points on the C line, s in [-rho, rho]")
    parser.add_argument("--n-line-log", type=int, default=200,
                        help="Log-spaced radii for A_log / B_log (and per sign for C_log)")
    parser.add_argument("--line-min-s", type=float, default=1e-5, help="Smallest log-spaced radius")
    parser.add_argument("--n-nbrs", type=int, default=10,
                        help="Real points around p whose convex hull directions set A samples along")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (kernel split, samples)")
    parser.add_argument("--kernel", choices=["gauss", "laplace"], default="gauss",
                        help="KernelScore kernel: Gauss e^{-|x|^2/(2 sigma^2)} (default) or the paper's "
                             "regularized Laplace e^{-sqrt(|x|^2/sigma^2 + r)}")
    parser.add_argument("--kernel-r", type=float, default=1e-3, help="Laplace kernel regularization r")
    parser.add_argument("--tag", type=str, default="",
                        help="Appended to the output file names (..._manifold[_r{R}]_{tag}_c{NN}.h5) to keep runs apart")
    parser.add_argument("--log-radii", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="Draw the uniform sets' radii (and |s| on the C / cC lines, random sign) log-uniformly "
                             "in [LO, HI] instead of volume-uniform in the radius ball / evenly on [-rho, rho]")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output files (default: refuse, so earlier runs are never lost)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_det_zero_manifold(
        cfg,
        interp_path=args.interpolation,
        out_dir=args.out_dir,
        crossings=None if args.crossing == ["all"] else [int(c) for c in args.crossing],
        bisect_iters=args.bisect_iters,
        m=args.m,
        n_cand=args.n_cand,
        chart_reach=args.chart_reach,
        project_iters=args.project_iters,
        fixed_radius=args.radius,
        radius_units=args.radius_units,
        n_points=args.n_points,
        n_line=args.n_line,
        n_line_log=args.n_line_log,
        line_min_s=args.line_min_s,
        n_nbrs=args.n_nbrs,
        seed=args.seed,
        kernel=args.kernel,
        kernel_r=args.kernel_r,
        tag=args.tag,
        log_radii=tuple(args.log_radii) if args.log_radii else None,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
