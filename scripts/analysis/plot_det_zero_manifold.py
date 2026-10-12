"""Plots for scripts/python/generate_det_zero_manifold.py output.

Given one or more ..._manifold_c{NN}.h5 files (typically every crossing of
one A -> B pair), writes:

  {pair}_manifold_overview.png -- across crossings:
    det along the A -> B path with the crossings marked; distance from the
    det = 0 point c to its nearest manifold point p (in spacings) with the
    kernel score u at c and p; sigma_min of J at c vs p; and how much of the
    kernel direction v_min lies in the m-dim tangent space at p (vs a random
    direction's m / d).
  {stem}.png, one per file -- around p (A on manifold, B ambient, C v_min(p) line)
  and around the critical point c (cB ambient, cC v_min(c) line); every
  distance is from the set's own centre (p or c):
    1. kernel score u of every set;
    2. chart residual vs u per point (where on- vs off-manifold points sit);
    3. ||f(x) - f(centre)|| vs ||x - centre||, log-log, incl. the log-spaced sets;
    4. each point's K_L value ||f(x) - f(centre)|| / ||x - centre|| (3's y over x) vs
       ||x - centre||, with the float32 floor 2e-6 / ||x - centre||;
    5. histogram of that ratio (uniform sets);
    6. log|det| vs distance from the centre;
    7. pairwise Lipschitz ratio g = ||f(x_i) - f(x_j)|| / ||x_i - x_j|| over every pair within
       each uniform set (the K_L values; min g = that set's empirical K_L), pairs with
       ||f(x_i) - f(x_j)|| < --fp32-floor dropped (default 0: distances taken as exact; pairs
       with identical outputs are left out since log10(0) can't be shown).

CPU, anywhere:
    uv run python -m scripts.analysis.plot_det_zero_manifold \\
        onnx_models/simple_stories/det_zero_manifold/add_4.add_6/row26685_row26140_manifold_c*.h5
"""
import argparse
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np

SETS = {"A": ("p: on manifold (convex nbrs)", "#2a78d6"), "B": ("p: ambient ball", "#eb6834"),
        "C": ("p: smallest-sigma line v_min(p)", "#1baf7a"), "cB": ("c: ambient ball", "#eda100"),
        "cC": ("c: kernel line v_min(c)", "#e87ba4")}
CENTRE = {"A": "p", "B": "p", "C": "p", "cB": "c", "cC": "c"}
INK, MUTED = "#0b0b0b", "#52514e"
FP32_FLOOR = 0.0  # --fp32-floor: drop output changes below this as float32 noise (0 = take distances as exact)


def _style(ax) -> None:
    ax.grid(True, color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _load(path: str) -> dict:
    with h5py.File(path, "r") as f:
        d = {"attrs": dict(f.attrs)}
        for k in ("p", "p_out", "c_out", "path_t", "path_sign", "path_log_abs_det", "singular_values_p", "singular_values_c"):
            d[k] = f[k][:]
        for g in SETS:
            for name in (g, g + "_log"):
                if name in f:  # cB / cC only in files made after they were added
                    d[name] = {k: f[name][k][:] for k in f[name]}
        d["u_heldout"] = f["calib/u_heldout"][:]
    return d


def _all_pairs(vec_in: np.ndarray, vec_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(||x_i - x_j||, ||f(x_i) - f(x_j)||) for every i < j, float64, row by row
    (same as plot_det_zero_sphere_lipschitz.all_pair_dists)."""
    x, y = vec_in.astype(np.float64), vec_out.astype(np.float64)
    in_d = [np.linalg.norm(x[i + 1:] - x[i], axis=1) for i in range(len(x) - 1)]
    out_d = [np.linalg.norm(y[i + 1:] - y[i], axis=1) for i in range(len(y) - 1)]
    return np.concatenate(in_d), np.concatenate(out_d)


def _stretch(d: dict, g: str) -> tuple[np.ndarray, np.ndarray]:
    """(||x - centre||, ||f(x) - f(centre)||), centre = p or c per CENTRE."""
    dx = d[g]["dist"]
    df = np.linalg.norm(d[g]["vec_out"] - d[CENTRE[g.removesuffix("_log")] + "_out"], axis=1)
    return dx, df


def plot_overview(runs: list[tuple[str, dict]], out_path: str) -> None:
    d0 = runs[0][1]
    a0 = d0["attrs"]
    cr = np.array([r["attrs"]["crossing"] for _, r in runs])
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"{a0['sublayer_input']}.{a0['sublayer_output']} pos {a0['pos']}, rows {a0['a_row']} -> {a0['b_row']}: "
                 f"det = 0 points c projected onto the manifold (m = {a0['m']})", fontsize=11, color=INK)

    ax = axes[0, 0]
    t, sgn, ld = d0["path_t"], d0["path_sign"], d0["path_log_abs_det"]
    ax.plot(t, ld, color=MUTED, lw=1.2, zorder=1)
    for s, col, lab in ((1, "#2a78d6", "det > 0"), (-1, "#eb6834", "det < 0")):
        ax.scatter(t[sgn == s], ld[sgn == s], s=6, color=col, label=lab, zorder=2)
    for _, r in runs:
        ax.axvline(r["attrs"]["t_star"], color=MUTED, lw=0.6, ls=":")
        ax.annotate(str(r["attrs"]["crossing"]), (r["attrs"]["t_star"], 1.0), xycoords=("data", "axes fraction"),
                    fontsize=7, color=MUTED, ha="center", va="bottom")
    ax.set_xlabel("t along A -> B")
    ax.set_ylabel("log |det J|")
    ax.set_title("det along the straight path; dotted = bisected crossings c", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[0, 1]
    dist = np.array([r["attrs"]["dist_c_p_units"] for _, r in runs])
    ax.bar(cr, dist, width=0.6, color="#2a78d6")
    ax.set_xticks(cr)
    ax.set_xlabel("crossing")
    ax.set_ylabel("|c - p|  (local spacings)")
    ax.set_title("how far each det = 0 point c is from the manifold", fontsize=10)

    ax = axes[0, 2]
    ax.plot(cr, [r["attrs"]["u_c"] for _, r in runs], "o", color="#eb6834", ms=7, label="u(c)")
    ax.plot(cr, [r["attrs"]["u_p"] for _, r in runs], "s", color="#1baf7a", ms=7, label="u(p)")
    ax.set_xticks(cr)
    ax.set_xlabel("crossing")
    ax.set_ylabel("kernel score u (1 = on manifold)")
    ax.set_title("kernel score at c and p", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 0]
    ax.semilogy(cr, [r["singular_values_c"][-1] for _, r in runs], "o", color="#eb6834", ms=7, label="at c (det = 0 point)")
    ax.semilogy(cr, [r["singular_values_p"][-1] for _, r in runs], "s", color="#1baf7a", ms=7, label="at p (nearest manifold point)")
    ax.semilogy(cr, [r["singular_values_p"][-2] for _, r in runs], "^", color="#52514e", ms=5, mfc="none",
                label="2nd smallest at p")
    ax.set_xticks(cr)
    ax.set_xlabel("crossing")
    ax.set_ylabel("sigma_min of J")
    ax.set_title("is p itself singular?", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 1]
    w = 0.38
    ax.bar(cr - w / 2, [r["attrs"]["tangent_overlap_p"] for _, r in runs], width=w, color="#1baf7a", label="v_min at p")
    ax.bar(cr + w / 2, [r["attrs"]["tangent_overlap_c"] for _, r in runs], width=w, color="#eb6834", label="v_min at c")
    ax.axhline(a0["m"] / len(d0["p"]), color=MUTED, ls="--", lw=1, label=f"random direction (m/d = {a0['m'] / len(d0['p']):.3f})")
    ax.set_xticks(cr)
    ax.set_ylim(0, 1)
    ax.set_xlabel("crossing")
    ax.set_ylabel("||U^T v_min||^2")
    ax.set_title("share of the kernel direction inside the tangent space", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1, 2]
    for off, (g, (lab, col)) in zip((-0.2, 0, 0.2), SETS.items()):
        first = []
        for _, r in runs:
            dd = r[g + "_log"]
            opp = dd["dist"][dd["sign"] != r["attrs"]["p_sign"]]
            first.append(opp.min() if len(opp) else np.nan)
        ax.semilogy(cr + off, first, "o", color=col, ms=7, label=f"{lab}")
    ax.semilogy(cr, [r["attrs"]["dist_c_p"] for _, r in runs], "_", color=INK, ms=18, mew=2, label="|c - p|")
    ax.set_xticks(cr)
    ax.set_xlabel("crossing")
    ax.set_ylabel("||x - p|| of nearest opposite-sign point")
    ax.set_title("how far from p the det sign first flips (log-spaced sets)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes.flat:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def _hist_peak1(ax, values: dict[str, np.ndarray], sets: dict, n_bins: int, lw: float = 2.0,
                labels: dict[str, str] | None = None) -> None:
    """One step histogram per set on shared bins spanning the full finite range of all
    values, each scaled to a peak height of 1 -- a narrow spike and a broad spread
    stay equally visible."""
    finite = {g: v[np.isfinite(v)] for g, v in values.items()}
    allv = np.concatenate([v for v in finite.values() if v.size])
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, n_bins + 1)
    # thickest first, so sets that peak in the same bins stay visible as nested outlines
    widths = np.linspace(lw + 2.5, max(lw - 0.5, 1.0), len(sets))
    for (g, (lab, col)), w in zip(sets.items(), widths):
        counts, edges = np.histogram(finite[g], bins=bins)
        if counts.max() > 0:
            ax.stairs(counts / counts.max(), edges, color=col, lw=w, label=(labels or {}).get(g, lab))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("count / peak count")


def plot_crossing(d: dict, out_path: str) -> None:
    a = d["attrs"]
    sets = {g: v for g, v in SETS.items() if g in d}
    fig = plt.figure(figsize=(22, 10))
    grid = fig.add_gridspec(2, 12)
    axes = [fig.add_subplot(grid[0, j:j + 3]) for j in (0, 3, 6, 9)] + \
           [fig.add_subplot(grid[1, j:j + 4]) for j in (0, 4, 8)]
    fig.suptitle(f"{a['sublayer_input']}.{a['sublayer_output']} pos {a['pos']}, crossing {a['crossing']} "
                 f"(t* = {a['t_star']:.4f}): |c - p| = {a['dist_c_p_units']:.2f} spacings, radius around p and c = "
                 f"{a['radius']:.3g} ({a['radius'] / a['spacing']:.2g} spacings); log|det| at p {a['p_log_abs_det']:.2f}, "
                 f"at c {a['c_log_abs_det']:.2f}; sigma_min p {d['singular_values_p'][-1]:.2g}, "
                 f"c {d['singular_values_c'][-1]:.2g}", fontsize=11, color=INK)
    x_left = 0.5 * np.abs(d["C_log"]["s"]).min()

    # 1. kernel score
    ax = axes[0]
    _hist_peak1(ax, {g: d[g]["u"] for g in sets}, sets, 50)
    ax.set_xlabel("kernel score u  (1 = on manifold)")
    ax.set_title("1. is it on the manifold?", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    # 2. residual vs u
    ax = axes[1]
    for g, (lab, col) in sets.items():
        ax.scatter(d[g]["residual"], d[g]["u"], s=8, alpha=0.5, lw=0, color=col, label=lab)
    ax.set_xlabel("chart residual (spacings off the tangent chart)")
    ax.set_ylabel("kernel score u")
    ax.set_title("2. two validators per point", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, markerscale=1.5)

    # 3. stretch, log-log
    ax = axes[2]
    for g, (lab, col) in sets.items():
        dx, df = _stretch(d, g)
        ax.scatter(dx, df, s=6, alpha=0.35, lw=0, color=col)
        dx, df = _stretch(d, g + "_log")
        ax.scatter(dx, df, s=12, alpha=0.9, lw=0, color=col, label=lab)
    xs = np.logspace(np.log10(a["radius"] * 1e-5), np.log10(a["radius"]), 10)
    ax.plot(xs, xs, color=MUTED, ls=":", lw=1, label="||Δf|| = ||Δx||")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=x_left)
    ax.set_xlabel("||x - centre||")
    ax.set_ylabel("||f(x) - f(centre)||")
    ax.set_title("3. output vs input distance from the centre (p or c)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    # 4. per-point K_L value (panel 3's y / x) vs distance
    ax = axes[3]
    for g, (lab, col) in sets.items():
        for suffix, sz, al in (("", 6, 0.35), ("_log", 12, 0.9)):
            dx, df = _stretch(d, g + suffix)
            ok = dx > 0
            ax.scatter(dx[ok], df[ok] / dx[ok], s=sz, alpha=al, lw=0, color=col, label=lab if suffix else None)
    xs = np.logspace(np.log10(x_left), np.log10(a["radius"]), 50)
    if FP32_FLOOR > 0:
        ax.plot(xs, FP32_FLOOR / xs, color=MUTED, ls="--", lw=1, label=f"float32 floor ({FP32_FLOOR:g} / ||Δx||)")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(left=x_left)
    ax.set_xlabel("||x - centre||")
    ax.set_ylabel("K_L value  ||f(x) - f(centre)|| / ||x - centre||")
    ax.set_title("4. K_L value of each point vs distance from the centre", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    # 5. ratio histogram
    ax = axes[4]
    ratios = {}
    for g in sets:
        dx, df = _stretch(d, g)
        ok = dx > 0  # the uniform v_min lines include the centre itself (0 / 0)
        ratios[g] = np.log10(df[ok] / dx[ok])
    _hist_peak1(ax, ratios, sets, 60)
    ax.set_xlabel("log10 ( ||f(x) - f(centre)|| / ||x - centre|| )")
    ax.set_title("5. stretch ratio from the centre (uniform sets)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    # 6. log|det| vs distance
    ax = axes[5]
    for g, (lab, col) in sets.items():
        for suffix, sz, al in (("", 6, 0.35), ("_log", 12, 0.9)):
            dd = d[g + suffix]
            ax.scatter(dd["dist"], dd["log_abs_det"], s=sz, alpha=al, lw=0, color=col, label=lab if suffix else None)
    ax.axhline(a["p_log_abs_det"], color=MUTED, ls=":", lw=1, label="log|det| at p")
    ax.axhline(a["c_log_abs_det"], color=MUTED, ls="--", lw=1, label="log|det| at c")
    ax.set_xscale("log")
    ax.set_xlim(left=x_left)
    ax.set_xlabel("||x - centre||")
    ax.set_ylabel("log |det J|")
    ax.set_title("6. log|det| vs distance from the centre", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)

    # 7. pairwise ratio g (K_L values) within each uniform set
    ax = axes[6]
    gs_ = {}
    for g in sets:
        din, dout = _all_pairs(d[g]["vec_in"], d[g]["vec_out"])
        keep = (dout >= FP32_FLOOR) & (dout > 0) & (din > 0)
        gs_[g] = np.log10(dout[keep] / din[keep])
    _hist_peak1(ax, gs_, sets, 80, lw=1.8, labels={g: f"{sets[g][0]}: min {10 ** gs_[g].min():.2g}" for g in sets})
    for g, (_, col) in sets.items():
        ax.axvline(gs_[g].min(), color=col, ls=":", lw=1)
    ax.set_xlabel("log10 g,  g = ||f(x_i) - f(x_j)|| / ||x_i - x_j||  (all pairs in a set)")
    ax.set_title("7. K_L values: pairwise ratio per set (dotted = min)", fontsize=10)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_p_validation(runs: list[tuple[str, dict]], out_path: str) -> None:
    """Kernel score u(p) of every crossing's projected point p against the
    held-out real points' u (real cloud points the KernelScore was not fit
    on -- the cloud, kernel split and seed are the same for every crossing
    of one pair, so one held-out distribution serves them all)."""
    d0 = runs[0][1]
    a0 = d0["attrs"]
    held = np.sort(d0["u_heldout"])
    cr = np.array([r["attrs"]["crossing"] for _, r in runs])
    u_p = np.array([r["attrs"]["u_p"] for _, r in runs])
    u_c = np.array([r["attrs"]["u_c"] for _, r in runs])
    pct_p = 100 * np.searchsorted(held, u_p) / len(held)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.suptitle(f"{a0['sublayer_input']}.{a0['sublayer_output']} pos {a0['pos']}, rows {a0['a_row']} -> {a0['b_row']}: "
                 f"is p on the manifold? kernel score u of p vs {len(held)} held-out real points", fontsize=11, color=INK)

    ax = axes[0]
    lo = min(held.min(), u_p.min(), u_c.min())
    hi = max(held.max(), u_p.max(), u_c.max())
    pad = 0.02 * (hi - lo)
    ax.hist(held, bins=np.linspace(lo - pad, hi + pad, 60), color="#b7d3f6", edgecolor="#fcfcfb", lw=0.5,
            label="held-out real points")
    for q, ls in ((5, ":"), (50, "--"), (95, ":")):
        ax.axvline(np.percentile(held, q), color=MUTED, ls=ls, lw=1, label=f"held-out {q}th percentile")
    ytop = ax.get_ylim()[1]
    ax.scatter(u_p, np.full_like(u_p, 0.04 * ytop), marker="|", s=400, lw=2.5, color="#1baf7a", zorder=5,
               label=f"u(p), {len(u_p)} crossings")
    ax.scatter(u_c, np.full_like(u_c, 0.04 * ytop), marker="|", s=250, lw=1.5, color="#eb6834", alpha=0.6, zorder=4,
               label="u(c), for contrast")
    ax.set_xlabel("kernel score u  (1 = on manifold)")
    ax.set_ylabel("held-out points")
    ax.set_title("held-out distribution with every p (and c) marked", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    ax.bar(cr, pct_p, width=0.6, color="#1baf7a")
    for x, y, u in zip(cr, pct_p, u_p):
        ax.annotate(f"{u:.3f}", (x, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5, color=MUTED)
    ax.axhspan(5, 95, color="#f0efec", zorder=0, label="central 90% of held-out")
    ax.set_xticks(cr)
    ax.set_ylim(0, 105)
    ax.set_xlabel("crossing")
    ax.set_ylabel("percentile of u(p) among held-out real")
    ax.set_title("where each p falls (label = u(p))", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_kernel_groups(runs: list[tuple[str, dict]], out_path: str) -> None:
    """Kernel score u of three groups side by side: held-out real points (real
    cloud points the kernel score was not fit on), the critical (det = 0)
    points c, and their nearest manifold points p."""
    d0 = runs[0][1]
    a0 = d0["attrs"]
    groups = [
        ("held-out real points", d0["u_heldout"], "#52514e"),
        ("critical points c\n(det J = 0)", np.array([r["attrs"]["u_c"] for _, r in runs]), "#eb6834"),
        ("nearest manifold points p", np.array([r["attrs"]["u_p"] for _, r in runs]), "#1baf7a"),
    ]
    kern = a0.get("kernel", "gauss")
    kdesc = (f"Laplace kernel e^(-sqrt(|x|^2/sigma^2 + r)), r = {a0['kernel_r']:g}" if kern == "laplace"
             else "Gauss kernel e^(-|x|^2 / 2 sigma^2)")
    sig = f", sigma = {a0['kernel_sigma']:.3g}" if "kernel_sigma" in a0 else ""

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{a0['sublayer_input']}.{a0['sublayer_output']} pos {a0['pos']}, rows {a0['a_row']} -> {a0['b_row']}: "
                 f"kernel score by group\n{kdesc}{sig}", fontsize=11, color=INK)
    rng = np.random.default_rng(0)
    held = groups[0][1]
    ax.axhspan(*np.percentile(held, [5, 95]), color="#f0efec", zorder=0, label="central 90% of held-out")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1, label="u = 1 (on the fitted data)")
    for i, (lab, u, col) in enumerate(groups):
        x = i + rng.uniform(-0.18, 0.18, len(u))
        ax.scatter(x, u, s=10 if len(u) > 50 else 40, color=col, alpha=0.45 if len(u) > 50 else 0.9,
                   lw=0, zorder=3)
        med = np.median(u)
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2, zorder=4)
        ax.annotate(f"median {med:.4f}\nn = {len(u)}", (i + 0.32, med), fontsize=8, color=MUTED, va="center")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups])
    ax.set_xlim(-0.6, len(groups) - 0.2)
    ax.set_ylabel("kernel score u  (1 = on manifold)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", help="generate_det_zero_manifold.py output h5 file(s)")
    parser.add_argument("--fp32-floor", type=float, default=0.0,
                        help="treat output changes below this as float32 noise: drawn as a floor line in panel 4 "
                             "and dropped from panel 7 (default 0: distances taken as exact)")
    args = parser.parse_args()
    global FP32_FLOOR
    FP32_FLOOR = args.fp32_floor
    runs = sorted(((p, _load(p)) for p in args.files), key=lambda r: r[1]["attrs"]["crossing"])
    for path, d in runs:
        plot_crossing(d, os.path.splitext(path)[0] + ".png")
    pair = re.sub(r"_c\d+\.h5$", "", runs[0][0])
    plot_overview(runs, pair + "_overview.png")
    plot_p_validation(runs, pair + "_p_validation.png")
    plot_kernel_groups(runs, pair + "_kernel_score.png")


if __name__ == "__main__":
    main()
