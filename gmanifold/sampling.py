"""Volume-uniform sampling from a fitted manifold, and fit-quality diagnostics.

Free functions over a fitted `manifold` (autoencoder.autoencoder.manifold),
migrated out of GlobalManifold's methods in gmanifold/manifold.py.
"""

from typing import Any

import torch
from torch import Tensor

from .autoencoder import knn, manifold


def uniform_ball(n, m, radius, generator=None, device=None):
    """Uniform points in m-balls of the given radii (n,): direction ~ normalised Gaussian, r = R s^{1/m}."""
    g = torch.randn(n, m, device=device, generator=generator)
    g = g / g.norm(dim=1, keepdim=True)
    return g * (radius.to(device) * torch.rand(n, device=device, generator=generator) ** (1.0 / m))[:, None]


# To get uniform sampling
def log_volume(model: manifold, z: Tensor, chunk: int = 1024) -> Tensor:
    """0.5 * log det(J^T J): log of the Riemannian volume element of the chart at z."""
    out: list[Tensor] = []
    for s in range(0, len(z), chunk):
        J = model.jacobian(z[s : s + chunk])
        out.append(0.5 * torch.linalg.slogdet(J.transpose(1, 2) @ J)[1])
    return torch.cat(out)


# ------------------------------------------------------------------ sampling
def radii(model: manifold, alpha: float, radius: str = "global") -> Tensor:
    """Latent ball radii: 'global' = alpha x median rho for every anchor (a uniform-thickness neighbourhood of
    the data; robust to outliers), 'local' = alpha x rho_i per anchor (the union of local balls is then dominated
    by the largest balls, since ball volume scales like rho^m)."""
    latent_scale = (
        model.latent_scale
        if radius == "local"
        else model.latent_scale.median().expand_as(model.latent_scale)
    )
    return alpha * latent_scale


@torch.no_grad()
def support_logq(
    model: manifold, z: Tensor, R: Tensor, anchor_logp: Tensor, chunk: int = 4096
) -> tuple[Tensor, Tensor]:
    """Exact log-density of the latent-ball mixture proposal (up to a constant) and ball multiplicity."""
    logq: list[Tensor] = []
    cnt: list[Tensor] = []
    for s in range(0, len(z), chunk):
        d = torch.cdist(z[s : s + chunk], model.latent_codes)
        inside = d <= R * (1 + 1e-5)
        term = torch.where(
            inside,
            anchor_logp - model.latent_dim * R.log(),
            torch.full_like(d, -float("inf")),
        )
        logq.append(term.logsumexp(1))
        cnt.append(inside.sum(1))
    return torch.cat(logq), torch.cat(cnt)


@torch.no_grad()
def sample(
    model: manifold,
    n: int,
    alpha: float = 0.3,
    radius: str = "global",
    n_candidates: int | None = None,
    anchor_power: float | None = None,
    reweight: bool = True,
    tau: float = 1.0,
    auto_tau: bool = False,
    min_ess_frac: float = 0.05,
    seed: int | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Approximately volume-uniform samples on G(Omega), Omega = union_i B(z_i, R_i) with R_i from `radii`.
    1. anchor i ~ P(i) ∝ R_i^p (p = m: the proposal is then uniform on Omega up to ball multiplicity);
    2. z uniform in B(z_i, R_i);  3. importance weight w = sqrt(det J^T J)^tau / q(z) with the exact mixture
    density q (tau = 1: exact surface-area weighting);  4. multinomial resampling of n points;  5. x = G(z).
    The weights are checked every call: if their effective sample size falls below `min_ess_frac` of the
    candidates (a few candidates with extreme volume elements absorb the mass), a warning is printed with the
    tau that would restore the floor. With `auto_tau=True` that tau is applied automatically.
    Returns (x, info) with info = z, anchor, multiplicity, ess, tau, tau_suggested, warning, anchor participation."""
    g = torch.Generator(device=model.device)
    if seed is not None:
        g.manual_seed(seed)
    n_c = n_candidates or 8 * n
    p = model.latent_dim if anchor_power is None else anchor_power
    R = radii(model, alpha, radius)
    anchor_logp = p * R.log()
    anchor_logp = anchor_logp - anchor_logp.logsumexp(0)
    i = torch.multinomial(anchor_logp.exp(), n_c, replacement=True, generator=g)
    z = model.latent_codes[i] + uniform_ball(n_c, model.latent_dim, R[i], g, model.device)
    logq, cnt = support_logq(model, z, R, anchor_logp)
    logvol = (
        torch.nan_to_num(
            log_volume(model, z), nan=-float("inf"), posinf=-float("inf")
        )
        if reweight
        else torch.zeros(n_c, device=model.device)
    )

    def weights(t: float) -> Tensor:
        lw = (
            t * logvol - logq if reweight else torch.zeros(n_c, device=model.device)
        )  # no reweighting = the proposal itself
        lw = torch.nan_to_num(lw, nan=-float("inf"), posinf=-float("inf"))
        return (lw - lw.logsumexp(0)).exp()

    ess = lambda w: float(1 / (w**2).sum())
    w = weights(tau)
    info: dict[str, Any] = dict(tau=tau, tau_suggested=tau, warning=None)
    if reweight and min_ess_frac and ess(w) < min_ess_frac * n_c:  # the checker
        lo, hi = 0.0, tau
        for _ in range(20):  # largest tempering exponent meeting the ESS floor
            mid_ = 0.5 * (lo + hi)
            lo, hi = (
                (mid_, hi)
                if ess(weights(mid_)) >= min_ess_frac * n_c
                else (lo, mid_)
            )
        top = float(w.sort(descending=True).values[:10].sum())
        info["tau_suggested"] = lo
        if auto_tau:
            w = weights(lo)
            info["tau"] = lo
        else:
            info["warning"] = (
                f"degenerate importance weights: ESS {ess(w):.0f} of {n_c} candidates, the 10 heaviest carry {top:.0%} of the mass; "
                f"the samples will be near-copies of a few points. Temper the volume weights with tau≈{lo:.2f} "
                f"(sample(..., tau={lo:.2f}) or auto_tau=True), and check the fit (held-out recon, latent dim, outlier states)."
            )
            print("[autoencoder] WARNING " + info["warning"])
    pick = torch.multinomial(w, n, replacement=True, generator=g)
    info.update(
        z=z[pick],
        anchor=i[pick],
        multiplicity=cnt[pick],
        ess=ess(w),
        alpha=alpha,
        radius=radius,
        n_candidates=n_c,
        anchor_participation=float(1 / (anchor_logp.exp() ** 2).sum()),
    )
    return model.decode(z[pick]), info


# ------------------------------------------------------------------ diagnostics
def unit(model: manifold, j: Tensor) -> Tensor:
    """Length unit for points whose nearest fitted state is j: that state's spacing, or the global median."""
    return (
        model.ambient_spacing[j]
        if model.spacing_unit == "anchor"
        else model.ambient_spacing.median().expand(len(j))
    )


@torch.no_grad()
def recon_error(model: manifold, Y: Tensor) -> Tensor:
    """||G(E(y)) - y|| in spacing units (the off-manifold distance to the ansatz)."""
    Y = Y.to(model.device).float()
    j = knn(model.states, 1, Y)[1][:, 0]
    return (model.project(Y) - Y).norm(dim=1) / unit(model, j)


@torch.no_grad()
def nearest_real(model: manifold, Y: Tensor) -> Tensor:
    """Distance from Y to the nearest fitted real state, in spacing units."""
    d, j = knn(model.states, 1, Y.to(model.device).float())
    return d[:, 0] / unit(model, j[:, 0])


@torch.no_grad()
def coverage(model: manifold, Y: Tensor) -> float:
    """Fraction of fitted real states that have a sample within one spacing unit."""
    assert model.states is not None
    d = knn(Y.to(model.device).float(), 1, model.states)[0][:, 0]
    return float(
        (d / unit(model, torch.arange(len(model.states), device=model.device)) <= 1)
        .float()
        .mean()
    )


@torch.no_grad()
def jacobian_rank(model: manifold, n: int = 256, tol: float = 1e-3) -> dict[str, Any]:
    sv = torch.linalg.svdvals(model.jacobian(model.latent_codes[:n]))
    return dict(
        rank_median=float((sv > tol * sv[:, :1]).sum(1).float().median()),
        singular_values=sv.mean(0).cpu(),
    )
