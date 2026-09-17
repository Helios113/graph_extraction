"""GlobalManifold: learn G: R^m -> R^D from real states X and sample ~uniformly (w.r.t. manifold volume)
near the observed support. Knows nothing about Transformers."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from .geometry import knn, uniform_ball


def _mlp(sizes: Sequence[int], act: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [
    return nn.Sequential(*layers[:-1])


class GlobalManifold(nn.Module):
    """Encoder E: D -> hidden -> m, decoder G: m -> reversed(hidden) -> D (SiLU).
    Preprocessing: subtract the training centre, divide by one global RMS scale (a similarity, so Euclidean
    geometry is preserved). The manifold ansatz is G(Omega) with Omega = union of latent balls
    B(z_i, alpha*latent_scale_i)."""

    def __init__(
        self,
        latent_dim: int,
        # Size of MLP
        hidden: Sequence[int] = (512, 256),
        n_neighbors: int = 32,
        n_scale_neighbors: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim, self.hidden, self.n_neighbors, self.n_scale_neighbors = (
            int(latent_dim),
            tuple(hidden),
            n_neighbors,
            n_scale_neighbors,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.encoder = self.encoder = nn.Sequential(
                *[layer for a, b in zip(sizes[:-1], sizes[1:]) for layer in (nn.Linear(a, b), act())][:-1]
            )
        self.decoder: nn.Sequential | None = None
        
        self.register_buffer("center", torch.zeros(0))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer(
            "latent_codes", torch.zeros(0, self.latent_dim)
        )  # latent codes of the fitted states
        self.register_buffer(
            "latent_scale", torch.zeros(0)
        )  # local latent scale (n_scale_neighbors-th latent neighbour)
        self.register_buffer(
            "ambient_spacing", torch.zeros(0)
        )  # local ambient spacing (n_scale_neighbors-th neighbour) of states
        self.states: Tensor | None = None
        self.history: list[dict[str, Any]] = []
        self.spacing_unit = "anchor"  # 'anchor': divide by the nearest anchor's spacing; 'global': by the median
        self.hub_share = 0.0  # share of points whose nearest neighbour is one of the 5 most common anchors

    # ------------------------------------------------------------------ maps
    def encode(self, x: Tensor) -> Tensor:
        assert self.encoder is not None
        return self.encoder((x.to(self.device) - self.center) / self.scale)

    def decode(self, z: Tensor) -> Tensor:
        assert self.decoder is not None
        return self.decoder(z.to(self.device)) * self.scale + self.center

    def project(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def jacobian(self, z: Tensor, chunk: int = 1024) -> Tensor:
        """J_G(z) in ambient units, (B, D, m)."""
        assert self.decoder is not None
        f = torch.func.vmap(torch.func.jacfwd(self.decoder))
        return (
            torch.cat([f(z[s : s + chunk]) for s in range(0, len(z), chunk)])
            * self.scale
        )

    # To get uniform sampling
    def log_volume(self, z: Tensor, chunk: int = 1024) -> Tensor:
        """0.5 * log det(J^T J): log of the Riemannian volume element of the chart at z."""
        out: list[Tensor] = []
        for s in range(0, len(z), chunk):
            J = self.jacobian(z[s : s + chunk])
            out.append(0.5 * torch.linalg.slogdet(J.transpose(1, 2) @ J)[1])
        return torch.cat(out)

    # ------------------------------------------------------------------ sampling
    def radii(self, alpha: float, radius: str = "global") -> Tensor:
        """Latent ball radii: 'global' = alpha x median rho for every anchor (a uniform-thickness neighbourhood of
        the data; robust to outliers), 'local' = alpha x rho_i per anchor (the union of local balls is then dominated
        by the largest balls, since ball volume scales like rho^m)."""
        latent_scale = (
            self.latent_scale
            if radius == "local"
            else self.latent_scale.median().expand_as(self.latent_scale)
        )
        return alpha * latent_scale

    @torch.no_grad()
    def support_logq(
        self, z: Tensor, R: Tensor, anchor_logp: Tensor, chunk: int = 4096
    ) -> tuple[Tensor, Tensor]:
        """Exact log-density of the latent-ball mixture proposal (up to a constant) and ball multiplicity."""
        logq: list[Tensor] = []
        cnt: list[Tensor] = []
        for s in range(0, len(z), chunk):
            d = torch.cdist(z[s : s + chunk], self.latent_codes)
            inside = d <= R * (1 + 1e-5)
            term = torch.where(
                inside,
                anchor_logp - self.latent_dim * R.log(),
                torch.full_like(d, -float("inf")),
            )
            logq.append(term.logsumexp(1))
            cnt.append(inside.sum(1))
        return torch.cat(logq), torch.cat(cnt)

    @torch.no_grad()
    def sample(
        self,
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
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        n_c = n_candidates or 8 * n
        p = self.latent_dim if anchor_power is None else anchor_power
        R = self.radii(alpha, radius)
        anchor_logp = p * R.log()
        anchor_logp = anchor_logp - anchor_logp.logsumexp(0)
        i = torch.multinomial(anchor_logp.exp(), n_c, replacement=True, generator=g)
        z = self.latent_codes[i] + uniform_ball(n_c, self.latent_dim, R[i], g, self.device)
        logq, cnt = self.support_logq(z, R, anchor_logp)
        logvol = (
            torch.nan_to_num(
                self.log_volume(z), nan=-float("inf"), posinf=-float("inf")
            )
            if reweight
            else torch.zeros(n_c, device=self.device)
        )

        def weights(t: float) -> Tensor:
            lw = (
                t * logvol - logq if reweight else torch.zeros(n_c, device=self.device)
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
                print("[gmanifold] WARNING " + info["warning"])
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
        return self.decode(z[pick]), info

    # ------------------------------------------------------------------ diagnostics
    def unit(self, j: Tensor) -> Tensor:
        """Length unit for points whose nearest fitted state is j: that state's spacing, or the global median."""
        return (
            self.ambient_spacing[j]
            if self.spacing_unit == "anchor"
            else self.ambient_spacing.median().expand(len(j))
        )

    
    @torch.no_grad()
    def recon_error(self, Y: Tensor) -> Tensor:
        """||G(E(y)) - y|| in spacing units (the off-manifold distance to the ansatz)."""
        Y = Y.to(self.device).float()
        j = knn(self.states, 1, Y)[1][:, 0]
        return (self.project(Y) - Y).norm(dim=1) / self.unit(j)

    @torch.no_grad()
    def nearest_real(self, Y: Tensor) -> Tensor:
        """Distance from Y to the nearest fitted real state, in spacing units."""
        d, j = knn(self.states, 1, Y.to(self.device).float())
        return d[:, 0] / self.unit(j[:, 0])

    @torch.no_grad()
    def coverage(self, Y: Tensor) -> float:
        """Fraction of fitted real states that have a sample within one spacing unit."""
        assert self.states is not None
        d = knn(Y.to(self.device).float(), 1, self.states)[0][:, 0]
        return float(
            (d / self.unit(torch.arange(len(self.states), device=self.device)) <= 1)
            .float()
            .mean()
        )

    @torch.no_grad()
    def jacobian_rank(self, n: int = 256, tol: float = 1e-3) -> dict[str, Any]:
        sv = torch.linalg.svdvals(self.jacobian(self.latent_codes[:n]))
        return dict(
            rank_median=float((sv > tol * sv[:, :1]).sum(1).float().median()),
            singular_values=sv.mean(0).cpu(),
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        assert self.states is not None
        # States is the bigeest issue
        torch.save(
            dict(
                latent_dim=self.latent_dim,
                hidden=self.hidden,
                n_neighbors=self.n_neighbors,
                n_scale_neighbors=self.n_scale_neighbors,
                ambient_dim=self.states.shape[1],
                states=self.states.cpu(),
                state=self.state_dict(),
                history=self.history,
                spacing_unit=self.spacing_unit,
                hub_share=self.hub_share,
            ),
            path,
        )

    @classmethod
    def load(cls, path: str, device: torch.device | str | None = None) -> GlobalManifold:
        s: dict[str, Any] = torch.load(path, map_location="cpu")
        M = cls(s["latent_dim"], s["hidden"], s["n_neighbors"], s["n_scale_neighbors"], device)
        M.encoder = _mlp([s["ambient_dim"], *M.hidden, M.latent_dim]).to(M.device)
        M.decoder = _mlp([M.latent_dim, *reversed(M.hidden), s["ambient_dim"]]).to(M.device)
        M.latent_codes, M.latent_scale, M.ambient_spacing, M.center = (
            torch.zeros_like(s["state"][k])
            for k in ("latent_codes", "latent_scale", "ambient_spacing", "center")
        )
        M.load_state_dict(s["state"])
        M.to(M.device)
        M.states = s["states"].to(M.device)
        M.history = s["history"]
        M.spacing_unit, M.hub_share = (
            s.get("spacing_unit", "anchor"),
            s.get("hub_share", 0.0),
        )
        return M.eval()

