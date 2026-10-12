"""Volume-uniform sampling from a fitted autoencoder / batched_autoencoder:
samples ~uniformly (w.r.t. manifold volume) near the observed support.

Single-model functions (sample, radii, support_logq, log_volume) operate on
autoencoder.autoencoder and are a direct port of the old
gmanifold/sampling.py's free functions. batched_sample is a fully vectorized
equivalent over batched_autoencoder -- every step gains a leading n_models
dim (multinomial per model row, cdist/jacobian already batched).

Both require fit_stats() to have been called first (latent_codes/latent_scale
populated from states)."""

from typing import Any

import torch
from torch import Tensor

from autoencoder.batched_autoencoder import autoencoder, batched_autoencoder, uniform_ball


# ============================================================== single-model


def radii(model: autoencoder, alpha: float, radius: str = "global") -> Tensor:
    """Latent ball radii: 'global' = alpha x median rho for every anchor (a uniform-thickness neighbourhood of
    the data; robust to outliers), 'local' = alpha x rho_i per anchor (the union of local balls is then dominated
    by the largest balls, since ball volume scales like rho^m)."""
    latent_scale = (
        model.latent_scale
        if radius == "local"
        else model.latent_scale.median().expand_as(model.latent_scale)
    )
    return alpha * latent_scale


def log_volume(model: autoencoder, z: Tensor, chunk: int = 1024) -> Tensor:
    """0.5 * log det(J^T J): log of the Riemannian volume element of the chart at z."""
    out: list[Tensor] = []
    for s in range(0, len(z), chunk):
        J = model.jacobian(z[s : s + chunk])
        out.append(0.5 * torch.linalg.slogdet(J.transpose(1, 2) @ J)[1])
    return torch.cat(out)


@torch.no_grad()
def support_logq(
    model: autoencoder, z: Tensor, R: Tensor, anchor_logp: Tensor, chunk: int = 4096
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
    model: autoencoder,
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
    assert model.latent_codes.shape[0] > 0, "call model.fit_stats() before sampling"
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
            print("[autoencoder.sampling] WARNING " + info["warning"])
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


# ================================================================== batched


def batched_radii(model: batched_autoencoder, alpha: float, radius: str = "global") -> Tensor:
    """Per model row: (n_models, n_train_rows)."""
    latent_scale = (
        model.latent_scale
        if radius == "local"
        else model.latent_scale.median(dim=1, keepdim=True).values.expand_as(model.latent_scale)
    )
    return alpha * latent_scale


def batched_log_volume(
    model: batched_autoencoder, z: Tensor, chunk: int = 1024, model_chunk: int | None = None,
) -> Tensor:
    """Per model row: (n_models, B). `chunk`/`model_chunk` bound model.jacobian's
    point/model dims -- see its docstring for why both matter (forward-mode
    AD's memory scales with points x models processed together, not just
    points)."""
    out: list[Tensor] = []
    for s in range(0, z.shape[1], chunk):
        # .contiguous(): see batched_autoencoder.jacobian's docstring/comment -- a
        # non-contiguous slice into vmap(vmap(jacfwd(...))) can blow its internal
        # allocation up to a size tied to z's full (unsliced) extent.
        J = model.jacobian(z[:, s : s + chunk].contiguous(), chunk=chunk, model_chunk=model_chunk)  # (n_models, b, D, m)
        out.append(0.5 * torch.linalg.slogdet(J.transpose(-1, -2) @ J)[1])
    return torch.cat(out, dim=1)


@torch.no_grad()
def batched_support_logq(
    model: batched_autoencoder, z: Tensor, R: Tensor, anchor_logp: Tensor, chunk: int = 2048,
    model_chunk: int | None = 4,
) -> tuple[Tensor, Tensor]:
    """z, R, anchor_logp: (n_models, ...). Returns (logq, cnt), each (n_models, B).

    `model_chunk` additionally bounds the model dim (default: 4; pass None
    for all models at once) -- the (n_models, chunk, n_train_rows) cdist
    result scales with n_models too, so a large n_models can OOM even with a
    small point `chunk` (e.g. n_models=100, n_train_rows=28800, chunk=4096
    tried to allocate 69GiB here). `base` (anchor_logp - latent_dim*log(R))
    doesn't depend on the point chunk `s`, so it's hoisted out of the inner
    loop instead of rebuilt (as a torch.where against a fresh -inf tensor)
    every chunk -- that rebuild was itself a second full-sized tensor live
    alongside `d`/`inside`, which is what was still OOMing one chunking
    level down (11GiB at model_chunk=8, chunk=4096, close to the ~11GiB free
    margin left once the rest of sampling's tensors are resident)."""
    n_models = z.shape[0]
    model_chunk = model_chunk or n_models
    logq = torch.empty(n_models, z.shape[1], device=z.device, dtype=z.dtype)
    cnt = torch.empty(n_models, z.shape[1], device=z.device, dtype=torch.long)
    for m in range(0, n_models, model_chunk):
        me = m + model_chunk
        base = anchor_logp[m:me, None, :] - model.latent_dim * R[m:me, None, :].log()  # (mb, 1, n_train_rows)
        radius = R[m:me, None, :] * (1 + 1e-5)  # (mb, 1, n_train_rows)
        for s in range(0, z.shape[1], chunk):
            e = s + chunk
            d = torch.cdist(z[m:me, s:e], model.latent_codes[m:me])  # (mb, b, n_train_rows)
            inside = d <= radius
            cnt[m:me, s:e] = inside.sum(-1)
            del d
            term = torch.where(inside, base.expand_as(inside), -float("inf"))
            del inside
            logq[m:me, s:e] = term.logsumexp(-1)
            del term
    return logq, cnt


@torch.no_grad()
def batched_sample(
    model: batched_autoencoder,
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
    jacobian_chunk: int = 256,
    jacobian_model_chunk: int | None = None,
    support_chunk: int = 4096,
    support_model_chunk: int | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Fully vectorized equivalent of `sample`, one independent sampler per
    model row. Returns (x, info) with x: (n_models, n, D); info's per-row
    tensors (ess, tau, ...) are (n_models,), and z/anchor/multiplicity are
    (n_models, n, ...). tau tempering (the ESS floor search) is done
    per-model-row via a vectorized bisection, since different rows may need
    different tau to clear the floor.

    `jacobian_chunk`/`jacobian_model_chunk`: passed to batched_log_volume /
    model.jacobian, bounding the point/model dims of the volume-reweighting
    Jacobian pass -- forward-mode AD's memory scales with both together, so
    a large n_models needs jacobian_model_chunk set even with a small
    jacobian_chunk (see model.jacobian's docstring).

    `support_chunk`/`support_model_chunk`: passed to batched_support_logq,
    same reason -- its cdist result also scales with n_models."""
    assert model.latent_codes.shape[0] > 0, "call model.fit_stats() before sampling"
    n_models = model.latent_codes.shape[0]
    g = torch.Generator(device=model.device)
    if seed is not None:
        g.manual_seed(seed)
    n_c = n_candidates or 8 * n
    p = model.latent_dim if anchor_power is None else anchor_power
    R = batched_radii(model, alpha, radius)  # (n_models, n_train_rows)
    anchor_logp = p * R.log()
    anchor_logp = anchor_logp - anchor_logp.logsumexp(-1, keepdim=True)

    # multinomial doesn't batch over a leading dim directly with a shared generator seed
    # in a fully vectorized way across rows with different distributions; sample per row
    # (still one call per model, not per model-and-point) via Categorical, which does batch.
    anchor_dist = torch.distributions.Categorical(probs=anchor_logp.exp())
    i = anchor_dist.sample((n_c,)).T  # (n_models, n_c)

    model_arange = torch.arange(n_models, device=model.device)[:, None]
    R_i = R[model_arange, i]  # (n_models, n_c)
    offsets = uniform_ball(n_models * n_c, model.latent_dim, R_i.reshape(-1), g, model.device)
    z = model.latent_codes[model_arange, i] + offsets.reshape(n_models, n_c, model.latent_dim)
    logq, cnt = batched_support_logq(model, z, R, anchor_logp, chunk=support_chunk, model_chunk=support_model_chunk)
    logvol = (
        torch.nan_to_num(
            batched_log_volume(model, z, chunk=jacobian_chunk, model_chunk=jacobian_model_chunk),
            nan=-float("inf"), posinf=-float("inf"),
        )
        if reweight
        else torch.zeros(n_models, n_c, device=model.device)
    )

    def weights(t: Tensor) -> Tensor:
        """t: (n_models, 1) or scalar-broadcastable -> w: (n_models, n_c)."""
        lw = t * logvol - logq if reweight else torch.zeros_like(logq)
        lw = torch.nan_to_num(lw, nan=-float("inf"), posinf=-float("inf"))
        return (lw - lw.logsumexp(-1, keepdim=True)).exp()

    def ess(w: Tensor) -> Tensor:
        """w: (n_models, n_c) -> (n_models,)."""
        return 1 / (w**2).sum(-1)

    tau_t = torch.full((n_models, 1), float(tau), device=model.device)
    w = weights(tau_t)
    info: dict[str, Any] = dict(
        tau=tau_t.squeeze(-1).clone(), tau_suggested=tau_t.squeeze(-1).clone(), warning=[None] * n_models,
    )
    ess_val = ess(w)
    needs_temper = reweight and min_ess_frac and (ess_val < min_ess_frac * n_c)
    if bool(needs_temper.any()):
        lo = torch.zeros(n_models, 1, device=model.device)
        hi = tau_t.clone()
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            ok = ess(weights(mid)) >= min_ess_frac * n_c
            lo = torch.where(ok[:, None], mid, lo)
            hi = torch.where(ok[:, None], hi, mid)
        info["tau_suggested"] = torch.where(needs_temper, lo.squeeze(-1), tau_t.squeeze(-1))
        if auto_tau:
            tau_t = torch.where(needs_temper[:, None], lo, tau_t)
            w = weights(tau_t)
            info["tau"] = tau_t.squeeze(-1).clone()
        else:
            for m in range(n_models):
                if bool(needs_temper[m]):
                    top = float(w[m].sort(descending=True).values[:10].sum())
                    msg = (
                        f"model {m}: degenerate importance weights: ESS {float(ess_val[m]):.0f} of {n_c} candidates, "
                        f"the 10 heaviest carry {top:.0%} of the mass; the samples will be near-copies of a few points. "
                        f"Temper the volume weights with tau≈{float(lo[m]):.2f} (tau=..., auto_tau=True), "
                        "and check the fit (held-out recon, latent dim, outlier states)."
                    )
                    info["warning"][m] = msg
                    print("[autoencoder.sampling] WARNING " + msg)

    pick = torch.multinomial(w, n, replacement=True, generator=g)  # (n_models, n)
    info.update(
        z=z[model_arange, pick],
        anchor=i[model_arange, pick],
        multiplicity=cnt[model_arange, pick],
        ess=ess(w),
        alpha=alpha,
        radius=radius,
        n_candidates=n_c,
        anchor_participation=1 / (anchor_logp.exp() ** 2).sum(-1),
    )
    return model.decode(z[model_arange, pick]), info
