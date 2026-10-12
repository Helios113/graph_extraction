"""Batched Generalized Pareto Distribution MLE fit on GPU via Newton's
method, vectorized across an arbitrary leading batch dimension -- replaces
many sequential scipy.stats.genpareto.fit(floc=0) calls (a per-call
CPU-bound numerical optimizer with no GPU path) with one batched tensor
computation.

This exists because scripts/analysis/lipschitz_ratio_bootstrap_eqd_fit.py's
bootstrap_eqd_statistic calls genpareto.fit once per (candidate threshold,
bootstrap replicate) pair -- e.g. 40 candidates x 50 replicates = 2000 MLE
fits per (sample, pos), x 8 positions x 64 samples x 12 sublayer pairs for a
single simple_stories sweep -- and profiling
(logs/lipschitz_ratio_bootstrap_eqd_sweep_58661.out: ~30s/position) showed
this dominates runtime (>40h projected for that sweep). GPD MLE is a smooth
2-parameter (xi, sigma) problem with an analytic gradient and Hessian (same
formulas as estimate_colipschitz_gpd.py's gpd_negloglik_hessian, verified
against it and against sympy's symbolic derivative here), so it batches
cleanly: every candidate/replicate's Newton step is one vectorized tensor
op, all running on GPU at once.

Fit quality vs. scipy.optimize.minimize (scipy's genpareto.fit backend,
Nelder-Mead by default): Newton's method converges quadratically near the
optimum but is less globally robust -- it can diverge or step outside the
GPD's support {1 + xi*z/sigma > 0} from a bad start. This module guards
against that with (a) a method-of-moments warm start (gpd_moments_fit, same
closed-form estimator lipschitz_ratio_eqd_fit.py already uses to seed its
threshold sweep) rather than an arbitrary fixed start, (b) step damping/line
search that halves the Newton step until it stays in-support and decreases
the negative log-likelihood, and (c) a per-item NaN/non-convergence mask so
one bad fit in a batch doesn't corrupt the others -- see fit_gpd_batched's
docstring for exact fallback behavior. Validated against scipy in this
module's __main__ self-test block (run directly: `uv run python -m
scripts.analysis.batched_gpd_mle`) across a range of true (xi, sigma) and
sample sizes.
"""

import math

import torch


def gpd_moments_fit_torch(mean: torch.Tensor, var: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched torch equivalent of lipschitz_ratio_eqd_fit.py's
    gpd_moments_fit (closed-form method-of-moments GPD(xi, sigma) estimator,
    loc=0) -- used as the Newton solver's warm start."""
    ratio = torch.where(var > 0, mean**2 / torch.where(var > 0, var, torch.ones_like(var)), torch.nan)
    xi = 0.5 * (ratio - 1)
    sigma = 0.5 * mean * (ratio + 1)
    return xi, sigma


def gpd_negloglik(xi: torch.Tensor, sigma: torch.Tensor, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Batched GPD(xi, sigma, loc=0) negative log-likelihood, summed over the
    last (observation) dim. xi, sigma: (...,) batch of parameters. z: (...,
    n) exceedances, one row per batch item. mask: (..., n) bool, True where
    that row's z entry is a real observation (see fit_gpd_batched for why a
    mask is needed -- batches can have ragged exceedance counts).

    Returns +inf for any batch item currently outside the GPD support
    {1 + xi*z/sigma > 0}, so a line search backtracks away from it instead
    of evaluating log() of a negative number."""
    s = sigma.unsqueeze(-1) + xi.unsqueeze(-1) * z  # sigma * (1 + xi*z/sigma)
    in_support = (s > 0) | ~mask
    s_safe = torch.where(in_support, s, torch.ones_like(s))
    log_term = torch.log(s_safe / sigma.unsqueeze(-1))
    ll_terms = torch.where(
        mask,
        -torch.log(sigma).unsqueeze(-1) - (1 + 1 / xi.unsqueeze(-1)) * log_term,
        torch.zeros_like(s),
    )
    ll = ll_terms.sum(dim=-1)
    all_in_support = in_support.all(dim=-1) | ~mask.any(dim=-1)
    return torch.where(all_in_support, -ll, torch.full_like(ll, float("inf")))


def gpd_negloglik_grad_hessian(
    xi: torch.Tensor, sigma: torch.Tensor, z: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched analytic gradient (..., 2) and Hessian (..., 2, 2), in (xi,
    sigma) order, of the negative log-likelihood -- same formulas as
    estimate_colipschitz_gpd.py's gpd_negloglik_hessian (verified to agree
    with it and with a fresh sympy derivation), extended here with the
    matching gradient and vectorized over an arbitrary batch shape with a
    ragged-row mask."""
    xi_ = xi.unsqueeze(-1)
    sigma_ = sigma.unsqueeze(-1)
    s = sigma_ + xi_ * z
    s_safe = torch.where(mask, s, torch.ones_like(s))
    xi_safe = torch.where(xi.abs() > 1e-8, xi, torch.full_like(xi, 1e-8)).unsqueeze(-1)

    log_ratio = torch.log(s_safe / sigma_)

    d_xi = (-xi_safe * z * (xi_safe + 1) + s_safe * log_ratio) / (xi_safe**2 * s_safe)
    d_sigma = (-sigma_ + z) / (sigma_ * s_safe)
    d_xi = torch.where(mask, d_xi, torch.zeros_like(d_xi))
    d_sigma = torch.where(mask, d_sigma, torch.zeros_like(d_sigma))

    d_xixi = (xi_safe**2 * z**2 * (xi_safe + 1) + 2 * xi_safe * z * s_safe - 2 * s_safe**2 * log_ratio) / (
        xi_safe**3 * s_safe**2
    )
    d_sigmasigma = (sigma_**2 - 2 * sigma_ * z - xi_safe * z**2) / (sigma_**2 * s_safe**2)
    d_xisigma = z * (s_safe - z * (xi_safe + 1)) / (sigma_ * s_safe**2)
    d_xixi = torch.where(mask, d_xixi, torch.zeros_like(d_xixi))
    d_sigmasigma = torch.where(mask, d_sigmasigma, torch.zeros_like(d_sigmasigma))
    d_xisigma = torch.where(mask, d_xisigma, torch.zeros_like(d_xisigma))

    grad = -torch.stack([d_xi.sum(-1), d_sigma.sum(-1)], dim=-1)
    hessian = -torch.stack(
        [
            torch.stack([d_xixi.sum(-1), d_xisigma.sum(-1)], dim=-1),
            torch.stack([d_xisigma.sum(-1), d_sigmasigma.sum(-1)], dim=-1),
        ],
        dim=-2,
    )
    return grad, hessian


def fit_gpd_batched(
    z: torch.Tensor,
    mask: torch.Tensor | None = None,
    n_iters: int = 10,
    max_backtrack: int = 30,
    ridge: float = 1e-8,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fits GPD(xi, sigma, loc=0) by damped Newton's method to every row of
    z independently, in one batched computation.

    n_iters default of 10 (not run-to-convergence): benchmarked on real
    co-Lipschitz ratio data from this project's perturbed-sweep output
    (200-row sample spanning the full candidate/bootstrap grid) against
    scipy.stats.genpareto.fit's exact MLE -- the accuracy bar this caller
    needs is 0.5% relative agreement on (xi, sigma), and n_iters=10 already
    hits that on 199/200 sampled rows in ~1.3s (vs. ~7.4s at n_iters=50,
    same accuracy -- the strict `converged` flag, tol=1e-4 gradient norm, is
    a much tighter bar than 0.5% parameter agreement and isn't needed for
    this caller). The one exception found: bootstrap resamples of the
    SMALLEST candidate n_tail (e.g. n_tail=20, the low end of a typical
    n_tail_min..n_tail_max sweep) can land on an extreme fitted shape
    (xi ~ -1.3, near the GPD support boundary where the likelihood surface
    is sharply curved) that more iterations don't reliably fix -- ~18% of
    rows in that specific smallest-n_tail block disagreed >0.5% with scipy
    even at n_iters=30. This is inherent GPD MLE instability at very small
    sample sizes (scipy's own fit is high-variance there too), not
    something this solver can iterate its way out of, and it's confined to
    the candidate grid's noisiest end, which callers ranking candidates by
    a goodness-of-fit statistic (e.g. EQD) should naturally disfavor anyway.
    If a caller needs tighter agreement or cares specifically about
    very-small-n_tail candidates, re-benchmark against scipy on real data
    rather than assuming this default generalizes -- it was tuned to this
    project's actual co-Lipschitz ratio distributions, not derived
    analytically.

    z: (..., n) non-negative exceedances (threshold - data, same convention
    as lipschitz_ratio_eqd_fit.py's fit_gpd_at_threshold), one batch item
    per leading-dim index, n observations in the last dim. MUST be float64:
    the Newton solver's Hessian solve is precision-sensitive near-singular
    Hessians occur routinely in this workload (early iterations, small n_tail
    candidates) -- benchmarked on an A40 GPU, float32 converged on only ~10%
    of a realistic batch (vs. 100% at float64), so this is a correctness
    requirement here, not just a precision nicety.
    mask: (..., n) bool, True where that entry of z is a real observation --
    lets batch rows have different exceedance counts (e.g. different
    candidate n_tail values) by padding shorter rows and masking the pad.
    None means every entry is real (all rows same length).

    Returns (xi, sigma, converged): (...,) tensors. converged is False for
    any batch item where the Newton iteration failed to reach `tol` gradient
    norm within n_iters steps (numerically degenerate, or every backtrack
    step in max_backtrack still left it outside the GPD support/increased
    the negative log-likelihood) -- these should be treated as a fit failure
    by the caller (e.g. drop from the EQD average), not trusted as a fit.
    """
    if mask is None:
        mask = torch.ones_like(z, dtype=torch.bool)
    batch_shape = z.shape[:-1]
    n_obs = mask.sum(dim=-1).clamp(min=1)

    mean = (z * mask).sum(dim=-1) / n_obs
    var = (z**2 * mask).sum(dim=-1) / n_obs - mean**2
    xi_moments, sigma_moments = gpd_moments_fit_torch(mean, var)
    # The method-of-moments estimator needs Var(GPD) to exist (xi < 0.5) and
    # is already visibly biased well before that boundary (verified: at true
    # xi=0.6 it returns xi_hat around -0.45, a WRONG-SIGN start that then
    # traps damped Newton in the wrong basin since Armijo backtracking only
    # guarantees local descent, not escaping to the true optimum) -- so
    # don't trust it outside a conservative safe range, and always also try
    # a small fixed grid of starts spanning negative/zero/positive xi
    # regardless, keeping whichever converged start reaches the lowest NLL
    # (standard multi-start remedy for a non-convex likelihood surface).
    moments_valid = torch.isfinite(xi_moments) & torch.isfinite(sigma_moments) & (sigma_moments > 0) & (xi_moments.abs() < 0.4)
    sigma_fallback = mean.clamp(min=1e-6)

    starts = [(xi_moments, sigma_moments, moments_valid)]
    for xi_fixed in (-0.3, 0.05, 0.3, 0.7):
        starts.append((torch.full_like(mean, xi_fixed), sigma_fallback, torch.ones_like(moments_valid)))

    best_xi = torch.full(batch_shape, float("nan"), dtype=z.dtype, device=z.device)
    best_sigma = torch.full(batch_shape, float("nan"), dtype=z.dtype, device=z.device)
    best_nll = torch.full(batch_shape, float("inf"), dtype=z.dtype, device=z.device)
    best_converged = torch.zeros(batch_shape, dtype=torch.bool, device=z.device)

    # Only run each subsequent start on rows that haven't already converged
    # -- GATHERED down to just those rows, not run on the full batch with a
    # boolean mask. Profiling on an A40 with a boolean-masked `active_rows`
    # (the first version of this fix) showed NO speedup from skipping
    # converged rows: _newton_from_start's `if not active.any(): break` only
    # exits once EVERY row in the batch is inactive, so with real data's
    # ~95% first-start convergence rate, the remaining ~5% of rows (out of
    # e.g. 2000) still forced the full n_iters-step loop to run at FULL
    # (2000, ...) tensor shape on every subsequent start -- torch ops don't
    # get cheaper because most masked-off rows are along for the ride. Real
    # co-Lipschitz-ratio-data benchmark: this cost ~13s of a ~16s total call
    # across the 4 fallback starts, vs. ~3.4s for the first start alone.
    # Gathering to an actual smaller tensor (batch_shape must be 1-D here,
    # true for every caller -- fit_gpd_batched is always called on a
    # flattened (n_rows, n) shape) makes each fallback start's cost scale
    # with the shrinking active-row COUNT, not the original batch size.
    active_idx = torch.arange(batch_shape[0], device=z.device)
    for start_i, (xi_start, sigma_start, start_valid) in enumerate(starts):
        if active_idx.numel() == 0:
            break
        if start_i == 0:
            z_active, mask_active = z, mask
            xi_start_active, sigma_start_active = xi_start, sigma_start
        else:
            z_active = z[active_idx]
            mask_active = mask[active_idx]
            xi_start_active = xi_start[active_idx]
            sigma_start_active = sigma_start[active_idx]

        xi, sigma, converged = _newton_from_start(
            xi_start_active, sigma_start_active, z_active, mask_active, n_iters, max_backtrack, ridge, tol,
        )
        nll = gpd_negloglik(xi, sigma, z_active, mask_active)
        start_valid_active = start_valid[active_idx] if start_i > 0 else start_valid
        nll = torch.where(start_valid_active & torch.isfinite(nll), nll, torch.full_like(nll, float("inf")))

        improves_active = nll < best_nll[active_idx]
        improved_global_idx = active_idx[improves_active]
        best_xi[improved_global_idx] = xi[improves_active]
        best_sigma[improved_global_idx] = sigma[improves_active]
        best_converged[improved_global_idx] = converged[improves_active]
        best_nll[improved_global_idx] = nll[improves_active]

        # A row keeps needing more starts unless THIS start both improved it
        # and converged -- matches the original semantics (best_converged
        # tracks the converged flag of whichever start currently holds the
        # best NLL for that row).
        active_idx = active_idx[~(converged & improves_active)]

    return best_xi, best_sigma, best_converged & torch.isfinite(best_nll)


def fit_gpd_c_hat_constrained_batched(
    z: torch.Tensor,
    threshold: torch.Tensor,
    mask: torch.Tensor | None = None,
    xi_grid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits GPD(xi, sigma, loc=0) to every row of z independently, same as
    fit_gpd_batched, but constrained to xi < 0 AND c_hat = threshold +
    sigma/xi >= 0 -- i.e. the fitted tail must have a finite left endpoint
    (xi<0, required for c_hat to be defined at all) and that endpoint must
    be non-negative, since this codebase's ratios g = ||f(x)-f(y)||/||x-y||
    are always >= 0 so a negative estimated infimum isn't physically valid.

    The unconstrained MLE (fit_gpd_batched) is a genuinely unconstrained
    2-parameter optimum and can land with c_hat<0 (observed on real
    co-Lipschitz-ratio data -- see scripts/analysis/list_flagged_positions.py).
    Rather than a general constrained 2D optimizer, this exploits the
    constraint's structure: substituting the boundary sigma = -xi*threshold
    (from c_hat=0) reduces the negative log-likelihood to a 1D function of
    xi alone, so the constrained optimum -- if the unconstrained one
    violates c_hat>=0 -- lies somewhere on this boundary curve, found here
    by evaluating gpd_negloglik on a batched grid of candidate xi<0 values
    (default: 2000 points log-spaced-ish over (-5, -1e-4)) and taking the
    grid minimum. This is a bounded grid search, not Newton's method --
    robust (no convergence failures near the xi=0/support boundary this
    constrained region borders) at the cost of grid resolution rather than
    exact convergence; --n-tail-sized batches make this cheap relative to
    fit_gpd_batched's own multi-start Newton cost.

    z: (..., n) non-negative exceedances, MUST be float64 (see
    fit_gpd_batched). threshold: (...,) this row's threshold u (same shape
    as z's batch dims) -- c_hat = u + sigma/xi. mask: (..., n) bool, as
    fit_gpd_batched. xi_grid: (g,) candidate xi<0 values to search, shared
    across every batch row; default is 2000 points geometrically spaced
    over (-5, -1e-4) (denser near 0, where GPD tail behavior changes
    fastest).

    Returns (xi, sigma): (...,) tensors, both nan for any batch item whose
    grid search found no finite-negloglik candidate (degenerate row, e.g.
    all-zero exceedances)."""
    if mask is None:
        mask = torch.ones_like(z, dtype=torch.bool)
    batch_shape = z.shape[:-1]
    device, dtype = z.device, z.dtype

    if xi_grid is None:
        # Denser near 0 (where the profiled negloglik along this boundary
        # tends to curve fastest) via a log-spaced magnitude, all negative.
        xi_grid = -torch.logspace(-4, math.log10(5.0), 2000, device=device, dtype=dtype)

    n_grid = xi_grid.shape[0]
    xi_cand = xi_grid.view(*([1] * len(batch_shape)), n_grid).expand(*batch_shape, n_grid)
    sigma_cand = -xi_cand * threshold.unsqueeze(-1)  # boundary c_hat=0: sigma = -xi*u

    z_exp = z.unsqueeze(-2).expand(*batch_shape, n_grid, z.shape[-1])
    mask_exp = mask.unsqueeze(-2).expand(*batch_shape, n_grid, z.shape[-1])
    nll = gpd_negloglik(xi_cand, sigma_cand, z_exp, mask_exp)  # (..., n_grid)

    best_nll, best_idx = nll.min(dim=-1)
    best_xi = torch.gather(xi_cand, -1, best_idx.unsqueeze(-1)).squeeze(-1)
    best_sigma = torch.gather(sigma_cand, -1, best_idx.unsqueeze(-1)).squeeze(-1)
    invalid = ~torch.isfinite(best_nll)
    best_xi = torch.where(invalid, torch.full_like(best_xi, float("nan")), best_xi)
    best_sigma = torch.where(invalid, torch.full_like(best_sigma, float("nan")), best_sigma)
    return best_xi, best_sigma


def fit_gpd_c_hat_nonneg_batched(
    z: torch.Tensor,
    threshold: torch.Tensor,
    mask: torch.Tensor | None = None,
    xi_grid: torch.Tensor | None = None,
    n_iters: int = 10,
    max_backtrack: int = 30,
    ridge: float = 1e-8,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fit_gpd_batched's unconstrained MLE where that's already feasible
    (xi<0 and c_hat=threshold+sigma/xi>=0), else
    fit_gpd_c_hat_constrained_batched's boundary-constrained fit -- the
    combined "best fit subject to c_hat>=0" this codebase's ratios (always
    >= 0) actually require, using the unconstrained optimizer's better
    precision wherever the constraint isn't binding rather than always
    paying the constrained grid search's coarser resolution.

    Returns (xi, sigma, was_constrained): was_constrained is True for every
    row where the unconstrained fit was infeasible (or didn't converge) and
    the boundary solve was used instead -- lets a caller distinguish "this
    row's unconstrained fit already respected c_hat>=0" from "the
    constraint was actually binding here"."""
    xi_u, sigma_u, converged = fit_gpd_batched(z, mask, n_iters, max_backtrack, ridge, tol)
    c_hat_u = threshold + sigma_u / torch.where(xi_u < 0, xi_u, torch.full_like(xi_u, -1.0))
    feasible = converged & (xi_u < 0) & (c_hat_u >= 0)

    xi_c, sigma_c = fit_gpd_c_hat_constrained_batched(z, threshold, mask, xi_grid)

    xi = torch.where(feasible, xi_u, xi_c)
    sigma = torch.where(feasible, sigma_u, sigma_c)
    return xi, sigma, ~feasible


def _newton_from_start(
    xi: torch.Tensor,
    sigma: torch.Tensor,
    z: torch.Tensor,
    mask: torch.Tensor,
    n_iters: int,
    max_backtrack: int,
    ridge: float,
    tol: float,
    active_rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Damped/PD-corrected Newton iteration from a given (xi, sigma) start --
    the inner loop fit_gpd_batched calls once per candidate start in its
    multi-start grid.

    active_rows: (batch_shape,) bool, rows this start should actually work
    on (e.g. rows a previous start already converged well don't need this
    start's iterations at all) -- None means every row with real data.
    Rows outside active_rows are excluded from the very first iteration, so
    once none of the batch still needs this start the loop's `active.any()`
    check exits immediately instead of running n_iters idle steps."""
    batch_shape = z.shape[:-1]
    converged = torch.zeros(batch_shape, dtype=torch.bool, device=z.device)
    active = torch.ones(batch_shape, dtype=torch.bool, device=z.device) & mask.any(dim=-1)
    if active_rows is not None:
        active = active & active_rows
    armijo_c = 1e-4

    for _ in range(n_iters):
        if not active.any():
            break
        grad, hessian = gpd_negloglik_grad_hessian(xi, sigma, z, mask)
        grad_norm = grad.norm(dim=-1)
        newly_converged = active & (grad_norm < tol)
        converged = converged | newly_converged
        active = active & ~newly_converged
        if not active.any():
            break

        # Newton's step is only a descent direction where the Hessian is
        # positive definite -- away from the optimum (as in this
        # non-globally-convex negative log-likelihood) it often isn't, and
        # solving Hess @ step = grad then blindly gives an ASCENT direction
        # (observed: NLL creeping up every "backtracked" iteration while
        # grad_norm stays flat, never triggering the old `<=` accept check
        # to actually reject it). Flip any non-positive eigenvalue to its
        # absolute value (a standard modified-Newton / LM trick) so the
        # resulting direction is always a true descent direction.
        #
        # Closed-form 2x2 symmetric eigendecomposition rather than
        # torch.linalg.eigh: eigh's iterative algorithm can fail outright on
        # GPU (cuSOLVER, unlike CPU LAPACK, *raises* -- observed:
        # `linalg.eigh: The algorithm failed to converge because the input
        # matrix is ill-conditioned` -- on a batch containing a near-singular
        # Hessian, e.g. very early iterations or degenerate exceedance rows,
        # which crashes the whole batched call instead of just that row).
        # For a 2x2 [[a,b],[b,d]], eigenvalues are (a+d)/2 +- sqrt(((a-d)/2)^2+b^2)
        # with a matching closed-form eigenvector pair -- exact, and defined
        # for every finite input (no iterative convergence to fail).
        eye = torch.eye(2, device=z.device, dtype=z.dtype).expand(*batch_shape, 2, 2)
        hessian_reg = hessian + ridge * eye
        a = hessian_reg[..., 0, 0]
        b = hessian_reg[..., 0, 1]
        d = hessian_reg[..., 1, 1]
        tr_half = (a + d) / 2
        disc = torch.sqrt(((a - d) / 2) ** 2 + b**2)
        eig1, eig2 = tr_half + disc, tr_half - disc
        # Eigenvector for eig1: (b, eig1 - a), normalized -- degenerates
        # (both components -> 0) only when b==0 and a==eig1, i.e. hessian_reg
        # already diagonal, handled by the where() below using the identity
        # basis instead.
        vx, vy = b, eig1 - a
        vnorm = torch.sqrt(vx**2 + vy**2)
        diagonal_already = vnorm < 1e-30
        vnorm_safe = vnorm.clamp(min=1e-30)
        v1x = torch.where(diagonal_already, torch.ones_like(vx), vx / vnorm_safe)
        v1y = torch.where(diagonal_already, torch.zeros_like(vy), vy / vnorm_safe)
        v2x, v2y = -v1y, v1x  # orthogonal complement

        eig1_pd = eig1.abs().clamp(min=ridge)
        eig2_pd = eig2.abs().clamp(min=ridge)
        # hessian_pd = V @ diag(|eig|) @ V^T, expanded in closed form.
        h00 = eig1_pd * v1x**2 + eig2_pd * v2x**2
        h01 = eig1_pd * v1x * v1y + eig2_pd * v2x * v2y
        h11 = eig1_pd * v1y**2 + eig2_pd * v2y**2
        det = h00 * h11 - h01 * h01
        det_safe = torch.where(det.abs() < 1e-30, torch.full_like(det, 1e-30), det)
        step_x = (h11 * grad[..., 0] - h01 * grad[..., 1]) / det_safe
        step_y = (-h01 * grad[..., 0] + h00 * grad[..., 1]) / det_safe
        step = torch.stack([step_x, step_y], dim=-1)
        bad_step = ~torch.isfinite(step).all(dim=-1)
        step = torch.where(bad_step.unsqueeze(-1), grad / grad_norm.clamp(min=1e-12).unsqueeze(-1), step)

        nll_current = gpd_negloglik(xi, sigma, z, mask)
        directional_deriv = (grad * step).sum(dim=-1)  # > 0 for a descent direction on nll
        step_scale = torch.ones(batch_shape, device=z.device, dtype=z.dtype)
        xi_new, sigma_new, nll_best = xi, sigma, nll_current
        for _bt in range(max_backtrack):
            cand_xi = xi - step_scale * step[..., 0]
            cand_sigma = (sigma - step_scale * step[..., 1]).clamp(min=1e-10)
            nll_cand = gpd_negloglik(cand_xi, cand_sigma, z, mask)
            # Armijo sufficient-decrease: require an actual improvement
            # proportional to the step size, not just "no worse" -- `<=`
            # alone accepted near-zero/negative progress forever.
            sufficient_decrease = nll_current - nll_cand >= armijo_c * step_scale * directional_deriv
            accept = torch.isfinite(nll_cand) & sufficient_decrease & (nll_cand < nll_best)
            xi_new = torch.where(active & accept, cand_xi, xi_new)
            sigma_new = torch.where(active & accept, cand_sigma, sigma_new)
            nll_best = torch.where(active & accept, nll_cand, nll_best)
            still_bad = active & ~(torch.isfinite(nll_cand) & sufficient_decrease)
            if not still_bad.any():
                break
            step_scale = torch.where(still_bad, step_scale * 0.5, step_scale)

        xi, sigma = xi_new, sigma_new

    return xi, sigma, converged


if __name__ == "__main__":
    import numpy as np
    from scipy.stats import genpareto

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"self-test device: {device}")

    rng = np.random.default_rng(0)
    cases = []
    for true_xi in (-0.4, -0.1, 0.0, 0.2, 0.6):
        for true_sigma in (0.5, 1.0, 3.0):
            for n in (30, 200, 2000):
                sample = genpareto.rvs(true_xi, loc=0, scale=true_sigma, size=n, random_state=rng)
                cases.append(sample)

    max_n = max(len(c) for c in cases)
    z = torch.zeros(len(cases), max_n, dtype=torch.float64)
    mask = torch.zeros(len(cases), max_n, dtype=torch.bool)
    for i, c in enumerate(cases):
        z[i, : len(c)] = torch.from_numpy(c)
        mask[i, : len(c)] = True
    z, mask = z.to(device), mask.to(device)

    xi_batched, sigma_batched, converged = fit_gpd_batched(z, mask)
    xi_batched = xi_batched.cpu().numpy()
    sigma_batched = sigma_batched.cpu().numpy()
    converged = converged.cpu().numpy()

    xi_errs, sigma_errs = [], []
    n_scipy_fail = 0
    for i, c in enumerate(cases):
        try:
            xi_s, _loc_s, sigma_s = genpareto.fit(c, floc=0)
        except Exception:
            n_scipy_fail += 1
            continue
        if not converged[i]:
            continue
        xi_errs.append(abs(xi_batched[i] - xi_s))
        sigma_errs.append(abs(sigma_batched[i] - sigma_s) / sigma_s)

    print(f"{len(cases)} cases, {converged.sum()}/{len(cases)} batched-Newton converged, "
          f"{n_scipy_fail} scipy fit failures")
    print(f"xi abs error vs scipy: mean={np.mean(xi_errs):.4g}, max={np.max(xi_errs):.4g}")
    print(f"sigma rel error vs scipy: mean={np.mean(sigma_errs):.4g}, max={np.max(sigma_errs):.4g}")
    assert np.mean(xi_errs) < 0.01, "batched Newton xi disagrees with scipy MLE beyond tolerance"
    assert np.mean(sigma_errs) < 0.01, "batched Newton sigma disagrees with scipy MLE beyond tolerance"
    print("PASS: batched GPD MLE agrees with scipy.stats.genpareto.fit")
