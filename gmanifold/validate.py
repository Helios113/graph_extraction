"""Independent checks of sampled points against a real reference cloud (never used for sampling).
KernelScore is Guidotti's kernel signature (arXiv:2404.00427): u(x) = (1/N) sum_j Lambda_j K_sigma(x, x_j),
(M + N lambda I) Lambda = N 1; u = 1 on the fitted states, decays away from the cloud."""
import torch

from .geometry import knn


class KernelScore:
    def __init__(self, X, sigma=None, lam=1e-6, n_max=8192, seed=0):
        X = X.double()
        if len(X) > n_max:
            X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:n_max].to(X.device)]
        self.X, self.lam, self.N = X, lam, len(X)
        self.sigma = float(sigma or knn(X.float(), 32)[0][:, -1].median())      # default: median 32-NN distance
        M = self._kernel(X); M.diagonal().add_(self.N * lam)
        self.Lam = torch.linalg.solve(M, self.N * torch.ones(self.N, dtype=X.dtype, device=X.device))

    def _kernel(self, x):
        return torch.exp(-torch.cdist(x.double(), self.X) ** 2 / (2 * self.sigma ** 2))

    @torch.no_grad()
    def __call__(self, Y, chunk=4096):
        return torch.cat([self._kernel(Y[s:s + chunk]) @ self.Lam / self.N for s in range(0, len(Y), chunk)]).float()


@torch.no_grad()
def support_report(M, samples, X_heldout=None, kernel=None, tangent=None, noise=(0.5, 1.0), seed=0):
    """Medians of the support diagnostics for samples, next to held-out real states and to held-out states
    displaced by `noise` x local spacing (the calibration bands). Columns: nearest_real, recon (decoder off-manifold
    error), u (Guidotti kernel score, if `kernel`), tangent (normal residual to the nearest local chart, if `tangent`)."""
    def row(Y):
        r = dict(nearest_real=float(M.nearest_real(Y).median()), recon=float(M.recon_error(Y).median()))   # in M.spacing_unit units
        if kernel is not None:
            r["u"] = float(kernel(Y).median())
        if tangent is not None:
            r["tangent"] = float(tangent.residual(Y).median())
        return r
    out = {"samples": row(samples)}
    if X_heldout is not None:
        Xh = X_heldout.to(M.device).float()
        out["held-out real"] = row(Xh)
        g = torch.Generator(device=M.device).manual_seed(seed)
        j = knn(M.X, 1, Xh)[1][:, 0]
        for f in noise:
            n = torch.randn(Xh.shape, device=M.device, generator=g)
            out[f"real + {f:g}x noise"] = row(Xh + n / n.norm(dim=1, keepdim=True) * (f * M.unit(j))[:, None])
    return out


def check_sampling(info, min_ess_frac=0.05, verbose=True):
    """Inspect the `info` returned by `GlobalManifold.sample` for the degenerate-weights signature (a few candidates with
    extreme volume elements taking all the importance mass). Returns (ok, message). With verbose=True the message is
    printed loudly when the check fails, together with the tau that would restore the ESS floor."""
    n_c = info["n_candidates"]; ess_frac = info["ess"] / n_c
    mult = info["multiplicity"]; dup = float((mult > 1).float().mean()) if hasattr(mult, "float") else float("nan")
    ok = ess_frac >= min_ess_frac and info.get("warning") is None
    msg = (f"ESS {info['ess']:.0f} of {n_c} candidates ({ess_frac:.1%}), tau = {info['tau']:.2f}"
           + (f", suggested tau = {info['tau_suggested']:.2f}" if info.get("tau_suggested", info["tau"]) != info["tau"] else "")
           + f", anchors participating ≈ {info['anchor_participation']:.0f}")
    if not ok:
        msg = ("DEGENERATE SAMPLING: " + msg + ". A handful of candidates dominate the weights, so the samples are near-copies of a few points. "
               f"Temper the volume weights: sample(..., tau={info.get('tau_suggested', 0.5):.2f}) or sample(..., auto_tau=True); "
               "also check the fit (held-out recon error, latent dim, isolated outlier states in X).")
        if verbose:
            print("[gmanifold] " + "!" * 20 + "\n[gmanifold] " + msg + "\n[gmanifold] " + "!" * 20)
    elif verbose:
        print("[gmanifold] sampling ok: " + msg)
    return ok, msg
