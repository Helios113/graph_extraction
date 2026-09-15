"""Local tangent charts as an independent validator and as a conservative cross-check sampler.
Tangent bases U_i come from local PCA over K = max(4m, 32) > m Euclidean neighbours (a centred K-neighbourhood
has rank <= K-1) and are computed lazily for the anchors that are queried, so memory is bounded by the query size.
Validation: normal residual of a point to the nearest anchor's chart, in units of that anchor's spacing.
Sampling: anchor ∝ r_i^m, u uniform in the intrinsic m-ball of radius alpha*r_i, x = x_i + U_i u
(approximately density-corrected local sampling, not exact uniform manifold volume)."""
import torch

from .geometry import knn, uniform_ball


class TangentCharts:
    def __init__(self, X, m, K=None, K_s=8):
        self.X, self.m, self.K_s = X.float(), m, K_s
        self.K = K or max(4 * m, 32)
        if self.K <= m:
            raise ValueError(f"K={self.K} neighbours cannot span an m={m} tangent space (need K > m)")
        dist, self.idx = knn(self.X, self.K)
        self.r = dist[:, K_s - 1]                                   # local spacing of each anchor
        hub = float(self.r[self.idx[:, 0]].median()) < 0.5 * float(self.r.median())   # hub-dominated cloud (see GlobalManifold)
        self.r_unit = self.r.median().expand_as(self.r) if hub else self.r      # length unit for residuals

    @torch.no_grad()
    def bases(self, anchors, chunk=256):
        """Tangent bases U (n, D, m) of the given anchors: centred SVD of each K-neighbourhood."""
        U = []
        for s in range(0, len(anchors), chunk):
            nb = self.X[self.idx[anchors[s:s + chunk]]]; nb = nb - nb.mean(1, keepdim=True)
            U.append(torch.linalg.svd(nb, full_matrices=False)[2][:, :self.m].transpose(1, 2))
        return torch.cat(U)

    @torch.no_grad()
    def residual(self, Y):
        """Normal residual to the nearest anchor's tangent chart / that anchor's spacing (0 on the chart)."""
        Y = Y.to(self.X.device).float()
        j = knn(self.X, 1, Y)[1][:, 0]
        uniq, inv = torch.unique(j, return_inverse=True)
        U = self.bases(uniq)[inv]
        d = Y - self.X[j]
        u = (d[:, None] @ U)[:, 0]
        return (d - (U @ u[..., None])[..., 0]).norm(dim=1) / self.r_unit[j]

    @torch.no_grad()
    def sample(self, n, alpha=0.3, seed=None):
        """Conservative local samples x = x_i + U_i u, anchor i ∝ r_i^m, u uniform in the m-ball of radius alpha*r_i."""
        g = torch.Generator(device=self.X.device)
        if seed is not None:
            g.manual_seed(seed)
        logp = self.m * (alpha * self.r).log(); logp = logp - logp.logsumexp(0)
        i = torch.multinomial(logp.exp(), n, replacement=True, generator=g)
        uniq, inv = torch.unique(i, return_inverse=True)
        U = self.bases(uniq)[inv]
        u = uniform_ball(n, self.m, alpha * self.r[i], g, self.X.device)
        return self.X[i] + (U @ u[..., None])[..., 0], dict(anchor=i, u=u, alpha=alpha)
