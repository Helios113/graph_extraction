"""Small Euclidean helpers (no normalisation to the sphere anywhere)."""
import torch


@torch.no_grad()
def knn(X, k, Q=None, chunk=4096):
    """k nearest neighbours of Q (default X itself, self excluded) in X -> (dist, idx), each (n, k)."""
    self_q = Q is None
    Q = X if self_q else Q
    kk = min(k + int(self_q), len(X))
    ds, js = [], []
    for s in range(0, len(Q), chunk):
        d = torch.cdist(Q[s:s + chunk], X)
        if self_q:
            d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = float("inf")
        dd, jj = d.topk(kk, largest=False)
        ds.append(dd[:, :k] if self_q else dd); js.append(jj[:, :k] if self_q else jj)
    return torch.cat(ds), torch.cat(js)


def uniform_ball(n, m, radius, generator=None, device=None):
    """Uniform points in m-balls of the given radii (n,): direction ~ normalised Gaussian, r = R s^{1/m}."""
    g = torch.randn(n, m, device=device, generator=generator)
    g = g / g.norm(dim=1, keepdim=True)
    return g * (radius.to(device) * torch.rand(n, device=device, generator=generator) ** (1.0 / m))[:, None]


@torch.no_grad()
def intrinsic_dimension(X, K=32):
    """TwoNN (Facco et al. 2017) and Levina-Bickel MLE estimates; duplicates removed first."""
    X = torch.unique(X, dim=0)
    dist, _ = knn(X, K)
    mu = (dist[:, 1] / dist[:, 0].clamp_min(1e-12)).sort().values
    keep = int(len(mu) * 0.9); mu = mu[:keep]
    F = torch.arange(1, keep + 1, dtype=mu.dtype, device=mu.device) / len(X)
    x, y = mu.log(), -(1 - F).log()
    inv = (dist[:, K - 1:K].clamp_min(1e-12) / dist[:, :K - 1].clamp_min(1e-12)).log().sum(1) / (K - 2)
    return dict(twonn=float((x * y).sum() / (x * x).sum()), mle=float(1 / inv.mean()), spacing_median=float(dist[:, 7].median()))


@torch.no_grad()
def degenerate_mask(X, rel_tol=0.05):
    """True for rows whose nearest neighbour is closer than rel_tol x the median nearest-neighbour distance
    (near-duplicate states, e.g. never-trained embedding rows). Such rows make the local spacing meaningless and
    should be dropped before fitting."""
    d = knn(X, 1)[0][:, 0]
    return d < rel_tol * d.median()


@torch.no_grad()
def outlier_mask(X, rel_tol=10.0):
    """True for isolated rows whose nearest neighbour is farther than rel_tol x the median nearest-neighbour
    distance (e.g. massive-activation states in deep Llama layers). Such points dominate a volume-uniform
    sampler (the decoder must stretch enormously to reach them) and should be dropped before fitting."""
    d = knn(X, 1)[0][:, 0]
    return d > rel_tol * d.median()
