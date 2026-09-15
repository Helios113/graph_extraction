import torch
import pytest


@pytest.fixture(scope="session")
def sheet():
    """A curved 2-D sheet in R^64 with volume element sqrt(1 + (2.4 cos 3s)^2); returns (X_train, X_val, st_train, vol)."""
    g = torch.Generator(device="cuda").manual_seed(0)
    A = torch.linalg.qr(torch.randn(64, 3, device="cuda", generator=g))[0].T * 3
    st = torch.rand(5000, 2, device="cuda", generator=g)
    X = torch.stack([st[:, 0], st[:, 1], 0.8 * torch.sin(3 * st[:, 0])], 1) @ A
    vol = lambda s: torch.sqrt(1 + (2.4 * torch.cos(3 * s)) ** 2)
    return X[:4500], X[4500:], st[:4500], vol


@pytest.fixture(scope="session")
def sheet_model(sheet):
    import gmanifold as gm
    Xtr, Xval, _, _ = sheet
    return gm.GlobalManifold(latent_dim=2, hidden=(128, 64)).fit(Xtr, epochs=200, X_val=Xval, seed=0)
