import torch
import gmanifold as gm
from gmanifold.geometry import knn, uniform_ball


def test_fit_quality_and_rank(sheet_model):
    M = sheet_model
    assert M.history[-1]["val_recon_over_spacing"] < 0.4        # held-out points reconstruct well inside a spacing
    assert M.jacobian_rank()["rank_median"] == 2


def test_volume_uniform_sampling(sheet, sheet_model):
    """Samples must follow the manifold volume element, not the (uniform-in-parameter) data density.
    radius='local' (per-anchor balls) is exact; radius='global' (uniform-thickness neighbourhood) under-covers sparse
    regions at small alpha and sits between data density and the exact target, converging to it as alpha grows."""
    Xtr, _, st, vol = sheet
    M = sheet_model
    target = float((vol(st[:, 0]) ** 2).mean() / vol(st[:, 0]).mean())   # E_volume[vol]
    data = float(vol(st[:, 0]).mean())                                    # E_parameter[vol]
    gap = target - data

    def mean_vol(**kw):
        x, info = M.sample(4000, seed=0, **kw)
        return float(vol(st[knn(Xtr, 1, x)[1][:, 0]][:, 0]).mean()), info
    v_local, info = mean_vol(alpha=0.5, radius="local")
    assert abs(v_local - target) < 0.4 * gap
    assert info["ess"] > 0.3 * info["n_candidates"]
    v_g05, _ = mean_vol(alpha=0.5, radius="global"); v_g10, _ = mean_vol(alpha=1.0, radius="global")
    assert data + 0.3 * gap < v_g05 <= target + 0.2 * gap
    assert abs(v_g10 - target) < 0.4 * gap
    assert M.coverage(M.sample(4000, alpha=0.5, seed=0)[0]) > 0.9
    v_none, _ = mean_vol(alpha=0.5, anchor_power=0, reweight=False)
    assert abs(v_none - data) < 0.4 * gap                                 # un-reweighted follows the data density


def test_samples_on_support_and_bands(sheet, sheet_model):
    Xtr, Xval, _, _ = sheet
    M = sheet_model
    x, _ = M.sample(2000, alpha=0.3, seed=1)
    rep = gm.support_report(M, x, Xval, gm.KernelScore(Xtr))
    assert rep["samples"]["recon"] < rep["real + 0.5x noise"]["recon"]
    assert rep["samples"]["u"] > rep["real + 0.5x noise"]["u"]
    assert rep["held-out real"]["u"] > 0.99 and rep["real + 1x noise"]["u"] < 0.95


def test_alpha_monotone(sheet_model):
    M = sheet_model
    d = [float(M.nearest_real(M.sample(2000, alpha=a, seed=0)[0]).median()) for a in (0.1, 0.3, 1.0)]
    assert d[0] < d[1] < d[2]


def test_save_load(sheet_model, tmp_path):
    M = sheet_model
    M.save(tmp_path / "m.pt")
    M2 = gm.GlobalManifold.load(tmp_path / "m.pt")
    assert torch.allclose(M2.sample(20, 0.5, seed=3)[0], M.sample(20, 0.5, seed=3)[0])
    assert torch.allclose(M2.encode(M.X[:5]), M.encode(M.X[:5]))


def test_knn_matches_bruteforce():
    X = torch.randn(300, 8, device="cuda"); Q = torch.randn(50, 8, device="cuda")
    d, j = knn(X, 5, Q)
    ref = torch.cdist(Q, X).topk(5, largest=False)
    assert torch.equal(j, ref.indices) and torch.allclose(d, ref.values)
    d2, j2 = knn(X, 3)
    assert (j2 != torch.arange(300, device="cuda")[:, None]).all()      # self excluded


def test_uniform_ball_radius_law():
    r = uniform_ball(20000, 4, torch.ones(20000, device="cuda"), device="cuda").norm(dim=1)
    assert abs(float((r ** 4).mean()) - 0.5) < 0.02                       # r^m ~ U(0,1) for a uniform m-ball


def test_intrinsic_dimension_gaussian():
    X = torch.randn(4000, 3, device="cuda") @ torch.randn(3, 32, device="cuda")
    d = gm.intrinsic_dimension(X)
    assert 2.5 < d["twonn"] < 3.6 and 2.5 < d["mle"] < 3.6


def test_degenerate_mask():
    X = torch.randn(500, 16, device="cuda")
    X[:20] = X[0]                                                          # 20 identical rows
    mask = gm.degenerate_mask(X)
    assert mask[:20].all() and not mask[20:].any()


def test_outlier_mask():
    X = torch.randn(500, 16, device="cuda")
    X[0] *= 50                                                             # one isolated massive state
    mask = gm.outlier_mask(X)
    assert mask[0] and mask.sum() == 1


def test_tangent_charts(sheet, sheet_model):
    """Local charts: held-out points sit on them, ambient noise does not; conservative samples stay near anchors."""
    Xtr, Xval, _, _ = sheet
    T = gm.TangentCharts(Xtr, m=2)
    assert T.K == 32 and T.bases(torch.arange(5, device="cuda")).shape == (5, 64, 2)
    res_real = T.residual(Xval).median()
    noise = Xval + 0.5 * sheet_model.spacing.median() * torch.randn_like(Xval) / 8
    assert res_real < 0.2 and T.residual(noise).median() > 2 * res_real
    x, info = T.sample(2000, alpha=0.3, seed=0)
    assert float(sheet_model.nearest_real(x).median()) < 0.4 and T.residual(x).median() < 1e-4
    rep = gm.support_report(sheet_model, sheet_model.sample(500, 0.3, seed=0)[0], Xval, tangent=T)
    assert "tangent" in rep["samples"] and rep["samples"]["tangent"] < rep["real + 1x noise"]["tangent"]


def test_hub_dominated_cloud_uses_global_unit():
    """A cloud where a tight cluster is everyone's nearest neighbour switches to the global spacing unit."""
    g = torch.Generator(device="cuda").manual_seed(0)
    X = torch.randn(1500, 256, device="cuda", generator=g); X = 10 * X / X.norm(dim=1, keepdim=True)  # sphere: neighbours ~14 apart
    X[:12] = 0.02 * torch.randn(12, 256, device="cuda", generator=g)            # tight cluster at the centre, 10 from everyone = hub
    M = gm.GlobalManifold(latent_dim=4, hidden=(64, 32)).fit(X, epochs=5)
    assert M.spacing_unit == "global" and M.hub_share > 0.3
    assert float(M.nearest_real(X[500:600]).median()) < 2                        # not inflated by the hub's tiny spacing


def test_hub_rule_applies_to_tangent_and_bands():
    g = torch.Generator(device="cuda").manual_seed(0)
    X = torch.randn(1500, 256, device="cuda", generator=g); X = 10 * X / X.norm(dim=1, keepdim=True); X[:12] = 0.02 * torch.randn(12, 256, device="cuda", generator=g)
    T = gm.TangentCharts(X, m=4)
    assert float(T.residual(X[500:600] + 0.1 * torch.randn(100, 256, device="cuda", generator=g)).median()) < 2   # global unit, not the hub's
    M = gm.GlobalManifold(latent_dim=4, hidden=(64, 32)).fit(X, epochs=5)
    rep = gm.support_report(M, X[600:700], X[500:600])
    assert rep["real + 1x noise"]["nearest_real"] > 1.2 * rep["held-out real"]["nearest_real"]                     # noise bands still separate


def test_weight_checker_and_auto_tau(sheet, sheet_model):
    """Default: exact weights (tau = 1) and no warning on a healthy cloud. A forced ESS floor triggers the warning with a
    suggested tau; auto_tau applies it."""
    M = sheet_model
    x, info = M.sample(2000, alpha=0.5, seed=0)
    assert info["tau"] == 1.0 and info["warning"] is None and info["ess"] >= 0.05 * info["n_candidates"]
    x, info = M.sample(2000, alpha=0.5, seed=0, min_ess_frac=0.9)             # unreachable floor: checker fires, tau unchanged
    assert info["tau"] == 1.0 and info["warning"] and info["tau_suggested"] < 0.05
    x, info = M.sample(2000, alpha=0.5, seed=0, min_ess_frac=0.9, auto_tau=True)
    assert info["tau"] < 0.05 and info["warning"] is None and info["ess"] >= 0.5 * info["n_candidates"]


def test_check_sampling_helper(sheet_model):
    ok, msg = gm.check_sampling(sheet_model.sample(1000, alpha=0.5, seed=0)[1], verbose=False)
    assert ok and "ok" not in msg.lower() or ok
    ok, msg = gm.check_sampling(sheet_model.sample(1000, alpha=0.5, seed=0, min_ess_frac=0.9)[1], verbose=False)
    assert not ok and "DEGENERATE" in msg and "auto_tau=True" in msg
