"""Recreates the gmanifold PCA + kernel-score diagnostic plots (see the
pa511@taranaki:gmanifold snippet this is ported from) for a trained
batched_autoencoder in this repo.

For one (src, dst) sublayer and position `pos`:
  - left column: `src`'s embedding space (the autoencoder's own ambient
    space at that (layer, pos));
  - right column: `dst`, i.e. those same sampled src-points mapped through
    the REAL ONNX sublayer (via run_autoencoder_sampled_sweep's splice-and-run
    mechanism, so attention-crossing sublayers are handled correctly).

Row 1: 2D PCA (fit on held-out real states) of real states vs. samples at
two alphas (tight / loose latent balls).
Row 2: KernelScore below, a Guidotti kernel-density signature (arXiv:2404.00427,
ported from gmanifold.validate -- gmanifold itself is dead/unimportable on
this branch, see its own syntax error in manifold.py) fit on held-out real
states, evaluated on held-out real states, held-out real + Gaussian noise
(0.5x / 1x local spacing -- the calibration bands), and the two sample sets.
Samples should track the "held-out real" band, not the noised ones.

Usage:
  uv run python -m scripts.python.plot_autoencoder_sample_diagnostics configs/smoll_jac.toml \\
      --sublayer <src>.<dst> --pos 0 --alpha 0.3 1.5 --out sample_diagnostics.png

--ae-checkpoints-dir defaults to autoencoder.data.ae_checkpoints_dir_for(cfg,
args.sample_idx) (requires $NEW_JAC_SCRATCH); pass it explicitly to point at a
different save_all() output dir. Requires the same --sample-idx/--layers used
at training time, to rebuild the exact full_activations pool
reattach()/fit_stats() need (see autoencoder/data.py).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from autoencoder import sampling
from autoencoder.batched_autoencoder import knn_single, load_batched
from autoencoder.data import ae_checkpoints_dir_for, load_sweep_data, sweep_dir_for
from config import load_config
from execute.sharded_writer import discover_layer_names
from scripts.python.run_autoencoder_sampled_sweep import (
    context_activations,
    load_base_seq,
    make_subgraph_map,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KernelScore:
    """Guidotti's kernel signature (arXiv:2404.00427): u(x) = (1/N) sum_j
    Lambda_j K_sigma(x, x_j), (M + N lambda I) Lambda = N 1; u = 1 on the
    fitted states, decays away from the cloud. Ported from
    gmanifold.validate.KernelScore (gmanifold itself is dead/unimportable on
    this branch -- see its own syntax error in gmanifold/manifold.py)."""

    def __init__(self, X: torch.Tensor, sigma: float | None = None, lam: float = 1e-6,
                 n_max: int = 8192, seed: int = 0) -> None:
        X = X.double()
        if len(X) > n_max:
            X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:n_max].to(X.device)]
        self.X, self.lam, self.N = X, lam, len(X)
        self.sigma = float(sigma or knn_single(X.float(), 32)[0][:, -1].median())  # default: median 32-NN distance
        M = self._kernel(X)
        M.diagonal().add_(self.N * lam)
        self.Lam = torch.linalg.solve(M, self.N * torch.ones(self.N, dtype=X.dtype, device=X.device))

    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-torch.cdist(x.double(), self.X) ** 2 / (2 * self.sigma ** 2))

    @torch.no_grad()
    def __call__(self, Y: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        return torch.cat([self._kernel(Y[s:s + chunk]) @ self.Lam / self.N for s in range(0, len(Y), chunk)]).float()


def _val_split(model, pool: torch.Tensor, model_index: int) -> torch.Tensor:
    """Held-out states for one model row, using the same seeded-randperm
    recipe as validate_autoencoder.py (burn `model_index` draws from the
    shared per-batch generator stream, then this row's own draw's tail is
    its val split)."""
    n_points = pool.shape[0]
    gen = torch.Generator().manual_seed(model.seed)
    for _ in range(model_index):
        torch.randperm(n_points, generator=gen)
    perm = torch.randperm(n_points, generator=gen)
    val_idx = perm[model.n_train_rows:]
    return pool[val_idx.to(pool.device)]


def _pca_scatter(ax, ref: torch.Tensor, sets: dict[str, torch.Tensor], title: str) -> None:
    mu = ref.mean(0)
    W = torch.linalg.svd(ref - mu, full_matrices=False)[2][:2].T

    def P(Z: torch.Tensor) -> torch.Tensor:
        return ((Z - mu) @ W).cpu()

    p = P(ref)
    ax.scatter(p[:, 0], p[:, 1], s=5, c="#1baf7a", alpha=0.45, lw=0, label="real (held-out)")
    for (lab, Z), col in zip(sets.items(), ("#2a78d6", "#eb6834")):
        p = P(Z)
        ax.scatter(p[:, 0], p[:, 1], s=4, c=col, alpha=0.5, lw=0, label=lab)
    ax.set_title(f"PCA of real states and samples — {title}")
    ax.legend(fontsize=8, markerscale=3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def _kernel_hist(ax, kern: KernelScore, X_train: torch.Tensor, X_val: torch.Tensor, unit: torch.Tensor,
                  sets: dict[str, torch.Tensor], title: str, seed: int) -> None:
    j = knn_single(X_train, 1, X_val)[1][:, 0]
    g = torch.Generator(device=X_val.device).manual_seed(seed)
    nz = torch.randn(X_val.shape, device=X_val.device, generator=g)
    nz = nz / nz.norm(dim=1, keepdim=True) * unit[j][:, None]
    series = {"held-out real": (kern(X_val), "#1baf7a"),
              "real + 0.5x noise": (kern(X_val + 0.5 * nz), "#9085e9"),
              "real + 1x noise": (kern(X_val + nz), "#4a3aa7")}
    colors = ("#2a78d6", "#eb6834")
    for (lab, Z), col in zip(sets.items(), colors):
        series[lab] = (kern(Z), col)
    lo = min(float(v.quantile(0.01)) for v, _ in series.values())
    hi = max(float(v.quantile(0.99)) for v, _ in series.values())
    for lab, (v, col) in series.items():
        ax.hist(v.cpu().numpy(), bins=40, range=(lo, hi), density=True, histtype="step", lw=1.6, color=col, label=lab)
    ax.set_xlabel("kernel score u")
    ax.set_yticks([])
    ax.set_title(f"kernel score distributions — {title}")
    ax.legend(fontsize=8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=str, help="Path to a toml ModelConfig")
    p.add_argument("--ae-checkpoints-dir", default=None,
                   help="save_all() output dir from train_batched_manifold.py (default: "
                        "autoencoder.data.ae_checkpoints_dir_for(cfg), i.e. "
                        "$NEW_JAC_SCRATCH/onnx_models/{cfg.name}/ae_checkpoints)")
    p.add_argument("--sublayer", required=True, help="'<src>.<dst>' -- src must have a trained autoencoder")
    p.add_argument("--pos", type=int, required=True, help="sequence position to sample at")
    p.add_argument("--sample-idx", type=int, default=0, help="must match the --sample-idx used at training time")
    p.add_argument("--layers", nargs="*", default=None,
                   help="must match the --layers used at training time (default: every layer in sweep_dir)")
    p.add_argument("--n", type=int, default=2000, help="samples per alpha")
    p.add_argument("--alpha", type=float, nargs=2, default=[0.3, 1.5], help="two alphas to compare (tight, loose)")
    p.add_argument("--radius", choices=["global", "local"], default="global")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--auto-tau", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--knn-chunk", type=int, default=512)
    p.add_argument("--knn-model-chunk", type=int, default=8)
    p.add_argument("--jacobian-chunk", type=int, default=256)
    p.add_argument("--jacobian-model-chunk", type=int, default=8)
    p.add_argument("--support-chunk", type=int, default=2048)
    p.add_argument("--support-model-chunk", type=int, default=4)
    p.add_argument("--out", default=None, help="if set, saves the figure here instead of plt.show()")
    args = p.parse_args()

    src, dst = args.sublayer.split(".")

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    full_layers = discover_layer_names(sweep_dir)
    layers = args.layers or full_layers
    assert src in layers, f"{src} not in layers {layers}"
    seq = cfg.seq
    model_index = full_layers.index(src) * seq + args.pos

    ae_checkpoints_dir = args.ae_checkpoints_dir or ae_checkpoints_dir_for(cfg, args.sample_idx)
    print(f"loading batched_autoencoder from {ae_checkpoints_dir}", flush=True)
    # reattach()'s row alignment against full_activations (built from `layers` below)
    # requires the checkpoint dir to hold exactly this global index set -- see
    # validate_autoencoder.py's equivalent check.
    expected_indices = {full_layers.index(layer) * seq + pos for layer in layers for pos in range(seq)}
    model = load_batched(ae_checkpoints_dir, device=device, expected_indices=expected_indices)
    model.eval()

    print("reattaching states and computing fit_stats", flush=True)
    full_activations = load_sweep_data(sweep_dir, layers, args.sample_idx)
    model.reattach(full_activations.to(device))
    model.fit_stats(knn_chunk=args.knn_chunk, knn_model_chunk=args.knn_model_chunk)

    pool = full_activations[model_index].to(device)  # (n_vocab, d_model) real src activations at (src, pos)
    X_val = _val_split(model, pool, model_index)

    print(f"sampling n={args.n} at alpha={args.alpha}", flush=True)
    samples_emb: dict[str, torch.Tensor] = {}
    for a in args.alpha:
        x, info = sampling.batched_sample(
            model, args.n, alpha=a, radius=args.radius, tau=args.tau, auto_tau=args.auto_tau, seed=args.seed,
            jacobian_chunk=args.jacobian_chunk, jacobian_model_chunk=args.jacobian_model_chunk,
            support_chunk=args.support_chunk, support_model_chunk=args.support_model_chunk,
        )
        print(f"  alpha={a}: ess={float(info['ess'][model_index]):.0f}/{info['n_candidates']}, "
              f"tau={float(info['tau'][model_index]):.2f}", flush=True)
        samples_emb[f"samples α={a:g}"] = x[model_index]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # see run_autoencoder_sampled_sweep.run_sublayer for why

    print(f"getting real context activations for sample_idx={args.sample_idx}", flush=True)
    base_ids = load_base_seq(sweep_dir, args.sample_idx)
    ctx = context_activations(cfg, src, base_ids)  # (seq, d_model), real values at every position
    map_fn = make_subgraph_map(cfg, src, dst)

    def through_real_model(x: torch.Tensor) -> torch.Tensor:
        """x: (n, d_model) samples at (src, pos) -> (n, d_model) real dst at pos, splicing
        each sample into position `pos` of the real per-position context (see
        run_autoencoder_sampled_sweep's module docstring for why this is correct
        even for attention-crossing sublayers, unlike zero-padding every other position)."""
        n = x.shape[0]
        spliced = np.broadcast_to(ctx, (n, seq, ctx.shape[-1])).copy()
        spliced[:, args.pos] = x.detach().cpu().numpy().astype(np.float32)
        out = map_fn(spliced.astype(np.float32))
        return torch.from_numpy(np.ascontiguousarray(out[:, args.pos])).to(device)

    print(f"mapping real states and samples through {src}->{dst}", flush=True)
    X_layer_val = through_real_model(X_val)
    samples_layer = {lab: through_real_model(z) for lab, z in samples_emb.items()}

    print("fitting kernel scores", flush=True)
    X_train = model.states[model_index]  # (n_train_rows, d_model), what fit_stats() fit latent_codes on
    X_train_layer = through_real_model(X_train)
    k_in = KernelScore(X_train)
    k_out = KernelScore(X_train_layer)

    # Per-train-point local spacing in each space (gmanifold.manifold.unit's
    # anchor-spacing recipe): src's is already computed by fit_stats();
    # dst's has no batched_autoencoder equivalent, so it's the plain
    # nearest-neighbor distance of the real dst cloud, same definition.
    unit_in = model.ambient_spacing[model_index]
    unit_out = knn_single(X_train_layer, 1)[0][:, 0]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        (X_val, samples_emb, src, k_in, X_train, unit_in),
        (X_layer_val, samples_layer, f"{dst} (images through the real model)", k_out, X_train_layer, unit_out),
    )
    for col, (ref, sets, title, kern, X_train_k, unit) in enumerate(panels):
        _pca_scatter(ax[0, col], ref, sets, title)
        _kernel_hist(ax[1, col], kern, X_train_k, ref, unit, sets, title, args.seed)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
