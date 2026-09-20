"""Loads a trained autoencoder (single checkpoint or batched_autoencoder
directory) and draws volume-uniform samples from it via autoencoder.sampling.

Requires the same config/layers/sample_idx used at training time, to rebuild
the exact full_activations pool reattach() needs (see autoencoder/data.py and
scripts/python/train_batched_manifold.py).
"""

import argparse

import torch

from autoencoder.batched_autoencoder import load, load_batched
from autoencoder.data import load_sweep_data, sweep_dir_for
from autoencoder import sampling
from config import load_config
from execute.sharded_writer import discover_layer_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="path to a single model_{i}.pt checkpoint")
    p.add_argument("--batched-dir", default=None, help="directory of model_{i}.pt checkpoints (save_all output)")
    p.add_argument("config", type=str, help="Path to the same toml ModelConfig used at training time -- "
                   "sweep_dir is derived as onnx_models/{cfg.name}/perturbed_sweep")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--layers", nargs="*", default=None, help="defaults to every layer in sweep_dir; must match training")
    p.add_argument("--n", type=int, default=1000, help="number of samples to draw")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--radius", choices=["global", "local"], default="global")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--auto-tau", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jacobian-chunk", type=int, default=256,
                   help="point-dim chunk size for the volume-reweighting Jacobian pass (batched mode only)")
    p.add_argument("--jacobian-model-chunk", type=int, default=8,
                   help="model-dim chunk size for the same pass -- bounds forward-mode AD memory when many "
                        "models are loaded at once (batched mode only; default: all models at once)")
    p.add_argument("--knn-chunk", type=int, default=512, help="point-dim chunk size for fit_stats()'s knn passes")
    p.add_argument("--knn-model-chunk", type=int, default=8,
                   help="model-dim chunk size for fit_stats()'s knn passes (batched mode only; default: "
                        "all models at once)")
    p.add_argument("--support-chunk", type=int, default=2048,
                   help="point-dim chunk size for the support-density cdist pass (batched mode only)")
    p.add_argument("--support-model-chunk", type=int, default=4,
                   help="model-dim chunk size for the same cdist pass -- its result scales with n_models too "
                        "(batched mode only)")
    p.add_argument("--out", default=None, help="if set, torch.save the sampled points here")
    args = p.parse_args()

    assert (args.model is None) != (args.batched_dir is None), "pass exactly one of --model / --batched-dir"

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    full_layers = discover_layer_names(sweep_dir)
    layers = args.layers or full_layers
    full_activations = load_sweep_data(sweep_dir, layers, args.sample_idx)

    if args.model:
        model = load(args.model, device=device)
        model.reattach(full_activations[model.model_index])
        model.fit_stats()
        x, info = sampling.sample(
            model, args.n, alpha=args.alpha, radius=args.radius, tau=args.tau, auto_tau=args.auto_tau, seed=args.seed,
        )
        print(f"sampled {x.shape} points, ess={info['ess']:.0f}/{info['n_candidates']}, tau={info['tau']:.2f}")
    else:
        # See validate_autoencoder.py's equivalent check -- reattach()'s row
        # alignment against full_activations (built from `layers`) requires the
        # checkpoint dir to hold exactly this global index set, not a subset from
        # an in-progress multi-batch training run.
        expected_indices = {
            full_layers.index(layer) * cfg.seq + pos for layer in layers for pos in range(cfg.seq)
        }
        model = load_batched(args.batched_dir, device=device, expected_indices=expected_indices)
        model.reattach(full_activations.to(device))
        model.fit_stats(knn_chunk=args.knn_chunk, knn_model_chunk=args.knn_model_chunk)
        x, info = sampling.batched_sample(
            model, args.n, alpha=args.alpha, radius=args.radius, tau=args.tau, auto_tau=args.auto_tau, seed=args.seed,
            jacobian_chunk=args.jacobian_chunk, jacobian_model_chunk=args.jacobian_model_chunk,
            support_chunk=args.support_chunk, support_model_chunk=args.support_model_chunk,
        )
        print(f"sampled {x.shape} points")
        for m in range(x.shape[0]):
            print(f"  model {m}: ess={info['ess'][m]:.0f}/{info['n_candidates']}, tau={info['tau'][m]:.2f}")

    if args.out:
        torch.save(x.cpu(), args.out)
        print(f"saved samples to {args.out}")


if __name__ == "__main__":
    main()
