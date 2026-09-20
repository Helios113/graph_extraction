"""Random hyperparameter search for scripts.python.train_batched_manifold's
train(): loads sweep data once, then runs N trials with network-dims/lr/
batch-size drawn randomly and lam-geom drawn from a small fixed set, each
trial a short (--epochs, default 50) training run within this one process --
no per-trial SLURM job, just N sequential train() calls in one srun session.

Score = val_recon_loss + lam_geom * val_geom_loss, matching exactly what
that trial trained on (no separate fixed scoring weight), each term meaned
over models. Lower is better. Prints a ranked table and (with --out) saves
every trial's config + score to a json file.
"""

import argparse
import json
import math
import random
import time

import torch

from autoencoder.data import load_sweep_data, sweep_dir_for
from config import load_config
from execute.sharded_writer import discover_layer_names
from scripts.python.train_batched_manifold import train

NETWORK_DIMS_CHOICES = [[256, 16], [512, 32], [512, 64], [1024, 64]]
BATCH_SIZE_CHOICES = [32, 64, 128, 256]
LAM_GEOM_CHOICES = [0.01, 0.1, 1.0]
LR_LOG_RANGE = (1e-4, 1e-2)


def sample_config(rng: random.Random) -> dict:
    lr_lo, lr_hi = math.log10(LR_LOG_RANGE[0]), math.log10(LR_LOG_RANGE[1])
    return dict(
        network_dims=rng.choice(NETWORK_DIMS_CHOICES),
        lr=10 ** rng.uniform(lr_lo, lr_hi),
        batch_size=rng.choice(BATCH_SIZE_CHOICES),
        lam_geom=rng.choice(LAM_GEOM_CHOICES),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", type=str, help="Path to a toml ModelConfig -- sweep_dir is derived as "
                   "onnx_models/{cfg.name}/perturbed_sweep")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--layers", nargs="*", default=None, help="defaults to every layer in sweep_dir")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=50, help="short training run per trial; not searched")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--knn-chunk", type=int, default=512)
    p.add_argument("--knn-model-chunk", type=int, default=8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--data-seed", type=int, default=0, help="seed for the train/val split, shared across trials")
    p.add_argument("--search-seed", type=int, default=0, help="seed for sampling the hyperparameter configs")
    p.add_argument("--out", default=None, help="if set, write every trial's config + score to this json file")
    args = p.parse_args()

    sweep_dir = sweep_dir_for(load_config(args.config))
    print(f"sweep_dir: {sweep_dir}")
    layers = args.layers or discover_layer_names(sweep_dir)
    print(f"layers ({len(layers)}): {layers}")

    print("loading sweep data", flush=True)
    a = time.time()
    data = load_sweep_data(sweep_dir, layers, args.sample_idx)
    print(f"loaded {data.shape} in {time.time() - a:.1f}s", flush=True)

    rng = random.Random(args.search_seed)
    results = []

    for trial in range(args.n_trials):
        cfg = sample_config(rng)
        print(f"\n=== trial {trial + 1}/{args.n_trials}: {cfg} ===", flush=True)

        t0 = time.time()
        _, metrics = train(
            data,
            network_dims=cfg["network_dims"],
            k=args.k,
            knn_chunk=args.knn_chunk,
            knn_model_chunk=args.knn_model_chunk,
            epochs=args.epochs,
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            lam_geom=cfg["lam_geom"],
            val_frac=args.val_frac,
            seed=args.data_seed,
            verbose=False,
        )
        elapsed = time.time() - t0

        val_recon = float(metrics["val_recon_loss"].mean())
        val_geom = float(metrics["val_geom_loss"].mean())
        score = val_recon + cfg["lam_geom"] * val_geom

        print(f"trial {trial + 1}: val_recon={val_recon:.4f} geom={val_geom:.4f} score={score:.4f} ({elapsed:.0f}s)")
        results.append(dict(trial=trial, **cfg, val_recon_loss=val_recon, geom_loss=val_geom, score=score, elapsed=elapsed))

    results.sort(key=lambda r: r["score"])
    print("\n=== ranked results (best first) ===")
    for r in results:
        print(
            f"score={r['score']:.4f}  val_recon={r['val_recon_loss']:.4f}  geom={r['geom_loss']:.4f}  "
            f"network_dims={r['network_dims']}  lr={r['lr']:.2e}  batch_size={r['batch_size']}  lam_geom={r['lam_geom']}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {len(results)} trial results to {args.out}")


if __name__ == "__main__":
    main()
