"""Generalization check for a trained autoencoder (single checkpoint or
batched_autoencoder directory): reattaches the full activation pool,
recomputes the exact train/val split used at training time (same seed,
n_train_rows), and reports:

  - recon/geom loss, train vs. val (the same terms train_batched_manifold.py
    optimizes, so this is a direct generalization-gap number)
  - recon_error / nearest_real, train vs. val, in local-spacing units (see
    batched_autoencoder.unit) -- off-manifold distance and distance-to-real-
    support, normalized so results are comparable across models/layers with
    different local densities, not just raw MSE.
"""

import argparse

import torch

from autoencoder.batched_autoencoder import load, load_batched
from autoencoder.data import load_sweep_data, sweep_dir_for
from config import load_config
from execute.sharded_writer import discover_layer_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _recon_geom(
    model, x: torch.Tensor, n_neighbors: int, dim: int, chunk: int = 512, model_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same recon/geom loss terms as training, evaluated on x (no gradient).
    x: (..., N, D) -- knn is computed against x itself (self-excluded).

    `chunk`/`model_chunk` (batched path only) bound the point/model dims of
    the knn and encode/decode passes below -- x here is a full states/val
    pool (not a training-loop-sized minibatch), so left unchunked these
    scale the same way train_batched_manifold.py's chunked_recon_loss was
    written to avoid (see its docstring) -- x=(100, ~28800, 768) OOMs here
    too, same root cause as fit_stats/sampling.batched_support_logq."""
    from autoencoder.batched_autoencoder import knn, knn_single

    with torch.no_grad():
        if dim == 3:  # batched: (n_models, N, D)
            dist, nbr_idx = knn(x, n_neighbors, chunk=chunk, model_chunk=model_chunk, device=x.device)
            model_arange = torch.arange(x.shape[0], device=x.device)[:, None, None]
            n_models = x.shape[0]
            recon_sq_sum = torch.zeros(n_models, device=x.device)
            diff_sq_sum = torch.zeros(n_models, device=x.device)
            ambient_sq_sum = torch.zeros(n_models, device=x.device)
            for s in range(0, x.shape[1], chunk):
                e = s + chunk
                x_c = x[:, s:e]
                nbr_c = nbr_idx[:, s:e]
                neighbors = x[model_arange, nbr_c]  # (n_models, c, k, D)
                z = model.encode(x_c)
                z_neighbors = model.encode(
                    neighbors.reshape(n_models, -1, x.shape[-1])
                ).reshape(n_models, x_c.shape[1], n_neighbors, model.latent_dim)
                x_recon = model.decode(z)
                ambient_dist = (neighbors - x_c[:, :, None]).norm(dim=-1)
                latent_dist = (z_neighbors - z[:, :, None]).norm(dim=-1)
                recon_sq_sum += ((x_recon - x_c) ** 2).sum(-1).sum(-1)
                diff_sq_sum += ((latent_dist - ambient_dist) ** 2).sum((-1, -2))
                ambient_sq_sum += (ambient_dist ** 2).sum((-1, -2))
            recon_loss = recon_sq_sum / x.shape[1]  # (n_models,)
            geom_loss = diff_sq_sum / ambient_sq_sum
        else:  # single: (N, D)
            dist, nbr_idx = knn_single(x, n_neighbors)
            neighbors = x[nbr_idx]  # (N, k, D)
            z = model.encode(x)
            z_neighbors = model.encode(neighbors.reshape(-1, x.shape[-1])).reshape(x.shape[0], n_neighbors, model.latent_dim)
            x_recon = model.decode(z)
            ambient_dist = (neighbors - x[:, None]).norm(dim=-1)
            latent_dist = (z_neighbors - z[:, None]).norm(dim=-1)
            recon_loss = ((x_recon - x) ** 2).sum(-1).mean()
            geom_loss = ((latent_dist - ambient_dist) ** 2).mean() / (ambient_dist ** 2).mean()
    return recon_loss, geom_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="path to a single model_{i}.pt checkpoint")
    p.add_argument("--batched-dir", default=None, help="directory of model_{i}.pt checkpoints (save_all output)")
    p.add_argument("config", type=str, help="Path to the same toml ModelConfig used at training time -- "
                   "sweep_dir is derived as onnx_models/{cfg.name}/perturbed_sweep")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--layers", nargs="*", default=None, help="defaults to every layer in sweep_dir; must match training")
    p.add_argument("--k", type=int, default=32, help="neighbors for the recon/geom loss terms; should match training's --k")
    p.add_argument("--chunk", type=int, default=512,
                   help="point-dim chunk size for knn/encode/decode passes (batched mode only)")
    p.add_argument("--model-chunk", type=int, default=8,
                   help="model-dim chunk size for the same passes -- their memory scales with n_models too "
                        "(batched mode only; default: all models at once)")
    args = p.parse_args()

    assert (args.model is None) != (args.batched_dir is None), "pass exactly one of --model / --batched-dir"

    cfg = load_config(args.config)
    sweep_dir = sweep_dir_for(cfg)
    full_layers = discover_layer_names(sweep_dir)
    layers = args.layers or full_layers
    full_activations = load_sweep_data(sweep_dir, layers, args.sample_idx)

    if args.model:
        model = load(args.model, device=device)
        pool = full_activations[model.model_index].to(device)
        model.reattach(pool)
        model.fit_stats()

        # Same recipe as autoencoder.reattach: burn model_index preceding
        # randperm draws from the shared per-batch generator stream, then
        # this model's own draw's tail (n_train_rows:) is its val split.
        n_points = pool.shape[0]
        gen = torch.Generator().manual_seed(model.seed)
        for _ in range(model.model_index):
            torch.randperm(n_points, generator=gen)
        perm = torch.randperm(n_points, generator=gen)
        val_idx = perm[model.n_train_rows:]
        val_data = pool[val_idx.to(device)]

        train_recon, train_geom = _recon_geom(model, model.states, args.k, dim=2)
        val_recon, val_geom = _recon_geom(model, val_data, args.k, dim=2)

        train_re = model.recon_error(model.states).mean()
        val_re = model.recon_error(val_data).mean()
        train_nr = model.nearest_real(model.states).mean()
        val_nr = model.nearest_real(val_data).mean()

        print(f"model_index={model.model_index}  n_train={model.n_train_rows}  n_val={len(val_idx)}")
        print(f"  recon_loss   train={train_recon.item():.4f}  val={val_recon.item():.4f}")
        print(f"  geom_loss    train={train_geom.item():.4f}  val={val_geom.item():.4f}")
        print(f"  recon_error  train={train_re.item():.4f}  val={val_re.item():.4f}  (spacing units)")
        print(f"  nearest_real train={train_nr.item():.4f}  val={val_nr.item():.4f}  (spacing units)")

    else:
        # `layers` here is what full_activations was built from -- reattach()'s row
        # alignment only holds if the checkpoint dir has exactly this set of global
        # indices (layer_idx * seq + pos over `layers`), no more, no fewer; e.g. a
        # directory only some of a multi-batch train_batched_manifold.py run's
        # batches have landed in would otherwise silently reattach a partial/
        # misaligned model instead of failing loudly -- see load_batched's docstring.
        expected_indices = {
            full_layers.index(layer) * cfg.seq + pos for layer in layers for pos in range(cfg.seq)
        }
        model = load_batched(args.batched_dir, device=device, expected_indices=expected_indices)
        pool = full_activations.to(device)
        model.reattach(pool)
        model.fit_stats(knn_chunk=args.chunk, knn_model_chunk=args.model_chunk)

        n_models, n_points = pool.shape[0], pool.shape[1]
        gen = torch.Generator().manual_seed(model.seed)
        perms = torch.stack([torch.randperm(n_points, generator=gen) for _ in range(n_models)])
        val_idx = perms[:, model.n_train_rows:]
        model_arange = torch.arange(n_models)[:, None]
        val_data = pool[model_arange, val_idx.to(pool.device)]

        train_recon, train_geom = _recon_geom(model, model.states, args.k, dim=3, chunk=args.chunk, model_chunk=args.model_chunk)
        val_recon, val_geom = _recon_geom(model, val_data, args.k, dim=3, chunk=args.chunk, model_chunk=args.model_chunk)

        train_re = model.recon_error(model.states, chunk=args.chunk, model_chunk=args.model_chunk).mean(-1)
        val_re = model.recon_error(val_data, chunk=args.chunk, model_chunk=args.model_chunk).mean(-1)
        train_nr = model.nearest_real(model.states, chunk=args.chunk, model_chunk=args.model_chunk).mean(-1)
        val_nr = model.nearest_real(val_data, chunk=args.chunk, model_chunk=args.model_chunk).mean(-1)

        print(f"n_models={n_models}  n_train={model.n_train_rows}  n_val={val_idx.shape[1]}")
        for m in range(n_models):
            print(f"model {m}:")
            print(f"  recon_loss   train={train_recon[m].item():.4f}  val={val_recon[m].item():.4f}")
            print(f"  geom_loss    train={train_geom[m].item():.4f}  val={val_geom[m].item():.4f}")
            print(f"  recon_error  train={train_re[m].item():.4f}  val={val_re[m].item():.4f}  (spacing units)")
            print(f"  nearest_real train={train_nr[m].item():.4f}  val={val_nr[m].item():.4f}  (spacing units)")


if __name__ == "__main__":
    main()
