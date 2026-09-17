"""GlobalManifold: learn G: R^m -> R^D from real states X and sample ~uniformly (w.r.t. manifold volume)
near the observed support. Knows nothing about Transformers."""

import copy
import math
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


@torch.no_grad()
def knn(X, k, Q=None, chunk=4096, progress=False):
    """k nearest neighbours of Q (default X itself, self excluded) in X -> (dist, idx), each (n, k)."""
    self_q = Q is None
    Q = X if self_q else Q
    kk = min(k + int(self_q), len(X))
    ds, js = [], []
    steps = range(0, len(Q), chunk)
    if progress:
        steps = tqdm(steps, desc="knn", total=math.ceil(len(Q) / chunk))
    for s in steps:
        d = torch.cdist(Q[s:s + chunk], X)
        if self_q:
            d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = float("inf")
        dd, jj = d.topk(kk, largest=False)
        ds.append(dd[:, :k] if self_q else dd); js.append(jj[:, :k] if self_q else jj)
    return torch.cat(ds), torch.cat(js)



class manifold(nn.Module):
    def __init__(
        self,
        network_dims: list[int] = [512, 256],
        n_neighbors: int = 32,
        n_scale_neighbors: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_scale_neighbors = n_scale_neighbors
        self.latent_dim = network_dims[-1]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.encoder: nn.Sequential = self.block(network_dims).to(self.device)

        self.decoder: nn.Sequential = self.block(list(reversed(network_dims))).to(
            self.device
        )

        self.register_buffer("center", torch.zeros(0))
        self.register_buffer("scale", torch.ones(()))

        self.register_buffer(
            "latent_codes",
            torch.zeros(0, self.latent_dim),
        )  # latent codes of the fitted states
        self.register_buffer(
            "latent_scale",
            torch.zeros(0),
        )  # local latent scale (n_scale_neighbors-th latent neighbour)
        self.register_buffer(
            "ambient_spacing",
            torch.zeros(0),
        )  # local ambient spacing (n_scale_neighbors-th neighbour) of states
        self.states: Tensor | None = None
        self.history: list[dict[str, Any]] = []
        self.spacing_unit = "anchor"  # 'anchor': divide by the nearest anchor's spacing; 'global': by the median
        self.hub_share = 0.0  # share of points whose nearest neighbour is one of the 5 most common anchors

    def block(self, network_dims: list[int]) -> nn.Module:
        layers = []
        for i in range(len(network_dims) - 1):
            layers.append(nn.Linear(network_dims[i], network_dims[i + 1]))
            if i != len(network_dims) - 2:
                layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder((x.to(self.device) - self.center) / self.scale)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z.to(self.device)) * self.scale + self.center

    def project(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    def jacobian(self, z: Tensor, chunk: int = 1024) -> Tensor:
        """J_G(z) in ambient units, (B, D, m)."""
        assert self.decoder is not None
        f = torch.func.vmap(torch.func.jacfwd(self.decoder))
        return (
            torch.cat([f(z[s : s + chunk]) for s in range(0, len(z), chunk)])
            * self.scale
        )


# ---------------------------------------------------------------------- fitting
def train(
    model: manifold,
    train_activations: Tensor,
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 2e-3,
    lam_geom: float = 0.1,
    lam_curv: float = 1e-3,
    curv_delta: float = 0.5,
    seed: int = 0,
    val_activations: Tensor | None = None,
    log_every: int = 50,
    verbose: bool = False,
) -> manifold:
    train_activations = train_activations.to(model.device)

    # n_states is number of points
    ambient_dim, n_states = train_activations.shape[1], len(train_activations)

    model.center = train_activations.mean(0)
    model.scale = ((train_activations - model.center) ** 2).sum(1).mean().sqrt()
    train_activations_normalised = (train_activations - model.center) / model.scale

    if val_activations is not None:
        val_activations = val_activations.to(model.device)
        val_activations_normalised = (val_activations - model.center) / model.scale

    dist, neighbor_idx = knn(train_activations, model.n_neighbors, progress=True)

    model.ambient_spacing = dist[:, model.n_scale_neighbors - 1]

    model.states = train_activations

    counts = torch.bincount(neighbor_idx[:, 0], minlength=n_states).float()

    model.hub_share = float(counts.sort(descending=True).values[:5].sum() / n_states)

    if float(model.ambient_spacing[neighbor_idx[:, 0]].median()) < 0.5 * float(
        model.ambient_spacing.median()
    ):
        model.spacing_unit = "global"
        print(
            f"[autoencoder] hub-dominated cloud (top-5 anchors are the nearest neighbour of {model.hub_share:.0%} of the points): "
            "distances are reported in units of the global median spacing"
        )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=epochs * math.ceil(n_states / batch_size), pct_start=0.05
    )

    model.train()

    for ep in tqdm(range(epochs), desc="fit"):
        perm = torch.randperm(n_states, device=model.device)
        losses: dict[str, float] = {}
        for s in range(0, n_states, batch_size):
            batch_idx = perm[s : s + batch_size]
            x, neighbors = (
                train_activations_normalised[batch_idx],
                train_activations_normalised[neighbor_idx[batch_idx]],
            )  # (B,D), (B,K,D)
            z, z_neighbors = (
                model.encoder(x),
                model.encoder(neighbors.reshape(-1, ambient_dim)).reshape(
                    len(batch_idx),
                    model.n_neighbors,
                    model.latent_dim,
                ),
            )

            x_recon = model.decoder(z)

            # distances between neighbours and input points in domain and image spaces
            ambient_dist, latent_dist = (
                (neighbors - x[:, None]).norm(dim=2),
                (z_neighbors - z[:, None]).norm(dim=2),
            )

            # Reconstruction loss
            # Simple geometry loss
            terms = {
                "recon": ((x_recon - x) ** 2).sum(1).mean(),
                "geom": ((latent_dist - ambient_dist) ** 2).mean()
                / (ambient_dist**2).mean(),
            }
            # lam_curv is 1e-3
            # curvature minimising loss
            if lam_curv > 0:
                direction = torch.randn_like(z)
                direction = (
                    direction
                    / direction.norm(dim=1, keepdim=True)
                    * (curv_delta * latent_dist.detach().mean(1, keepdim=True))
                )
                second_diff = (
                    model.decoder(z + direction)
                    - 2 * x_recon
                    + model.decoder(z - direction)
                )
                terms["curv"] = ((second_diff**2).sum(1) / (direction**2).sum(1)).mean()

            # total loss
            loss = (
                terms["recon"]
                + lam_geom * terms["geom"]
                + lam_curv * terms.get("curv", 0.0)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            # History storage
            for k, v in terms.items():
                losses[k] = losses.get(k, 0.0) + float(v.detach())
        rec = {k: v / math.ceil(n_states / batch_size) for k, v in losses.items()} | {
            "epoch": ep
        }

        if val_activations is not None:
            model.eval()
            with torch.no_grad():
                val_recon = model.decoder(model.encoder(val_activations_normalised))
                rec["val_recon"] = float(
                    ((val_recon - val_activations_normalised) ** 2).sum(1).mean()
                )
            model.train()

        model.history.append(rec)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()},
            )
    return model


# ------------------------------------------------------------ batched fitting
def train_batch(
    models: list[manifold],
    train_activations: Tensor,
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 2e-3,
    lam_geom: float = 0.1,
    lam_curv: float = 1e-3,
    curv_delta: float = 0.5,
    log_every: int = 50,
    verbose: bool = False,
) -> list[manifold]:
    """Fit len(models) manifolds at once on a GPU, one per row of
    `train_activations` (N, n_states, D) -- same n_states for every model.
    All models must share network_dims/n_neighbors/n_scale_neighbors. Each
    model's parameters are stacked into one batched tensor per layer and
    trained with a single vmapped forward/backward per step, so N small
    models share GPU kernels instead of running N sequential train() loops.
    """
    print("in train batch")
    device = models[0].device
    n_models = len(models)
    train_activations = train_activations.to(device)

    n_neighbors = models[0].n_neighbors
    n_scale_neighbors = models[0].n_scale_neighbors
    latent_dim = models[0].latent_dim
    ambient_dim, n_states = train_activations.shape[2], train_activations.shape[1]

    center = train_activations.mean(1)  # (N, D)
    scale = ((train_activations - center[:, None]) ** 2).sum(2).mean(1).sqrt()  # (N,)

    def normalise(idx: Tensor) -> Tensor:
        """Normalised activations for state indices `idx` (N, ...), computed
        on the fly instead of materialising a second (N, n_states, D) copy of
        `train_activations` up front -- that copy is what blows GPU memory
        budgets at realistic (n_models, n_states, ambient_dim) sizes."""
        pad = (1,) * (idx.dim() - 1)
        model_arange = torch.arange(n_models, device=device).reshape((n_models,) + pad)
        x = train_activations[model_arange, idx]
        return (x - center.reshape((n_models,) + pad + (ambient_dim,))) / scale.reshape(
            (n_models,) + pad + (1,)
        )

    ambient_spacing = torch.empty(n_models, n_states, device=device)
    neighbor_idx = torch.empty(n_models, n_states, n_neighbors, dtype=torch.long, device=device)
   
    for i in range(n_models):
        dist_i, idx_i = knn(train_activations[i], n_neighbors)
        ambient_spacing[i] = dist_i[:, n_scale_neighbors - 1]
        neighbor_idx[i] = idx_i

    encoder_params, encoder_buffers = torch.func.stack_module_state(
        [m.encoder for m in models]
    )
    decoder_params, decoder_buffers = torch.func.stack_module_state(
        [m.decoder for m in models]
    )
    base_encoder = copy.deepcopy(models[0].encoder).to("meta")
    base_decoder = copy.deepcopy(models[0].decoder).to("meta")

    def encode(params, buffers, x):
        return torch.func.functional_call(base_encoder, (params, buffers), (x,))

    def decode(params, buffers, z):
        return torch.func.functional_call(base_decoder, (params, buffers), (z,))

    v_encode = torch.func.vmap(encode)
    v_decode = torch.func.vmap(decode)

    for p in [*encoder_params.values(), *decoder_params.values()]:
        p.requires_grad_(True)

    opt = torch.optim.Adam(
        [*encoder_params.values(), *decoder_params.values()], lr=lr
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=epochs * math.ceil(n_states / batch_size), pct_start=0.05
    )

    history: list[dict[str, Any]] = []
    for ep_idx in tqdm(range(epochs), desc="fit_batch"):
        perm = torch.stack(
            [torch.randperm(n_states, device=device) for _ in range(n_models)]
        )
        losses: dict[str, float] = {}
        n_chunks = 0
        for s in range(0, n_states, batch_size):
            batch_idx = perm[:, s : s + batch_size]  # (N, B)
            b = batch_idx.shape[1]
            model_arange = torch.arange(n_models, device=device)[:, None]
            x = normalise(batch_idx)  # (N, B, D)
            nb_idx = neighbor_idx[model_arange, batch_idx]  # (N, B, K)
            neighbors = normalise(nb_idx)  # (N, B, K, D)

            z = v_encode(encoder_params, encoder_buffers, x)  # (N, B, m)
            z_neighbors = v_encode(
                encoder_params,
                encoder_buffers,
                neighbors.reshape(n_models, b * n_neighbors, ambient_dim),
            ).reshape(n_models, b, n_neighbors, latent_dim)
            x_recon = v_decode(decoder_params, decoder_buffers, z)

            ambient_dist, latent_dist = (
                (neighbors - x[:, :, None]).norm(dim=3),
                (z_neighbors - z[:, :, None]).norm(dim=3),
            )

            terms = {
                "recon": ((x_recon - x) ** 2).sum(2).mean(),
                "geom": ((latent_dist - ambient_dist) ** 2).mean()
                / (ambient_dist**2).mean(),
            }
            if lam_curv > 0:
                direction = torch.randn_like(z)
                direction = (
                    direction
                    / direction.norm(dim=2, keepdim=True)
                    * (curv_delta * latent_dist.detach().mean(2, keepdim=True))
                )
                second_diff = (
                    v_decode(decoder_params, decoder_buffers, z + direction)
                    - 2 * x_recon
                    + v_decode(decoder_params, decoder_buffers, z - direction)
                )
                terms["curv"] = ((second_diff**2).sum(2) / (direction**2).sum(2)).mean()

            loss = (
                terms["recon"]
                + lam_geom * terms["geom"]
                + lam_curv * terms.get("curv", 0.0)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

            n_chunks += 1
            for k, v in terms.items():
                losses[k] = losses.get(k, 0.0) + float(v.detach())

        rec = {k: v / n_chunks for k, v in losses.items()} | {"epoch": ep_idx}
        history.append(rec)
        if verbose and (ep_idx % log_every == 0 or ep_idx == epochs - 1):
            print(
                {k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()},
            )

    with torch.no_grad():
        latent_codes = torch.cat(
            [
                v_encode(
                    encoder_params,
                    encoder_buffers,
                    normalise(
                        torch.arange(s, min(s + batch_size, n_states), device=device)
                        .unsqueeze(0)
                        .expand(n_models, -1)
                    ),
                )
                for s in range(0, n_states, batch_size)
            ],
            dim=1,
        )  # (N, n_states, m)

    for i, model in enumerate(models):
        model.encoder.load_state_dict(
            {k: v[i].detach() for k, v in encoder_params.items()}
            | {k: v[i].detach() for k, v in encoder_buffers.items()}
        )
        model.decoder.load_state_dict(
            {k: v[i].detach() for k, v in decoder_params.items()}
            | {k: v[i].detach() for k, v in decoder_buffers.items()}
        )
        model.center = center[i]
        model.scale = scale[i]
        model.ambient_spacing = ambient_spacing[i]
        model.states = train_activations[i]
        model.latent_codes = latent_codes[i]
        latent_dist_i, _ = knn(model.latent_codes, n_neighbors)
        model.latent_scale = latent_dist_i[:, n_scale_neighbors - 1]
        model.history = history  # shared across the batch: losses are averaged over all n_models
        model.eval()

    return models
