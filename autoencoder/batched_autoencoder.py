import copy
import math
from typing import Any

import torch
from torch import Tensor, nn
from tqdm.auto import tqdm


@torch.no_grad()
def knn(x, k, query_set=None, chunk=4096, model_chunk=None, device=None, progress=False):
    """Batched k-nearest-neighbours. x, query_set: CPU or GPU tensors, shape
    (n_models, N, D) / (n_models, Q, D). Each (model_chunk, chunk) slice is
    moved to `device` only for the duration of its cdist/topk call, if given
    -- pass device=None to compute in place on x's own device."""
    self_q = query_set is None
    query_set = x if self_q else query_set
    data_size = x.shape[1]

    kk = min(k + int(self_q), data_size)
    batch, q_size = query_set.shape[0], query_set.shape[1]
    model_chunk = model_chunk or batch
    out_device = device or x.device
    ds = torch.empty(batch, q_size, k, dtype=x.dtype, device=out_device)
    js = torch.empty(batch, q_size, k, dtype=torch.long, device=out_device)
    model_steps = range(0, batch, model_chunk)
    if progress:
        model_steps = tqdm(model_steps)
    for m in model_steps:
        x_m = x[m:m + model_chunk].to(device) if device else x[m:m + model_chunk]
        for s in range(0, q_size, chunk):
            q = query_set[m:m + model_chunk, s:s + chunk]
            q = q.to(device) if device else q
            d = torch.cdist(q, x_m)
            if self_q:
                b = torch.arange(d.shape[0], device=d.device)[:, None]
                r = torch.arange(d.shape[1], device=d.device)[None, :]
                d[b, r, s + r] = float("inf")
            dd, jj = d.topk(kk, largest=False, dim=-1)
            ds[m:m + model_chunk, s:s + chunk] = (dd[..., :k] if self_q else dd).to(out_device)
            js[m:m + model_chunk, s:s + chunk] = (jj[..., :k] if self_q else jj).to(out_device)
    return ds, js


@torch.no_grad()
def knn_single(x, k, query_set=None, chunk=4096):
    """k nearest neighbours of query_set (default x itself, self excluded) in
    x -> (dist, idx), each (n, k). Non-batched (x, query_set: (N, D))."""
    self_q = query_set is None
    query_set = x if self_q else query_set
    kk = min(k + int(self_q), len(x))
    ds, js = [], []
    for s in range(0, len(query_set), chunk):
        d = torch.cdist(query_set[s:s + chunk], x)
        if self_q:
            d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = float("inf")
        dd, jj = d.topk(kk, largest=False)
        ds.append(dd[:, :k] if self_q else dd)
        js.append(jj[:, :k] if self_q else jj)
    return torch.cat(ds), torch.cat(js)


def uniform_ball(n, m, radius, generator=None, device=None):
    """Uniform points in m-balls of the given radii (n,): direction ~ normalised Gaussian, r = R s^{1/m}."""
    g = torch.randn(n, m, device=device, generator=generator)
    g = g / g.norm(dim=1, keepdim=True)
    return g * (radius.to(device) * torch.rand(n, device=device, generator=generator) ** (1.0 / m))[:, None]


class BatchedMLP_no_bias(nn.Module):
    def __init__(self, n_models: int, network_dims: list[int], activation: type[nn.Module] = nn.SiLU) -> None:
        super().__init__()
        self.n_models = n_models
        self.network_dims = network_dims
        self.n_layers = len(network_dims) - 1
        self.activation = activation()

        weights = []
        for i in range(self.n_layers):
            fan_in, fan_out = network_dims[i], network_dims[i + 1]
            w = torch.empty(n_models, fan_in, fan_out)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(w, -bound, bound)
            weights.append(w)
            print(w.shape)
        self.weights = nn.ParameterList(nn.Parameter(w) for w in weights)
        # print(self.weights)
        # self.weights shape = [n_layers, n_models, network_dims[i]]

    def forward(self, x: Tensor) -> Tensor:
        """x: (n_models, B, in) -> (n_models, B, out)."""
        for i in range(self.n_layers):
            x = torch.bmm(x, self.weights[i])
            if i != self.n_layers - 1:
                x = self.activation(x)
        return x

    @torch.no_grad()
    def load_from_sequential(self, index: int, seq: nn.Sequential) -> None:
        """Copies one model's-worth of weights (a plain nn.Sequential built
        by manifold.block) into row `index` of this batch -- used to seed a
        BatchedMLP from manifold instances constructed the usual way."""
        linears = [m for m in seq if isinstance(m, nn.Linear)]
        assert len(linears) == self.n_layers
        for i, lin in enumerate(linears):
            self.weights[i][index] = lin.weight.T

    @torch.no_grad()
    def state_dict_for(self, index: int) -> dict[str, Tensor]:
        """The nn.Linear-shaped state_dict for model `index`, e.g. to load
        back into a plain manifold's encoder/decoder (block()'s Sequential)
        after training."""
        sd = {}
        for i in range(self.n_layers):
            sd[f"{2 * i}.weight"] = self.weights[i][index].detach().T.contiguous()
            # sd[f"{2 * i}.bias"] = self.biases[i][index, 0].detach().contiguous()
        return sd

    @torch.no_grad()
    def load_state_dict_for(self, index: int, state_dict: dict[str, Tensor]) -> None:
        """Inverse of state_dict_for: writes an nn.Linear-shaped state_dict
        (keys "0.weight", "2.weight", ...) into row `index` of this batch."""
        for i in range(self.n_layers):
            self.weights[i][index] = state_dict[f"{2 * i}.weight"].T


class batched_autoencoder(nn.Module):
    def __init__(
        self,
        center : torch.Tensor,
        scale : torch.Tensor,
        n_models: int = 6,
        network_dims: list[int] = [512, 256],
        n_neighbors: int = 32,
        n_scale_neighbors: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.n_scale_neighbors = n_scale_neighbors
        self.network_dims = network_dims
        self.latent_dim = network_dims[-1]
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.encoder = BatchedMLP_no_bias(n_models, network_dims)
        self.decoder = BatchedMLP_no_bias(n_models, list(reversed(network_dims)))

        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

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
        self.states: Tensor | None = None  # (n_models, n_train_rows, D) subset states was fit on -- not a
        # parameter/buffer, so it never lands in state_dict(); re-derivable from seed/n_train_rows via reattach()
        self.seed: int | None = None
        self.n_train_rows: int | None = None
        self.model_index_offset: int = 0  # global index of this model's row 0 -- see reattach(); nonzero when
        # this batch is one chunk of a larger chunked training run (scripts.autoencoder.train_batched_manifold's
        # --n-models-per-chunk), or when load_batched reassembled several chunks' checkpoints into one batch
        self.history: list[dict[str, Any]] = []
        self.spacing_unit = "anchor"  # 'anchor': divide by the nearest anchor's spacing; 'global': by the median
        self.hub_share = 0.0  # share of points whose nearest neighbour is one of the 5 most common anchors

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder((x.to(self.device) - self.center) / self.scale)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z.to(self.device)) * self.scale + self.center

    def project(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def chunked_apply(self, fn, x: Tensor, chunk: int) -> Tensor:
        """fn(x[:, s:s+chunk]) concatenated along the point dim -- use this
        instead of calling encode/decode/project directly on a full
        (n_models, N, D)-shaped pool. Every one of those ops runs the whole
        point dim through the batched MLP (and, for project, its intermediate
        encoder hidden activation) in one shot, which OOMs once N is a full
        states/val pool rather than a training-loop-sized minibatch (see
        train_batched_manifold.py's chunked_recon_loss, the same fix applied
        there for exactly this reason)."""
        return torch.cat([fn(x[:, s:s + chunk]) for s in range(0, x.shape[1], chunk)], dim=1)

    def reattach(self, full_activations: Tensor) -> None:
        """Re-derives self.states -- a per-row seeded torch.randperm(row_seed)
        [:n_train_rows] subset, row_seed = self.seed + model_index_offset + row
        -- from a freshly supplied (n_models, N, D) pool of the same activations
        used at fit time. Each row's split is entirely self-contained (its own
        Generator, not a shared stream burned through in row order), so this is
        correct regardless of how training was chunked: reattaching a batch
        reassembled (e.g. via load_batched) from several separately-trained
        chunks still reproduces each row's own original split. Requires
        seed/n_train_rows to already be set (set directly after training, or
        via `load`/`load_batched`)."""
        assert self.seed is not None and self.n_train_rows is not None
        full_activations = full_activations.to(self.device)
        n_models = full_activations.shape[0]
        perms = torch.stack([
            torch.randperm(
                full_activations.shape[1],
                generator=torch.Generator(device=full_activations.device).manual_seed(
                    self.seed + self.model_index_offset + row,
                ),
                device=full_activations.device,
            )
            for row in range(n_models)
        ])
        idx = perms[:, : self.n_train_rows]
        self.states = torch.gather(
            full_activations, 1, idx[..., None].expand(-1, -1, full_activations.shape[-1])
        )

    @staticmethod
    def _decode_one_point(weights: list[Tensor], z_point: Tensor, activation: nn.Module) -> Tensor:
        x = z_point
        n_layers = len(weights)
        for i, w in enumerate(weights):
            x = x @ w
            if i != n_layers - 1:
                x = activation(x)
        return x

    def jacobian(self, z: Tensor, chunk: int = 1024, model_chunk: int | None = None) -> Tensor:
        """J_G(z) in ambient units, per model and per point: (n_models, B, D, m).
        BatchedMLP_no_bias.forward always needs the full (n_models, B, in) batch
        (it's not a per-model function), so jacfwd can't differentiate it
        directly -- instead this builds a plain per-model, per-point decode
        function over the raw weight list and nests vmap over (model, point).

        `chunk` bounds the point dim; `model_chunk` (default: all models at
        once) additionally bounds the model dim -- forward-mode AD tracks
        `m` (latent_dim) tangent directions through every decoder activation
        for every model *and* every point in a chunk simultaneously, so
        leaving model_chunk unset can OOM at a large n_models even with a
        small point chunk (e.g. n_models=104, chunk=1024 tried to allocate
        208GiB here)."""
        assert self.decoder is not None
        weights = list(self.decoder.weights)  # each (n_models, fan_in, fan_out)
        n_models = z.shape[0]
        model_chunk = model_chunk or n_models
        f = torch.func.vmap(
            torch.func.vmap(
                torch.func.jacfwd(self._decode_one_point, argnums=1), in_dims=(None, 0, None),
            ),
            in_dims=(0, 0, None),
        )
        d_model = self.decoder.network_dims[-1]
        out = torch.empty(n_models, z.shape[1], d_model, self.latent_dim, device=z.device)
        for m in range(0, n_models, model_chunk):
            m_weights = [w[m:m + model_chunk] for w in weights]
            for s in range(0, z.shape[1], chunk):
                out[m:m + model_chunk, s:s + chunk] = f(
                    m_weights, z[m:m + model_chunk, s:s + chunk], self.decoder.activation,
                )
        return out * self.scale[..., None]

    @torch.no_grad()
    def fit_stats(self, knn_chunk: int = 512, knn_model_chunk: int | None = None) -> None:
        """Populates latent_codes/latent_scale/ambient_spacing/hub_share/
        spacing_unit from self.states, exactly as the training script does at
        the end of fitting -- call after training or after reattach().

        `knn_model_chunk` (default: all models at once) bounds every knn/
        encode call below along the model dim too, not just the point dim
        (`knn_chunk`) -- both self.states and a full checkpoint's n_models
        can be large together (e.g. GPT-2: 100 models x ~28800 train rows),
        and cdist/the batched MLP scale with the product of the two (this is
        what was OOMing: "Tried to allocate 12.94 GiB" for GPT-2's full
        self.states, then 69GiB in sampling.batched_support_logq's cdist --
        same root cause, model dim left unchunked)."""
        assert self.states is not None
        n_models, n_train_rows = self.states.shape[0], self.states.shape[1]

        dist, neighbor_idx = knn(
            self.states, self.n_neighbors, chunk=knn_chunk, model_chunk=knn_model_chunk, device=self.device,
        )
        self.ambient_spacing = dist[:, :, self.n_scale_neighbors - 1]

        offset = torch.arange(n_models, device=dist.device)[:, None] * n_train_rows
        counts = torch.bincount(
            (neighbor_idx[:, :, 0] + offset).flatten(), minlength=n_models * n_train_rows,
        ).reshape(n_models, n_train_rows)
        self.hub_share = (counts.sort(dim=-1, descending=True).values[:, :5].sum(dim=-1) / n_train_rows)

        if float(self.ambient_spacing[torch.arange(n_models)[:, None], neighbor_idx[:, :, 0]].median()) < 0.5 * float(
            self.ambient_spacing.median()
        ):
            self.spacing_unit = "global"

        self.latent_codes = self.chunked_apply(self.encode, self.states, knn_chunk)
        lat_dist, _ = knn(
            self.latent_codes, self.n_scale_neighbors, chunk=knn_chunk, model_chunk=knn_model_chunk, device=self.device,
        )
        self.latent_scale = lat_dist[:, :, self.n_scale_neighbors - 1]

    def unit(self, j: Tensor) -> Tensor:
        """Length unit per model row for points whose nearest fitted state is
        j (n_models, n): that state's spacing, or the global median."""
        anchor = torch.gather(self.ambient_spacing, 1, j)
        glob = self.ambient_spacing.median(dim=1, keepdim=True).values.expand_as(j)
        return anchor if self.spacing_unit == "anchor" else glob

    @torch.no_grad()
    def recon_error(self, Y: Tensor, chunk: int = 512, model_chunk: int | None = None) -> Tensor:
        """||G(E(y)) - y|| in spacing units, per model row: (n_models, n).

        `chunk`/`model_chunk`: bound the point/model dims of both the knn
        query against self.states and the project() call over Y -- Y may be
        a full states/val pool, same OOM risk as fit_stats (see its
        docstring)."""
        assert self.states is not None
        Y = Y.to(self.device).float()
        j = knn(self.states, 1, Y, chunk=chunk, model_chunk=model_chunk, device=self.device)[1][..., 0]
        return (self.chunked_apply(self.project, Y, chunk) - Y).norm(dim=-1) / self.unit(j)

    @torch.no_grad()
    def nearest_real(self, Y: Tensor, chunk: int = 512, model_chunk: int | None = None) -> Tensor:
        """Distance from Y to the nearest fitted real state, in spacing units, per model row."""
        assert self.states is not None
        d, j = knn(self.states, 1, Y.to(self.device).float(), chunk=chunk, model_chunk=model_chunk, device=self.device)
        return d[..., 0] / self.unit(j[..., 0])

    @torch.no_grad()
    def coverage(self, Y: Tensor, chunk: int = 512, model_chunk: int | None = None) -> Tensor:
        """Fraction of fitted real states (per model row) with a sample within one spacing unit."""
        assert self.states is not None
        n_train_rows = self.states.shape[1]
        d = knn(Y.to(self.device).float(), 1, self.states, chunk=chunk, model_chunk=model_chunk, device=self.device)[0][..., 0]
        idx = torch.arange(n_train_rows, device=self.device)[None, :].expand(self.states.shape[0], -1)
        return (d / self.unit(idx) <= 1).float().mean(dim=-1)


def _block(network_dims: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers = []
    for i in range(len(network_dims) - 1):
        layers.append(nn.Linear(network_dims[i], network_dims[i + 1], bias=False))
        if i != len(network_dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class autoencoder(nn.Module):
    """Single, non-batched autoencoder -- one row of a batched_autoencoder,
    same architecture (Linear, no bias, SiLU between layers) built as a
    plain nn.Sequential instead of BatchedMLP_no_bias."""

    def __init__(
        self,
        center: Tensor,
        scale: Tensor,
        network_dims: list[int],
        n_neighbors: int = 32,
        n_scale_neighbors: int = 8,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.network_dims = network_dims
        self.latent_dim = network_dims[-1]
        self.n_neighbors = n_neighbors
        self.n_scale_neighbors = n_scale_neighbors
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.encoder = _block(network_dims).to(self.device)
        self.decoder = _block(list(reversed(network_dims))).to(self.device)

        self.register_buffer("center", center)
        self.register_buffer("scale", scale)

        self.register_buffer("latent_codes", torch.zeros(0, self.latent_dim))
        self.register_buffer("latent_scale", torch.zeros(0))
        self.register_buffer("ambient_spacing", torch.zeros(0))
        self.spacing_unit = "anchor"
        self.hub_share = 0.0

        self.states: Tensor | None = None  # (n_train_rows, D) subset states was fit on -- not a
        # parameter/buffer, so it never lands in state_dict(); re-derivable from seed/n_train_rows via reattach()
        self.seed: int | None = None
        self.n_train_rows: int | None = None
        self.model_index: int | None = None  # this model's GLOBAL row index in the batched_autoencoder run it
        # came from (== its checkpoint filename number) -- reattach uses seed + model_index as this row's own
        # independent split seed, so it's correct regardless of chunking (see batched_autoencoder.reattach)

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

    def reattach(self, full_activations: Tensor) -> None:
        """Re-derives self.states -- the seeded torch.randperm(seed + model_index)
        [:n_train_rows] subset -- from a freshly supplied (N, D) pool of the same
        activations used at fit time. Requires seed/n_train_rows/model_index to
        already be set (set directly after training, or via `load`). Each row's
        split is its own independent Generator (see batched_autoencoder.reattach),
        so this is correct regardless of how training was chunked."""
        assert self.seed is not None and self.n_train_rows is not None and self.model_index is not None
        full_activations = full_activations.to(self.device)
        gen = torch.Generator(device=full_activations.device).manual_seed(self.seed + self.model_index)
        perm = torch.randperm(full_activations.shape[0], generator=gen, device=full_activations.device)
        self.states = full_activations[perm[: self.n_train_rows]]

    @torch.no_grad()
    def fit_stats(self) -> None:
        """Populates latent_codes/latent_scale/ambient_spacing/hub_share/
        spacing_unit from self.states, exactly as the training script does at
        the end of fitting -- call after training or after reattach()."""
        assert self.states is not None
        n_train_rows = self.states.shape[0]

        dist, neighbor_idx = knn_single(self.states, self.n_neighbors)
        self.ambient_spacing = dist[:, self.n_scale_neighbors - 1]

        counts = torch.bincount(neighbor_idx[:, 0], minlength=n_train_rows)
        self.hub_share = float(counts.sort(descending=True).values[:5].sum() / n_train_rows)

        if float(self.ambient_spacing[neighbor_idx[:, 0]].median()) < 0.5 * float(
            self.ambient_spacing.median()
        ):
            self.spacing_unit = "global"

        self.latent_codes = self.encode(self.states)
        self.latent_scale = knn_single(self.latent_codes, self.n_scale_neighbors)[0][:, self.n_scale_neighbors - 1]

    def unit(self, j: Tensor) -> Tensor:
        """Length unit for points whose nearest fitted state is j: that state's spacing, or the global median."""
        return (
            self.ambient_spacing[j]
            if self.spacing_unit == "anchor"
            else self.ambient_spacing.median().expand(len(j))
        )

    @torch.no_grad()
    def recon_error(self, Y: Tensor) -> Tensor:
        """||G(E(y)) - y|| in spacing units (the off-manifold distance to the ansatz)."""
        assert self.states is not None
        Y = Y.to(self.device).float()
        j = knn_single(self.states, 1, Y)[1][:, 0]
        return (self.project(Y) - Y).norm(dim=1) / self.unit(j)

    @torch.no_grad()
    def nearest_real(self, Y: Tensor) -> Tensor:
        """Distance from Y to the nearest fitted real state, in spacing units."""
        assert self.states is not None
        d, j = knn_single(self.states, 1, Y.to(self.device).float())
        return d[:, 0] / self.unit(j[:, 0])

    @torch.no_grad()
    def coverage(self, Y: Tensor) -> float:
        """Fraction of fitted real states that have a sample within one spacing unit."""
        assert self.states is not None
        d = knn_single(Y.to(self.device).float(), 1, self.states)[0][:, 0]
        return float(
            (d / self.unit(torch.arange(len(self.states), device=self.device)) <= 1).float().mean()
        )


def save_all(model: batched_autoencoder, out_dir: str, index_offset: int | None = None) -> None:
    """Splits a trained batched_autoencoder into model.encoder.n_models
    individual checkpoints, written to out_dir/model_{global_index}.pt --
    each loadable via `load` into a plain autoencoder. Does not save
    model.states itself -- only seed/global_index, so the caller must
    `reattach` a freshly-supplied copy of that model's full activation pool
    after loading to get `states` back (see autoencoder.reattach).

    `index_offset`: defaults to model.model_index_offset (0 unless this batch
    is one chunk of a larger chunked training run -- see
    scripts.autoencoder.train_batched_manifold's --n-models-per-chunk). Each
    row's saved `model_index` is index_offset + i: its GLOBAL position, used
    directly as part of that row's own independent split seed
    (seed + model_index) by autoencoder.reattach -- so it must match whatever
    global slot this checkpoint actually occupies, which is also why it's the
    same number as the output filename."""
    import os

    assert model.seed is not None and model.n_train_rows is not None, (
        "model.seed/n_train_rows must be set (e.g. right after training) before save_all"
    )
    index_offset = model.model_index_offset if index_offset is None else index_offset
    os.makedirs(out_dir, exist_ok=True)
    for i in range(model.encoder.n_models):
        global_index = index_offset + i
        torch.save(
            dict(
                network_dims=model.network_dims,
                n_neighbors=model.n_neighbors,
                n_scale_neighbors=model.n_scale_neighbors,
                encoder_state_dict=model.encoder.state_dict_for(i),
                decoder_state_dict=model.decoder.state_dict_for(i),
                center=model.center[i].detach().cpu(),
                scale=model.scale[i].detach().cpu(),
                seed=model.seed,
                n_train_rows=model.n_train_rows,
                model_index=global_index,
            ),
            os.path.join(out_dir, f"model_{global_index}.pt"),
        )


def load(path: str, device: torch.device | str | None = None) -> autoencoder:
    ckpt = torch.load(path, map_location="cpu")
    model = autoencoder(
        ckpt["center"], ckpt["scale"], network_dims=ckpt["network_dims"],
        n_neighbors=ckpt.get("n_neighbors", 32), n_scale_neighbors=ckpt.get("n_scale_neighbors", 8),
        device=device,
    )
    model.encoder.load_state_dict(ckpt["encoder_state_dict"])
    model.decoder.load_state_dict(ckpt["decoder_state_dict"])
    model.to(model.device)
    model.seed = ckpt["seed"]
    model.n_train_rows = ckpt["n_train_rows"]
    model.model_index = ckpt["model_index"]
    return model.eval()


def load_batched(out_dir: str, device: torch.device | str | None = None) -> batched_autoencoder:
    """Inverse of save_all: reassembles a batched_autoencoder from every
    model_{i}.pt checkpoint in `out_dir`. All checkpoints must share
    network_dims/n_neighbors/n_scale_neighbors/seed/n_train_rows (true of
    any dir written by one save_all call) and cover model indices 0..n-1
    contiguously. Does not restore states -- call reattach() on the result
    with a freshly-supplied (n_models, N, D) pool, as with a single
    autoencoder's load()."""
    import glob
    import os
    import re

    paths = glob.glob(os.path.join(out_dir, "model_*.pt"))
    by_index = {}
    for p in paths:
        m = re.match(r"model_(\d+)\.pt$", os.path.basename(p))
        if m:
            by_index[int(m.group(1))] = p
    n_models = len(by_index)
    assert n_models > 0, f"no model_{{i}}.pt checkpoints found in {out_dir}"
    assert set(by_index) == set(range(n_models)), (
        f"expected contiguous model indices 0..{n_models - 1} in {out_dir}, got {sorted(by_index)}"
    )

    ckpts = [torch.load(by_index[i], map_location="cpu") for i in range(n_models)]
    first = ckpts[0]
    for i, ckpt in enumerate(ckpts):
        assert ckpt["network_dims"] == first["network_dims"]
        assert ckpt["seed"] == first["seed"]
        assert ckpt["n_train_rows"] == first["n_train_rows"]
        assert ckpt.get("model_index", i) == i, (
            f"model_{i}.pt has model_index={ckpt.get('model_index')}, expected {i} -- "
            "its own global split seed (seed + model_index) wouldn't match this row's position"
        )

    center = torch.stack([c["center"] for c in ckpts])
    scale = torch.stack([c["scale"] for c in ckpts])

    model = batched_autoencoder(
        center, scale, n_models, network_dims=first["network_dims"],
        n_neighbors=first.get("n_neighbors", 32), n_scale_neighbors=first.get("n_scale_neighbors", 8),
        device=device,
    )
    for i, ckpt in enumerate(ckpts):
        model.encoder.load_state_dict_for(i, ckpt["encoder_state_dict"])
        model.decoder.load_state_dict_for(i, ckpt["decoder_state_dict"])
    model.to(model.device)
    model.seed = first["seed"]
    model.model_index_offset = 0  # row i's global index is i, by the model_index assertion above
    model.n_train_rows = first["n_train_rows"]
    return model.eval()
