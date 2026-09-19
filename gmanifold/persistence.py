"""Save/load a fitted manifold's weights without re-saving its (large) state
point-cloud. `save` stores only the seed and row count used to select
`model.states` out of the full activation pool that was available at fit
time; `load` re-derives the identical subset from a freshly-supplied copy of
that pool via the same torch.randperm(seed)[:n_train_rows] slice, then
recomputes latent_codes/latent_scale/ambient_spacing/spacing_unit/hub_share
from it exactly as train()/train_batch() do -- same idea as
gmanifold.persistence.reattach, but deriving the row selection deterministically
instead of requiring the caller to already have the exact same subset in hand.
"""

from typing import Any

import torch
from torch import Tensor

from .autoencoder import knn, manifold


def save(model: manifold, path: str, seed: int, n_train_rows: int) -> None:
    """`seed`/`n_train_rows` must be the same values used to select
    `model.states` out of the full activation pool (e.g. via
    `torch.randperm(len(full_activations), generator=Generator().manual_seed(seed))[:n_train_rows]`),
    so `load` can reselect the identical rows later instead of storing them."""
    assert model.states is not None
    assert len(model.states) == n_train_rows, (
        f"n_train_rows={n_train_rows} doesn't match model.states "
        f"({len(model.states)} rows) -- pass the seed/cutoff actually used to "
        "select model.states, not an arbitrary value."
    )
    torch.save(
        dict(
            network_dims=[
                model.encoder[0].in_features,
                *(
                    layer.out_features
                    for layer in model.encoder
                    if isinstance(layer, torch.nn.Linear)
                ),
            ],
            n_neighbors=model.n_neighbors,
            n_scale_neighbors=model.n_scale_neighbors,
            state_dict=model.state_dict(),
            history=model.history,
            spacing_unit=model.spacing_unit,
            hub_share=model.hub_share,
            seed=seed,
            n_train_rows=n_train_rows,
        ),
        path,
    )


@torch.no_grad()
def load(
    path: str, full_activations: Tensor, device: torch.device | str | None = None
) -> manifold:
    """`full_activations` must be the same pool of rows (same order) that
    `save`'s caller selected `model.states` from -- e.g. a fresh pull of the
    same layer's activations. Re-derives states/latent_codes/latent_scale/
    ambient_spacing exactly as train()/train_batch() do at the end of
    fitting."""
    ckpt: dict[str, Any] = torch.load(path, map_location="cpu")

    model = manifold(
        network_dims=ckpt["network_dims"],
        n_neighbors=ckpt["n_neighbors"],
        n_scale_neighbors=ckpt["n_scale_neighbors"],
        device=device,
    )
    # buffers below are (0,)-shaped until fit; resize to the checkpoint's
    # shapes first so load_state_dict's shape check passes
    for k in ("center", "latent_codes", "latent_scale", "ambient_spacing"):
        setattr(model, k, torch.zeros_like(ckpt["state_dict"][k]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(model.device)
    model.history = ckpt["history"]
    model.spacing_unit = ckpt["spacing_unit"]
    model.hub_share = ckpt["hub_share"]

    full_activations = full_activations.to(model.device)
    perm = torch.randperm(
        len(full_activations), generator=torch.Generator().manual_seed(ckpt["seed"])
    )
    model.states = full_activations[perm[: ckpt["n_train_rows"]]]

    dist, _ = knn(model.states, model.n_neighbors)
    model.ambient_spacing = dist[:, model.n_scale_neighbors - 1]
    model.latent_codes = model.encode(model.states)
    model.latent_scale = knn(model.latent_codes, model.n_scale_neighbors)[0][
        :, model.n_scale_neighbors - 1
    ]

    return model.eval()
