"""Re-attaching a real point cloud to a weights-only-loaded GlobalManifold."""
from __future__ import annotations

import torch

from .geometry import knn
from .manifold import GlobalManifold


@torch.no_grad()
def reattach(model: GlobalManifold, states: torch.Tensor) -> GlobalManifold:
    """Re-attaches a real point cloud to a fitted model whose states aren't
    loaded (e.g. a weights-only checkpoint), recomputing states/latent_codes/
    latent_scale/ambient_spacing/spacing_unit/hub_share exactly as
    gmanifold.manifold.fit does at the end of training (same
    n_neighbors/n_scale_neighbors already fixed on the model) -- states need
    not be the same rows the model was trained on (e.g. a fresh pull of the
    same layer's activations), but should be drawn from the same
    distribution for the local-spacing units and nearest-neighbour
    diagnostics to mean what they claim to."""
    states = states.to(model.device).float()
    model.states = states
    dist, idx = knn(states, model.n_neighbors)
    model.ambient_spacing = dist[:, model.n_scale_neighbors - 1]
    counts = torch.bincount(idx[:, 0], minlength=len(states)).float()
    model.hub_share = float(counts.sort(descending=True).values[:5].sum() / len(states))
    model.spacing_unit = (
        "global"
        if float(model.ambient_spacing[idx[:, 0]].median()) < 0.5 * float(model.ambient_spacing.median())
        else "anchor"
    )
    model.latent_codes = model.encode(states)
    model.latent_scale = knn(model.latent_codes, model.n_scale_neighbors)[0][:, model.n_scale_neighbors - 1]
    return model
