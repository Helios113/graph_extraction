"""Run several independent GlobalManifold.fit() calls concurrently on one GPU, each on its own
CUDA stream, so the GPU scheduler can overlap kernels from different fits instead of running them
one after another on the default stream. GlobalManifold/fit are used completely unmodified: every
model produced is a normal, independently save()/load()-able GlobalManifold, exactly as if fit() had
been called on it directly and sequentially -- this only changes when the kernels are scheduled, not
what each model is or how it's persisted."""
from __future__ import annotations

import threading
from typing import Any

import torch
from torch import Tensor

from .manifold import GlobalManifold, fit


def fit_concurrent(
    models: list[GlobalManifold],
    Xs: list[Tensor],
    max_workers: int | None = None,
    **fit_kwargs: Any,
) -> list[GlobalManifold]:
    """Fits `models[i]` on `Xs[i]` for every i, concurrently. Each model/data pair is otherwise
    exactly a normal `fit(models[i], Xs[i], **fit_kwargs)` call -- same architecture-per-model
    freedom (they need not share latent_dim/hidden), same independent seeding if `seed` is passed
    via fit_kwargs (every model gets the identical seed unless you pass per-model seeds by calling
    this in groups). `max_workers` caps how many fits run at once (default: len(models), i.e. all
    of them); use it to keep memory bounded when fitting hundreds of small models. Returns the same
    `models` list, each fitted in place (matching fit()'s in-place-and-returns-it contract).

    Concurrency comes from giving each fit its own torch.cuda.Stream and running the fits on
    separate Python threads -- CUDA kernel launches release the GIL, so the GPU can genuinely
    interleave kernels from different streams (real overlap when any single model's kernels don't
    saturate the GPU's SMs, which is the common case for these small MLPs). This is weaker than a
    single fused batched op (see git history for the abandoned BatchedGlobalManifold approach) but
    keeps every model 100% normal -- no un-batching step needed before saving."""
    assert len(models) == len(Xs), f"{len(models)} models but {len(Xs)} data tensors"
    max_workers = max_workers or len(models)

    device = models[0].device
    streams = [torch.cuda.Stream(device=device) for _ in models] if device.type == "cuda" else [None] * len(models)
    sem = threading.Semaphore(max_workers)
    errors: list[BaseException | None] = [None] * len(models)

    def run_one(i: int) -> None:
        with sem:
            try:
                if streams[i] is not None:
                    with torch.cuda.stream(streams[i]):
                        fit(models[i], Xs[i], **fit_kwargs)
                    streams[i].synchronize()
                else:
                    fit(models[i], Xs[i], **fit_kwargs)
            except BaseException as e:  # noqa: BLE001 -- surfaced to the caller after join()
                errors[i] = e

    threads = [threading.Thread(target=run_one, args=(i,)) for i in range(len(models))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, e in enumerate(errors):
        if e is not None:
            raise RuntimeError(f"fit_concurrent: model {i} failed") from e
    return models
