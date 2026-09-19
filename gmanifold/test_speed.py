"""Quick timing check: train manifold autoencoders on random data at the
real production shape -- one manifold per (position, sample), fit via
train_batch(). The full (n_models, n_states, ambient_dim) batch doesn't fit
on one GPU at this scale, so models are trained in GPU-sized chunks, calling
train_batch() once per chunk.

Run with: uv run python -m autoencoder.test_speed
"""

import time

import torch

from autoencoder import manifold, train_batch

torch.manual_seed(0)

n_positions = 5
n_samples = 64
n_models = n_positions * n_samples  # one manifold per (position, sample)
n_states = 50_000  # vocab_sample
ambient_dim = 2000  # d_model
network_dims = [ambient_dim, 256, 32]
epochs = 100
batch_size = 512
models_per_chunk = 8  # how many manifolds train_batch() fits on one GPU at once
device = "cuda" if torch.cuda.is_available() else "cpu"

print(
    f"device={device}  n_models={n_models} ({n_positions} positions x {n_samples} samples)  "
    f"n_states={n_states}  ambient_dim={ambient_dim}  network_dims={network_dims}  "
    f"epochs={epochs}  models_per_chunk={models_per_chunk}"
)

models = [manifold(network_dims=network_dims, device=device) for _ in range(n_models)]

t0 = time.perf_counter()
for c in range(0, n_models, models_per_chunk):
    chunk = models[c : c + models_per_chunk]
    # generated directly on `device`, one chunk of models at a time, instead
    # of materializing all n_models up front -- at full scale that's ~128GB,
    # nowhere near fitting in CPU or GPU memory at once.
    Xb = torch.randn(len(chunk), n_states, ambient_dim, device=device)
    train_batch(chunk, Xb, epochs=epochs, batch_size=batch_size, verbose=False)
    del Xb
    print(f"  chunk {c}:{c + len(chunk)} done, {time.perf_counter() - t0:.1f}s elapsed", flush=True)
t1 = time.perf_counter()

print(
    f"\n[train_batch, chunked] {t1 - t0:.2f}s total for {n_models} models "
    f"({models_per_chunk}/chunk), {(t1 - t0) / n_models:.2f}s/model, "
    f"final recon={models[0].history[-1]['recon']:.4f}"
)
