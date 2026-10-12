# Codebase overview

Research code for studying the **Jacobians of transformer sublayers** (GPT-2, TinyLlama,
SimpleStories) via ONNX graphs: where the Jacobian determinant crosses zero, how the
layer map folds, and how Lipschitz-ratio tails (EVT / generalized Pareto fits) behave
near those singularities. A learned-manifold sampler (autoencoders) generates
on-manifold activations to probe this. Everything runs through `uv run` (see `CLAUDE.md`).

## Top-level directories

| Dir | What it does |
|---|---|
| `construct/` | Builds the ONNX artifacts. Exports the base model (`model_export.py`, plus a KV-cache variant in `kv_cache_export.py` / `kv_cache_subgraph.py`), extracts per-sublayer subgraphs (`subgraph_extract.py`), and attaches loss/gradient/pullback blocks via onnxruntime-training (`loss.py`, `gradient_block.py`, `pullback.py`, `generate_artifact.py`). `construct.py` is the entry point that ensures all of it exists for a config. |
| `execute/` | Runs the ONNX graphs and moves data. `jacobian.py` / `find_jacobians.py` compute Jacobians with one-hot masked-sum pullbacks; `optimization_loop.py` / `run_optimization.py` hold optimizers (SGD, AdamW, Newton-Raphson) over a graph; `io_onnx.py` handles ORT I/O. Data transport: `sharded_writer.py` (crash-resilient sharded h5 writer) pushes through `node_relay.py` (NFS staging, the default). `remote_storage.py` is the old Hetzner/SFTP path, **banned for new work**. |
| `autoencoder/` | Batched PyTorch autoencoders (`batched_autoencoder.py`: many models trained in parallel on `(n_models, n_points, D)` tensors, with chunked `knn` / `chunked_apply`), volume-uniform manifold sampling (`sampling.py`), and loaders for sweep point clouds (`data.py`). Remember to chunk over both the point and model dims. |
| `gmanifold/` | Single-manifold library the batched code descends from: `GlobalManifold` encoder/decoder, tangent charts, geometry (kNN, intrinsic dimension, outliers), sampling, validation, persistence, and an ONNX transformer front-end. Its own `README.md` explains the "volume-uniform" sampling idea. |
| `scripts/python/` | Main pipeline stages, one CLI each: base activations/sequences, perturbed / ambient / tangent sweeps, tangent bases, base-Jacobian stats, kernel-tangent overlap, perturbed Jacobian det, det-zero manifold/sphere/interpolation generators, autoencoder train / sample / validate / hparam search, and layer-batched drivers (`run_layer_batched_*`). |
| `scripts/batch_scripts/` | SLURM `sbatch` wrappers for the stages above (mauao / ruapehu variants), with `gpt2/`, `simple_stories/` and `tiny_llama/` subdirs for per-model launches, and the scratch↔NFS copy scripts. |
| `scripts/analysis/` | Post-hoc analysis and paper plots: GPD / EQD fits and bootstraps of Lipschitz ratios, `plot_det_zero_*` figures, kernel-tangent overlap plots, toy examples (`toy_singular_identity.py`, `toy_kl_x_squared.py`). |
| `scripts/lipschitz/` | GPU-batched generalized-Pareto MLE and the bootstrap EQD sweep drivers (Python and sbatch). |
| `scripts/helper/` | Ops utilities: node-scratch sync (`sync_node_scratch.sh`), corrupt-shard cleanup, manifest reconstruction, shard reclaiming, GPU / node availability checks, max-batch-size finders, remote-storage stress test, and a KV-cache sweep variant. |
| `configs/` | TOML model configs (`gpt2`, `tiny_llama`, `simple_stories`) listing the sublayer `input → output` pairs, plus `toy_singular_identity/*.yaml` parameter sets for the toy singular-map experiments. |
| `generate_data/` | `datagen.py`: ambient point sampling helpers. |
| `lapac/` | Standalone numerics checks: LAPACK LU sign/determinant behaviour and torch `slogdet` float32 sign stress tests (is the det sign trustworthy near zero?). |
| `analyze/` | Small h5 inspection (`plot_h5.py`) and a scratch notebook. |
| `figures/` | Paper figures (`paper/` PDFs/PNGs plus `list_of_figures.txt`), standalone kernel/tangent PDFs, helper notebook. |
| `data/` | Local sweep outputs and their manifest/progress JSON, per model (e.g. `data/tiny_llama/*.h5`). Generated, not source. |
| `onnx_models/` | Generated ONNX artifacts and sweep outputs per model. Only some `simple_stories/det_zero_manifold` PDFs are tracked. Large data goes to node-local scratch (`$NEW_JAC_SCRATCH`). |
| `examples/` | `gmanifold/simple_stories.py`: example use of the `gmanifold` library. |
| `papers/` | Reference paper PDF (arXiv 2404.00427). |

## Top-level files

- `config.py`: TOML config loader (`ModelConfig`, sublayer pairs, paths, node-name helpers).
- `upload.py` / `browser.py` / `upload.cfg`: DVC-style SFTP store CLI and its Textual file browser (legacy Hetzner path).
- `analytical_manifold.py`: toy map `f(x)_0 = x_0²` with a known fold, used to sanity-check the ratio/tail/distance views.
- `CLAUDE.md`: project rules (uv-only, NFS relay, scratch vs NFS output, chunking). `SCRIPTS.md`: per-script flags and workflows. `README.md`: setup plus tiny_llama launch commands.
- `justfile`, `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`: tooling.
- `adversarial.ipynb`, `playground.ipynb`: exploratory notebooks.

## Typical data flow

`configs/*.toml` → `construct/` (export model, sublayer subgraphs, pullback graphs) →
`scripts/python/generate_*_sweep` (perturbed / ambient / tangent samples, run via `execute/`,
written by `ShardedLayerWriter` to scratch, relayed over NFS) → Jacobian det / kernel /
Lipschitz stats → `scripts/analysis/` fits and plots → `figures/paper/`.
Autoencoder branch: sweep clouds → `autoencoder/` training → manifold sampling → sampled sweeps.

## Notes from the review

- `examples/` and `onnx_models/` are tracked in git but not present on disk in this checkout.
- `README.md` is partly a scratchpad (commented-out section, loose notes, a long tiny_llama sbatch recipe); it could be tidied.
- `autoencoder/` and `gmanifold/` overlap (the batched version supersedes the single-model one); `autoencoder/test_script.py` still has a commented-out import from a module that no longer exists.
- `scripts/*/__pycache__` directories sit in the tree; check they are gitignored.
