---
name: construct-onnx
description: Use when you need gradients, VJPs, full Jacobians or gradient-based optimisation of an intermediate residual-stream stage of a Hugging Face causal LM through ONNX Runtime. The construct-onnx package exports the model to ONNX (cacheless or KV-cache), cuts out the subgraph between two residual boundaries, and builds a pullback graph (loss value + forward output + input gradient) for it. Triggers - "pullback", "Jacobian of a transformer sublayer", "adversarial sample / optimise a hidden state", "last-token gradient with a KV cache", "extract subgraph between add nodes".
---

# construct-onnx

Builds ONNX artifacts for a model **config** (a TOML): base model export -> subgraph between two tensors -> **pullback graph**. A pullback graph takes the subgraph input (plus loss inputs) and returns the loss, the subgraph output, and `d loss / d input` computed by ONNX Runtime training (`onnxruntime-training`, a custom build).

Import name is `construct` (distribution name `construct-onnx`). Everything is cached on disk and guarded by lock files, so every `ensure_*` call is idempotent and safe to call from several processes.

## Install and environment

```
uv add construct-onnx            # or: uv add --editable /nfs-share/pa511/code_bases/new_jac
```

`onnxruntime-training` is a **custom wheel** on NFS, pinned via `[tool.uv.sources]` in this repo's pyproject. `uv` does not inherit sources from a dependency, so a consuming project must add the same line itself:

```toml
[tool.uv.sources]
onnxruntime-training = { path = "/nfs-share/pa511/code_bases/onnxruntime_build_output/1.20.0_py311_cuda12_sm86/ort_build/RelWithDebInfo/dist/onnxruntime_training-1.20.0+cu126-cp311-cp311-linux_x86_64.whl" }
```

Python 3.11 only (`>=3.11,<3.12`), `transformers>=4.57,<4.58`. Cluster rules: run via sbatch on mauao (`ampere`) or ruapehu (`hopper`), never the login node. **Do not use ngongotaha (ngog)**: its CUDA version does not match the custom ORT build.

Where artifacts go: `construct.config.ONNX_DIR` (default `Path("onnx_models")`, relative to the cwd). Set it once before building configs, e.g. `from construct import config; config.ONNX_DIR = Path("/nfs-share/me/onnx_cache")`. Each config gets `ONNX_DIR/<cfg.name>/`.

## Concepts

- **Boundary j**: the j-th residual-stream tensor. `0` = embeddings, odd `2l+1` = post-attention of layer l, even `2l+2` = post-MLP of layer l. A model with n layers has `2n+1` boundaries; `cfg.sublayers[j].output` is the ONNX tensor name of boundary j (`sublayers[0]` is `input_ids -> embedding`).
- **Sublayer / span**: any `boundary a -> boundary b`. One attention or MLP step is `j -> j+1`; a layer is `2l -> 2l+2`.
- **Losses** (`construct.config.LossType`):
  - `JacobianExtractorLoss` = `sum(output * mask)`. Linear, **no minimum, never optimise it**. With a one-hot `mask` the gradient is one row of the Jacobian `d out[i] / d in`. Use it only to extract Jacobians.
  - `AdversarialLoss_SquaredDiffLoss`, `AdversarialLoss_NormalizedSquaredDiffLoss`, `AdversarialLoss_EnvelopeDiffLoss`: optimisation objectives (per-row, summed over the last axis). Inputs `x`, `target`.

## Workflow A: cacheless pipeline (static batch, full sequence)

1. Make a config for the model (once): `uv run python -m construct.make_config <hf-id> --out configs/<name>.toml [--seq 4 --batch 1024]`. It finds the boundary tensors by running the model and matching the ONNX values; it fails loudly if the architecture has no `x + attn(norm x)`, `+ mlp(norm .)` residual structure. Existing configs: `configs/gpt2.toml`, `tests/series3/configs/` (tinystories-1m, gpt2, opt-125m, smollm2-135m, qwen2_5-0_5b).
2. Build artifacts:

```python
from dataclasses import replace
from construct.config import load_config, LossType, SublayerConfig
from construct.construct import ensure_base_model, ensure_subgraphs, ensure_subgraph_pullback

cfg = load_config("configs/gpt2.toml")
cfg = replace(cfg, batch=2, seq=8, batch_by_node={})          # static shapes are baked into the export
ensure_base_model(cfg)                                          # torch.onnx export -> cfg.base_model_path
b = [s.output for s in cfg.sublayers]                           # boundary tensor names
sub = SublayerConfig(loss_type=LossType.AdversarialLoss_SquaredDiffLoss,
                     input_shape=[2, 8, cfg.d_model], input=b[2], output=b[4])   # span h2 -> h4
cfg = replace(cfg, sublayers=[sub])
spans = ensure_subgraphs(cfg)                                   # [(forward_onnx_path, SublayerConfig)]
pullbacks = ensure_subgraph_pullback(spans)                     # [pullback_onnx_path]
```

Use a **different `cfg.name` (hence directory) per loss type / span set**: the pullback file name does not encode the loss type, so two losses for the same span in one directory collide (the series 3 runner uses `name=f"{name}/{loss.value}/h{a}h{b}"`).

3. Run the pullback with onnxruntime. Pass `("CUDAExecutionProvider", {"use_tf32": "0"})` and set `NVIDIA_TF32_OVERRIDE=0` for fp32-accurate gradients (ORT's default TF32 on A40/H100 gives ~1e-3 relative noise).

Pullback graph I/O:

| | name | shape |
|---|---|---|
| in | `<span input tensor name>` | `input_shape` |
| in | `x`, `target` (adversarial losses), `mask` (JacobianExtractor) | same as the span **output** (x: as the input) |
| in | extra graph inputs the span needs (e.g. `input_ids` for a span starting at the embedding) | feed what the session lists |
| out | loss `(batch, 1)` per row, forward output `<span output name>`, `<span input name>_grad` | |

Always read `sess.get_inputs()/get_outputs()` rather than assuming; feed only names the session lists.

### Full Jacobian recipe

Feed the same input in every batch row and a one-hot `mask` per row (row i -> output element i); `_grad` row i is `d out[i] / d in`. Batch = number of output elements (d_model for a last-token Jacobian).

## Atomic operations: no config, any ONNX file

The config layer is only bookkeeping. Two functions do the actual work on **any** ONNX model, so use them directly when you just want a subgraph or a Jacobian. (Verified end to end on a small residual net: Jacobian vs torch autograd max abs diff 1.2e-7.)

```python
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from construct.config import LossType
from construct.subgraph_extract import extract_subgraph
from construct.pullback import generate_subgraph_pullback

model = Path("m.onnx")                      # any ONNX file, static shapes
g = onnx.load(str(model), load_external_data=False).graph
print([o for n in g.node for o in n.output])  # pick tensor names (inputs and outputs of the span)

# 1. Subgraph between two tensors (any tensors, not only residual boundaries). Several outputs: pass a list.
extract_subgraph(model, Path("blk.forward.onnx"), "add", "add_1")        # saves weights to blk.forward.onnx.data

# 2. Pullback of that subgraph. input_shape = shape of the subgraph input (static).
pb = generate_subgraph_pullback(Path("blk.forward.onnx"), LossType.JacobianExtractorLoss,
                                [B, S, D], "add", "add_1")               # -> Path to ...pullback.onnx
sess = ort.InferenceSession(str(pb), providers=["CPUExecutionProvider"])  # or CUDA with use_tf32=0
```

- **File names matter:** the pullback is written next to the subgraph as `<stem minus its last dotted part>.pullback.onnx`. `blk.forward.onnx` -> `blk.pullback.onnx`; a stem without a dot (`sub.onnx`) gives the hidden file `.pullback.onnx`. Use the returned path, name subgraphs `<name>.forward.onnx`, and use one directory per (span, loss) since the name does not encode the loss.
- **Shapes:** `input_shape` is also used for the loss inputs (`mask`, `target`, `x`), so the span must be shape-preserving (residual-stream spans are). Batch and sequence are baked in; a different batch needs a new pullback.
- The returned/graph outputs are `[loss, <output tensor name>, <input tensor name>_grad]`; the loss output has an auto-generated name (e.g. `onnx::reducesum_output::3`), so take outputs by position or by `sess.get_outputs()`, not by name.
- Inputs: the subgraph input tensor, `mask` (JacobianExtractor) or `target` (+ `x` for the normalized/envelope losses), plus any other graph input the subgraph still needs (feed it, e.g. `input_ids`).
- Give the pullback the value of the input tensor, not the model input: get it by adding that tensor as an extra output of the full model (`model.graph.output.append(onnx.ValueInfoProto(name=...))`) or by running the cut `extract_subgraph(model, p, model_input, "add")`.

### VJP / gradient of a scalar objective

`JacobianExtractorLoss` computes `sum(output * mask)`, so `mask` is the cotangent: `grad = J^T mask` for any `mask` of the output shape. A single call gives a vector-Jacobian product.

### Jacobian by hand

Row i of `J = d out / d in` is the gradient with a one-hot mask at output element i. Feed the same input value in every row.

```python
names = [o.name for o in sess.get_outputs()]            # [loss, out, in_grad]
pos = S - 1                                             # e.g. last token
J = np.zeros((D, D), np.float32)
for i in range(D):                                      # B = 1 pullback: one call per output element
    mask = np.zeros((B, S, D), np.float32); mask[0, pos, i] = 1
    _, _, grad = sess.run(names, {"add": y, "mask": mask})
    J[i] = grad[0, pos]                                 # d out[pos, i] / d in[pos, :]
```

Faster: build the pullback with `input_shape=[D, S, D]` (batch = number of output elements), tile `y` over the batch, set `mask[i, pos, i] = 1`, and read `grad[:, pos, :]` as the full Jacobian in one call. Rows are independent only if the batch rows do not interact (true for per-sample transformers). A diagonal block across tokens (only the last position) is what the KV workflow below uses; for the full multi-token Jacobian loop over `pos` too.

### Cutting several tensors at once / checking

- `extract_subgraph(model, out, "input_ids", ["add", "add_2", ...])` gives a graph returning all those tensors (this is how the all-boundaries graph is built).
- Check any result against torch: wrap the same span as a torch function and use `torch.autograd.functional.jacobian`; set `NVIDIA_TF32_OVERRIDE=0` and `use_tf32=0` on GPU or expect ~1e-3 noise.

## Workflow B: KV cache, last token only

For "fix a prefix, vary the last token, gradients only for that token". Uses `optimum` to export a `text-generation-with-past` graph; needs `block_prefix` in the config (`make_config` writes it).

```python
from construct.config import load_config, LossType
from construct.kv_cache_export import ensure_kv_cache_model
from construct.kv_cache_subgraph import (ensure_kv_cache_boundaries, ensure_kv_cache_sublayer_pullback,
                                         KV_SUBLAYER_IN, KV_SUBLAYER_OUT, past_kv_input_names, present_kv_names)

cfg = load_config("tests/series3/configs/gpt2.toml")
ensure_kv_cache_model(cfg)                                   # optimum export (slow the first time)
bpath, bnames, n_layers = ensure_kv_cache_boundaries(cfg)    # graph returning all boundaries + present K/V
sub, pb = ensure_kv_cache_sublayer_pullback(cfg, j_in=2, batch=B, loss_type=LossType.AdversarialLoss_SquaredDiffLoss)
```

- Prefix pass: run the boundaries graph on the prefix `(1, L)` with an empty past `(1, kv_heads, 0, head_dim)`; its `present.*` outputs, renamed `past_key_values.*`, are the prefix cache. For `L == 0` just feed the zero-length past (the graph cannot run an empty `input_ids`).
- Pullback inputs: `hidden_in` `(B, 1, d_model)` (last-token boundary j_in), `mask`/`target`/`x` `(B, 1, d_model)`, plus `sub.extra_inputs` (that layer's `past_key_values.*` broadcast to batch B, `attention_mask` ones `(B, L+1)`, `input_ids` zeros `(B, 1)` - shape only). Outputs: loss, `hidden_out`, `hidden_in_grad`.
- Only single sublayers `j_in -> j_in+1` are supported; the pullback is cached per `(loss, batch)`.
- Reference implementation (feeds, prefix handling, torch baselines): `tests/series4/run_series4.py`.

## Validated behaviour (see `test_plan.md`, `results/seriesN/FINDINGS.md`)

Against torch autograd: forward/loss/gradient/Jacobian agree to ~1e-5 or better on GPT-2, OPT-125m, SmolLM2-135M, Qwen2.5-0.5B, TinyStories-1M, cacheless and KV; SGD/AdamW trajectories agree to <= 1e-3. Control-flow ops (`If`/`Loop`) in simple nets: see `results/series2/FINDINGS.md` for what breaks. Not tested: transformers whose export contains control flow.

## Pitfalls

- Plain SGD on adversarial losses can diverge for a fixed lr (in torch and ORT alike); that is the objective, not a bug.
- ORT pullback generation can SIGSEGV on some graphs; series 2 builds it in a child process.
- `ensure_*` reuse stale artifacts if the config's shapes changed but the name did not: change `cfg.name` or pass `force=True`.
- Batch/seq are fixed at export time for the cacheless path; the KV path has dynamic batch and sequence axes.
- Use `uv run` for everything; GPU work goes through sbatch, never the login node.
