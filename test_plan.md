# Test Plan

**JacobianExtractorLoss is the Jacobian extractor, not an optimisation objective.** Its loss is sum(output × mask); with a one-hot mask the pullback gradient is one row of the Jacobian. Every series therefore uses it to assemble the full Jacobian (one pullback call per output element) and compares that with the torch autograd Jacobian; SGD/AdamW optimisation is run only on the three adversarial losses.

Each series has its own `results/seriesN/` directory with `FINDINGS.md` (numbers and conclusions) and plotly figures (`.html` + `.png`). Below, for each series: the goal, the tests performed, and what each plot is meant to show.

## Series 1: Validity on simple cases

### Goal
Check that the export → subgraph extraction → pullback generation → optimisation pipeline is correct on simple networks with no control flow.

### Tests performed
Six small networks (`mlp_tanh`, `mlp_gelu_deep`, `resmlp`, `resmlp_layernorm`, `conv_relu`, `mlp_softmax_mixer`) exported with `torch.onnx.export`. For each network:
1. **Extraction:** subgraphs of every single stage and of the whole network (`ensure_subgraphs`); forward output compared with the torch stage composition.
2. **Pullback:** `ensure_subgraph_pullback` for all four `LossType`s; loss value and input gradient (random mask for JacobianExtractor) compared with torch autograd.
3. **Full Jacobian:** with `JacobianExtractorLoss` and one-hot masks, the whole Jacobian of every span compared with the torch autograd Jacobian.
4. **Optimisation:** 100-step SGD and AdamW trajectories (the three adversarial losses) driven by the ORT pullback gradient compared with torch autograd + `torch.optim`.

Code: `tests/series1/run_series1.py`, plots `series1_plots.py`, run on mauao CPU (CPU EP, so no TF32 noise) via `tests/series1/run.sh`.

### Plots
- **Per-network `results.png` (4 panels):** (a) forward error of each extracted span, (b) pullback loss-value error, (c) pullback gradient error ∂L/∂input, per span and loss type, plus a black ✕ per span for the JacobianExtractor full-Jacobian error, (d) the optimisation trajectory error over the 100 steps (SGD solid, AdamW dashed). Every mark is a measured ORT-vs-torch discrepancy, with the pass-criterion line drawn: the pipeline is correct if all marks lie below the line.
- **`summary.png`:** worst case per network for the static checks (forward / loss / gradient) and for the trajectory, against the criteria, to compare networks at a glance.

Findings: [results/series1/FINDINGS.md](results/series1/FINDINGS.md).

## Series 2: Validity on more complex cases (control flow)

### Goal
Check the same pipeline on simple networks **with** control flow.

### Tests performed
Networks `lin → control-flow stage → lin` (3 stages) exported through TorchScript so `if` / `for` become ONNX `If` / `Loop`: `cond_branch_pos`, `cond_branch_neg` (each branch of an `If` exercised separately), `loop_fixed` (unrolled by TorchScript), `loop_break` (data-dependent `Loop`), `loop_with_cond` (`If` inside a loop). Same checks as series 1 (extraction, four-loss pullback, JacobianExtractor full Jacobian, SGD/AdamW trajectories) on the plain stages, the control-flow stage and the whole net. Pullback generation runs in a child process because ORT can SIGSEGV. Failures are recorded as results. Run on ruapehu GPU (`tests/series2/run_series2.py`, `results/series2/`; the older CPU run in `results/series2_cpu/` predates the Jacobian change).

### Plots
Same figures as series 1 (per-experiment `results.png`, `summary.png`). Where extraction, pullback or optimisation failed there is nothing to plot for that part; the figure shows which parts of the pipeline work (errors at float32 level) and which do not. `summary.png` only contains experiments for which all three stages exist.

Findings: [results/series2/FINDINGS.md](results/series2/FINDINGS.md).

## Series 3: Validity on transformers

### Goal
Check the pipeline on Hugging Face transformer models, with and without control flow.

### Tests performed
Five models (`tinystories-1m`, `gpt2`, `smollm2-135m`, `opt-125m`, `qwen2_5-0_5b`), exported with the project's `construct.model_export` (configs in `tests/series3/configs`), random token ids, batch 2, seq 8. Boundaries are embeddings, post-attention and post-MLP of each layer. For spans (single sublayers, one layer, several layers, last layer): extraction vs torch hooks, full-model logits vs torch, pullbacks for all four losses vs torch autograd (layer-aligned spans), the **full last-token Jacobian** (d_model JacobianExtractor one-hot pullbacks) vs torch autograd, and 30-step SGD/AdamW trajectories (adversarial losses) on the multi-layer span. Run on ruapehu GPU, TF32 off (`tests/series3/run_series3.py`, `results/series3/`; the older CPU run in `results/series3_cpu/` predates the Jacobian change). None of the exports contained control flow, so only the "without control flow" half is covered.

### Plots
Same layout as series 1: per-model `results.png` with forward / loss / gradient / full-Jacobian error per span and the trajectory error, against the criteria, and `summary.png` comparing the models. They show whether the error stays at float32 level as depth and operator variety (LayerNorm, RMSNorm, rotary, Where, ...) grow, and which model/loss/optimiser combination exceeds the trajectory criterion (SquaredDiff + SGD for two models).

Findings: [results/series3/FINDINGS.md](results/series3/FINDINGS.md).

## Series 4: Validity on transformers with KV cache

### Goal
KV-cache exports (optimum) of the series 3 transformers: are forward and gradients correct, can we optimise well; can we fix a prefix and compute forward, grad and loss only for the last token; does batching of candidates over one shared prefix work.

### Tests performed
Script-based (like series 1-3), `tests/series4/run_series4.py`, on ruapehu H100 (ORT CUDA, fp32, TF32 off), five models (`tinystories-1m`, `gpt2`, `smollm2-135m`, `opt-125m`, `qwen2_5-0_5b`, configs from `tests/series3/configs`), prefix length L in {0, 1, 8, 32}. Baseline: the HF model in torch.
- **Forward:** all 2n+1 residual boundaries and the logits of the last token, KV (prefix past + candidates) vs HF on the full sequence, 16 candidates per prefix.
- **Gradients:** the full d_model x d_model last-token Jacobian (`JacobianExtractorLoss`, one-hot masks, batch = d_model) of six single sublayers vs torch autograd.
- **Optimisation:** 30-step SGD and AdamW x 3 adversarial losses x attention (h2->h3) / MLP (h3->h4) x L = 0 / 8 / 32, 8 candidates, KV pullback vs torch autograd.
- Batching over one shared prefix is inherent in all of the above.

The earlier GPT-2-only pytest suite (`tests/kv_tests/`, 121 tests, includes the cacheless comparison and timing) is kept; its findings are in `results/series4/FINDINGS_pytest.md`.

### Plots
- **Per-model `results.png` (4 panels):** (a) last-token boundary activation error per boundary and prefix length, (b) logits error per prefix length, (c) full-Jacobian error per sublayer and prefix length, (d) optimisation trajectory error per sublayer type / prefix length (colour = loss, filled SGD, open AdamW), each against its pass criterion; values above the criterion are shown, not clipped.
- **`summary.png`:** worst case per model for forward, Jacobian and trajectory.

Findings: [results/series4/FINDINGS.md](results/series4/FINDINGS.md).

## Series 5: Time and memory use for ONNX vs torch

### Goal
Compare time and memory of ONNX vs torch operations, with KV-cache transformers as the main target.

### Tests performed
GPT-2, ruapehu H100, fp32, one group of 768 candidate last tokens after a shared prefix of length L ∈ {0, 8, 32, 128}. Two operations: forward to boundary 24, and the JacobianExtractor pullback of one attention sublayer (full 768×768 Jacobian). Four variants: `kv_onnx`, `cacheless_onnx`, `torch_cache` (HF DynamicCache), `torch`. Each (op, variant, L) runs in its own subprocess; time is the median of 10 runs after 3 warm-ups; memory is polled device usage above the CUDA context. Outputs of the variants are checked to agree. Code: `tests/series5/run_series5.py`.

### Plots
- **`time.png`:** time per group vs prefix length for each variant (forward and pullback panels): shows how cost scales with L, i.e. where the KV cache pays off versus recomputing the whole sequence.
- **`memory.png`:** peak device memory vs prefix length for each variant: shows the memory cost of each approach.

Findings: [results/series5/FINDINGS.md](results/series5/FINDINGS.md).

Plots for series 4 and 5 are generated by `tests/plot_series45.py` (`tests/plot_series45.sh`).
