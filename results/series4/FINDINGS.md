# Series 4 findings: transformers with KV cache (five models)

See [test_plan.md](../../test_plan.md). Code: `tests/series4/run_series4.py` (+ `series4_plots.py`, `run.sh`); jobs 61965 (tinystories-1m, gpt2) and 61979 (smollm2-135m, opt-125m, qwen2_5-0_5b), ruapehu H100, ORT CUDA EP, fp32, TF32 off. Configs are the series 3 ones. The earlier GPT-2-only pytest suite is kept in [FINDINGS_pytest.md](FINDINGS_pytest.md).

Setup: optimum `text-generation-with-past` export; one shared random prefix of length L in {0, 1, 8, 32}; 16 candidate last tokens (forward), 768 / d_model candidates (Jacobian), 8 candidates (optimisation). Baseline is the HF model in torch (hooks for activations and logits; autograd through an injected boundary, full sequence prefix + candidate, only the last position optimised). Criteria: activations and logits < 1e-4 (relative to max |HF|), full last-token Jacobian < 1e-3 (relative Frobenius), SGD/AdamW trajectories < 1e-3 (max |L_ORT - L_torch| / max |L_torch| over 30 steps). Jacobian on single sublayers h0->h1, h1->h2, h2->h3, h3->h4 and the last two; optimisation on one attention (h2->h3) and one MLP (h3->h4) sublayer, three adversarial losses, L = 0 / 8 / 32. `JacobianExtractorLoss` is used for the Jacobian only.

## Results (worst case per model)

| Model | Layers | KV heads x dim | Activations | Logits | Jacobian | Trajectories (36) |
|---|---|---|---|---|---|---|
| [tinystories-1m](tinystories-1m/) | 8 | 16 x 4 | 2.3e-6 | 2.1e-6 | 4.6e-7 | 2.0e-7, all pass |
| [gpt2](gpt2/) | 12 | 12 x 64 | 2.4e-6 | 2.8e-6 | 2.1e-6 | **0.19** (1 run), 3.6e-3 (2 runs), rest <= 2e-5 |
| [smollm2-135m](smollm2-135m/) | 30 | 3 x 64 | 3.7e-6 | 3.4e-6 | 4.9e-7 | **2.7e-3** (1 run), rest pass |
| [qwen2_5-0_5b](qwen2_5-0_5b/) | 24 | 2 x 64 | 4.4e-5 | 2.4e-5 | 2.1e-5 | 2.3e-7, all pass |
| [opt-125m](opt-125m/) | 12 | 12 x 64 | 4.2e-6 | 3.3e-6 | **27.8** (flattened-MLP sublayers) | **invalid**, see below |

Summary: [summary.png](summary.png) ([html](summary.html)); per-model `results.html` / `results.png`. The `results.json` / `summary.json` files are not tracked in git (`*.json` is ignored); regenerate with `tests/series4/run.sh`.

## Conclusion

- **KV forward is correct on all five models**: all 2n+1 boundary activations of the last token and the last-token logits match HF (prefix + continuation vs the full pass), at <= 4.4e-5, for every prefix length including L = 0.
- **KV gradients are correct on four models**: the full last-token Jacobian from the KV pullback matches torch autograd to <= 2.1e-5 on tinystories, gpt2, smollm2 and qwen2.5 (all 24 sublayer x prefix-length combinations each).
- **Optimisation agrees with torch on tinystories-1m and qwen2.5 (all 36 runs)**; on gpt2 33 of 36 and smollm2 35 of 36 are below 1e-3. The misses are all plain SGD on the MLP sublayer at L = 0, see below.
- **OPT-125m is not supported by the KV sublayer pullback** (forward is fine).

## Findings and caveats

- **The SGD misses are a diverging-objective effect, not (shown to be) a gradient error.** gpt2 `mlp|L0|SGD|SquaredDiff` (0.19) and smollm2 `mlp|L0|SGD|SquaredDiff` (2.7e-3): with lr 0.01 the loss runs away (gpt2: 8.7 -> 9e4 in both ORT and torch, end values 88943 vs 90014) and fp32 differences are amplified. gpt2's NormalizedSquaredDiff and Envelope with SGD at the same setting (3.6e-3) converge (to 0.7123 in both) but also exceed the criterion at some step; I have not diagnosed that one. Not tested in float64 or with a smaller lr. The `loss_decreased` flag in `results.json` is false for the two diverging runs.
- **OPT-125m**: the optimum export flattens the residual stream to 2D `(tokens, d)` between the attention and MLP blocks, whereas the KV sublayer code assumes `(batch, 1, d)`. Sublayers starting at an even boundary >= 2 fail to run (`Invalid rank for input hidden_in: Got 3 Expected 2`), and the sublayers that run through the flattened tensor (h1->h2, h3->h4, h23->h24) return wrong Jacobians (rel. error ~28) and trajectories (all 18 optimisation runs exceed 1e-3); this matches the shape mismatch but I did not verify each wrong value. Only h0->h1 is right (2e-7). The runner records these as failures rather than hiding them. Supporting OPT would need the KV sublayer extraction to reshape the boundary tensors.
- **Library changes made to run these models**: (1) `construct/kv_cache_subgraph.py` now looks through the `Cast` at the start of a Llama-style layer when finding the embedding tensor (smollm2 / qwen failed there); (2) `construct/gradient_block.py` disables ORT's `SimplifiedLayerNormFusion` for the gradient build (the fused RMSNorm node had no usable gradient, `Bad node spec`), and sets the log level to errors only (the build otherwise prints thousands of INFO lines); (3) the opt-125m config `block_prefix` was `/model/decoder/layers`, the optimum graph uses `/decoder/layers` (`construct/make_config.py` generates the wrong value for such models).
- No `If` / `Loop` / `Scan` in any of the KV graphs (`control_flow: false`), so transformer control flow is still untested.
- Not repeated here (covered in series 3, 5): cacheless-vs-KV comparison, timing.
