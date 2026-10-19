# Series 4 findings: transformer KV cache (GPT-2)

See [test_plan.md](../../test_plan.md). Code: `tests/kv_tests/` (test list in `tests/kv_tests/TESTS.md`), run on ruapehu H100 via `tests/kv_tests/run_pytest.sh` (job 61896; ORT CUDA EP, fp32, TF32 off). Raw data: [junit.xml](junit.xml), [optimisation.json](optimisation.json), `tests/kv_tests/results/timing.{md,json}`. Plots: [test_outcomes.png](test_outcomes.png), [trajectory_diff.png](trajectory_diff.png), [trajectories.png](trajectories.png).

## Result: 120 / 121 pass (rerun, job 61955; JacobianExtractor no longer optimised)

| Question | Tests | Outcome |
|---|---|---|
| Is the forward pass correct (KV vs HF vs cacheless, all 25 boundaries, L = 0/1/8/32)? | prefix_lengths, export_and_subgraph, sublayer_subgraphs | pass |
| Is prefix + 1 token with KV cache the same as the full pass / as seq len 1 without a prefix? | export_and_subgraph (L=0 case and several splits) | pass |
| Are gradients correct (768x768 last-token Jacobian vs cacheless pullback vs torch autograd)? | pullback_jacobian (24) | pass |
| Pullback I/O contract, batching of candidates over one shared prefix | pullback_construction, export_and_subgraph | pass |
| Can we optimise (SGD / AdamW, the 3 adversarial losses, attn/MLP, L = 0/8/32; ORT vs torch trajectories) | optimisation (35/36) | 35 pass, trajectories agree to <= 8e-5 (most <= 1e-6). JacobianExtractorLoss is the Jacobian extractor and is tested in pullback_jacobian, not as an optimiser |
| Is KV faster than cacheless? | timing | pass (see below) |

## The one failure is a test-design issue, not a pipeline error

`test_kv_optimisation_matches_torch[0-mlp-SGD-AdversarialLoss_SquaredDiffLoss]`: the ORT and torch trajectories agree (diff 5.1e-5 < 1e-3), but plain SGD with the fixed learning rate diverges in **both** (loss 8.76 -> 9.3e4), so the extra assertion "loss decreased" fails. The step size is too large for that one configuration. Nothing was changed in the test (not asked); a smaller lr or dropping the decrease check for that case would fix it.

## Timing (one 768-candidate group, A40, from `timing.md`)

KV is 0.9x at L=0 (no prefix to reuse), 1.2x at L=1, 4.4x at L=8 and 11.3x at L=32 faster than cacheless ONNX; for L=32 it is 17.8 ms vs 200.8 ms (HF: 184.6, HF with DynamicCache: 21.1). Series 5 covers this in more depth.

## Caveats
- GPT-2 only; the optimisation tests are on single sublayers (last token), not the whole network.
- Secant optimisers (GaussNewton / NewtonRaphson) are not in this run's trajectory set (see TESTS.md known issues).
