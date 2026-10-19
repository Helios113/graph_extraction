# Series 1 findings

See [test_plan.md](../../test_plan.md) for the plan; figures are in this directory (HTML + PNG each).

## Experiments and results

All values are the worst case over every span and loss. Errors are relative unless noted. The loss-value error is relative to sum|f*mask| for JacobianExtractor (a signed sum that can cancel to ~0) and to max|loss| otherwise.

Pass criteria: static checks (forward, loss, gradient) below 1e-5 (~100x float32 epsilon, 1.2e-7); optimisation trajectories below 1e-3, measured as max |L_ORT - L_torch| / max |L_torch| over 100 steps. Each plot shows the measured values against the criterion line, so a pass is every mark below the line.

| Experiment | Ops | Extraction max abs err | Grad rel err | Loss rel err | Optimisation trajectory diff |
|---|---|---|---|---|---|
| [mlp_tanh](mlp_tanh/) | Gemm, Tanh | 1.2e-07 | 3.0e-07 | 1.6e-07 | 3.7e-07 |
| [mlp_gelu_deep](mlp_gelu_deep/) | Gemm, Gelu | 1.8e-07 | 8.2e-07 | 2.3e-07 | 2.1e-05 |
| [resmlp](resmlp/) | Gemm, Sigmoid, Mul, Add | 2.4e-07 | 2.3e-07 | 1.4e-07 | 1.5e-06 |
| [resmlp_layernorm](resmlp_layernorm/) | + LayerNormalization | 3.0e-07 | 2.7e-07 | 1.7e-07 | 1.6e-06 |
| [conv_relu](conv_relu/) | Conv, Relu | 4.8e-07 | 3.9e-07 | 2.2e-07 | 1.5e-06 |
| [mlp_softmax_mixer](mlp_softmax_mixer/) | Gemm, Softmax | 3.0e-08 | 3.1e-07 | 2.1e-07 | 4.6e-05 |

Summary: [summary.json](summary.json), figure [summary.png](summary.png) ([html](summary.html)).

## Conclusion

All six networks match the torch baseline to float32 precision. The extracted subgraphs, the pullback loss values and gradients, and the optimisation trajectories all agree, for all four loss types.

## Findings and caveats

- **Duplicate gradient names with default LayerNorm parameters (fixed).** With three or more stacked `LayerNorm`s at default init (weight = 1, bias = 0), pullback generation failed with `Duplicate definition of name (1.n.weight_grad)`. Cause: `torch.onnx.export` deduplicates identical parameters into one initializer plus an `Identity` node per extra consumer. ORT folds the Identities away, so several `LayerNormalizationGrad` nodes read one shared tensor and each emits the same `<param>_grad` output. Fix: `_unshare_identity_initializers` in `construct/pullback.py` gives each `Identity(initializer)` its own initializer copy before the gradient graph is built. `resmlp_layernorm` now runs with default LayerNorm parameters (3 and 5 blocks build) and still matches torch.
- **Equal input and output shapes.** The adversarial losses create `x` and `target` with `input_shape` and compare `target` against the subgraph output. The test networks therefore keep the shape equal across every stage.
- **Optimisation setup.** The optimisation loop was removed from the repo (`execute/`), so the script uses a small inline SGD/AdamW. Targets are reachable, set to `f(y*)`.
- **Envelope loss.** `AdversarialLoss_EnvelopeDiffLoss` is currently identical in implementation to `AdversarialLoss_NormalizedSquaredDiffLoss`, so it is checked against the same formula.
- **Dependencies.** plotly is not a project dependency. Plots use plotly, run with `uv run --with plotly --with kaleido`. PNG export needs Chrome. Home directories are per node, so Chrome is installed on NFS at `/nfs-share/pa511/tools/chrome` and the plot script sets `BROWSER_PATH` to it.

## JacobianExtractorLoss is checked as a Jacobian extractor, not optimised

`JacobianExtractorLoss` (loss = sum(output * mask), linear, no minimum) is the Jacobian extractor: with a one-hot mask at output element j the pullback gradient is row j of the Jacobian. It is therefore no longer used as an optimisation objective (SGD/AdamW trajectories are run for the three adversarial losses only). Instead, for every span the full Jacobian is assembled from one pullback call per output element (`jacobian_check` in `tests/series1/run_series1.py`) and compared with torch autograd rows. Worst case over spans, relative to max|J_torch| (criterion 1e-5):

| Network | Full-Jacobian rel err |
|---|---|
| conv_relu | 2.3e-07 |
| mlp_gelu_deep | 7.3e-07 |
| mlp_softmax_mixer | 2.4e-07 |
| mlp_tanh | 2.4e-07 |
| resmlp | 1.1e-07 |
| resmlp_layernorm | 1.1e-07 |

(`conv_relu` single stages agree bit-for-bit, 0.0.) Rerun on mauao CPU, job 61956.
