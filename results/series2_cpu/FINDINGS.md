# Series 2 findings: networks with control flow

See [test_plan.md](../../test_plan.md). Code: `tests/series2/run_series2.py` (reuses the series 1 harness, same criteria as series 1: static checks < 1e-5, optimisation trajectories < 1e-3 where they could run), run on ngog CPU via `tests/series2/run.sh`. Networks are `lin -> control-flow stage -> lin` (3 stages, boundaries h0..h3, input (8,16)), exported through TorchScript so `if` / `for` become ONNX `If` / `Loop`. Torch autograd on the un-scripted Python module is the baseline. Failures are results, not test errors: they are recorded in each `results.json` under `errors`.

## Results

| Experiment | Control-flow op in ONNX | Plain-stage spans (h0->h1, h2->h3) | Control-flow stage (h1->h2) | Whole net (h0->h3) pullback | Optimisation |
|---|---|---|---|---|---|
| [cond_branch_pos](cond_branch_pos/) | `If` (taken: tanh branch) | pass (extract 1.2e-7, grad 3.9e-7) | **extraction fails** | **pullback crashes (SIGSEGV)** | not possible |
| [cond_branch_neg](cond_branch_neg/) | `If` (taken: sigmoid branch) | pass (extract 3.6e-7, grad 2.7e-7) | **extraction fails** | **pullback crashes (SIGSEGV)** | not possible |
| [loop_fixed](loop_fixed/) | none: TorchScript unrolled the fixed 4-trip loop | pass | pass (it is plain ops) | pass | pass, trajectory diff <= 8.5e-6 |
| [loop_break](loop_break/) | `Loop` (data-dependent exit) | h0->h1 pass (6e-8, grad 3.3e-7) | **extraction fails** (`Field 'shape' of 'type' is required but missing`; h2->h3 too, since the Loop's value-infos lose their shapes) | **pullback fails**: ORT has no gradient builder for `Loop` (`gradient_builder_registry.cc:29`) | not possible |
| [loop_with_cond](loop_with_cond/) | `If` (the loop was unrolled, the branch stayed) | pass (grad 4.3e-7) | **extraction fails** | **pullback crashes (SIGSEGV)** | not possible |

Figures: each directory has `results.html` + `results.png`; [summary.png](summary.png) only contains `loop_fixed` because the summary needs extraction, pullback and optimisation to all exist.

## Conclusion

- The pipeline is correct (float32 level) on every stage **outside** a control-flow node, and on a fixed loop, which TorchScript turns into plain ops.
- It does **not** work through `If` or `Loop` nodes:
  1. `construct.subgraph_extract` cannot extract a subgraph containing an `If` ("Bad node spec for node ... If") or a `Loop` (missing shapes on the nested graph's values).
  2. Pullback generation on a graph with `If` kills the process with SIGSEGV inside ORT's gradient builder; with `Loop` it raises (no gradient registered). The harness runs pullback generation in a child process to survive this.
- So a model with data-dependent control flow cannot currently be differentiated by this pipeline. The viable workaround is to export the branch actually taken (trace, or fix the condition before export), which is what `cond_branch_*` would reduce to.

## Caveats

- Only ngog CPU was used. `If` conditions were fixed by a `shift` on the input so each branch was exercised separately (`pos` / `neg`); the branch is chosen at run time by the data, so a pullback through `If` that did work would be per-branch.
- Not investigated: whether upstream ORT fixes (gradient for `If`/`Loop` subgraphs) or inlining `If` when the condition is a constant would remove the crash. That would be a code change beyond this test series.
