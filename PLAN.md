# Plan

Write what the code needs to do as items below. Claude works from this file.
Status: `[ ]` todo, `[~]` in progress, `[x]` done.

## 1. KV cache ONNX model (GPT-2)

- [x] **Done.** KV-cache export, last-token sublayer subgraphs, KV pullbacks and the KV perturbed sweep
  all match HF and the cacheless pipeline.
  - Code: `construct/kv_cache_export.py`, `construct/kv_cache_subgraph.py`,
    `scripts/helper/generate_perturbed_sweep_kvcache.py`, `execute/optimization_loop.py`.
  - Evidence: `logs/kv_tests_full2.log` -- 184 passed (GPU, mauao, 2026-10-13 15:32), run after the last edit
    to any source file. Test plan: `tests/kv_tests/TESTS.md`.
  - Speedup vs cacheless, 768-candidate group: 0.9x at prefix 0, 1.3x at 1, 4.6x at 8, 10.6x at 32
    (`tests/kv_tests/results/timing.md`).
  - Known limitation: pos=0 uses a zero-length-past workaround (tested).
