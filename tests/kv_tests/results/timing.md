# KV-cache vs cacheless vs HF timing (GPT-2, 768 candidate last tokens, boundary 24, fp32, 1x A40)

Median of 10 runs after 3 warm-ups, ms per 768-candidate group. `kv_onnx` = prefix pass + continuation.

| prefix | kv_onnx | (prefix only) | cacheless_onnx | hf | hf_cache | kv vs cacheless | kv vs hf | full-vocab group: kv / cacheless / hf (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 14.87 | 0.00 | 13.99 | 25.25 | 20.12 | 0.9x | 1.7x | 0.97 / 0.92 / 1.65 |
| 1 | 20.04 | 4.30 | 26.98 | 36.32 | 27.42 | 1.3x | 1.8x | 1.03 / 1.77 / 2.38 |
| 8 | 24.17 | 4.57 | 110.93 | 105.20 | 29.65 | 4.6x | 4.4x | 1.29 / 7.26 / 6.88 |
| 32 | 39.01 | 4.55 | 414.01 | 354.64 | 40.05 | 10.6x | 9.1x | 2.26 / 27.09 / 23.21 |
