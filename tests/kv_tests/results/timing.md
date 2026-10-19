# KV-cache vs cacheless vs HF timing (GPT-2, 768 candidate last tokens, boundary 24, fp32, 1x A40)

Median of 10 runs after 3 warm-ups, ms per 768-candidate group. `kv_onnx` = prefix pass + continuation.

| prefix | kv_onnx | (prefix only) | cacheless_onnx | hf | hf_cache | kv vs cacheless | kv vs hf | full-vocab group: kv / cacheless / hf (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 7.40 | 0.00 | 6.65 | 14.26 | 10.39 | 0.9x | 1.9x | 0.48 / 0.44 / 0.93 |
| 1 | 10.36 | 2.65 | 11.45 | 17.07 | 14.16 | 1.1x | 1.6x | 0.51 / 0.75 / 1.12 |
| 8 | 11.44 | 2.92 | 43.73 | 46.18 | 15.33 | 3.8x | 4.0x | 0.56 / 2.86 / 3.02 |
| 32 | 15.44 | 3.08 | 167.10 | 147.83 | 19.44 | 10.8x | 9.6x | 0.81 / 10.93 / 9.67 |
