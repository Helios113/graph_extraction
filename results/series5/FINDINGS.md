# Series 5 findings: time and memory, ONNX vs torch (GPT-2, KV cache)

See [test_plan.md](../../test_plan.md). Code: `tests/series5/run_series5.py` (job 61897, ruapehu H100, fp32, TF32 off, 1 GPU). One group of 768 candidate last tokens after a shared prefix of length L = 0/8/32/128. `fwd` = residual stream after the last block; `pb` = JacobianExtractor pullback of one attention sublayer, i.e. the full 768x768 last-token Jacobian. Variants: `kv_onnx`, `cacheless_onnx`, `torch_cache` (HF DynamicCache), `torch`. Each run is its own subprocess; time is the median of 10 runs after 3 warm-ups; memory is device-wide used memory minus the CUDA-context baseline. Data: [results.json](results.json). Plots: [time.png](time.png), [memory.png](memory.png). 32 rows, 0 errors.

## Correctness
All variants agree: max relative error vs the reference <= 2.6e-6 (forward) and <= 1.6e-6 (pullback).

## Time (ms)

| op | L | kv_onnx | cacheless_onnx | torch_cache | torch |
|---|---:|---:|---:|---:|---:|
| fwd | 0 | 7.6 | 6.6 | 10.3 | 13.4 |
| fwd | 8 | 11.6 | 44.7 | 15.1 | 44.2 |
| fwd | 32 | 15.7 | 167.2 | 19.5 | 146.9 |
| fwd | 128 | 29.3 | 697.4 | 39.2 | 617.6 |
| pb | 0 | 0.87 | 0.60 | 1.33 | 1.27 |
| pb | 8 | 4.12 | 2.85 | 5.19 | 2.93 |
| pb | 32 | 4.86 | 9.28 | 6.01 | 7.78 |
| pb | 128 | 7.85 | 40.8 | 10.2 | 33.1 |

## Peak device memory (MB, above CUDA context)

| op | L | kv_onnx | cacheless_onnx | torch_cache | torch |
|---|---:|---:|---:|---:|---:|
| fwd | 128 | 17228 | 17590 | 14552 | 8898 |
| pb | 128 | 4354 | 9446 | 2734 | 4672 |

(Full table in results.json / memory.png.)

## Findings
- **Forward:** KV cache ONNX cost grows slowly with L (7.6 -> 29 ms) while cacheless ONNX and plain torch grow roughly linearly (-> ~700 / ~620 ms at L=128): 24x faster at L=128. It also beats HF DynamicCache by 1.3x at L=128. At L=0 there is no prefix to reuse, so KV is not faster than cacheless (7.6 vs 6.6 ms).
- **Pullback:** the KV advantage only appears at larger L (5.2x vs cacheless at L=128; 1.9x at L=32); at L <= 8 cacheless ONNX / torch are as fast or faster. HF DynamicCache torch is slightly slower than KV ONNX at L >= 32.
- **Memory:** the pullback KV uses 2.2x less peak memory than cacheless ONNX at L=128 (4.4 vs 9.4 GB), and similar to torch. For the forward pass at L=128 the ONNX variants peak around 17 GB versus 9 GB for plain torch and 14.6 GB for torch with cache: ORT's arena is not tighter than torch there; the KV model does not save memory in the forward pass.

## Caveats
- Single GPU type (H100), one model (GPT-2), one sublayer for the pullback. ORT's reported "peak" includes its arena allocation, so it is an upper bound on what is needed.
