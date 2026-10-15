# KV-cache test plan (GPT-2)

Everything here runs on a GPU node (ORT CUDA provider, fp32, TF32 off via `tests/conftest.py`) against
`configs/gpt2.toml`. Ground truth is the HF torch model; the second reference is the project's existing
**cacheless** pipeline (static-shape `torch.onnx` export, `ensure_subgraphs`, `ensure_subgraph_pullback`).

```
srun --partition=interactive --nodelist=mauao --gres=gpu:a40:1 --time=01:00:00 --cpus-per-task=8 --mem=64G \
  uv run pytest tests/kv_tests -q
```

Notation: **boundary j** = j-th residual-stream tensor (0 = embeddings, odd = post-attention, even = post-MLP; the
config's `sublayers[j].output`). **Prefix length L** in {0, 1, 8, 32}. **Last token** = position L, the one a
perturbed-sweep row substitutes. 768 candidates per prefix (= d_model, so one batch is one full Jacobian).

Shared code: `kv_helpers.py`. Library code under test: `construct/kv_cache_export.py`, `construct/kv_cache_subgraph.py`,
`scripts/helper/generate_perturbed_sweep_kvcache.py`, `execute/optimization_loop.py`.

| # | File | Status | What it verifies |
|---|------|--------|------------------|
| 1 | `test_kv_export_and_subgraph.py` | existing | optimum KV export I/O signature; logits and present K/V vs HF; prefix+continuation == full pass (several splits); one prefix broadcast over many candidates; the pos=0 limitation (empty `input_ids` fails in ORT) and its zero-length-past workaround; `block_prefix` from config; `boundary_for_index`; boundary subgraph extraction (embed / attn / mlp, first / middle / last layer) vs HF hooks, no `logits`, K/V untouched; extracted add_N names (`add`, `add_2`, ...) == cacheless export |
| 2 | `test_kv_sweep.py` | existing | the real `run_perturbed_sweep_kvcache` end to end (embed, attn_out, mlp_out, last layer): row order, token ids, every activation vs HF with the token substituted; shards/ragged chunks; idempotent resume |
| 3 | `test_kv_prefix_lengths_vs_hf_and_cacheless.py` | new | all 25 boundaries of the last token for L = 0/1/8/32: **KV vs HF vs cacheless**, pairwise |
| 4 | `test_kv_sublayer_subgraphs.py` | new | single-sublayer last-token subgraphs (`ensure_kv_cache_sublayer`; input `hidden_in`, that layer's past K/V) for attention (0->1, 2->3, 22->23) and MLP (1->2, 3->4, 23->24), L = 0/1/8/32: vs HF and vs the cacheless sublayer subgraphs; unused layers' K/V pruned from the inputs |
| 5 | `test_kv_pullback_construction.py` | new | `ensure_kv_cache_sublayer_pullback` for all four loss types x attention/MLP: I/O contract (mask/target/x are `(B,1,d)`; outputs loss, forward out, `hidden_in_grad`), caching, no collisions across batch/loss, squared-diff loss value vs numpy |
| 6 | `test_kv_pullback_jacobian.py` | new | last-token Jacobian `d out[pos]/d in[pos]` (768x768) via the KV MaskedSum pullback vs the **cacheless pullback** vs **torch autograd on HF**, L = 0/1/8/32, 6 sublayers; also non-degeneracy and relative Frobenius error < 1e-3 |
| 7 | `test_kv_timing.py` | new | wall-clock for one 768-candidate group: KV ONNX (prefix + continuation) vs cacheless ONNX vs HF vs HF-with-DynamicCache, L = 0/1/8/32, + projected full-vocab group time; correctness of each variant vs HF; writes `results/timing.{md,json}`; asserts KV faster than cacheless at L = 32 |
| 8 | `test_kv_optimizers_vs_cacheless.py` | new | the adversarial-sample optimisers (SGD, AdamW, GaussNewton, NewtonRaphson) x the 3 adversarial losses (squared diff, normalized, envelope) x attention/MLP x L = 0/8/32: updating **only the last position**, KV pullback and cacheless pullback agree. (a) loss and gradient at fixed y agree (optimiser-independent core claim); (b) SGD/AdamW trajectories agree tightly; (c) GaussNewton/NewtonRaphson: first two steps tight, rest bounded by the method's own float-noise sensitivity; (d) SGD/AdamW reduce the squared-diff loss |

## Known issues the tests pin down

* **ORT CUDA `DivGrad` does not broadcast `dY`** (FIXED in `construct/generate_artifact.py`): the gradient builder
  seeds backprop with a `[1]` ones initializer even when the loss output is `[B, 1]`. CPU `DivGrad` broadcasts it, the
  CUDA one reads only element 0, so the normalized / envelope losses (a Div of two per-row sums) got wrong gradients
  for rows > 0 (zeros in the cacheless pullback). `_expand_loss_seed_grads` now replaces the seed with
  `ConstantOfShape(Shape(out))`; `test_cuda_divgrad_seed_regression` checks CUDA == CPU and all adversarial losses
  run on the GPU. Pullbacks generated before the fix must be regenerated. Repros: `onnx_models/_divrepro/`.
* **pos = 0**: the exported graph cannot run an empty `input_ids`; the sweep skips the prefix call and feeds a
  zero-length past (#1, #2).
* **Secant optimisers** (GaussNewton / NewtonRaphson) divide by a tiny `y - y_prev`, which amplifies ~1e-6
  float differences between any two equivalent graphs; their first secant step already blows the loss up in both
  pipelines. Tested against their own noise sensitivity instead of a fixed tolerance (#8c).
* `execute/optimization_loop.py` picked the loss output by iterating a `set` of output names (could return the
  forward output instead of the loss) and could not feed extra constant inputs; now the loss is the first output and
  `extra_feeds` is accepted.
