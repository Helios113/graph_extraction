# KV-cache perturbed sweep: status

Goal: exploit that perturbed-sweep rows sharing (sample_idx, pos) share the
same prefix tokens[:pos], by running that prefix once through a KV-cache
ONNX export and reusing its key/value cache across all n_vocab candidate
continuations at that position, instead of generate_perturbed_sweep.py's
current approach of a full from-scratch forward pass per row.

## Dependency change (done)

`transformers` downgraded from `>=5.14.1` to `>=4.57,<4.58` and
`optimum-onnx` added, in the main project env (pyproject.toml / uv.lock) --
`optimum-onnx==0.1.0` requires `transformers<4.58`, incompatible with the
previous pin. Verified this doesn't break the 3 existing `transformers`
call sites (`construct/model_export.py`, `generate_data/datagen.py`,
`scripts/generate_base_sequences.py` -- all plain `AutoTokenizer`/
`AutoModelForCausalLM` usage, nothing version-specific).

## New files (done, not yet fully working)

- `construct/kv_cache_export.py` -- `ensure_kv_cache_model(cfg)`: shells out
  to `optimum-cli export onnx --task text-generation-with-past` (via
  `python -m optimum.commands.optimum_cli`), lock-guarded like
  `ensure_base_model`. Output: `<cfg.dir>/kv_cache_export/model.onnx`, a
  graph with dynamic batch/seq/past-length axes and
  `past_key_values.{i}.{key,value}` / `present.{i}.{key,value}` I/O for
  every layer.

- `construct/kv_cache_subgraph.py` -- resolves a decoder layer index +
  boundary ("attn_out" = post-attention residual, "mlp_out" = post-MLP
  residual) to the KV-cache graph's tensor names
  (`/model/layers.{i}/Add_output_0` / `/model/layers.{i}/Add_1_output_0`),
  and `ensure_kv_cache_extra_outputs()` extracts+caches a subgraph whose
  outputs are [that sublayer tensor] + every layer's `present.*` (drops
  `logits`, same rationale as `construct.construct.ensure_extra_outputs`).

  **Verified numerically correct** for SimpleStories-V2-11M layer 0: the
  extracted tensors exactly match the old cacheless export's `add_6`/`add_8`
  (max diff ~1e-7, fp32 noise). See git history / conversation for the
  verification script if it needs re-running -- it lived at
  `/tmp/.../scratchpad/verify_layer_mapping2.py` (scratchpad, not saved in
  repo).

- `scripts/generate_perturbed_sweep_kvcache.py` -- the new sweep script.
  Structure: for each `(sample_idx, pos)` group of `n_vocab` rows, run
  `_run_prefix` once (batch=1, `tokens[:pos]`, empty incoming KV) to get
  that prefix's `present.*` KV, then `_run_continuation` in
  `--candidate-batch`-sized chunks (each candidate token as an independent
  length-1 continuation, attending to the prefix KV broadcast across the
  batch via `torch.Tensor.expand` -- zero-copy view, bound straight into
  ORT via IOBinding + dlpack, no host round-trip). Output written via the
  existing `ShardedLayerWriter` unchanged, so format/resume semantics match
  `generate_perturbed_sweep.py`.

  **The two-call approach (prefix once + broadcast-batched continuation) is
  verified correct** against a full forward pass, including the batched
  case with 37 arbitrary candidates against one broadcast prefix (max diff
  ~9e-8) -- see prototype scripts referenced above.

## Known bug -- NOT yet fixed

`_run_prefix` crashes at `pos=0` (the first position swept for every
sample, i.e. every 9th... no, every `group_idx` where `pos == 0`, so once
per sample): ONNX Runtime's `Reshape` op inside the graph's attention block
fails going from a `{1, 0, 128}` tensor to the requested `{1, 0, -1, 64}`
when the incoming sequence length is 0 -- the `-1` inferred dim is
ambiguous when the total element count is already 0. This is a real bug in
how the graph handles an empty `input_ids` (`(1, 0)`), not a shape mismatch
in my binding code. Repro: run the script (or its `_run_prefix` function)
with `prefix_ids = base_seqs[i:i+1, :0]`.

**Not yet investigated**: whether this needs (a) special-casing `pos == 0`
in `_run_perturbed_sweep_kvcache_once` to skip the prefix call entirely and
run the continuation batch directly with an empty/zero-length past KV
(should be equivalent and avoids the empty-input_ids reshape entirely), or
(b) something export-side (optimum-cli flags, or a different task/config)
that produces a graph robust to a zero-length `input_ids` prefix call.
(a) is probably simpler and should be tried first.

## Testing note

Last dry-run attempt monkeypatched the hardcoded `CUDAExecutionProvider`
and GPU IOBinding calls to test on this (GPU-less) dev node -- **user said
not to monkeypatch**. Next attempt at testing should add a real,
non-monkeypatched way to run on CPU (e.g. a `--provider` CLI arg threaded
through both `_run_prefix`/`_run_continuation`'s session creation and
IOBinding vs. plain `sess.run` codepath), or just test on an actual GPU
node (mauao/ruapehu via srun, per CLAUDE.md) instead of trying to make CPU
testing work.

## Also not yet done

- The `_run_continuation` `io_binding.bind_output` calls bind every
  `present.*` output too (required since they're declared graph outputs)
  but the results are never read/used -- wasted compute on every
  continuation call, not fixed since it's not a correctness issue, just an
  efficiency one worth revisiting once the pos=0 bug is fixed and this is
  running end-to-end.
- Cleanup of scratch/dry-run files under `/tmp/claude-*/scratchpad/` --
  not in the repo, safe to ignore/let expire.
- `onnx_models/simple_stories/kv_cache_export/model.onnx` and the extracted
  `extra_outputs.layers_0_mlp_out.onnx` are sitting in the repo (gitignored,
  same as other `*.onnx` build artifacts) from testing -- fine to leave or
  delete, they're regenerable via `ensure_kv_cache_model`/
  `ensure_kv_cache_extra_outputs`.
