"""Find the largest batch size a model's forward pass can run at on the
current node's GPU without going OOM.

The scripts under `scripts/` (compute_base_jacobian_stats.py,
generate_perturbed_sweep.py, generate_tangent_manifold_distances.py, ...) all
use `cfg.batch_for_node()` (see config.py) as their ONNX subgraphs' fixed
batch dimension -- `construct/model_export.py` exports the base model with
`torch.onnx.export` using a `(batch, cfg.seq, ...)` dummy input and no
`dynamic_axes`, so the batch size is baked into the graph at export time
rather than being a free runtime knob. Different nodes (e.g. mauao's A40 vs
ruapehu's H100) fit different max batch sizes, hence `batch_for_node` and
this script being node-aware.

This probes the *actual* ONNX Runtime graph the real workload runs --
export_base_model + ensure_extra_outputs with every configured sublayer
output requested, exactly like generate_perturbed_sweep.py /
generate_base_activations.py -- rather than a plain PyTorch forward pass.
That distinction matters: retaining every sublayer's activation
simultaneously (and, before ensure_extra_outputs started truncating the
graph, the unused final logits tensor too) has a different memory profile
than a single forward pass computing only the final output, so a batch size
measured against plain PyTorch can be too large for the real ONNX graph and
OOM there (see the `node_linear_4` crash this replaced). Each candidate
batch size needs its own export (batch is baked into the graph, no
dynamic_axes), so this re-exports per probe under a disposable cfg.name
("<name>_batch_probe_<node>") that's deleted after every probe -- slower
per probe than a plain torch forward pass, but it measures the thing that
actually has to fit. The HF model itself is loaded once, up front, and
reused across every probe (export_base_model accepts a pre-loaded model)
rather than reloaded from disk on each of the ~30 probes a full
doubling+bisection search needs -- for a multi-GB model that reload cost
alone would otherwise dominate wall time.

Usage:
    uv run python -m scripts.helper.find_max_batch_size configs/tiny_llama.toml
    uv run python -m scripts.helper.find_max_batch_size configs/tiny_llama.toml --update-config
"""

import argparse
import dataclasses
import re
import shutil
import time

import numpy as np
import onnxruntime as ort
import torch

from config import ModelConfig, current_node_name, load_config
from construct.construct import ensure_base_model, ensure_extra_outputs
from construct.model_export import _load_model, _random_init


def _probe_cfg(cfg: ModelConfig, node: str, batch: int) -> ModelConfig:
    """A disposable copy of `cfg` pinned to `batch` on `node`, exporting
    under its own onnx_models/<name>_batch_probe_<node>/... directory
    (cfg.dir derives from cfg.name) so repeated probe exports never touch
    the real cached export at onnx_models/<cfg.name>/..., and concurrent
    finder runs on different nodes never contend for the same probe path."""
    return dataclasses.replace(cfg, name=f"{cfg.name}_batch_probe_{node}", batch_by_node={node: batch})


def _is_oom_error(e: Exception) -> bool:
    msg = str(e)
    return "Failed to allocate memory" in msg or "OutOfMemoryError" in type(e).__name__


def _load_probe_model(cfg: ModelConfig):
    """Loads cfg.model from HF once, applying random_init the same way
    export_base_model would -- the caller passes the result into every
    fits() call for this search so the (potentially multi-GB) weights are
    read from disk/HF exactly once rather than once per probe."""
    model = _load_model(cfg.model)
    if cfg.random_init:
        _random_init(model, cfg.random_init_seed)
    return model


def fits(
    cfg: ModelConfig, node: str, batch: int, extra_outputs: list[str], model,
) -> bool:
    """True if a forward pass at this batch size completes on the real
    (extra-outputs) ONNX graph without OOM. Exports fresh at `batch` (the
    graph's batch dim can't be changed after export) and always cleans up
    the probe export directory afterward, success or failure.

    `model`: already-loaded (see `_load_probe_model`), reused across every
    probe instead of reloading `cfg.model` from HF on each one."""
    probe = _probe_cfg(cfg, node, batch)
    shutil.rmtree(probe.dir, ignore_errors=True)
    t0 = time.monotonic()
    try:
        ensure_base_model(probe, force=True, verbose=False, model=model)
        model_path = ensure_extra_outputs(probe, extra_outputs, force=True)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.intra_op_num_threads = 8
        sess = ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=["CUDAExecutionProvider"],
        )
        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]

        io_binding = sess.io_binding()
        dummy = np.random.randint(0, cfg.vocab_size, (batch, cfg.seq)).astype(np.int64)
        io_binding.bind_cpu_input(input_name, dummy)
        for name in output_names:
            io_binding.bind_output(name, device_type="cuda", device_id=0)
        sess.run_with_iobinding(io_binding)
        ok = True
    except RuntimeError as e:
        if not _is_oom_error(e):
            raise
        ok = False
        print(f"[fits] batch={batch}: OOM ({e})", flush=True)
    else:
        print(f"[fits] batch={batch}: OK", flush=True)
    finally:
        shutil.rmtree(probe.dir, ignore_errors=True)

    dt = time.monotonic() - t0
    print(f"[fits] batch={batch}: result={ok} time={dt:.1f}s", flush=True)
    return ok


def find_max_batch_size(
    cfg: ModelConfig,
    node: str,
    extra_outputs: list[str],
    model,
    start: int = 1,
    max_batch: int = 1 << 16,
) -> int:
    """Doubles `batch` until it no longer fits (or max_batch is hit), then
    binary-searches the (last-good, first-bad) interval for the largest
    batch size that still fits. `model`: see `_load_probe_model`."""
    print(f"[find_max_batch_size] start={start} max_batch={max_batch}", flush=True)
    if not fits(cfg, node, start, extra_outputs, model):
        raise RuntimeError(f"Model does not fit even at batch={start} (seq={cfg.seq})")

    lo = start
    hi = None
    probe = start
    while probe <= max_batch:
        probe *= 2
        print(f"[doubling] probe={probe} (lo={lo}, hi={hi})", flush=True)
        if fits(cfg, node, probe, extra_outputs, model):
            lo = probe
        else:
            hi = probe
            break

    if hi is None:
        print(f"warning: reached max_batch={max_batch} without hitting OOM", flush=True)
        return lo

    print(f"[doubling done] bracket=({lo}, {hi})", flush=True)

    n_iter = 0
    while hi - lo > 1:
        n_iter += 1
        mid = (lo + hi) // 2
        print(f"[bisect {n_iter}] lo={lo} hi={hi} mid={mid}", flush=True)
        if fits(cfg, node, mid, extra_outputs, model):
            lo = mid
        else:
            hi = mid
        print(f"[bisect {n_iter}] -> lo={lo} hi={hi}", flush=True)

    print(f"[find_max_batch_size] converged: lo={lo} hi={hi} in {n_iter} bisection step(s)", flush=True)
    return lo


def _update_batch_by_node(config_path: str, node: str, batch: int) -> None:
    """Set `batch_by_node.<node> = <batch>` in the TOML file at `config_path`,
    editing the text directly (rather than parsing with tomllib and
    rewriting with a TOML-writer library) so hand-written comments and
    formatting elsewhere in the file survive untouched.

    If a `[batch_by_node]` table already exists, its `node = ...` line is
    replaced in place (or a new line appended to the table if that node
    isn't there yet). Otherwise a new `[batch_by_node]` table is appended at
    the end of the file.
    """
    with open(config_path) as f:
        text = f.read()

    table_re = re.compile(r"^\[batch_by_node\]\s*$", re.MULTILINE)
    table_match = table_re.search(text)

    if table_match is None:
        if not text.endswith("\n"):
            text += "\n"
        text += f"\n[batch_by_node]\n{node} = {batch}\n"
        with open(config_path, "w") as f:
            f.write(text)
        return

    table_start = table_match.end()
    next_table_match = re.search(r"^\[", text[table_start:], re.MULTILINE)
    table_end = table_start + next_table_match.start() if next_table_match else len(text)
    table_body = text[table_start:table_end]

    node_line_re = re.compile(rf"^{re.escape(node)}\s*=\s*\d+\s*$", re.MULTILINE)
    node_match = node_line_re.search(table_body)
    if node_match is not None:
        new_body = table_body[: node_match.start()] + f"{node} = {batch}" + table_body[node_match.end():]
    else:
        new_body = table_body.rstrip("\n") + f"\n{node} = {batch}\n"

    text = text[:table_start] + new_body + text[table_end:]
    with open(config_path, "w") as f:
        f.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str, help="Path to a toml ModelConfig")
    parser.add_argument(
        "--start-batch",
        type=int,
        default=5000,
        help="Smallest batch size to try (fails loudly if even this OOMs)",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1 << 16,
        help="Upper bound to stop doubling at if nothing OOMs first",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Write the discovered max batch size into [batch_by_node] for the "
        "current node ($SLURMD_NODENAME, or hostname) directly in the config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    node = current_node_name()
    extra_outputs = [sublayer.output for sublayer in cfg.sublayers]
    print(
        f"{cfg.name}: probing {cfg.model} (seq={cfg.seq}, node={node}) "
        f"with extra_outputs={extra_outputs}"
    )

    # A prior run killed mid-probe (SIGKILL/scancel can't be caught, so
    # fits()'s own finally-block cleanup never ran) can leave this probe
    # directory behind with a stale base_model_extra_outputs.onnx -- e.g.
    # built by an older ensure_extra_outputs before a graph-shape change.
    # Wipe it up front so this run never mistakes that leftover for its own.
    stale_probe_dir = _probe_cfg(cfg, node, args.start_batch).dir
    shutil.rmtree(stale_probe_dir, ignore_errors=True)

    print(f"{cfg.name}: loading {cfg.model} once, reused across every probe", flush=True)
    model = _load_probe_model(cfg)

    max_batch = find_max_batch_size(
        cfg, node, extra_outputs, model, start=args.start_batch, max_batch=args.max_batch,
    )

    free, total = torch.cuda.mem_get_info()
    configured_batch = cfg.batch_for_node(node)
    print(
        f"{cfg.name}: max batch size at seq={cfg.seq} on node {node!r} is {max_batch} "
        f"(configured batch={configured_batch}) -- GPU memory free/total: "
        f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB"
    )

    if args.update_config:
        _update_batch_by_node(args.config, node, max_batch)
        print(f"{cfg.name}: wrote batch_by_node.{node} = {max_batch} -> {args.config}")


if __name__ == "__main__":
    main()
