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
this script being node-aware. Re-exporting (and re-extracting every sublayer
subgraph) at each candidate batch size to measure this directly would be
slow and disk-heavy (~1GB+ of ONNX + external-data files per export).

Instead, this loads the underlying HF model once in plain PyTorch (the same
`from_pretrained` call `construct/model_export.py` uses, before it ever gets
traced to ONNX) and binary-searches over batch size, running a real forward
pass at `(batch, cfg.seq)` on GPU and catching CUDA OOM. This is fast (no
export/extraction per candidate) and representative of how memory scales
with batch size, though it won't exactly match ONNX Runtime's CUDA EP memory
footprint for the same graph.

Usage:
    uv run python -m scripts.find_max_batch_size configs/tiny_llama.toml
    uv run python -m scripts.find_max_batch_size configs/tiny_llama.toml --update-config
"""

import argparse
import re
import time

import torch
from transformers.models.auto import AutoModelForCausalLM

from config import current_node_name, load_config

DEVICE = torch.device("cuda")


def _load_model(spec: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(spec)
    model.eval()
    model.config.use_cache = False
    return model.to(DEVICE)


def _dummy_input(cfg, batch: int) -> torch.Tensor:
    if cfg.input_name == "input_ids":
        return torch.randint(0, cfg.vocab_size, (batch, cfg.seq), device=DEVICE)
    raise ValueError(
        f"find_max_batch_size only supports input_name == 'input_ids', got {cfg.input_name!r}"
    )


def fits(model: torch.nn.Module, cfg, batch: int) -> bool:
    """True if a forward pass at this batch size completes without OOM."""
    t0 = time.monotonic()
    try:
        with torch.no_grad():
            model(_dummy_input(cfg, batch))
        torch.cuda.synchronize()
        ok = True
    except torch.cuda.OutOfMemoryError as e:
        ok = False
        print(f"[fits] batch={batch}: OOM ({type(e).__name__}: {e})", flush=True)
    else:
        print(f"[fits] batch={batch}: OK", flush=True)
    finally:
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    dt = time.monotonic() - t0
    print(
        f"[fits] batch={batch}: result={ok} time={dt:.1f}s "
        f"free/total={free / 2**30:.2f}/{total / 2**30:.2f} GiB after empty_cache",
        flush=True,
    )
    return ok


def find_max_batch_size(model: torch.nn.Module, cfg, start: int = 1, max_batch: int = 1 << 16) -> int:
    """Doubles `batch` until it no longer fits (or max_batch is hit), then
    binary-searches the (last-good, first-bad) interval for the largest
    batch size that still fits."""
    print(f"[find_max_batch_size] start={start} max_batch={max_batch}", flush=True)
    if not fits(model, cfg, start):
        raise RuntimeError(f"Model does not fit even at batch={start} (seq={cfg.seq})")

    lo = start
    hi = None
    probe = start
    while probe <= max_batch:
        probe *= 2
        print(f"[doubling] probe={probe} (lo={lo}, hi={hi})", flush=True)
        if fits(model, cfg, probe):
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
        if fits(model, cfg, mid):
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
        default=1,
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
    print(f"{cfg.name}: loading {cfg.model} (seq={cfg.seq}) onto {DEVICE} (node={node})")
    model = _load_model(cfg.model)

    max_batch = find_max_batch_size(model, cfg, start=args.start_batch, max_batch=args.max_batch)

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
