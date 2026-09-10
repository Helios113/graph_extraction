"""Find the largest batch size a model's forward pass can run at on the
current GPU without going OOM.

The scripts under `scripts/` (compute_base_jacobian_stats.py,
generate_perturbed_sweep.py, generate_tangent_manifold_distances.py, ...) all
use `cfg.batch` (32 in configs/smoll_jac.toml) as their ONNX subgraphs' fixed
batch dimension -- `construct/model_export.py` exports the base model with
`torch.onnx.export` using a `(cfg.batch, cfg.seq, ...)` dummy input and no
`dynamic_axes`, so the batch size is baked into the graph at export time
rather than being a free runtime knob. Re-exporting (and re-extracting every
sublayer subgraph) at each candidate batch size to measure this directly
would be slow and disk-heavy (~1GB+ of ONNX + external-data files per
export).

Instead, this loads the underlying HF model once in plain PyTorch (the same
`from_pretrained` call `construct/model_export.py` uses, before it ever gets
traced to ONNX) and binary-searches over batch size, running a real forward
pass at `(batch, cfg.seq)` on GPU and catching CUDA OOM. This is fast (no
export/extraction per candidate) and representative of how memory scales
with batch size, though it won't exactly match ONNX Runtime's CUDA EP memory
footprint for the same graph.

Usage:
    uv run python -m scripts.find_max_batch_size configs/smoll_jac.toml
"""

import argparse

import torch
from transformers.models.auto import AutoModelForCausalLM

from config import load_config

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
    try:
        with torch.no_grad():
            model(_dummy_input(cfg, batch))
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        torch.cuda.empty_cache()


def find_max_batch_size(model: torch.nn.Module, cfg, start: int = 1, max_batch: int = 1 << 16) -> int:
    """Doubles `batch` until it no longer fits (or max_batch is hit), then
    binary-searches the (last-good, first-bad) interval for the largest
    batch size that still fits."""
    if not fits(model, cfg, start):
        raise RuntimeError(f"Model does not fit even at batch={start} (seq={cfg.seq})")

    lo = start
    hi = None
    probe = start
    while probe <= max_batch:
        probe *= 2
        if fits(model, cfg, probe):
            lo = probe
        else:
            hi = probe
            break

    if hi is None:
        print(f"warning: reached max_batch={max_batch} without hitting OOM")
        return lo

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(model, cfg, mid):
            lo = mid
        else:
            hi = mid

    return lo


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
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"{cfg.name}: loading {cfg.model} (seq={cfg.seq}) onto {DEVICE}")
    model = _load_model(cfg.model)

    max_batch = find_max_batch_size(model, cfg, start=args.start_batch, max_batch=args.max_batch)

    free, total = torch.cuda.mem_get_info()
    print(
        f"{cfg.name}: max batch size at seq={cfg.seq} is {max_batch} "
        f"(configured batch={cfg.batch}) -- GPU memory free/total: "
        f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB"
    )


if __name__ == "__main__":
    main()
