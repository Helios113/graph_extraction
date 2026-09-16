"""Exports a KV-cache-enabled ONNX graph for a causal LM via `optimum-cli
export onnx --task text-generation-with-past`, as an alternative to
construct/model_export.py's cacheless, static-batch export.

Unlike export_base_model (use_cache=False, single (batch, seq) forward pass,
static batch dim baked in at export time), the graph produced here takes
`past_key_values.{layer}.{key,value}` as additional inputs (dynamic batch and
sequence-length axes throughout) and returns `present.{layer}.{key,value}` --
a prior call's `present` tensors can be fed back in as the next call's `past`,
so a shared prefix only needs to be run once. See
scripts/generate_perturbed_sweep_kvcache.py for the sweep that exploits this.

optimum-cli is not importable as a clean Python API (it's a CLI-first tool),
so this shells out to it rather than reimplementing the export by hand --
hand-tracing an HF DynamicCache through torch.onnx.export directly runs into
real tracer issues (e.g. SymBool branches in HF's attention masking code,
complex-dtype RoPE tensors the legacy exporter can't lower) that optimum's
export path already handles.
"""

import subprocess
import sys
from pathlib import Path

from config import ModelConfig
from construct.fs_utils import exclusive_lock


def kv_cache_model_dir(cfg: ModelConfig) -> Path:
    """Directory holding this config's KV-cache-enabled export (model.onnx
    plus the tokenizer/config files optimum-cli writes alongside it) --
    kept separate from cfg.base_model_path's cacheless export since they're
    different graphs (different inputs/outputs) produced by a different
    export path."""
    return cfg.dir / "kv_cache_export"


def kv_cache_model_path(cfg: ModelConfig) -> Path:
    return kv_cache_model_dir(cfg) / "model.onnx"


def ensure_kv_cache_model(cfg: ModelConfig, force: bool = False) -> Path:
    """Exports a KV-cache-enabled ONNX graph for cfg.model if it doesn't
    already exist (or always, if force), guarded by a lockfile like
    ensure_base_model. Only supports input_name == "input_ids" (the
    text-generation-with-past task assumes token-id input, not
    inputs_embeds)."""
    if cfg.input_name != "input_ids":
        raise ValueError(
            f"{cfg.name}: KV-cache export only supports input_name == 'input_ids', "
            f"got {cfg.input_name!r}",
        )

    out_dir = kv_cache_model_dir(cfg)
    model_path = kv_cache_model_path(cfg)
    lock_path = out_dir.with_suffix(".kv_cache_export.lock")
    with exclusive_lock(lock_path):
        if not force and model_path.exists():
            print(f"KV-cache model already exists, skipping export -> {model_path}")
            return model_path

        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "optimum.commands.optimum_cli", "export", "onnx",
            "--model", cfg.model,
            "--task", "text-generation-with-past",
            "--trust-remote-code",
            str(out_dir),
        ]
        print(f"Exporting KV-cache model: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"KV-cache model exported -> {model_path}")

    return model_path
