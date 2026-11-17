import os
from pathlib import Path

from optimum.exporters.onnx import main_export

from construct.fs_utils import exclusive_lock


def allowed_cpu_count() -> int:
    """CPUs this process may actually use (Slurm cpuset-aware)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1

    except AttributeError:
        return os.cpu_count() or 1


def ensure_kv_cache_model(
    input_name,
    hf_name,
    local_save_dir: Path,
    force: bool = False,
) -> Path:
    """Exports a KV-cache-enabled ONNX graph for cfg.model if it doesn't
    already exist (or always, if force), guarded by a lockfile like
    ensure_base_model. Only supports input_name == "input_ids" (the
    text-generation-with-past task assumes token-id input, not
    inputs_embeds)."""
    if input_name != "input_ids":
        raise ValueError(
            f"{hf_name}: KV-cache export only supports input_name == 'input_ids', "
            f"got {input_name!r}",
        )

    out_dir = local_save_dir / "kv_cache_export"
    model_path = out_dir / "model.onnx"

    lock_path = out_dir.with_suffix(".kv_cache_export.lock")
    with exclusive_lock(lock_path):
        if not force and model_path.exists():
            print(f"KV-cache model already exists, skipping export -> {model_path}")
            return model_path

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Exporting KV-cache model: {hf_name} -> {out_dir}", flush=True)

        # Set thread count Doesn't work now
        os.environ["ORT_DISABLE_THREAD_AFFINITY"] = "1"
        os.environ["OMP_NUM_THREADS"] = str(allowed_cpu_count)
        print(os.environ["ORT_DISABLE_THREAD_AFFINITY"])
        print(os.environ["OMP_NUM_THREADS"])

        main_export(
            model_name_or_path=hf_name,
            output=out_dir,
            task="text-generation-with-past",
            trust_remote_code=True,
        )
        print(f"KV-cache model exported -> {model_path}")

    return model_path
