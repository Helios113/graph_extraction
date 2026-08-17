from pathlib import Path

import torch
from transformers.models.auto import AutoModelForCausalLM

from config import ModelConfig


def _dummy_input(cfg: ModelConfig) -> torch.Tensor:
    if cfg.input_name == "input_ids":
        return torch.randint(0, cfg.vocab_size, (cfg.batch, cfg.seq))
    assert cfg.input_dim is not None, (
        f"{cfg.name}: input_dim required in config for non-input_ids models"
    )
    return torch.randn(cfg.batch, cfg.seq, cfg.input_dim)


def _needs_embeds_kwarg(cfg: ModelConfig) -> bool:
    """`inputs_embeds` isn't the first positional arg in HF forward signatures (input_ids
    is), so it must be exported as a kwarg rather than a positional tensor -- otherwise it
    silently lands in e.g. `past_key_values` instead."""
    return cfg.input_name == "inputs_embeds" and ":" not in cfg.model


def _load_model(spec: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(spec)
    model.eval()
    model.config.use_cache = False
    return model


def _random_init(model: torch.nn.Module, seed: int) -> None:
    """Reinitialize every parameter in-place with fresh random weights (matching each
    tensor's existing shape/dtype), instead of the pretrained/factory-built values
    _load_model produced -- works for any module (HF or custom factory) since it doesn't
    rely on architecture-specific init_weights()/reset_parameters() methods."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(
                torch.randn(param.shape, generator=generator, dtype=param.dtype),
            )


def export_base_model(
    cfg: ModelConfig,
    verbose,
):
    cfg.dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(cfg.model)
    if cfg.random_init:
        _random_init(model, cfg.random_init_seed)
    dummy_input = _dummy_input(cfg)
    if _needs_embeds_kwarg(cfg):
        args = ({"inputs_embeds": dummy_input},)
    else:
        args = (dummy_input,)
    torch.onnx.export(
        model,
        args,
        str(cfg.base_model_path),
        input_names=[cfg.input_name],
        output_names=["output"],
        verbose=verbose,
    )
