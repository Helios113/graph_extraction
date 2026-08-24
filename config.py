from typing import Any
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ONNX_DIR = Path("onnx_models")

from enum import Enum


class LossType(Enum):
    MaskedSumLoss = "MaskedSumLoss"
    AdversarialLoss_SquaredDiffLoss = "AdversarialLoss_SquaredDiffLoss"
    AdversarialLoss_NormalizedSquaredDiffLoss = (
        "AdversarialLoss_NormalizedSquaredDiffLoss"
    )
    AdversarialLoss_EnvelopeDiffLoss = "AdversarialLoss_EnvelopeDiffLoss"


class OptimizerType(Enum):
    SGD = "SGD"
    AdamW = "AdamW"


@dataclass
class InputSourceConfig:
    tokenizer: str | None = None
    dataset: str | None = None
    dataset_config: str | None = None
    split: str = "train"
    text_column: str = "text"
    data_batch_size: int = 32 # the numbe of sequence lengths


@dataclass
class SublayerConfig:
    loss_type: LossType
    input_shape: list[int]
    output: str
    input: str
    optimizer_type: OptimizerType | None = None

    def __post_init__(self):
        if isinstance(self.loss_type, str):
            self.loss_type = LossType(self.loss_type)
        if isinstance(self.optimizer_type, str):
            self.optimizer_type = OptimizerType(self.optimizer_type)


_UNSET: Any = object()


@dataclass
class ModelConfig:
    name: str

    model: str        
    vocab_size: int
    batch: int
    seq: int
    d_model: int
    input_name: str
    sublayers: list[SublayerConfig] = field(default_factory=list)
    input_source: InputSourceConfig = field(default_factory=InputSourceConfig)
    jacobian_position: int | None = None
    random_init: bool = False

    @property
    def dir(self) -> Path:
        """This config's own subfolder -- every artifact it produces lives under here,
        except h5 outputs when save_dir is set -- see h5_dir."""
        return ONNX_DIR / self.name
    
    @property
    def base_model_path(self) -> Path:
        return self.dir / "base_model.onnx"
    
    @property
    def sub_block_path(self) -> Path:
        return self.dir / "sub_blocks"

    @property
    def base_model_path(self) -> Path:
        assert self.dir is not None
        return self.dir / "base_model.onnx"

def load_config(path: str) -> ModelConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    input_source = raw.pop("input_source", None)
    if input_source is not None:
        input_source = InputSourceConfig(**input_source)
    else:
        input_source = InputSourceConfig()
    sublayers = [SublayerConfig(**s) for s in raw.pop("sublayers", [])]
    return ModelConfig(**raw, input_source=input_source, sublayers=sublayers)
