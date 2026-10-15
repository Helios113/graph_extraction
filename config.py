import os
import socket
from typing import Any
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ONNX_DIR = Path("onnx_models")


def current_node_name() -> str:
    """The Slurm-allocated node name, or this host's hostname outside Slurm."""
    return os.environ.get("SLURMD_NODENAME") or socket.gethostname()

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
    batch_by_node: dict[str, int] = field(default_factory=dict)
    network_dims: list[int] = field(default_factory=lambda: [512, 32])
    # Name prefix of the decoder blocks in the optimum KV-cache ONNX export, as
    # "{block_prefix}.{layer_idx}/Add" -- e.g. "/transformer/h" for GPT-2,
    # "/model/layers" for Llama-style models. Only needed for the KV-cache path
    # (construct/kv_cache_subgraph.py).
    block_prefix: str | None = None

    def batch_for_node(self, node: str | None = None) -> int:
        """The fixed ONNX batch dim to use on `node` (default: the current
        node, per `current_node_name()`) -- `batch_by_node[node]` if set,
        else the plain `batch` default. Different nodes have different GPUs
        (e.g. mauao's A40 vs ruapehu's H100) and so fit different max batch
        sizes; see scripts/helper/find_max_batch_size.py."""
        node = node or current_node_name()
        return self.batch_by_node.get(node, self.batch)

    @property
    def dir(self) -> Path:
        """This config's own subfolder -- every artifact it produces (base
        model export, subgraphs, pullbacks) lives under here, except h5
        outputs when save_dir is set -- see h5_dir.

        The fixed ONNX batch dim (see `batch_for_node`) is baked into the
        base model at export time, so once a config sets `batch_by_node` for
        more than one node, each node needs its own export -- suffixed here
        by node name so mauao and ruapehu don't clobber/misuse each other's
        cached export. Configs with at most one `batch_by_node` entry keep
        the old unsuffixed path so existing caches stay valid."""
        if len(self.batch_by_node) <= 1:
            return ONNX_DIR / self.name
        return ONNX_DIR / self.name / current_node_name()

    @property
    def base_model_path(self) -> Path:
        return self.dir / "base_model.onnx"

    @property
    def sub_block_path(self) -> Path:
        return self.dir / "sub_blocks"

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
