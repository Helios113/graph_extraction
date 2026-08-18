from pathlib import Path
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from config import SublayerConfig, load_config
from construct.construct import (
    ensure_base_model,
    ensure_subgraph_pullback,
    ensure_subgraphs,
    generate_subgraph_pullback,
)
from construct.subgraph_extract import generate_union_of_subgraphs



cfg = load_config("configs/qwen_jac_sublayer.toml")
force = False

ensure_base_model(cfg, force=force, verbose=False)


sub_graphs = ensure_subgraphs(cfg, force=force)

output = ensure_subgraph_pullback(
    sub_graphs,
    force=force,
)