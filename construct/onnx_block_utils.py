from pathlib import Path
from typing import TypeVar
from onnxruntime.training.onnxblock.blocks import Block

T = TypeVar("T", bound=Block)

def set_temp_file_name(
    block: T, temp_dir: Path, temp_file_name: str
) -> T:
    block.temp_onnx_file_path = str(temp_dir / temp_file_name)
    block.temp_external_data_file_name = temp_file_name + ".data"
    return block