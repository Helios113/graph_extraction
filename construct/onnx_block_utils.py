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



def set_temp_file_names(block: T, temp_dir: Path, tag: str) -> T:
    """Recursively stamp temp-file paths onto a block and its child blocks."""
    set_temp_file_name(block, temp_dir, f"temp_{tag}.onnx")
    for attr_name, attr_value in vars(block).items():
        if isinstance(attr_value, Block) and attr_value is not block:
            child_tag = attr_name.lstrip("_")
            set_temp_file_name(attr_value, temp_dir, f"temp_{child_tag}_{tag}.onnx")
    return block
