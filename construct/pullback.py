import tempfile
from pathlib import Path

import onnx
import onnxruntime.training.onnxblock as onnxblock
from onnx import helper

from config import LossType
from fs_utils import _strip_last_quantifier
from generate_artifact import generate_artifacts
from loss import (
    AdversarialLoss_EnvelopeDiffLoss,
    AdversarialLoss_NormalizedSquaredDiffLoss,
    AdversarialLoss_SquaredDiffLoss,
    MaskedSumLoss,
)


def generate_loss(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
    output_dtype: int,
    temp_dir: Path | None = None,
) -> tuple[Path, onnxblock.blocks.Block]:
    match loss_type:
        case LossType.MaskedSumLoss:
            assert input_shape is not None
            assert temp_dir is not None, "temp_dir required for MaskedSumLoss"

            target_path = _jacobian_modification(
                sub_graph_path,
                temp_dir,
                input_shape,
                output_dtype,
            )
            tag = f"{sub_graph_path.stem}_pullback"
            loss = MaskedSumLoss(tag, target_path.parent)

        case LossType.AdversarialLoss_SquaredDiffLoss:
            target_path = sub_graph_path
            loss = AdversarialLoss_SquaredDiffLoss()

        case LossType.AdversarialLoss_NormalizedSquaredDiffLoss:
            target_path = sub_graph_path
            loss = AdversarialLoss_NormalizedSquaredDiffLoss()

        case LossType.AdversarialLoss_EnvelopeDiffLoss:
            target_path = sub_graph_path
            loss = AdversarialLoss_EnvelopeDiffLoss()

        case _:
            raise ValueError(f"Invalid loss type: {loss_type}")

    return target_path, loss


def _jacobian_modification(
    path: Path,
    save_dir: Path,
    input_shape: list[int],
    output_dtype: int,
) -> Path:
    save_path = save_dir / f"{path.stem}_mod.onnx"
    data_filename = f"{save_path.name}.data"

    base_model = onnx.load(str(path))
    base_model.graph.input.append(
        helper.make_tensor_value_info(
            "mask",
            output_dtype,
            input_shape,
        ),
    )

    onnx.save(
        base_model,
        str(save_path),
        save_as_external_data=True,
        location=data_filename,
    )
    return save_path


def generate_subgraph_pullback(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
    input_name: str,
    output_name: str,
    base_model_path: Path | None = None,
) -> dict[str, Path]:
    # This will always override -- make sure file protection is elsewhere

    """Builds one sublayer's own independent gradient graph"""
    # ONNX Save removes base_dir info so when saving multiple tensors, we end up with errors
    # Using the absolute path remedies this directly.
    sub_graph_path = sub_graph_path.resolve()
    tag = ".pullback"

    # get the dtype of the output.
    # we load the model
    model_graph = onnx.load(base_model_path or sub_graph_path, load_external_data=False)
    # get the node called output_name
    output_node = next(
        vi for vi in model_graph.graph.value_info if vi.name == output_name
    )
    output_dtype = output_node.type.tensor_type.elem_type
    del model_graph

    with tempfile.TemporaryDirectory() as tmp:
        target_path, loss = generate_loss(
            base_model_path or sub_graph_path,
            loss_type,
            input_shape,
            output_dtype,
            temp_dir=Path(tmp),
        )

        prefix = _strip_last_quantifier(sub_graph_path.stem) + tag
        generated_paths = generate_artifacts(
            target_path,
            save_directory=sub_graph_path.parent,
            requires_grad=[input_name],
            loss_input_names=[output_name],
            loss=loss,
            gradient_model_name=f"{prefix}.onnx",
            save_loss_model=False,
            optimizer_model_name=f"{prefix}_optimizer.onnx",
        )

    return generated_paths
