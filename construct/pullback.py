import tempfile
from pathlib import Path

import onnx
import onnxruntime.training.onnxblock as onnxblock
from onnxruntime.training.onnxblock.optim import SGD,AdamW

from onnx import helper

from config import LossType, OptimizerType

from construct.fs_utils import _strip_last_quantifier
from construct.generate_artifact import generate_artifacts
from construct.loss import (
    AdversarialLoss_EnvelopeDiffLoss,
    AdversarialLoss_NormalizedSquaredDiffLoss,
    AdversarialLoss_SquaredDiffLoss,
    MaskedSumLoss,
)

import numpy as np
from onnx import numpy_helper


def _add_loss_inputs(
    path: Path,
    save_dir: Path,
    tag: str,
    new_inputs: list[tuple[str, int, list[int]]] | None = None,
    trainable: dict[str, np.ndarray] | None = None,
) -> Path:
    """Adds runtime-only inputs and/or makes existing inputs trainable.

    new_inputs: (name, dtype, shape) tuples appended as fresh graph
        inputs, fed every step, never touched by the optimizer.
    trainable: {name: init_value} for graph inputs (new or pre-existing)
        that should also get an initializer, giving them a default value
        the optimizer can update between steps.
    """
    save_path = save_dir / f"{path.stem}_{tag}.onnx"
    data_filename = f"{save_path.name}.data"

    model = onnx.load(str(path))

    for name, dtype, shape in new_inputs or []:
        model.graph.input.append(helper.make_tensor_value_info(name, dtype, shape))

    for name, init_value in (trainable or {}).items():
        existing = next((vi for vi in model.graph.input if vi.name == name), None)
        if existing is None:
            raise ValueError(
                f"{name!r} is not a graph input; add it via new_inputs first",
            )
        dtype = existing.type.tensor_type.elem_type
        model.graph.initializer.append(
            numpy_helper.from_array(
                init_value.astype(onnx.helper.tensor_dtype_to_np_dtype(dtype)),
                name=name,
            ),
        )

    onnx.save(
        model,
        str(save_path),
        save_as_external_data=True,
        location=data_filename,
    )
    return save_path


def generate_loss(
    sub_graph_path: Path,
    loss_type: LossType,
    input_name :str,
    output_name:str,
    input_shape: list[int] | None,
    output_dtype: int,
    temp_dir: Path | None = None,
) -> tuple[Path, onnxblock.blocks.Block, list[str]]:
    names = [output_name]
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
            loss = MaskedSumLoss(target_path.parent)

        case LossType.AdversarialLoss_SquaredDiffLoss:
            assert input_shape is not None
            assert temp_dir is not None, (
                "temp_dir required for AdversarialLoss_SquaredDiffLoss"
            )

            target_path = _add_loss_inputs(
                sub_graph_path,
                temp_dir,
                "target",
                new_inputs=[("target", output_dtype, input_shape)],
            )

            loss = AdversarialLoss_SquaredDiffLoss(target_path.parent)
            names = ["target"] + names

        case (
            LossType.AdversarialLoss_NormalizedSquaredDiffLoss
            | LossType.AdversarialLoss_EnvelopeDiffLoss
        ):
            assert input_shape is not None
            assert temp_dir is not None, "temp_dir required for this loss"

            target_path = _add_loss_inputs(
                sub_graph_path,
                temp_dir,
                "adv_inputs",
                new_inputs=[
                    ("x", output_dtype, input_shape),
                    ("target", output_dtype, input_shape),
                ],
                # trainable={input_name: np.zeros(input_shape, dtype=np.float32)},
            )
            loss_cls = (
                AdversarialLoss_NormalizedSquaredDiffLoss
                if loss_type == LossType.AdversarialLoss_NormalizedSquaredDiffLoss
                else AdversarialLoss_EnvelopeDiffLoss
            )
            loss = loss_cls(target_path.parent)
            names = ["x", input_name, "target"] + names

        case _:
            raise ValueError(f"Invalid loss type: {loss_type}")

    return target_path, loss, names


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
def find_output_node(model_graph, output_name):
    """Search value_info, output, and input for a ValueInfoProto matching
    output_name. Returns the ValueInfoProto (has .name and .type.tensor_type).
    Initializers are excluded — TensorProto has no .type field, so it can't
    be returned as a compatible object; see note below if you need those too.
    """
    all_value_info = (
        list(model_graph.graph.value_info)
        + list(model_graph.graph.output)
        + list(model_graph.graph.input)
    )

    try:
        return next(vi for vi in all_value_info if vi.name == output_name)
    except StopIteration:
        available = [vi.name for vi in all_value_info]
        raise ValueError(f"'{output_name}' not found. Available: {available}")


def generate_subgraph_pullback(
    sub_graph_path: Path,
    loss_type: LossType,
    input_shape: list[int] | None,
    input_name: str,
    output_name: str,
    base_model_path: Path | None = None,
    optimizer_type:  OptimizerType| None = None,
) -> Path:
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
    output_node = find_output_node(model_graph, output_name)
    output_dtype = output_node.type.tensor_type.elem_type
    del model_graph

    with tempfile.TemporaryDirectory() as tmp:
        target_path, loss , output_names= generate_loss(
            base_model_path or sub_graph_path,
            loss_type,
            input_name,
            output_name,
            input_shape,
            output_dtype,
            temp_dir=Path(tmp),
        )
    
        prefix = _strip_last_quantifier(sub_graph_path.stem) + tag

        generated_path = generate_artifacts(
            target_path,
            save_directory=sub_graph_path.parent,
            requires_grad=[input_name],
            loss_input_names=output_names,
            loss=loss,
            gradient_model_name=f"{prefix}.onnx",
            save_loss_model=False,
            optimizer = optimizer_type,
            optimizer_model_name=f"{prefix}_optimizer.onnx",
            
        )

    return generated_path
