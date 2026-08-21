# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from pathlib import Path
from enum import Enum
from construct.onnx_block_utils import set_temp_file_names
from construct.gradient_block import GradientBlock
from onnxruntime.training.onnxblock.optim import AdamW, SGD
from config import OptimizerType
import onnx

from onnxruntime.training import onnxblock


def generate_artifacts(
    model_path: Path,
    save_directory: Path,
    requires_grad: list[str] | None = None,
    frozen_params: list[str] | None = None,
    loss: onnxblock.Block | None = None,
    optimizer: OptimizerType | None = None,
    gradient_model_name: str = "gradients_model.onnx",
    loss_model_name: str = "loss_model.onnx",
    optimizer_model_name: str = "optimizer_model.onnx",
    save_loss_model: bool = False,
    additional_output_names: list[str] | None = None,
    loss_input_names: list[str] | None = None,
) -> Path:

    loaded_model = onnx.load(model_path)

    if loss is None:
        loss = onnxblock.blocks.PassThrough()
        logging.info(
            "No loss block provided. Loss node will not be added to the graph."
        )

    class _GradientsBlock(GradientBlock):
        def __init__(self, _loss, _loss_input_names=None, accumulate_gradients=False):
            super().__init__(accumulate_gradients=accumulate_gradients)
            self._loss = _loss
            self._loss_input_names = _loss_input_names

        def build(self, *inputs_to_loss):
            # If loss_input_names is passed, only pass the specified input names to the loss function.

            if self._loss_input_names:
                inputs_to_loss = self._loss_input_names

            if additional_output_names:
                # If additional output names is not a list, raise an error
                if not isinstance(additional_output_names, list):
                    raise RuntimeError(
                        f"Unknown type provided for additional output names {type(additional_output_names)}. "
                        "Expected additional output names to be a list of strings."
                    )

                loss_output = self._loss(*inputs_to_loss)
                if isinstance(loss_output, tuple):
                    return (*loss_output, *tuple(additional_output_names))
                else:
                    return (loss_output, *tuple(additional_output_names))
            return self._loss(*inputs_to_loss)

    gradients_block = set_temp_file_names(
        _GradientsBlock(
            loss,
            loss_input_names,
            accumulate_gradients=optimizer is not None,
        ),
        temp_dir=save_directory,
        tag="gradient",
    )

    if (
        requires_grad is not None
        and frozen_params is not None
        and set(requires_grad).intersection(set(frozen_params))
    ):
        raise RuntimeError(
            "A parameter cannot be frozen and require gradient computation at the same "
            f"time {set(requires_grad).intersection(set(frozen_params))}",
        )

    if requires_grad is not None:
        for arg in requires_grad:
            gradients_block.requires_grad(arg)

    if frozen_params is not None:
        for arg in frozen_params:
            gradients_block.requires_grad(arg, False)

    with onnxblock.base(loaded_model, str(model_path)):
        _ = gradients_block(*[output.name for output in loaded_model.graph.output])
        gradients_model, loss_model = gradients_block.to_model_proto()
        model_params = gradients_block.parameters()

    def _save_with_data_file(model_proto, model_path, data_file_name):
        onnx.save(
            model_proto,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_file_name,
        )

    gradient_model_path = save_directory / gradient_model_name
    if gradient_model_path.exists():
        logging.info(
            "gradients model path %s already exists. Overwriting.", gradient_model_path
        )
    _save_with_data_file(
        gradients_model, gradient_model_path, f"{gradient_model_name}.data"
    )
    onnx.checker.check_model(gradient_model_path)
    return gradient_model_path
    return_dict = {"gradient": gradient_model_path}

    # logging.info("Saved gradients model to %s", gradient_model_path)

    # if save_loss_model:
    #     loss_model_path = save_directory / loss_model_name
    #     if loss_model_path.exists():
    #         logging.info(
    #             "loss model path %s already exists. Overwriting.", loss_model_path
    #         )
    #     _save_with_data_file(loss_model, loss_model_path, f"{loss_model_name}.data")
    #     onnx.checker.check_model(loss_model_path)

    #     logging.info("Saved loss model to %s", loss_model_path)
    #     return_dict["loss"] = loss_model_path

    # else:
    #     logging.info("save_loss_model is False. Skipping loss model generation.")

    # # If optimizer is not specified, skip creating the optimizer model
    # if optimizer is None:
    #     logging.info("No optimizer enum provided. Skipping optimizer model generation.")
    #     return return_dict

    # opset_version = None
    # for domain in loaded_model.opset_import:
    #     if domain.domain == "" or domain.domain == "ai.onnx":
    #         opset_version = domain.version
    #         break
    # optim_dict = {
    #     "SGD": SGD,
    #     "AdamW": AdamW,
    # }
    # optim_block = optim_dict[optimizer.value]()
    # with onnxblock.empty_base(opset_version=opset_version):
    #     _ = optim_block(model_params)
    #     optim_model = optim_block.to_model_proto()

    # optimizer_model_path = save_directory / optimizer_model_name
    # _save_with_data_file(
    #     optim_model,
    #     optimizer_model_path,
    #     f"{optimizer_model_name}.data",
    # )
    # onnx.checker.check_model(optimizer_model_path)

    # logging.info("Saved optimizer model to %s", optimizer_model_path)
    # return_dict["optimizer"] = optimizer_model_path

    # return return_dict
