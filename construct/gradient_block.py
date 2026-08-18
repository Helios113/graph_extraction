# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from abc import abstractmethod
from typing import List, Tuple

import onnx

import onnxruntime.training.onnxblock._graph_utils as _graph_utils
import onnxruntime.training.onnxblock._training_graph_utils as _training_graph_utils
import onnxruntime.training.onnxblock.blocks as blocks
import onnxruntime.training.onnxblock.model_accessor as accessor


class GradientBlock(blocks.Block):
    

    def __init__(self, accumulate_gradients = False):
        super().__init__()
        self._requires_grad = set()
        self._frozen_params = set()
        self._parameters = None
        self._training_model = None
        self._eval_model = None
        # Add way to change this
        self._accumulate_gradients = accumulate_gradients

    @abstractmethod
    def build(self, *args, **kwargs):
        """Customize the forward graph for this model by stacking up blocks on top of the inputs to this function.

        This method should be overridden by the subclass. The output of this method should
        be the name of the output from where backpropagation must begin (typically the name
        of the output from the loss function).
        """

    def requires_grad(self, argument_name: str, value: bool = True):
        """Specify whether the argument requires gradient or not.

        The auto-diff will compute the gradient graph for only the arguments that
        require gradient. By default, none of the arguments require gradient.
        The user must explicitly specify which arguments require gradient.

        Args:
            argument_name (str): The name of the argument that require/does not require gradient.
            value (bool): True if the argument requires gradient, False otherwise.
        """
        if value is True:
            self._requires_grad.add(argument_name)
            if argument_name in self._frozen_params:
                self._frozen_params.remove(argument_name)
        else:
            if argument_name in self._requires_grad:
                self._requires_grad.remove(argument_name)
            self._frozen_params.add(argument_name)

    def parameters(self) -> Tuple[List[onnx.TensorProto], List[onnx.TensorProto]]:
        """Trainable as well as non-trainable (frozen) parameters of the model.

        Model parameters that are extracted while building the training model
        are returned by this method.

        Note that the parameters are not known before the training model is
        built. As a result, if this method is invoked before the training model
        is built, an exception will be raised.

        Returns:
            trainable_params (list of onnx.TensorProto): The trainable parameters of the model.
            frozen_params (list of onnx.TensorProto): The non-trainable parameters of the model.

        Raises:
            RuntimeError: If the build method has not been invoked (i.e. the training model has not been built yet).
        """
        if self._parameters is None:
            raise RuntimeError("Please build the training model first before trying to retrieve the parameters.")

        return self._parameters

    def to_model_proto(self) -> Tuple[onnx.ModelProto, onnx.ModelProto]:
        """Returns the training and eval models.

        Once the gradient graph is built, the training and eval models can be retrieved
        by invoking this method.

        Returns:
            training_model (onnx.ModelProto): The training model.
            eval_model (onnx.ModelProto): The eval model.

        Raises:
            RuntimeError: If the build method has not been invoked (i.e. the training model has not been built yet).
        """
        if self._training_model is None or self._eval_model is None:
            raise RuntimeError("Please build the training and eval models first before trying to retrieve them.")
        return self._training_model, self._eval_model

    def __call__(self, *args, **kwargs):
        self.base = accessor._GLOBAL_ACCESSOR.model

        logging.debug("Building training block %s", self.__class__.__name__)

        output = self.build(*args, **kwargs)

        model = self.infer_shapes_on_base()

        _graph_utils.register_graph_outputs(model, output)

        logging.debug("Building gradient graph for training block %s", self.__class__.__name__)

        self._parameters = _training_graph_utils.get_model_parameters(model, self._requires_grad, self._frozen_params)

        self._training_model, self._eval_model = _training_graph_utils.build_gradient_graph(
            model, self._requires_grad, self._frozen_params, output, accessor._GLOBAL_CUSTOM_OP_LIBRARY
        )

        if self._accumulate_gradients:
            logging.debug("Adding gradient accumulation nodes for training block %s", self.__class__.__name__)
            _training_graph_utils.build_gradient_accumulation_graph(self._training_model, self._requires_grad)

        accessor._GLOBAL_ACCESSOR.model.CopyFrom(self._training_model)

        return output
