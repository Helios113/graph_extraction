from pathlib import Path

import onnxruntime.training.onnxblock as onnxblock
from construct.onnx_block_utils import set_temp_file_names


class MaskedSumLoss(
    onnxblock.blocks.Block,
):
    """loss = sum(downstream * mask). One per per-pair graph."""

    def __init__(self, temp_dir: Path):
        super().__init__()
        self._mul = onnxblock.blocks.Mul()
        self._reduce_sum = onnxblock.blocks.ReduceSum(keepdims=False)
        set_temp_file_names(self, temp_dir, f"loss")

    def build(self, output):
        masked = self._mul(output, "mask")
        return self._reduce_sum(masked)



class AdversarialLoss_SquaredDiffLoss(
    onnxblock.blocks.Block,
):
    """loss = sum((f(x) - f(y))**2)"""

    def __init__(self, temp_dir: Path):
        super().__init__()
        self._sub = onnxblock.blocks.Sub()
        self._pow = onnxblock.blocks.Pow(2.0)
        self._reduce_sum = onnxblock.blocks.ReduceSum(keepdims=False)
        set_temp_file_names(self, temp_dir, f"loss")

    def build(self, target, output):
        diff = self._sub(target, output)
        return self._reduce_sum(self._pow(diff))


class AdversarialLoss_NormalizedSquaredDiffLoss(
    onnxblock.blocks.Block,
):
    """loss = sum((f(x) - f(y))**2) / sum((x - y)**2)"""

    def __init__(self, temp_dir: Path):
        super().__init__()
        self._sub_inputs = onnxblock.blocks.Sub()
        self._sub_outputs = onnxblock.blocks.Sub()
        self._pow_inputs = onnxblock.blocks.Pow(2.0)
        self._pow_outputs = onnxblock.blocks.Pow(2.0)
        self._reduce_sum_inputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._reduce_sum_outputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._div = onnxblock.blocks.Div()
        set_temp_file_names(self, temp_dir, f"loss")

    def build(self, x, y, target, output):
        input_diff = self._sub_inputs(x, y)
        input_sq_dist = self._reduce_sum_inputs(self._pow_inputs(input_diff))

        output_diff = self._sub_outputs(target, output)
        output_sq_dist = self._reduce_sum_outputs(self._pow_outputs(output_diff))

        return self._div(output_sq_dist, input_sq_dist)


class AdversarialLoss_EnvelopeDiffLoss(
    onnxblock.blocks.Block,
):
    """loss = (||f(y) - f(x)||/|| f(x)||)**2 + 100(ReLU(r - |y - x||)/ ||x||)**2"""

    def __init__(self, temp_dir: Path):
        super().__init__()
        self._sub_inputs = onnxblock.blocks.Sub()
        self._sub_outputs = onnxblock.blocks.Sub()
        self._pow_inputs = onnxblock.blocks.Pow(2.0)
        self._pow_outputs = onnxblock.blocks.Pow(2.0)
        self._reduce_sum_inputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._reduce_sum_outputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._div = onnxblock.blocks.Div()
        set_temp_file_names(self, temp_dir, f"loss")

    def build(self, x, y, target, output):
        input_diff = self._sub_inputs(x, y)
        input_sq_dist = self._reduce_sum_inputs(self._pow_inputs(input_diff))

        output_diff = self._sub_outputs(target, output)
        output_sq_dist = self._reduce_sum_outputs(self._pow_outputs(output_diff))

        return self._div(output_sq_dist, input_sq_dist)
