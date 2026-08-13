from pathlib import Path

import onnx
import onnxruntime.training.onnxblock as onnxblock


def _set_temp_file_name(
    block: onnxblock.blocks.Block, temp_dir: Path, temp_file_name: str
) -> onnxblock.blocks.Block:
    # Block.__init__ hardcodes "temp.onnx" for every instance, so sibling blocks called
    # back-to-back under the same has_path build collide on temp.onnx.data (onnx.save
    # refuses to overwrite it). Give each block its own scratch file name instead, under
    # temp_dir (cfg.union_artifact_dir -- per-config, not the shared process cwd): two
    # configs/jobs running at once can otherwise collide on an identical tag (e.g. two GPT-2
    # variants both building "pair0_add_1_add_4"), and clear_stale_temp_files's cleanup
    # sweep would delete a concurrent job's in-progress scratch file if it swept a shared dir.
    block.temp_onnx_file_path = str(temp_dir / temp_file_name)
    block.temp_external_data_file_name = temp_file_name + ".data"
    return block


class MaskedSumLoss(
    onnxblock.blocks.Block,
):
    """loss = sum(downstream * mask). One per per-pair graph."""

    def __init__(self, tag: str, temp_dir: Path):
        super().__init__()
        _set_temp_file_name(self, temp_dir, f"temp_loss_{tag}.onnx")
        self._cast = _set_temp_file_name(
            onnxblock.blocks.Cast(onnx.TensorProto.FLOAT),
            temp_dir,
            f"temp_cast_{tag}.onnx",
        )
        self._mul = _set_temp_file_name(
            onnxblock.blocks.Mul(), temp_dir, f"temp_mul_{tag}.onnx"
        )
        self._reduce_sum = _set_temp_file_name(
            onnxblock.blocks.ReduceSum(keepdims=False),
            temp_dir,
            f"temp_reduce_sum_{tag}.onnx",
        )

    def build(self, downstream_name):
        downstream_f32 = self._cast(downstream_name)
        masked = self._mul(downstream_f32, "mask")
        return self._reduce_sum(masked)


class SquaredDiffLoss(
    onnxblock.blocks.Block,
):
    """loss = sum((f(x) - f(y))**2). Unlike MaskedSumLoss, no `mask` input -- every element of
    both copies' outputs contributes to the loss directly."""

    def __init__(self):
        super().__init__()
        self._cast_a = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_b = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._sub = onnxblock.blocks.Sub()
        self._pow = onnxblock.blocks.Pow(2.0)
        self._reduce_sum = onnxblock.blocks.ReduceSum(keepdims=False)

    def build(self, a_name, b_name):
        diff = self._sub(self._cast_a(a_name), self._cast_b(b_name))
        return self._reduce_sum(self._pow(diff))


class NormalizedSquaredDiffLoss(
    onnxblock.blocks.Block,
):
    """loss = sum((f(x) - f(y))**2) / sum((x - y)**2) -- output squared-distance normalized
    by input squared-distance, so shrinking ||x - y|| doesn't trivially shrink the loss on
    its own; the optimizer is rewarded only for closing the output gap faster than the input
    gap grows."""

    def __init__(self):
        super().__init__()
        self._cast_x = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_y = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_a = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._cast_b = onnxblock.blocks.Cast(onnx.TensorProto.FLOAT)
        self._sub_inputs = onnxblock.blocks.Sub()
        self._sub_outputs = onnxblock.blocks.Sub()
        self._pow_inputs = onnxblock.blocks.Pow(2.0)
        self._pow_outputs = onnxblock.blocks.Pow(2.0)
        self._reduce_sum_inputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._reduce_sum_outputs = onnxblock.blocks.ReduceSum(keepdims=False)
        self._div = onnxblock.blocks.Div()

    def build(self, x_name, y_name, a_name, b_name):
        input_diff = self._sub_inputs(self._cast_x(x_name), self._cast_y(y_name))
        input_sq_dist = self._reduce_sum_inputs(self._pow_inputs(input_diff))

        output_diff = self._sub_outputs(self._cast_a(a_name), self._cast_b(b_name))
        output_sq_dist = self._reduce_sum_outputs(self._pow_outputs(output_diff))

        return self._div(output_sq_dist, input_sq_dist)
