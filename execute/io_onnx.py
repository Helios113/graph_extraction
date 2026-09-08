import ml_dtypes
import numpy as np
import numpy.typing as npt
import onnxruntime
import torch


def _torch_to_np_dtype(dtype: torch.dtype) -> type:
    mapping = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.bfloat16: ml_dtypes.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(f"No numpy/ml_dtypes mapping registered for {dtype}")
    return mapping[dtype]


def run_iobinding_gpu(
    session: onnxruntime.InferenceSession,
    input_name: str,
    input_tensor: torch.Tensor,
    output_name: str,
    device_id: int = 0,
) -> torch.Tensor:
    """
    Run an onnxruntime session with a GPU-resident torch input tensor,
    returning a GPU-resident torch output tensor.
    Uses IOBinding + dlpack to avoid host<->device copies on input and output.
    """
    input_tensor = input_tensor.contiguous()
    io_binding = session.io_binding()

    io_binding.bind_input(
        name=input_name,
        device_type="cuda",
        device_id=device_id,
        element_type=_torch_to_np_dtype(input_tensor.dtype),
        shape=tuple(input_tensor.shape),
        buffer_ptr=input_tensor.data_ptr(),
    )
    io_binding.bind_output(output_name, device_type="cuda", device_id=device_id)

    session.run_with_iobinding(io_binding)

    ort_output = io_binding.get_outputs()[0]
    return torch.utils.dlpack.from_dlpack(ort_output._ortvalue.to_dlpack())


def run_iobinding_cpu_input_gpu_output(
    session: onnxruntime.InferenceSession,
    input_name: str,
    input_array: npt.NDArray,
    output_name: str,
    device_id: int = 0,
) -> torch.Tensor:
    """
    For host-side inputs (e.g. int64 token ids as plain numpy arrays).
    ORT copies the input host->device once internally; output stays
    GPU-resident as a torch tensor via dlpack.
    """
    io_binding = session.io_binding()
    io_binding.bind_cpu_input(input_name, input_array)
    io_binding.bind_output(output_name, device_type="cuda", device_id=device_id)

    session.run_with_iobinding(io_binding)

    ort_output = io_binding.get_outputs()[0]
    return torch.utils.dlpack.from_dlpack(ort_output._ortvalue.to_dlpack())
