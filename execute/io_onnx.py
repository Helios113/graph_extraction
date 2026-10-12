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


def run_iobinding_gpu_inputs(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, torch.Tensor],
    output_names: list[str],
    device_id: int = 0,
    io_binding: onnxruntime.IOBinding | None = None,
) -> list[torch.Tensor]:
    """
    Multi-input/multi-output version of run_iobinding_gpu: every input is a
    GPU-resident torch tensor (bound in place, no host<->device copy), every
    output stays GPU-resident as a torch tensor via dlpack. For hot loops
    whose inputs are already on the GPU (e.g. a one-hot mask plus a large
    activation batch fed to a pullback graph), where
    run_iobinding_cpu_inputs_gpu_output would round-trip them through host.

    The caller must keep `inputs` alive until this returns. `io_binding`:
    same reuse semantics as run_iobinding_cpu_inputs_gpu_output.

    ORT runs on its own CUDA stream, not torch's current one, so the inputs
    are synchronized first -- otherwise ORT can read a buffer torch has
    queued (but not finished) writing, e.g. a reused buffer refilled for
    this call.
    """
    io_binding = io_binding or session.io_binding()
    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()
    inputs = {name: t.contiguous() for name, t in inputs.items()}
    torch.cuda.synchronize()
    for name, t in inputs.items():
        io_binding.bind_input(
            name=name,
            device_type="cuda",
            device_id=device_id,
            element_type=_torch_to_np_dtype(t.dtype),
            shape=tuple(t.shape),
            buffer_ptr=t.data_ptr(),
        )
    for output_name in output_names:
        io_binding.bind_output(output_name, device_type="cuda", device_id=device_id)

    session.run_with_iobinding(io_binding)

    return [torch.utils.dlpack.from_dlpack(ort_output._ortvalue.to_dlpack()) for ort_output in io_binding.get_outputs()]


def run_iobinding_cpu_input_gpu_output(
    session: onnxruntime.InferenceSession,
    input_name: str,
    input_array: npt.NDArray,
    output_names: list[str],
    device_id: int = 0,
    io_binding: onnxruntime.IOBinding | None = None,
) -> list[torch.Tensor]:
    """
    For host-side inputs (e.g. int64 token ids as plain numpy arrays).
    ORT copies the input host->device once internally; output stays
    GPU-resident as a torch tensor via dlpack.
    """
    return run_iobinding_cpu_inputs_gpu_output(
        session, {input_name: input_array}, output_names, device_id=device_id, io_binding=io_binding
    )


def run_iobinding_cpu_inputs_gpu_output(
    session: onnxruntime.InferenceSession,
    inputs: dict[str, npt.NDArray],
    output_names: list[str],
    device_id: int = 0,
    io_binding: onnxruntime.IOBinding | None = None,
) -> list[torch.Tensor]:
    """
    Same as run_iobinding_cpu_input_gpu_output but for sessions that take
    more than one host-side input (e.g. a data input plus a one-hot mask).
    ORT copies each input host->device once internally; outputs stay
    GPU-resident as torch tensors via dlpack.

    `io_binding`: pass a `session.io_binding()` created once by the caller
    (e.g. in a hot loop making many calls against the same session) to reuse
    it instead of allocating a new one every call -- its previous
    input/output bindings are cleared first, so this is safe to call
    repeatedly with different `inputs`/shapes. If omitted, a fresh one is
    created and used only for this call, as before.
    """
    io_binding = io_binding or session.io_binding()
    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()
    for name, array in inputs.items():
        io_binding.bind_cpu_input(name, array)
    for output_name in output_names:
        io_binding.bind_output(output_name, device_type="cuda", device_id=device_id)

    session.run_with_iobinding(io_binding)

    return [torch.utils.dlpack.from_dlpack(ort_output._ortvalue.to_dlpack()) for ort_output in io_binding.get_outputs()]
