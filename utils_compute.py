import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from config import ModelConfig
from utils_construct import sublayer_gradient_path
from utils_post import (
    _session_options, sample_jacobian_points, _save_activations,
)


def compute_sublayer_jacobian(
    cfg: ModelConfig,
    sublayer_idx: int,
    example_input_np: np.ndarray | None = None,
    session: ort.InferenceSession | None = None,
    disable_progress: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:


    sublayer = cfg.sublayers[sublayer_idx]
    positions = range(cfg.seq) if cfg.jacobian_position is None else [cfg.jacobian_position % cfg.seq]
    num_elements = len(positions) * cfg.d_model

    if session is None:
        session = ort.InferenceSession(
            str(sublayer_gradient_path(cfg, sublayer_idx)),
            sess_options=_session_options(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    session_output_names = {o.name for o in session.get_outputs()}

    def _f32_output_name(tensor_name: str) -> str:
        cast_name = f"{tensor_name}_f32"
        return cast_name if cast_name in session_output_names else tensor_name

    grad_output_name = _f32_output_name(f"{sublayer.input}_grad")
    grad_related_names = {f"{sublayer.input}_grad", grad_output_name}
    forward_output_names = [
        name for name in session_output_names
        if name not in grad_related_names and f"{name}_f32" not in session_output_names
    ]

    mask_full_np = np.zeros((num_elements, cfg.seq, cfg.d_model), dtype=np.float32)
    for row, position in enumerate(positions):
        mask_full_np[row * cfg.d_model:(row + 1) * cfg.d_model, position, :] = np.eye(cfg.d_model, dtype=np.float32)
    
    sample_np = example_input_np if example_input_np is not None else sample_jacobian_points(cfg, sublayer_idx, 0)
    
    input_batch_np = np.tile(sample_np, (cfg.batch,) + (1,) * (sample_np.ndim - 1))

    if num_elements % cfg.batch != 0:
        raise ValueError(
            f"{cfg.name}: batch ({cfg.batch}) must evenly divide {num_elements} "
            f"({'seq*d_model' if cfg.jacobian_position is None else 'd_model'}) -- "
            "the ORT session's batch dim is fixed at export time, so a remainder chunk can't be run."
        )
    num_chunks = num_elements // cfg.batch
    full_jacobian = np.zeros((num_elements, cfg.seq, cfg.d_model), dtype=np.float32)
    activations: dict[str, np.ndarray] = {}

    for chunk_idx in tqdm(range(num_chunks), desc=f"ORT sublayer{sublayer_idx} graph", disable=disable_progress):
        chunk_mask = mask_full_np[chunk_idx * cfg.batch:(chunk_idx + 1) * cfg.batch]
        fetch_names = [grad_output_name] + (forward_output_names if chunk_idx == 0 else [])
        outputs = session.run(
            fetch_names,
            {sublayer.input: input_batch_np, "mask": chunk_mask},
        )
        out = dict(zip(fetch_names, outputs))
        full_jacobian[chunk_idx * cfg.batch:(chunk_idx + 1) * cfg.batch] = out[grad_output_name]
        if chunk_idx == 0:
            activations = {name: out[name] for name in forward_output_names}

    diagonal_block = np.stack([
        full_jacobian[row * cfg.d_model:(row + 1) * cfg.d_model, position, :]
        for row, position in enumerate(positions)
    ]).astype(np.float32)
    return diagonal_block, activations

