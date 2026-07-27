import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ModelConfig, load_config
from new_code.construct_utils import sublayer_gradient_path, build_single_pair_gradient_graph, ensure_base_model
from new_code.post_utils import (
    _session_options, _example_input_np, _save_activations,
)


def compute_sublayer_jacobian(
    cfg: ModelConfig,
    sublayer_idx: int,
    example_input_np: np.ndarray | None = None,
    session: ort.InferenceSession | None = None,
    disable_progress: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Diagonal block(s) (D_MODEL x D_MODEL) of d(downstream)/d(upstream) for one sublayer,
    computed from that sublayer's own independent gradient graph (see
    build_single_pair_gradient_graph) rather than the full multi-pair union graph
    jacobians/run_model.py's compute_diagonal_jacobians uses.

    Same shape/argument contract as compute_diagonal_jacobians, restricted to one pair:
    diagonal_block is (SEQ, D_MODEL, D_MODEL) if cfg.jacobian_position is None, else
    (1, D_MODEL, D_MODEL). activations is the graph's own forward-pass outputs (e.g. the
    sublayer's input/output tensors), keyed by graph output name, each (batch, seq, d_model).
    """
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
    example_np = example_input_np if example_input_np is not None else _example_input_np(cfg)
    input_batch_np = np.tile(example_np, (cfg.batch,) + (1,) * (example_np.ndim - 1))

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
            {cfg.input_name: input_batch_np, "mask": chunk_mask, "lazy_reset_grad": np.array([True])},
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


def save_sublayer_jacobian(
    cfg: ModelConfig, sublayer_idx: int, diagonal_block: np.ndarray, activations: dict[str, np.ndarray],
) -> None:
    """Saves one sublayer's diagonal Jacobian block(s) (SEQ, D_MODEL, D_MODEL) plus its
    sign/log|det| (np.linalg.slogdet, batched over the SEQ axis) and forward-pass
    `activations`, to cfg.sublayer_jacobians_dir/sublayer{idx}_{input}_{output}.h5.
    """
    sublayer = cfg.sublayers[sublayer_idx]
    tag = f"sublayer{sublayer_idx}_{sublayer.input}_{sublayer.output}"
    sign, logabsdet = np.linalg.slogdet(diagonal_block)

    out_path = cfg.sublayer_jacobians_dir / f"{tag}.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("diagonal_block", data=diagonal_block, compression="gzip")
        f.create_dataset("sign", data=sign, compression="gzip")
        f.create_dataset("logabsdet", data=logabsdet, compression="gzip")
        f.attrs["pair"] = f"{sublayer.input}_{sublayer.output}"
        _save_activations(f, activations)
    print(f"Saved sublayer Jacobian {diagonal_block.shape} -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to a TOML config, e.g. configs/qwen_block0_sublayer_jacobians.toml")
    parser.add_argument("sublayer_idx", type=int, help="index into the config's [[sublayers]] list")
    args = parser.parse_args()
    cfg = load_config(args.config)

    ensure_base_model(cfg)
    build_single_pair_gradient_graph(cfg, args.sublayer_idx)

    print(f"Computing sublayer {args.sublayer_idx} diagonal-block Jacobian...")
    diagonal_block, activations = compute_sublayer_jacobian(cfg, args.sublayer_idx)
    save_sublayer_jacobian(cfg, args.sublayer_idx, diagonal_block, activations)


if __name__ == "__main__":
    main()
