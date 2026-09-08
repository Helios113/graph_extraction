import numpy as np
import onnxruntime as ort
import tqdm


def compute_jacobian(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    mask_name: str,
    mask_dtype: np.dtype,
    gradient_output_name: list[str],
    output_shape: list[int],
    mode: str = "diagonal",
    seq_positions: list[int] | None = None,
) -> np.ndarray:
    if mode == "full":
        return _compute_jacobian_full(
            session,
            input_data,
            input_name,
            mask_name,
            mask_dtype,
            gradient_output_name,
            output_shape,
        )
    if mode == "diagonal":
        return _compute_jacobian_diagonal(
            session,
            input_data,
            input_name,
            mask_name,
            mask_dtype,
            gradient_output_name,
            output_shape,
            seq_positions=seq_positions,
        )
    raise ValueError(f"mode must be 'full' or 'diagonal', got {mode!r}")


def _compute_jacobian_full(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    mask_name: str,
    mask_dtype: np.dtype,
    gradient_output_name: list[str],
    output_shape: list[int],
) -> np.ndarray:
    M = int(np.prod(output_shape))  # Flattened output dimension

    jacobian_rows = []

    # Iterate through standard basis vectors e_i
    for i in range(M):
        # Create one-hot mask with identical shape and dtype as the output
        mask = np.zeros(M, dtype=mask_dtype)
        mask[i] = 1.0
        mask = mask.reshape(output_shape)

        # Run session to get grad_x (dL_i / dx = dy_i / dx)
        outputs = session.run(
            gradient_output_name,
            {
                input_name: input_data,
                mask_name: mask,
            },
        )
        jacobian_rows.append(outputs)

    jacobian_mat = np.stack(jacobian_rows, axis=0).reshape(
        *output_shape, *input_data.shape
    )

    return jacobian_mat


def _compute_jacobian_diagonal_single(
    session: ort.InferenceSession,
    sample: np.ndarray,
    input_name: str,
    mask_name: str,
    mask_dtype: np.dtype,
    gradient_output_name: list[str],
    compute_batch: int,
    y: int,
    z: int,
    seq_positions: list[int] | None = None,
) -> np.ndarray:
    """Diagonal Jacobian for one real data sample, shape (seq,).

    Uses the model's batch dimension (compute_batch) purely for throughput:
    `sample` is broadcast across compute_batch identical rows, and each row
    of one session.run call computes a different (seq_pos, dim) entry of the
    diagonal. Returns an array of shape (y, z, z).

    seq_positions: sequence positions to actually sweep (e.g. [y - 1] for
    just the last token). Positions not included are left uninitialized in
    the returned array. Defaults to every position in range(y).
    """
    input_data = np.broadcast_to(sample, (compute_batch, *sample.shape))

    swept_positions = seq_positions if seq_positions is not None else range(y)
    positions = [(s, d) for s in swept_positions for d in range(z)]
    diagonal = np.empty((len(gradient_output_name), y, z, z), dtype=mask_dtype)

    for chunk_start in range(0, len(positions), compute_batch):
        chunk = positions[chunk_start : chunk_start + compute_batch]

        mask = np.zeros((compute_batch, y, z), dtype=mask_dtype)
        for slot, (s, d) in enumerate(chunk):
            mask[slot, s, d] = 1.0

        # Last chunk may be smaller than compute_batch; only feed as many
        # broadcast input rows as there are mask slots in use.
        outputs = session.run(
            gradient_output_name,
            {
                input_name: input_data[: len(chunk)],
                mask_name: mask[: len(chunk)],
            },
        )
        for i in range(len(gradient_output_name)):
            for slot, (s, d) in enumerate(chunk):
                diagonal[i, s, d, :] = outputs[i][slot, s, :]

    return diagonal


def _compute_jacobian_diagonal(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    mask_name: str,
    mask_dtype: np.dtype,
    gradient_output_name: list[str],
    output_shape: list[int],
    seq_positions: list[int] | None = None,
) -> np.ndarray:
    """Diagonal Jacobian over a batch of *distinct* data samples.

    input_data: (data_batch, seq, d_model) -- one real sample per row along
    axis 0. This is independent from the model's compute batch dimension
    (output_shape[0]), which is used purely to parallelize the one-hot sweep
    over (seq_pos, dim) positions for a single sample at a time.

    Returns an array of shape (data_batch, seq, d_model, d_model).
    """
    compute_batch, y, z = output_shape

    data_batch = input_data.shape[0]
    diagonals = np.empty(
        (data_batch, len(gradient_output_name), y, z, z),
        dtype=mask_dtype,
    )

    for i in tqdm.tqdm(range(data_batch)):
        diagonals[i] = _compute_jacobian_diagonal_single(
            session,
            input_data[i],
            input_name,
            mask_name,
            mask_dtype,
            gradient_output_name,
            compute_batch,
            y,
            z,
            seq_positions=seq_positions,
        )
    return diagonals
