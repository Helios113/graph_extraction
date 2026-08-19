import numpy as np
import onnxruntime as ort


def compute_jacobian(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    mask_name: str,
    gradient_output_name: str,
    output_shape: list[int],
    mode: str = "diagonal",
) -> np.ndarray:
    if mode == "full":
        return _compute_jacobian_full(
            session, input_data, input_name, mask_name, gradient_output_name, output_shape,
        )
    if mode == "diagonal":
        return _compute_jacobian_diagonal(
            session, input_data, input_name, mask_name, gradient_output_name, output_shape,
        )
    raise ValueError(f"mode must be 'full' or 'diagonal', got {mode!r}")


def _compute_jacobian_full(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    mask_name: str,
    gradient_output_name: str,
    output_shape: list[int],
) -> np.ndarray:
    M = int(np.prod(output_shape))  # Flattened output dimension

    jacobian_rows = []

    # Iterate through standard basis vectors e_i
    for i in range(M):
        # Create one-hot mask with identical shape and dtype as the output
        mask = np.zeros(M, dtype=input_data.dtype)
        mask[i] = 1.0
        mask = mask.reshape(output_shape)

        # Run session to get grad_x (dL_i / dx = dy_i / dx)
        outputs = session.run(
            [gradient_output_name],
            {
                input_name: input_data,
                mask_name: mask,
            },
        )
        grad_x = outputs[0]  # Shape matches input_data.shape
        jacobian_rows.append(grad_x)

    jacobian_mat = np.stack(jacobian_rows, axis=0).reshape(*output_shape, *input_data.shape)

    return jacobian_mat


def _compute_jacobian_diagonal(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    mask_name: str,
    gradient_output_name: str,
    output_shape: list[int],
) -> np.ndarray:
    x, y, z = output_shape
    if input_data.shape != (x, y, z):
        raise ValueError(
            f"input_data.shape {input_data.shape} must match output_shape {output_shape}",
        )
    if not np.all(input_data == input_data[:1]):
        raise ValueError(
            "mode='diagonal' requires input_data's batch slots to all be identical "
            "(the real input duplicated across the batch axis for throughput) -- "
            "got distinct values across the batch dimension",
        )

    positions = [(s, d) for s in range(y) for d in range(z)]  # y*z (seq_pos, dim) pairs
    diagonal = np.empty((x, y, z, z), dtype=input_data.dtype)

    for chunk_start in range(0, len(positions), x):
        chunk = positions[chunk_start : chunk_start + x]

        mask = np.zeros((x, y, z), dtype=input_data.dtype)
        for slot, (s, d) in enumerate(chunk):
            mask[slot, s, d] = 1.0

        outputs = session.run(
            [gradient_output_name],
            {
                input_name: input_data,
                mask_name: mask,
            },
        )
        grad_x = outputs[0]  # (x, y, z)

        for slot, (s, d) in enumerate(chunk):
            # grad_x[slot] = d(output[slot, s, d]) / d(input[slot, :, :]); every
            # batch slot holds the same input, so this is also d(output at
            # position s)/d(input at position s) for every other slot -- read it
            # off slot's own row and store it for every batch entry.
            diagonal[:, s, d, :] = grad_x[slot, s, :]

    return diagonal
