import numpy as np
import onnxruntime as ort

def compute_jacobian(
    session: ort.InferenceSession,
    input_data: np.ndarray,
    input_name: str,
    output_name: str,
    mask_name: str,
    gradient_output_name: str,
    output_shape: list[int],
) -> np.ndarray:
    """
    Computes the Jacobian d(output)/d(input) using one-hot masks.
    
    Parameters:
        session: ORT session loaded with the training/gradient graph.
        input_data: Input array of shape (N,) or (1, N).
        input_name: Name of the input tensor.
        output_name: Name of the downstream vector output.
        mask_name: Name of the mask input tensor ("mask").
        gradient_output_name: Name of the output tensor representing d(Loss)/d(input).
        
    Returns:
        jacobian: Array of shape (M, N) where M is output dimension and N is input dimension.
    """
    # 1. Forward run to determine output dimension M
    # (or obtain M directly if known in advance)

    M = int(np.prod(output_shape))  # Flattened output dimension
    
    jacobian_rows = []

    # 2. Iterate through standard basis vectors e_i
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