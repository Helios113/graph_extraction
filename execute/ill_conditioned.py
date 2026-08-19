import numpy as np


def generate_ill_conditioned_matrix(
    n: int,
    condition_number: float = 1e12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an n x n near-singular matrix with a prescribed condition number.

    Built via SVD: A = U @ diag(s) @ V^T, with singular values log-spaced
    between 1 and 1/condition_number, and U, V random orthogonal matrices.
    """
    if rng is None:
        rng = np.random.default_rng()

    singular_values = np.logspace(0, -np.log10(condition_number), n)

    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))

    return U @ np.diag(singular_values) @ V.T
