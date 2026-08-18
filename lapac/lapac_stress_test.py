import numpy as np
import numpy.typing as npt

import scipy.linalg as sla
from scipy.linalg import lapack


def random_orthogonal_det_plus1(n, rng):
    """Random orthogonal matrix with det = +1 exactly (up to fp roundoff in QR)."""
    M = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(M)
    # fix Q's sign ambiguity from QR, then fix overall det sign
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_ill_conditioned(
    n: int,
    cond: float,
    n_negative: int,
    rng: np.random.Generator,
    dtype: npt.DTypeLike,
):

    assert 0 <= n_negative <= n

    mags = np.logspace(0, np.log10(cond), n)  # from 1 to cond

    signs = np.array([-1.0] * n_negative + [1.0] * (n - n_negative))

    rng.shuffle(signs)

    d = signs * mags

    Q1 = random_orthogonal_det_plus1(n, rng)
    Q2 = random_orthogonal_det_plus1(n, rng)
    A = (Q1 * d) @ Q2  # = Q1 @ diag(d) @ Q2

    known_sign = 1 if (n_negative % 2 == 0) else -1
    known_log_abs_det = np.sum(np.log(mags))  # sum log|d_i|

    known_condition_number = np.max(mags) / np.min(mags)

    return A.astype(dtype), known_sign, known_log_abs_det, known_condition_number


def lapack_lu_sign(A):
    """
    Sign of det(A) computed the way LAPACK-based det() implementations do:
    sign = parity(pivot permutation) * prod(sign(diag(U))).
    Uses scipy's raw ?getrf binding directly (dgetrf/sgetrf under the hood).
    Returns (sign, log_abs_det, info).
    """
    getrf = lapack.get_lapack_funcs("getrf", (A,))
    lu, piv, info = getrf(A, overwrite_a=False)
    n = A.shape[0]
    # permutation parity from getrf's interchange-format piv.
    # NB: scipy's compiled getrf binding returns piv 0-indexed (row i was
    # swapped with row piv[i], both 0-based) -- NOT raw Fortran IPIV, which
    # would be 1-indexed. Verified empirically against scipy.linalg.lu_factor,
    # whose piv output is byte-identical to this. Using the 1-indexed formula
    # here silently produces a wrong parity on roughly half of all inputs.
    parity = 1
    for i in range(n):
        if piv[i] != i:
            parity = -parity
    diag = np.diag(lu)
    sign = parity * np.prod(np.sign(diag))
    with np.errstate(divide="ignore"):
        log_abs_det = np.sum(
            np.log(np.abs(diag))
        )  # -inf if a diag entry underflowed to 0
    return sign, log_abs_det, info


def run_stress_test(size, condition_number, number_negatives, rng, dtype):
    A, known_sign, known_log_abs, known_condition_number = make_ill_conditioned(
        size,
        condition_number,
        number_negatives,
        rng,
        dtype=dtype,
    )

    # 1) manual LAPACK getrf-based sign (what det() does internally)
    sign_lapack, log_abs_lapack, info = lapack_lu_sign(A)

    # 2) numpy.linalg.slogdet cross-check (also LAPACK LU, fp64 internally
    #    for numpy; numpy upcasts float32 input to float64 before calling
    #    LAPACK, so this is NOT an independent float32 check)
    sign_np, logdet_np = np.linalg.slogdet(A)

    actual_cond = np.linalg.cond(A)
    print(sign_lapack, sign_np, actual_cond)
    print(known_sign, known_condition_number)
    


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    run_stress_test(
        891,
        1e4,
        300,
        rng,
        np.float32,
    )
