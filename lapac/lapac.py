import numpy as np
import scipy.linalg as sla
from scipy.linalg import lapack
import numpy as np
import tqdm
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
   
    parity = 1
    for i in range(n):
        if piv[i] != i:
            parity = -parity
    diag = np.diag(lu)
    sign = parity * np.prod(np.sign(diag))
    with np.errstate(divide="ignore"):
        log_abs_det = np.sum(
            np.log(np.abs(diag)),
        )  # -inf if a diag entry underflowed to 0
    return sign, log_abs_det, info


def _random_orthogonal_det_plus1(n, rng):
    """Random orthogonal matrix with det = +1 exactly (up to fp roundoff in QR)."""
    M = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def generate_matrix(n: int, cond: float = 1e6, seed: int = 100, verbose: bool = True):
    """
    Builds A = Q1 @ diag(d) @ Q2 with Q1, Q2 random orthogonal (det=+1).
    Since Q1, Q2 are orthogonal, the singular values of A are exactly |d_i|,
    so the smallest singular value and cond(A) = max|d_i|/min|d_i| are pinned
    exactly by construction (up to fp roundoff), independent of the LAPACK
    factorization being tested.
    Returns (A, target_sign).
    """
    base_numer = 1e-3
    target_sign = 1

    rng = np.random.default_rng(seed)

    # Log-spaced singular values from base_numer to base_numer*cond, so the
    # smallest singular value is pinned at base_numer (1e-3) and
    # max|d_i| / min|d_i| == cond exactly (up to fp roundoff).
    mags = np.logspace(np.log10(base_numer), np.log10(base_numer * cond), n)
    signs = rng.choice([-1.0, 1.0], size=n)
    # Force the product of signs to match target_sign by fixing the last one.
    current_sign = np.prod(signs[:-1])
    signs[-1] = target_sign * current_sign
    d = signs * mags

    Q1 = _random_orthogonal_det_plus1(n, rng)
    Q2 = _random_orthogonal_det_plus1(n, rng)
    A = (Q1 * d) @ Q2  # = Q1 @ diag(d) @ Q2
    A = A.astype(np.float32)

    if verbose:
        print(lapack_lu_sign(A))
        S = np.linalg.svd(A).S
        print("smallest singular value", np.min(S))
        print("largest singular value", np.max(S))

        print(np.linalg.slogdet(A))
        print("cond number", np.linalg.cond(A))

    return A, target_sign


def generate_matrix_with_target(
    n: int,
    target_logabsdet: float,
    cond: float = 1e5,
    target_sign: int = 1,
    seed: int = 100,
    verbose: bool = True,
):
    """
    Builds A = Q1 @ diag(d) @ Q2 (Q1, Q2 random orthogonal, det=+1) whose
    singular values are log-spaced so that, exactly by construction:
      - cond(A) == cond
      - sum(log|d_i|) == target_logabsdet  (so log|det(A)| == target_logabsdet)
      - sign(det(A)) == target_sign

    For log-spaced |d_i| = d_min * cond**(i/(n-1)), i = 0..n-1:
        sum(log|d_i|) = n*log(d_min) + log(cond) * n/2
    Solving for d_min given the target logabsdet:
        log(d_min) = (target_logabsdet - log(cond)*n/2) / n

    Returns (A, target_sign).
    """
    rng = np.random.default_rng(seed)

    log_cond = np.log(cond)
    log_d_min = (target_logabsdet - log_cond * n / 2) / n
    d_min = np.exp(log_d_min)

    mags = np.logspace(np.log10(d_min), np.log10(d_min * cond), n)
    signs = rng.choice([-1.0, 1.0], size=n)
    # Force the product of signs to match target_sign by fixing the last one.
    current_sign = np.prod(signs[:-1])
    signs[-1] = target_sign * current_sign
    d = signs * mags

    Q1 = _random_orthogonal_det_plus1(n, rng)
    Q2 = _random_orthogonal_det_plus1(n, rng)
    A = (Q1 * d) @ Q2  # = Q1 @ diag(d) @ Q2
    A = A.astype(np.float32)

    if verbose:
        print(lapack_lu_sign(A))
        S = np.linalg.svd(A).S
        print("smallest singular value", np.min(S))
        print("largest singular value", np.max(S))
        print(np.linalg.slogdet(A))
        print("cond number", np.linalg.cond(A))

    return A, target_sign


def run_sign_mismatch_trial(n_trials: int = 1000, n: int = 900, cond: float = 1e5):
    """
    Runs generate_matrix n_trials times (one seed per trial) and counts how
    often the LAPACK-derived determinant sign disagrees with the known
    target sign (which is always +1 by construction).
    """
    n_mismatch = 0
    log_abs_det_mean = 0
    mean_cond = 0
    for trial in tqdm.tqdm(range(n_trials),):
        A, target_sign = generate_matrix_with_target(n,500, cond=cond, seed=trial, verbose=False)
        sign_lapack, log_abs_det, _ = lapack_lu_sign(A)
        cond = np.linalg.cond(A)
        mean_cond+=cond
        log_abs_det_mean+=log_abs_det
        if int(np.sign(sign_lapack)) != target_sign:
            n_mismatch += 1
            
        print(np.linalg.svd(A).S[[0,1,2,3,-5,-4,-3,-2,-1]])
    print("mean det", log_abs_det_mean/n_trials)
    
    print(f"n={n} cond={cond:.1e} trials={n_trials}")
    print(f"n={n} cond measured={mean_cond/n_trials} trials={n_trials}")
    print(f"sign mismatches: {n_mismatch}/{n_trials} ({n_mismatch / n_trials:.2%})")
    return n_mismatch, n_trials


if __name__ == "__main__":
    run_sign_mismatch_trial(n_trials=10, n=900, cond=1e8)

# Create poorly conditioned matrix with random initialisations
# Create near singular matrix with N dims which are near zero -- check the smallest singular vals
# Check condition number
