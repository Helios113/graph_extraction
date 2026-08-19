"""
Stress test: does LAPACK's LU-based determinant sign (as wrapped by
numpy.linalg.det/slogdet and scipy.linalg.lapack.?getrf) get the sign
of det(A) right when A is ill-conditioned?

Construction (gives an EXACTLY known sign, independent of the numerics
you're testing):

    A = Q1 @ diag(d) @ Q2

  - Q1, Q2: random orthogonal matrices with det(Q) = +1 exactly (proper
    rotations, built by QR + a column-sign fix).
  - d: real vector, |d_i| log-spaced to hit a target condition number
    (cond(A) = max|d_i| / min|d_i|, since Q1,Q2 are orthogonal so the
    singular values of A are exactly |d_i|), with a chosen number of
    negative entries.

  det(A) = det(Q1) * prod(d) * det(Q2) = prod(d)   (since det Q1=det Q2=1)

  sign(det A) = (-1)^(number of negative d_i)  -- known exactly, by
  construction, with no floating point involved in deriving it.

This isolates the thing being tested: whether the numerical LU
factorization recovers the correct sign, not whether some other
"known-answer" formula (e.g. a Vandermonde or Hilbert matrix determinant
formula) is itself accurate.
"""
import numpy as np
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


def make_ill_conditioned(n, cond, n_negative, rng, dtype=np.float64):
    """
    Returns (A, known_sign, known_log_abs_det).
    cond: target condition number (max|d_i|/min|d_i|).
    n_negative: how many of the n singular directions get a negative sign.
    """
    assert 0 <= n_negative <= n
    mags = np.logspace(0, np.log10(cond), n)         # from 1 to cond
    signs = np.array([-1.0] * n_negative + [1.0] * (n - n_negative))
    rng.shuffle(signs)
    d = signs * mags

    Q1 = random_orthogonal_det_plus1(n, rng)
    Q2 = random_orthogonal_det_plus1(n, rng)
    A = (Q1 * d) @ Q2  # = Q1 @ diag(d) @ Q2

    known_sign = 1 if (n_negative % 2 == 0) else -1
    known_log_abs_det = np.sum(np.log(mags))  # sum log|d_i|
    return A.astype(dtype), known_sign, known_log_abs_det


def lapack_lu_sign(A):
    """
    Sign of det(A) computed the way LAPACK-based det() implementations do:
    sign = parity(pivot permutation) * prod(sign(diag(U))).
    Uses scipy's raw ?getrf binding directly (dgetrf/sgetrf under the hood).
    Returns (sign, log_abs_det, info).
    """
    getrf = lapack.get_lapack_funcs('getrf', (A,))
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
    with np.errstate(divide='ignore'):
        log_abs_det = np.sum(np.log(np.abs(diag)))  # -inf if a diag entry underflowed to 0
    return sign, log_abs_det, info


def run_stress_test(sizes=(10, 50, 100), conds=None, trials=5, seed=0):
    if conds is None:
        conds = np.logspace(-18, 18, 19)  # 1e0 ... 1e18

    rng = np.random.default_rng(seed)
    results = []

    for n in sizes:
        for cond in conds:
            for trial in range(trials):
                n_neg = rng.integers(0, n + 1)
                for dtype, label in ((np.float64, 'f64'), (np.float32, 'f32')):
                    A, known_sign, known_log_abs = make_ill_conditioned(
                        n, cond, n_neg, rng, dtype=dtype
                    )

                    # 1) manual LAPACK getrf-based sign (what det() does internally)
                    sign_lapack, log_abs_lapack, info = lapack_lu_sign(A)

                    # 2) numpy.linalg.slogdet cross-check (also LAPACK LU, fp64 internally
                    #    for numpy; numpy upcasts float32 input to float64 before calling
                    #    LAPACK, so this is NOT an independent float32 check)
                    sign_np, logdet_np = np.linalg.slogdet(A.astype(np.float64))

                    actual_cond = np.linalg.cond(A.astype(np.float64))

                    results.append(dict(
                        n=n, target_cond=cond, actual_cond=actual_cond,
                        dtype=label, n_negative=int(n_neg),
                        known_sign=known_sign,
                        lapack_sign=int(np.sign(sign_lapack)),
                        lapack_match=int(np.sign(sign_lapack)) == known_sign,
                        info=info,
                    ))
                    print(results[-1])
    return results


if __name__ == '__main__':
    results = run_stress_test(
        sizes=(900,),
        conds=np.logspace(5, 10, 5),
        trials=10,
        seed=0,
    )

    import collections
    # failure rate by (dtype, condition-number decade)
    buckets = collections.defaultdict(lambda: [0, 0])  # key -> [n_fail, n_total]
    for r in results:
        key = (r['dtype'], round(np.log10(r['target_cond'])))
        buckets[key][1] += 1
        if not r['lapack_match']:
            buckets[key][0] += 1

    print(f"{'dtype':5s} {'log10(cond)':>11s} {'fail/total':>12s} {'fail rate':>10s}")
    for key in sorted(buckets):
        dtype, logc = key
        nfail, ntot = buckets[key]
        print(f"{dtype:5s} {logc:11d} {nfail:5d}/{ntot:<5d}    {nfail/ntot:8.2%}")

    n_fail_total = sum(1 for r in results if not r['lapack_match'])
    print(f"\nTotal failures: {n_fail_total}/{len(results)}")

    # show a few concrete failing cases if any
    fails = [r for r in results if not r['lapack_match']]
    if fails:
        print("\nExample failures (target_cond, actual_cond, known_sign, lapack_sign):")
        for r in fails[:10]:
            print(f"  n={r['n']:4d} dtype={r['dtype']} target_cond={r['target_cond']:.1e} "
                  f"actual_cond={r['actual_cond']:.3e} known={r['known_sign']:+d} "
                  f"lapack={r['lapack_sign']:+d}")