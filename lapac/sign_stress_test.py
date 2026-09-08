"""Stress test for the *actual* sign/determinant path used in production:
torch.linalg.slogdet on a float32 tensor (see run_jacobian_tokenized.py).

This is deliberately independent from lapac.py / lapac_stress_test.py, which
probe scipy's CPU LAPACK ?getrf binding directly. That is a useful proxy but
NOT the code path that produced results/signs.h5 -- the real pipeline calls
torch.linalg.slogdet on CUDA float32 tensors. This module builds matrices
with a known-by-construction sign and log|det|, runs them through
torch.linalg.slogdet at float32 (on CUDA if available, else CPU), and checks
whether the recovered sign/log|det| match the ground truth.

Two stress tests:
  1. run_synthetic_sweep: log-spaced synthetic singular values, sweeping
     matrix size n and condition number, to map out where float32 sign
     recovery starts to fail.
  2. run_characteristic_sweep: uses the *actual* singular value spectra
     recorded in results/singular_values.h5 (shape (32, 2, 8, 896) here) as
     sigma, with many random sign/orthogonal draws per spectrum, to check
     whether the real Jacobians' singular value distributions are the kind
     that cause sign flips.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm


def _random_orthogonal_det_plus1(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix with det = +1 exactly (up to fp roundoff in QR)."""
    M = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_matrix_from_singular_values(
    sigma: np.ndarray,
    target_sign: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Builds A = Q1 @ diag(d) @ Q2 with |d_i| = sigma_i (so svdvals(A) == sigma
    up to fp roundoff), Q1/Q2 random Haar orthogonal, and sign(det(A)) ==
    target_sign exactly by construction. Signs of d_i are randomized subject
    to that determinant-sign constraint.

    Returns (A, known_log_abs_det) where known_log_abs_det = sum(log(sigma)).
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    if np.any(sigma < 0):
        raise ValueError("singular values must be non-negative")
    n = sigma.shape[0]

    signs = rng.choice([-1.0, 1.0], size=n)
    current_sign = np.prod(signs[:-1]) if n > 1 else 1.0
    signs[-1] = target_sign * current_sign
    d = signs * sigma

    Q1 = _random_orthogonal_det_plus1(n, rng)
    Q2 = _random_orthogonal_det_plus1(n, rng)
    A = (Q1 * d) @ Q2  # = Q1 @ diag(d) @ Q2

    with np.errstate(divide="ignore"):
        known_log_abs_det = float(np.sum(np.log(sigma)))
    return A, known_log_abs_det


def torch_slogdet_sign(A: np.ndarray, dtype: torch.dtype, device: torch.device) -> tuple[int, float]:
    """Runs the exact op used in production (torch.linalg.slogdet) on A cast
    to `dtype`. Returns (sign, log_abs_det)."""
    t = torch.from_numpy(A).to(device=device, dtype=dtype)
    sign, log_abs_det = torch.linalg.slogdet(t)
    return int(torch.sign(sign).item()), float(log_abs_det.item())


@dataclasses.dataclass
class TrialResult:
    n: int
    cond: float
    known_sign: int
    recovered_sign: int
    known_log_abs_det: float
    recovered_log_abs_det: float

    @property
    def sign_mismatch(self) -> bool:
        return self.known_sign != self.recovered_sign

    @property
    def log_abs_det_err(self) -> float:
        return abs(self.recovered_log_abs_det - self.known_log_abs_det)


def run_synthetic_sweep(
    ns: list[int] = (64, 128, 256, 512, 896),
    conds: list[float] = (1e2, 1e4, 1e6, 1e8, 1e10),
    trials_per_setting: int = 20,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> list[TrialResult]:
    """Sweeps matrix size x condition number, building log-spaced synthetic
    singular values in [1, cond] each trial, and checks torch.linalg.slogdet
    (at `dtype`, matching production) against the known-by-construction sign.
    """
    device = _device()
    rng = np.random.default_rng(seed)
    results: list[TrialResult] = []

    settings = [(n, cond) for n in ns for cond in conds]
    for n, cond in tqdm.tqdm(settings, desc="synthetic sweep"):
        for _ in range(trials_per_setting):
            sigma = np.logspace(0, np.log10(cond), n)
            target_sign = int(rng.choice([-1, 1]))
            A, known_log_abs_det = build_matrix_from_singular_values(sigma, target_sign, rng)
            recovered_sign, recovered_log_abs_det = torch_slogdet_sign(A, dtype, device)
            results.append(
                TrialResult(
                    n=n,
                    cond=cond,
                    known_sign=target_sign,
                    recovered_sign=recovered_sign,
                    known_log_abs_det=known_log_abs_det,
                    recovered_log_abs_det=recovered_log_abs_det,
                )
            )
    return results


def run_characteristic_sweep(
    singular_values_path: Path | str = "results/singular_values.h5",
    dataset_name: str = "singular_values",
    total_trials: int = 20,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> list[TrialResult]:
    """Loads the real singular-value spectra recorded from the actual
    Jacobian pipeline (results/singular_values.h5, shape e.g.
    (32, 2, 8, 896): batch x layer_pair x head x singular_value_index),
    randomly samples `total_trials` (spectrum, random sign/orthogonal draw,
    random target sign) trials from it, and checks torch.linalg.slogdet's
    recovered sign against the known-by-construction ground truth.

    This directly tests whether the *actual* conditioning your Jacobians
    exhibit is enough to flip signs under float32 slogdet.
    """
    device = _device()
    rng = np.random.default_rng(seed)

    with h5py.File(singular_values_path, "r") as f:
        sing_vals = np.asarray(f[dataset_name])

    flat_shape = sing_vals.shape[:-1]
    n = sing_vals.shape[-1]
    flat = sing_vals.reshape(-1, n)

    n_spectra = flat.shape[0]
    print(f"loaded {n_spectra} spectra with shape {flat_shape}, n={n}; sampling {total_trials} trials")
    sample_indices = rng.integers(0, n_spectra, size=total_trials)

    results: list[TrialResult] = []
    for idx in tqdm.tqdm(sample_indices, desc="characteristic sweep"):
        sigma = flat[idx].astype(np.float64)
        sigma = np.clip(sigma, a_min=0.0, a_max=None)  # guard tiny negative fp noise
        cond = float(sigma.max() / sigma.min()) if sigma.min() > 0 else float("inf")

        target_sign = int(rng.choice([-1, 1]))
        A, known_log_abs_det = build_matrix_from_singular_values(sigma, target_sign, rng)
        recovered_sign, recovered_log_abs_det = torch_slogdet_sign(A, dtype, device)
        results.append(
            TrialResult(
                n=n,
                cond=cond,
                known_sign=target_sign,
                recovered_sign=recovered_sign,
                known_log_abs_det=known_log_abs_det,
                recovered_log_abs_det=recovered_log_abs_det,
            )
        )

    return results


def summarize(results: list[TrialResult], group_by: str = "cond") -> None:
    """Prints mismatch rate and log|det| error grouped by `group_by` ("cond" or "n")."""
    keys = sorted({getattr(r, group_by) for r in results})
    print(f"\n{'='*60}\nsummary grouped by {group_by} (dtype used inside slogdet call)\n{'='*60}")
    total_mismatch = 0
    for key in keys:
        group = [r for r in results if getattr(r, group_by) == key]
        n_mismatch = sum(r.sign_mismatch for r in group)
        total_mismatch += n_mismatch
        max_err = max((r.log_abs_det_err for r in group), default=0.0)
        mean_err = float(np.mean([r.log_abs_det_err for r in group])) if group else 0.0
        print(
            f"{group_by}={key:.3e}  n_trials={len(group):5d}  "
            f"sign_mismatches={n_mismatch:4d} ({n_mismatch/len(group):6.2%})  "
            f"log|det| err mean={mean_err:.3e} max={max_err:.3e}"
        )
    print(f"\nTOTAL: {total_mismatch}/{len(results)} sign mismatches "
          f"({total_mismatch/len(results):.2%})")


if __name__ == "__main__":
    print(f"device: {_device()}")

    print("\n### synthetic sweep (torch.linalg.slogdet, float32) ###")
    synthetic_results = run_synthetic_sweep()
    summarize(synthetic_results, group_by="cond")
    summarize(synthetic_results, group_by="n")

    print("\n### characteristic sweep (your real singular value spectra) ###")
    try:
        characteristic_results = run_characteristic_sweep()
    except FileNotFoundError as e:
        print(f"skipping characteristic sweep: {e}")
    else:
        summarize(characteristic_results, group_by="cond")
