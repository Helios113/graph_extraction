"""Probability-weighted-moments (PWM) / L-moments GPD(xi, sigma, loc=0)
estimator (Hosking & Wallis 1987), as a closed-form alternative to
scripts/lipschitz/batched_gpd_mle.py's Newton MLE.

MLE for the GPD shape parameter is known to be poorly behaved near xi=0 (the
likelihood surface flattens/becomes multimodal there) and non-regular for
xi < -0.5 -- this estimator behaves differently in that region (no
likelihood optimization at all, just sample L-moments), so comparing the two
on the same exceedances is a useful check for whether a near-zero or
sign-flipped xi from the MLE is a genuine feature of the data or a Newton
optimizer artifact.

Formulas (loc=0 GPD, z ascending-sorted exceedances z_(1) <= ... <= z_(n)):
    b0 = mean(z)
    b1 = mean_i [ (n-i)/(n-1) * z_(i) ]   (i = 1..n)
    xi_hat    = 2 - b0 / (b0 - 2*b1)
    sigma_hat = 2 * b0 * b1 / (b0 - 2*b1)

Verified against scipy.stats.genpareto.rvs-generated samples across a range
of true (xi, sigma) in this module's __main__ self-test block.
"""

import numpy as np


def gpd_pwm_fit(z: np.ndarray) -> tuple[float, float]:
    """Fits GPD(xi, sigma, loc=0) to 1-D exceedances z (any order; sorted
    internally) via PWM/L-moments. Returns (xi, sigma), both nan if the
    sample is degenerate (n < 2, or b0 - 2*b1 == 0)."""
    z = np.sort(np.asarray(z, dtype=np.float64))
    n = z.size
    if n < 2:
        return float("nan"), float("nan")

    b0 = z.mean()
    i = np.arange(1, n + 1)
    b1 = np.mean((n - i) / (n - 1) * z)

    denom = b0 - 2 * b1
    if denom == 0:
        return float("nan"), float("nan")

    xi = 2 - b0 / denom
    sigma = 2 * b0 * b1 / denom
    return float(xi), float(sigma)


if __name__ == "__main__":
    from scipy.stats import genpareto

    rng = np.random.default_rng(0)
    print(f"{'true_xi':>8} {'true_sigma':>10} {'n':>6} {'xi_pwm':>10} {'sigma_pwm':>10} "
          f"{'xi_mle':>10} {'sigma_mle':>10}")
    for true_xi in (-0.4, -0.1, 0.0, 0.2):
        for true_sigma in (0.5, 1.0, 3.0):
            for n in (30, 200, 2000):
                sample = genpareto.rvs(true_xi, loc=0, scale=true_sigma, size=n, random_state=rng)
                xi_pwm, sigma_pwm = gpd_pwm_fit(sample)
                try:
                    xi_mle, _loc, sigma_mle = genpareto.fit(sample, floc=0)
                except Exception:
                    xi_mle, sigma_mle = float("nan"), float("nan")
                print(f"{true_xi:>8.2f} {true_sigma:>10.2f} {n:>6} {xi_pwm:>10.4f} {sigma_pwm:>10.4f} "
                      f"{xi_mle:>10.4f} {sigma_mle:>10.4f}")
