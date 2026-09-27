"""
src/utils/sample_size.py
─────────────────────────
Sample size calculation, power analysis, and pre-experiment diagnostics for LiftLab.

Includes:
  - Two-proportion z-test sample size (Cohen's h)
  - Power curve generation
  - MDE (Minimum Detectable Effect) calculator
  - Duration estimator
  - Sample Ratio Mismatch (SRM) detection
  - CUPED variance reduction estimator
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from loguru import logger
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Data Contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleSizeResult:
    """Output of sample size calculation."""
    n_per_group: int
    n_total: int
    baseline_rate: float
    variant_rate: float
    mde_absolute: float
    mde_relative: float
    alpha: float
    power: float
    cohen_h: float
    days_required: float            # given daily_traffic kwarg
    daily_traffic: int | None       # users per day across both groups


@dataclass
class SRMResult:
    """Sample Ratio Mismatch detection result."""
    srm_detected: bool
    chi2_stat: float
    p_value: float
    assigned_a: int
    assigned_b: int
    expected_ratio: float
    observed_ratio: float
    message: str


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_size(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_traffic: int | None = None,
    two_tailed: bool = True,
) -> SampleSizeResult:
    """
    Compute required sample size per group using Cohen's h effect size.

    Why Cohen's h instead of the simple z-test formula?
    The arcsine transformation stabilises variance of proportions,
    giving a more accurate sample size especially when rates are
    near 0 or 1 (e.g. 1% or 99% base rate).

    Formula
    -------
    n = ((z_α + z_β) / h)²
    where h = 2·arcsin(√p2) - 2·arcsin(√p1)   (Cohen's h)

    Parameters
    ----------
    baseline_rate   : conversion rate of control group (0 < p < 1)
    mde_relative    : minimum detectable effect as fraction of baseline
                      e.g. 0.10 means a 10% relative lift (12% → 13.2%)
    alpha           : Type-I error rate (default 0.05)
    power           : 1 - Type-II error rate (default 0.80)
    daily_traffic   : users per day across both groups (for duration estimate)
    two_tailed      : whether hypothesis is directional
    """
    if not 0 < baseline_rate < 1:
        raise ValueError(f"baseline_rate must be in (0,1), got {baseline_rate}")
    if mde_relative <= -1:
        raise ValueError("mde_relative must be > -1")

    variant_rate = baseline_rate * (1 + mde_relative)
    variant_rate = min(max(variant_rate, 0.001), 0.999)

    # Cohen's h effect size for proportions
    phi1 = 2 * np.arcsin(np.sqrt(baseline_rate))
    phi2 = 2 * np.arcsin(np.sqrt(variant_rate))
    h = abs(phi2 - phi1)

    if h < 1e-9:
        raise ValueError("MDE is too small — effect size is effectively zero.")

    # Critical values
    alpha_adj = alpha / 2 if two_tailed else alpha
    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta = stats.norm.ppf(power)

    n = int(np.ceil(((z_alpha + z_beta) / h) ** 2))
    n_total = n * 2

    days = None
    if daily_traffic and daily_traffic > 0:
        days = round(n_total / daily_traffic, 1)

    logger.debug(
        f"Sample size: n={n}/group, h={h:.4f}, "
        f"base={baseline_rate:.2%}, variant={variant_rate:.2%}"
    )
    return SampleSizeResult(
        n_per_group=n,
        n_total=n_total,
        baseline_rate=baseline_rate,
        variant_rate=variant_rate,
        mde_absolute=variant_rate - baseline_rate,
        mde_relative=mde_relative,
        alpha=alpha,
        power=power,
        cohen_h=float(h),
        days_required=float(days) if days else None,
        daily_traffic=daily_traffic,
    )


def power_curve(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    sample_sizes: list[int] | None = None,
) -> list[dict]:
    """
    Generate a power curve: power vs. sample size.

    Useful for visualising how power grows as experiment runs longer.

    Returns
    -------
    List of {n_per_group, power} dicts for plotting
    """
    variant_rate = baseline_rate * (1 + mde_relative)
    variant_rate = min(max(variant_rate, 0.001), 0.999)

    if sample_sizes is None:
        # Auto-generate a sensible range
        req = compute_sample_size(baseline_rate, mde_relative, alpha=alpha, power=0.80)
        max_n = req.n_per_group * 3
        sample_sizes = list(range(10, max_n, max(max_n // 50, 1)))

    curve = []
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    for n in sample_sizes:
        se = np.sqrt(
            baseline_rate * (1 - baseline_rate) / n
            + variant_rate * (1 - variant_rate) / n
        )
        ncp = abs(variant_rate - baseline_rate) / max(se, 1e-12)
        pwr = float(stats.norm.cdf(ncp - z_alpha))
        curve.append({"n_per_group": n, "power": round(pwr, 4)})
    return curve


def mde_curve(
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    mde_range: list[float] | None = None,
) -> list[dict]:
    """
    MDE vs required sample size tradeoff curve.

    Shows how demanding it is to detect smaller effects.
    Helps teams set realistic experiment goals.

    Returns
    -------
    List of {mde_relative, n_per_group} dicts
    """
    if mde_range is None:
        mde_range = [i / 100 for i in range(1, 51)]  # 1% to 50%

    curve = []
    for mde in mde_range:
        try:
            res = compute_sample_size(baseline_rate, mde, alpha=alpha, power=power)
            curve.append({"mde_relative": mde, "n_per_group": res.n_per_group})
        except ValueError:
            continue
    return curve


def check_sample_ratio_mismatch(
    assigned_a: int,
    assigned_b: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.01,          # more stringent than 0.05 for SRM
) -> SRMResult:
    """
    Detect Sample Ratio Mismatch (SRM) — a critical quality check.

    SRM occurs when the observed assignment ratio significantly deviates from
    the expected ratio (usually 50/50). It indicates a bug in:
      - Assignment hash function
      - Bot filtering (applied after assignment)
      - Load balancer sticky sessions
      - Logging pipeline dropping events unevenly

    SRM invalidates the experiment — you cannot trust any metric result
    until the root cause is fixed and the experiment re-run.

    Parameters
    ----------
    assigned_a     : users assigned to control
    assigned_b     : users assigned to variant
    expected_ratio : fraction expected in group A (0.5 for 50/50 split)
    alpha          : significance threshold (0.01 is standard for SRM checks)
    """
    total = assigned_a + assigned_b
    if total == 0:
        raise ValueError("Total assignments is 0.")

    expected_a = total * expected_ratio
    expected_b = total * (1 - expected_ratio)

    chi2 = (
        (assigned_a - expected_a) ** 2 / expected_a
        + (assigned_b - expected_b) ** 2 / expected_b
    )
    p_val = float(1 - stats.chi2.cdf(chi2, df=1))
    srm_detected = bool(p_val < alpha)
    observed_ratio = assigned_a / total

    if srm_detected:
        msg = (
            f"⚠ SRM DETECTED (p={p_val:.5f} < {alpha}). "
            f"Expected ratio={expected_ratio:.2f}, observed={observed_ratio:.3f}. "
            "STOP the experiment and investigate assignment logic before analysing results."
        )
        logger.warning(msg)
    else:
        msg = (
            f"SRM check passed (p={p_val:.4f} ≥ {alpha}). "
            f"Assignment ratio {observed_ratio:.3f} is consistent with expected {expected_ratio:.2f}."
        )

    return SRMResult(
        srm_detected=srm_detected,
        chi2_stat=float(chi2),
        p_value=p_val,
        assigned_a=assigned_a,
        assigned_b=assigned_b,
        expected_ratio=expected_ratio,
        observed_ratio=float(observed_ratio),
        message=msg,
    )


def cuped_variance_reduction(
    metric_post: np.ndarray,
    covariate_pre: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    CUPED — Controlled-experiment Using Pre-Experiment Data.

    Reduces metric variance by regressing out the pre-experiment covariate
    (typically the same metric measured before the experiment started).

    CUPED estimator:
        Y_cuped = Y - θ·(X - E[X])
        where θ = Cov(Y, X) / Var(X)

    This is equivalent to including the covariate as a regression control.
    Variance reduction = 1 - ρ(Y, X)²  where ρ is Pearson correlation.

    Parameters
    ----------
    metric_post   : post-experiment metric values (1-D array)
    covariate_pre : pre-experiment values of same metric (1-D array)

    Returns
    -------
    (adjusted_metric, variance_reduction_fraction)
    variance_reduction_fraction: 0.30 means 30% variance reduction,
                                  equivalent to running 43% more samples.
    """
    metric_post = np.asarray(metric_post, dtype=float)
    covariate_pre = np.asarray(covariate_pre, dtype=float)

    if len(metric_post) != len(covariate_pre):
        raise ValueError("metric_post and covariate_pre must have the same length.")

    # OLS coefficient: θ = Cov(Y, X) / Var(X)
    theta = np.cov(metric_post, covariate_pre)[0, 1] / max(np.var(covariate_pre, ddof=1), 1e-12)

    # Adjusted metric
    adjusted = metric_post - theta * (covariate_pre - np.mean(covariate_pre))

    # Variance reduction
    var_before = np.var(metric_post, ddof=1)
    var_after = np.var(adjusted, ddof=1)
    reduction = 1 - var_after / max(var_before, 1e-12)

    logger.info(f"CUPED: θ={theta:.4f}, variance reduction={reduction:.1%}")
    return adjusted, float(reduction)


def sequential_alpha_spending(
    n_looks: int,
    alpha: float = 0.05,
    method: str = "obrien_fleming",
) -> list[float]:
    """
    Compute alpha spending boundaries for sequential analysis.

    Prevents p-value inflation from interim peeking at results.

    Methods
    -------
    - obrien_fleming : conservative early, liberal late (preferred)
    - pocock         : constant boundary (more permissive early)

    Returns
    -------
    List of alpha thresholds for each interim analysis
    """
    t_vals = np.linspace(1 / n_looks, 1, n_looks)

    if method == "obrien_fleming":
        # O'Brien-Fleming: z(t) = z_alpha / sqrt(t)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        boundaries = [2 * (1 - stats.norm.cdf(z_alpha / np.sqrt(t))) for t in t_vals]
    elif method == "pocock":
        # Pocock: constant boundary (approximate — iterative solution omitted)
        boundaries = [alpha / n_looks * 1.5] * n_looks
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'obrien_fleming' or 'pocock'.")

    return [round(b, 6) for b in boundaries]
