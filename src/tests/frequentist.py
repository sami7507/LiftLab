"""
src/tests/frequentist.py
─────────────────────────
Production-grade frequentist A/B testing module.

Implements:
  - Welch's t-test         → continuous metrics (revenue, session time)
  - Student's t-test       → continuous metrics (equal variance assumed)
  - Chi-square test        → binary outcomes (conversion, click)
  - Mann-Whitney U test    → non-normal / ordinal metrics (ratings, NPS)
  - Z-test for proportions → large-sample binary outcomes

Each test returns a typed TestResult dataclass so downstream code
never has to inspect raw scipy objects.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# Data Contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Standardised output for every statistical test."""

    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool

    # Effect size
    effect_size: float
    effect_size_label: str          # e.g. "Cohen's d", "Cramér's V"
    effect_magnitude: str           # "negligible" | "small" | "medium" | "large"

    # Confidence / credible interval (None for non-parametric or chi-square)
    ci_lower: float | None
    ci_upper: float | None
    ci_level: float | None          # e.g. 0.95

    # Human-readable
    interpretation: str
    recommendation: str

    # Raw group summaries (always populated)
    control_mean: float
    variant_mean: float
    control_std: float | None = None
    variant_std: float | None = None
    control_n: int = 0
    variant_n: int = 0

    # Metadata
    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cohens_d_magnitude(d: float) -> str:
    """Interpret Cohen's d by conventional benchmarks."""
    d = abs(d)
    if d < 0.20:
        return "negligible"
    if d < 0.50:
        return "small"
    if d < 0.80:
        return "medium"
    return "large"


def _cramers_v_magnitude(v: float) -> str:
    """Interpret Cramér's V for 2×2 tables."""
    if v < 0.10:
        return "negligible"
    if v < 0.30:
        return "small"
    if v < 0.50:
        return "medium"
    return "large"


def _check_sample_size(n: int, minimum: int = 30) -> list[str]:
    """Return warnings if sample size is dangerously small."""
    w = []
    if n < minimum:
        w.append(f"Sample size n={n} < {minimum}. Results may be unreliable.")
    return w


def _check_normality(data: np.ndarray) -> list[str]:
    """Shapiro-Wilk normality check. Returns warnings if violated."""
    w = []
    if len(data) < 3:
        return w
    try:
        # Shapiro-Wilk is reliable for n < 5000
        sample = data[:5000] if len(data) > 5000 else data
        _, p = stats.shapiro(sample)
        if p < 0.05:
            w.append(
                f"Normality assumption may be violated (Shapiro-Wilk p={p:.4f}). "
                "Consider Mann-Whitney U instead."
            )
    except Exception:
        pass
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class FrequentistTests:
    """
    A/B testing using classical frequentist methods.

    Usage
    -----
    >>> tester = FrequentistTests(alpha=0.05)
    >>> result = tester.welch_ttest(control_data, variant_data)
    >>> print(result.significant, result.p_value)
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        logger.info(f"FrequentistTests initialised with α={alpha}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Welch's t-test  (continuous, unequal variance — DEFAULT for most A/Bs)
    # ─────────────────────────────────────────────────────────────────────────

    def welch_ttest(
        self,
        control: np.ndarray,
        variant: np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> TestResult:
        """
        Welch's t-test — does NOT assume equal variances.

        Best for: revenue, session duration, page views (continuous metrics).
        Unlike Student's t-test, robust when group SDs differ.

        Parameters
        ----------
        control   : 1-D array of observations for the control group
        variant   : 1-D array of observations for the variant group
        alternative: hypothesis direction

        Returns
        -------
        TestResult with Cohen's d effect size and 95% CI on the mean difference
        """
        control = np.asarray(control, dtype=float)
        variant = np.asarray(variant, dtype=float)

        warn_msgs: list[str] = []
        warn_msgs += _check_sample_size(len(control))
        warn_msgs += _check_sample_size(len(variant))
        warn_msgs += _check_normality(control)
        warn_msgs += _check_normality(variant)

        t_stat, p_val = stats.ttest_ind(control, variant,
                                         equal_var=False,
                                         alternative=alternative)

        # Cohen's d using pooled SD (Hedges' correction for small n)
        n1, n2 = len(control), len(variant)
        s1, s2 = np.std(control, ddof=1), np.std(variant, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohen_d = (np.mean(variant) - np.mean(control)) / (pooled_sd + 1e-12)

        # Hedges' g correction (unbiased for small samples)
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        hedges_g = cohen_d * correction

        # 95% CI on mean difference via Welch-Satterthwaite df
        se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
        df_welch = (s1**2 / n1 + s2**2 / n2)**2 / (
            (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        )
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=df_welch)
        mean_diff = np.mean(variant) - np.mean(control)
        ci_lo = mean_diff - t_crit * se_diff
        ci_hi = mean_diff + t_crit * se_diff

        sig = bool(p_val < self.alpha)
        magnitude = _cohens_d_magnitude(hedges_g)

        interpretation = (
            f"Variant mean ({np.mean(variant):.4f}) is "
            f"{'significantly' if sig else 'NOT significantly'} different from "
            f"control mean ({np.mean(control):.4f}). "
            f"Effect size (Hedges' g) = {hedges_g:.3f} — {magnitude}."
        )
        recommendation = (
            "Ship variant." if sig and np.mean(variant) > np.mean(control)
            else "Hold. No significant improvement detected."
        )

        logger.debug(f"Welch t-test: t={t_stat:.4f}, p={p_val:.4f}, g={hedges_g:.4f}")
        return TestResult(
            test_name="Welch's t-test",
            statistic=float(t_stat),
            p_value=float(p_val),
            alpha=self.alpha,
            significant=sig,
            effect_size=float(hedges_g),
            effect_size_label="Hedges' g",
            effect_magnitude=magnitude,
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            ci_level=1 - self.alpha,
            interpretation=interpretation,
            recommendation=recommendation,
            control_mean=float(np.mean(control)),
            variant_mean=float(np.mean(variant)),
            control_std=float(s1),
            variant_std=float(s2),
            control_n=n1,
            variant_n=n2,
            warnings=warn_msgs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Student's t-test  (continuous, equal variance assumed)
    # ─────────────────────────────────────────────────────────────────────────

    def student_ttest(
        self,
        control: np.ndarray,
        variant: np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> TestResult:
        """
        Student's t-test — assumes equal variances (homoscedastic).

        Use only when Levene's / Bartlett's test confirms equal variances.
        In practice, prefer Welch's — it's strictly more general.
        """
        control = np.asarray(control, dtype=float)
        variant = np.asarray(variant, dtype=float)

        warn_msgs = _check_sample_size(len(control)) + _check_sample_size(len(variant))

        # Levene's test for equal variances
        _, lev_p = stats.levene(control, variant)
        if lev_p < 0.05:
            warn_msgs.append(
                f"Levene's test p={lev_p:.4f} < 0.05 — variances differ. "
                "Welch's t-test is more appropriate."
            )

        t_stat, p_val = stats.ttest_ind(control, variant,
                                         equal_var=True,
                                         alternative=alternative)
        n1, n2 = len(control), len(variant)
        s1, s2 = np.std(control, ddof=1), np.std(variant, ddof=1)
        pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        cohen_d = (np.mean(variant) - np.mean(control)) / (pooled_sd + 1e-12)

        se_diff = pooled_sd * np.sqrt(1 / n1 + 1 / n2)
        t_crit = stats.t.ppf(1 - self.alpha / 2, df=n1 + n2 - 2)
        mean_diff = np.mean(variant) - np.mean(control)

        sig = bool(p_val < self.alpha)
        magnitude = _cohens_d_magnitude(cohen_d)

        return TestResult(
            test_name="Student's t-test",
            statistic=float(t_stat),
            p_value=float(p_val),
            alpha=self.alpha,
            significant=sig,
            effect_size=float(cohen_d),
            effect_size_label="Cohen's d",
            effect_magnitude=magnitude,
            ci_lower=float(mean_diff - t_crit * se_diff),
            ci_upper=float(mean_diff + t_crit * se_diff),
            ci_level=1 - self.alpha,
            interpretation=f"Cohen's d = {cohen_d:.3f} ({magnitude} effect).",
            recommendation="Ship variant." if sig and mean_diff > 0 else "Hold.",
            control_mean=float(np.mean(control)),
            variant_mean=float(np.mean(variant)),
            control_std=float(s1),
            variant_std=float(s2),
            control_n=n1,
            variant_n=n2,
            warnings=warn_msgs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Chi-square test  (binary outcomes — conversions, clicks)
    # ─────────────────────────────────────────────────────────────────────────

    def chi_square_test(
        self,
        n_control: int,
        conversions_control: int,
        n_variant: int,
        conversions_variant: int,
        yates_correction: bool = False,
    ) -> TestResult:
        """
        Pearson's Chi-square test for independence (2×2 contingency table).

        Best for: conversion rate, click-through rate, sign-up rate.

        Parameters
        ----------
        n_control            : total users in control
        conversions_control  : converted users in control
        n_variant            : total users in variant
        conversions_variant  : converted users in variant
        yates_correction     : apply Yates' continuity correction (conservative)
                               — recommended when any cell count < 5

        Effect size: Cramér's V (appropriate for 2×2 tables)
        """
        warn_msgs: list[str] = []

        # Build 2×2 contingency table
        table = np.array([
            [conversions_control,   n_control - conversions_control],
            [conversions_variant,   n_variant - conversions_variant],
        ])

        # Check expected counts ≥ 5 (chi-square assumption)
        n_total = n_control + n_variant
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / n_total
        if (expected < 5).any():
            warn_msgs.append(
                "Some expected cell counts < 5. Consider Fisher's exact test "
                "or apply Yates' correction."
            )

        chi2, p_val, dof, _ = stats.chi2_contingency(
            table, correction=yates_correction
        )

        # Cramér's V
        cramers_v = np.sqrt(chi2 / (n_total * (min(table.shape) - 1)))

        # Risk difference and relative risk
        cr_ctrl = conversions_control / max(n_control, 1)
        cr_var = conversions_variant / max(n_variant, 1)
        risk_diff = cr_var - cr_ctrl
        relative_risk = cr_var / max(cr_ctrl, 1e-9)

        # 95% CI on risk difference (Newcombe method approximation)
        se_rd = np.sqrt(
            cr_ctrl * (1 - cr_ctrl) / n_control
            + cr_var * (1 - cr_var) / n_variant
        )
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lo = risk_diff - z_crit * se_rd
        ci_hi = risk_diff + z_crit * se_rd

        sig = bool(p_val < self.alpha)
        magnitude = _cramers_v_magnitude(cramers_v)

        interpretation = (
            f"Control CVR: {cr_ctrl:.2%} | Variant CVR: {cr_var:.2%}. "
            f"Absolute lift: {risk_diff:+.2%} | Relative lift: {(relative_risk - 1):+.2%}. "
            f"Cramér's V = {cramers_v:.3f} ({magnitude} association)."
        )

        logger.debug(f"Chi-square: χ²={chi2:.4f}, p={p_val:.4f}, V={cramers_v:.4f}")
        return TestResult(
            test_name=f"Chi-square{'+ Yates' if yates_correction else ''}",
            statistic=float(chi2),
            p_value=float(p_val),
            alpha=self.alpha,
            significant=sig,
            effect_size=float(cramers_v),
            effect_size_label="Cramér's V",
            effect_magnitude=magnitude,
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            ci_level=1 - self.alpha,
            interpretation=interpretation,
            recommendation="Ship variant." if sig and risk_diff > 0 else "Hold.",
            control_mean=float(cr_ctrl),
            variant_mean=float(cr_var),
            control_n=n_control,
            variant_n=n_variant,
            warnings=warn_msgs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Mann-Whitney U  (non-parametric — ordinal, skewed, or heavy-tailed)
    # ─────────────────────────────────────────────────────────────────────────

    def mann_whitney_test(
        self,
        control: np.ndarray,
        variant: np.ndarray,
        alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    ) -> TestResult:
        """
        Mann-Whitney U test — non-parametric test for stochastic dominance.

        Best for:
          - NPS scores, star ratings (ordinal data)
          - Revenue with heavy right tail (many zeros, a few large values)
          - Any metric where normality is strongly violated

        Effect size: Common Language Effect Size (CLES) = P(X_B > X_A)
        Interpreted as: probability that a random variant user scores higher.
        """
        control = np.asarray(control, dtype=float)
        variant = np.asarray(variant, dtype=float)

        warn_msgs = _check_sample_size(len(control), 20) + _check_sample_size(len(variant), 20)

        u_stat, p_val = stats.mannwhitneyu(
            variant, control, alternative=alternative
        )

        n1, n2 = len(control), len(variant)

        # Common Language Effect Size (probability of superiority)
        cles = u_stat / (n1 * n2)

        # Rank-biserial correlation (equivalent to Cohen's d for non-parametric)
        rank_biserial = 2 * cles - 1  # ranges [-1, +1]

        # Glass's delta (non-parametric effect interpretation)
        if abs(rank_biserial) < 0.10:
            magnitude = "negligible"
        elif abs(rank_biserial) < 0.30:
            magnitude = "small"
        elif abs(rank_biserial) < 0.50:
            magnitude = "medium"
        else:
            magnitude = "large"

        # Hodges-Lehmann estimator for median shift (location difference)
        # Compute all pairwise differences (expensive for large n — sample if needed)
        if n1 * n2 <= 100_000:
            diffs = np.subtract.outer(variant, control).ravel()
            hl_estimate = np.median(diffs)
        else:
            # Sample 10k pairs for large datasets
            rng = np.random.default_rng(42)
            idx1 = rng.choice(n2, 10_000)
            idx2 = rng.choice(n1, 10_000)
            hl_estimate = np.median(variant[idx1] - control[idx2])

        interpretation = (
            f"P(Variant > Control) = {cles:.1%} | Rank-biserial r = {rank_biserial:+.3f} ({magnitude}). "
            f"Hodges-Lehmann shift estimate: {hl_estimate:+.4f}. "
            f"{'Variant stochastically dominates control.' if cles > 0.5 else 'Control stochastically dominates variant.'}"
        )

        logger.debug(f"Mann-Whitney: U={u_stat:.1f}, p={p_val:.4f}, CLES={cles:.4f}")
        return TestResult(
            test_name="Mann-Whitney U",
            statistic=float(u_stat),
            p_value=float(p_val),
            alpha=self.alpha,
            significant=bool(p_val < self.alpha),
            effect_size=float(cles),
            effect_size_label="CLES  P(B>A)",
            effect_magnitude=magnitude,
            ci_lower=None,
            ci_upper=None,
            ci_level=None,
            interpretation=interpretation,
            recommendation=(
                "Ship variant." if p_val < self.alpha and cles > 0.5 else "Hold."
            ),
            control_mean=float(np.median(control)),   # median for non-parametric
            variant_mean=float(np.median(variant)),
            control_std=float(stats.iqr(control)),    # IQR instead of SD
            variant_std=float(stats.iqr(variant)),
            control_n=n1,
            variant_n=n2,
            warnings=warn_msgs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Z-test for proportions  (large-sample binary outcomes)
    # ─────────────────────────────────────────────────────────────────────────

    def z_test_proportions(
        self,
        n_control: int,
        conversions_control: int,
        n_variant: int,
        conversions_variant: int,
        alternative: Literal["two-sided", "larger", "smaller"] = "two-sided",
    ) -> TestResult:
        """
        Two-proportion z-test using statsmodels.

        Equivalent to chi-square for large n (n > 100 per group recommended).
        Provides a directional z-statistic and is faster to compute than chi-square
        for very large datasets.
        """
        from statsmodels.stats.proportion import proportions_ztest

        warn_msgs = []
        cr_ctrl = conversions_control / max(n_control, 1)
        cr_var = conversions_variant / max(n_variant, 1)

        if n_control * cr_ctrl < 5 or n_control * (1 - cr_ctrl) < 5:
            warn_msgs.append("Expected counts < 5 in control. Use chi-square or Fisher's exact.")

        counts = np.array([conversions_variant, conversions_control])
        nobs = np.array([n_variant, n_control])

        z_stat, p_val = proportions_ztest(counts, nobs, alternative=alternative)

        risk_diff = cr_var - cr_ctrl
        se_rd = np.sqrt(
            cr_ctrl * (1 - cr_ctrl) / n_control
            + cr_var * (1 - cr_var) / n_variant
        )
        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci_lo, ci_hi = risk_diff - z_crit * se_rd, risk_diff + z_crit * se_rd

        # Effect size: Cohen's h (arcsine transformation of proportions)
        cohen_h = 2 * (np.arcsin(np.sqrt(cr_var)) - np.arcsin(np.sqrt(cr_ctrl)))
        magnitude = _cohens_d_magnitude(cohen_h)

        sig = bool(p_val < self.alpha)
        return TestResult(
            test_name="Z-test (proportions)",
            statistic=float(z_stat),
            p_value=float(p_val),
            alpha=self.alpha,
            significant=sig,
            effect_size=float(abs(cohen_h)),
            effect_size_label="Cohen's h",
            effect_magnitude=magnitude,
            ci_lower=float(ci_lo),
            ci_upper=float(ci_hi),
            ci_level=1 - self.alpha,
            interpretation=(
                f"CVR control={cr_ctrl:.2%}, variant={cr_var:.2%}. "
                f"Absolute diff={risk_diff:+.2%}. Cohen's h={cohen_h:.3f} ({magnitude})."
            ),
            recommendation="Ship variant." if sig and risk_diff > 0 else "Hold.",
            control_mean=float(cr_ctrl),
            variant_mean=float(cr_var),
            control_n=n_control,
            variant_n=n_variant,
            warnings=warn_msgs,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Multiple Testing Correction
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> dict:
        """
        Benjamini-Hochberg False Discovery Rate correction.

        Use when testing multiple metrics simultaneously (primary + guardrails).
        Controls expected proportion of false discoveries rather than family-wise error rate.

        Returns
        -------
        dict with:
          - adjusted_p_values : BH-adjusted p-values
          - significant       : boolean mask
          - fdr_threshold     : actual alpha threshold used
        """
        from statsmodels.stats.multitest import multipletests

        rejected, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
        return {
            "adjusted_p_values": p_adjusted.tolist(),
            "significant": rejected.tolist(),
            "method": "Benjamini-Hochberg FDR",
            "alpha_input": alpha,
        }

    @staticmethod
    def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> dict:
        """
        Bonferroni correction — controls family-wise error rate (FWER).

        More conservative than BH. Use when even one false positive is costly.
        """
        from statsmodels.stats.multitest import multipletests

        rejected, p_adjusted, _, alpha_corrected = multipletests(
            p_values, alpha=alpha, method="bonferroni"
        )
        return {
            "adjusted_p_values": p_adjusted.tolist(),
            "significant": rejected.tolist(),
            "method": "Bonferroni",
            "corrected_alpha": float(alpha_corrected),
        }
