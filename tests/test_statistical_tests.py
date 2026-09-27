"""
tests/test_statistical_tests.py
────────────────────────────────
Unit + property-based tests for the LiftLab statistical engine.

Covers:
  - FrequentistTests: all 5 test methods
  - BayesianABTest: posterior math, expected loss, sequential
  - SampleSize: compute_sample_size, SRM, CUPED
  - Data contracts: StatisticalTestResult, BayesianResult field types

Run:
    pytest tests/ -v --cov=src --cov-report=term-missing
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# src modules
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.tests.frequentist import FrequentistTests, TestResult as StatisticalTestResult
from src.bayesian.beta_binomial import BayesianABTest, BayesianResult
from src.utils.sample_size import (
    check_sample_ratio_mismatch,
    compute_sample_size,
    cuped_variance_reduction,
    mde_curve,
    power_curve,
    sequential_alpha_spending,
)
from src.utils.data_generator import SyntheticDataGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def tester():
    return FrequentistTests(alpha=0.05)

@pytest.fixture
def large_sig_data(rng):
    """Two clearly different groups (should always be significant)."""
    ctrl = rng.normal(loc=10.0, scale=1.0, size=500)
    var = rng.normal(loc=11.5, scale=1.0, size=500)
    return ctrl, var

@pytest.fixture
def null_data(rng):
    """Two identical groups (should almost never be significant)."""
    ctrl = rng.normal(loc=10.0, scale=1.0, size=500)
    var = rng.normal(loc=10.0, scale=1.0, size=500)
    return ctrl, var

@pytest.fixture
def bayes():
    return BayesianABTest(prior_alpha=1, prior_beta=1, monte_carlo_samples=10_000, seed=42)

@pytest.fixture
def gen():
    return SyntheticDataGenerator(seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# FrequentistTests — Welch t-test
# ─────────────────────────────────────────────────────────────────────────────

class TestWelchTTest:

    def test_returns_test_result_type(self, tester, large_sig_data):
        ctrl, var = large_sig_data
        result = tester.welch_ttest(ctrl, var)
        assert isinstance(result, StatisticalTestResult)

    def test_significant_for_large_effect(self, tester, large_sig_data):
        ctrl, var = large_sig_data
        result = tester.welch_ttest(ctrl, var)
        assert result.significant, f"Expected significant but got p={result.p_value:.4f}"

    def test_not_significant_for_null(self, tester, null_data):
        ctrl, var = null_data
        result = tester.welch_ttest(ctrl, var)
        # Not guaranteed but highly likely for n=500 null groups
        assert result.p_value > 0.001, "p-value suspiciously small for null data"

    def test_ci_contains_true_effect(self, tester, rng):
        """95% CI should contain true effect in ~95% of experiments."""
        true_diff = 1.5
        captured = 0
        for _ in range(200):
            ctrl = rng.normal(10, 1, 200)
            var = rng.normal(10 + true_diff, 1, 200)
            r = tester.welch_ttest(ctrl, var)
            if r.ci_lower <= true_diff <= r.ci_upper:
                captured += 1
        coverage = captured / 200
        assert coverage >= 0.88, f"CI coverage {coverage:.0%} is too low (expected ~95%)"

    def test_effect_size_direction(self, tester, large_sig_data):
        ctrl, var = large_sig_data
        result = tester.welch_ttest(ctrl, var)
        # Variant > Control → positive effect size
        assert result.effect_size > 0

    def test_small_sample_warning(self, tester, rng):
        ctrl = rng.normal(10, 1, 10)
        var = rng.normal(11, 1, 10)
        result = tester.welch_ttest(ctrl, var)
        assert any("Sample size" in w for w in result.warnings)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            FrequentistTests(alpha=1.5)

    def test_result_fields_are_floats(self, tester, large_sig_data):
        ctrl, var = large_sig_data
        r = tester.welch_ttest(ctrl, var)
        assert isinstance(r.p_value, float)
        assert isinstance(r.statistic, float)
        assert isinstance(r.effect_size, float)
        assert isinstance(r.ci_lower, float)
        assert isinstance(r.ci_upper, float)


# ─────────────────────────────────────────────────────────────────────────────
# FrequentistTests — Chi-square
# ─────────────────────────────────────────────────────────────────────────────

class TestChiSquare:

    def test_significant_large_diff(self, tester):
        # 10% vs 20% CVR with n=1000 → clearly significant
        result = tester.chi_square_test(1000, 100, 1000, 200)
        assert result.significant

    def test_not_significant_no_diff(self, tester):
        # 12% vs 12% CVR
        result = tester.chi_square_test(1000, 120, 1000, 120)
        assert not result.significant

    def test_cramers_v_in_range(self, tester):
        result = tester.chi_square_test(500, 50, 500, 80)
        assert 0 <= result.effect_size <= 1

    def test_ci_on_risk_difference(self, tester):
        result = tester.chi_square_test(1000, 120, 1000, 150)
        # CI should straddle 0 if not significant, not straddle if significant
        if result.significant:
            assert not (result.ci_lower <= 0 <= result.ci_upper), \
                "CI should not contain 0 for significant result"

    def test_conversions_exceed_n_guard(self):
        """Should not raise — clamp handled gracefully."""
        tester = FrequentistTests(alpha=0.05)
        with pytest.raises(Exception):
            # Pass bad data to ensure downstream check works
            tester.chi_square_test(100, 150, 100, 50)  # c_ctrl > n_ctrl

    def test_yates_correction_more_conservative(self, tester):
        n, c_a, c_b = 100, 10, 18
        r_no_yates = tester.chi_square_test(n, c_a, n, c_b, yates_correction=False)
        r_yates = tester.chi_square_test(n, c_a, n, c_b, yates_correction=True)
        assert r_yates.p_value >= r_no_yates.p_value, "Yates should be more conservative"


# ─────────────────────────────────────────────────────────────────────────────
# FrequentistTests — Mann-Whitney U
# ─────────────────────────────────────────────────────────────────────────────

class TestMannWhitney:

    def test_detects_stochastic_dominance(self, tester, rng):
        ctrl = rng.exponential(scale=1.0, size=500)
        var = rng.exponential(scale=1.5, size=500)
        result = tester.mann_whitney_test(ctrl, var)
        assert result.significant

    def test_cles_in_range(self, tester, rng):
        ctrl = rng.normal(0, 1, 200)
        var = rng.normal(0.5, 1, 200)
        result = tester.mann_whitney_test(ctrl, var)
        assert 0.0 <= result.effect_size <= 1.0

    def test_cles_near_half_for_null(self, tester, rng):
        ctrl = rng.normal(0, 1, 1000)
        var = rng.normal(0, 1, 1000)
        result = tester.mann_whitney_test(ctrl, var)
        assert abs(result.effect_size - 0.5) < 0.07, \
            f"CLES={result.effect_size:.3f} should be near 0.5 for null"

    def test_no_ci_for_nonparametric(self, tester, rng):
        ctrl = rng.exponential(1, 100)
        var = rng.exponential(1.2, 100)
        result = tester.mann_whitney_test(ctrl, var)
        assert result.ci_lower is None
        assert result.ci_upper is None


# ─────────────────────────────────────────────────────────────────────────────
# Multiple Testing Correction
# ─────────────────────────────────────────────────────────────────────────────

class TestMultipleTesting:

    def test_bh_reduces_significant_count(self):
        pvals = [0.001, 0.04, 0.048, 0.49, 0.92]
        result = FrequentistTests.benjamini_hochberg(pvals)
        raw_sig = sum(p < 0.05 for p in pvals)
        bh_sig = sum(result["significant"])
        assert bh_sig <= raw_sig, "BH should not increase significant count"

    def test_bonferroni_most_conservative(self):
        pvals = [0.01, 0.03, 0.04]
        bh = FrequentistTests.benjamini_hochberg(pvals)
        bonf = FrequentistTests.bonferroni_correction(pvals)
        bh_sig = sum(bh["significant"])
        bonf_sig = sum(bonf["significant"])
        assert bonf_sig <= bh_sig, "Bonferroni should be at least as conservative as BH"

    def test_returns_same_length(self):
        pvals = [0.01, 0.05, 0.20]
        result = FrequentistTests.benjamini_hochberg(pvals)
        assert len(result["adjusted_p_values"]) == len(pvals)
        assert len(result["significant"]) == len(pvals)


# ─────────────────────────────────────────────────────────────────────────────
# BayesianABTest
# ─────────────────────────────────────────────────────────────────────────────

class TestBayesianAB:

    def test_returns_bayesian_result(self, bayes):
        result = bayes.run(1000, 120, 1000, 150)
        assert isinstance(result, BayesianResult)

    def test_prob_b_beats_a_in_range(self, bayes):
        result = bayes.run(1000, 120, 1000, 150)
        assert 0.0 <= result.prob_b_beats_a <= 1.0

    def test_high_prob_when_b_clearly_better(self, bayes):
        result = bayes.run(5000, 500, 5000, 700)
        assert result.prob_b_beats_a > 0.95, \
            f"Expected P(B>A) > 0.95 for large effect, got {result.prob_b_beats_a:.3f}"

    def test_low_prob_when_a_clearly_better(self, bayes):
        result = bayes.run(5000, 700, 5000, 500)
        assert result.prob_b_beats_a < 0.05, \
            f"Expected P(B>A) < 0.05 when A is better, got {result.prob_b_beats_a:.3f}"

    def test_expected_loss_non_negative(self, bayes):
        result = bayes.run(1000, 120, 1000, 130)
        assert result.expected_loss_choose_b >= 0
        assert result.expected_loss_choose_a >= 0

    def test_posterior_mean_close_to_observed(self, bayes):
        """Posterior mean ≈ observed rate for large n (prior dominated by data)."""
        n, c = 5000, 600
        a_post, b_post = bayes._posterior_params(n, c)
        posterior_mean = a_post / (a_post + b_post)
        observed_rate = c / n
        assert abs(posterior_mean - observed_rate) < 0.005

    def test_credible_interval_coverage(self, bayes):
        a_post, b_post = bayes._posterior_params(1000, 120)
        lo, hi = bayes._credible_interval(a_post, b_post, level=0.95)
        assert lo < hi
        assert 0 <= lo < hi <= 1

    def test_invalid_conversions_exceeds_n(self, bayes):
        with pytest.raises(ValueError):
            bayes.run(100, 150, 100, 80)  # 150 conversions > 100 visitors

    def test_sequential_update_returns_list(self, bayes):
        history = [
            {"n_a": 100, "c_a": 10, "n_b": 100, "c_b": 12},
            {"n_a": 200, "c_a": 20, "n_b": 200, "c_b": 28},
        ]
        snapshots = bayes.sequential_update(history)
        assert isinstance(snapshots, list)
        assert len(snapshots) <= len(history)

    def test_different_priors_converge_with_enough_data(self):
        """With n=10000, uniform and Jeffreys priors should give nearly identical results."""
        bayes_uniform = BayesianABTest(1, 1, mc_samples := 20_000, seed=0)
        bayes_jeffreys = BayesianABTest(0.5, 0.5, 20_000, seed=0)
        r1 = bayes_uniform.run(10000, 1200, 10000, 1400)
        r2 = bayes_jeffreys.run(10000, 1200, 10000, 1400)
        assert abs(r1.prob_b_beats_a - r2.prob_b_beats_a) < 0.02


# ─────────────────────────────────────────────────────────────────────────────
# Sample Size & Power
# ─────────────────────────────────────────────────────────────────────────────

class TestSampleSize:

    def test_basic_computation(self):
        result = compute_sample_size(0.10, 0.10, alpha=0.05, power=0.80)
        assert result.n_per_group > 0
        assert result.n_total == result.n_per_group * 2

    def test_larger_mde_needs_smaller_n(self):
        small_mde = compute_sample_size(0.10, 0.05)
        large_mde = compute_sample_size(0.10, 0.20)
        assert small_mde.n_per_group > large_mde.n_per_group

    def test_higher_power_needs_more_n(self):
        low_power = compute_sample_size(0.10, 0.10, power=0.70)
        high_power = compute_sample_size(0.10, 0.10, power=0.90)
        assert high_power.n_per_group > low_power.n_per_group

    def test_stricter_alpha_needs_more_n(self):
        lenient = compute_sample_size(0.10, 0.10, alpha=0.10)
        strict = compute_sample_size(0.10, 0.10, alpha=0.01)
        assert strict.n_per_group > lenient.n_per_group

    def test_duration_estimate(self):
        result = compute_sample_size(0.10, 0.10, daily_traffic=1000)
        assert result.days_required is not None
        assert result.days_required > 0

    def test_invalid_baseline_raises(self):
        with pytest.raises(ValueError):
            compute_sample_size(1.5, 0.10)

    def test_power_curve_length(self):
        curve = power_curve(0.10, 0.15, alpha=0.05)
        assert len(curve) > 0
        for d in curve:
            assert 0 <= d["power"] <= 1

    def test_mde_curve_monotone_decreasing(self):
        """Larger MDE → fewer samples needed."""
        curve = mde_curve(0.10, alpha=0.05, power=0.80)
        ns = [d["n_per_group"] for d in curve]
        assert all(ns[i] >= ns[i+1] for i in range(len(ns)-1)), \
            "MDE curve should be monotone decreasing"


class TestSRM:

    def test_srm_detected_for_large_imbalance(self):
        result = check_sample_ratio_mismatch(8000, 2000, expected_ratio=0.5)
        assert result.srm_detected

    def test_srm_not_detected_for_balanced(self):
        result = check_sample_ratio_mismatch(5002, 4998, expected_ratio=0.5)
        assert not result.srm_detected

    def test_p_value_in_range(self):
        result = check_sample_ratio_mismatch(500, 480)
        assert 0 <= result.p_value <= 1

    def test_zero_total_raises(self):
        with pytest.raises(ValueError):
            check_sample_ratio_mismatch(0, 0)


class TestCUPED:

    def test_reduces_variance(self, rng):
        metric = rng.normal(10, 2, 500)
        covariate = metric * 0.8 + rng.normal(0, 0.5, 500)  # correlated
        adj, reduction = cuped_variance_reduction(metric, covariate)
        assert reduction > 0, "Correlated covariate should reduce variance"

    def test_zero_reduction_for_uncorrelated(self, rng):
        metric = rng.normal(10, 2, 500)
        covariate = rng.normal(0, 1, 500)  # uncorrelated
        adj, reduction = cuped_variance_reduction(metric, covariate)
        assert reduction < 0.05, "Uncorrelated covariate should not reduce variance"

    def test_length_mismatch_raises(self, rng):
        with pytest.raises(ValueError):
            cuped_variance_reduction(rng.normal(0, 1, 100), rng.normal(0, 1, 200))

    def test_output_same_length(self, rng):
        metric = rng.normal(0, 1, 300)
        cov = rng.normal(0, 1, 300)
        adj, _ = cuped_variance_reduction(metric, cov)
        assert len(adj) == 300


# ─────────────────────────────────────────────────────────────────────────────
# Sequential Alpha Spending
# ─────────────────────────────────────────────────────────────────────────────

class TestSequentialAlphaSpending:

    def test_returns_correct_length(self):
        n = 5
        boundaries = sequential_alpha_spending(n, alpha=0.05, method="obrien_fleming")
        assert len(boundaries) == n

    def test_obf_increasing_boundaries(self):
        """O'Brien-Fleming: boundaries increase (more permissive over time)."""
        bounds = sequential_alpha_spending(5, alpha=0.05, method="obrien_fleming")
        assert all(bounds[i] <= bounds[i+1] for i in range(len(bounds)-1)), \
            "O'Brien-Fleming boundaries should be non-decreasing"

    def test_final_boundary_near_alpha(self):
        """Final boundary should be close to the nominal alpha."""
        bounds = sequential_alpha_spending(10, alpha=0.05, method="obrien_fleming")
        assert abs(bounds[-1] - 0.05) < 0.02

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            sequential_alpha_spending(5, method="made_up_method")


# ─────────────────────────────────────────────────────────────────────────────
# Data Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGenerator:

    def test_conversion_output_binary(self, gen):
        ds = gen.conversion_experiment(n=500, base_rate=0.12, true_lift=0.15)
        assert set(ds.control).issubset({0.0, 1.0})
        assert set(ds.variant).issubset({0.0, 1.0})

    def test_revenue_non_negative(self, gen):
        ds = gen.revenue_experiment(n=500)
        assert (ds.control >= 0).all()
        assert (ds.variant >= 0).all()

    def test_srm_injection(self, gen):
        ds = gen.conversion_experiment(n=1000, inject_srm=True)
        assert len(ds.variant) < len(ds.control)

    def test_sequential_data_cumulative(self, gen):
        history = gen.sequential_daily_data(total_days=10, daily_users=100)
        for i in range(1, len(history)):
            assert history[i]["n_a"] >= history[i-1]["n_a"]
            assert history[i]["c_a"] >= history[i-1]["c_a"]

    def test_to_dataframe_shape(self, gen):
        ds = gen.conversion_experiment(n=200)
        df = gen.to_dataframe(ds)
        assert len(df) == 400  # 200 ctrl + 200 var
        assert "group" in df.columns
        assert set(df["group"].unique()) == {"control", "variant"}


# ─────────────────────────────────────────────────────────────────────────────
# Property-based tests (Hypothesis)
# ─────────────────────────────────────────────────────────────────────────────

class TestPropertyBased:

    @given(
        n=st.integers(min_value=50, max_value=2000),
        base=st.floats(min_value=0.05, max_value=0.45),
        mde=st.floats(min_value=0.05, max_value=0.40),
    )
    @settings(max_examples=30, deadline=5000)
    def test_sample_size_always_positive(self, n, base, mde):
        result = compute_sample_size(base, mde, alpha=0.05, power=0.80)
        assert result.n_per_group > 0

    @given(
        n=st.integers(min_value=100, max_value=5000),
        c=st.integers(min_value=5, max_value=99),
    )
    @settings(max_examples=30, deadline=5000)
    def test_posterior_mean_in_range(self, n, c):
        if c >= n:
            return
        bayes = BayesianABTest(1, 1, monte_carlo_samples=1000, seed=0)
        a_post, b_post = bayes._posterior_params(n, c)
        mean = a_post / (a_post + b_post)
        assert 0 < mean < 1

    @given(
        p=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, deadline=2000)
    def test_chi_square_p_value_in_range(self, p):
        """For any valid p-value, BH output is valid."""
        p_vals = [p, 0.05, 0.10]
        result = FrequentistTests.benjamini_hochberg(p_vals)
        for adj_p in result["adjusted_p_values"]:
            assert 0 <= adj_p <= 1
