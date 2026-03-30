"""
conftest.py
────────────
pytest shared fixtures and global configuration.

This file is auto-loaded by pytest before any test module runs.
Fixtures defined here are available to ALL test files without importing.

Fixtures provided
─────────────────
  rng                 — seeded numpy default_rng for reproducible test data
  tester              — FrequentistTests(alpha=0.05) ready to use
  bayes               — BayesianABTest with 5k MC samples (fast for tests)
  gen                 — SyntheticDataGenerator(seed=42)
  small_effect_data   — two groups with a small but detectable difference
  large_effect_data   — two groups with a large, always-significant difference
  null_data           — two identical groups (should rarely be significant)
  binary_counts       — (n_ctrl, c_ctrl, n_var, c_var) for proportion tests
"""

from __future__ import annotations

import numpy as np
import pytest

from src.bayesian.beta_binomial import BayesianABTest
from src.tests.frequentist import FrequentistTests
from src.utils.data_generator import SyntheticDataGenerator


# ─────────────────────────────────────────────────────────────────────────────
# Primitive fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """
    Session-scoped seeded RNG.
    Re-used across all tests — same seed guarantees reproducible data.
    Use scope="session" so the RNG state is shared (faster than per-test).
    """
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def tester() -> FrequentistTests:
    """
    FrequentistTests instance at standard α = 0.05.
    Session-scoped — stateless class, safe to share.
    """
    return FrequentistTests(alpha=0.05)


@pytest.fixture(scope="session")
def bayes() -> BayesianABTest:
    """
    BayesianABTest with uniform prior Beta(1,1) and 5,000 MC samples.
    Reduced sample count (vs. production 50k) keeps tests fast.
    Session-scoped — deterministic with fixed seed.
    """
    return BayesianABTest(
        prior_alpha=1.0,
        prior_beta=1.0,
        monte_carlo_samples=5_000,
        seed=42,
    )


@pytest.fixture(scope="session")
def gen() -> SyntheticDataGenerator:
    """
    SyntheticDataGenerator with fixed seed for reproducible datasets.
    Session-scoped — stateless generator.
    """
    return SyntheticDataGenerator(seed=42)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def large_effect_data(rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Two groups with a large, always-detectable difference.
    Control: N(10, 1),  Variant: N(12, 1),  n=500 each.
    Cohen's d ≈ 2.0 — this should ALWAYS be significant.
    """
    ctrl = rng.normal(loc=10.0, scale=1.0, size=500)
    var  = rng.normal(loc=12.0, scale=1.0, size=500)
    return ctrl, var


@pytest.fixture(scope="session")
def small_effect_data(rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Two groups with a small but real difference.
    Control: N(10, 1),  Variant: N(10.3, 1),  n=500 each.
    Cohen's d ≈ 0.3 — detectable at n=500 with 80% power.
    """
    ctrl = rng.normal(loc=10.0, scale=1.0, size=500)
    var  = rng.normal(loc=10.3, scale=1.0, size=500)
    return ctrl, var


@pytest.fixture(scope="session")
def null_data(rng) -> tuple[np.ndarray, np.ndarray]:
    """
    Two identical groups — ground truth H₀ is true.
    Used to verify Type-I error rate behaves correctly (p-values should be
    uniformly distributed; significant ≈ 5% of the time across many runs).
    """
    ctrl = rng.normal(loc=10.0, scale=1.0, size=500)
    var  = rng.normal(loc=10.0, scale=1.0, size=500)
    return ctrl, var


@pytest.fixture(scope="session")
def binary_counts() -> dict:
    """
    Pre-computed conversion counts for proportion tests.
    Returns a dict with keys: n_ctrl, c_ctrl, n_var, c_var, cr_ctrl, cr_var.

    Scenario:
      Control:  1000 visitors, 120 conversions  → 12.0% CVR
      Variant:  1000 visitors, 155 conversions  → 15.5% CVR
      Relative lift: +29.2%  (large, clearly significant at n=1000)
    """
    n_ctrl, c_ctrl = 1000, 120
    n_var,  c_var  = 1000, 155
    return {
        "n_ctrl":  n_ctrl,
        "c_ctrl":  c_ctrl,
        "n_var":   n_var,
        "c_var":   c_var,
        "cr_ctrl": c_ctrl / n_ctrl,
        "cr_var":  c_var  / n_var,
    }


@pytest.fixture(scope="session")
def null_binary_counts() -> dict:
    """
    Equal conversion counts — ground truth H₀ is true.
    Control: 1000 visitors, 120 conversions  → 12.0% CVR
    Variant: 1000 visitors, 120 conversions  → 12.0% CVR
    """
    return {
        "n_ctrl": 1000, "c_ctrl": 120,
        "n_var":  1000, "c_var":  120,
    }


@pytest.fixture(scope="session")
def sequential_history(gen) -> list[dict]:
    """
    30-day cumulative conversion history for sequential analysis tests.
    True lift = 15%,  base CVR = 10%,  200 users/day.
    """
    return gen.sequential_daily_data(
        total_days=30,
        daily_users=200,
        base_rate=0.10,
        true_lift=0.15,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pytest configuration hooks
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to suppress PytestUnknownMarkWarning."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with -m 'not slow')",
    )
    config.addinivalue_line(
        "markers",
        "integration: mark as integration test requiring external services",
    )
    config.addinivalue_line(
        "markers",
        "bayesian: mark as Bayesian inference test (Monte Carlo)",
    )
    config.addinivalue_line(
        "markers",
        "frequentist: mark as frequentist statistical test",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Auto-apply markers based on test class names.
    Tests in TestBayesianAB  → @pytest.mark.bayesian
    Tests in TestWelch* etc  → @pytest.mark.frequentist
    Property-based tests     → @pytest.mark.slow
    """
    for item in items:
        cls_name = item.cls.__name__ if item.cls else ""

        if "Bayesian" in cls_name:
            item.add_marker(pytest.mark.bayesian)

        if any(k in cls_name for k in ("Welch", "Chi", "Mann", "Student",
                                        "ZTest", "MultipleTest", "Frequentist")):
            item.add_marker(pytest.mark.frequentist)

        if "PropertyBased" in cls_name or "Property" in cls_name:
            item.add_marker(pytest.mark.slow)
