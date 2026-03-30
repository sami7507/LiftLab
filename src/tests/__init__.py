"""
src.tests
─────────
Frequentist statistical testing module.

Public API
──────────

``FrequentistTests``
    Main class. Instantiate once with a chosen alpha level, then call any
    test method. All methods return a ``TestResult`` dataclass.

    Methods:
      - ``welch_ttest(control, variant)``       — Welch's t-test (unequal variance)
      - ``student_ttest(control, variant)``     — Student's t-test (equal variance)
      - ``chi_square_test(n_ctrl, c_ctrl, …)``  — Pearson chi-square (binary outcomes)
      - ``z_test_proportions(n_ctrl, c_ctrl, …)``— Two-proportion z-test
      - ``mann_whitney_test(control, variant)`` — Mann-Whitney U (non-parametric)
      - ``benjamini_hochberg(p_values)``        — BH FDR correction (static method)
      - ``bonferroni_correction(p_values)``     — Bonferroni FWER correction (static)

``TestResult``
    Typed dataclass returned by every test method. Fields include:
    test_name, statistic, p_value, significant, effect_size,
    effect_size_label, effect_magnitude, ci_lower, ci_upper,
    interpretation, recommendation, warnings.

Usage::

    from src.tests import FrequentistTests, TestResult
    import numpy as np

    tester = FrequentistTests(alpha=0.05)
    result: TestResult = tester.welch_ttest(control_data, variant_data)
    print(result.significant, result.p_value, result.effect_magnitude)

    # Binary outcome (conversion rate)
    result = tester.chi_square_test(
        n_control=1000, conversions_control=120,
        n_variant=1000,  conversions_variant=150,
    )
    print(result.recommendation)  # "Ship variant." or "Hold."
"""

from .frequentist import FrequentistTests, TestResult

__all__ = [
    "FrequentistTests",  # main testing class
    "TestResult",        # typed result dataclass returned by all test methods
]
