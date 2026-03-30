"""
tests
─────
pytest test suite for the A/B Testing & Statistical Inference Dashboard.

Test modules
────────────

``test_statistical_tests.py``
    Covers every public class and function in ``src/``:

    TestWelchTTest          — Welch's t-test: significance, CI coverage, effect size
    TestChiSquare           — Chi-square: Yates correction, Cramér's V, risk difference
    TestMannWhitney         — Mann-Whitney: CLES range, stochastic dominance
    TestMultipleTesting     — BH FDR and Bonferroni correction outputs
    TestBayesianAB          — P(B>A), expected loss, posterior means, sequential update
    TestSampleSize          — Cohen's h, monotonicity of MDE/power curves
    TestSRM                 — SRM detection, edge cases, zero-total guard
    TestCUPED               — variance reduction, correlated vs. uncorrelated covariate
    TestSequentialAlpha     — O'Brien-Fleming boundary monotonicity
    TestDataGenerator       — binary output, revenue non-negativity, cumulative history
    TestPropertyBased       — Hypothesis property tests (always-positive n, valid posteriors)

Run the full suite::

    pytest                                        # all tests with coverage
    pytest -m frequentist                         # only frequentist tests
    pytest -m bayesian                            # only Bayesian tests
    pytest -m "not slow"                          # skip slow property-based tests
    pytest tests/test_statistical_tests.py -v     # verbose, single file

Coverage target: 80% (enforced in pyproject.toml via --cov-fail-under=80)
"""
