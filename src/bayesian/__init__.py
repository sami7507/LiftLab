"""
src.bayesian
────────────
Bayesian A/B testing using the Beta-Binomial conjugate model.

Public API
──────────

``BayesianABTest``
    Main inference class. Uses Monte Carlo sampling from Beta posteriors
    to estimate P(B > A) and expected loss — no MCMC or external sampler
    required (conjugate model gives exact posteriors).

    Key methods:
      - ``run(n_a, c_a, n_b, c_b)``  — full Bayesian analysis, returns BayesianResult
      - ``sequential_update(history)``— day-by-day posterior updates with early stop
      - ``posterior_samples(a, b)``   — raw Beta samples for custom plotting

``BayesianResult``
    Typed dataclass returned by ``BayesianABTest.run()``. Fields include:
    prob_b_beats_a, expected_loss_choose_b, expected_loss_choose_a,
    posterior_mean_a/b, posterior_std_a/b, ci_lower/upper_a/b,
    alpha_a/b, beta_a/b, relative_uplift, absolute_uplift,
    recommendation, confidence_label, warnings.

Statistical model::

    Prior:      θ  ~  Beta(prior_alpha, prior_beta)
    Likelihood: k  ~  Binomial(n, θ)
    Posterior:  θ | k, n  ~  Beta(prior_alpha + k, prior_beta + n - k)

    P(B > A) = Monte Carlo estimate from 50,000 posterior samples
    Expected loss = E[max(θ_A - θ_B, 0)]  (opportunity cost of choosing B)

Usage::

    from src.bayesian import BayesianABTest, BayesianResult

    model = BayesianABTest(prior_alpha=1, prior_beta=1, monte_carlo_samples=50_000)
    result: BayesianResult = model.run(
        n_a=1200, conversions_a=144,
        n_b=1200, conversions_b=174,
    )
    print(f"P(B > A) = {result.prob_b_beats_a:.1%}")
    print(f"Expected loss = {result.expected_loss_choose_b:.4%}")
    print(result.recommendation)
"""

from .beta_binomial import BayesianABTest, BayesianResult

__all__ = [
    "BayesianABTest",   # main Bayesian inference class
    "BayesianResult",   # typed result dataclass returned by .run()
]
