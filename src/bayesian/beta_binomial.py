"""
src/bayesian/beta_binomial.py
──────────────────────────────
Bayesian A/B testing using the Beta-Binomial conjugate model. Part of LiftLab.

Why Bayesian?
  - Gives P(B beats A) — a direct business-friendly probability
  - No fixed sample size required; posterior updates incrementally
  - Expected loss framework answers "what do we lose if we're wrong?"
  - Credible intervals have direct probability interpretation unlike CIs

Model
─────
  Prior:    θ ~ Beta(α₀, β₀)           (belief before seeing data)
  Likelihood: k | n, θ ~ Binomial(n, θ) (observed conversions)
  Posterior:  θ | k, n ~ Beta(α₀+k, β₀+n-k)  (updated belief)

The Beta-Binomial is the exact conjugate — no MCMC required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy import stats
from scipy.special import betaln


# ─────────────────────────────────────────────────────────────────────────────
# Data Contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BayesianResult:
    """Full output of a Bayesian A/B test."""

    # Core inference
    prob_b_beats_a: float           # P(θ_B > θ_A | data) via Monte Carlo
    expected_loss_choose_b: float   # E[max(θ_A - θ_B, 0)] — loss if we deploy B and A was better
    expected_loss_choose_a: float   # E[max(θ_B - θ_A, 0)] — loss if we keep A and B was better

    # Posterior summaries
    posterior_mean_a: float
    posterior_mean_b: float
    posterior_std_a: float
    posterior_std_b: float

    # Credible intervals
    ci_lower_a: float
    ci_upper_a: float
    ci_lower_b: float
    ci_upper_b: float
    ci_level: float

    # Posterior parameters (for plotting)
    alpha_a: float
    beta_a: float
    alpha_b: float
    beta_b: float

    # Uplift
    relative_uplift: float          # (mean_B - mean_A) / mean_A
    absolute_uplift: float          # mean_B - mean_A

    # Decision
    recommendation: str
    confidence_label: str           # "High" | "Medium" | "Low"

    # Prior used
    prior_alpha: float
    prior_beta: float

    # Raw inputs
    n_a: int
    conversions_a: int
    n_b: int
    conversions_b: int

    warnings: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main Class
# ─────────────────────────────────────────────────────────────────────────────

class BayesianABTest:
    """
    Bayesian A/B testing with Beta-Binomial conjugate model.

    Supports:
      - Jeffreys prior (non-informative): Beta(0.5, 0.5)  — recommended default
      - Uniform prior (Laplace):          Beta(1, 1)
      - Informed prior from historical data

    Usage
    -----
    >>> test = BayesianABTest(prior_alpha=1, prior_beta=1)
    >>> result = test.run(n_a=1000, conversions_a=120, n_b=1000, conversions_b=145)
    >>> print(f"P(B beats A) = {result.prob_b_beats_a:.1%}")
    """

    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        monte_carlo_samples: int = 100_000,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        prior_alpha : α parameter of Beta prior (successes + pseudo-count)
        prior_beta  : β parameter of Beta prior (failures  + pseudo-count)
        monte_carlo_samples : number of samples for P(B>A) estimation
        seed : random seed for reproducibility
        """
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Prior parameters must be positive.")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.mc_samples = monte_carlo_samples
        self.rng = np.random.default_rng(seed)
        logger.info(
            f"BayesianABTest init: prior=Beta({prior_alpha}, {prior_beta}), "
            f"MC_samples={monte_carlo_samples}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Posterior Update (conjugate — exact, no MCMC)
    # ─────────────────────────────────────────────────────────────────────────

    def _posterior_params(self, n: int, conversions: int) -> tuple[float, float]:
        """
        Return (alpha_post, beta_post) after observing data.

        Prior:     Beta(prior_alpha, prior_beta)
        Posterior: Beta(prior_alpha + conversions, prior_beta + n - conversions)
        """
        alpha_post = self.prior_alpha + conversions
        beta_post = self.prior_beta + (n - conversions)
        return alpha_post, beta_post

    # ─────────────────────────────────────────────────────────────────────────
    # P(B > A)  via Monte Carlo
    # ─────────────────────────────────────────────────────────────────────────

    def _prob_b_beats_a(
        self, alpha_a: float, beta_a: float, alpha_b: float, beta_b: float
    ) -> float:
        """
        Estimate P(θ_B > θ_A) via Monte Carlo sampling from posteriors.

        Analytical formula exists for the Beta case but is numerically
        unstable for large α, β. MC is fast, exact to 3 decimal places
        with 100k samples, and trivially extensible to non-conjugate models.
        """
        theta_a = self.rng.beta(alpha_a, beta_a, self.mc_samples)
        theta_b = self.rng.beta(alpha_b, beta_b, self.mc_samples)
        return float(np.mean(theta_b > theta_a))

    # ─────────────────────────────────────────────────────────────────────────
    # Expected Loss  (decision-theoretic stopping criterion)
    # ─────────────────────────────────────────────────────────────────────────

    def _expected_loss(
        self,
        alpha_a: float,
        beta_a: float,
        alpha_b: float,
        beta_b: float,
    ) -> tuple[float, float]:
        """
        Compute expected loss for both choices.

        loss(choose B) = E[max(θ_A - θ_B, 0)]  — B might be worse than A
        loss(choose A) = E[max(θ_B - θ_A, 0)]  — A might be worse than B

        Decision rule: deploy B when expected_loss(choose_B) < threshold (e.g. 0.001)
        This is the "regret" or "opportunity cost" framing from Bayesian decision theory.
        """
        theta_a = self.rng.beta(alpha_a, beta_a, self.mc_samples)
        theta_b = self.rng.beta(alpha_b, beta_b, self.mc_samples)

        loss_choose_b = float(np.mean(np.maximum(theta_a - theta_b, 0.0)))
        loss_choose_a = float(np.mean(np.maximum(theta_b - theta_a, 0.0)))
        return loss_choose_b, loss_choose_a

    # ─────────────────────────────────────────────────────────────────────────
    # Credible Interval
    # ─────────────────────────────────────────────────────────────────────────

    def _credible_interval(
        self, alpha_post: float, beta_post: float, level: float = 0.95
    ) -> tuple[float, float]:
        """
        Highest Posterior Density (HPD) interval using Beta quantiles.

        Unlike frequentist CIs, we can say:
        "There is a 95% probability that the true rate lies in [lo, hi]."
        """
        lo = (1 - level) / 2
        hi = 1 - lo
        return (
            float(stats.beta.ppf(lo, alpha_post, beta_post)),
            float(stats.beta.ppf(hi, alpha_post, beta_post)),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Entry Point
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        n_a: int,
        conversions_a: int,
        n_b: int,
        conversions_b: int,
        ci_level: float = 0.95,
        loss_threshold: float = 0.005,
    ) -> BayesianResult:
        """
        Run a full Bayesian A/B test.

        Parameters
        ----------
        n_a, conversions_a   : control group visitors and conversions
        n_b, conversions_b   : variant group visitors and conversions
        ci_level             : credible interval coverage (default 95%)
        loss_threshold       : max acceptable expected loss before shipping

        Returns
        -------
        BayesianResult with all inference outputs
        """
        warnings_list: list[str] = []

        # Input validation
        if conversions_a > n_a or conversions_b > n_b:
            raise ValueError("Conversions cannot exceed total visitors.")
        if n_a < 10 or n_b < 10:
            warnings_list.append("Very small sample. Posterior will be dominated by prior.")

        # Posterior parameters
        alpha_a, beta_a = self._posterior_params(n_a, conversions_a)
        alpha_b, beta_b = self._posterior_params(n_b, conversions_b)

        # Core quantities
        prob_b_beats_a = self._prob_b_beats_a(alpha_a, beta_a, alpha_b, beta_b)
        loss_choose_b, loss_choose_a = self._expected_loss(alpha_a, beta_a, alpha_b, beta_b)

        # Posterior summaries
        mean_a = alpha_a / (alpha_a + beta_a)
        mean_b = alpha_b / (alpha_b + beta_b)
        std_a = np.sqrt(alpha_a * beta_a / ((alpha_a + beta_a)**2 * (alpha_a + beta_a + 1)))
        std_b = np.sqrt(alpha_b * beta_b / ((alpha_b + beta_b)**2 * (alpha_b + beta_b + 1)))

        ci_a = self._credible_interval(alpha_a, beta_a, ci_level)
        ci_b = self._credible_interval(alpha_b, beta_b, ci_level)

        # Uplift
        absolute_uplift = mean_b - mean_a
        relative_uplift = absolute_uplift / max(mean_a, 1e-9)

        # Decision logic
        if prob_b_beats_a >= 0.95 and loss_choose_b <= loss_threshold:
            recommendation = f"Ship variant B. P(B>A)={prob_b_beats_a:.1%}, loss={loss_choose_b:.4%}"
            confidence_label = "High"
        elif prob_b_beats_a >= 0.80:
            recommendation = f"Lean toward B but collect more data. P(B>A)={prob_b_beats_a:.1%}"
            confidence_label = "Medium"
        elif prob_b_beats_a <= 0.10:
            recommendation = "Keep A. Strong evidence variant B is worse."
            confidence_label = "High"
        else:
            recommendation = "Inconclusive. Continue experiment."
            confidence_label = "Low"

        logger.info(
            f"Bayesian run complete: P(B>A)={prob_b_beats_a:.3f}, "
            f"loss_B={loss_choose_b:.5f}, {recommendation}"
        )
        return BayesianResult(
            prob_b_beats_a=prob_b_beats_a,
            expected_loss_choose_b=loss_choose_b,
            expected_loss_choose_a=loss_choose_a,
            posterior_mean_a=float(mean_a),
            posterior_mean_b=float(mean_b),
            posterior_std_a=float(std_a),
            posterior_std_b=float(std_b),
            ci_lower_a=ci_a[0],
            ci_upper_a=ci_a[1],
            ci_lower_b=ci_b[0],
            ci_upper_b=ci_b[1],
            ci_level=ci_level,
            alpha_a=float(alpha_a),
            beta_a=float(beta_a),
            alpha_b=float(alpha_b),
            beta_b=float(beta_b),
            relative_uplift=float(relative_uplift),
            absolute_uplift=float(absolute_uplift),
            recommendation=recommendation,
            confidence_label=confidence_label,
            prior_alpha=self.prior_alpha,
            prior_beta=self.prior_beta,
            n_a=n_a,
            conversions_a=conversions_a,
            n_b=n_b,
            conversions_b=conversions_b,
            warnings=warnings_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Sequential Monitoring  (update posterior as data arrives)
    # ─────────────────────────────────────────────────────────────────────────

    def sequential_update(
        self,
        history: list[dict],  # [{n_a, c_a, n_b, c_b}, ...]  cumulative snapshots
        loss_threshold: float = 0.005,
        early_stop_prob: float = 0.95,
    ) -> list[dict]:
        """
        Run Bayesian updating across a time series of cumulative data.

        Enables continuous monitoring without the p-value inflation of
        frequentist sequential testing. The posterior legitimately updates
        with each new batch — no correction needed.

        Parameters
        ----------
        history : list of dicts, each with keys n_a, c_a, n_b, c_b (cumulative)
        loss_threshold : stop when expected loss drops below this
        early_stop_prob : stop when P(B>A) exceeds this

        Returns
        -------
        List of snapshots with posterior stats and stop flag
        """
        snapshots = []
        for i, snap in enumerate(history):
            result = self.run(snap["n_a"], snap["c_a"], snap["n_b"], snap["c_b"])
            stop = (
                result.prob_b_beats_a >= early_stop_prob
                and result.expected_loss_choose_b <= loss_threshold
            )
            snapshots.append({
                "day": i + 1,
                "n_a": snap["n_a"],
                "n_b": snap["n_b"],
                "prob_b_beats_a": result.prob_b_beats_a,
                "expected_loss": result.expected_loss_choose_b,
                "mean_a": result.posterior_mean_a,
                "mean_b": result.posterior_mean_b,
                "early_stop": stop,
                "recommendation": result.recommendation,
            })
            if stop:
                logger.info(f"Early stop triggered at snapshot {i+1}.")
                break
        return snapshots

    # ─────────────────────────────────────────────────────────────────────────
    # Posterior Predictive Samples  (for plotting)
    # ─────────────────────────────────────────────────────────────────────────

    def posterior_samples(
        self,
        alpha_post: float,
        beta_post: float,
        n_samples: int = 10_000,
    ) -> np.ndarray:
        """Draw samples from the posterior for density plotting."""
        return self.rng.beta(alpha_post, beta_post, n_samples)
