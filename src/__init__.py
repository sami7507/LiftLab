"""
src
───
A/B Testing & Statistical Inference Dashboard — source package.

Sub-packages
────────────
  src.tests          Frequentist statistical tests
                     (Welch t, Student t, Chi-square, Mann-Whitney U, Z-test)

  src.bayesian       Bayesian A/B testing
                     (Beta-Binomial conjugate model, Monte Carlo inference)

  src.utils          Supporting utilities
                     (sample size, power analysis, SRM detection, CUPED,
                      synthetic data generation)

  src.visualization  Plotly chart functions
                     (posteriors, CIs, power curve, MDE, sequential monitoring)

Typical imports::

    from src.tests.frequentist       import FrequentistTests, TestResult
    from src.bayesian.beta_binomial  import BayesianABTest, BayesianResult
    from src.utils.sample_size       import compute_sample_size
    from src.utils.data_generator    import SyntheticDataGenerator
    from src.visualization.plots     import plot_posterior_distributions
"""

__version__ = "2.0.0"
__author__  = "Your Name"

__all__ = ["__version__", "__author__"]
