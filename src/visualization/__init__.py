"""
src.visualization
─────────────────
Plotly chart functions for LiftLab.

Every function in this package:
  - Accepts plain Python types and numpy arrays (no Streamlit dependency)
  - Returns a ``plotly.graph_objects.Figure`` ready for ``st.plotly_chart()``
  - Uses a consistent color scheme:
      control  →  #58a6ff  (blue)
      variant  →  #3fb950  (green)
  - Ships with a dark, WCAG-aware layout (see ``LAYOUT_BASE`` in ``plots.py``)

Public API
──────────

``plot_posterior_distributions(alpha_a, beta_a, alpha_b, beta_b)``
    Beta posterior density curves with credible interval shading.
    Used in: Section 2 — Bayesian A/B.

``plot_conversion_distributions(control, variant)``
    KDE density curves for comparing metric distributions.
    Used in: Section 1 — Frequentist Tests.

``plot_confidence_intervals(control_mean, …, variant_mean, …)``
    Forest-plot style CI bars for two groups.
    Used in: Section 1 — Frequentist Tests.

``plot_power_curve(power_data, required_n, target_power)``
    Power vs. sample size curve with threshold annotations.
    Used in: Section 3 — Sample Size & Power.

``plot_mde_curve(mde_data)``
    MDE vs. required sample size tradeoff curve.
    Used in: Section 3 — Sample Size & Power.

``plot_sequential_monitoring(history, early_stop_idx)``
    Dual-panel time series: CVR over time + P(B>A) over time.
    Used in: Section 5 — Sequential Monitoring.

``plot_expected_loss(loss_choose_b, loss_choose_a, loss_threshold)``
    Bar chart comparing expected loss for each shipping decision.
    Used in: Section 2 — Bayesian A/B.

``plot_srm_diagnostic(assigned_a, assigned_b, expected_ratio)``
    Grouped bar chart: observed vs. expected assignment counts.
    Used in: Section 4 — Data Quality (SRM).

Usage::

    from src.visualization import plot_posterior_distributions
    import streamlit as st

    fig = plot_posterior_distributions(
        alpha_a=145, beta_a=1057,
        alpha_b=175, beta_b=1027,
        ci_level=0.95,
    )
    st.plotly_chart(fig, use_container_width=True)
"""

from .plots import (
    plot_confidence_intervals,
    plot_conversion_distributions,
    plot_expected_loss,
    plot_mde_curve,
    plot_posterior_distributions,
    plot_power_curve,
    plot_sequential_monitoring,
    plot_srm_diagnostic,
)

__all__ = [
    "plot_posterior_distributions",   # Beta posteriors with CI shading (Bayesian)
    "plot_conversion_distributions",  # KDE density comparison (Frequentist)
    "plot_confidence_intervals",      # Forest-plot CI bars (Frequentist)
    "plot_power_curve",               # Power vs. sample size (Sample Size)
    "plot_mde_curve",                 # MDE vs. required n tradeoff (Sample Size)
    "plot_sequential_monitoring",     # CVR + P(B>A) time series (Sequential)
    "plot_expected_loss",             # Decision theory loss bars (Bayesian)
    "plot_srm_diagnostic",            # Observed vs. expected assignment (SRM)
]
