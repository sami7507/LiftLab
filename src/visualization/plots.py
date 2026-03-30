"""
src/visualization/plots.py
───────────────────────────
All Plotly visualisation functions for the A/B Testing Dashboard.

Design principles:
  - Each function returns a go.Figure (composable, testable)
  - Consistent dark color scheme: control=#58a6ff (blue), variant=#3fb950 (green)
  - WCAG-compliant contrast on dark backgrounds
  - Minimal chartjunk — every pixel earns its place
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# Color palette — bright colors that pop on dark #161b27 background
# ─────────────────────────────────────────────────────────────────────────────
CTRL_COLOR = "#58a6ff"              # bright blue
VAR_COLOR  = "#3fb950"              # bright green
CTRL_FILL  = "rgba(88,166,255,0.15)"
VAR_FILL   = "rgba(63,185,80,0.15)"
NEUTRAL    = "#8b949e"
WARN       = "#d29922"
DANGER     = "#f85149"
GRID_COLOR = "rgba(255,255,255,0.06)"
BG_DARK    = "#161b27"
BG_CARD    = "#0e1117"

# ─────────────────────────────────────────────────────────────────────────────
# Shared layout helpers
# ─────────────────────────────────────────────────────────────────────────────

# LAYOUT_BASE: only keys that are safe to unpack in EVERY update_layout call.
# Never put title / xaxis / yaxis here — each chart passes those individually
# and Python raises "multiple values for keyword argument" if they appear twice.
LAYOUT_BASE = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_DARK,
    font=dict(family="sans-serif", size=12, color="#c9d1d9"),
    margin=dict(l=50, r=20, t=44, b=40),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color="#8b949e", size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
    title_font=dict(color="#e6edf3", size=13),
    hoverlabel=dict(
        bgcolor="#1c2233",
        bordercolor="#2a2f3e",
        font=dict(color="#e6edf3", size=12),
    ),
)

# Shared axis style dict — applied via update_xaxes / update_yaxes
_AXIS = dict(
    gridcolor=GRID_COLOR,
    linecolor="#2a2f3e",
    tickcolor="#484f58",
    tickfont=dict(color="#8b949e", size=11),
    title_font=dict(color="#8b949e", size=11),
    zerolinecolor="#2a2f3e",
    showgrid=True,
)


def _dark(fig: go.Figure) -> go.Figure:
    """Apply dark axis styling to every axis on the figure."""
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Posterior Distribution Plot (Bayesian)
# ─────────────────────────────────────────────────────────────────────────────

def plot_posterior_distributions(
    alpha_a: float, beta_a: float,
    alpha_b: float, beta_b: float,
    ci_level: float = 0.95,
) -> go.Figure:
    """Beta posterior density curves with credible interval shading."""
    lo = min(stats.beta.ppf(0.001, alpha_a, beta_a), stats.beta.ppf(0.001, alpha_b, beta_b))
    hi = max(stats.beta.ppf(0.999, alpha_a, beta_a), stats.beta.ppf(0.999, alpha_b, beta_b))
    x = np.linspace(max(lo - 0.01, 0), min(hi + 0.01, 1), 500)

    pdf_a = stats.beta.pdf(x, alpha_a, beta_a)
    pdf_b = stats.beta.pdf(x, alpha_b, beta_b)

    ci_lo_a = stats.beta.ppf((1 - ci_level) / 2, alpha_a, beta_a)
    ci_hi_a = stats.beta.ppf(1 - (1 - ci_level) / 2, alpha_a, beta_a)
    ci_lo_b = stats.beta.ppf((1 - ci_level) / 2, alpha_b, beta_b)
    ci_hi_b = stats.beta.ppf(1 - (1 - ci_level) / 2, alpha_b, beta_b)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=pdf_a, name="Control A",
        line=dict(color=CTRL_COLOR, width=2),
        fill="tozeroy", fillcolor=CTRL_FILL,
        hovertemplate="p=%{x:.4f}<br>density=%{y:.2f}<extra>Control A</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=pdf_b, name="Variant B",
        line=dict(color=VAR_COLOR, width=2),
        fill="tozeroy", fillcolor=VAR_FILL,
        hovertemplate="p=%{x:.4f}<br>density=%{y:.2f}<extra>Variant B</extra>",
    ))

    # CI shading
    for x_ci, pdf_fn, a, b, clr in [
        (x[(x >= ci_lo_a) & (x <= ci_hi_a)], lambda xx: stats.beta.pdf(xx, alpha_a, beta_a),
         alpha_a, beta_a, "rgba(88,166,255,0.22)"),
        (x[(x >= ci_lo_b) & (x <= ci_hi_b)], lambda xx: stats.beta.pdf(xx, alpha_b, beta_b),
         alpha_b, beta_b, "rgba(63,185,80,0.22)"),
    ]:
        if len(x_ci) > 1:
            pdf_ci = pdf_fn(x_ci)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_ci, x_ci[::-1]]),
                y=np.concatenate([pdf_ci, np.zeros(len(pdf_ci))]),
                fill="toself", fillcolor=clr,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

    mean_a = alpha_a / (alpha_a + beta_a)
    mean_b = alpha_b / (alpha_b + beta_b)
    fig.add_vline(x=mean_a, line=dict(color=CTRL_COLOR, dash="dot", width=1),
                  annotation_text=f"μ_A={mean_a:.3f}",
                  annotation_font=dict(color=CTRL_COLOR, size=11))
    fig.add_vline(x=mean_b, line=dict(color=VAR_COLOR, dash="dot", width=1),
                  annotation_text=f"μ_B={mean_b:.3f}",
                  annotation_font=dict(color=VAR_COLOR, size=11))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Posterior distributions (Beta)",
        xaxis=dict(title="Conversion rate", tickformat=".1%"),
        yaxis=dict(title="Density", showgrid=False),
        height=320,
    )
    return _dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Frequentist Distribution Comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_conversion_distributions(
    control: np.ndarray,
    variant: np.ndarray,
    alpha: float = 0.05,
    n_bins: int = 30,
) -> go.Figure:
    """KDE density curves for continuous metric comparison."""
    fig = go.Figure()

    kde_ctrl = stats.gaussian_kde(control)
    kde_var  = stats.gaussian_kde(variant)
    lo = min(control.min(), variant.min())
    hi = max(control.max(), variant.max())
    x  = np.linspace(lo, hi, 500)

    fig.add_trace(go.Scatter(
        x=x, y=kde_ctrl(x), name="Control A",
        line=dict(color=CTRL_COLOR, width=2),
        fill="tozeroy", fillcolor=CTRL_FILL,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=kde_var(x), name="Variant B",
        line=dict(color=VAR_COLOR, width=2),
        fill="tozeroy", fillcolor=VAR_FILL,
    ))

    fig.add_vline(x=np.mean(control), line=dict(color=CTRL_COLOR, dash="dot", width=1),
                  annotation_text=f"x̄_A={np.mean(control):.3f}",
                  annotation_font=dict(color=CTRL_COLOR, size=11))
    fig.add_vline(x=np.mean(variant), line=dict(color=VAR_COLOR, dash="dot", width=1),
                  annotation_text=f"x̄_B={np.mean(variant):.3f}",
                  annotation_font=dict(color=VAR_COLOR, size=11))

    fig.update_layout(
        **LAYOUT_BASE,
        title="Metric distribution comparison",
        xaxis=dict(title="Metric value"),
        yaxis=dict(title="Density", showgrid=False),
        height=300,
    )
    return _dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Confidence Interval Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_confidence_intervals(
    control_mean: float, control_ci_lo: float, control_ci_hi: float,
    variant_mean: float, variant_ci_lo: float, variant_ci_hi: float,
    alpha: float = 0.05,
    metric_label: str = "Conversion Rate",
) -> go.Figure:
    """Forest-plot style confidence interval comparison."""
    fig = go.Figure()

    groups = ["Control A", "Variant B"]
    means  = [control_mean, variant_mean]
    lo     = [control_ci_lo, variant_ci_lo]
    hi     = [control_ci_hi, variant_ci_hi]
    colors = [CTRL_COLOR, VAR_COLOR]

    for i, (grp, mean, l, h, color) in enumerate(zip(groups, means, lo, hi, colors)):
        fig.add_trace(go.Scatter(
            x=[l, h], y=[grp, grp], mode="lines",
            line=dict(color=color, width=3), showlegend=False,
            hovertemplate=f"{int((1-alpha)*100)}% CI: [{l:.4f}, {h:.4f}]<extra>{grp}</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[mean], y=[grp], mode="markers",
            marker=dict(color=color, size=10, line=dict(color=BG_DARK, width=2)),
            name=f"{grp} (mean={mean:.4f})",
            hovertemplate=f"Mean: {mean:.4f}<extra>{grp}</extra>",
        ))
        for endpoint in [l, h]:
            fig.add_shape(
                type="line",
                x0=endpoint, x1=endpoint,
                y0=i - 0.12, y1=i + 0.12,
                line=dict(color=color, width=2),
            )

    fig.update_layout(
        **LAYOUT_BASE,
        title=f"{int((1-alpha)*100)}% confidence intervals — {metric_label}",
        xaxis=dict(title=metric_label, tickformat=".3f"),
        yaxis=dict(showgrid=False),
        height=200,
    )
    return _dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Power Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_power_curve(
    power_data: list[dict],
    required_n: int,
    target_power: float = 0.80,
) -> go.Figure:
    """Statistical power vs. sample size curve."""
    ns     = [d["n_per_group"] for d in power_data]
    powers = [d["power"] for d in power_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ns, y=[p * 100 for p in powers],
        name="Power", mode="lines",
        line=dict(color=CTRL_COLOR, width=2),
        fill="tozeroy", fillcolor=CTRL_FILL,
    ))
    fig.add_hline(
        y=target_power * 100,
        line=dict(color=WARN, dash="dash", width=1.5),
        annotation_text=f"{target_power:.0%} target",
        annotation_font=dict(color=WARN, size=11),
        annotation_position="right",
    )
    fig.add_vline(
        x=required_n,
        line=dict(color=DANGER, dash="dot", width=1.5),
        annotation_text=f"n={required_n:,}",
        annotation_font=dict(color=DANGER, size=11),
        annotation_position="top",
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title="Power curve vs. sample size",
        xaxis=dict(title="Sample size (per group)", tickformat=","),
        yaxis=dict(title="Power (%)", range=[0, 105]),
        height=300,
    )
    return _dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 5. MDE Tradeoff Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_mde_curve(mde_data: list[dict]) -> go.Figure:
    """MDE vs. required sample size tradeoff."""
    mdes = [d["mde_relative"] * 100 for d in mde_data]
    ns   = [d["n_per_group"] for d in mde_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mdes, y=ns,
        name="Required n", mode="lines+markers",
        line=dict(color=VAR_COLOR, width=2),
        marker=dict(size=4, color=VAR_COLOR),
        fill="tozeroy", fillcolor=VAR_FILL,
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title="Required sample size vs. MDE",
        xaxis=dict(title="Minimum detectable effect (%)", ticksuffix="%"),
        yaxis=dict(title="Sample size (per group)", tickformat=","),
        height=300,
    )
    return _dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Sequential / Time-Series Monitoring
# ─────────────────────────────────────────────────────────────────────────────

def plot_sequential_monitoring(
    history: list[dict],
    early_stop_idx: int | None = None,
) -> go.Figure:
    """Day-by-day CVR and P(B>A) dual-panel monitoring chart."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Conversion rate over time", "P(B beats A)"),
        vertical_spacing=0.15,
    )

    days   = [d["day"] for d in history]
    cvr_a  = [d["cvr_a"] * 100 for d in history]
    cvr_b  = [d["cvr_b"] * 100 for d in history]
    prob_b = [d.get("prob_b_beats_a", 0.5) * 100 for d in history]

    fig.add_trace(go.Scatter(
        x=days, y=cvr_a, name="Control CVR",
        line=dict(color=CTRL_COLOR, width=2), mode="lines",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=days, y=cvr_b, name="Variant CVR",
        line=dict(color=VAR_COLOR, width=2), mode="lines",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=days, y=prob_b, name="P(B>A)",
        line=dict(color=WARN, width=2), mode="lines",
        fill="tozeroy", fillcolor="rgba(210,153,34,0.10)",
    ), row=2, col=1)
    fig.add_hline(
        y=95, line=dict(color=DANGER, dash="dash", width=1),
        row=2, col=1,
        annotation_text="95% threshold",
        annotation_font=dict(color=DANGER, size=11),
    )

    if early_stop_idx is not None and early_stop_idx < len(days):
        stop_day = days[early_stop_idx]
        for row in [1, 2]:
            fig.add_vline(x=stop_day, line=dict(color=DANGER, dash="dot", width=1.5),
                          row=row, col=1)
        fig.add_annotation(
            x=stop_day, y=0.5, xref="x", yref="paper",
            text="Early stop", showarrow=True, arrowhead=2,
            font=dict(color=DANGER, size=11),
        )

    fig.update_layout(
        **LAYOUT_BASE,
        title="Sequential experiment monitoring",
        height=450,
        showlegend=True,
    )
    # Style both subplot axes
    fig.update_xaxes(**_AXIS, title_text="Day")
    fig.update_yaxes(**_AXIS)
    # Override subplot title font color
    for ann in fig.layout.annotations:
        ann.font.color = "#8b949e"
        ann.font.size  = 12
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 7. Expected Loss Chart (Bayesian)
# ─────────────────────────────────────────────────────────────────────────────

def plot_expected_loss(
    loss_choose_b: float,
    loss_choose_a: float,
    loss_threshold: float = 0.005,
) -> go.Figure:
    """Bar chart: expected loss for each deployment decision."""
    categories = ["Deploy B (switch)", "Keep A (hold)"]
    losses     = [loss_choose_b * 100, loss_choose_a * 100]
    bar_colors = [
        VAR_COLOR if loss_choose_b < loss_threshold else DANGER,
        CTRL_COLOR if loss_choose_a < loss_threshold else NEUTRAL,
    ]

    fig = go.Figure(go.Bar(
        x=categories, y=losses,
        marker_color=bar_colors,
        marker_line=dict(color="#2a2f3e", width=1),
        text=[f"{l:.4f}%" for l in losses],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=11),
    ))
    fig.add_hline(
        y=loss_threshold * 100,
        line=dict(color=WARN, dash="dash", width=1.5),
        annotation_text=f"Threshold = {loss_threshold:.1%}",
        annotation_font=dict(color=WARN, size=11),
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title="Expected loss (decision theory)",
        xaxis=dict(title="Decision"),
        yaxis=dict(title="Expected loss (%)", tickformat=".4f"),
        height=280,
        showlegend=False,
    )
    return _dark(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SRM Diagnostic Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_srm_diagnostic(
    assigned_a: int, assigned_b: int, expected_ratio: float = 0.5
) -> go.Figure:
    """Grouped bar: observed vs. expected assignment counts."""
    total    = assigned_a + assigned_b
    exp_a    = total * expected_ratio
    exp_b    = total * (1 - expected_ratio)
    groups   = ["Control A", "Variant B"]
    observed = [assigned_a, assigned_b]
    expected = [exp_a, exp_b]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Observed", x=groups, y=observed,
        marker_color=[CTRL_COLOR, VAR_COLOR],
        marker_line=dict(color="#2a2f3e", width=1),
    ))
    fig.add_trace(go.Bar(
        name="Expected", x=groups, y=expected,
        marker_color=["rgba(88,166,255,0.25)", "rgba(63,185,80,0.25)"],
        marker_line=dict(color=[CTRL_COLOR, VAR_COLOR], width=2),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        barmode="group",
        title="SRM check: observed vs. expected assignment",
        xaxis=dict(title="Group"),
        yaxis=dict(title="Users assigned", tickformat=","),
        height=280,
    )
    return _dark(fig)