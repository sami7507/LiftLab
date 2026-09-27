"""
app.py
───────
LiftLab — A/B Testing & Statistical Inference Dashboard

Run:
    streamlit run app.py

Sections:
    1. Frequentist Tests      — Welch t-test, Chi-square, Mann-Whitney U, Z-test
    2. Bayesian A/B           — Beta-Binomial model, Monte Carlo, Expected Loss
    3. Sample Size & Power    — Cohen's h, Power curve, MDE tradeoff, CUPED
    4. Data Quality (SRM)     — Sample Ratio Mismatch, A/A test calibration
    5. Sequential Monitoring  — Bayesian updating, early stopping, daily tracker

Author : Sami  (sami757007@gmail.com · linkedin.com/in/sami7507)
Version: 2.1.0
Python : 3.10+
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Make src/ importable regardless of cwd ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st

from config.settings import settings
from src.bayesian.beta_binomial import BayesianABTest
from src.tests.frequentist import FrequentistTests
from src.utils.data_generator import SyntheticDataGenerator
from src.utils.sample_size import (
    check_sample_ratio_mismatch,
    compute_sample_size,
    cuped_variance_reduction,
    mde_curve,
    power_curve,
    sequential_alpha_spending,
)
from src.visualization.plots import (
    plot_confidence_intervals,
    plot_conversion_distributions,
    plot_expected_loss,
    plot_mde_curve,
    plot_posterior_distributions,
    plot_power_curve,
    plot_sequential_monitoring,
    plot_srm_diagnostic,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

APP_NAME = settings.app_title
GITHUB_URL = "https://github.com/sami7507/AB-Besting-Statistical-Inference-Dashboard"

st.set_page_config(
    page_title=f"{APP_NAME} — Statistical Inference Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": GITHUB_URL,
        "Report a bug": f"{GITHUB_URL}/issues",
        "About": (
            f"{APP_NAME} · Industry-grade A/B Testing & Statistical Inference "
            "Dashboard v2.1 — built by Sami (sami757007@gmail.com)"
        ),
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ══════════════════════════════════════════════════════════════════
       DARK THEME — LiftLab
       Forces a consistent, polished dark aesthetic regardless of the
       Streamlit theme the user has selected.
       ══════════════════════════════════════════════════════════════════ */

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre, .stCodeBlock, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    /* ── App background & main container ── */
    .stApp {
        background: radial-gradient(circle at 20% 0%, #131924 0%, #0e1117 45%) !important;
    }
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2rem !important;
        max-width: 1220px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #10141d !important;
        border-right: 1px solid #232838 !important;
    }
    [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2 { color: #e6edf3 !important; letter-spacing: -0.01em; }

    /* ── All text defaults ── */
    h1, h2, h3, h4 { color: #e6edf3 !important; letter-spacing: -0.02em; }
    p, span, label { color: #c9d1d9; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #171d2b 0%, #0e1117 100%);
        border: 1px solid #262c3d;
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: -60%; right: -8%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(56,139,253,0.16) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-banner::after {
        content: "";
        position: absolute;
        bottom: -70%; left: -6%;
        width: 260px; height: 260px;
        background: radial-gradient(circle, rgba(63,185,80,0.10) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 1.95rem; font-weight: 800; color: #e6edf3; margin: 0;
        display: flex; align-items: center; gap: 10px; position: relative;
    }
    .hero-sub {
        color: #8b949e; font-size: 0.92rem; margin-top: 8px; max-width: 720px;
        line-height: 1.55; position: relative;
    }
    .hero-badges { margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; position: relative; }
    .hero-badge {
        background: #1a2233; color: #79c0ff; border: 1px solid #2d5a8e;
        padding: 4px 13px; border-radius: 20px; font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.01em;
    }

    /* ── Metric cards ── */
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #e6edf3 !important;
    }
    [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.78rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.76rem !important; }
    [data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #262c3d;
        border-radius: 12px;
        padding: 14px 16px !important;
        transition: border-color 0.15s ease, transform 0.15s ease;
    }
    [data-testid="metric-container"]:hover {
        border-color: #388bfd55;
        transform: translateY(-1px);
    }

    /* ── Sidebar section labels ── */
    .sidebar-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #4c5566 !important;
        margin: 1.2rem 0 0.3rem;
    }

    /* ── Result badges ── */
    .badge-sig {
        display: inline-block;
        background: #0d2118; color: #3fb950;
        padding: 5px 16px; border-radius: 20px;
        font-size: 0.80rem; font-weight: 600;
        margin-bottom: 0.75rem;
        border: 1px solid #238636;
        letter-spacing: 0.01em;
    }
    .badge-not-sig {
        display: inline-block;
        background: #2d1d00; color: #d29922;
        padding: 5px 16px; border-radius: 20px;
        font-size: 0.80rem; font-weight: 600;
        margin-bottom: 0.75rem;
        border: 1px solid #9e6a03;
    }
    .badge-high   { background:#0d2118; color:#3fb950; border:1px solid #238636; }
    .badge-medium { background:#2d1d00; color:#d29922; border:1px solid #9e6a03; }
    .badge-low    { background:#2d0c0c; color:#f85149; border:1px solid #da3633; }

    /* ── Section header tag ── */
    .section-tag {
        display: inline-block;
        background: #16233a; color: #79c0ff;
        padding: 3px 12px; border-radius: 4px;
        font-size: 0.70rem; font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase;
        margin-bottom: 0.5rem;
        border: 1px solid #2d5a8e;
    }

    /* ── Interpretation / info box ── */
    .interp-box {
        background: #161b27;
        border-left: 3px solid #388bfd;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.84rem;
        line-height: 1.7;
        margin: 0.6rem 0;
        color: #c9d1d9;
    }

    /* ── Streamlit info / success / warning overrides ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border-width: 1px !important;
    }

    /* ── Expanders ── */
    [data-testid="stExpander"] {
        background: #141a26 !important;
        border: 1px solid #262c3d !important;
        border-radius: 10px !important;
    }
    details summary {
        font-weight: 500;
        font-size: 0.90rem;
        color: #c9d1d9 !important;
    }
    details summary:hover { color: #e6edf3 !important; }

    /* ── DataFrames / tables ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #262c3d !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] table {
        font-size: 0.80rem !important;
        background: #161b27 !important;
    }
    [data-testid="stDataFrame"] th {
        background: #1a2030 !important;
        color: #8b949e !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.68rem !important;
        letter-spacing: 0.06em;
    }
    [data-testid="stDataFrame"] td { color: #c9d1d9 !important; }
    [data-testid="stDataFrame"] tr:hover td {
        background: #1a2030 !important;
    }

    /* ── Plotly chart wrapper ── */
    [data-testid="stPlotlyChart"] {
        background: #141a26;
        border: 1px solid #262c3d;
        border-radius: 12px;
        overflow: hidden;
        padding: 4px;
    }

    /* ── Buttons ── */
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #388bfd 0%, #2d6ee0 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: filter 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.1); }

    /* ── Sliders ── */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: #388bfd !important;
        border-color: #388bfd !important;
    }

    /* ── Select boxes & inputs ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input {
        background: #141a26 !important;
        border-color: #262c3d !important;
        color: #e6edf3 !important;
    }

    /* ── Radio buttons ── */
    [data-testid="stRadio"] label { color: #c9d1d9 !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tab"] { color: #8b949e !important; }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #388bfd !important;
    }

    /* ── Dividers ── */
    hr {
        margin: 1.5rem 0 !important;
        border-color: #232838 !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] { color: #58a6ff !important; }

    /* ── Caption text ── */
    [data-testid="stCaptionContainer"] p { color: #4c5566 !important; font-size: 0.76rem !important; }

    /* ── Title / header spacing ── */
    .stTitle, h1 { margin-bottom: 0.2rem !important; }

    /* ── Footer ── */
    .app-footer {
        margin-top: 1.5rem;
        padding: 18px 22px;
        border: 1px solid #262c3d;
        border-radius: 12px;
        background: #10141d;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .app-footer a { color: #58a6ff !important; text-decoration: none; }
    .app-footer a:hover { text-decoration: underline; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0e1117; }
    ::-webkit-scrollbar-thumb { background: #262c3d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #388bfd; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"## ⚡ {APP_NAME}")
    st.caption("Statistical Inference Dashboard · v2.1")
    st.divider()

    st.markdown('<p class="sidebar-label">Navigation</p>', unsafe_allow_html=True)
    section = st.radio(
        "nav",
        options=[
            "🔬 Frequentist Tests",
            "🧠 Bayesian A/B",
            "📐 Sample Size & Power",
            "🔍 Data Quality (SRM)",
            "📈 Sequential Monitoring",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<p class="sidebar-label">Global Settings</p>', unsafe_allow_html=True)

    alpha = st.slider(
        "Significance level α",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
        help="Type-I error rate (false positive rate). Industry standard: 0.05.",
    )
    power_target = st.slider(
        "Target power (1 − β)",
        min_value=0.60, max_value=0.99, value=0.80, step=0.05,
        help="Probability of detecting a true effect. Typical minimum: 0.80.",
    )
    ci_level = 1.0 - alpha

    st.divider()
    st.markdown('<p class="sidebar-label">Data Source</p>', unsafe_allow_html=True)

    data_mode = st.selectbox(
        "Input mode",
        options=["Synthetic (demo)", "Manual input", "Upload CSV"],
        help="Synthetic uses generated data. Manual lets you type raw numbers. CSV uploads your data.",
    )
    rng_seed = st.number_input("Random seed", value=42, step=1,
                               help="Seed for reproducible synthetic data generation.")

    st.divider()
    st.caption(
        "Stack: Python 3.10 · scipy · numpy · pymc · streamlit · plotly\n\n"
        "Tests: Welch t · χ² · Mann-Whitney U · Beta-Binomial · CUPED · SRM"
    )
    st.divider()
    st.caption(f"Built by **Sami** · [LinkedIn](https://www.linkedin.com/in/samikhan07)")


# ─────────────────────────────────────────────────────────────────────────────
# Hero banner (main area)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">⚡ {APP_NAME}</div>
    <div class="hero-sub">
        A production-style Frequentist &amp; Bayesian A/B testing engine — statistical
        rigor, guardrail checks, and sequential monitoring in one dashboard, built for
        teams who need to ship experiment decisions with confidence.
    </div>
    <div class="hero-badges">
        <span class="hero-badge">Frequentist</span>
        <span class="hero-badge">Bayesian</span>
        <span class="hero-badge">Power Analysis</span>
        <span class="hero-badge">SRM Detection</span>
        <span class="hero-badge">Sequential Monitoring</span>
        <span class="hero-badge">CUPED</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_synthetic(seed: int, metric_type: str, n: int,
                    base_rate: float, true_lift: float):
    """Cache synthetic dataset so sliders don't regenerate on every interaction."""
    gen = SyntheticDataGenerator(seed=seed)
    if metric_type == "Conversion (binary)":
        return gen.conversion_experiment(n=n, base_rate=base_rate, true_lift=true_lift)
    elif metric_type == "Revenue (log-normal)":
        return gen.revenue_experiment(n=n, base_mean=max(base_rate * 200, 5.0),
                                      true_lift=true_lift)
    elif metric_type == "Session Time (Gamma)":
        return gen.session_time_experiment(n=n, base_minutes=max(base_rate * 50, 1.0),
                                           true_lift=true_lift)
    else:  # NPS
        return gen.nps_experiment(n=n, base_nps=base_rate * 100 - 50,
                                  true_lift=true_lift * 20)


def _sig_badge(significant: bool, p_value: float) -> str:
    if significant:
        return f'<span class="badge-sig">✓ Significant — p = {p_value:.5f}</span>'
    return f'<span class="badge-not-sig">✗ Not significant — p = {p_value:.5f}</span>'


def _interp(text: str) -> None:
    st.markdown(f'<div class="interp-box">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████  SECTION 1: FREQUENTIST  ██████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

if section == "🔬 Frequentist Tests":

    st.markdown('<span class="section-tag">Frequentist</span>', unsafe_allow_html=True)
    st.title("Frequentist A/B Testing")
    st.caption(
        "Implements Welch's t-test · Student's t-test · Chi-square · "
        "Mann-Whitney U · Z-test for proportions · BH/Bonferroni correction"
    )

    # ── 1a. Experiment Parameters ─────────────────────────────────────────────
    with st.expander("⚙ Experiment Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_type = st.selectbox(
                "Metric type",
                ["Conversion (binary)", "Revenue (log-normal)",
                 "Session Time (Gamma)", "NPS (ordinal)"],
                help=(
                    "Binary → Chi-square + Z-test\n"
                    "Continuous → Welch / Student t-test\n"
                    "Ordinal / skewed → Mann-Whitney U"
                ),
            )
        with c2:
            n_per_group = st.slider("n per group", 50, 5000, 500, 50,
                                    help="Number of users in each variant arm.")
        with c3:
            base_rate = st.slider("Baseline rate / scale", 0.01, 0.50, 0.12, 0.01,
                                  help="Control group conversion rate (binary) or scale factor.")
        with c4:
            true_lift_pct = st.slider("True lift (relative %)", -30, 50, 15, 1,
                                      help="Simulated ground-truth effect of the variant.")
            true_lift = true_lift_pct / 100.0

    # ── 1b. Load data & run tests ─────────────────────────────────────────────
    dataset = _load_synthetic(rng_seed, metric_type, n_per_group, base_rate, true_lift)
    control = dataset.control
    variant = dataset.variant
    is_binary = (metric_type == "Conversion (binary)")

    tester = FrequentistTests(alpha=alpha)

    if is_binary:
        n_ctrl = len(control);  c_ctrl = int(control.sum())
        n_var  = len(variant);  c_var  = int(variant.sum())
        result_chi  = tester.chi_square_test(n_ctrl, c_ctrl, n_var, c_var)
        result_z    = tester.z_test_proportions(n_ctrl, c_ctrl, n_var, c_var)
        result_mw   = tester.mann_whitney_test(control, variant)
        results     = [result_chi, result_z, result_mw]
        primary     = result_chi
    else:
        result_welch   = tester.welch_ttest(control, variant)
        result_student = tester.student_ttest(control, variant)
        result_mw      = tester.mann_whitney_test(control, variant)
        results        = [result_welch, result_student, result_mw]
        primary        = result_welch

    # ── 1c. Summary metrics ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### Results")
    st.markdown(_sig_badge(primary.significant, primary.p_value), unsafe_allow_html=True)

    ctrl_label = "Control CVR"   if is_binary else "Control mean"
    var_label  = "Variant CVR"   if is_binary else "Variant mean"
    fmt        = "{:.2%}"        if is_binary else "{:.4f}"
    delta_fmt  = "{:+.2%}"      if is_binary else "{:+.4f}"

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(ctrl_label,  fmt.format(primary.control_mean))
    m2.metric(var_label,   fmt.format(primary.variant_mean),
              delta=delta_fmt.format(primary.variant_mean - primary.control_mean))
    m3.metric("p-value",   f"{primary.p_value:.5f}",
              delta="< α ✓" if primary.significant else "> α ✗",
              delta_color="normal" if primary.significant else "inverse")
    m4.metric(primary.effect_size_label, f"{primary.effect_size:.4f}",
              delta=primary.effect_magnitude)
    m5.metric("Recommendation",
              "Ship" if primary.significant and primary.variant_mean > primary.control_mean
              else "Hold")

    # ── 1d. Statistical warnings ──────────────────────────────────────────────
    all_warnings = list({w for r in results for w in r.warnings})
    if all_warnings:
        for w in all_warnings:
            st.warning(w)

    # ── 1e. Charts ────────────────────────────────────────────────────────────
    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Distribution comparison**")
        fig_dist = plot_conversion_distributions(control, variant, alpha=alpha)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_r:
        st.markdown("**Confidence interval comparison**")
        if primary.ci_lower is not None and primary.ci_upper is not None:
            se = abs(primary.ci_upper - primary.ci_lower) / 2
            fig_ci = plot_confidence_intervals(
                control_mean=primary.control_mean,
                control_ci_lo=primary.control_mean - se,
                control_ci_hi=primary.control_mean + se,
                variant_mean=primary.variant_mean,
                variant_ci_lo=primary.variant_mean - se,
                variant_ci_hi=primary.variant_mean + se,
                alpha=alpha,
                metric_label=metric_type,
            )
            st.plotly_chart(fig_ci, use_container_width=True)
        else:
            st.info("Confidence interval not available for this non-parametric test.")

    # ── 1f. All test results table ────────────────────────────────────────────
    st.divider()
    st.markdown("### All statistical test results")

    rows = []
    for r in results:
        rows.append({
            "Test": r.test_name,
            "Statistic": f"{r.statistic:.4f}",
            "p-value": f"{r.p_value:.5f}",
            "Significant": "Yes" if r.significant else "No",
            "Effect size": f"{r.effect_size:.4f}",
            "Metric": r.effect_size_label,
            "Magnitude": r.effect_magnitude.title(),
            "Recommendation": r.recommendation,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 1g. Interpretation ────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Interpretation")
    for r in results:
        with st.expander(f"Interpretation — {r.test_name}"):
            _interp(r.interpretation)
            if r.warnings:
                for w in r.warnings:
                    st.warning(w)

    # ── 1h. Multiple testing correction ──────────────────────────────────────
    st.divider()
    with st.expander("Multiple testing correction (multi-metric experiments)"):
        st.markdown(
            "When analysing multiple metrics simultaneously, apply a correction to "
            "control the false discovery rate and prevent spurious findings."
        )

        raw_pvals = [r.p_value for r in results]
        bh_result   = FrequentistTests.benjamini_hochberg(raw_pvals, alpha=alpha)
        bonf_result = FrequentistTests.bonferroni_correction(raw_pvals, alpha=alpha)

        tab_bh, tab_bonf = st.tabs(["Benjamini-Hochberg (FDR)", "Bonferroni (FWER)"])

        with tab_bh:
            st.caption(
                "Controls the **expected proportion** of false discoveries. "
                "Less conservative than Bonferroni — preferred for exploratory analysis."
            )
            st.dataframe(pd.DataFrame({
                "Test": [r.test_name for r in results],
                "Raw p-value": [f"{p:.5f}" for p in raw_pvals],
                "BH-adjusted p": [f"{p:.5f}" for p in bh_result["adjusted_p_values"]],
                "Significant at α": ["Yes" if s else "No" for s in bh_result["significant"]],
            }), use_container_width=True, hide_index=True)

        with tab_bonf:
            st.caption(
                "Controls the **family-wise error rate** (P of any false positive). "
                "More conservative — preferred when even one false positive is costly."
            )
            st.dataframe(pd.DataFrame({
                "Test": [r.test_name for r in results],
                "Raw p-value": [f"{p:.5f}" for p in raw_pvals],
                "Bonferroni-adjusted p": [f"{p:.5f}" for p in bonf_result["adjusted_p_values"]],
                "Significant at α": ["Yes" if s else "No" for s in bonf_result["significant"]],
            }), use_container_width=True, hide_index=True)

    # ── 1i. Raw data sample ───────────────────────────────────────────────────
    with st.expander("Raw data sample (first 40 rows)"):
        gen_preview = SyntheticDataGenerator(seed=rng_seed)
        df_preview = gen_preview.to_dataframe(dataset).head(40)
        st.dataframe(df_preview, use_container_width=True, hide_index=True)
        st.caption(
            f"Control n={len(control):,} | Variant n={len(variant):,} | "
            f"True lift injected: {true_lift:+.1%}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████  SECTION 2: BAYESIAN  █████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

elif section == "🧠 Bayesian A/B":

    st.markdown('<span class="section-tag">Bayesian</span>', unsafe_allow_html=True)
    st.title("Bayesian A/B Testing")
    st.caption(
        "Beta-Binomial conjugate model · Monte Carlo sampling · "
        "Expected loss decision framework · Credible intervals"
    )

    # ── 2a. Data input ─────────────────────────────────────────────────────────
    with st.expander("⚙ Experiment data & prior", expanded=True):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Control (A)**")
            n_a = st.number_input("Visitors (A)", min_value=10, max_value=500_000,
                                  value=1200, step=50)
            c_a = st.number_input("Conversions (A)", min_value=1,
                                  max_value=int(n_a), value=144, step=5)
            st.caption(f"Observed CVR: {c_a/n_a:.2%}")

        with c2:
            st.markdown("**Variant (B)**")
            n_b = st.number_input("Visitors (B)", min_value=10, max_value=500_000,
                                  value=1200, step=50)
            c_b = st.number_input("Conversions (B)", min_value=1,
                                  max_value=int(n_b), value=174, step=5)
            st.caption(f"Observed CVR: {c_b/n_b:.2%}")

        st.divider()
        c3, c4 = st.columns(2)

        with c3:
            prior_choice = st.selectbox(
                "Prior distribution",
                ["Uniform  Beta(1, 1)", "Jeffreys  Beta(0.5, 0.5)",
                 "Weakly informative  Beta(2, 18)"],
                help=(
                    "Uniform: equal weight on all rates — standard default.\n"
                    "Jeffreys: non-informative, mathematically optimal for proportions.\n"
                    "Weakly informative: encodes a prior belief of ~10% baseline CVR."
                ),
            )

        with c4:
            loss_threshold = st.slider(
                "Expected loss threshold",
                min_value=0.001, max_value=0.050, value=0.005, step=0.001,
                format="%.3f",
                help=(
                    "Ship B when expected loss < this threshold.\n"
                    "0.005 = tolerate at most 0.5% expected revenue loss."
                ),
            )

        mc_samples = st.select_slider(
            "Monte Carlo samples",
            options=[10_000, 25_000, 50_000, 100_000],
            value=50_000,
            help="More samples → more precise P(B>A) estimate. 50k is a good balance.",
        )

    # ── 2b. Run Bayesian inference ─────────────────────────────────────────────
    prior_map = {
        "Uniform  Beta(1, 1)":               (1.0,  1.0),
        "Jeffreys  Beta(0.5, 0.5)":          (0.5,  0.5),
        "Weakly informative  Beta(2, 18)":   (2.0, 18.0),
    }
    pa, pb = prior_map[prior_choice]

    bayes = BayesianABTest(
        prior_alpha=pa,
        prior_beta=pb,
        monte_carlo_samples=mc_samples,
        seed=int(rng_seed),
    )

    with st.spinner("Running Monte Carlo inference…"):
        try:
            result = bayes.run(
                n_a=int(n_a), conversions_a=int(c_a),
                n_b=int(n_b), conversions_b=int(c_b),
                ci_level=ci_level,
                loss_threshold=loss_threshold,
            )
        except ValueError as e:
            st.error(f"Input error: {e}")
            st.stop()

    # ── 2c. Key metrics ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Inference results")

    conf_cls = {
        "High":   "badge-sig badge-high",
        "Medium": "badge-not-sig badge-medium",
        "Low":    "badge-not-sig badge-low",
    }.get(result.confidence_label, "badge-not-sig")
    st.markdown(
        f'<span class="badge-sig">P(B beats A) = {result.prob_b_beats_a:.1%}</span>&nbsp;'
        f'<span class="{conf_cls}">Confidence: {result.confidence_label}</span>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("P(B > A)",          f"{result.prob_b_beats_a:.1%}")
    m2.metric("Expected loss (B)", f"{result.expected_loss_choose_b:.4%}",
              delta="Below threshold ✓" if result.expected_loss_choose_b < loss_threshold
                    else "Above threshold ✗",
              delta_color="normal" if result.expected_loss_choose_b < loss_threshold
                          else "inverse")
    m3.metric("Relative uplift",   f"{result.relative_uplift:+.2%}")
    m4.metric("Absolute uplift",   f"{result.absolute_uplift:+.4f}")
    m5.metric("Expected loss (A)", f"{result.expected_loss_choose_a:.4%}")

    st.info(f"**Recommendation:** {result.recommendation}")

    for w in result.warnings:
        st.warning(w)

    # ── 2d. Charts ─────────────────────────────────────────────────────────────
    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Posterior distributions**")
        fig_post = plot_posterior_distributions(
            result.alpha_a, result.beta_a,
            result.alpha_b, result.beta_b,
            ci_level=ci_level,
        )
        st.plotly_chart(fig_post, use_container_width=True)

    with col_r:
        st.markdown("**Expected loss (decision theory)**")
        fig_loss = plot_expected_loss(
            result.expected_loss_choose_b,
            result.expected_loss_choose_a,
            loss_threshold=loss_threshold,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # ── 2e. Full posterior summary ────────────────────────────────────────────
    st.divider()
    with st.expander("Full posterior parameter summary"):
        col_a, col_b = st.columns(2)

        def _post_table(mean, std, lo, hi, alpha_p, beta_p, ci_lvl):
            return pd.DataFrame({
                "Parameter": [
                    "Posterior mean", "Posterior std",
                    f"{ci_lvl:.0%} credible interval — lower",
                    f"{ci_lvl:.0%} credible interval — upper",
                    "Posterior α", "Posterior β",
                ],
                "Value": [
                    f"{mean:.4%}", f"{std:.4%}",
                    f"{lo:.4%}",   f"{hi:.4%}",
                    f"{alpha_p:.2f}", f"{beta_p:.2f}",
                ],
            })

        with col_a:
            st.markdown(f"**Control A** — prior Beta({pa}, {pb})")
            st.dataframe(
                _post_table(result.posterior_mean_a, result.posterior_std_a,
                            result.ci_lower_a, result.ci_upper_a,
                            result.alpha_a, result.beta_a, ci_level),
                use_container_width=True, hide_index=True,
            )

        with col_b:
            st.markdown(f"**Variant B** — prior Beta({pa}, {pb})")
            st.dataframe(
                _post_table(result.posterior_mean_b, result.posterior_std_b,
                            result.ci_lower_b, result.ci_upper_b,
                            result.alpha_b, result.beta_b, ci_level),
                use_container_width=True, hide_index=True,
            )

    # ── 2f. Prior sensitivity analysis ────────────────────────────────────────
    with st.expander("Prior sensitivity analysis"):
        st.markdown(
            "How much does the choice of prior matter? "
            "For large n the posterior converges — prior influence should be minimal."
        )
        sensitivity_rows = []
        for pname, (pa2, pb2) in prior_map.items():
            b2 = BayesianABTest(pa2, pb2, 20_000, seed=0)
            r2 = b2.run(int(n_a), int(c_a), int(n_b), int(c_b))
            sensitivity_rows.append({
                "Prior": pname.split("  ")[0],
                "P(B > A)": f"{r2.prob_b_beats_a:.1%}",
                "Expected loss": f"{r2.expected_loss_choose_b:.4%}",
                "Posterior mean A": f"{r2.posterior_mean_a:.4%}",
                "Posterior mean B": f"{r2.posterior_mean_b:.4%}",
                "Recommendation": r2.recommendation.split(".")[0],
            })
        st.dataframe(pd.DataFrame(sensitivity_rows), use_container_width=True, hide_index=True)
        st.caption(
            "If P(B>A) is stable across all three priors, the result is robust. "
            "Large swings indicate the data is insufficient to overwhelm the prior."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ████████████████████  SECTION 3: SAMPLE SIZE & POWER  ████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

elif section == "📐 Sample Size & Power":

    st.markdown('<span class="section-tag">Power Analysis</span>', unsafe_allow_html=True)
    st.title("Sample Size & Power Analysis")
    st.caption(
        "Cohen's h effect size · Two-proportion z-test · Power curve · "
        "MDE tradeoff · CUPED variance reduction · O'Brien-Fleming α-spending"
    )

    # ── 3a. Parameters ─────────────────────────────────────────────────────────
    with st.expander("⚙ Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            ss_base = st.slider("Baseline CVR", 0.01, 0.50, 0.10, 0.01,
                                help="Current conversion rate of the control group.")
        with c2:
            ss_mde = st.slider("MDE (relative)", 0.01, 0.50, 0.10, 0.01,
                               help=(
                                   "Minimum Detectable Effect as a fraction of baseline. "
                                   "0.10 = a 10% relative lift (e.g. 10% → 11%)."
                               ))
        with c3:
            ss_power = st.slider("Target power (1 − β)", 0.60, 0.99,
                                 float(power_target), 0.05)
        with c4:
            daily_traffic = st.number_input(
                "Daily traffic (both groups)", value=500, step=50, min_value=10,
                help="Total daily users split across control and variant.",
            )

    # ── 3b. Compute & display ──────────────────────────────────────────────────
    try:
        ss_result = compute_sample_size(
            baseline_rate=ss_base,
            mde_relative=ss_mde,
            alpha=alpha,
            power=ss_power,
            daily_traffic=int(daily_traffic),
        )
    except ValueError as e:
        st.error(f"Calculation error: {e}")
        st.stop()

    st.divider()
    st.markdown("### Required sample size")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("n per group",   f"{ss_result.n_per_group:,}")
    m2.metric("Total users",   f"{ss_result.n_total:,}")
    m3.metric("Duration",
              f"{ss_result.days_required:.1f} days" if ss_result.days_required else "—")
    m4.metric("Cohen's h",     f"{ss_result.cohen_h:.4f}")
    m5.metric("Absolute MDE",  f"{ss_result.mde_absolute:+.2%}")

    st.caption(
        f"Baseline: **{ss_base:.1%}** → Variant: **{ss_result.variant_rate:.1%}** "
        f"(+{ss_mde:.0%} relative = {ss_result.mde_absolute:+.2%} absolute) | "
        f"α = {alpha} | Power = {ss_power:.0%} | Two-tailed"
    )

    # ── 3c. Power & MDE charts ─────────────────────────────────────────────────
    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Power curve vs. sample size**")
        pwr_data = power_curve(ss_base, ss_mde, alpha=alpha)
        fig_pwr = plot_power_curve(pwr_data, ss_result.n_per_group, ss_power)
        st.plotly_chart(fig_pwr, use_container_width=True)

    with col_r:
        st.markdown("**MDE vs. required sample size**")
        mde_data = mde_curve(ss_base, alpha=alpha, power=ss_power)
        fig_mde = plot_mde_curve(mde_data)
        st.plotly_chart(fig_mde, use_container_width=True)

    # ── 3d. Duration estimation table ─────────────────────────────────────────
    with st.expander("Duration estimation for different traffic levels"):
        traffic_scenarios = [100, 250, 500, 1000, 2000, 5000]
        duration_rows = []
        for t in traffic_scenarios:
            days = ss_result.n_total / t
            duration_rows.append({
                "Daily traffic (both groups)": f"{t:,}",
                "Days required": f"{days:.1f}",
                "Weeks": f"{days/7:.1f}",
                "Feasible?": (
                    "Yes" if days <= 90
                    else "Long" if days <= 180
                    else "Impractical"
                ),
            })
        st.dataframe(pd.DataFrame(duration_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Rule of thumb: experiments > 90 days face novelty effect and seasonal confounds."
        )

    # ── 3e. Sequential α-spending ─────────────────────────────────────────────
    st.divider()
    with st.expander("Sequential testing boundaries — anti-peeking protection"):
        st.markdown(
            "If you must look at results before the experiment ends, use an α-spending function "
            "to preserve the overall Type-I error rate. This prevents the peeking problem "
            "that inflates false positive rates in standard frequentist tests."
        )
        c1, c2 = st.columns(2)
        with c1:
            n_looks = st.slider("Number of interim analyses", 2, 10, 5,
                                help="How many times will you check the results mid-experiment?")
        with c2:
            spend_method = st.radio(
                "Spending function",
                ["obrien_fleming", "pocock"],
                horizontal=True,
                help=(
                    "O'Brien-Fleming: very conservative early, nearly full α at the end. "
                    "Pocock: constant boundary — easier to cross early."
                ),
            )

        boundaries = sequential_alpha_spending(n_looks, alpha=alpha, method=spend_method)
        looks_df = pd.DataFrame({
            "Analysis": [
                f"Interim {i+1}" if i < n_looks - 1 else f"Final (#{i+1})"
                for i in range(n_looks)
            ],
            "Fraction of data": [f"{(i+1)/n_looks:.0%}" for i in range(n_looks)],
            "Alpha boundary": [f"{b:.6f}" for b in boundaries],
            "Reject H₀ if p <": [f"{b:.6f}" for b in boundaries],
        })
        st.dataframe(looks_df, use_container_width=True, hide_index=True)
        st.caption(
            "O'Brien-Fleming is conservative early (hard to stop) and nearly full α at the "
            "final look — the industry standard. Pocock uses a constant boundary but requires "
            "a lower final-look threshold, sacrificing some power."
        )

    # ── 3f. CUPED variance reduction ──────────────────────────────────────────
    st.divider()
    with st.expander("CUPED — Variance Reduction via Pre-Experiment Data"):
        st.markdown(
            "CUPED regresses out a pre-experiment covariate (e.g. last week's revenue) "
            "from the post-experiment metric. This reduces metric variance, which "
            "either shortens experiment duration or increases power at the same n."
        )

        c1, c2 = st.columns(2)
        with c1:
            cuped_n = st.slider("Sample size for CUPED demo", 200, 2000, 500, 50)
        with c2:
            cuped_rho = st.slider(
                "Pre/post correlation ρ", 0.10, 0.95, 0.70, 0.05,
                help="Higher correlation → more variance reduction.",
            )

        gen_cuped = SyntheticDataGenerator(seed=int(rng_seed))
        ds_cuped  = gen_cuped.revenue_experiment(n=cuped_n)

        rng_c = np.random.default_rng(0)
        noise  = rng_c.normal(0, 1, cuped_n)
        pre_ctrl = (cuped_rho * ds_cuped.control
                    + np.sqrt(1 - cuped_rho**2) * noise * np.std(ds_cuped.control))

        adj_ctrl, var_reduction = cuped_variance_reduction(ds_cuped.control, pre_ctrl)
        effective_n_mult = 1.0 / max(1.0 - var_reduction, 0.01)

        m1, m2, m3 = st.columns(3)
        m1.metric("Variance reduction",   f"{var_reduction:.1%}")
        m2.metric("Effective n multiplier", f"{effective_n_mult:.2f}×",
                  help="CUPED gives you this many times more effective samples.")
        m3.metric("Equiv. extra users",
                  f"{int(cuped_n * (effective_n_mult - 1)):,}",
                  help="How many extra users you'd need without CUPED for the same power.")

        st.caption(
            f"Original variance: {np.var(ds_cuped.control, ddof=1):.4f} → "
            f"CUPED-adjusted: {np.var(adj_ctrl, ddof=1):.4f} | "
            f"ρ(post, pre) = {cuped_rho:.2f}"
        )
        _interp(
            "CUPED works best when the pre-experiment metric is strongly correlated with the "
            "post-experiment metric (ρ > 0.5). In practice, using last week's data as the "
            "covariate for this week's experiment typically gives 20–50% variance reduction, "
            "equivalent to running 25–100% more users."
        )


# ─────────────────────────────────────────────────────────────────────────────
# ████████████████████  SECTION 4: DATA QUALITY (SRM)  █████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

elif section == "🔍 Data Quality (SRM)":

    st.markdown('<span class="section-tag">Data Quality</span>', unsafe_allow_html=True)
    st.title("Data Quality & Guardrail Checks")
    st.caption(
        "Sample Ratio Mismatch detection · A/A test calibration · "
        "Pre-experiment sanity checks"
    )

    # ── 4a. SRM check ─────────────────────────────────────────────────────────
    st.markdown("### Sample Ratio Mismatch (SRM) detection")
    _interp(
        "SRM occurs when the actual assignment ratio deviates significantly from the expected "
        "ratio. It is a critical quality signal — any metric result from an experiment with "
        "SRM is untrustworthy until the root cause is identified and fixed. Common causes: "
        "buggy hash function, bot traffic filtered unevenly, sticky sessions in load balancer, "
        "logging pipeline dropping events at different rates per variant."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        srm_a = st.number_input("Users assigned to Control (A)",
                                min_value=1, value=5021, step=100)
    with c2:
        srm_b = st.number_input("Users assigned to Variant (B)",
                                min_value=1, value=4620, step=100)
    with c3:
        exp_ratio = st.slider(
            "Expected fraction in Control",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="For a standard 50/50 split, set to 0.50.",
        )

    try:
        srm_result = check_sample_ratio_mismatch(
            int(srm_a), int(srm_b),
            expected_ratio=exp_ratio,
            alpha=0.01,   # SRM uses stricter threshold
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    if srm_result.srm_detected:
        st.markdown(f"""
        <div style="background:#2d0c0c;border:1px solid #da3633;border-radius:8px;
                    padding:12px 16px;margin:8px 0;color:#f85149;font-size:14px;font-weight:500;">
            SRM DETECTED &mdash; p = {srm_result.p_value:.5f} &lt; 0.01 &nbsp;&nbsp;
            <span style="font-weight:400;color:#ffa198;">
            Stop the experiment immediately. Results are invalid until root cause is fixed.
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:#0d2118;border:1px solid #238636;border-radius:8px;
                    padding:12px 16px;margin:8px 0;color:#3fb950;font-size:14px;font-weight:500;">
            SRM not detected &mdash; p = {srm_result.p_value:.4f} &ge; 0.01 &nbsp;&nbsp;
            <span style="font-weight:400;color:#56d364;">
            Assignment ratio is consistent with the expected {exp_ratio:.0%} split.
            </span>
        </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("χ² statistic",      f"{srm_result.chi2_stat:.4f}")
    m2.metric("p-value",            f"{srm_result.p_value:.5f}")
    m3.metric("Observed ratio (A)", f"{srm_result.observed_ratio:.4f}")
    m4.metric("Expected ratio (A)", f"{exp_ratio:.4f}",
              delta=f"{srm_result.observed_ratio - exp_ratio:+.4f}")

    fig_srm = plot_srm_diagnostic(int(srm_a), int(srm_b), exp_ratio)
    st.plotly_chart(fig_srm, use_container_width=True)

    with st.expander("SRM root cause guide"):
        st.markdown("""
| Root cause | Signal | Fix |
|---|---|---|
| Hash function bias | Consistent skew toward one variant | Audit hash seed and modulo operation |
| Bot filtering post-assignment | Bots disproportionately in one group | Apply bot filter before assignment |
| Sticky sessions | Users re-assigned across sessions | Use user-level not session-level hashing |
| Log pipeline drop | Logging server overloaded | Fix logging infra; check drop rates per variant |
| Feature flag lag | Variant flag slow to roll out | Ensure simultaneous flag activation |
        """)

    # ── 4b. A/A test calibration ───────────────────────────────────────────────
    st.divider()
    st.markdown("### A/A test — pipeline calibration")
    _interp(
        "An A/A test exposes two identical groups to the same experience. Your platform "
        "should produce significant results at exactly the α rate (≈5% of A/A tests "
        "should be falsely flagged as significant). If you see FPR > 8% or < 2%, "
        "your assignment or analysis pipeline has a systematic bug."
    )

    with st.expander("Simulate A/A false positive rate"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_aa = st.slider("Sample size per group", 100, 2000, 500, 50)
        with c2:
            n_sims = st.slider("Number of A/A simulations", 50, 500, 200, 50)
        with c3:
            base_cr_aa = st.slider("Baseline CVR for A/A", 0.01, 0.50, 0.12, 0.01)

        gen_aa    = SyntheticDataGenerator(seed=int(rng_seed))
        tester_aa = FrequentistTests(alpha=alpha)
        false_pos = 0
        p_values_aa: list[float] = []

        for _ in range(n_sims):
            ds_aa = gen_aa.conversion_experiment(n=n_aa, base_rate=base_cr_aa,
                                                  true_lift=0.0)
            r_aa = tester_aa.chi_square_test(
                len(ds_aa.control), int(ds_aa.control.sum()),
                len(ds_aa.variant),  int(ds_aa.variant.sum()),
            )
            p_values_aa.append(r_aa.p_value)
            if r_aa.significant:
                false_pos += 1

        fpr = false_pos / n_sims
        col1, col2, col3 = st.columns(3)
        col1.metric("Observed FPR",   f"{fpr:.1%}")
        col2.metric("Expected FPR (α)", f"{alpha:.1%}")
        col3.metric("Deviation",      f"{abs(fpr - alpha):.1%}",
                    delta_color="inverse" if abs(fpr - alpha) > 0.03 else "off")

        if abs(fpr - alpha) <= 0.03:
            st.success(
                f"FPR {fpr:.1%} is within 3pp of alpha = {alpha:.1%}. "
                "Pipeline appears correctly calibrated."
            )
        else:
            st.warning(
                f"FPR {fpr:.1%} deviates more than 3pp from alpha = {alpha:.1%}. "
                "Investigate assignment mechanism or check for SRM."
            )

        # p-value distribution — should be uniform under H₀
        st.markdown("**p-value distribution** (should be uniform under H₀)")
        pval_series = pd.Series(p_values_aa)
        bin_counts = pval_series.value_counts(bins=10, normalize=True).sort_index()
        st.bar_chart(bin_counts, height=200)
        st.caption(
            "A flat histogram confirms your pipeline is unbiased. "
            "A spike at p < 0.05 means systematic anti-conservative bias."
        )

    # ── 4c. Pre-experiment checklist ───────────────────────────────────────────
    st.divider()
    st.markdown("### Pre-experiment quality checklist")

    srm_ok = not srm_result.srm_detected

    # Build styled HTML table — dataframe can't colour individual cells
    def _status_badge(status: str, is_auto: bool) -> str:
        if status == "PASSED":
            return '<span style="background:#0d2118;color:#3fb950;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;border:1px solid #238636;">PASSED</span>'
        if status == "FAILED":
            return '<span style="background:#2d0c0c;color:#f85149;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;border:1px solid #da3633;">FAILED</span>'
        return '<span style="background:#1c2233;color:#8b949e;padding:2px 10px;border-radius:12px;font-size:12px;border:1px solid #2a2f3e;">Pending</span>'

    def _auto_badge(is_auto: bool) -> str:
        if is_auto:
            return '<span style="color:#58a6ff;font-size:12px;font-weight:500;">Auto</span>'
        return '<span style="color:#4c5566;font-size:12px;">Manual</span>'

    checklist_items = [
        ("SRM ratio check",                   "PASSED" if srm_ok else "FAILED", True),
        ("Sample size meets power target",     "PENDING", False),
        ("Holdout period observed (>= 1 week)","PENDING", False),
        ("Novelty effect window excluded",     "PENDING", False),
        ("Guardrail metrics defined",          "PENDING", False),
        ("A/A test run and calibrated",        "PENDING", False),
        ("Sequential testing boundaries set",  "PENDING", False),
        ("Experiment duration < 90 days",      "PENDING", False),
    ]

    rows_html = ""
    for check, status, is_auto in checklist_items:
        rows_html += f"""
        <tr style="border-bottom:1px solid #262c3d;">
          <td style="padding:10px 14px;color:#c9d1d9;font-size:13px;">{check}</td>
          <td style="padding:10px 14px;">{_status_badge(status, is_auto)}</td>
          <td style="padding:10px 14px;">{_auto_badge(is_auto)}</td>
        </tr>"""

    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;background:#161b27;
                  border:1px solid #262c3d;border-radius:10px;overflow:hidden;">
      <thead>
        <tr style="background:#1a2030;border-bottom:1px solid #262c3d;">
          <th style="padding:9px 14px;text-align:left;color:#8b949e;
                     font-size:11px;font-weight:600;letter-spacing:0.06em;
                     text-transform:uppercase;">Check</th>
          <th style="padding:9px 14px;text-align:left;color:#8b949e;
                     font-size:11px;font-weight:600;letter-spacing:0.06em;
                     text-transform:uppercase;">Status</th>
          <th style="padding:9px 14px;text-align:left;color:#8b949e;
                     font-size:11px;font-weight:600;letter-spacing:0.06em;
                     text-transform:uppercase;">Verified by</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)
    st.caption("PASSED = verified automatically by the dashboard. Pending = confirm before shipping.")


# ─────────────────────────────────────────────────────────────────────────────
# ████████████████████  SECTION 5: SEQUENTIAL MONITORING  ██████████████████████
# ─────────────────────────────────────────────────────────────────────────────

elif section == "📈 Sequential Monitoring":

    st.markdown('<span class="section-tag">Sequential</span>', unsafe_allow_html=True)
    st.title("Sequential Experiment Monitoring")
    st.caption(
        "Bayesian posterior updating · Early stopping rules · "
        "Day-by-day CVR and P(B>A) tracking · Expected loss monitoring"
    )

    _interp(
        "Sequential Bayesian monitoring lets you check results continuously without "
        "inflating Type-I error. The posterior updates each day as new data arrives. "
        "You can stop early when P(B > A) ≥ 95% AND expected loss < threshold — "
        "this is the correct decision rule, not a raw p-value check."
    )

    # ── 5a. Simulation parameters ──────────────────────────────────────────────
    with st.expander("⚙ Simulation parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            seq_days = st.slider("Experiment duration (days)", 10, 90, 30)
        with c2:
            seq_daily = st.slider("Daily users (both groups)", 100, 2000, 300, 50)
        with c3:
            seq_base = st.slider("Baseline CVR", 0.02, 0.40, 0.10, 0.01)
        with c4:
            seq_lift_pct = st.slider("True lift (%)", -15, 40, 15, 1)
            seq_lift = seq_lift_pct / 100.0

        c5, c6 = st.columns(2)
        with c5:
            seq_loss_thresh = st.slider(
                "Early stop — loss threshold", 0.001, 0.020, 0.005, 0.001,
                format="%.3f",
                help="Stop when expected loss drops below this value.",
            )
        with c6:
            seq_prob_thresh = st.slider(
                "Early stop — P(B>A) threshold", 0.80, 0.99, 0.95, 0.01,
                help="Stop when P(B>A) exceeds this probability.",
            )

    # ── 5b. Generate data & run Bayesian sequential update ─────────────────────
    gen_seq = SyntheticDataGenerator(seed=int(rng_seed))
    history = gen_seq.sequential_daily_data(
        total_days=seq_days,
        daily_users=seq_daily,
        base_rate=seq_base,
        true_lift=seq_lift,
    )

    bayes_seq = BayesianABTest(
        prior_alpha=1.0, prior_beta=1.0,
        monte_carlo_samples=20_000,
        seed=int(rng_seed),
    )

    with st.spinner("Running sequential Bayesian updates…"):
        seq_snapshots = bayes_seq.sequential_update(
            history,
            loss_threshold=seq_loss_thresh,
            early_stop_prob=seq_prob_thresh,
        )

    # Merge Bayesian outputs back into history
    for i, snap in enumerate(seq_snapshots):
        if i < len(history):
            history[i]["prob_b_beats_a"] = snap["prob_b_beats_a"]
            history[i]["expected_loss"]  = snap["expected_loss"]

    # Find early stop index
    early_stop_idx: int | None = None
    for i, snap in enumerate(seq_snapshots):
        if snap.get("early_stop"):
            early_stop_idx = i
            break

    # ── 5c. Key outcome metrics ────────────────────────────────────────────────
    st.divider()
    final_snap    = seq_snapshots[-1]
    days_run      = final_snap["day"]
    stopped_early = early_stop_idx is not None

    if stopped_early:
        stop_day = seq_snapshots[early_stop_idx]["day"]
        st.success(
            f"Early stop triggered on Day {stop_day} "
            f"— P(B>A) ≥ {seq_prob_thresh:.0%} and expected loss < {seq_loss_thresh:.3f}."
        )
    else:
        st.info(
            f"Experiment ran the full {seq_days}-day duration. "
            f"Final P(B>A) = {final_snap['prob_b_beats_a']:.1%}."
        )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Final P(B > A)",  f"{final_snap['prob_b_beats_a']:.1%}")
    m2.metric("Expected loss",   f"{final_snap['expected_loss']:.4%}")
    m3.metric("Days run",        f"{days_run}",
              delta=f"{seq_days - days_run} days saved" if stopped_early else "full duration",
              delta_color="normal" if stopped_early else "off")
    m4.metric("Final CVR (A)",   f"{final_snap['mean_a']:.2%}")
    m5.metric("Final CVR (B)",   f"{final_snap['mean_b']:.2%}",
              delta=f"{final_snap['mean_b'] - final_snap['mean_a']:+.2%}")

    # ── 5d. Time series chart ─────────────────────────────────────────────────
    st.divider()
    st.markdown("### Sequential monitoring chart")

    monitored_history = history[:len(seq_snapshots)]
    fig_seq = plot_sequential_monitoring(monitored_history, early_stop_idx)
    st.plotly_chart(fig_seq, use_container_width=True)

    # ── 5e. Day-by-day snapshot table ─────────────────────────────────────────
    st.divider()
    st.markdown("### Day-by-day snapshot table")

    snapshot_df = pd.DataFrame(seq_snapshots).rename(columns={
        "day":            "Day",
        "n_a":            "N(A) cumulative",
        "n_b":            "N(B) cumulative",
        "prob_b_beats_a": "P(B > A)",
        "expected_loss":  "Expected loss",
        "mean_a":         "CVR(A)",
        "mean_b":         "CVR(B)",
        "early_stop":     "Early stop",
        "recommendation": "Recommendation",
    })

    snapshot_df["P(B > A)"]      = snapshot_df["P(B > A)"].map("{:.1%}".format)
    snapshot_df["Expected loss"] = snapshot_df["Expected loss"].map("{:.4%}".format)
    snapshot_df["CVR(A)"]        = snapshot_df["CVR(A)"].map("{:.2%}".format)
    snapshot_df["CVR(B)"]        = snapshot_df["CVR(B)"].map("{:.2%}".format)
    snapshot_df["Early stop"]    = snapshot_df["Early stop"].map(
        lambda x: "⛔ Stop" if x else ""
    )

    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

    # ── 5f. Sequential decision guide ─────────────────────────────────────────
    with st.expander("Sequential decision guide"):
        st.markdown("""
**When to stop early:**
- P(B > A) ≥ 95% **AND** expected loss < threshold → **Ship B**
- P(B > A) ≤ 5%  **AND** expected loss (A) < threshold → **Revert to A**

**When to keep running:**
- P(B > A) between 5% and 95% → **Inconclusive — collect more data**
- Never stop based on a raw p-value mid-experiment (peeking problem)

**Bayesian vs Frequentist peeking:**
- Frequentist: each interim check at α = 0.05 inflates true FPR — use α-spending
- Bayesian: the posterior is always valid; early stopping does not inflate error rates

**Expected loss threshold guidance:**

| Threshold | Use case |
|---|---|
| 0.001 (0.1%) | High-stakes revenue experiments |
| 0.005 (0.5%) | Standard conversion experiments |
| 0.010 (1.0%) | Exploratory / low-stakes tests |
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.markdown(f"""
<div class="app-footer">
    <div>
        <strong style="color:#e6edf3;">⚡ {APP_NAME}</strong>
        <span style="color:#4c5566;"> · v2.1.0 · scipy · numpy · streamlit · plotly</span>
        <div style="color:#4c5566;font-size:0.78rem;margin-top:2px;">
            Welch t · χ² · Mann-Whitney U · Bayesian Beta-Binomial · CUPED · SRM · O'Brien-Fleming
        </div>
    </div>
    <div style="color:#8b949e;font-size:0.85rem;text-align:right;">
        Built by <strong style="color:#c9d1d9;">Sami</strong><br/>
        <a href="mailto:sami757007@gmail.com">sami757007@gmail.com</a> ·
        <a href="https://www.linkedin.com/in/samikhan07" target="_blank">LinkedIn</a>
    </div>
</div>
""", unsafe_allow_html=True)
