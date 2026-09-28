<div align="center">

# ⚡ LiftLab

### A/B Testing & Statistical Inference Dashboard

*Frequentist and Bayesian experiment analysis, data-quality guardrails, and sequential monitoring — bring your own data and get a defensible ship / hold decision.*

[![Live Demo](https://img.shields.io/badge/live%20demo-liftlab--dashboard.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://liftlab-dashboard.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-59%20%7C%20pytest%20%2B%20hypothesis-0A9EDC)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-8b949e)](LICENSE)

**[🚀 Open the live app](https://liftlab-dashboard.streamlit.app)** · [Features](#features) · [Bring your own data](#bring-your-own-data) · [Statistical methods](#statistical-methods) · [Run locally](#run-locally) · [About the author](#about-the-author)

</div>

---

## Overview

**LiftLab** reproduces the analysis workflow used by product-analytics and growth teams:

1. **Check the data first** — detect Sample Ratio Mismatch before trusting any metric.
2. **Test the result** — run the right frequentist test for the metric type, with effect sizes and confidence intervals.
3. **Cross-check with Bayesian inference** — `P(B > A)` and expected loss give a decision framed in business terms.
4. **Size the next experiment** — power analysis, minimum detectable effect, and CUPED variance reduction.
5. **Monitor without peeking errors** — sequential Bayesian updating with early-stopping rules.

Every screen explains the *why* (assumptions, interpretation, pitfalls), not just the number.

---

## Try it in 30 seconds

1. Open the **[live app](https://liftlab-dashboard.streamlit.app)**.
2. In the sidebar, set **Input mode → Upload CSV**.
3. Upload one of the ready-made files from [`data/samples/`](data/samples):
   - [`conversion_long_format.csv`](data/samples/conversion_long_format.csv) — set *Metric type → Conversion (binary)*. Result: significant.
   - [`revenue_wide_format.csv`](data/samples/revenue_wide_format.csv) — set *Metric type → Revenue*. Result: not significant → **Hold**.

---

## Features

| Section | What it does |
|---|---|
| **🔬 Frequentist Tests** | Welch t, Student t, Chi-square, Mann-Whitney U, Z-test for proportions, BH and Bonferroni multiple-testing correction. Runs on synthetic, pasted, or uploaded data. |
| **🧠 Bayesian A/B** | Beta-Binomial model, Monte Carlo `P(B > A)`, expected-loss decision rule, credible intervals, prior-sensitivity analysis. |
| **📐 Sample Size & Power** | Cohen's h power analysis, MDE trade-off curve, duration estimator, CUPED variance reduction, O'Brien-Fleming α-spending. |
| **🔍 Data Quality (SRM)** | Sample Ratio Mismatch detection, A/A false-positive-rate simulation, pre-experiment checklist. |
| **📈 Sequential Monitoring** | Day-by-day Bayesian updating, early-stopping rules, live `P(B > A)` and CVR tracking (simulation). |

---

## Bring your own data

Which sections take your real numbers today:

| Section | Data input |
|---|---|
| Frequentist Tests | **Real data supported** — paste raw values, or upload a CSV (sidebar → *Input mode*). Synthetic demo mode is also available. |
| Bayesian A/B | **Real numbers** — type visitors and conversions for each arm. |
| Sample Size & Power | **Real numbers** — type baseline rate, MDE, and traffic. (The CUPED sub-demo uses simulated revenue.) |
| Data Quality (SRM) | **Real numbers** — type assignment counts for each arm. (The A/A calibration is a simulation by design.) |
| Sequential Monitoring | Simulation only. |

### CSV formats accepted (Frequentist Tests)

**Long format** — one row per user:

```csv
group,value
control,0
control,1
variant,1
variant,0
```

**Wide format** — one column per arm (columns may differ in length):

```csv
control,variant
5.10,6.20
4.80,5.90
```

- Group labels accepted: `control` / `variant`, `a` / `b`, or `0` / `1` (case-insensitive).
- For **Conversion (binary)**, any value `> 0` counts as a conversion.
- Pick the matching **Metric type** so the right tests run: binary → Chi-square + Z-test; continuous → Welch + Student t. Mann-Whitney U is always included.
- The app has a **Download sample template** button, and `data/samples/` has two working examples.

---

## Statistical Methods

### Frequentist

| Test | Use case | Effect size |
|---|---|---|
| Welch's t-test | Continuous metrics, unequal variance (default) | Hedges' g |
| Student's t-test | Continuous metrics, equal variance | Cohen's d |
| Chi-square | Binary outcomes (CVR, CTR) | Cramér's V |
| Z-test (proportions) | Large-sample binary outcomes | Cohen's h |
| Mann-Whitney U | Skewed / ordinal metrics (revenue, NPS) | CLES `P(B > A)` |

Multiple-testing correction: **Benjamini-Hochberg (FDR)** and **Bonferroni (FWER)**.

### Bayesian

Exact conjugate posterior — no MCMC required:

```
Prior:      θ  ~  Beta(α₀, β₀)
Likelihood: k  ~  Binomial(n, θ)
Posterior:  θ | k, n  ~  Beta(α₀ + k, β₀ + n − k)
```

**Decision rule:** ship B when `P(B > A) ≥ 95%` **and** `expected loss < threshold`.

### Power analysis (Cohen's h)

```
h = 2·arcsin(√p₂) − 2·arcsin(√p₁)
n = ((z_α + z_β) / h)²        # per group
```

The arcsine transform stabilises the variance of proportions, so it stays accurate at extreme base rates where the plain z-test formula drifts.

### CUPED

```
Y_cuped = Y − θ·(X − E[X])      θ = Cov(Y, X) / Var(X)
Variance reduction = 1 − ρ(Y, X)²
```

### Sequential testing

- **Frequentist:** O'Brien-Fleming α-spending boundaries prevent Type-I error inflation from interim peeking.
- **Bayesian:** posteriors remain valid at any stopping time, so the monitoring module updates daily and stops on `P(B > A)` plus expected loss instead of raw p-values.

### SRM detection

Chi-square on assignment counts at a stricter **α = 0.01**. A mismatch invalidates every downstream metric until the root cause (hashing bug, bot filtering, sticky sessions, logging drop, flag lag) is fixed.

---

## Architecture

```
LiftLab/
├── app.py                          # Streamlit UI + orchestration
├── requirements.txt                # runtime dependencies (pinned)
├── requirements-dev.txt            # + pytest, pytest-cov, hypothesis
├── pyproject.toml                  # tooling / pytest config
├── Makefile · Dockerfile · .env.template · conftest.py
├── LICENSE
│
├── .streamlit/config.toml          # dark theme (no server/port overrides — see Deployment)
├── config/settings.py              # pydantic-settings configuration
├── data/samples/                   # ready-to-upload example CSVs
│
├── src/
│   ├── tests/frequentist.py        # frequentist tests + corrections
│   ├── bayesian/beta_binomial.py   # Bayesian engine
│   ├── utils/
│   │   ├── sample_size.py          # power, MDE, SRM, CUPED, α-spending
│   │   └── data_generator.py       # synthetic dataset factory
│   └── visualization/plots.py      # Plotly figures
│
└── tests/test_statistical_tests.py # 59 unit + property-based tests
```

**Design principle:** the statistics layer (`src/`) has no Streamlit imports. Every test and model is a pure function returning a typed result object, so it can be unit-tested and reused in notebooks or batch jobs.

---

## Run locally

Requires **Python 3.11+**.

```bash
git clone https://github.com/sami7507/LiftLab.git
cd LiftLab

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # add requirements-dev.txt for the test suite
streamlit run app.py               # http://localhost:8501
```

Optional: `cp .env.template .env` to override defaults (α, power, Monte Carlo samples).

### Docker

```bash
docker build -t liftlab .
docker run -p 8501:8501 liftlab
```

---

## Deployment (Streamlit Community Cloud)

The live app is deployed from `main` on [Streamlit Community Cloud](https://streamlit.io/cloud): connect the repo, set the main file to `app.py`, choose Python 3.11.

Two lessons from getting it live, kept here because they cost real debugging time:

1. **Pin your dependencies.** With open `>=` ranges, a fresh Cloud install can resolve untested new majors (pandas 3.0, numpy 2.5). `requirements.txt` pins the tested set instead.
2. **Don't override server networking in `.streamlit/config.toml`.** Setting `[server] port`, `headless`, or `[browser] serverAddress` works locally but made the Cloud health check fail with `connection refused` on `/healthz`. Cloud manages port binding and the public URL itself. The committed config only sets theme, upload size, and logging.

---

## Testing

```bash
pip install -r requirements-dev.txt

pytest                     # full suite with coverage
pytest -m "not slow"       # skip slow property-based tests
pytest tests/ -v           # verbose
```

59 tests cover all five frequentist methods (CI coverage checks, effect-size direction, small-sample warnings), Bayesian posterior math and expected loss, sample-size monotonicity, SRM, CUPED, α-spending, the data generator, and Hypothesis property-based checks.

---

## Key design decisions

**Why not PyMC?** Beta-Binomial is conjugate — the posterior is exact. MCMC would add heavy dependencies and runtime with no accuracy gain for a conversion metric.

**Why Cohen's h for sample size?** Variance-stabilising arcsine transform; accurate at 1% or 45% base rates where the simple formula over/under-estimates.

**Why Welch as the default t-test?** No equal-variance assumption; matches Student's when variances agree and stays valid when they don't.

**Why Mann-Whitney for revenue?** Revenue is zero-inflated and right-skewed; Mann-Whitney tests stochastic dominance without a distribution assumption.

**Why expected loss, not just `P(B > A)`?** A 96% chance of winning by a hair can be worse than a 90% chance of a large win. Expected loss prices the downside of being wrong.

---

## Roadmap

- [x] Real data input for Frequentist Tests (paste or CSV upload)
- [x] Live deployment on Streamlit Community Cloud
- [ ] CSV upload for Sequential Monitoring (cumulative daily counts)
- [ ] Multi-variant (A/B/n) tests
- [ ] Export a one-click experiment readout as PDF
- [ ] Alerting hook (Slack/webhook) when SRM is detected

---

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Statistics | scipy, numpy, statsmodels |
| Dashboard | Streamlit |
| Charts | Plotly |
| Config / logging | pydantic-settings, loguru |
| Testing | pytest, pytest-cov, hypothesis |
| Container | Docker |

---

## Contributing

Issues and pull requests are welcome. Please run `pytest` before opening a PR.

## License

MIT — see [LICENSE](LICENSE).

---

## About the Author

**Sami**
📧 [sami757007@gmail.com](mailto:sami757007@gmail.com)
🔗 [linkedin.com/in/samikhan07](https://www.linkedin.com/in/samikhan07)

Built to demonstrate applied statistics end to end: experiment design, frequentist and Bayesian inference, data-quality guardrails, and sequential decision-making — packaged as a tool an experimentation team could actually use. Happy to walk through any design decision above.
