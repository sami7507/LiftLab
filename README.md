<div align="center">

# ⚡ LiftLab

### A/B Testing & Statistical Inference Dashboard

*Frequentist and Bayesian experimentation, guardrail checks, and sequential monitoring — in one production-style tool.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-pytest%20%7C%20hypothesis-0A9EDC)](#testing)
[![Coverage](https://img.shields.io/badge/coverage-target%2080%25-3fb950)](#testing)
[![License: MIT](https://img.shields.io/badge/license-MIT-8b949e)](#license)
[![Made by Sami](https://img.shields.io/badge/made%20by-Sami-58a6ff)](https://www.linkedin.com/in/samikhan07)

[Live Demo](#) · [Features](#features) · [Statistical Methods](#statistical-methods) · [Quickstart](#quickstart) · [About the Author](#about-the-author)

</div>

---

## Overview

**LiftLab** is a full-stack experimentation dashboard that reproduces the analysis pipeline used by real product-analytics and growth teams: run a frequentist test, cross-check it against a Bayesian model, size the next experiment correctly, catch broken instrumentation *before* trusting the result, and monitor a live rollout without falling into the peeking trap.

It's built to demonstrate applied statistics, not just wrap `scipy.stats` in a UI — every screen includes the *why*, not only the *what*: effect sizes, confidence vs. credible intervals, expected-loss decision theory, SRM root-cause guidance, and anti-peeking sequential boundaries.

> 💡 **Why this project exists:** most portfolios show a model. This shows a *decision system* — the kind of tool a data scientist or analytics engineer builds internally to stop shipping A/B tests on vibes.

---

## Features

| Module | What it does |
|---|---|
| **🔬 Frequentist Tests** | Welch's t-test, Student's t-test, Chi-square, Mann-Whitney U, Z-test for proportions, with Benjamini-Hochberg (FDR) and Bonferroni (FWER) multiple-testing correction |
| **🧠 Bayesian A/B** | Beta-Binomial conjugate model, Monte Carlo `P(B > A)`, expected-loss decision framework, credible intervals, prior-sensitivity analysis |
| **📐 Sample Size & Power** | Cohen's h power analysis, MDE tradeoff curves, CUPED variance reduction, O'Brien-Fleming α-spending for interim looks |
| **🔍 Data Quality (SRM)** | Sample Ratio Mismatch detection, A/A test calibration (false-positive rate check), pre-experiment quality checklist |
| **📈 Sequential Monitoring** | Day-by-day Bayesian posterior updates, early-stopping rules, live CVR + `P(B > A)` tracking |

---

## Screenshot

> _Add a screenshot or GIF of the dashboard here — this single image does more for recruiter attention than any paragraph of text._
>
> `docs/screenshot-frequentist.png` · `docs/screenshot-bayesian.png`

---

## Statistical Methods

### Frequentist

| Test | Use case | Effect size |
|---|---|---|
| Welch's t-test | Continuous metrics, unequal variance | Hedges' g |
| Student's t-test | Continuous metrics, equal variance | Cohen's d |
| Chi-square | Binary outcomes (CVR, CTR) | Cramér's V |
| Z-test (proportions) | Large-sample binary outcomes | Cohen's h |
| Mann-Whitney U | Non-normal / ordinal metrics (NPS, revenue) | CLES `P(B>A)` |

Multiple-testing correction: **Benjamini-Hochberg (FDR)** and **Bonferroni (FWER)**.

### Bayesian

Exact conjugate posterior — no MCMC required:

```
Prior:      θ  ~  Beta(α₀, β₀)
Likelihood: k  ~  Binomial(n, θ)
Posterior:  θ|k,n  ~  Beta(α₀+k, β₀+n-k)
```

**Decision rule:** ship B when `P(B > A) ≥ 95%` **and** `expected_loss < threshold`.

### Power Analysis

Sample size via **Cohen's h** (arcsine transform of proportions) — more accurate than the plain z-test formula at extreme base rates:

```
h = 2·arcsin(√p₂) − 2·arcsin(√p₁)
n = ((z_α + z_β) / h)²
```

### CUPED

Variance reduction via pre-experiment covariate regression:

```
Y_cuped = Y − θ·(X − E[X])     where θ = Cov(Y,X) / Var(X)
Variance reduction = 1 − ρ(Y,X)²
```

Typical reduction: **20–50%**, equivalent to running 25–100% more users for free.

### Sequential Testing

- **Frequentist:** O'Brien-Fleming α-spending boundaries prevent Type-I error inflation from interim peeking.
- **Bayesian:** posteriors are valid at any stopping time — no correction needed, which is why the Sequential Monitoring module uses daily Bayesian updates instead of raw p-values.

### SRM Detection

Sample Ratio Mismatch is checked via a chi-square test on assignment counts, at a stricter **α = 0.01**. SRM invalidates every downstream metric until the root cause (hashing bug, bot filtering, sticky sessions, logging drop, flag-rollout lag) is found and fixed.

---

## Architecture

```
liftlab/
│
├── app.py                         ← Streamlit entry point (UI + orchestration)
├── requirements.txt               ← pip dependencies
├── pyproject.toml                 ← packaging, pytest, ruff config
├── .env.template                  ← environment variables template
├── conftest.py                    ← shared pytest fixtures
├── Makefile                       ← developer workflow shortcuts
├── Dockerfile                     ← containerised deployment
│
├── config/
│   └── settings.py                ← pydantic BaseSettings
│
├── src/
│   ├── tests/frequentist.py       ← all frequentist tests + corrections
│   ├── bayesian/beta_binomial.py  ← Bayesian inference engine
│   ├── utils/
│   │   ├── sample_size.py         ← power analysis, SRM, CUPED
│   │   └── data_generator.py      ← synthetic dataset factory
│   └── visualization/plots.py     ← all Plotly chart functions
│
└── tests/
    └── test_statistical_tests.py  ← unit + property-based test suite
```

**Design principle:** the statistics layer (`src/`) has zero Streamlit imports — every test, model, and calculation is a pure function that returns a typed result object, independently unit-testable and reusable outside the dashboard (e.g. in a notebook or a batch job).

---

## Quickstart

### 1. Clone and set up the environment

```bash
git clone https://github.com/sami7507/liftlab.git
cd liftlab

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.template .env
# Edit .env if you want to change defaults (optional)
```

### 3. Run the dashboard

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### Or use Make

```bash
make install-dev   # full setup including dev tools
make run           # start dashboard
make test          # run test suite
make lint          # ruff linter
make format        # auto-format code
```

### Docker

```bash
# Build
docker build -t liftlab .

# Run
docker run -p 8501:8501 liftlab

# Run with a custom env file
docker run -p 8501:8501 --env-file .env liftlab

# Open http://localhost:8501
```

---

## Testing

```bash
# Full suite with coverage
pytest

# Fast only (skip slow property-based tests)
pytest -m "not slow"

# Only Bayesian tests
pytest -m bayesian

# Only frequentist tests
pytest -m frequentist

# Verbose single file
pytest tests/test_statistical_tests.py -v

# HTML coverage report
make test-cov
```

Coverage target: **80%** (enforced in `pyproject.toml`). Property-based tests (via `hypothesis`) fuzz the statistical functions against edge cases — zero variance, tiny samples, extreme base rates — that hand-picked unit tests tend to miss.

---

## Key Design Decisions

**Why not PyMC for the Bayesian module?**
The Beta-Binomial pair is conjugate — the posterior is analytically exact. MCMC adds heavy dependencies and runtime with no accuracy benefit for a binomial conversion metric.

**Why Cohen's h instead of the plain z-test sample-size formula?**
The arcsine transform stabilises the variance of a proportion. At extreme base rates (1% or 45%) the plain formula over- or under-estimates the required sample size significantly.

**Why Welch's t-test as the default, not Student's?**
Welch's test doesn't assume equal variances and is strictly more general — it matches Student's when variances are equal, but doesn't silently break when they aren't.

**Why Mann-Whitney for revenue metrics?**
Revenue is typically zero-inflated and heavily right-skewed. A t-test's normality assumption is shaky there; Mann-Whitney tests stochastic dominance without assuming a distribution shape.

---

## Roadmap

- [ ] CSV upload → real experiment data (currently synthetic-only end to end)
- [ ] Multi-armed / multi-variant tests (not just A vs. B)
- [ ] Stratified / CUPED-adjusted sequential monitoring
- [ ] Export a one-click experiment readout as PDF
- [ ] Slack/webhook alert on SRM detection

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Statistics | scipy, numpy, statsmodels |
| Dashboard | Streamlit |
| Charts | Plotly |
| Config | pydantic-settings |
| Logging | loguru |
| Testing | pytest, hypothesis |
| Linting | ruff |
| Container | Docker |

---

## Contributing

Issues and PRs are welcome. Please run `make lint` and `make test` before opening a pull request.

## License

MIT — see [LICENSE](LICENSE) for details.

---

## About the Author

**Sami**
📧 [sami757007@gmail.com](mailto:sami757007@gmail.com)
🔗 [linkedin.com/in/samikhan07](https://www.linkedin.com/in/sami7507)

Built as a demonstration of applied statistical reasoning end-to-end: frequentist and Bayesian inference, experiment design, data-quality guardrails, and sequential decision-making — packaged as a tool a real experimentation team could actually use.

If you're a hiring manager or engineer reviewing this: happy to walk through any design decision above, especially the trade-offs between the frequentist and Bayesian modules.
