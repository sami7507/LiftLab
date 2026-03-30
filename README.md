# ⚡ A/B Testing & Statistical Inference Dashboard

> Industry-grade experimentation platform built with Python, scipy, and Streamlit.
> Implements frequentist and Bayesian A/B testing with production-ready code, guardrail checks, and sequential monitoring.

---

## Features

| Section | What it does |
|---|---|
| **Frequentist Tests** | Welch t-test, Student t-test, Chi-square, Mann-Whitney U, Z-test for proportions, BH/Bonferroni correction |
| **Bayesian A/B** | Beta-Binomial conjugate model, Monte Carlo P(B>A), expected loss decision framework, prior sensitivity analysis |
| **Sample Size & Power** | Cohen's h power analysis, MDE tradeoff curves, CUPED variance reduction, O'Brien-Fleming α-spending |
| **Data Quality (SRM)** | Sample Ratio Mismatch detection, A/A test calibration, pre-experiment checklist |
| **Sequential Monitoring** | Day-by-day Bayesian updating, early stopping rules, daily CVR + P(B>A) tracking |

---

## Project Structure

```
ab_testing_dashboard/
│
├── app.py                         ← Streamlit entry point
├── requirements.txt               ← pip dependencies
├── pyproject.toml                 ← packaging, pytest, ruff config
├── .env.template                  ← environment variables template
├── conftest.py                    ← shared pytest fixtures
├── Makefile                       ← developer workflow shortcuts
├── Dockerfile                     ← containerised deployment
│
├── config/
│   ├── __init__.py
│   └── settings.py                ← pydantic BaseSettings
│
├── src/
│   ├── tests/
│   │   └── frequentist.py         ← all frequentist tests + corrections
│   ├── bayesian/
│   │   └── beta_binomial.py       ← Bayesian inference engine
│   ├── utils/
│   │   ├── sample_size.py         ← power analysis, SRM, CUPED
│   │   └── data_generator.py      ← synthetic dataset factory
│   └── visualization/
│       └── plots.py               ← all Plotly chart functions
│
└── tests/
    └── test_statistical_tests.py  ← unit + property-based test suite
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/your-username/ab-testing-dashboard.git
cd ab-testing-dashboard

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

---

## Docker

```bash
# Build
docker build -t ab-dashboard .

# Run
docker run -p 8501:8501 ab-dashboard

# Run with custom env file
docker run -p 8501:8501 --env-file .env ab-dashboard

# Open http://localhost:8501
```

---

## Running Tests

```bash
# Full suite with coverage
pytest

# Fast only (skip slow property tests)
pytest -m "not slow"

# Only Bayesian tests
pytest -m bayesian

# Only frequentist tests
pytest -m frequentist

# Verbose single file
pytest tests/test_statistical_tests.py -v

# View HTML coverage report
make test-cov
```

Coverage target: **80%** (enforced in `pyproject.toml`).

---

## Statistical Methods

### Frequentist

| Test | Use case | Effect size |
|---|---|---|
| Welch's t-test | Continuous metrics, unequal variance | Hedges' g |
| Student's t-test | Continuous metrics, equal variance | Cohen's d |
| Chi-square | Binary outcomes (CVR, CTR) | Cramér's V |
| Z-test (proportions) | Large-sample binary outcomes | Cohen's h |
| Mann-Whitney U | Non-normal / ordinal metrics (NPS, revenue) | CLES P(B>A) |

Multiple testing corrections: **Benjamini-Hochberg FDR** and **Bonferroni FWER**.

### Bayesian

**Model:** Beta-Binomial conjugate (exact posteriors, no MCMC needed)

```
Prior:      θ  ~  Beta(α₀, β₀)
Likelihood: k  ~  Binomial(n, θ)
Posterior:  θ|k,n  ~  Beta(α₀+k, β₀+n-k)
```

**Decision rule:** Deploy B when `P(B>A) ≥ 95%` **AND** `expected_loss < threshold`.

### Power Analysis

Sample size computed using **Cohen's h** (arcsine transformation of proportions) — more accurate than the plain z-test formula, especially for extreme base rates.

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

Typical reduction: **20–50%**, equivalent to running 25–100% more users.

### Sequential Testing

Alpha spending with **O'Brien-Fleming** boundaries prevents Type-I error inflation from interim peeking. Bayesian sequential monitoring updates posteriors daily — valid to check at any time.

### SRM Detection

Sample Ratio Mismatch detected via chi-square test on assignment counts. Uses `α = 0.01` (stricter than experiment alpha). SRM invalidates all metric results until root cause is fixed.

---

## Key Design Decisions

**Why not pymc for Bayesian?**
The Beta-Binomial is a conjugate model — the posterior is analytically exact. MCMC (pymc) adds heavy dependencies and runtime with no accuracy benefit for this use case.

**Why Cohen's h instead of the simple z-test formula?**
The arcsine transformation stabilises variance of proportions. At extreme base rates (1% or 45%), the simple formula over- or under-estimates required sample size significantly.

**Why Welch's t-test as default, not Student's?**
Welch's test does not assume equal variances and is strictly more general. It gives identical results when variances are equal, but doesn't break when they differ.

**Why Mann-Whitney for revenue?**
Revenue distributions are zero-inflated and heavily right-skewed. t-tests can be unreliable for these. Mann-Whitney (nonparametric) tests stochastic dominance without assuming any distribution.

---

## Stack

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

## License

MIT — see [LICENSE](LICENSE) for details.
