"""
src.utils
─────────
Supporting utilities for experiment design and data generation. Part of LiftLab.

Sub-modules
───────────

``sample_size``
    Functions:
      - ``compute_sample_size``         — required n per group (Cohen's h)
      - ``power_curve``                 — power vs. sample size data points
      - ``mde_curve``                   — MDE vs. required sample data points
      - ``check_sample_ratio_mismatch`` — SRM detection via chi-square
      - ``cuped_variance_reduction``    — CUPED adjusted metric + reduction %
      - ``sequential_alpha_spending``   — O'Brien-Fleming / Pocock boundaries
    Dataclasses:
      - ``SampleSizeResult``            — output of compute_sample_size()
      - ``SRMResult``                   — output of check_sample_ratio_mismatch()

``data_generator``
    Classes:
      - ``SyntheticDataGenerator``      — generates synthetic A/B datasets
    Dataclasses:
      - ``ABDataset``                   — container for control/variant arrays

Usage::

    # Sample size calculation
    from src.utils import compute_sample_size, SampleSizeResult

    result: SampleSizeResult = compute_sample_size(
        baseline_rate=0.10, mde_relative=0.10, alpha=0.05, power=0.80
    )
    print(f"Required n per group: {result.n_per_group:,}")

    # SRM detection
    from src.utils import check_sample_ratio_mismatch, SRMResult

    srm: SRMResult = check_sample_ratio_mismatch(assigned_a=5021, assigned_b=4620)
    if srm.srm_detected:
        print("SRM detected — stop the experiment!")

    # Synthetic data
    from src.utils import SyntheticDataGenerator, ABDataset

    gen = SyntheticDataGenerator(seed=42)
    ds: ABDataset = gen.conversion_experiment(n=1000, base_rate=0.12, true_lift=0.15)
    print(ds.control.mean(), ds.variant.mean())
"""

from .data_generator import ABDataset, SyntheticDataGenerator
from .sample_size import (
    SampleSizeResult,
    SRMResult,
    check_sample_ratio_mismatch,
    compute_sample_size,
    cuped_variance_reduction,
    mde_curve,
    power_curve,
    sequential_alpha_spending,
)

__all__ = [
    # ── sample_size ───────────────────────────────────────────────────────────
    "compute_sample_size",          # required n per group using Cohen's h
    "power_curve",                  # power vs sample size list of dicts
    "mde_curve",                    # MDE vs required sample list of dicts
    "check_sample_ratio_mismatch",  # SRM detection (chi-square)
    "cuped_variance_reduction",     # CUPED adjusted array + reduction fraction
    "sequential_alpha_spending",    # O'Brien-Fleming / Pocock alpha boundaries
    "SampleSizeResult",             # dataclass: n_per_group, n_total, days, h, …
    "SRMResult",                    # dataclass: srm_detected, p_value, ratio, …

    # ── data_generator ────────────────────────────────────────────────────────
    "SyntheticDataGenerator",       # generates conversion, revenue, session, NPS data
    "ABDataset",                    # dataclass: control array, variant array, metric_type
]
