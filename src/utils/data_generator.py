"""
src/utils/data_generator.py
────────────────────────────
Generates realistic synthetic A/B test data for LiftLab demos, testing, and onboarding.

Data patterns modelled:
  - Revenue: log-normal (right-skewed, heavy tail)
  - Conversion: Bernoulli
  - Session time: Gamma (bounded, right-skewed)
  - NPS: discrete [-100, 100]
  - Daily cumulative data (for sequential analysis demo)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from loguru import logger


@dataclass
class ABDataset:
    """Container for a synthetic A/B experiment."""
    experiment_name: str
    metric_type: str       # "conversion" | "revenue" | "session_time" | "nps"
    control: np.ndarray
    variant: np.ndarray
    true_lift: float       # ground truth for simulation
    has_srm: bool = False  # whether SRM was injected


class SyntheticDataGenerator:
    """
    Generate synthetic A/B test datasets with configurable parameters.

    Usage
    -----
    >>> gen = SyntheticDataGenerator(seed=42)
    >>> dataset = gen.conversion_experiment(n=1000, base_rate=0.12, true_lift=0.15)
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        logger.info(f"SyntheticDataGenerator initialised with seed={seed}")

    def conversion_experiment(
        self,
        n: int = 1000,
        base_rate: float = 0.12,
        true_lift: float = 0.15,      # relative lift (0.15 = 15%)
        inject_srm: bool = False,
    ) -> ABDataset:
        """
        Binary conversion experiment (click, sign-up, purchase).

        Control: Bernoulli(base_rate)
        Variant: Bernoulli(base_rate * (1 + true_lift))
        """
        variant_rate = base_rate * (1 + true_lift)
        n_ctrl = n
        n_var = int(n * 0.6) if inject_srm else n   # inject SRM: variant gets fewer

        control = self.rng.binomial(1, base_rate, n_ctrl).astype(float)
        variant = self.rng.binomial(1, min(variant_rate, 1.0), n_var).astype(float)

        logger.debug(
            f"Conversion experiment: n={n}, base={base_rate:.2%}, "
            f"lift={true_lift:.1%}, SRM={inject_srm}"
        )
        return ABDataset(
            experiment_name=f"Conversion (lift={true_lift:+.0%})",
            metric_type="conversion",
            control=control,
            variant=variant,
            true_lift=true_lift,
            has_srm=inject_srm,
        )

    def revenue_experiment(
        self,
        n: int = 1000,
        base_mean: float = 25.0,
        base_sigma: float = 0.8,       # log-normal sigma (controls skew)
        true_lift: float = 0.10,
    ) -> ABDataset:
        """
        Revenue per user experiment — log-normal distribution.

        Log-normal is the standard model for revenue because:
        - Non-negative (revenue ≥ 0)
        - Right-skewed (a few high-value users)
        - Multiplicative growth (percentage lifts)
        """
        mu_ctrl = np.log(base_mean) - base_sigma**2 / 2
        mu_var = mu_ctrl + np.log(1 + true_lift)

        control = self.rng.lognormal(mu_ctrl, base_sigma, n)
        variant = self.rng.lognormal(mu_var, base_sigma, n)

        # Add zero-inflation (not all users purchase)
        zero_mask_ctrl = self.rng.random(n) > 0.60
        zero_mask_var = self.rng.random(n) > 0.55   # variant has slightly higher purchase rate
        control = control * zero_mask_ctrl
        variant = variant * zero_mask_var

        return ABDataset(
            experiment_name=f"Revenue per User (lift={true_lift:+.0%})",
            metric_type="revenue",
            control=control,
            variant=variant,
            true_lift=true_lift,
        )

    def session_time_experiment(
        self,
        n: int = 1000,
        base_minutes: float = 5.0,
        true_lift: float = 0.08,
    ) -> ABDataset:
        """
        Session duration — Gamma distribution (bounded, right-skewed).

        Gamma is appropriate for time-to-event and duration metrics.
        Shape parameter controls skewness; scale controls mean.
        """
        shape = 2.0    # controls shape of distribution
        scale_ctrl = base_minutes / shape
        scale_var = scale_ctrl * (1 + true_lift)

        control = self.rng.gamma(shape, scale_ctrl, n)
        variant = self.rng.gamma(shape, scale_var, n)

        return ABDataset(
            experiment_name=f"Session Time (lift={true_lift:+.0%})",
            metric_type="session_time",
            control=control,
            variant=variant,
            true_lift=true_lift,
        )

    def nps_experiment(
        self,
        n: int = 1000,
        base_nps: float = 20.0,     # scale: -100 to 100
        true_lift: float = 10.0,    # absolute NPS points
    ) -> ABDataset:
        """
        Net Promoter Score experiment — ordinal, non-normal, bounded.

        NPS = % Promoters (9-10) - % Detractors (0-6)
        Best analysed with Mann-Whitney U (non-parametric).
        """
        # Simulate individual scores 0-10, then convert to NPS
        scores_ctrl = self._simulate_nps_scores(n, base_nps)
        scores_var = self._simulate_nps_scores(n, base_nps + true_lift)

        return ABDataset(
            experiment_name=f"NPS (lift={true_lift:+.0f} pts)",
            metric_type="nps",
            control=scores_ctrl,
            variant=scores_var,
            true_lift=true_lift,
        )

    def _simulate_nps_scores(self, n: int, target_nps: float) -> np.ndarray:
        """Generate 0-10 Likert scores consistent with a given NPS level."""
        # NPS ~ 20 → roughly 40% promoters, 20% detractors
        # We use a mixture of beta distributions scaled to 0-10
        nps_norm = (target_nps + 100) / 200   # map to [0, 1]
        a, b = 1 + 4 * nps_norm, 1 + 4 * (1 - nps_norm)
        raw = self.rng.beta(a, b, n) * 10
        return np.round(raw).clip(0, 10)

    def sequential_daily_data(
        self,
        total_days: int = 30,
        daily_users: int = 200,
        base_rate: float = 0.10,
        true_lift: float = 0.15,
    ) -> list[dict]:
        """
        Generate day-by-day cumulative conversion data for sequential analysis.

        Returns list of {day, n_a, c_a, n_b, c_b} cumulative snapshots.
        """
        variant_rate = base_rate * (1 + true_lift)
        n_a_cum, c_a_cum = 0, 0
        n_b_cum, c_b_cum = 0, 0
        history = []

        for day in range(1, total_days + 1):
            n_day = int(self.rng.poisson(daily_users))
            n_a_day = n_day // 2
            n_b_day = n_day - n_a_day

            c_a_day = int(self.rng.binomial(n_a_day, base_rate))
            c_b_day = int(self.rng.binomial(n_b_day, variant_rate))

            n_a_cum += n_a_day
            n_b_cum += n_b_day
            c_a_cum += c_a_day
            c_b_cum += c_b_day

            history.append({
                "day": day,
                "n_a": n_a_cum, "c_a": c_a_cum,
                "n_b": n_b_cum, "c_b": c_b_cum,
                "cvr_a": c_a_cum / max(n_a_cum, 1),
                "cvr_b": c_b_cum / max(n_b_cum, 1),
            })
        return history

    def to_dataframe(self, dataset: ABDataset) -> pd.DataFrame:
        """Convert ABDataset to tidy long-format DataFrame."""
        ctrl_df = pd.DataFrame({
            "group": "control",
            "value": dataset.control,
            "converted": (dataset.control > 0).astype(int)
            if dataset.metric_type == "conversion" else None,
        })
        var_df = pd.DataFrame({
            "group": "variant",
            "value": dataset.variant,
            "converted": (dataset.variant > 0).astype(int)
            if dataset.metric_type == "conversion" else None,
        })
        df = pd.concat([ctrl_df, var_df], ignore_index=True)
        df["metric_type"] = dataset.metric_type
        df["experiment_name"] = dataset.experiment_name
        return df
