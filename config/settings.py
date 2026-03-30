"""
config/settings.py
──────────────────
Central configuration using Pydantic BaseSettings.
All values can be overridden via environment variables or .env file.
"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings. Override via environment variables."""

    # ── App ────────────────────────────────────────────────────────────────
    app_env: str = Field("development", description="Runtime environment")
    app_port: int = Field(8501, description="Streamlit port")
    app_title: str = Field("A/B Testing Dashboard", description="UI title")

    # ── Statistical Defaults ───────────────────────────────────────────────
    default_alpha: float = Field(0.05, ge=0.001, le=0.20,
                                 description="Significance threshold")
    default_power: float = Field(0.80, ge=0.50, le=0.99,
                                 description="Desired statistical power (1-β)")
    default_mde: float = Field(0.10, ge=0.01, le=0.50,
                               description="Minimum detectable effect (relative)")
    monte_carlo_samples: int = Field(50_000, ge=1_000, le=1_000_000,
                                     description="MC samples for Bayesian inference")

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", description="Logging verbosity")
    log_file: str = Field("logs/app.log", description="Log file path")

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = Field("data/samples", description="Sample data directory")
    export_dir: str = Field("data/exports", description="Export output directory")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
