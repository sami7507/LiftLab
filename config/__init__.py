"""
config
──────
Central configuration package.

Exposes the singleton `settings` instance loaded from .env / environment
variables via pydantic-settings.

Usage (anywhere in the project)::

    from config import settings

    print(settings.app_env)        # "development"
    print(settings.default_alpha)  # 0.05
"""

from .settings import Settings, settings

__all__ = [
    "Settings",   # the pydantic model class (useful for type hints)
    "settings",   # the singleton instance — import this everywhere
]
