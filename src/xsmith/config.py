"""Centralized settings. Env-loaded via pydantic-settings.

CLI flags override env via explicit construction in `cli.py`.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    ANTHROPIC_API_KEY: str | None = None
    MODEL: str = "claude-sonnet-4-6"
    STEP_BUDGET: int = 24
    K: int = 5
    GAMMA: float = 0.5
    MAX_TURNS_GEN: int = 8
    MAX_TURNS_SCORE: int = 3
    MAX_USD: float | None = None

    # Docker
    DOCKER_IMAGE: str = "xsmith-runner:latest"
    DOCKER_TIMEOUT_S: float = 60.0

    # Subprocess (used by integration tests + local dev)
    SUBPROCESS_TIMEOUT_S: float = 30.0


def load_settings() -> Settings:
    return Settings()
