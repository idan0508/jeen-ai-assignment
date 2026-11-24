"""Application configuration loaded from the environment / .env file."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from doc_indexer.errors import ConfigError


@dataclass(frozen=True)
class Config:
    # repr=False keeps secrets out of logs, tracebacks and debugger output.
    gemini_api_key: str = field(repr=False)
    postgres_url: str = field(repr=False)


def load_config() -> Config:
    """Load settings from .env (if present) and the process environment.

    Only variable *names* are ever reported in errors, never their values.
    """
    load_dotenv()
    values = {
        name: os.getenv(name, "").strip()
        for name in ("GEMINI_API_KEY", "POSTGRES_URL")
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
    return Config(
        gemini_api_key=values["GEMINI_API_KEY"],
        postgres_url=values["POSTGRES_URL"],
    )
