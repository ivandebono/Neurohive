"""
Configuration loader for the Neurohive pipeline.

Reads ``config.toml`` at the repository root (next to ``main.py``).

Resolution order (highest priority wins)
-----------------------------------------
  1. CLI flags        --model, --node-top-k, --chunk-top-k, etc.
  2. Environment vars ANTHROPIC_MODEL (set in .env or the shell)
  3. config.toml      the values in this file — the sole source of truth

This module only handles layer 3: reading config.toml.
Layer 1 and 2 are handled in main.py after calling load_config().
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from neurohive.schema import Config  # re-exported for callers

# Repo-root config.toml, one level above this package directory.
_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"


def load_config(path: Path | str | None = None) -> Config:
    """
    Load and return the pipeline configuration from config.toml.

    Parameters
    ----------
    path : Optional override path for the TOML file (useful in tests).

    Returns
    -------
    Config instance with typed, attribute-accessible configuration.

    Raises
    ------
    FileNotFoundError if the config file does not exist.
    ValueError if the config fails schema validation.
    """
    resolved = Path(path) if path else _CONFIG_PATH

    if not resolved.exists():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")

    with open(resolved, "rb") as fh:
        raw = tomllib.load(fh)

    from pydantic import ValidationError  # noqa: PLC0415
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
        raise ValueError(f"config.toml validation failed — {problems}") from exc
