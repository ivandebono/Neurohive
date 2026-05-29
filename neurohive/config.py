"""
Configuration loader for the Neurohive pipeline.

Reads ``config.toml`` at the repository root (next to ``main.py``).
Returns a plain dict so callers can use it without an extra dependency.

Resolution order (highest priority wins)
-----------------------------------------
  1. CLI flags        --model, --node-top-k, --chunk-top-k, etc.
  2. Environment vars ANTHROPIC_MODEL (set in .env or the shell)
  3. config.toml      the values in this file
  4. Hard-coded defaults inside each module

This module only handles layer 3: reading config.toml.
Layer 1 and 2 are handled in main.py after calling load_config().
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Repo-root config.toml, one level above this package directory.
_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

_DEFAULTS: dict = {
    "pipeline": {
        "node_top_k":          6,
        "chunk_top_k":         6,
        "confidence_threshold": 0.8,
    },
    # Model names are intentionally absent from these defaults.
    # config.toml is the single source of truth for all model identifiers.
    "anthropic":  {},
    "embeddings": {},
    "nli":        {"entailment_threshold": 0.5},
}


def load_config(path: Path | str | None = None) -> dict:
    """
    Load and return the pipeline configuration.

    Parameters
    ----------
    path : Optional override path for the TOML file (useful in tests).

    Returns
    -------
    dict with keys "pipeline", "anthropic", "embeddings", and "nli",
    each a nested dict.  Missing keys are filled in from
    ``_DEFAULTS``.
    """
    resolved = Path(path) if path else _CONFIG_PATH

    raw: dict = {}
    if resolved.exists():
        with open(resolved, "rb") as fh:
            raw = tomllib.load(fh)

    # Deep-merge: defaults provide the base; file values override section-by-section.
    merged: dict = {}
    for section, defaults in _DEFAULTS.items():
        merged[section] = {**defaults, **raw.get(section, {})}

    return merged
