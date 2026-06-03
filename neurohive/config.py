"""
Configuration loader for the Neurohive pipeline.

Reads ``config.toml`` at the repository root (next to ``main.py``).
Returns a plain dict so callers can use it without an extra dependency.

Resolution order (highest priority wins)
-----------------------------------------
  1. CLI flags        --model, --node-top-k, --chunk-top-k, etc.
  2. Environment vars ANTHROPIC_MODEL (set in .env or the shell)
  3. config.toml      the values in this file
  4. Fallback defaults in this loader

This module only handles layer 3: reading config.toml.
Layer 1 and 2 are handled in main.py after calling load_config().
"""

from __future__ import annotations

import tomllib
from pathlib import Path

# Repo-root config.toml, one level above this package directory.
_CONFIG_PATH = Path(__file__).parent.parent / "config.toml"

_DEFAULTS: dict = {
    "paths": {
        "data_dir":   "data",
        "env_file":   ".env",
        "logs_dir":   "logs",
        "models_dir": "models",
    },
    "pipeline": {
        "node_top_k":          6,
        "chunk_top_k":         6,
        "confidence_threshold": 0.8,
    },
    "retrieval": {
        "bm25_k1":            1.5,
        "bm25_b":             0.75,
        "node_rrf_k":         30,
        "chunk_rrf_k":        60,
        "sibling_discount":   0.5,
        "bm25_cache_version": 1,
        "bm25_meta_file":     "bm25_meta.json",
        "node_bm25_file":     "node_bm25.json",
        "chunk_bm25_file":    "chunk_bm25.json",
        "emb_meta_file":      "emb_meta.json",
        "node_emb_file":      "node_embs.npy",
        "chunk_emb_file":     "chunk_embs.npy",
    },
    # API model names are intentionally absent from these defaults.
    # config.toml is the single source of truth for Anthropic model identifiers.
    "anthropic":  {},
    "embeddings": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
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
    dict with keys "paths", "pipeline", "retrieval", "anthropic",
    "embeddings", and "nli", each a nested dict. Missing keys are filled
    in from ``_DEFAULTS``.
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
