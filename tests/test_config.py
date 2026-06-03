"""Tests for the config loader."""

from __future__ import annotations

import pytest

from neurohive.config import load_config


def test_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert cfg["pipeline"]["node_top_k"] == 6
    assert cfg["pipeline"]["chunk_top_k"] == 6
    assert cfg["pipeline"]["confidence_threshold"] == 0.8
    assert cfg["nli"]["entailment_threshold"] == 0.5


def test_file_value_overrides_default(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[pipeline]\nnode_top_k = 12\n")
    cfg = load_config(f)
    assert cfg["pipeline"]["node_top_k"] == 12
    assert cfg["pipeline"]["chunk_top_k"] == 6   # untouched default


def test_partial_section_merges_with_defaults(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[pipeline]\nchunk_top_k = 3\n")
    cfg = load_config(f)
    assert cfg["pipeline"]["chunk_top_k"] == 3
    assert cfg["pipeline"]["node_top_k"] == 6    # default preserved


def test_absent_section_uses_defaults(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[anthropic]\nmodel = 'claude-test'\n")
    cfg = load_config(f)
    assert cfg["anthropic"]["model"] == "claude-test"
    assert cfg["pipeline"]["node_top_k"] == 6


def test_all_expected_sections_present(tmp_path):
    cfg = load_config(tmp_path / "nonexistent.toml")
    assert set(cfg.keys()) >= {"pipeline", "anthropic", "embeddings", "nli"}
