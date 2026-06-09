"""Tests for the config loader and Config class."""

from __future__ import annotations

import pytest

from neurohive.config import load_config
from neurohive.schema import (
    Config,
    PipelineConfig,
    RetrievalConfig,
    CacheConfig,
    DriftConfig,
    NliConfig,
)

# A complete, valid TOML config used as a base for override tests.
_COMPLETE_TOML = """\
[paths]
data_dir   = "data"
env_file   = ".env"
logs_dir   = "logs"
models_dir = "models"

[pipeline]
node_top_k           = 6
chunk_top_k          = 6
confidence_threshold = 0.8

[retrieval]
bm25_k1            = 1.5
bm25_b             = 0.75
node_rrf_k         = 30
chunk_rrf_k        = 60
sibling_discount   = 0.5
bm25_cache_version = 1
bm25_meta_file     = "bm25_meta.json"
node_bm25_file     = "node_bm25.json"
chunk_bm25_file    = "chunk_bm25.json"
emb_meta_file      = "emb_meta.json"
node_emb_file      = "node_embs.npy"
chunk_emb_file     = "chunk_embs.npy"

[anthropic]
model = "claude-haiku-4-5-20251001"

[ollama]
model = "ministral-3:3b-instruct-2512-q4_K_M"
host  = "http://localhost:11434"

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"

[cache]
enabled     = false
ttl_seconds = 86400
db_file     = "query_cache.db"

[reranker]
enabled         = false
pool_multiplier = 3

[ingestor]
semantic_dedup_enabled   = true
semantic_dedup_threshold = 0.92

[drift]
baseline_file           = "drift_baseline.json"
vocab_warning_threshold = 0.05
vocab_alert_threshold   = 0.15
volume_warning_pct      = 0.20
volume_alert_pct        = 0.50
embedding_warning_dist  = 0.10
embedding_alert_dist    = 0.25

[nli]
model                = "cross-encoder/nli-deberta-v3-small"
entailment_threshold = 0.5
"""


def test_missing_file_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.toml")


def test_returns_config_instance(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text(_COMPLETE_TOML)
    cfg = load_config(f)
    assert isinstance(cfg, Config)


def test_attribute_access(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text(_COMPLETE_TOML)
    cfg = load_config(f)
    assert cfg.pipeline.node_top_k == 6
    assert cfg.pipeline.chunk_top_k == 6
    assert cfg.paths.data_dir == "data"
    assert cfg.retrieval.bm25_meta_file == "bm25_meta.json"
    assert cfg.ollama.model == "ministral-3:3b-instruct-2512-q4_K_M"
    assert cfg.nli.entailment_threshold == 0.5


def test_correct_types(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text(_COMPLETE_TOML)
    cfg = load_config(f)
    assert isinstance(cfg.pipeline.node_top_k, int)
    assert isinstance(cfg.pipeline.confidence_threshold, float)
    assert isinstance(cfg.retrieval.bm25_k1, float)
    assert isinstance(cfg.retrieval.node_rrf_k, int)
    assert isinstance(cfg.cache.enabled, bool)
    assert isinstance(cfg.cache.ttl_seconds, int)
    assert isinstance(cfg.drift.vocab_warning_threshold, float)
    assert isinstance(cfg.nli.entailment_threshold, float)


def test_custom_value_used(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text(_COMPLETE_TOML.replace("node_top_k           = 6", "node_top_k           = 12"))
    cfg = load_config(f)
    assert cfg.pipeline.node_top_k == 12
    assert cfg.pipeline.chunk_top_k == 6


def test_all_expected_sections_present(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text(_COMPLETE_TOML)
    cfg = load_config(f)
    assert isinstance(cfg.paths, object)
    assert isinstance(cfg.pipeline, PipelineConfig)
    assert isinstance(cfg.retrieval, RetrievalConfig)
    assert isinstance(cfg.cache, CacheConfig)
    assert isinstance(cfg.drift, DriftConfig)
    assert isinstance(cfg.nli, NliConfig)


def test_incomplete_config_raises_validation_error(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text("[pipeline]\nnode_top_k = 12\n")
    with pytest.raises(ValueError, match="validation failed"):
        load_config(f)


def test_drift_threshold_order_enforced(tmp_path):
    bad = _COMPLETE_TOML.replace(
        "vocab_warning_threshold = 0.05\nvocab_alert_threshold   = 0.15",
        "vocab_warning_threshold = 0.20\nvocab_alert_threshold   = 0.10",
    )
    f = tmp_path / "config.toml"
    f.write_text(bad)
    with pytest.raises(ValueError, match="validation failed"):
        load_config(f)
