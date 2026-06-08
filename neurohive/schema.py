"""Pydantic models for config.toml validation.

These models are used exclusively for validation in ``load_config()``.
They do not provide defaults — all defaults live in ``config.py:_DEFAULTS``.
Unknown extra keys are silently ignored so future config extensions don't
break older code.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PathsConfig(_Base):
    data_dir:   str
    env_file:   str
    logs_dir:   str
    models_dir: str


class PipelineConfig(_Base):
    node_top_k:           int   = Field(..., gt=0)
    chunk_top_k:          int   = Field(..., gt=0)
    confidence_threshold: float = Field(..., ge=0.0, le=1.0)


class RetrievalConfig(_Base):
    bm25_k1:          float = Field(..., gt=0.0)
    bm25_b:           float = Field(..., ge=0.0, le=1.0)
    node_rrf_k:       int   = Field(..., gt=0)
    chunk_rrf_k:      int   = Field(..., gt=0)
    sibling_discount: float = Field(..., ge=0.0, le=1.0)


class CacheConfig(_Base):
    enabled:     bool
    ttl_seconds: int = Field(..., ge=0)
    db_file:     str


class RerankerConfig(_Base):
    enabled:         bool
    pool_multiplier: int = Field(..., ge=1)


class IngestorConfig(_Base):
    semantic_dedup_enabled:   bool
    semantic_dedup_threshold: float = Field(..., ge=0.0, le=1.0)


class DriftConfig(_Base):
    baseline_file:           str
    vocab_warning_threshold: float = Field(..., ge=0.0, le=1.0)
    vocab_alert_threshold:   float = Field(..., ge=0.0, le=1.0)
    volume_warning_pct:      float = Field(..., ge=0.0)
    volume_alert_pct:        float = Field(..., ge=0.0)
    embedding_warning_dist:  float = Field(..., ge=0.0)
    embedding_alert_dist:    float = Field(..., ge=0.0)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> DriftConfig:
        if self.vocab_warning_threshold >= self.vocab_alert_threshold:
            raise ValueError(
                "drift.vocab_warning_threshold must be less than vocab_alert_threshold"
            )
        if self.volume_warning_pct >= self.volume_alert_pct:
            raise ValueError(
                "drift.volume_warning_pct must be less than volume_alert_pct"
            )
        if self.embedding_warning_dist >= self.embedding_alert_dist:
            raise ValueError(
                "drift.embedding_warning_dist must be less than embedding_alert_dist"
            )
        return self


class AnthropicConfig(_Base):
    model: Optional[str] = None


class OllamaConfig(_Base):
    model: str
    host:  str


class EmbeddingsConfig(_Base):
    model: str


class NliConfig(_Base):
    model:                Optional[str] = None
    entailment_threshold: float = Field(..., ge=0.0, le=1.0)


class NeurohiveConfig(_Base):
    paths:     PathsConfig
    pipeline:  PipelineConfig
    retrieval: RetrievalConfig
    cache:     CacheConfig
    reranker:  RerankerConfig
    ingestor:  IngestorConfig
    drift:     DriftConfig
    anthropic: AnthropicConfig
    ollama:    OllamaConfig
    embeddings: EmbeddingsConfig
    nli:       NliConfig
