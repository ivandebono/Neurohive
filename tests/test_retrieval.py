"""Tests for BM25 index, Retriever, and embedding cache."""

from __future__ import annotations

import json

import numpy as np
import pytest

from neurohive.config import load_config
from neurohive.retrieval import Retriever, _BM25Index


def _retrieval_cfg() -> dict:
    return load_config()["retrieval"]


def _build_bm25(docs: list[str]) -> _BM25Index:
    cfg = _retrieval_cfg()
    return _BM25Index.build(docs, k1=float(cfg["bm25_k1"]), b=float(cfg["bm25_b"]))


class TestBM25Index:
    def test_relevant_doc_scores_above_zero(self):
        idx = _build_bm25(["action potential sodium channel", "synaptic plasticity LTP"])
        scores = idx.score("action potential")
        assert scores[0] > 0
        assert scores[0] > scores[1]

    def test_irrelevant_query_scores_zero(self):
        idx = _build_bm25(["neural signaling", "synaptic plasticity"])
        scores = idx.score("quantum mechanics")
        assert all(s == 0.0 for s in scores)

    def test_exact_match_scores_highest(self):
        docs = ["action potential", "synaptic plasticity", "voltage-gated channel"]
        idx = _build_bm25(docs)
        scores = idx.score("synaptic plasticity")
        assert scores[1] == max(scores)

    def test_empty_corpus_returns_empty_scores(self):
        idx = _build_bm25([])
        assert idx.score("anything") == []


class TestRetriever:
    def test_retrieve_nodes_returns_ranked_results(self, kb):
        r = Retriever(kb)
        results = r.retrieve_nodes("action potential", top_k=3)
        assert len(results) <= 3
        assert all(isinstance(score, float) for _, score in results)

    def test_top_node_is_most_relevant(self, kb):
        r = Retriever(kb)
        results = r.retrieve_nodes("action potential spike generation", top_k=1)
        assert len(results) == 1
        node, _ = results[0]
        assert "Action" in node.name or "Potential" in node.name

    def test_retrieve_nodes_scores_descending(self, kb):
        r = Retriever(kb)
        results = r.retrieve_nodes("neural signaling electrical", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_chunks_returns_results(self, kb):
        r = Retriever(kb)
        results = r.retrieve_chunks("action potential sodium", top_k=2)
        assert len(results) >= 1
        assert all(score > 0 for _, score in results)

    def test_retrieve_chunks_scores_descending(self, kb):
        r = Retriever(kb)
        results = r.retrieve_chunks("voltage-gated sodium channels", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_sibling_boost_elevates_same_doi_chunks(self, kb):
        r = Retriever(kb)
        # Both sample chunks share the same DOI; if one ranks highly the other
        # should receive a sibling boost and also appear in results.
        results = r.retrieve_chunks("voltage-gated sodium action potential", top_k=2)
        dois = [c.doi for c, _ in results]
        assert len(set(dois)) == 1  # both from the same DOI

    def test_bm25_only_mode_when_no_model_dir(self, kb, tmp_path):
        r = Retriever(kb, model_dir=tmp_path / "nonexistent")
        assert not r.is_hybrid
        results = r.retrieve_nodes("action potential", top_k=2)
        assert len(results) >= 1


class TestEmbeddingCache:
    """Unit-test the cache helpers by injecting synthetic embeddings."""

    def _make_retriever(self, kb, cache_dir):
        """Return a BM25-only Retriever wired to cache_dir."""
        return Retriever(kb, model_dir=None, cache_dir=cache_dir)

    def _fake_embeddings(self, r):
        """Attach synthetic float32 arrays so save/load can be tested."""
        n_nodes  = len(r._node_ids)
        n_chunks = len(r.kb.chunks)
        r._node_embs  = np.zeros((n_nodes,  8), dtype="float32")
        r._chunk_embs = np.zeros((n_chunks, 8), dtype="float32")

    def test_save_creates_expected_files(self, kb, tmp_path):
        r = self._make_retriever(kb, tmp_path)
        self._fake_embeddings(r)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        r._save_emb_cache(model_dir)
        cfg = _retrieval_cfg()
        assert (tmp_path / cfg["node_emb_file"]).exists()
        assert (tmp_path / cfg["chunk_emb_file"]).exists()
        assert (tmp_path / cfg["emb_meta_file"]).exists()

    def test_meta_records_model_and_corpus_hash(self, kb, tmp_path):
        r = self._make_retriever(kb, tmp_path)
        self._fake_embeddings(r)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        r._save_emb_cache(model_dir)
        meta = json.loads((tmp_path / _retrieval_cfg()["emb_meta_file"]).read_text())
        assert meta["model"] == str(model_dir.resolve())
        assert "corpus_hash" in meta

    def test_load_succeeds_on_valid_cache(self, kb, tmp_path):
        r = self._make_retriever(kb, tmp_path)
        self._fake_embeddings(r)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        r._save_emb_cache(model_dir)
        # Clear in-memory arrays and reload
        r._node_embs = r._chunk_embs = None
        assert r._load_emb_cache(model_dir) is True
        assert r._node_embs is not None
        assert r._chunk_embs is not None

    def test_load_fails_when_model_path_differs(self, kb, tmp_path):
        r = self._make_retriever(kb, tmp_path)
        self._fake_embeddings(r)
        model_a = tmp_path / "model_a"; model_a.mkdir()
        model_b = tmp_path / "model_b"; model_b.mkdir()
        r._save_emb_cache(model_a)
        assert r._load_emb_cache(model_b) is False

    def test_load_fails_when_corpus_changes(self, kb, tmp_path):
        r = self._make_retriever(kb, tmp_path)
        self._fake_embeddings(r)
        model_dir = tmp_path / "model"; model_dir.mkdir()
        r._save_emb_cache(model_dir)
        # Corrupt the corpus hash in the metadata
        meta_path = tmp_path / _retrieval_cfg()["emb_meta_file"]
        meta = json.loads(meta_path.read_text())
        meta["corpus_hash"] = "deadbeef"
        meta_path.write_text(json.dumps(meta))
        assert r._load_emb_cache(model_dir) is False

    def test_load_fails_when_cache_absent(self, kb, tmp_path):
        r = self._make_retriever(kb, tmp_path)
        model_dir = tmp_path / "model"; model_dir.mkdir()
        assert r._load_emb_cache(model_dir) is False

    def test_no_cache_dir_disables_caching(self, kb, tmp_path):
        r = Retriever(kb, model_dir=None, cache_dir=None)
        self._fake_embeddings(r)
        model_dir = tmp_path / "model"; model_dir.mkdir()
        r._save_emb_cache(model_dir)   # should silently do nothing
        assert not (tmp_path / _retrieval_cfg()["node_emb_file"]).exists()

    def test_corpus_hash_changes_when_nodes_differ(self, kb, tmp_path):
        r1 = self._make_retriever(kb, tmp_path)
        h1 = r1._corpus_hash()
        # Tamper with internal node IDs
        r1._node_ids = r1._node_ids + ["FAKE_EXTRA_NODE"]
        h2 = r1._corpus_hash()
        assert h1 != h2


class TestBM25Cache:
    """BM25 indices are persisted in the retrieval cache directory."""

    def test_retriever_writes_bm25_cache_files(self, kb, tmp_path):
        Retriever(kb, model_dir=tmp_path / "nonexistent", cache_dir=tmp_path)
        cfg = _retrieval_cfg()
        assert (tmp_path / cfg["bm25_meta_file"]).exists()
        assert (tmp_path / cfg["node_bm25_file"]).exists()
        assert (tmp_path / cfg["chunk_bm25_file"]).exists()

    def test_load_bm25_cache_succeeds(self, kb, tmp_path):
        first = Retriever(kb, model_dir=tmp_path / "nonexistent", cache_dir=tmp_path)
        second = Retriever(kb, model_dir=tmp_path / "nonexistent", cache_dir=tmp_path)
        assert second._node_bm25._N == first._node_bm25._N
        assert second._chunk_bm25._N == first._chunk_bm25._N
        assert second.retrieve_chunks("action potential sodium", top_k=1)

    def test_load_bm25_cache_rejects_stale_corpus_hash(self, kb, tmp_path):
        Retriever(kb, model_dir=tmp_path / "nonexistent", cache_dir=tmp_path)
        meta_path = tmp_path / _retrieval_cfg()["bm25_meta_file"]
        meta = json.loads(meta_path.read_text())
        meta["corpus_hash"] = "stale"
        meta_path.write_text(json.dumps(meta))

        r = Retriever(kb, model_dir=tmp_path / "nonexistent", cache_dir=tmp_path)
        refreshed = json.loads(meta_path.read_text())
        assert refreshed["corpus_hash"] == r._corpus_hash()
