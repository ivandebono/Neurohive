"""Tests for BM25 index and Retriever."""

from __future__ import annotations

import pytest

from neurohive.retrieval import Retriever, _BM25Index


class TestBM25Index:
    def test_relevant_doc_scores_above_zero(self):
        idx = _BM25Index.build(["action potential sodium channel", "synaptic plasticity LTP"])
        scores = idx.score("action potential")
        assert scores[0] > 0
        assert scores[0] > scores[1]

    def test_irrelevant_query_scores_zero(self):
        idx = _BM25Index.build(["neural signaling", "synaptic plasticity"])
        scores = idx.score("quantum mechanics")
        assert all(s == 0.0 for s in scores)

    def test_exact_match_scores_highest(self):
        docs = ["action potential", "synaptic plasticity", "voltage-gated channel"]
        idx = _BM25Index.build(docs)
        scores = idx.score("synaptic plasticity")
        assert scores[1] == max(scores)

    def test_empty_corpus_returns_empty_scores(self):
        idx = _BM25Index.build([])
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
