"""Tests for the DriftDetector and DriftSnapshot."""

from __future__ import annotations

import json
import math

import pytest

from neurohive.drift import (
    DriftDetector,
    DriftReport,
    DriftSnapshot,
    _build_vocab,
    _cosine_distance,
    _js_divergence,
    _to_distribution,
    _tokenize,
    _year_stats,
)
from neurohive.entities import PaperChunk
from neurohive.ingestor import IncrementalIngestor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_lowercase_alphanumeric(self):
        assert _tokenize("Hello, World!") == ["hello", "world"]

    def test_numbers_kept(self):
        assert "123" in _tokenize("voltage 123 mV")

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = _tokenize("action-potential spike.")
        assert "action" in tokens
        assert "potential" in tokens
        assert "spike" in tokens


class TestBuildVocab:
    def test_counts_tokens(self):
        vocab, total = _build_vocab(["the cat sat", "the cat"])
        assert vocab["the"] == 2
        assert vocab["cat"] == 2
        assert vocab["sat"] == 1

    def test_total_is_all_tokens(self):
        _, total = _build_vocab(["a b c", "d e"])
        assert total == 5

    def test_empty_corpus(self):
        vocab, total = _build_vocab([])
        assert vocab == {}
        assert total == 0

    def test_top_n_trimmed(self):
        import neurohive.drift as drift_module
        original = drift_module._VOCAB_TOP_N
        drift_module._VOCAB_TOP_N = 2
        try:
            vocab, _ = _build_vocab(["a b c d e"])
            assert len(vocab) <= 2
        finally:
            drift_module._VOCAB_TOP_N = original


class TestJsDivergence:
    def test_identical_distributions_zero(self):
        p = {"a": 0.5, "b": 0.5}
        assert _js_divergence(p, p) == pytest.approx(0.0, abs=1e-9)

    def test_completely_disjoint_distributions(self):
        p = {"a": 1.0}
        q = {"b": 1.0}
        div = _js_divergence(p, q)
        assert div == pytest.approx(math.log(2), abs=1e-9)

    def test_symmetric(self):
        p = {"a": 0.7, "b": 0.3}
        q = {"a": 0.2, "b": 0.8}
        assert _js_divergence(p, q) == pytest.approx(_js_divergence(q, p), abs=1e-9)

    def test_bounded(self):
        p = {"a": 0.6, "b": 0.4}
        q = {"c": 1.0}
        div = _js_divergence(p, q)
        assert 0.0 <= div <= math.log(2) + 1e-9

    def test_empty_distributions_zero(self):
        assert _js_divergence({}, {}) == pytest.approx(0.0, abs=1e-9)


class TestCosineDistance:
    def test_identical_vectors_zero_distance(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors_distance_one(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors_distance_two(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_distance(a, b) == pytest.approx(2.0, abs=1e-9)

    def test_zero_vector_returns_one(self):
        assert _cosine_distance([0.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


class TestYearStats:
    def test_single_year(self):
        s = _year_stats([2010])
        assert s["min"] == 2010
        assert s["max"] == 2010
        assert s["mean"] == 2010
        assert s["median"] == 2010

    def test_odd_count(self):
        s = _year_stats([2000, 2010, 2020])
        assert s["median"] == 2010

    def test_even_count(self):
        s = _year_stats([2000, 2010, 2020, 2030])
        assert s["median"] == pytest.approx(2015.0)

    def test_empty_returns_empty(self):
        assert _year_stats([]) == {}


# ---------------------------------------------------------------------------
# DriftSnapshot serialization
# ---------------------------------------------------------------------------

class TestDriftSnapshot:
    def _make_snapshot(self) -> DriftSnapshot:
        return DriftSnapshot(
            timestamp="2026-01-01T00:00:00+00:00",
            n_nodes=5,
            n_edges=4,
            n_chunks=2,
            n_papers=1,
            node_type_counts={"Pillar": 1, "Subpillar": 1, "Research_area": 2, "Theory": 1},
            edge_type_counts={"HAS_SUBPILLAR": 1, "HAS_RESEARCH_AREA": 1, "HAS_THEORY": 1, "EXPLAINS": 1},
            paper_year_stats={"min": 2020.0, "max": 2020.0, "mean": 2020.0, "median": 2020.0},
            vocab={"voltage": 3, "channel": 2, "action": 1},
            vocab_total=6,
        )

    def test_json_round_trip(self):
        snap = self._make_snapshot()
        restored = DriftSnapshot.from_json(snap.to_json())
        assert restored.n_nodes == snap.n_nodes
        assert restored.n_chunks == snap.n_chunks
        assert restored.vocab == snap.vocab
        assert restored.timestamp == snap.timestamp

    def test_save_and_load(self, tmp_path):
        snap = self._make_snapshot()
        path = tmp_path / "baseline.json"
        snap.save(path)
        assert path.exists()
        loaded = DriftSnapshot.load(path)
        assert loaded.n_edges == snap.n_edges
        assert loaded.node_type_counts == snap.node_type_counts

    def test_saved_file_is_valid_json(self, tmp_path):
        snap = self._make_snapshot()
        path = tmp_path / "baseline.json"
        snap.save(path)
        data = json.loads(path.read_text())
        assert data["n_nodes"] == 5
        assert data["vocab"]["voltage"] == 3

    def test_none_fields_serialized(self, tmp_path):
        snap = self._make_snapshot()
        assert snap.chunk_emb_centroid is None
        path = tmp_path / "baseline.json"
        snap.save(path)
        loaded = DriftSnapshot.load(path)
        assert loaded.chunk_emb_centroid is None


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetectorSnapshot:
    def test_snapshot_counts(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert snap.n_nodes == 5
        assert snap.n_edges == 4
        assert snap.n_chunks == 2
        assert snap.n_papers == 1

    def test_snapshot_node_type_counts(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert snap.node_type_counts["Pillar"] == 1
        assert snap.node_type_counts["Research_area"] == 2
        assert snap.node_type_counts["Theory"] == 1

    def test_snapshot_edge_type_counts(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert snap.edge_type_counts["HAS_SUBPILLAR"] == 1
        assert snap.edge_type_counts["EXPLAINS"] == 1

    def test_snapshot_vocab_not_empty(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert snap.vocab_total > 0
        assert len(snap.vocab) > 0

    def test_snapshot_year_stats(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert snap.paper_year_stats["min"] == 2020
        assert snap.paper_year_stats["max"] == 2020

    def test_snapshot_timestamp_is_string(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert isinstance(snap.timestamp, str)
        assert "2026" in snap.timestamp or "202" in snap.timestamp

    def test_no_embeddings_centroid_is_none(self, kb):
        detector = DriftDetector(kb, cache_dir=kb.cache_dir)
        snap = detector.snapshot()
        assert snap.chunk_emb_centroid is None
        assert snap.node_emb_centroid is None


class TestDriftDetectorBaseline:
    def test_save_creates_file(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        saved = detector.save_baseline()
        assert saved == path
        assert path.exists()

    def test_baseline_exists_false_before_save(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        assert not detector.baseline_exists()

    def test_baseline_exists_true_after_save(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        assert detector.baseline_exists()

    def test_load_baseline_matches_saved(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        loaded = detector.load_baseline()
        assert loaded.n_nodes == 5
        assert loaded.n_chunks == 2

    def test_load_baseline_raises_if_missing(self, kb, tmp_path):
        path = tmp_path / "nonexistent.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        with pytest.raises(FileNotFoundError):
            detector.load_baseline()


class TestDriftDetectorCheck:
    def test_no_drift_status_ok(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        report = detector.check()
        assert report.status == "ok"

    def test_no_drift_volume_changes_zero(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        report = detector.check()
        for metric, info in report.volume_changes.items():
            assert info["pct_change"] == pytest.approx(0.0)

    def test_no_drift_vocab_divergence_zero(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        report = detector.check()
        assert report.vocab_divergence == pytest.approx(0.0, abs=1e-9)

    def test_volume_growth_triggers_warning(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        # Use a very low threshold so small growth triggers a warning.
        detector = DriftDetector(
            kb, cache_dir=kb.cache_dir, baseline_path=path,
            thresholds={"volume_warning": 0.01, "volume_alert": 0.99},
        )
        detector.save_baseline()

        # Add one chunk (50% growth over 2 baseline chunks).
        new_chunk = PaperChunk(
            id=999, doi="10.9/new", title="New Paper", authors="A, B",
            year=2023, chunk_index=0,
            text="Synaptic plasticity involves long-term changes in synaptic strength.",
            taxonomy_node_ids=(),
        )
        ingestor = IncrementalIngestor(kb)
        ingestor.add_chunks([new_chunk])

        report = detector.check()
        assert report.status in ("warning", "alert")
        assert report.volume_changes["chunks"]["current"] == 3

    def test_volume_alert_threshold(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(
            kb, cache_dir=kb.cache_dir, baseline_path=path,
            thresholds={"volume_warning": 0.01, "volume_alert": 0.10},
        )
        detector.save_baseline()

        # Adding 1 chunk to a 2-chunk baseline is +50%, well above the 10% alert threshold.
        new_chunk = PaperChunk(
            id=998, doi="10.9/alert", title="Alert Paper", authors="C",
            year=2024, chunk_index=0,
            text="Dopaminergic modulation alters prefrontal cortex activity.",
            taxonomy_node_ids=(),
        )
        IncrementalIngestor(kb).add_chunks([new_chunk])
        report = detector.check()
        assert report.status == "alert"

    def test_new_node_type_triggers_warning(self, kb, tmp_path):
        from neurohive.entities import Node

        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()

        # Add a node with a novel type.
        new_node = Node(
            id="XX001", name="Novel_Concept", type="Paradigm",
            description="A node whose type does not exist in the baseline.",
        )
        IncrementalIngestor(kb).add_nodes([new_node])

        report = detector.check()
        assert "Paradigm" in report.new_node_types
        assert report.status in ("warning", "alert")

    def test_check_report_has_timestamps(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        report = detector.check()
        assert isinstance(report.baseline_timestamp, str)
        assert isinstance(report.current_timestamp, str)

    def test_check_uses_passed_baseline(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        snap = detector.snapshot()
        # Pass the baseline explicitly (bypasses file I/O).
        report = detector.check(baseline=snap)
        assert report.status == "ok"

    def test_report_str_contains_status(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=path)
        detector.save_baseline()
        report = detector.check()
        text = str(report)
        assert "OK" in text or "WARNING" in text or "ALERT" in text
        assert "Volume changes" in text

    def test_vocab_drift_detected_with_low_threshold(self, kb, tmp_path):
        path = tmp_path / "baseline.json"
        # Set a tiny threshold so any vocabulary change triggers a warning.
        detector = DriftDetector(
            kb, cache_dir=kb.cache_dir, baseline_path=path,
            thresholds={"vocab_warning": 0.001, "vocab_alert": 0.99,
                        "volume_warning": 0.99, "volume_alert": 0.99},
        )
        detector.save_baseline()

        # Add a chunk with completely unrelated vocabulary.
        new_chunk = PaperChunk(
            id=997, doi="10.9/vocab", title="Quantum Computing Review",
            authors="Z, Y", year=2022, chunk_index=0,
            text=(
                "Qubit entanglement enables superposition states that surpass classical "
                "binary computation. Quantum gates manipulate probability amplitudes "
                "in Hilbert space, exploiting interference for algorithmic speedup."
            ),
            taxonomy_node_ids=(),
        )
        IncrementalIngestor(kb).add_chunks([new_chunk])

        report = detector.check()
        # Vocabulary divergence must be positive.
        assert report.vocab_divergence is not None
        assert report.vocab_divergence > 0
        # With threshold=0.001 and new vocabulary, status must be at least warning.
        assert report.status in ("warning", "alert")
