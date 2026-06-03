"""Tests for KnowledgeBase: ingestion, dedup, lookups, and graph helpers."""

from __future__ import annotations

import warnings

import pytest

from neurohive.knowledge_base import KnowledgeBase
from neurohive.models import Node


# ---------------------------------------------------------------------------
# Verbose ingestion and rebuild
# ---------------------------------------------------------------------------

class TestIngestFlags:
    def test_first_run_populates_database(self, data_dir):
        db_path = data_dir / "database" / "neurohive.db"
        assert not db_path.exists()
        KnowledgeBase(data_dir)
        assert db_path.exists()

    def test_verbose_prints_progress(self, data_dir, capsys):
        KnowledgeBase(data_dir, verbose=True)
        out = capsys.readouterr().out
        assert "Building knowledge base" in out
        assert "taxonomy_nodes.csv" in out
        assert "taxonomy_edges.csv" in out
        assert "paper_chunks.json" in out
        assert "done" in out

    def test_silent_by_default(self, data_dir, capsys):
        KnowledgeBase(data_dir)
        assert capsys.readouterr().out == ""

    def test_rebuild_recreates_database(self, data_dir):
        kb1 = KnowledgeBase(data_dir)
        db_path = data_dir / "database" / "neurohive.db"
        mtime_before = db_path.stat().st_mtime

        import time; time.sleep(0.05)
        KnowledgeBase(data_dir, rebuild=True)
        mtime_after = db_path.stat().st_mtime

        assert mtime_after > mtime_before  # file was replaced

    def test_rebuild_preserves_data(self, data_dir):
        KnowledgeBase(data_dir)
        kb = KnowledgeBase(data_dir, rebuild=True)
        assert len(kb.nodes) == 5
        assert len(kb.edges) == 4

    def test_no_rebuild_skips_ingestion(self, data_dir, capsys):
        KnowledgeBase(data_dir, verbose=True)   # first run: ingests
        capsys.readouterr()
        KnowledgeBase(data_dir, verbose=True)   # second run: DB already populated
        out = capsys.readouterr().out
        assert out == ""  # nothing printed — ingestion was skipped

    def test_rebuild_verbose_prints_progress(self, data_dir, capsys):
        KnowledgeBase(data_dir)                          # silent first build
        KnowledgeBase(data_dir, rebuild=True, verbose=True)
        out = capsys.readouterr().out
        assert "Building knowledge base" in out


# ---------------------------------------------------------------------------
# _dedup
# ---------------------------------------------------------------------------

class TestDedup:
    def _node(self, id_, name="X", type_="Pillar", desc="d"):
        return Node(id=id_, name=name, type=type_, description=desc)

    def test_unique_items_all_kept(self):
        items = [self._node("A", "Foo"), self._node("B", "Bar")]
        result = KnowledgeBase._dedup(items, lambda n: (n.name, n.type, n.description), "node")
        assert len(result) == 2

    def test_duplicate_content_drops_second(self):
        items = [self._node("A"), self._node("B")]  # same content, different id
        result = KnowledgeBase._dedup(items, lambda n: (n.name, n.type, n.description), "node")
        assert len(result) == 1
        assert result[0].id == "A"

    def test_duplicate_emits_warning(self):
        items = [self._node("A"), self._node("B")]
        with pytest.warns(UserWarning, match="Duplicate node"):
            KnowledgeBase._dedup(items, lambda n: (n.name, n.type, n.description), "node")

    def test_warning_identifies_both_items(self):
        items = [self._node("A"), self._node("B")]
        with pytest.warns(UserWarning) as record:
            KnowledgeBase._dedup(items, lambda n: (n.name, n.type, n.description), "node")
        msg = str(record[0].message)
        assert "id='A'" in msg or "A" in msg
        assert "id='B'" in msg or "B" in msg

    def test_no_warning_for_unique_items(self):
        items = [self._node("A", "Foo"), self._node("B", "Bar")]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KnowledgeBase._dedup(items, lambda n: (n.name, n.type, n.description), "node")

    def test_only_id_differs_still_detected(self):
        # Identical content, different id → content duplicate
        a = Node(id="X1", name="Same", type="Pillar", description="same")
        b = Node(id="X2", name="Same", type="Pillar", description="same")
        with pytest.warns(UserWarning):
            result = KnowledgeBase._dedup([a, b], lambda n: (n.name, n.type, n.description), "node")
        assert len(result) == 1

    def test_three_duplicates_keeps_first_only(self):
        items = [self._node("A"), self._node("B"), self._node("C")]
        with pytest.warns(UserWarning):
            result = KnowledgeBase._dedup(items, lambda n: (n.name, n.type, n.description), "node")
        assert len(result) == 1
        assert result[0].id == "A"


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class TestIngestion:
    def test_correct_counts(self, kb):
        assert len(kb.nodes) == 5
        assert len(kb.edges) == 4
        assert len(kb.chunks) == 2

    def test_node_fields_loaded(self, kb):
        n = kb.nodes_by_id["P001"]
        assert n.name == "Neural_Signaling"
        assert n.type == "Pillar"
        assert "pillar" in n.description.lower()

    def test_edge_fields_loaded(self, kb):
        edges = kb.outgoing.get("P001", [])
        sp_edge = next(e for e in edges if e.to_id == "SP001")
        assert sp_edge.relationship_type == "HAS_SUBPILLAR"
        assert sp_edge.confidence == 1.0
        assert sp_edge.map_source == "canonical"

    def test_chunk_fields_loaded(self, kb):
        chunk = kb.chunks[0]
        assert chunk.doi == "10.1/ap"
        assert "RA001" in chunk.taxonomy_node_ids or "TH001" in chunk.taxonomy_node_ids


# ---------------------------------------------------------------------------
# Isolated-node detection
# ---------------------------------------------------------------------------

class TestIsolated:
    def test_isolated_node_flagged_true(self, kb):
        lone = kb.nodes_by_id["LONE"]
        assert lone.isolated is True

    def test_connected_node_flagged_false(self, kb):
        assert kb.nodes_by_id["P001"].isolated is False
        assert kb.nodes_by_id["RA001"].isolated is False

    def test_isolated_count(self, kb):
        isolated = [n for n in kb.nodes if n.isolated]
        assert len(isolated) == 1
        assert isolated[0].id == "LONE"


# ---------------------------------------------------------------------------
# Dict-like proxies
# ---------------------------------------------------------------------------

class TestNodeIndex:
    def test_getitem_returns_correct_node(self, kb):
        n = kb.nodes_by_id["RA001"]
        assert n.id == "RA001"
        assert n.name == "Action_Potential"

    def test_getitem_missing_raises_key_error(self, kb):
        with pytest.raises(KeyError):
            _ = kb.nodes_by_id["DOES_NOT_EXIST"]

    def test_contains_true_for_existing(self, kb):
        assert "P001" in kb.nodes_by_id

    def test_contains_false_for_missing(self, kb):
        assert "FAKE" not in kb.nodes_by_id

    def test_get_returns_node(self, kb):
        n = kb.nodes_by_id.get("SP001")
        assert n is not None
        assert n.id == "SP001"

    def test_get_returns_default_for_missing(self, kb):
        assert kb.nodes_by_id.get("FAKE") is None
        assert kb.nodes_by_id.get("FAKE", "fallback") == "fallback"


class TestEdgeIndex:
    def test_outgoing_returns_edges(self, kb):
        edges = kb.outgoing.get("P001")
        assert edges is not None
        assert any(e.to_id == "SP001" for e in edges)

    def test_incoming_returns_edges(self, kb):
        edges = kb.incoming.get("SP001")
        assert edges is not None
        assert any(e.from_id == "P001" for e in edges)

    def test_outgoing_missing_returns_default(self, kb):
        assert kb.outgoing.get("FAKE") is None
        assert kb.outgoing.get("FAKE", []) == []

    def test_incoming_missing_returns_default(self, kb):
        assert kb.incoming.get("LONE") is None


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

class TestAncestors:
    def test_research_area_has_subpillar_and_pillar_ancestors(self, kb):
        ancestor_ids = {a.id for a in kb.ancestors("RA001")}
        assert "SP001" in ancestor_ids
        assert "P001" in ancestor_ids

    def test_pillar_has_no_ancestors(self, kb):
        assert kb.ancestors("P001") == []

    def test_isolated_node_has_no_ancestors(self, kb):
        assert kb.ancestors("LONE") == []


class TestBreadcrumb:
    def test_leaf_node_includes_full_path(self, kb):
        crumb = kb.breadcrumb("RA001")
        assert "Neural Signaling" in crumb
        assert "Electrical Signaling" in crumb
        assert "Action Potential" in crumb
        assert "→" in crumb

    def test_pillar_breadcrumb_is_just_name(self, kb):
        crumb = kb.breadcrumb("P001")
        assert crumb == "Neural Signaling"

    def test_missing_id_returns_id_string(self, kb):
        assert kb.breadcrumb("NONEXISTENT") == "NONEXISTENT"


class TestSmartExpand:
    def test_expands_to_parent_subpillar(self, kb):
        # One hop from RA001 reaches its immediate parent SP001 (HAS_RESEARCH_AREA)
        expanded = kb.smart_expand({"RA001"})
        assert "SP001" in expanded

    def test_one_hop_does_not_reach_grandparent(self, kb):
        # P001 is two hops from RA001 — not included in a single expansion
        expanded = kb.smart_expand({"RA001"})
        assert "P001" not in expanded

    def test_expands_to_pillar_from_subpillar(self, kb):
        # One hop from SP001 reaches P001 via incoming HAS_SUBPILLAR
        expanded = kb.smart_expand({"SP001"})
        assert "P001" in expanded

    def test_seed_always_in_result(self, kb):
        expanded = kb.smart_expand({"RA001"})
        assert "RA001" in expanded

    def test_isolated_node_expands_to_itself_only(self, kb):
        expanded = kb.smart_expand({"LONE"})
        assert expanded == {"LONE"}

    def test_confidence_threshold_filters_low_confidence(self, kb):
        # EXPLAINS edge has confidence 0.9; threshold above that should exclude it
        expanded_strict = kb.smart_expand({"RA001"}, confidence_threshold=0.95)
        # TH001 is reached via incoming EXPLAINS from TH001 → RA001 (conf 0.9, semantic)
        # semantic edges get +0.1 boost to threshold, so effective threshold = 1.05 > 0.9 → excluded
        assert "TH001" not in expanded_strict

    def test_zero_threshold_follows_all_edges(self, kb):
        expanded = kb.smart_expand({"RA001"}, confidence_threshold=0.0)
        assert "TH001" in expanded


class TestChunksForNodes:
    def test_returns_chunks_linked_to_node(self, kb):
        chunks = kb.chunks_for_nodes({"RA001"})
        assert len(chunks) == 1
        assert chunks[0].doi == "10.1/ap"

    def test_returns_empty_for_unlinked_node(self, kb):
        assert kb.chunks_for_nodes({"P001"}) == []

    def test_returns_empty_for_empty_input(self, kb):
        assert kb.chunks_for_nodes(set()) == []

    def test_deduplicates_across_multiple_nodes(self, kb):
        # Both RA001 and TH001 link to chunks from the same DOI
        chunks = kb.chunks_for_nodes({"RA001", "TH001"})
        dois = [c.doi for c in chunks]
        assert len(dois) == len(set(dois)) or len(chunks) == 2  # at most one chunk per unique chunk id
