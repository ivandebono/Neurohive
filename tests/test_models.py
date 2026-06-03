"""Tests for Node, Edge, and PaperChunk dataclasses."""

from __future__ import annotations

import pytest

from neurohive.models import Edge, Node, PaperChunk


class TestNode:
    def test_basic_construction(self):
        n = Node(id="P001", name="Neural_Signaling", type="Pillar", description="desc")
        assert n.id == "P001"
        assert n.name == "Neural_Signaling"

    def test_isolated_defaults_to_false(self):
        n = Node(id="x", name="X", type="Pillar", description="d")
        assert n.isolated is False

    def test_isolated_set_at_instantiation(self):
        n = Node(id="x", name="X", type="Pillar", description="d", isolated=True)
        assert n.isolated is True

    def test_from_row_loads_source_column(self):
        row = {"id": "P001", "name": "N", "type": "Pillar", "description": "d", "source": "v2"}
        n = Node.from_row(row)
        assert n.id == "P001"
        assert n.source == "v2"

    def test_from_row_source_defaults_to_empty_when_absent(self):
        row = {"id": "P001", "name": "N", "type": "Pillar", "description": "d"}
        n = Node.from_row(row)
        assert n.source == ""

    def test_frozen_prevents_mutation(self):
        n = Node(id="x", name="X", type="Pillar", description="d")
        with pytest.raises((AttributeError, TypeError)):
            n.name = "Y"  # type: ignore[misc]

    def test_new_node_type_accepted(self):
        # RelationshipType / NodeType are now plain str — arbitrary values load cleanly
        n = Node(id="x", name="X", type="Novel_Type", description="d")
        assert n.type == "Novel_Type"


class TestEdge:
    def test_basic_construction(self):
        e = Edge(
            from_id="A", to_id="B",
            relationship_type="RELATED_TO",
            confidence=0.85,
            map_source="semantic",
            notes="test note",
        )
        assert e.from_id == "A"
        assert e.confidence == 0.85

    def test_from_row_parses_confidence_as_float(self):
        row = {
            "from_id": "A", "to_id": "B",
            "relationship_type": "RELATED_TO",
            "confidence": "0.85",
            "map_source": "canonical",
            "notes": "",
        }
        e = Edge.from_row(row)
        assert isinstance(e.confidence, float)
        assert e.confidence == 0.85

    def test_new_relationship_type_accepted(self):
        e = Edge(from_id="A", to_id="B", relationship_type="NOVEL_TYPE",
                 confidence=1.0, map_source="canonical", notes="")
        assert e.relationship_type == "NOVEL_TYPE"


class TestPaperChunk:
    def _chunk(self, **kwargs):
        defaults = dict(
            id=1, doi="10.1/x", title="Title", authors="Smith, J;Jones, B",
            year=2020, chunk_index=0, text="Some text.", taxonomy_node_ids=(),
        )
        return PaperChunk(**{**defaults, **kwargs})

    def test_citation_format(self):
        assert self._chunk().citation == "Smith et al., 2020"

    def test_first_author_surname_single_author(self):
        c = self._chunk(authors="Catterall, W")
        assert c.first_author_surname == "Catterall"

    def test_first_author_surname_multiple_authors(self):
        c = self._chunk(authors="Turrigiano, G;Nelson, S;Abbott, L")
        assert c.first_author_surname == "Turrigiano"

    def test_taxonomy_node_ids_stored_as_tuple(self):
        c = self._chunk(taxonomy_node_ids=("RA001", "RA002"))
        assert isinstance(c.taxonomy_node_ids, tuple)
        assert "RA001" in c.taxonomy_node_ids
