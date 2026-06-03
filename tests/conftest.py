"""Shared fixtures for the Neurohive test suite."""

from __future__ import annotations

import csv
import json

import pytest

from neurohive.backends import GenerationBackend

# ---------------------------------------------------------------------------
# Minimal sample data
# ---------------------------------------------------------------------------

SAMPLE_NODES = [
    {"id": "P001", "name": "Neural_Signaling",     "type": "Pillar",        "description": "Top-level pillar for neural signalling", "source": "v1"},
    {"id": "SP001", "name": "Electrical_Signaling", "type": "Subpillar",     "description": "Electrical aspects of neural signalling", "source": "v1"},
    {"id": "RA001", "name": "Action_Potential",     "type": "Research_area", "description": "Spike generation and propagation",       "source": "v1"},
    {"id": "TH001", "name": "Hodgkin_Huxley_Model", "type": "Theory",        "description": "Conductance-based model of the AP",      "source": "v1"},
    {"id": "LONE",  "name": "Isolated_Concept",     "type": "Research_area", "description": "A concept with no edges at all",         "source": "v1"},
]

SAMPLE_EDGES = [
    {"from_id": "P001",  "to_id": "SP001", "relationship_type": "HAS_SUBPILLAR",    "confidence": "1.0", "map_source": "canonical", "notes": ""},
    {"from_id": "SP001", "to_id": "RA001", "relationship_type": "HAS_RESEARCH_AREA","confidence": "1.0", "map_source": "canonical", "notes": ""},
    {"from_id": "P001",  "to_id": "TH001", "relationship_type": "HAS_THEORY",       "confidence": "1.0", "map_source": "canonical", "notes": ""},
    {"from_id": "TH001", "to_id": "RA001", "relationship_type": "EXPLAINS",         "confidence": "0.9", "map_source": "semantic",  "notes": "HH model explains AP generation"},
]

SAMPLE_CHUNKS = [
    {
        "id": 1, "doi": "10.1/ap", "title": "Voltage-Gated Channels",
        "authors": "Smith, J;Jones, B", "year": 2020, "chunk_index": 0,
        "text": "Voltage-gated sodium channels initiate action potentials by opening rapidly.",
        "taxonomy_node_ids": ["RA001"],
    },
    {
        "id": 2, "doi": "10.1/ap", "title": "Voltage-Gated Channels",
        "authors": "Smith, J;Jones, B", "year": 2020, "chunk_index": 1,
        "text": "The Hodgkin-Huxley model describes channel kinetics mathematically.",
        "taxonomy_node_ids": ["TH001"],
    },
]


# ---------------------------------------------------------------------------
# Mock backend (no API calls)
# ---------------------------------------------------------------------------

class MockBackend(GenerationBackend):
    """Returns an empty answer; avoids any real API calls."""

    def generate(self, query: str, taxonomy_ctx: str, paper_ctx: str) -> dict:
        return {"answer": []}

    @property
    def model_id(self) -> str:
        return "mock-model"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def data_dir(tmp_path):
    """Temporary data directory populated with sample CSV / JSON files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (tmp_path / "database").mkdir()

    with open(raw_dir / "taxonomy_nodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "type", "description", "source"])
        w.writeheader()
        w.writerows(SAMPLE_NODES)

    with open(raw_dir / "taxonomy_edges.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["from_id", "to_id", "relationship_type", "confidence", "map_source", "notes"])
        w.writeheader()
        w.writerows(SAMPLE_EDGES)

    with open(raw_dir / "paper_chunks.json", "w") as f:
        json.dump(SAMPLE_CHUNKS, f)

    return tmp_path


@pytest.fixture
def kb(data_dir):
    from neurohive.knowledge_base import KnowledgeBase
    return KnowledgeBase(data_dir)


@pytest.fixture
def pipeline(data_dir):
    from neurohive.pipeline import QueryPipeline
    return QueryPipeline(data_dir=data_dir, backend=MockBackend())
