from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


NodeType = str
RelationshipType = str
MapSource = str


@dataclass(frozen=True)
class Node:
    """A concept node in the neuroscience taxonomy."""

    id: str
    name: str
    type: NodeType
    description: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Node:
        return cls(
            id=row["id"],
            name=row["name"],
            type=row["type"],  # type: ignore[arg-type]
            description=row["description"],
            # "source" column (always "v1") is intentionally not loaded.
        )

    @classmethod
    def load_all(cls, path: Path) -> list[Node]:
        with open(path, newline="") as f:
            return [cls.from_row(row) for row in csv.DictReader(f)]

    def isolated(self, edges: Iterable[Edge]) -> bool:
        """Return True if this node has at least one edge, False if it has none."""
        return any(e.from_id == self.id or e.to_id == self.id for e in edges)


@dataclass(frozen=True)
class Edge:
    """A directed relationship between two taxonomy nodes."""

    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: float  # 0.0–1.0
    map_source: MapSource
    notes: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Edge:
        return cls(
            from_id=row["from_id"],
            to_id=row["to_id"],
            relationship_type=row["relationship_type"],
            confidence=float(row["confidence"]),
            map_source=row["map_source"],
            notes=row["notes"],
        )

    @classmethod
    def load_all(cls, path: Path) -> list[Edge]:
        with open(path, newline="") as f:
            return [cls.from_row(row) for row in csv.DictReader(f)]


@dataclass(frozen=True)
class PaperChunk:
    """A text excerpt from an academic paper, linked to taxonomy nodes."""

    id: int
    doi: str
    title: str
    authors: str          # semicolon-separated author list
    year: int
    chunk_index: int      # zero-based position within the paper
    text: str
    taxonomy_node_ids: tuple[str, ...]  # node IDs from taxonomy_nodes.csv

    @classmethod
    def from_dict(cls, item: dict) -> PaperChunk:
        return cls(
            id=item["id"],
            doi=item["doi"],
            title=item["title"],
            authors=item["authors"],
            year=item["year"],
            chunk_index=item["chunk_index"],
            text=item["text"],
            taxonomy_node_ids=tuple(item["taxonomy_node_ids"]),
        )

    @classmethod
    def load_all(cls, path: Path) -> list[PaperChunk]:
        with open(path) as f:
            return [cls.from_dict(item) for item in json.load(f)]

    @property
    def first_author_surname(self) -> str:
        """Returns the surname of the first author."""
        return self.authors.split(";")[0].strip().split(",")[0].strip()

    @property
    def citation(self) -> str:
        """Short citation string, e.g. 'Catterall et al., 2017'."""
        return f"{self.first_author_surname} et al., {self.year}"
