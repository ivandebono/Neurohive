from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


RelationshipType = str
MapSource = str


@dataclass(frozen=True)
class Edge:
    """A directed relationship between two taxonomy nodes."""

    from_id: str
    to_id: str
    relationship_type: RelationshipType
    confidence: float  # 0.0-1.0
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
