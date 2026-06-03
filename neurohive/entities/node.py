from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


NodeType = str


@dataclass(frozen=True)
class Node:
    """A concept node in the neuroscience taxonomy."""

    id: str
    name: str
    type: NodeType
    description: str
    source: str = ""
    isolated: bool = False  # True when the node has no edges in the graph

    @classmethod
    def from_row(cls, row: dict[str, str]) -> Node:
        return cls(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            description=row["description"],
            source=row.get("source", ""),
        )

    @classmethod
    def load_all(cls, path: Path) -> list[Node]:
        with open(path, newline="") as f:
            return [cls.from_row(row) for row in csv.DictReader(f)]
