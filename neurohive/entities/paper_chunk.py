from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


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
