"""
Incremental ingestion: add paper chunks, nodes, and edges to the live database
without a full rebuild from raw source files.

After each ingestion, the retrieval cache files (BM25 and dense embeddings) are
invalidated automatically. Run ``python main.py --ingest`` (or instantiate a
new ``Retriever``) to rebuild indices from the updated database.

Usage
-----
    from neurohive.knowledge_base import KnowledgeBase
    from neurohive.ingestor import IncrementalIngestor
    from neurohive.entities import PaperChunk

    kb = KnowledgeBase("data")
    ingestor = IncrementalIngestor(kb)

    new_chunks = PaperChunk.load_all("my_new_papers.json")
    result = ingestor.add_chunks(new_chunks, source="my_new_papers.json")
    print(result)
"""
from __future__ import annotations

import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from neurohive.entities import Edge, Node, PaperChunk

if TYPE_CHECKING:
    from neurohive.knowledge_base import KnowledgeBase


def _cache_files_from_config(config: dict) -> tuple[str, ...]:
    """Return cache filenames read from the [retrieval] config section."""
    r = config.get("retrieval", {})
    return (
        r.get("bm25_meta_file",  "bm25_meta.json"),
        r.get("node_bm25_file",  "node_bm25.json"),
        r.get("chunk_bm25_file", "chunk_bm25.json"),
        r.get("emb_meta_file",   "emb_meta.json"),
        r.get("node_emb_file",   "node_embs.npy"),
        r.get("chunk_emb_file",  "chunk_embs.npy"),
    )


@dataclass
class IngestResult:
    """Summary of a single incremental ingestion batch."""

    batch_id: str
    added_nodes: int = 0
    added_edges: int = 0
    added_chunks: int = 0
    skipped_nodes: int = 0
    skipped_edges: int = 0
    skipped_chunks: int = 0
    cache_invalidated: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def any_added(self) -> bool:
        return bool(self.added_nodes or self.added_edges or self.added_chunks)

    def __str__(self) -> str:
        lines = [f"Batch: {self.batch_id}"]
        lines.append(f"  Nodes:  +{self.added_nodes} added, {self.skipped_nodes} skipped")
        lines.append(f"  Edges:  +{self.added_edges} added, {self.skipped_edges} skipped")
        lines.append(f"  Chunks: +{self.added_chunks} added, {self.skipped_chunks} skipped")
        if self.cache_invalidated:
            lines.append(
                "  Cache:  invalidated — run 'python main.py --ingest' to rebuild indices"
            )
        for w in self.warnings:
            lines.append(f"  ⚠  {w}")
        return "\n".join(lines)


class IncrementalIngestor:
    """
    Adds new records to the knowledge base without a full rebuild.

    Duplicate detection is based on natural keys:
      - Nodes:  ``id``
      - Edges:  ``(from_id, to_id, relationship_type)``
      - Chunks: ``(doi, chunk_index)``

    After any successful insert the retrieval caches are deleted so the next
    ``Retriever`` instantiation (or ``--ingest`` run) will rebuild them.

    Parameters
    ----------
    kb        : Live KnowledgeBase instance.
    cache_dir : Directory holding BM25 / embedding cache files.
                Defaults to ``kb.cache_dir``.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        cache_dir: Path | None = None,
        config: dict | None = None,
    ):
        self._kb = kb
        self._conn: sqlite3.Connection = kb._conn
        self._cache_dir = cache_dir or kb.cache_dir
        if config is None:
            from neurohive.config import load_config  # noqa: PLC0415
            config = load_config()
        self._config = config
        self._cache_files: tuple[str, ...] = _cache_files_from_config(config)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_id() -> str:
        return "batch_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def _invalidate_caches(self) -> bool:
        """Delete on-disk cache files and clear the KB's in-memory caches."""
        deleted = False
        for name in self._cache_files:
            p = self._cache_dir / name
            if p.exists():
                p.unlink()
                deleted = True
        self._kb._nodes_cache = None
        self._kb._edges_cache = None
        self._kb._chunks_cache = None
        return deleted

    def _semantic_dedup(
        self,
        chunks: list[PaperChunk],
        threshold: float,
    ) -> list[PaperChunk]:
        """
        Remove near-duplicate chunks using cosine similarity of embeddings.

        Compares each candidate against the existing corpus embeddings AND
        against previously accepted candidates in this batch. Requires the
        embedding model and ``chunk_embs.npy`` to be present; silently skips
        if either is missing or sentence-transformers is not installed.
        """
        try:
            import numpy as np                                # noqa: PLC0415
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            return chunks

        cfg = self._config
        repo_root = Path(__file__).parent.parent
        models_dir = Path(cfg["paths"]["models_dir"])
        if not models_dir.is_absolute():
            models_dir = repo_root / models_dir
        model_name = cfg["embeddings"]["model"].split("/")[-1]
        model_path = models_dir / model_name
        emb_file   = self._cache_dir / cfg["retrieval"]["chunk_emb_file"]

        if not model_path.exists() or not emb_file.exists():
            return chunks

        existing_embs = np.load(str(emb_file))  # (N, D), L2-normalised
        model = SentenceTransformer(str(model_path))
        new_embs = model.encode(
            [c.text for c in chunks], normalize_embeddings=True
        )  # (M, D)

        accepted: list[PaperChunk] = []
        # Build a comparison matrix that grows as we accept new chunks.
        comparison = list(existing_embs)

        for i, chunk in enumerate(chunks):
            if comparison:
                sims = np.array(comparison) @ new_embs[i]
                max_sim = float(sims.max())
            else:
                max_sim = 0.0

            if max_sim >= threshold:
                msg = (
                    f"Semantic duplicate skipped: doi={chunk.doi!r} "
                    f"index={chunk.chunk_index} "
                    f"(max cosine similarity {max_sim:.3f} ≥ {threshold})"
                )
                warnings.warn(msg, stacklevel=3)
            else:
                accepted.append(chunk)
                comparison.append(new_embs[i])

        return accepted

    def _record_batch(self, batch_id: str, source: str, result: IngestResult) -> None:
        if not result.any_added:
            return
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO batches"
                "(batch_id, source, n_nodes, n_edges, n_chunks) VALUES(?,?,?,?,?)",
                (batch_id, source, result.added_nodes, result.added_edges, result.added_chunks),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[PaperChunk], source: str = "") -> IngestResult:
        """
        Add paper chunks to the knowledge base.

        Skips chunks whose ``(doi, chunk_index)`` already exists in the database.
        """
        bid = self._batch_id()
        result = IngestResult(batch_id=bid)
        now = datetime.now(timezone.utc).isoformat()

        existing: set[tuple[str, int]] = {
            (r[0], r[1])
            for r in self._conn.execute("SELECT doi, chunk_index FROM chunks").fetchall()
        }

        to_insert: list[PaperChunk] = []
        for chunk in chunks:
            key = (chunk.doi, chunk.chunk_index)
            if key in existing:
                result.skipped_chunks += 1
                msg = f"Duplicate chunk skipped: doi={chunk.doi!r} index={chunk.chunk_index}"
                result.warnings.append(msg)
                warnings.warn(msg, stacklevel=2)
            else:
                existing.add(key)
                to_insert.append(chunk)

        # Semantic deduplication (requires embedding model + existing embeddings).
        icfg = self._config.get("ingestor", {})
        if to_insert and icfg.get("semantic_dedup_enabled", True):
            before = len(to_insert)
            to_insert = self._semantic_dedup(
                to_insert, float(icfg.get("semantic_dedup_threshold", 0.92))
            )
            result.skipped_chunks += before - len(to_insert)

        if to_insert:
            max_id: int = self._conn.execute("SELECT COALESCE(MAX(id),0) FROM chunks").fetchone()[0]
            rows = []
            cn_rows = []
            for i, c in enumerate(to_insert):
                cid = max_id + i + 1
                rows.append((cid, c.doi, c.title, c.authors, c.year,
                              c.chunk_index, c.text, now, bid))
                for nid in c.taxonomy_node_ids:
                    cn_rows.append((cid, nid))

            with self._conn:
                self._conn.executemany(
                    "INSERT INTO chunks"
                    "(id, doi, title, authors, year, chunk_index, text, created_at, batch_id)"
                    " VALUES(?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                self._conn.executemany(
                    "INSERT OR IGNORE INTO chunk_nodes(chunk_id, node_id) VALUES(?,?)",
                    cn_rows,
                )

            result.added_chunks = len(to_insert)
            result.cache_invalidated = self._invalidate_caches()

        self._record_batch(bid, source, result)
        return result

    def add_nodes(self, nodes: list[Node], source: str = "") -> IngestResult:
        """
        Add taxonomy nodes to the knowledge base.

        Skips nodes whose ``id`` already exists.
        """
        bid = self._batch_id()
        result = IngestResult(batch_id=bid)
        now = datetime.now(timezone.utc).isoformat()

        existing_ids: set[str] = {
            r[0] for r in self._conn.execute("SELECT id FROM nodes").fetchall()
        }

        to_insert: list[Node] = []
        for node in nodes:
            if node.id in existing_ids:
                result.skipped_nodes += 1
                msg = f"Duplicate node skipped: id={node.id!r}"
                result.warnings.append(msg)
                warnings.warn(msg, stacklevel=2)
            else:
                existing_ids.add(node.id)
                to_insert.append(node)

        if to_insert:
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO nodes"
                    "(id, name, type, description, source, created_at, batch_id)"
                    " VALUES(?,?,?,?,?,?,?)",
                    [(n.id, n.name, n.type, n.description, n.source, now, bid)
                     for n in to_insert],
                )
            result.added_nodes = len(to_insert)
            result.cache_invalidated = self._invalidate_caches()

        self._record_batch(bid, source, result)
        return result

    def add_edges(self, edges: list[Edge], source: str = "") -> IngestResult:
        """
        Add taxonomy edges to the knowledge base.

        Skips edges whose ``(from_id, to_id, relationship_type)`` already exists.
        """
        bid = self._batch_id()
        result = IngestResult(batch_id=bid)
        now = datetime.now(timezone.utc).isoformat()

        existing: set[tuple[str, str, str]] = {
            (r[0], r[1], r[2])
            for r in self._conn.execute(
                "SELECT from_id, to_id, relationship_type FROM edges"
            ).fetchall()
        }

        to_insert: list[Edge] = []
        for edge in edges:
            key = (edge.from_id, edge.to_id, edge.relationship_type)
            if key in existing:
                result.skipped_edges += 1
                msg = (
                    f"Duplicate edge skipped: {edge.from_id!r} → {edge.to_id!r}"
                    f" ({edge.relationship_type!r})"
                )
                result.warnings.append(msg)
                warnings.warn(msg, stacklevel=2)
            else:
                existing.add(key)
                to_insert.append(edge)

        if to_insert:
            with self._conn:
                self._conn.executemany(
                    "INSERT INTO edges"
                    "(from_id, to_id, relationship_type, confidence,"
                    " map_source, notes, created_at, batch_id)"
                    " VALUES(?,?,?,?,?,?,?,?)",
                    [(e.from_id, e.to_id, e.relationship_type, e.confidence,
                      e.map_source, e.notes, now, bid)
                     for e in to_insert],
                )
            result.added_edges = len(to_insert)
            result.cache_invalidated = self._invalidate_caches()

        self._record_batch(bid, source, result)
        return result

    def add_from_file(self, path: Path | str, source: str = "") -> IngestResult:
        """
        Detect file type by extension and ingest accordingly.

        - ``*.json`` → paper chunks (schema: ``data/raw/paper_chunks.json``)
        - ``*.csv``  → taxonomy nodes (schema: ``data/raw/taxonomy_nodes.csv``)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        source = source or str(path)
        ext = path.suffix.lower()
        if ext == ".json":
            return self.add_chunks(PaperChunk.load_all(path), source=source)
        if ext == ".csv":
            return self.add_nodes(Node.load_all(path), source=source)
        raise ValueError(
            f"Unsupported extension {path.suffix!r}. Use .json for chunks or .csv for nodes."
        )

    def list_batches(self) -> list[dict]:
        """Return all recorded ingestion batches, newest first."""
        rows = self._conn.execute(
            "SELECT batch_id, created_at, source, n_nodes, n_edges, n_chunks"
            " FROM batches ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
