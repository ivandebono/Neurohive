"""SQLite-backed store for nodes, edges, and paper chunks with graph/retrieval helpers.

Raw source files live in ``data/raw/``.  The database is written to
``data/database/neurohive.db`` and auto-populated on first run.  Delete the
``.db`` file to force a rebuild from the raw files.

Full-table reads (``nodes``, ``edges``, ``chunks``) are cached once per session
because the Retriever needs them at startup to build BM25 and dense indices.
Per-query lookups (adjacency, breadcrumb walks, chunk fetches) go through
indexed SQL queries.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

from neurohive.models import Edge, Node, PaperChunk

_BIDIRECTIONAL_EDGE_TYPES = frozenset({"RELATED_TO"})
_HIERARCHY_EDGE_TYPES = frozenset({"HAS_SUBPILLAR", "HAS_RESEARCH_AREA"})
_KNOWN_RELATIONSHIP_TYPES = (
    _HIERARCHY_EDGE_TYPES
    | _BIDIRECTIONAL_EDGE_TYPES
    | frozenset({"HAS_DIMENSION", "HAS_THEORY", "EXPLAINS", "IS_DERIVED_FROM"})
)

_SCHEMA = """
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    type        TEXT NOT NULL,
    description TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS edges (
    id                INTEGER PRIMARY KEY,
    from_id           TEXT NOT NULL,
    to_id             TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence        REAL NOT NULL,
    map_source        TEXT NOT NULL,
    notes             TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_id);
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    doi         TEXT NOT NULL,
    title       TEXT NOT NULL,
    authors     TEXT NOT NULL,
    year        INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk_nodes (
    chunk_id INTEGER NOT NULL,
    node_id  TEXT    NOT NULL,
    PRIMARY KEY (chunk_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_chunk_nodes_node ON chunk_nodes(node_id);
"""


class KnowledgeBase:
    """
    SQLite-backed store for nodes, edges, and paper chunks.

    Full-table reads (``nodes``, ``edges``, ``chunks``) are cached for the
    lifetime of the instance — they are needed once at startup to build BM25
    and dense retrieval indices.  All other access (node/edge lookups,
    graph expansion, chunk fetches) runs as indexed SQL queries.

    Parameters
    ----------
    data_dir : Project data directory.  Raw source files are read from
               ``data_dir/raw/``; the SQLite database is written to
               ``data_dir/database/neurohive.db``.
    verbose  : Print step-by-step progress during ingestion.
    rebuild  : Delete and recreate the database even if it already exists.
    """

    def __init__(self, data_dir: Path | str, *, verbose: bool = False, rebuild: bool = False):
        data_dir = Path(data_dir)
        self._raw_dir = data_dir / "raw"
        db_path = data_dir / "database" / "neurohive.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        if rebuild and db_path.exists():
            db_path.unlink()

        self.cache_dir: Path = db_path.parent  # shared location for generated files

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

        if self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0:
            self._ingest(verbose=verbose)

        # Session caches for full-table reads (populated on first access)
        self._nodes_cache: list[Node] | None = None
        self._edges_cache: list[Edge] | None = None
        self._chunks_cache: list[PaperChunk] | None = None

        # Dict-like proxies that match the old attribute interface
        self.nodes_by_id = _NodeIndex(self)
        self.outgoing    = _EdgeIndex(self._conn, self._to_edge, "from_id")
        self.incoming    = _EdgeIndex(self._conn, self._to_edge, "to_id")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup(items: list, content_key, label: str) -> list:
        """
        Return a deduplicated copy of *items*, warning about any entry whose
        content (all fields except id) matches a previously seen entry.
        The first occurrence is kept; subsequent duplicates are dropped.
        """
        seen: dict = {}
        result = []
        for item in items:
            key = content_key(item)
            if key in seen:
                warnings.warn(
                    f"Duplicate {label} content skipped during ingestion: "
                    f"{item!r} duplicates {seen[key]!r}",
                    stacklevel=4,
                )
            else:
                seen[key] = item
                result.append(item)
        return result

    @staticmethod
    def _step(label: str, count: int | None = None) -> None:
        """Print a single completed ingestion step."""
        suffix = f"  {count}" if count is not None else ""
        print(f"  \033[32m✓\033[0m  {label}{suffix}", flush=True)

    def _ingest(self, verbose: bool = False) -> None:
        """Populate the database from the raw source files."""
        if verbose:
            print("Building knowledge base from source files …")

        nodes  = Node.load_all(self._raw_dir / "taxonomy_nodes.csv")
        nodes  = self._dedup(nodes,  lambda n: (n.name, n.type, n.description),                                                "node")
        if verbose:
            self._step(f"taxonomy_nodes.csv", len(nodes))

        edges  = Edge.load_all(self._raw_dir / "taxonomy_edges.csv")
        edges  = self._dedup(edges,  lambda e: (e.from_id, e.to_id, e.relationship_type, e.confidence, e.map_source, e.notes), "edge")
        if verbose:
            self._step(f"taxonomy_edges.csv", len(edges))

        chunks = PaperChunk.load_all(self._raw_dir / "paper_chunks.json")
        chunks = self._dedup(chunks, lambda c: (c.doi, c.title, c.authors, c.year, c.chunk_index, c.text),                     "chunk")
        if verbose:
            self._step(f"paper_chunks.json ", len(chunks))

        if verbose:
            print("  Writing to database …", end=" ", flush=True)

        with self._conn:
            self._conn.executemany(
                "INSERT INTO nodes(id, name, type, description, source) VALUES(?,?,?,?,?)",
                [(n.id, n.name, n.type, n.description, n.source) for n in nodes],
            )
            self._conn.executemany(
                "INSERT INTO edges(from_id, to_id, relationship_type, confidence, map_source, notes) "
                "VALUES(?,?,?,?,?,?)",
                [(e.from_id, e.to_id, e.relationship_type, e.confidence, e.map_source, e.notes)
                 for e in edges],
            )
            self._conn.executemany(
                "INSERT INTO chunks(id, doi, title, authors, year, chunk_index, text) "
                "VALUES(?,?,?,?,?,?,?)",
                [(c.id, c.doi, c.title, c.authors, c.year, c.chunk_index, c.text)
                 for c in chunks],
            )
            self._conn.executemany(
                "INSERT OR IGNORE INTO chunk_nodes(chunk_id, node_id) VALUES(?,?)",
                [(c.id, nid) for c in chunks for nid in c.taxonomy_node_ids],
            )

        if verbose:
            print("done  \033[32m✓\033[0m")
            print()

    # ------------------------------------------------------------------
    # Row → model converters
    # ------------------------------------------------------------------

    @staticmethod
    def _to_node(row: sqlite3.Row, isolated: bool = False) -> Node:
        return Node(id=row["id"], name=row["name"], type=row["type"],
                    description=row["description"], source=row["source"],
                    isolated=isolated)

    @staticmethod
    def _to_edge(row: sqlite3.Row) -> Edge:
        return Edge(
            from_id=row["from_id"], to_id=row["to_id"],
            relationship_type=row["relationship_type"],
            confidence=row["confidence"], map_source=row["map_source"],
            notes=row["notes"],
        )

    # ------------------------------------------------------------------
    # Full-table reads (cached — needed by Retriever at startup)
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        if self._nodes_cache is None:
            # Single query: left-join edges to detect nodes with no connections.
            rows = self._conn.execute(
                "SELECT n.*, "
                "  (COUNT(e.id) = 0) AS isolated "
                "FROM nodes n "
                "LEFT JOIN edges e ON e.from_id = n.id OR e.to_id = n.id "
                "GROUP BY n.id"
            ).fetchall()
            self._nodes_cache = [self._to_node(r, isolated=bool(r["isolated"])) for r in rows]
            self.nodes_by_id._warm({n.id: n for n in self._nodes_cache})
        return self._nodes_cache

    @property
    def edges(self) -> list[Edge]:
        if self._edges_cache is None:
            self._edges_cache = [
                self._to_edge(r)
                for r in self._conn.execute("SELECT * FROM edges").fetchall()
            ]
        return self._edges_cache

    @property
    def chunks(self) -> list[PaperChunk]:
        if self._chunks_cache is None:
            rows = self._conn.execute("SELECT * FROM chunks ORDER BY id").fetchall()
            cn_map: dict[int, list[str]] = {}
            for r in self._conn.execute("SELECT chunk_id, node_id FROM chunk_nodes").fetchall():
                cn_map.setdefault(r["chunk_id"], []).append(r["node_id"])
            self._chunks_cache = [
                PaperChunk(
                    id=r["id"], doi=r["doi"], title=r["title"], authors=r["authors"],
                    year=r["year"], chunk_index=r["chunk_index"], text=r["text"],
                    taxonomy_node_ids=tuple(cn_map.get(r["id"], [])),
                )
                for r in rows
            ]
        return self._chunks_cache

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def ancestors(self, node_id: str) -> list[Node]:
        """Walk up HAS_SUBPILLAR / HAS_RESEARCH_AREA edges to the root."""
        result: list[Node] = []
        current = node_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            row = self._conn.execute(
                "SELECT e.from_id, n.id, n.name, n.type, n.description "
                "FROM edges e JOIN nodes n ON n.id = e.from_id "
                "WHERE e.to_id = ? "
                "  AND e.relationship_type IN ('HAS_SUBPILLAR', 'HAS_RESEARCH_AREA') "
                "LIMIT 1",
                (current,),
            ).fetchone()
            if row is None:
                break
            result.append(self.nodes_by_id[row["from_id"]])
            current = row["from_id"]
        return result

    def breadcrumb(self, node_id: str) -> str:
        """Return a human-readable path, e.g. 'Neural Signaling → Action Potential'."""
        row = self._conn.execute("SELECT name FROM nodes WHERE id = ?", (node_id,)).fetchone()
        if not row:
            return node_id
        parts = [a.name.replace("_", " ") for a in reversed(self.ancestors(node_id))]
        parts.append(row["name"].replace("_", " "))
        return " → ".join(parts)

    def smart_expand(
        self,
        node_ids: set[str],
        confidence_threshold: float = 0.8,
    ) -> set[str]:
        """
        Expand a seed set of node IDs with semantically useful neighbours.

        Uses two bulk SQL queries (one for incoming edges, one for outgoing)
        across the entire seed set, then applies the same traversal rules as
        before.  Unknown relationship types are followed in both directions.

        Only edges with ``confidence >= confidence_threshold`` are considered;
        semantic edges (``map_source = 'semantic'``) require an additional +0.1.
        """
        expanded = set(node_ids)
        if not node_ids:
            return expanded

        ph = ",".join("?" * len(node_ids))
        ids = tuple(node_ids)

        # Incoming edges — candidate is the from_id node
        rows = self._conn.execute(
            f"SELECT e.*, n.type AS ctype "
            f"FROM edges e JOIN nodes n ON n.id = e.from_id "
            f"WHERE e.to_id IN ({ph}) AND e.confidence >= ?",
            (*ids, confidence_threshold),
        ).fetchall()
        for row in rows:
            threshold = (
                min(confidence_threshold + 0.1, 1.0)
                if row["map_source"] == "semantic"
                else confidence_threshold
            )
            if row["confidence"] < threshold:
                continue
            rt, ctype, from_id = row["relationship_type"], row["ctype"], row["from_id"]
            if rt in _HIERARCHY_EDGE_TYPES and ctype in ("Pillar", "Subpillar"):
                expanded.add(from_id)
            elif rt == "EXPLAINS" and ctype == "Theory":
                expanded.add(from_id)
            elif rt == "HAS_THEORY" and ctype == "Pillar":
                expanded.add(from_id)
            elif rt == "IS_DERIVED_FROM" and ctype == "Research_area":
                expanded.add(from_id)
            elif rt in _BIDIRECTIONAL_EDGE_TYPES:
                expanded.add(from_id)
            elif rt not in _KNOWN_RELATIONSHIP_TYPES:
                expanded.add(from_id)

        # Outgoing edges — candidate is the to_id node
        rows = self._conn.execute(
            f"SELECT e.*, n.type AS ctype "
            f"FROM edges e JOIN nodes n ON n.id = e.to_id "
            f"WHERE e.from_id IN ({ph}) AND e.confidence >= ?",
            (*ids, confidence_threshold),
        ).fetchall()
        for row in rows:
            threshold = (
                min(confidence_threshold + 0.1, 1.0)
                if row["map_source"] == "semantic"
                else confidence_threshold
            )
            if row["confidence"] < threshold:
                continue
            rt, ctype, to_id = row["relationship_type"], row["ctype"], row["to_id"]
            if rt == "HAS_THEORY" and ctype == "Theory":
                expanded.add(to_id)
            elif rt == "HAS_DIMENSION" and ctype == "Dimension":
                expanded.add(to_id)
            elif rt in _BIDIRECTIONAL_EDGE_TYPES:
                expanded.add(to_id)
            elif rt == "IS_DERIVED_FROM" and ctype == "Research_area":
                expanded.add(to_id)
            elif rt not in _KNOWN_RELATIONSHIP_TYPES:
                expanded.add(to_id)

        return expanded

    def chunks_for_nodes(self, node_ids: set[str]) -> list[PaperChunk]:
        """Return deduplicated chunks linked to any of the given node IDs."""
        if not node_ids:
            return []
        ph = ",".join("?" * len(node_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT c.* FROM chunks c "
            f"JOIN chunk_nodes cn ON cn.chunk_id = c.id "
            f"WHERE cn.node_id IN ({ph})",
            tuple(node_ids),
        ).fetchall()
        if not rows:
            return []
        chunk_ids = tuple(r["id"] for r in rows)
        cn_ph = ",".join("?" * len(chunk_ids))
        cn_map: dict[int, list[str]] = {}
        for r in self._conn.execute(
            f"SELECT chunk_id, node_id FROM chunk_nodes WHERE chunk_id IN ({cn_ph})",
            chunk_ids,
        ).fetchall():
            cn_map.setdefault(r["chunk_id"], []).append(r["node_id"])
        return [
            PaperChunk(
                id=r["id"], doi=r["doi"], title=r["title"], authors=r["authors"],
                year=r["year"], chunk_index=r["chunk_index"], text=r["text"],
                taxonomy_node_ids=tuple(cn_map.get(r["id"], [])),
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Dict-like proxies (preserve call-site compatibility)
# ---------------------------------------------------------------------------

class _NodeIndex:
    """
    Dict-like proxy for node lookups.

    Backed by a dict cache that is populated the first time ``kb.nodes`` is
    accessed (which happens at Retriever startup).  Subsequent lookups are O(1).
    Falls back to a SQL query for any node ID requested before the cache is warm.
    """

    def __init__(self, kb: KnowledgeBase):
        self._kb = kb
        self._cache: dict[str, Node] = {}

    def _warm(self, cache: dict[str, Node]) -> None:
        self._cache = cache

    def __getitem__(self, key: str) -> Node:
        if key in self._cache:
            return self._cache[key]
        row = self._kb._conn.execute(
            "SELECT n.*, (COUNT(e.id) = 0) AS isolated "
            "FROM nodes n "
            "LEFT JOIN edges e ON e.from_id = n.id OR e.to_id = n.id "
            "WHERE n.id = ? GROUP BY n.id",
            (key,),
        ).fetchone()
        if row is None:
            raise KeyError(key)
        node = KnowledgeBase._to_node(row, isolated=bool(row["isolated"]))
        self._cache[key] = node
        return node

    def __contains__(self, key: str) -> bool:
        if key in self._cache:
            return True
        return self._kb._conn.execute(
            "SELECT 1 FROM nodes WHERE id = ?", (key,)
        ).fetchone() is not None

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class _EdgeIndex:
    """Dict-like proxy for adjacency lookups (outgoing or incoming)."""

    def __init__(self, conn: sqlite3.Connection, convert, col: str):
        self._conn = conn
        self._convert = convert
        self._col = col

    def __getitem__(self, key: str) -> list[Edge]:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def get(self, key: str, default=None):
        rows = self._conn.execute(
            f"SELECT * FROM edges WHERE {self._col} = ?", (key,)
        ).fetchall()
        return [self._convert(r) for r in rows] if rows else default
