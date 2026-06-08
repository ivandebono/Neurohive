"""SQLite-backed query result cache.

Cache keys cover the query text, generation model, retrieval settings, and a
corpus fingerprint derived from the live row counts. Adding new data to the
knowledge base changes the fingerprint and automatically invalidates stale
entries.

Usage
-----
    from neurohive.cache import QueryCache

    cache = QueryCache(db_path, ttl_seconds=86400)
    key = cache.make_key(query, model_id, node_top_k, chunk_top_k,
                         confidence_threshold, rerank, corpus_fp)
    hit = cache.get(key)
    if hit is None:
        result = pipeline.run(query)
        cache.put(key, result.as_cache_dict())
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key   TEXT    PRIMARY KEY,
    query       TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class QueryCache:
    """
    Persistent query result cache backed by a SQLite database.

    Parameters
    ----------
    db_path     : Path to the SQLite cache file (created if absent).
    ttl_seconds : Seconds before a cache entry is considered stale.
                  0 means entries never expire.
    """

    def __init__(self, db_path: Path, ttl_seconds: int = 86400):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(
        query: str,
        model_id: str,
        node_top_k: int,
        chunk_top_k: int,
        confidence_threshold: float,
        rerank: bool,
        corpus_fingerprint: str,
    ) -> str:
        """Return a deterministic hex digest that covers all inputs."""
        payload = json.dumps(
            {
                "q":    query.strip().lower(),
                "m":    model_id,
                "ntk":  node_top_k,
                "ctk":  chunk_top_k,
                "ct":   round(confidence_threshold, 6),
                "rr":   rerank,
                "fp":   corpus_fingerprint,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Read / write
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict | None:
        """Return the cached payload dict, or None on miss / expiry."""
        row = self._conn.execute(
            "SELECT payload, created_at FROM query_cache WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if self._ttl > 0:
            created = datetime.fromisoformat(row["created_at"])
            age = (datetime.now(timezone.utc) - created).total_seconds()
            if age > self._ttl:
                self._conn.execute(
                    "DELETE FROM query_cache WHERE cache_key = ?", (key,)
                )
                self._conn.commit()
                return None
        return json.loads(row["payload"])

    def put(self, key: str, query: str, model_id: str, payload: dict) -> None:
        """Insert or replace a cache entry."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO query_cache"
                "(cache_key, query, payload, model, created_at)"
                " VALUES(?,?,?,?,?)",
                (key, query, json.dumps(payload, ensure_ascii=False), model_id, now),
            )

    def invalidate(self, key: str) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM query_cache WHERE cache_key = ?", (key,)
            )

    def clear(self) -> int:
        """Delete all entries. Returns the number of rows removed."""
        cur = self._conn.execute("SELECT COUNT(*) FROM query_cache")
        count = cur.fetchone()[0]
        with self._conn:
            self._conn.execute("DELETE FROM query_cache")
        return count

    def size(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM query_cache"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Corpus fingerprint helper
    # ------------------------------------------------------------------

    @staticmethod
    def corpus_fingerprint(n_nodes: int, n_edges: int, n_chunks: int) -> str:
        """Short hash of row counts — changes whenever data is added or removed."""
        raw = f"{n_nodes}:{n_edges}:{n_chunks}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
