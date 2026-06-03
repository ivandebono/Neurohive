"""
BM25 and hybrid (BM25 + dense) retrieval over taxonomy nodes and paper chunks.

At initialisation the Retriever checks for the configured sentence-transformers
model under the configured models directory. If the directory exists and
sentence-transformers is installed, corpus embeddings are loaded or computed
and hybrid retrieval via Reciprocal Rank Fusion is used. Otherwise pure BM25
Okapi is used. The public API is identical either way.

BM25 Okapi
----------
    score(q, d) = Σ_t  IDF(t) · f(t,d)·(k₁+1)
                                ─────────────────────────────────────
                                f(t,d) + k₁·(1 − b + b·|d|/avgdl)

    k1 and b are configured in config.toml.

Reciprocal Rank Fusion
----------------------
    rrf(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))

    RRF k values are configured separately for nodes and chunks in config.toml.

Sibling boost (chunk retrieval only)
-------------------------------------
After the initial ranking, every chunk that shares a DOI with any top-k seed
is guaranteed a configured score floor based on the best seed score for that DOI.
This ensures that if one passage from a paper is relevant, the paper's other
passages are also surfaced, providing fuller context from the same source.
The full corpus is re-ranked after boosting and top-k is then returned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from heapq import nlargest
from pathlib import Path
from typing import Any

from neurohive.knowledge_base import KnowledgeBase
from neurohive.entities import Node, PaperChunk

def _default_model_dir() -> Path:
    """
    Derive the default embedding model directory from config.toml.
    """
    from neurohive.config import load_config  # noqa: PLC0415
    cfg = load_config()
    repo_root = Path(__file__).parent.parent
    models_dir = Path(cfg["paths"]["models_dir"])
    if not models_dir.is_absolute():
        models_dir = repo_root / models_dir
    model_name = cfg["embeddings"]["model"].split("/")[-1]
    return models_dir / model_name


def _retrieval_config() -> dict:
    """Return retrieval settings from config.toml with config.py fallbacks."""
    from neurohive.config import load_config  # noqa: PLC0415
    return load_config()["retrieval"]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

@dataclass
class _BM25Index:
    """Minimal BM25 Okapi index over a flat list of string documents."""

    _tf: list[dict[str, int]]
    _df: dict[str, int]
    _dl: list[int]
    _idf: dict[str, float]
    _postings: dict[str, list[tuple[int, int]]]
    _avgdl: float
    _N: int
    _k1: float
    _b: float

    @classmethod
    def build(cls, docs: list[str], *, k1: float, b: float) -> "_BM25Index":
        tokenized = [_tokenize(d) for d in docs]
        N = len(tokenized)
        dl = [len(t) for t in tokenized]
        avgdl = sum(dl) / max(N, 1)
        df: dict[str, int] = {}
        tf_list: list[dict[str, int]] = []
        for tokens in tokenized:
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            tf_list.append(tf)
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        idf = {
            tok: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, tf in enumerate(tf_list):
            for tok, freq in tf.items():
                postings.setdefault(tok, []).append((i, freq))
        return cls(
            _tf=tf_list,
            _df=df,
            _dl=dl,
            _idf=idf,
            _postings=postings,
            _avgdl=avgdl,
            _N=N,
            _k1=k1,
            _b=b,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the index."""
        return {
            "tf": self._tf,
            "df": self._df,
            "dl": self._dl,
            "idf": self._idf,
            "postings": self._postings,
            "avgdl": self._avgdl,
            "N": self._N,
            "k1": self._k1,
            "b": self._b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, k1: float, b: float) -> "_BM25Index":
        """Rehydrate an index saved by ``to_dict``."""
        return cls(
            _tf=[{str(k): int(v) for k, v in tf.items()} for tf in data["tf"]],
            _df={str(k): int(v) for k, v in data["df"].items()},
            _dl=[int(v) for v in data["dl"]],
            _idf={str(k): float(v) for k, v in data["idf"].items()},
            _postings={
                str(tok): [(int(i), int(freq)) for i, freq in postings]
                for tok, postings in data["postings"].items()
            },
            _avgdl=float(data["avgdl"]),
            _N=int(data["N"]),
            _k1=k1,
            _b=b,
        )

    def score(self, query: str) -> list[float]:
        scores = [0.0] * self._N
        if self._N == 0:
            return scores
        for tok in dict.fromkeys(_tokenize(query)):
            postings = self._postings.get(tok)
            if not postings:
                continue
            idf = self._idf[tok]
            for i, f in postings:
                denom = f + self._k1 * (1 - self._b + self._b * self._dl[i] / max(self._avgdl, 1))
                scores[i] += idf * f * (self._k1 + 1) / denom if denom else 0
        return scores


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    BM25 retriever with optional dense-model hybrid upgrade via RRF.

    If the configured local embedding model exists and sentence-transformers is
    installed, corpus embeddings are loaded or computed and hybrid retrieval is
    used for every subsequent query. Otherwise pure BM25 is used. Check
    ``retriever.is_hybrid`` to see which mode is active.

    Parameters
    ----------
    kb        : Loaded KnowledgeBase instance.
    model_dir : Override the default model directory (useful for testing).
    """

    def __init__(self, kb: KnowledgeBase, model_dir: Path | str | None = None,
                 cache_dir: Path | str | None = None) -> None:
        self.kb = kb
        self._cache_dir = Path(cache_dir) if cache_dir else None
        cfg = _retrieval_config()
        self._bm25_k1 = float(cfg["bm25_k1"])
        self._bm25_b = float(cfg["bm25_b"])
        self._node_rrf_k = int(cfg["node_rrf_k"])
        self._chunk_rrf_k = int(cfg["chunk_rrf_k"])
        self._sibling_discount = float(cfg["sibling_discount"])
        self._bm25_cache_version = int(cfg["bm25_cache_version"])
        self._bm25_meta_file = str(cfg["bm25_meta_file"])
        self._node_bm25_file = str(cfg["node_bm25_file"])
        self._chunk_bm25_file = str(cfg["chunk_bm25_file"])
        self._emb_meta_file = str(cfg["emb_meta_file"])
        self._node_emb_file = str(cfg["node_emb_file"])
        self._chunk_emb_file = str(cfg["chunk_emb_file"])

        # Node document = name + type + description + all edge notes attached to
        # that node (outgoing and incoming).  Including notes means queries that
        # use the language of a relationship ("Hodgkin-Huxley predicts action
        # potential shape") can retrieve the correct node even when those words
        # don't appear in the node's own description.
        self._node_ids = [n.id for n in kb.nodes]
        # Collect all edge notes per node (both sides of each edge) so that
        # retrieval can match queries using the language of relationships.
        self._node_notes: dict[str, list[str]] = {n.id: [] for n in kb.nodes}
        for edge in kb.edges:
            if edge.notes.strip():
                if edge.from_id in self._node_notes:
                    self._node_notes[edge.from_id].append(edge.notes)
                if edge.to_id in self._node_notes:
                    self._node_notes[edge.to_id].append(edge.notes)
        node_docs = [
            " ".join([n.name, n.type, n.description] + self._node_notes[n.id])
            for n in kb.nodes
        ]
        chunk_docs = [self._chunk_doc(c) for c in kb.chunks]
        if not self._load_bm25_cache():
            self._node_bm25 = _BM25Index.build(node_docs, k1=self._bm25_k1, b=self._bm25_b)
            self._chunk_bm25 = _BM25Index.build(chunk_docs, k1=self._bm25_k1, b=self._bm25_b)
            self._save_bm25_cache()

        # Dense model (optional)
        self._dense_model = None
        self._node_embs = None   # shape (n_nodes, emb_dim), float32, L2-normalised
        self._chunk_embs = None  # shape (n_chunks, emb_dim), float32, L2-normalised

        resolved = Path(model_dir) if model_dir else _default_model_dir()
        self._try_load_dense(resolved)

    # ------------------------------------------------------------------
    # Dense model loading
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Embedding cache helpers
    # ------------------------------------------------------------------

    def _corpus_hash(self) -> str:
        """MD5 of indexed corpus content; changes when retrieval inputs change."""
        import hashlib  # noqa: PLC0415
        node_parts = (
            "\x1f".join([n.id, n.name, n.type, n.description, *self._node_notes[n.id]])
            for n in sorted(self.kb.nodes, key=lambda n: n.id)
        )
        chunk_parts = (
            "\x1f".join([
                str(c.id),
                c.doi,
                c.title,
                c.authors,
                str(c.year),
                str(c.chunk_index),
                c.text,
                *c.taxonomy_node_ids,
            ])
            for c in sorted(self.kb.chunks, key=lambda c: c.id)
        )
        node_ids = "\x1f".join(sorted(self._node_ids))
        token = node_ids + "\x1d" + "\x1e".join(node_parts) + "\x1d" + "\x1e".join(chunk_parts)
        return hashlib.md5(token.encode()).hexdigest()

    def _chunk_doc(self, chunk: PaperChunk) -> str:
        """Build the indexed text for a chunk, including its taxonomy links."""
        linked_parts: list[str] = []
        for nid in chunk.taxonomy_node_ids:
            if nid not in self.kb.nodes_by_id:
                continue
            node = self.kb.nodes_by_id[nid]
            linked_parts.extend([nid, node.name, node.type, node.description])
        return " ".join([
            chunk.title,
            f"chunk_index_{chunk.chunk_index}",
            *chunk.taxonomy_node_ids,
            *linked_parts,
            chunk.text,
        ])

    def _load_bm25_cache(self) -> bool:
        """Return True and populate BM25 indices if a valid cache exists."""
        if self._cache_dir is None:
            return False
        import json  # noqa: PLC0415
        meta_path = self._cache_dir / self._bm25_meta_file
        node_path = self._cache_dir / self._node_bm25_file
        chunk_path = self._cache_dir / self._chunk_bm25_file
        if not (meta_path.exists() and node_path.exists() and chunk_path.exists()):
            return False
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("version") != self._bm25_cache_version:
                return False
            if meta.get("corpus_hash") != self._corpus_hash():
                return False
            if float(meta.get("k1", 0.0)) != self._bm25_k1:
                return False
            if float(meta.get("b", 0.0)) != self._bm25_b:
                return False
            self._node_bm25 = _BM25Index.from_dict(
                json.loads(node_path.read_text()),
                k1=self._bm25_k1,
                b=self._bm25_b,
            )
            self._chunk_bm25 = _BM25Index.from_dict(
                json.loads(chunk_path.read_text()),
                k1=self._bm25_k1,
                b=self._bm25_b,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if self._node_bm25._N != len(self._node_ids):
            return False
        if self._chunk_bm25._N != len(self.kb.chunks):
            return False
        return True

    def _save_bm25_cache(self) -> None:
        """Persist BM25 indices and metadata to disk."""
        if self._cache_dir is None:
            return
        import json  # noqa: PLC0415
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        (self._cache_dir / self._node_bm25_file).write_text(json.dumps(self._node_bm25.to_dict()))
        (self._cache_dir / self._chunk_bm25_file).write_text(json.dumps(self._chunk_bm25.to_dict()))
        (self._cache_dir / self._bm25_meta_file).write_text(json.dumps({
            "version": self._bm25_cache_version,
            "corpus_hash": self._corpus_hash(),
            "k1": self._bm25_k1,
            "b": self._bm25_b,
        }))

    def _load_emb_cache(self, model_dir: Path) -> bool:
        """Return True and populate embedding arrays if a valid cache exists."""
        if self._cache_dir is None:
            return False
        import json  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        meta_path   = self._cache_dir / self._emb_meta_file
        node_path   = self._cache_dir / self._node_emb_file
        chunk_path  = self._cache_dir / self._chunk_emb_file
        if not (meta_path.exists() and node_path.exists() and chunk_path.exists()):
            return False
        meta = json.loads(meta_path.read_text())
        if meta.get("model") != str(model_dir.resolve()):
            return False
        if meta.get("corpus_hash") != self._corpus_hash():
            return False
        self._node_embs  = np.load(str(node_path))
        self._chunk_embs = np.load(str(chunk_path))
        if self._node_embs.shape[0] != len(self._node_ids):
            self._node_embs = self._chunk_embs = None
            return False
        if self._chunk_embs.shape[0] != len(self.kb.chunks):
            self._node_embs = self._chunk_embs = None
            return False
        return True

    def _print_emb_cache_hit(self) -> None:
        """Print where existing embedding files were loaded from."""
        if self._cache_dir is None:
            return
        print(f"Embedding cache already exists at {self._cache_dir}/", flush=True)
        print(f"  Taxonomy node embeddings: {self._cache_dir / self._node_emb_file}", flush=True)
        print(f"  Paper chunk embeddings:   {self._cache_dir / self._chunk_emb_file}", flush=True)
        print(f"  Cache metadata:           {self._cache_dir / self._emb_meta_file}", flush=True)

    def _save_emb_cache(self, model_dir: Path) -> None:
        """Persist embedding arrays and metadata to disk."""
        if self._cache_dir is None:
            return
        import json  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self._cache_dir / self._node_emb_file),  self._node_embs)
        np.save(str(self._cache_dir / self._chunk_emb_file), self._chunk_embs)
        (self._cache_dir / self._emb_meta_file).write_text(json.dumps({
            "model":       str(model_dir.resolve()),
            "corpus_hash": self._corpus_hash(),
        }))

    # ------------------------------------------------------------------
    # Dense model loading
    # ------------------------------------------------------------------

    def _try_load_dense(self, model_dir: Path) -> None:
        """
        Load the sentence-transformers model and obtain corpus embeddings.
        Embeddings are loaded from cache when available; otherwise computed
        and saved for subsequent runs.  Silently does nothing if the model
        directory is absent or sentence-transformers is not installed.
        """
        if not model_dir.exists():
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            return

        self._dense_model = SentenceTransformer(str(model_dir))

        if self._load_emb_cache(model_dir):
            self._print_emb_cache_hit()
            return  # cache hit — skip encoding

        node_docs = [
            " ".join([n.name, n.description] + self._node_notes[n.id])
            for n in self.kb.nodes
        ]
        chunk_docs = [self._chunk_doc(c) for c in self.kb.chunks]
        cache_target = f"{self._cache_dir}/" if self._cache_dir else "memory only"

        print("No valid embedding cache found for this model and corpus.", flush=True)
        print(f"Creating embeddings and storing them in: {cache_target}", flush=True)
        print(
            f"  Creating taxonomy node embeddings for {len(node_docs)} nodes "
            f"-> {self._cache_dir / self._node_emb_file if self._cache_dir else 'not cached'}",
            flush=True,
        )
        self._node_embs = self._dense_model.encode(
            node_docs, normalize_embeddings=True, show_progress_bar=False
        )
        print(
            f"  Creating paper chunk embeddings for {len(chunk_docs)} chunks "
            f"-> {self._cache_dir / self._chunk_emb_file if self._cache_dir else 'not cached'}",
            flush=True,
        )
        self._chunk_embs = self._dense_model.encode(
            chunk_docs, normalize_embeddings=True, show_progress_bar=False
        )
        self._save_emb_cache(model_dir)
        if self._cache_dir:
            print(f"  Wrote embedding metadata -> {self._cache_dir / self._emb_meta_file}", flush=True)

    @property
    def is_hybrid(self) -> bool:
        """True when the dense model is loaded and RRF fusion is active."""
        return (
            self._dense_model is not None
            and self._node_embs is not None
            and self._chunk_embs is not None
        )

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_scores(
        bm25_scores: list[float],
        dense_scores: list[float],
        rrf_k: int,
    ) -> list[float]:
        n = len(bm25_scores)
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
        dense_order = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)
        bm25_rank = {idx: r for r, idx in enumerate(bm25_order)}
        dense_rank = {idx: r for r, idx in enumerate(dense_order)}
        return [
            1 / (rrf_k + bm25_rank[i]) + 1 / (rrf_k + dense_rank[i])
            for i in range(n)
        ]

    @classmethod
    def _rrf_merge(
        cls,
        bm25_scores: list[float],
        dense_scores: list[float],
        items: list,
        top_k: int,
        rrf_k: int,
    ) -> list[tuple]:
        rrf = cls._rrf_scores(bm25_scores, dense_scores, rrf_k)
        ranked = sorted(enumerate(rrf), key=lambda x: x[1], reverse=True)
        return [(items[i], s) for i, s in ranked[:top_k]]

    # ------------------------------------------------------------------
    # BM25-only paths
    # ------------------------------------------------------------------

    def _bm25_nodes(self, query: str, top_k: int) -> list[tuple[Node, float]]:
        if top_k <= 0:
            return []
        scores = self._node_bm25.score(query)
        ranked = nlargest(
            top_k,
            ((self.kb.nodes_by_id[self._node_ids[i]], s)
             for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
        )
        return ranked

    def _bm25_chunks(self, query: str, top_k: int) -> list[tuple[PaperChunk, float]]:
        if top_k <= 0:
            return []
        scores = self._chunk_bm25.score(query)
        return self._sibling_boost(scores, top_k)

    def _sibling_boost(
        self,
        all_scores: list[float],
        top_k: int,
    ) -> list[tuple[PaperChunk, float]]:
        """
        Boost chunks that share a DOI with any initial top-k seed.

        After a first-pass ranking, every chunk whose DOI appears among the
        seed set is guaranteed a score of at least the configured sibling
        discount times the best seed score for that DOI.
        The full corpus is then re-ranked and top-k returned.
        """
        chunks = self.kb.chunks
        n = len(chunks)
        if top_k <= 0 or n == 0:
            return []

        # Best seed scores per DOI. Distance in chunk_index keeps adjacent
        # passages near the top without flattening a whole paper into one score.
        seed_indices = nlargest(top_k, range(n), key=lambda i: all_scores[i])
        doi_seeds: dict[str, list[tuple[int, float]]] = {}
        for i in seed_indices:
            s = all_scores[i]
            doi = chunks[i].doi
            if s > 0:
                doi_seeds.setdefault(doi, []).append((chunks[i].chunk_index, s))

        # Apply a distance-decayed floor to every chunk that shares a DOI with a seed.
        final_scores = list(all_scores)
        for i, chunk in enumerate(chunks):
            floors = [
                self._sibling_discount * seed_score / (abs(chunk.chunk_index - seed_index) + 1)
                for seed_index, seed_score in doi_seeds.get(chunk.doi, [])
            ]
            if floors:
                final_scores[i] = max(final_scores[i], max(floors))

        ranked = nlargest(
            top_k,
            ((chunks[i], final_scores[i]) for i in range(n) if final_scores[i] > 0),
            key=lambda x: x[1],
        )
        return ranked

    # ------------------------------------------------------------------
    # Hybrid (BM25 + dense) paths
    # ------------------------------------------------------------------

    def _hybrid_nodes(self, query: str, top_k: int) -> list[tuple[Node, float]]:
        if top_k <= 0:
            return []
        bm25_scores = self._node_bm25.score(query)
        q_emb = self._dense_model.encode([query], normalize_embeddings=True)[0]
        return self._hybrid_nodes_from_scores(bm25_scores, q_emb, top_k)

    def _hybrid_nodes_from_scores(self, bm25_scores: list[float], q_emb, top_k: int) -> list[tuple[Node, float]]:
        dense_scores = (self._node_embs @ q_emb).tolist()
        nodes = [self.kb.nodes_by_id[nid] for nid in self._node_ids]
        return self._rrf_merge(bm25_scores, dense_scores, nodes, top_k, self._node_rrf_k)

    def _hybrid_chunks(self, query: str, top_k: int) -> list[tuple[PaperChunk, float]]:
        if top_k <= 0:
            return []
        bm25_scores = self._chunk_bm25.score(query)
        q_emb = self._dense_model.encode([query], normalize_embeddings=True)[0]
        return self._hybrid_chunks_from_scores(bm25_scores, q_emb, top_k)

    def _hybrid_chunks_from_scores(self, bm25_scores: list[float], q_emb, top_k: int) -> list[tuple[PaperChunk, float]]:
        dense_scores = (self._chunk_embs @ q_emb).tolist()
        rrf_scores = self._rrf_scores(bm25_scores, dense_scores, self._chunk_rrf_k)
        return self._sibling_boost(rrf_scores, top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_nodes(self, query: str, top_k: int = 6) -> list[tuple[Node, float]]:
        """Return (node, score) pairs ranked by BM25 or RRF, highest first."""
        return self._hybrid_nodes(query, top_k) if self.is_hybrid else self._bm25_nodes(query, top_k)

    def retrieve_chunks(self, query: str, top_k: int = 6) -> list[tuple[PaperChunk, float]]:
        """Return (chunk, score) pairs ranked by BM25 or RRF, highest first."""
        return self._hybrid_chunks(query, top_k) if self.is_hybrid else self._bm25_chunks(query, top_k)

    def retrieve(
        self,
        query: str,
        node_top_k: int = 6,
        chunk_top_k: int = 6,
    ) -> tuple[list[tuple[Node, float]], list[tuple[PaperChunk, float]]]:
        """
        Return ranked taxonomy nodes and paper chunks for a query.

        This combined path keeps node and chunk retrieval independent while
        avoiding duplicate dense query encoding in hybrid mode.
        """
        if self.is_hybrid:
            q_emb = self._dense_model.encode([query], normalize_embeddings=True)[0]
            with ThreadPoolExecutor(max_workers=2) as executor:
                node_scores_f = executor.submit(self._node_bm25.score, query)
                chunk_scores_f = executor.submit(self._chunk_bm25.score, query)
                node_scores = node_scores_f.result()
                chunk_scores = chunk_scores_f.result()
            return (
                self._hybrid_nodes_from_scores(node_scores, q_emb, node_top_k),
                self._hybrid_chunks_from_scores(chunk_scores, q_emb, chunk_top_k),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            nodes_f = executor.submit(self._bm25_nodes, query, node_top_k)
            chunks_f = executor.submit(self._bm25_chunks, query, chunk_top_k)
            return nodes_f.result(), chunks_f.result()
