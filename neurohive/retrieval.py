"""
BM25 and hybrid (BM25 + dense) retrieval over taxonomy nodes and paper chunks.

At initialisation the Retriever checks for a sentence-transformers model at
models/all-MiniLM-L6-v2/ (relative to the repository root).  If the directory
exists and sentence-transformers is installed, corpus embeddings are
pre-computed once and hybrid retrieval via Reciprocal Rank Fusion is used.
Otherwise pure BM25 Okapi is used.  The public API is identical either way.

BM25 Okapi
----------
    score(q, d) = Σ_t  IDF(t) · f(t,d)·(k₁+1)
                                ─────────────────────────────────────
                                f(t,d) + k₁·(1 − b + b·|d|/avgdl)

    k₁ = 1.5,  b = 0.75  (standard Okapi defaults)

Reciprocal Rank Fusion
----------------------
    rrf(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))

    k = 60  (standard RRF default)

Sibling boost (chunk retrieval only)
-------------------------------------
After the initial ranking, every chunk that shares a DOI with any top-k seed
is guaranteed a score floor of  _SIBLING_DISCOUNT × best_seed_score_for_doi.
This ensures that if one passage from a paper is relevant, the paper's other
passages are also surfaced, providing fuller context from the same source.
The full corpus is re-ranked after boosting and top-k is then returned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from neurohive.knowledge_base import KnowledgeBase
from neurohive.models import Node, PaperChunk

# BM25 hyperparameters
_K1 = 1.5
_B = 0.75

# RRF constants — lower k sharpens rank differences at the top of the list.
# Taxonomy nodes use a lower k because the corpus is small and structured,
# so BM25 rank differences are meaningful; paper chunks use the standard k=60.
_NODE_RRF_K  = 30
_CHUNK_RRF_K = 60

# Sibling-boost: chunks sharing a DOI with a top-k seed receive at least this
# fraction of that seed's best score, surfacing context from the same paper.
_SIBLING_DISCOUNT = 0.5

def _default_model_dir() -> Path:
    """
    Derive the default embedding model directory from config.toml.
    Falls back to all-MiniLM-L6-v2 if the config cannot be loaded.
    """
    try:
        from neurohive.config import load_config  # noqa: PLC0415
        cfg = load_config()
        model_name = cfg["embeddings"]["model"].split("/")[-1]
    except Exception:
        model_name = "all-MiniLM-L6-v2"
    return Path(__file__).parent.parent / "models" / model_name


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
    _avgdl: float
    _N: int

    @classmethod
    def build(cls, docs: list[str]) -> "_BM25Index":
        tokenized = [_tokenize(d) for d in docs]
        N = len(tokenized)
        avgdl = sum(len(t) for t in tokenized) / max(N, 1)
        df: dict[str, int] = {}
        tf_list: list[dict[str, int]] = []
        for tokens in tokenized:
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            tf_list.append(tf)
            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1
        return cls(_tf=tf_list, _df=df, _avgdl=avgdl, _N=N)

    def score(self, query: str) -> list[float]:
        tokens = _tokenize(query)
        scores = [0.0] * self._N
        for tok in tokens:
            if tok not in self._df:
                continue
            idf = math.log((self._N - self._df[tok] + 0.5) / (self._df[tok] + 0.5) + 1)
            for i, tf in enumerate(self._tf):
                f = tf.get(tok, 0)
                dl = sum(tf.values())
                denom = f + _K1 * (1 - _B + _B * dl / max(self._avgdl, 1))
                scores[i] += idf * f * (_K1 + 1) / denom if denom else 0
        return scores


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    BM25 retriever with optional dense-model hybrid upgrade via RRF.

    If ``models/all-MiniLM-L6-v2/`` exists and sentence-transformers is
    installed, corpus embeddings are pre-computed once at construction time
    and hybrid retrieval is used for every subsequent query.  Otherwise pure
    BM25 is used.  Check ``retriever.is_hybrid`` to see which mode is active.

    Parameters
    ----------
    kb        : Loaded KnowledgeBase instance.
    model_dir : Override the default model directory (useful for testing).
    """

    def __init__(self, kb: KnowledgeBase, model_dir: Path | str | None = None) -> None:
        self.kb = kb

        # BM25 indices (always built)
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
        self._node_bm25 = _BM25Index.build([
            " ".join([n.name, n.type, n.description] + self._node_notes[n.id])
            for n in kb.nodes
        ])
        self._chunk_bm25 = _BM25Index.build(
            [f"{c.title} {c.text}" for c in kb.chunks]
        )

        # Dense model (optional)
        self._dense_model = None
        self._node_embs = None   # shape (n_nodes, emb_dim), float32, L2-normalised
        self._chunk_embs = None  # shape (n_chunks, emb_dim), float32, L2-normalised

        resolved = Path(model_dir) if model_dir else _default_model_dir()
        self._try_load_dense(resolved)

    # ------------------------------------------------------------------
    # Dense model loading
    # ------------------------------------------------------------------

    def _try_load_dense(self, model_dir: Path) -> None:
        """
        Try to load the sentence-transformers model and pre-compute embeddings.
        Silently does nothing if the directory is absent or the package is missing.
        """
        if not model_dir.exists():
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError:
            return

        self._dense_model = SentenceTransformer(str(model_dir))

        node_docs = [
            " ".join([n.name, n.description] + self._node_notes[n.id])
            for n in self.kb.nodes
        ]
        chunk_docs = [f"{c.title} {c.text}" for c in self.kb.chunks]

        self._node_embs = self._dense_model.encode(
            node_docs, normalize_embeddings=True, show_progress_bar=False
        )
        self._chunk_embs = self._dense_model.encode(
            chunk_docs, normalize_embeddings=True, show_progress_bar=False
        )

    @property
    def is_hybrid(self) -> bool:
        """True when the dense model is loaded and RRF fusion is active."""
        return self._dense_model is not None

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_merge(
        bm25_scores: list[float],
        dense_scores: list[float],
        items: list,
        top_k: int,
    ) -> list[tuple]:
        n = len(items)
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
        dense_order = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)
        bm25_rank = {idx: r for r, idx in enumerate(bm25_order)}
        dense_rank = {idx: r for r, idx in enumerate(dense_order)}

        rrf = [
            1 / (_NODE_RRF_K + bm25_rank[i]) + 1 / (_NODE_RRF_K + dense_rank[i])
            for i in range(n)
        ]
        ranked = sorted(enumerate(rrf), key=lambda x: x[1], reverse=True)
        return [(items[i], s) for i, s in ranked[:top_k]]

    # ------------------------------------------------------------------
    # BM25-only paths
    # ------------------------------------------------------------------

    def _bm25_nodes(self, query: str, top_k: int) -> list[tuple[Node, float]]:
        scores = self._node_bm25.score(query)
        ranked = sorted(
            ((self.kb.nodes_by_id[self._node_ids[i]], s)
             for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1], reverse=True,
        )
        return ranked[:top_k]

    def _bm25_chunks(self, query: str, top_k: int) -> list[tuple[PaperChunk, float]]:
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
        seed set is guaranteed a score of at least
            _SIBLING_DISCOUNT * best_seed_score_for_that_doi.
        The full corpus is then re-ranked and top-k returned.
        """
        chunks = self.kb.chunks
        n = len(chunks)

        # Best score per DOI among the initial top-k seeds
        seed_indices = sorted(range(n), key=lambda i: all_scores[i], reverse=True)[:top_k]
        doi_best: dict[str, float] = {}
        for i in seed_indices:
            s = all_scores[i]
            doi = chunks[i].doi
            if s > 0 and (doi not in doi_best or s > doi_best[doi]):
                doi_best[doi] = s

        # Apply the floor to every chunk that shares a DOI with a seed
        final_scores = list(all_scores)
        for i, chunk in enumerate(chunks):
            if chunk.doi in doi_best:
                floor = _SIBLING_DISCOUNT * doi_best[chunk.doi]
                if floor > final_scores[i]:
                    final_scores[i] = floor

        ranked = sorted(
            ((chunks[i], final_scores[i]) for i in range(n) if final_scores[i] > 0),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Hybrid (BM25 + dense) paths
    # ------------------------------------------------------------------

    def _hybrid_nodes(self, query: str, top_k: int) -> list[tuple[Node, float]]:
        bm25_scores = self._node_bm25.score(query)
        q_emb = self._dense_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = (self._node_embs @ q_emb).tolist()
        nodes = [self.kb.nodes_by_id[nid] for nid in self._node_ids]
        return self._rrf_merge(bm25_scores, dense_scores, nodes, top_k)

    def _hybrid_chunks(self, query: str, top_k: int) -> list[tuple[PaperChunk, float]]:
        bm25_scores = self._chunk_bm25.score(query)
        q_emb = self._dense_model.encode([query], normalize_embeddings=True)[0]
        dense_scores = (self._chunk_embs @ q_emb).tolist()
        n = len(self.kb.chunks)
        bm25_order = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)
        dense_order = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)
        bm25_rank = {idx: r for r, idx in enumerate(bm25_order)}
        dense_rank = {idx: r for r, idx in enumerate(dense_order)}
        rrf_scores = [
            1 / (_CHUNK_RRF_K + bm25_rank[i]) + 1 / (_CHUNK_RRF_K + dense_rank[i])
            for i in range(n)
        ]
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
