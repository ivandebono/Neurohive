"""
End-to-end query pipeline.

Flow
----
1. BM25 retrieval        — top-k taxonomy nodes + chunks matching the query
2. Graph expansion       — parents, theories, dimensions, lateral relations
3. Chunk harvest         — paper chunks linked to the expanded node set
4. Context formatting    — structured text blocks for taxonomy and papers
5. Generation            — backend produces {"answer": [paper/taxonomy triples]}
6. Citation verification — strips any reference not traceable to retrieved context
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from neurohive.backends import AnthropicBackend, GenerationBackend
from neurohive.knowledge_base import KnowledgeBase
from neurohive.entities import Node, PaperChunk
from neurohive.retrieval import Retriever

_MAP_SOURCE_LABELS: dict[str, str] = {
    "semantic": "inferred",
}

_EDGE_LABELS: dict[str, str] = {
    "HAS_SUBPILLAR":     "includes subpillar",
    "HAS_RESEARCH_AREA": "encompasses research area",
    "HAS_DIMENSION":     "is characterised by",
    "HAS_THEORY":        "is theorised by",
    "EXPLAINS":          "explains",
    "RELATED_TO":        "is related to",
    "IS_DERIVED_FROM":   "is derived from",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query: str
    answer: str
    citation: list[str]
    document: list[str]
    nodes_used: list[Node]
    chunks_used: list[PaperChunk]
    model: str
    retrieval_mode: str
    # citation string → (authors, title, doi) as stored in paper_chunks.json
    citation_meta: dict[str, tuple[str, str, str]]
    # raw verified triples produced by the LLM, for downstream use
    answer_triples: list[dict]
    cached: bool = False

    def as_dict(self) -> dict:
        """Return the primary output fields as a plain dict."""
        return {
            "answer":         self.answer,
            "citation":       self.citation,
            "document":       self.document,
            "answer_triples": self.answer_triples,
        }

    def __str__(self) -> str:
        cite_lines: list[str] = []
        for c in self.citation:
            meta = self.citation_meta.get(c)
            if meta:
                authors, title, doi = meta
                cite_lines.append(f"  • {authors} ({c.split(',')[-1].strip()})")
                cite_lines.append(f"    \"{title}\"")
                cite_lines.append(f"    DOI: {doi}")
            else:
                cite_lines.append(f"  • {c}")
        citations = "\n".join(cite_lines) or "  (none)"
        documents = "\n".join(f"  • {d.replace('_', ' ')}" for d in self.document) or "  (none)"
        return (
            f"\n{'=' * 70}\n"
            f"QUERY: {self.query}\n"
            f"{'=' * 70}\n"
            f"{self.answer}\n\n"
            f"Citations:\n{citations}\n\n"
            f"Taxonomy concepts:\n{documents}\n"
            f"{'=' * 70}\n"
            f"[{len(self.nodes_used)} nodes · {len(self.chunks_used)} chunks · "
            f"{self.retrieval_mode} · {self.model}]\n"
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class QueryPipeline:
    """
    Natural-language query → structured, grounded response.

    Parameters
    ----------
    data_dir             : Directory containing the three data files.
    backend              : Generation backend instance.  Defaults to AnthropicBackend.
    node_top_k           : Number of taxonomy nodes retrieved by BM25 / RRF.
    chunk_top_k          : Number of paper chunks retrieved by BM25 / RRF.
    confidence_threshold : Minimum edge confidence for graph expansion (default 0.8).
                           Filters weak RELATED_TO edges; hierarchy edges are
                           unaffected (always 1.0).  Set to 0.0 to disable.
    model_dir            : Override path for the sentence-transformers model directory.

    Examples
    --------
    # Anthropic API (default)
    pipeline = QueryPipeline()

    # Stronger model
    from neurohive.backends import AnthropicBackend
    pipeline = QueryPipeline(backend=AnthropicBackend("claude-sonnet-4-6"))
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        backend: GenerationBackend | None = None,
        node_top_k: int | None = None,
        chunk_top_k: int | None = None,
        confidence_threshold: float | None = None,
        verify_entailment: bool = False,
        rerank: bool = False,
        nli_model_dir: Path | str | None = None,
        entailment_threshold: float | None = None,
        model_dir: Path | str | None = None,
        cache_enabled: bool | None = None,
    ):
        from neurohive.config import load_config  # noqa: PLC0415
        cfg = load_config()
        repo_root = Path(__file__).parent.parent
        resolved_data_dir = Path(data_dir) if data_dir is not None else Path(cfg.paths.data_dir)
        if not resolved_data_dir.is_absolute():
            resolved_data_dir = repo_root / resolved_data_dir

        self.node_top_k = node_top_k if node_top_k is not None else cfg.pipeline.node_top_k
        self.chunk_top_k = chunk_top_k if chunk_top_k is not None else cfg.pipeline.chunk_top_k
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else cfg.pipeline.confidence_threshold
        )
        self._backend = backend or AnthropicBackend()

        self.kb = KnowledgeBase(resolved_data_dir)
        self.retriever = Retriever(self.kb, model_dir=model_dir, cache_dir=self.kb.cache_dir)

        # NLI cross-encoder: shared between re-ranking and entailment checking.
        self._nli_model = None
        self._nli_threshold: float = cfg.nli.entailment_threshold
        if verify_entailment or rerank:
            self._load_nli_model(nli_model_dir, entailment_threshold)

        self._verify_entailment = verify_entailment
        self._rerank = rerank and self._nli_model is not None
        self._rerank_pool = cfg.reranker.pool_multiplier

        # Query result cache
        cache_cfg = cfg.cache
        _cache_on = cache_enabled if cache_enabled is not None else cache_cfg.enabled
        if _cache_on:
            from neurohive.cache import QueryCache  # noqa: PLC0415
            cache_db = resolved_data_dir / "database" / cache_cfg.db_file
            self._cache: QueryCache | None = QueryCache(
                cache_db, ttl_seconds=cache_cfg.ttl_seconds
            )
        else:
            self._cache = None

    @property
    def retrieval_mode(self) -> str:
        return "hybrid (BM25 + dense, RRF)" if self.retriever.is_hybrid else "bm25"

    # ------------------------------------------------------------------
    # NLI model loading
    # ------------------------------------------------------------------

    def _load_nli_model(self, nli_model_dir: Path | str | None, entailment_threshold: float | None = None) -> None:
        """
        Load the cross-encoder NLI model for post-generation entailment checking.

        The model directory is resolved in this order:
          1. ``nli_model_dir`` argument (explicit override)
          2. ``<models_dir>/<model_name>/`` where ``models_dir`` comes from
             config.toml [paths] and ``model_name`` is the last component of
             the HuggingFace ID in config.toml (e.g. "nli-deberta-v3-small")

        The model must be pre-downloaded with::

            make setup-nli && make download-nli-model

        Raises
        ------
        FileNotFoundError   if the local model directory is absent or empty.
        ImportError         if sentence-transformers is not installed.
        """
        from neurohive.config import load_config  # noqa: PLC0415
        cfg = load_config()
        nli_cfg = cfg.nli
        hf_model_id: str = nli_cfg.model
        self._nli_threshold = entailment_threshold if entailment_threshold is not None else nli_cfg.entailment_threshold

        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for entailment checking.\n"
                "Run:  make setup-nli"
            ) from exc

        if nli_model_dir is not None:
            model_path = Path(nli_model_dir)
        else:
            model_name = hf_model_id.split("/")[-1]
            repo_root = Path(__file__).parent.parent
            models_dir = Path(cfg.paths.models_dir)
            if not models_dir.is_absolute():
                models_dir = repo_root / models_dir
            model_path = models_dir / model_name

        if not (model_path.exists() and any(model_path.iterdir())):
            raise FileNotFoundError(
                f"NLI model not found at {model_path}.\n"
                "Run:  make setup-nli && make download-nli-model"
            )

        self._nli_model = CrossEncoder(str(model_path), num_labels=3)

    # ------------------------------------------------------------------
    # Re-ranking
    # ------------------------------------------------------------------

    def _rerank_chunks(
        self, query: str, chunks: list[PaperChunk], keep: int
    ) -> list[PaperChunk]:
        """
        Score (query, chunk.text) pairs with the cross-encoder and return the
        top-``keep`` chunks by entailment probability (column 1 of the 3-class
        NLI output: contradiction=0, entailment=1, neutral=2).
        """
        if not chunks or len(chunks) <= keep:
            return chunks
        import numpy as np  # noqa: PLC0415
        pairs = [(query, c.text) for c in chunks]
        scores = np.atleast_2d(self._nli_model.predict(pairs, apply_softmax=True))
        entailment_scores = scores[:, 1].tolist()
        ranked = sorted(zip(entailment_scores, chunks), reverse=True)
        return [c for _, c in ranked[:keep]]

    # ------------------------------------------------------------------
    # Step 1 + 2: retrieve & expand
    # ------------------------------------------------------------------

    def _retrieve_and_expand(self, query: str) -> tuple[list[Node], list[PaperChunk]]:
        # Fetch a larger candidate pool when re-ranking is enabled.
        chunk_fetch_k = self.chunk_top_k * self._rerank_pool if self._rerank else self.chunk_top_k
        top_nodes_scored, top_chunks_scored = self.retriever.retrieve(
            query,
            self.node_top_k,
            chunk_fetch_k,
        )

        seed_ids = {n.id for n, _ in top_nodes_scored}
        seed_ids.update(
            nid
            for chunk, _ in top_chunks_scored
            for nid in chunk.taxonomy_node_ids
        )
        expanded_ids = self.kb.smart_expand(seed_ids, self.confidence_threshold)
        all_nodes = [self.kb.nodes_by_id[nid] for nid in expanded_ids
                     if nid in self.kb.nodes_by_id]

        bm25_chunks = {c.id: c for c, _ in top_chunks_scored}
        node_chunks = {c.id: c for c in self.kb.chunks_for_nodes(expanded_ids)}
        all_chunks = list({**node_chunks, **bm25_chunks}.values())

        if self._rerank:
            all_chunks = self._rerank_chunks(query, all_chunks, self.chunk_top_k)

        return all_nodes, all_chunks

    # ------------------------------------------------------------------
    # Step 3: format context
    # ------------------------------------------------------------------

    def _format_taxonomy_context(self, nodes: list[Node]) -> str:
        node_set = {n.id for n in nodes}
        lines: list[str] = []
        order = {"Pillar": 0, "Subpillar": 1, "Research_area": 2, "Theory": 3, "Dimension": 4}
        for node in sorted(nodes, key=lambda n: order.get(n.type, 5)):
            breadcrumb = self.kb.breadcrumb(node.id)
            lines.append(f"[{node.type}] {breadcrumb}")
            lines.append(f"  {node.description}")
            hidden = 0
            for edge in self.kb.outgoing.get(node.id, []):
                if edge.to_id in node_set:
                    target = self.kb.nodes_by_id[edge.to_id]
                    label = _EDGE_LABELS.get(edge.relationship_type, edge.relationship_type)
                    conf = f" [confidence: {edge.confidence:.0%}]" if edge.confidence < 1.0 else ""
                    source_label = _MAP_SOURCE_LABELS.get(edge.map_source, edge.map_source)
                    inferred = f" [{source_label}]" if edge.map_source != "canonical" else ""
                    lines.append(f"  {label} → {target.name.replace('_', ' ')}{conf}{inferred}")
                    if edge.notes.strip():
                        lines.append(f"    ({edge.notes})")
                else:
                    hidden += 1
            if hidden:
                lines.append(f"  [+ {hidden} further connection{'s' if hidden > 1 else ''} not retrieved for this query]")
            if node.isolated:
                lines.append("  [no connections]")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_paper_context(chunks: list[PaperChunk], kb: KnowledgeBase) -> str:
        by_doi: dict[str, list[PaperChunk]] = defaultdict(list)
        for chunk in chunks:
            by_doi[chunk.doi].append(chunk)
        lines: list[str] = []
        for doi, doi_chunks in by_doi.items():
            doi_chunks.sort(key=lambda c: c.chunk_index)
            first = doi_chunks[0]
            lines.append(f"[{first.citation}] \"{first.title}\" DOI:{doi}")
            linked_names = sorted({
                kb.nodes_by_id[nid].name.replace("_", " ")
                for c in doi_chunks
                for nid in c.taxonomy_node_ids
                if nid in kb.nodes_by_id
            })
            if linked_names:
                lines.append(f"  Concepts: {', '.join(linked_names)}")
            for chunk in doi_chunks:
                lines.append(f"  • {chunk.text}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step 4: generate
    # ------------------------------------------------------------------

    def _generate(self, query: str, nodes: list[Node], chunks: list[PaperChunk]) -> dict:
        taxonomy_ctx = self._format_taxonomy_context(nodes)
        paper_ctx = self._format_paper_context(chunks, self.kb)
        return self._backend.generate(query, taxonomy_ctx, paper_ctx)

    # ------------------------------------------------------------------
    # Step 5: verify
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(s: str) -> str:
        """Aggressive normalisation for citation/concept key matching."""
        return re.sub(r"[^a-z0-9]", "", s.lower())

    @staticmethod
    def _normalise_text(s: str) -> str:
        """Loose normalisation for verbatim quote substring matching."""
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower()).strip())

    @staticmethod
    def _assemble_answer(triples: list[dict]) -> str:
        """Render verified triples as a readable answer string."""
        if not triples:
            return (
                "(All model-generated claims were removed by citation verification. "
                "Try rephrasing the query or run with --debug to inspect raw triples.)"
            )
        parts = []
        for t in triples:
            if t["type"] == "paper":
                parts.append(f"{t['claim']} [{t['citation']}]")
            else:
                parts.append(f"{t['claim']} [{t['concept']}]")
        return " ".join(parts)

    def _verify(self, structured: dict, nodes: list[Node], chunks: list[PaperChunk]) -> dict:
        """
        Verify each triple in the LLM's structured answer against the retrieved context.

        Paper triples
        -------------
        • citation must match a retrieved PaperChunk citation (normalised key).
          Unmatched triples are dropped and a UserWarning is emitted.
        • quote is checked as a normalised substring of the matched chunk text.
          A mismatch warns but keeps the triple — the citation is verified.

        Taxonomy triples
        ----------------
        • concept must match a retrieved Node name (normalised key).
          Unmatched triples are dropped and a UserWarning is emitted.

        Returns a dict with:
          answer_triples : verified triples (paper + taxonomy)
          citation       : deduplicated citation strings from verified paper triples
          document       : deduplicated concept names from verified taxonomy triples
        """
        norm_to_chunks: dict[str, list[PaperChunk]] = {}
        for c in chunks:
            norm_to_chunks.setdefault(self._normalise(c.citation), []).append(c)
        valid_concepts: dict[str, str] = {self._normalise(n.name): n.name for n in nodes}

        raw = structured.get("answer", [])
        if not isinstance(raw, list):
            raw = []

        verified: list[dict] = []
        for triple in raw:
            if not isinstance(triple, dict):
                continue
            triple_type = triple.get("type")

            if triple_type == "paper":
                citation = str(triple.get("citation", "")).strip()
                quote    = str(triple.get("quote",    "")).strip()
                claim    = str(triple.get("claim",    "")).strip()
                norm_cite = self._normalise(citation)
                matching  = norm_to_chunks.get(norm_cite, [])
                if not matching:
                    warnings.warn(
                        f"Paper triple citation not in retrieved context: {citation!r}",
                        stacklevel=3,
                    )
                    continue
                if quote and not any(
                    self._normalise_text(quote) in self._normalise_text(c.text)
                    for c in matching
                ):
                    warnings.warn(
                        f"Quote not found verbatim in [{citation}]: {quote[:80]!r}",
                        stacklevel=3,
                    )
                verified.append({"type": "paper", "claim": claim,
                                  "quote": quote, "citation": citation})

            elif triple_type == "taxonomy":
                concept = str(triple.get("concept", "")).strip()
                claim   = str(triple.get("claim",   "")).strip()
                norm_concept = self._normalise(concept)
                if norm_concept not in valid_concepts:
                    # Model may have written the full breadcrumb path ("A → B → C");
                    # try just the last component.
                    last_seg = re.split(r"[→>|]", concept)[-1].strip()
                    norm_concept = self._normalise(last_seg)
                if norm_concept not in valid_concepts:
                    warnings.warn(
                        f"Taxonomy triple concept not in retrieved nodes: {concept!r}",
                        stacklevel=3,
                    )
                    continue
                verified.append({"type": "taxonomy", "claim": claim,
                                  "concept": valid_concepts[norm_concept]})

            else:
                warnings.warn(
                    f"Unknown triple type {triple_type!r}; skipping",
                    stacklevel=3,
                )

        citations = list(dict.fromkeys(
            t["citation"] for t in verified if t["type"] == "paper"
        ))
        documents = list(dict.fromkeys(
            t["concept"] for t in verified if t["type"] == "taxonomy"
        ))
        return {
            "answer_triples": verified,
            "citation":       citations,
            "document":       documents,
        }

    # ------------------------------------------------------------------
    # Step 6 (optional): NLI entailment check
    # ------------------------------------------------------------------

    def _entailment_check(
        self,
        triples: list[dict],
        chunks: list[PaperChunk],
    ) -> None:
        """
        Cross-encoder entailment check for every paper triple in the verified answer.

        For each paper triple the claim is used as the NLI hypothesis and the
        retrieved chunk texts for that citation are used as premises.  The
        cross-encoder predicts a 3-class probability vector
        (contradiction=0, entailment=1, neutral=2).  If the best entailment score
        across all premises falls below ``_nli_threshold``, a UserWarning is emitted.

        Taxonomy triples are skipped — concept membership was already verified by
        exact name matching in ``_verify``.

        This method only emits warnings; it does not modify the answer.
        It is a no-op when ``_nli_model`` is ``None`` (i.e. entailment checking is off).

        Performance note
        ----------------
        The DeBERTa-v3-small cross-encoder is CPU-capable but slow on long texts.
        Each claim/chunk pair takes ~50–200 ms on a modern CPU.
        """
        if self._nli_model is None:
            return

        import numpy as np  # noqa: PLC0415

        norm_to_chunks: dict[str, list[PaperChunk]] = {}
        for c in chunks:
            norm_to_chunks.setdefault(self._normalise(c.citation), []).append(c)

        work_items: list[tuple[dict, str, list[tuple[str, str]]]] = []
        all_pairs: list[tuple[str, str]] = []
        for triple in triples:
            if triple.get("type") != "paper":
                continue
            citation = triple["citation"]
            claim    = triple["claim"]
            matching = norm_to_chunks.get(self._normalise(citation), [])
            if not matching:
                continue

            pairs = [(c.text, claim) for c in matching]
            work_items.append((triple, citation, pairs))
            all_pairs.extend(pairs)

        if not all_pairs:
            return

        scores = self._nli_model.predict(all_pairs, apply_softmax=True)
        scores_arr = np.atleast_2d(scores)

        offset = 0
        for triple, citation, pairs in work_items:
            pair_count = len(pairs)
            triple_scores = scores_arr[offset:offset + pair_count]
            offset += pair_count
            # Column 1 = entailment probability (DeBERTa label ordering:
            # 0=contradiction, 1=entailment, 2=neutral).
            best_entailment = float(np.max(triple_scores[:, 1]))
            claim = triple["claim"]

            if best_entailment < self._nli_threshold:
                warnings.warn(
                    f"Low entailment score ({best_entailment:.0%}) for claim "
                    f"[{citation}]: {repr(claim[:120] + '...' if len(claim) > 120 else claim)}",
                    stacklevel=3,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _corpus_fingerprint(self) -> str:
        from neurohive.cache import QueryCache  # noqa: PLC0415
        conn = self.kb._conn
        n_nodes  = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges  = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return QueryCache.corpus_fingerprint(n_nodes, n_edges, n_chunks)

    def run(self, query: str, debug: bool = False) -> QueryResult:
        """Run the full pipeline and return a QueryResult."""
        # ── Cache lookup ───────────────────────────────────────────────
        cache_key: str | None = None
        if self._cache is not None:
            from neurohive.cache import QueryCache  # noqa: PLC0415
            cache_key = QueryCache.make_key(
                query,
                self._backend.model_id,
                self.node_top_k,
                self.chunk_top_k,
                self.confidence_threshold,
                self._rerank,
                self._corpus_fingerprint(),
            )
            hit = self._cache.get(cache_key)
            if hit is not None:
                return QueryResult(
                    query=query,
                    answer=hit["answer"],
                    citation=hit["citation"],
                    document=hit["document"],
                    nodes_used=[],
                    chunks_used=[],
                    model=hit["model"],
                    retrieval_mode=hit["retrieval_mode"],
                    citation_meta={k: tuple(v) for k, v in hit["citation_meta"].items()},
                    answer_triples=hit["answer_triples"],
                    cached=True,
                )

        # ── Full pipeline ──────────────────────────────────────────────
        nodes, chunks = self._retrieve_and_expand(query)
        structured = self._generate(query, nodes, chunks)

        if debug:
            print(f"[debug] full structured output: {structured}")
            raw = structured.get("answer", [])
            n = len(raw) if isinstance(raw, list) else f"non-list ({type(raw).__name__})"
            print(f"[debug] raw model output: {n} triple(s)")
            if isinstance(raw, list):
                for i, t in enumerate(raw):
                    print(f"[debug]   [{i}] {t}")

        verified = self._verify(structured, nodes, chunks)
        if self._verify_entailment:
            self._entailment_check(verified["answer_triples"], chunks)

        norm_to_meta: dict[str, tuple[str, str, str]] = {}
        for c in chunks:
            norm_to_meta[self._normalise(c.citation)] = (c.authors, c.title, c.doi)

        citation_meta: dict[str, tuple[str, str, str]] = {}
        for cite in verified["citation"]:
            meta = norm_to_meta.get(self._normalise(cite))
            if meta:
                citation_meta[cite] = meta

        result = QueryResult(
            query=query,
            answer=self._assemble_answer(verified["answer_triples"]),
            citation=verified["citation"],
            document=verified["document"],
            nodes_used=nodes,
            chunks_used=chunks,
            model=self._backend.model_id,
            retrieval_mode=self.retrieval_mode,
            citation_meta=citation_meta,
            answer_triples=verified["answer_triples"],
        )

        if self._cache is not None and cache_key is not None:
            self._cache.put(
                cache_key, query, self._backend.model_id,
                {
                    "answer":        result.answer,
                    "citation":      result.citation,
                    "document":      result.document,
                    "answer_triples": result.answer_triples,
                    "citation_meta": {k: list(v) for k, v in citation_meta.items()},
                    "model":         result.model,
                    "retrieval_mode": result.retrieval_mode,
                },
            )

        return result
