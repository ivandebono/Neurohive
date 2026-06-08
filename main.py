"""
Neurohive Query Pipeline
------------------------
Takes a natural language query and returns a grounded response from the
neuroscience knowledge store (taxonomy graph + paper chunks).

Usage
-----
    # Single query
    python main.py "What is the role of voltage-gated sodium channels?"

    # Interactive REPL
    python main.py

    # Override Anthropic model
    python main.py --model claude-sonnet-4-6 "query..."

    # Use local Ollama generation instead of Anthropic
    python main.py --ollama "query..."
    python main.py --ollama llama3.1:8b "query..."

    # Hybrid retrieval is enabled automatically when the configured model exists:
    #   uv sync --extra embeddings
    #   uv run python scripts/download_model.py

    # Show retrieved sources
    python main.py --show-sources "What are the main theories of synaptic plasticity?"

.env keys
---------
    ANTHROPIC_API_KEY   required only for Anthropic
    ANTHROPIC_MODEL     optional Anthropic model override, e.g. claude-sonnet-4-6

config.toml
-----------
    Provides defaults for pipeline, retrieval, and runtime paths. Every config.toml value
    can be overridden at query time via CLI flags (listed below).  Priority
    order (highest first): CLI flag → env var → config.toml → loader fallback.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).parent


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _REPO_ROOT / path


def _load_env(env_path: Path) -> None:
    """Load key=value pairs from .env into os.environ (does not overwrite)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        print("Add it to .env or export it:  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)


def _print_result(result, show_sources: bool = False) -> None:
    if result.cached:
        print("[cached]")
    print(result)
    if show_sources:
        print("TAXONOMY NODES RETRIEVED")
        for node in sorted(result.nodes_used, key=lambda n: (n.type, n.id)):
            print(f"  [{node.type}] {node.id} — {node.name.replace('_', ' ')}")
        print()
        print("PAPER CHUNKS RETRIEVED")
        seen_dois: set[str] = set()
        for chunk in result.chunks_used:
            if chunk.doi not in seen_dois:
                seen_dois.add(chunk.doi)
                print(f"  {chunk.citation} — \"{chunk.title}\"")
        print()


def _build_log_entry(query: str, result, config: dict, verification_warnings: list[str]) -> dict:
    return {
        "timestamp":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query":             query,
        "config":            config,
        "answer":            result.answer,
        "citations":         result.citation,
        "taxonomy_concepts": result.document,
        "answer_triples":    result.answer_triples,
        "nodes_retrieved":   [{"id": n.id, "name": n.name, "type": n.type} for n in result.nodes_used],
        "chunks_retrieved":  len(result.chunks_used),
        "retrieval_mode":    result.retrieval_mode,
        "model":             result.model,
        "warnings":          verification_warnings,
    }


def _write_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _run_and_print(
    pipeline,
    query: str,
    show_sources: bool,
    debug: bool = False,
    log_path: Path | None = None,
    log_config: dict | None = None,
) -> None:
    """Run the pipeline, print the result, then print any verification warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline.run(query, debug=debug)

    _print_result(result, show_sources)

    warning_messages: list[str] = []
    if caught:
        print("Verification warnings:")
        for w in caught:
            print(f"  ⚠  {w.message}")
            warning_messages.append(str(w.message))
        print()

    if log_path is not None:
        _write_log(log_path, _build_log_entry(query, result, log_config or {}, warning_messages))


def _interactive(
    pipeline,
    show_sources: bool,
    debug: bool = False,
    log_path: Path | None = None,
    log_config: dict | None = None,
) -> None:
    mode = pipeline.retrieval_mode
    model = pipeline._backend.model_id
    print(f"Neurohive Query Pipeline  [{mode} · {model}]")
    if log_path:
        print(f"Logging to {log_path}")
    print("Type your query and press Enter.  'quit' or Ctrl-C to exit.\n")
    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break
        _run_and_print(pipeline, query, show_sources, debug=debug,
                       log_path=log_path, log_config=log_config)


def main() -> None:
    from neurohive.config import load_config  # noqa: PLC0415
    cfg = load_config()
    data_dir = _repo_path(cfg["paths"]["data_dir"])
    env_path = _repo_path(cfg["paths"]["env_file"])
    logs_dir = _repo_path(cfg["paths"]["logs_dir"])
    db_path = data_dir / "database" / "neurohive.db"

    _load_env(env_path)

    parser = argparse.ArgumentParser(
        description="Query the Neurohive neuroscience knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "query", nargs="?",
        help="Question to answer (omit for interactive REPL mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Anthropic model ID to use (also reads ANTHROPIC_MODEL env var). "
            "Default: config.toml [anthropic] model."
        ),
    )
    parser.add_argument(
        "--ollama",
        nargs="?",
        const="",
        default=None,
        metavar="MODEL",
        help=(
            "Use local Ollama instead of Anthropic. Optionally pass a model name. "
            "Default model: config.toml [ollama] model."
        ),
    )
    parser.add_argument(
        "--node-top-k", type=int, default=None, metavar="N",
        help="Number of taxonomy nodes retrieved per query. Default: config.toml [pipeline] node_top_k.",
    )
    parser.add_argument(
        "--chunk-top-k", type=int, default=None, metavar="N",
        help="Number of paper chunks retrieved per query. Default: config.toml [pipeline] chunk_top_k.",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=None, metavar="F",
        help=(
            "Minimum edge confidence for graph expansion (0.0–1.0). "
            "Default: config.toml [pipeline] confidence_threshold."
        ),
    )
    parser.add_argument(
        "--embeddings-model", default=None, metavar="MODEL",
        help=(
            "HuggingFace model ID for dense retrieval "
            "(e.g. sentence-transformers/all-MiniLM-L6-v2). "
            "The model must already be downloaded under config.toml [paths] models_dir. "
            "Default: config.toml [embeddings] model."
        ),
    )
    parser.add_argument(
        "--nli-model", default=None, metavar="MODEL",
        help=(
            "HuggingFace model ID for NLI entailment checking "
            "(e.g. cross-encoder/nli-deberta-v3-small). "
            "Used only with --verify; model must be downloaded under config.toml [paths] models_dir. "
            "Default: config.toml [nli] model."
        ),
    )
    parser.add_argument(
        "--entailment-threshold", type=float, default=None, metavar="F",
        help=(
            "Minimum post-softmax entailment probability for a claim to pass NLI verification. "
            "Used only with --verify. Default: config.toml [nli] entailment_threshold."
        ),
    )
    parser.add_argument(
        "--show-sources", action="store_true",
        help="Print retrieved nodes and chunks after the response",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help=(
            "Enable post-generation NLI entailment verification. "
            "Requires: make setup-nli && make download-nli-model."
        ),
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help=(
            "Re-rank retrieved chunks with the cross-encoder before generation. "
            "Retrieves pool_multiplier × chunk_top_k candidates then keeps the "
            "top chunk_top_k by entailment score. "
            "Requires: make setup-nli && make download-nli-model."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable the query result cache for this run.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print the raw model output (triples before verification) for each query.",
    )
    parser.add_argument(
        "--log", action="store_true",
        help=(
            "Append a structured JSONL log entry for each query to "
            "logs/YYYY-MM-DD.jsonl (created automatically)."
        ),
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help=(
            "Build (or rebuild) the knowledge-base database from raw source files "
            "and exit.  Run this after adding or editing files in data/raw/."
        ),
    )
    parser.add_argument(
        "--add", metavar="FILE",
        help=(
            "Incrementally add records from FILE to the live database and exit. "
            "Accepts: *.json (paper chunks) or *.csv (taxonomy nodes). "
            "Duplicate records are skipped; retrieval caches are invalidated."
        ),
    )
    parser.add_argument(
        "--snapshot-drift", action="store_true",
        help="Save the current corpus state as the drift baseline and exit.",
    )
    parser.add_argument(
        "--check-drift", action="store_true",
        help="Compare the current corpus against the saved baseline and exit.",
    )
    args = parser.parse_args()

    from neurohive.knowledge_base import KnowledgeBase  # noqa: PLC0415

    # Resolve the embeddings model directory early — needed by both --ingest and normal run.
    models_root = _repo_path(cfg["paths"]["models_dir"])
    embeddings_model_dir = (models_root / args.embeddings_model.split("/")[-1]) if args.embeddings_model else None
    nli_model_dir        = (models_root / args.nli_model.split("/")[-1])        if args.nli_model        else None

    if args.ingest:
        from neurohive.retrieval import Retriever  # noqa: PLC0415
        kb = KnowledgeBase(data_dir, verbose=True, rebuild=True)
        print("Preparing retrieval caches ...", flush=True)
        r = Retriever(kb, model_dir=embeddings_model_dir, cache_dir=kb.cache_dir)
        print(f"BM25 indices cached to {kb.cache_dir}/")
        if r.is_hybrid:
            print(f"Embeddings cached to {kb.cache_dir}/")
        else:
            print("Embeddings skipped (no embedding model - run `make setup-embeddings` for hybrid retrieval)")
        return

    if args.add:
        from neurohive.ingestor import IncrementalIngestor  # noqa: PLC0415
        kb = KnowledgeBase(data_dir)
        ingestor = IncrementalIngestor(kb, config=cfg)
        result = ingestor.add_from_file(args.add)
        print(result)
        if result.cache_invalidated:
            print("\nRun 'python main.py --ingest' to rebuild retrieval indices.")
        return

    if args.snapshot_drift:
        from neurohive.drift import DriftDetector  # noqa: PLC0415
        kb = KnowledgeBase(data_dir)
        dcfg = cfg.get("drift", {})
        baseline_path = data_dir / "database" / dcfg.get("baseline_file", "drift_baseline.json")
        detector = DriftDetector(kb, cache_dir=kb.cache_dir, baseline_path=baseline_path)
        saved = detector.save_baseline()
        print(f"Drift baseline saved to {saved}")
        return

    if args.check_drift:
        from neurohive.drift import DriftDetector  # noqa: PLC0415
        kb = KnowledgeBase(data_dir)
        dcfg = cfg.get("drift", {})
        baseline_path = data_dir / "database" / dcfg.get("baseline_file", "drift_baseline.json")
        thresholds = {
            "vocab_warning":  dcfg.get("vocab_warning_threshold", 0.05),
            "vocab_alert":    dcfg.get("vocab_alert_threshold",   0.15),
            "volume_warning": dcfg.get("volume_warning_pct",      0.20),
            "volume_alert":   dcfg.get("volume_alert_pct",        0.50),
            "emb_warning":    dcfg.get("embedding_warning_dist",  0.10),
            "emb_alert":      dcfg.get("embedding_alert_dist",    0.25),
        }
        detector = DriftDetector(
            kb, cache_dir=kb.cache_dir,
            baseline_path=baseline_path, thresholds=thresholds,
        )
        if not detector.baseline_exists():
            print("No baseline found. Run 'python main.py --snapshot-drift' first.")
            return
        report = detector.check()
        print(report)
        return

    # Auto-build on first run (database absent) before loading the full pipeline.
    if not db_path.exists():
        KnowledgeBase(data_dir, verbose=True)

    if args.ollama is not None:
        from neurohive.backends import OllamaBackend  # noqa: PLC0415
        ollama_model = args.ollama or cfg["ollama"]["model"]
        backend = OllamaBackend(model=ollama_model, host=cfg["ollama"]["host"])
        model = backend.model_id
    else:
        from neurohive.backends import AnthropicBackend  # noqa: PLC0415
        _check_api_key()
        model = args.model or os.environ.get("ANTHROPIC_MODEL") or cfg["anthropic"]["model"]
        backend = AnthropicBackend(model=model)

    node_top_k          = args.node_top_k          if args.node_top_k          is not None else cfg["pipeline"]["node_top_k"]
    chunk_top_k         = args.chunk_top_k         if args.chunk_top_k         is not None else cfg["pipeline"]["chunk_top_k"]
    confidence_threshold = args.confidence_threshold if args.confidence_threshold is not None else cfg["pipeline"]["confidence_threshold"]

    from neurohive.pipeline import QueryPipeline  # noqa: PLC0415
    pipeline = QueryPipeline(
        data_dir=data_dir,
        backend=backend,
        node_top_k=node_top_k,
        chunk_top_k=chunk_top_k,
        confidence_threshold=confidence_threshold,
        verify_entailment=args.verify,
        rerank=args.rerank,
        nli_model_dir=nli_model_dir,
        entailment_threshold=args.entailment_threshold,
        model_dir=embeddings_model_dir,
        cache_enabled=False if args.no_cache else None,
    )

    log_path = (logs_dir / f"{datetime.date.today().isoformat()}.jsonl") if args.log else None
    log_config = {
        "model":                model,
        "node_top_k":           node_top_k,
        "chunk_top_k":          chunk_top_k,
        "confidence_threshold": confidence_threshold,
        "generation_backend":   "ollama" if args.ollama is not None else "anthropic",
        "verify_entailment":    args.verify,
        "entailment_threshold": args.entailment_threshold,
    } if args.log else None

    if args.query:
        _run_and_print(pipeline, args.query, args.show_sources, debug=args.debug,
                       log_path=log_path, log_config=log_config)
    else:
        _interactive(pipeline, args.show_sources, debug=args.debug,
                     log_path=log_path, log_config=log_config)


if __name__ == "__main__":
    main()
