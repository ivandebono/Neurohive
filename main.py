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

    # Override model
    python main.py --model claude-sonnet-4-6 "query..."

    # Hybrid retrieval is enabled automatically when models/ exists:
    #   uv sync --extra embeddings
    #   uv run python scripts/download_model.py

    # Show retrieved sources
    python main.py --show-sources "What are the main theories of synaptic plasticity?"

.env keys
---------
    ANTHROPIC_API_KEY   required
    ANTHROPIC_MODEL     optional model override, e.g. claude-sonnet-4-6

config.toml
-----------
    Provides defaults for all pipeline parameters.  Every config.toml value
    can be overridden at query time via CLI flags (listed below).  Priority
    order (highest first): CLI flag → env var → config.toml → hard-coded default.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ENV_PATH = Path(__file__).parent / ".env"


def _load_env() -> None:
    """Load key=value pairs from .env into os.environ (does not overwrite)."""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text().splitlines():
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


def _run_and_print(pipeline, query: str, show_sources: bool, debug: bool = False) -> None:
    """Run the pipeline, print the result, then print any verification warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = pipeline.run(query, debug=debug)

    _print_result(result, show_sources)

    if caught:
        print("Verification warnings:")
        for w in caught:
            print(f"  ⚠  {w.message}")
        print()


def _interactive(pipeline, show_sources: bool, debug: bool = False) -> None:
    mode = pipeline.retrieval_mode
    model = pipeline._backend.model_id
    print(f"Neurohive Query Pipeline  [{mode} · {model}]")
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
        _run_and_print(pipeline, query, show_sources, debug=debug)


def main() -> None:
    _load_env()

    from neurohive.config import load_config  # noqa: PLC0415
    cfg = load_config()

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
            "The model must already be downloaded under models/. "
            "Default: config.toml [embeddings] model."
        ),
    )
    parser.add_argument(
        "--nli-model", default=None, metavar="MODEL",
        help=(
            "HuggingFace model ID for NLI entailment checking "
            "(e.g. cross-encoder/nli-deberta-v3-small). "
            "Used only with --verify; model must be downloaded under models/. "
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
        "--debug", action="store_true",
        help="Print the raw model output (triples before verification) for each query.",
    )
    args = parser.parse_args()

    from neurohive.backends import AnthropicBackend  # noqa: PLC0415

    _check_api_key()
    model = args.model or os.environ.get("ANTHROPIC_MODEL") or cfg["anthropic"]["model"]
    backend = AnthropicBackend(model=model)

    node_top_k          = args.node_top_k          if args.node_top_k          is not None else cfg["pipeline"]["node_top_k"]
    chunk_top_k         = args.chunk_top_k         if args.chunk_top_k         is not None else cfg["pipeline"]["chunk_top_k"]
    confidence_threshold = args.confidence_threshold if args.confidence_threshold is not None else cfg["pipeline"]["confidence_threshold"]

    # Derive local model directory from a HuggingFace model ID (last path component).
    models_root = Path(__file__).parent / "models"
    embeddings_model_dir = (models_root / args.embeddings_model.split("/")[-1]) if args.embeddings_model else None
    nli_model_dir        = (models_root / args.nli_model.split("/")[-1])        if args.nli_model        else None

    from neurohive.pipeline import QueryPipeline  # noqa: PLC0415
    pipeline = QueryPipeline(
        data_dir=DATA_DIR,
        backend=backend,
        node_top_k=node_top_k,
        chunk_top_k=chunk_top_k,
        confidence_threshold=confidence_threshold,
        verify_entailment=args.verify,
        nli_model_dir=nli_model_dir,
        entailment_threshold=args.entailment_threshold,
        model_dir=embeddings_model_dir,
    )

    if args.query:
        _run_and_print(pipeline, args.query, args.show_sources, debug=args.debug)
    else:
        _interactive(pipeline, args.show_sources, debug=args.debug)


if __name__ == "__main__":
    main()
