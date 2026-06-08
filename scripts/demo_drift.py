"""
Demo: incremental ingestion and data-drift detection.

This script walks through the full workflow:

  1. Open the existing knowledge base.
  2. Save a drift baseline.
  3. Simulate adding new paper chunks.
  4. Check for drift — should report growth in volume.
  5. Accept the new state as the updated baseline.

Run from the repo root:
    uv run python scripts/demo_drift.py
"""
from __future__ import annotations

from pathlib import Path

from neurohive.config import load_config
from neurohive.drift import DriftDetector
from neurohive.entities import PaperChunk
from neurohive.ingestor import IncrementalIngestor
from neurohive.knowledge_base import KnowledgeBase

_REPO = Path(__file__).parent.parent
_SEP  = "─" * 60


def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def main() -> None:
    cfg      = load_config()
    data_dir = _REPO / cfg["paths"]["data_dir"]
    dcfg     = cfg.get("drift", {})

    baseline_path = data_dir / "database" / dcfg.get("baseline_file", "drift_baseline.json")
    thresholds = {
        "vocab_warning":  dcfg.get("vocab_warning_threshold", 0.05),
        "vocab_alert":    dcfg.get("vocab_alert_threshold",   0.15),
        "volume_warning": dcfg.get("volume_warning_pct",      0.20),
        "volume_alert":   dcfg.get("volume_alert_pct",        0.50),
        "emb_warning":    dcfg.get("embedding_warning_dist",  0.10),
        "emb_alert":      dcfg.get("embedding_alert_dist",    0.25),
    }

    # ── Step 1: open knowledge base ──────────────────────────────────────────
    _header("Step 1 · Open knowledge base")
    kb = KnowledgeBase(data_dir)

    # Remove any synthetic chunks left by a previous demo run so the demo is idempotent.
    _DEMO_DOIS = ("10.9999/demo.2024.001", "10.9999/demo.2024.002")
    with kb._conn:
        kb._conn.execute(
            "DELETE FROM chunk_nodes WHERE chunk_id IN "
            "(SELECT id FROM chunks WHERE doi IN (?,?))",
            _DEMO_DOIS,
        )
        kb._conn.execute(
            "DELETE FROM chunks WHERE doi IN (?,?)", _DEMO_DOIS
        )
    kb._chunks_cache = None  # clear in-memory cache

    print(f"  Nodes : {len(kb.nodes)}")
    print(f"  Edges : {len(kb.edges)}")
    print(f"  Chunks: {len(kb.chunks)}")

    # ── Step 2: save baseline ────────────────────────────────────────────────
    _header("Step 2 · Save drift baseline")
    detector = DriftDetector(
        kb, cache_dir=kb.cache_dir,
        baseline_path=baseline_path,
        thresholds=thresholds,
    )
    saved = detector.save_baseline()
    print(f"  Baseline written to: {saved}")

    # ── Step 3: simulate adding new chunks ───────────────────────────────────
    _header("Step 3 · Add new paper chunks (incremental ingestion)")

    # We use a handful of synthetic chunks to avoid needing real new papers.
    # In production, replace this list with PaperChunk.load_all("new_papers.json").
    synthetic_chunks = [
        PaperChunk(
            id=9001,
            doi="10.9999/demo.2024.001",
            title="Advances in Cortical Oscillation Research",
            authors="Demo, A.; Example, B.",
            year=2024,
            chunk_index=0,
            text=(
                "Recent advances in cortical oscillation research have revealed new mechanisms "
                "underlying gamma-band synchrony. Coupled with evidence from LFP recordings, "
                "these findings challenge earlier inhibitory interneuron models."
            ),
            taxonomy_node_ids=("P001",),  # links to an existing node
        ),
        PaperChunk(
            id=9002,
            doi="10.9999/demo.2024.001",
            title="Advances in Cortical Oscillation Research",
            authors="Demo, A.; Example, B.",
            year=2024,
            chunk_index=1,
            text=(
                "Computational models incorporating adaptive threshold dynamics can replicate "
                "the observed spectral peaks without requiring extensive parameter tuning."
            ),
            taxonomy_node_ids=("P001",),
        ),
        PaperChunk(
            id=9003,
            doi="10.9999/demo.2024.002",
            title="Synaptic Tagging and Memory Consolidation",
            authors="Smith, C.; Jones, D.; Williams, E.",
            year=2023,
            chunk_index=0,
            text=(
                "The synaptic tagging and capture hypothesis provides a molecular substrate "
                "for late-phase long-term potentiation, bridging early protein-synthesis-independent "
                "and late protein-synthesis-dependent phases."
            ),
            taxonomy_node_ids=(),
        ),
    ]

    ingestor = IncrementalIngestor(kb, config=cfg)
    result = ingestor.add_chunks(synthetic_chunks, source="demo_drift.py / synthetic")
    print(result)

    # ── Step 4: check for drift ──────────────────────────────────────────────
    _header("Step 4 · Check drift against baseline")
    report = detector.check()
    print(report)

    # ── Step 5: update baseline ──────────────────────────────────────────────
    _header("Step 5 · Accept new state as updated baseline")
    new_saved = detector.save_baseline()
    print(f"  Updated baseline written to: {new_saved}")

    # ── Step 6: confirm no drift after re-check ──────────────────────────────
    _header("Step 6 · Re-check drift (should be clean)")
    report2 = detector.check()
    print(report2)

    # ── Done ─────────────────────────────────────────────────────────────────
    _header("Done")
    print(
        "  In production:\n"
        "    1. python main.py --snapshot-drift        # save baseline\n"
        "    2. python main.py --add new_papers.json   # ingest new data\n"
        "    3. python main.py --ingest                # rebuild retrieval indices\n"
        "    4. python main.py --check-drift           # review drift report\n"
        "    5. python main.py --snapshot-drift        # accept as new baseline"
    )


if __name__ == "__main__":
    main()
