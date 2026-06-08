"""
Data drift detection for the Neurohive knowledge base.

Monitors the corpus for distribution shift by comparing a current snapshot
against a saved baseline. Detects changes in:

  - Corpus volume    — node / edge / chunk / paper counts
  - Vocabulary       — Jensen-Shannon divergence over chunk-text tokens
  - Type structure   — new node or edge types, shifted type distributions
  - Paper recency    — temporal shift in publication year distribution
  - Embedding space  — cosine distance between corpus centroids (needs numpy)

Quick start
-----------
    from neurohive.knowledge_base import KnowledgeBase
    from neurohive.drift import DriftDetector

    kb = KnowledgeBase("data")
    detector = DriftDetector(kb, cache_dir=kb.cache_dir)

    # Save the current state as baseline (once, after initial ingestion).
    detector.save_baseline()

    # After adding new data, check for drift:
    report = detector.check()
    print(report)
    if report.status == "alert":
        # Investigate, then accept the new distribution as the next baseline.
        detector.save_baseline()

Thresholds are configurable in config.toml under [drift].
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from neurohive.knowledge_base import KnowledgeBase

DriftStatus = Literal["ok", "warning", "alert"]

# ── Default thresholds ──────────────────────────────────────────────────────
_DEFAULT_VOCAB_WARNING  = 0.05   # Jensen-Shannon divergence
_DEFAULT_VOCAB_ALERT    = 0.15
_DEFAULT_VOLUME_WARNING = 0.20   # fractional change
_DEFAULT_VOLUME_ALERT   = 0.50
_DEFAULT_EMB_WARNING    = 0.10   # cosine distance
_DEFAULT_EMB_ALERT      = 0.25
_VOCAB_TOP_N            = 5_000  # terms kept in snapshot


# ── Snapshot ────────────────────────────────────────────────────────────────

@dataclass
class DriftSnapshot:
    """A point-in-time fingerprint of the corpus."""

    timestamp: str                              # ISO-8601 UTC
    n_nodes: int
    n_edges: int
    n_chunks: int
    n_papers: int
    node_type_counts: dict[str, int]            # e.g. {"Pillar": 6, ...}
    edge_type_counts: dict[str, int]            # e.g. {"RELATED_TO": 65, ...}
    paper_year_stats: dict[str, float]          # min, max, mean, median
    vocab: dict[str, int]                       # top-N term → raw count
    vocab_total: int                            # total tokens in corpus
    chunk_emb_centroid: list[float] | None = None
    node_emb_centroid:  list[float] | None = None
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> DriftSnapshot:
        d = json.loads(text)
        return cls(**d)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DriftSnapshot:
        return cls.from_json(path.read_text(encoding="utf-8"))


# ── Report ───────────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """Result of comparing the current corpus against a saved baseline."""

    status: DriftStatus
    baseline_timestamp: str
    current_timestamp: str
    findings: list[str] = field(default_factory=list)
    volume_changes: dict[str, dict] = field(default_factory=dict)
    vocab_divergence: float | None = None
    new_node_types: list[str] = field(default_factory=list)
    new_edge_types: list[str] = field(default_factory=list)
    emb_centroid_distance: float | None = None

    def __str__(self) -> str:
        icon = {"ok": "✓", "warning": "⚠", "alert": "✗"}[self.status]
        lines = [
            f"Drift status: {self.status.upper()}  {icon}",
            f"  Baseline : {self.baseline_timestamp}",
            f"  Current  : {self.current_timestamp}",
        ]
        if self.findings:
            lines.append("")
            lines.append("Findings:")
            for f in self.findings:
                lines.append(f"  {f}")
        if self.volume_changes:
            lines.append("")
            lines.append("Volume changes:")
            for metric, info in self.volume_changes.items():
                pct = info.get("pct_change", 0)
                sign = "+" if pct >= 0 else ""
                lines.append(
                    f"  {metric:<12} {info['baseline']:>6} → {info['current']:>6}"
                    f"  ({sign}{pct:.1%})"
                )
        if self.vocab_divergence is not None:
            lines.append(f"\nVocabulary Jensen-Shannon divergence : {self.vocab_divergence:.4f}")
        if self.emb_centroid_distance is not None:
            lines.append(f"Embedding centroid dist  : {self.emb_centroid_distance:.4f}")
        return "\n".join(lines)


# ── Detector ─────────────────────────────────────────────────────────────────

class DriftDetector:
    """
    Computes corpus snapshots and compares them to detect data drift.

    Parameters
    ----------
    kb         : Live KnowledgeBase instance.
    cache_dir  : Directory containing embedding cache files (.npy).
                 Defaults to ``kb.cache_dir``.
    baseline_path : Where to read/write the baseline snapshot JSON.
                    Defaults to ``<cache_dir>/drift_baseline.json``.
    thresholds : Optional dict overriding any of the default drift thresholds.
                 Keys: vocab_warning, vocab_alert, volume_warning, volume_alert,
                       emb_warning, emb_alert.
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        cache_dir: Path | None = None,
        baseline_path: Path | None = None,
        thresholds: dict | None = None,
    ):
        self._kb = kb
        self._conn: sqlite3.Connection = kb._conn
        self._cache_dir = cache_dir or kb.cache_dir
        self._baseline_path = baseline_path or (self._cache_dir / "drift_baseline.json")

        th = thresholds or {}
        self._vocab_warning  = th.get("vocab_warning",  _DEFAULT_VOCAB_WARNING)
        self._vocab_alert    = th.get("vocab_alert",    _DEFAULT_VOCAB_ALERT)
        self._vol_warning    = th.get("volume_warning", _DEFAULT_VOLUME_WARNING)
        self._vol_alert      = th.get("volume_alert",   _DEFAULT_VOLUME_ALERT)
        self._emb_warning    = th.get("emb_warning",    _DEFAULT_EMB_WARNING)
        self._emb_alert      = th.get("emb_alert",      _DEFAULT_EMB_ALERT)

    # ── Public API ──────────────────────────────────────────────────────────

    def snapshot(self) -> DriftSnapshot:
        """Capture a snapshot of the current corpus state."""
        conn = self._conn

        n_nodes  = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        n_edges  = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_papers = conn.execute("SELECT COUNT(DISTINCT doi) FROM chunks").fetchone()[0]

        node_type_counts = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT type, COUNT(*) FROM nodes GROUP BY type"
            ).fetchall()
        }
        edge_type_counts = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT relationship_type, COUNT(*) FROM edges GROUP BY relationship_type"
            ).fetchall()
        }

        year_rows = [
            r[0]
            for r in conn.execute("SELECT year FROM chunks WHERE year IS NOT NULL").fetchall()
        ]
        paper_year_stats = _year_stats(year_rows)

        texts = [
            r[0]
            for r in conn.execute("SELECT text FROM chunks").fetchall()
        ]
        vocab, vocab_total = _build_vocab(texts)

        chunk_emb_centroid = _load_centroid(self._cache_dir / "chunk_embs.npy")
        node_emb_centroid  = _load_centroid(self._cache_dir / "node_embs.npy")

        return DriftSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_nodes=n_nodes,
            n_edges=n_edges,
            n_chunks=n_chunks,
            n_papers=n_papers,
            node_type_counts=node_type_counts,
            edge_type_counts=edge_type_counts,
            paper_year_stats=paper_year_stats,
            vocab=vocab,
            vocab_total=vocab_total,
            chunk_emb_centroid=chunk_emb_centroid,
            node_emb_centroid=node_emb_centroid,
        )

    def save_baseline(self, path: Path | None = None) -> Path:
        """
        Take a fresh snapshot and save it as the drift baseline.

        Returns the path where the baseline was written.
        """
        snap = self.snapshot()
        dest = path or self._baseline_path
        snap.save(dest)
        return dest

    def load_baseline(self, path: Path | None = None) -> DriftSnapshot:
        """Load the saved baseline snapshot."""
        src = path or self._baseline_path
        if not src.exists():
            raise FileNotFoundError(
                f"No drift baseline found at {src}. "
                "Run detector.save_baseline() first."
            )
        return DriftSnapshot.load(src)

    def check(self, baseline: DriftSnapshot | None = None) -> DriftReport:
        """
        Compare the current corpus against *baseline* and return a DriftReport.

        If *baseline* is None the saved baseline file is loaded automatically.
        """
        if baseline is None:
            baseline = self.load_baseline()

        current = self.snapshot()
        findings: list[str] = []
        status: DriftStatus = "ok"

        def _upgrade(new: DriftStatus) -> None:
            nonlocal status
            if new == "alert" or (new == "warning" and status == "ok"):
                status = new

        # ── Volume drift ───────────────────────────────────────────────────
        volume_changes: dict[str, dict] = {}
        for metric, b_val, c_val in [
            ("nodes",  baseline.n_nodes,  current.n_nodes),
            ("edges",  baseline.n_edges,  current.n_edges),
            ("chunks", baseline.n_chunks, current.n_chunks),
            ("papers", baseline.n_papers, current.n_papers),
        ]:
            if b_val == 0:
                pct = float("inf") if c_val else 0.0
            else:
                pct = (c_val - b_val) / b_val
            volume_changes[metric] = {
                "baseline": b_val, "current": c_val, "pct_change": pct
            }
            abs_pct = abs(pct)
            if abs_pct >= self._vol_alert:
                _upgrade("alert")
                findings.append(
                    f"ALERT: {metric} count changed by {pct:+.1%}"
                    f" ({b_val} → {c_val})"
                )
            elif abs_pct >= self._vol_warning:
                _upgrade("warning")
                findings.append(
                    f"WARNING: {metric} count changed by {pct:+.1%}"
                    f" ({b_val} → {c_val})"
                )

        # ── Vocabulary drift ───────────────────────────────────────────────
        vocab_divergence: float | None = None
        if baseline.vocab and current.vocab:
            b_dist = _to_distribution(baseline.vocab, baseline.vocab_total)
            c_dist = _to_distribution(current.vocab, current.vocab_total)
            vocab_divergence = _js_divergence(b_dist, c_dist)
            if vocab_divergence >= self._vocab_alert:
                _upgrade("alert")
                findings.append(
                    f"ALERT: vocabulary distribution shifted significantly"
                    f" (JS={vocab_divergence:.4f} ≥ {self._vocab_alert})"
                )
            elif vocab_divergence >= self._vocab_warning:
                _upgrade("warning")
                findings.append(
                    f"WARNING: vocabulary distribution shifted"
                    f" (JS={vocab_divergence:.4f} ≥ {self._vocab_warning})"
                )

        # ── Type structure drift ───────────────────────────────────────────
        new_node_types = sorted(
            set(current.node_type_counts) - set(baseline.node_type_counts)
        )
        new_edge_types = sorted(
            set(current.edge_type_counts) - set(baseline.edge_type_counts)
        )
        if new_node_types:
            _upgrade("warning")
            findings.append(f"WARNING: new node type(s) detected: {new_node_types}")
        if new_edge_types:
            _upgrade("warning")
            findings.append(f"WARNING: new edge type(s) detected: {new_edge_types}")

        # ── Paper year drift ───────────────────────────────────────────────
        b_median = baseline.paper_year_stats.get("median")
        c_median = current.paper_year_stats.get("median")
        if b_median is not None and c_median is not None:
            year_shift = abs(c_median - b_median)
            if year_shift >= 5:
                _upgrade("warning")
                findings.append(
                    f"WARNING: median paper year shifted by {year_shift:.0f} years"
                    f" ({b_median:.0f} → {c_median:.0f})"
                )

        # ── Embedding centroid drift ───────────────────────────────────────
        emb_dist: float | None = None
        if baseline.chunk_emb_centroid and current.chunk_emb_centroid:
            if len(baseline.chunk_emb_centroid) == len(current.chunk_emb_centroid):
                emb_dist = _cosine_distance(
                    baseline.chunk_emb_centroid, current.chunk_emb_centroid
                )
                if emb_dist >= self._emb_alert:
                    _upgrade("alert")
                    findings.append(
                        f"ALERT: chunk embedding centroid shifted"
                        f" (cosine dist={emb_dist:.4f} ≥ {self._emb_alert})"
                    )
                elif emb_dist >= self._emb_warning:
                    _upgrade("warning")
                    findings.append(
                        f"WARNING: chunk embedding centroid shifted"
                        f" (cosine dist={emb_dist:.4f} ≥ {self._emb_warning})"
                    )

        if not findings:
            findings.append("No significant drift detected.")

        return DriftReport(
            status=status,
            baseline_timestamp=baseline.timestamp,
            current_timestamp=current.timestamp,
            findings=findings,
            volume_changes=volume_changes,
            vocab_divergence=vocab_divergence,
            new_node_types=new_node_types,
            new_edge_types=new_edge_types,
            emb_centroid_distance=emb_dist,
        )

    def baseline_exists(self) -> bool:
        return self._baseline_path.exists()


# ── Internal utilities ───────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _build_vocab(texts: list[str]) -> tuple[dict[str, int], int]:
    """Return top-N term counts and total token count for a list of texts."""
    counts: dict[str, int] = {}
    total = 0
    for text in texts:
        for tok in _tokenize(text):
            counts[tok] = counts.get(tok, 0) + 1
            total += 1
    if len(counts) > _VOCAB_TOP_N:
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:_VOCAB_TOP_N]
        counts = dict(top)
    return counts, total


def _to_distribution(vocab: dict[str, int], total: int) -> dict[str, float]:
    if total == 0:
        return {}
    return {k: v / total for k, v in vocab.items()}


def _js_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two sparse probability distributions."""
    keys = set(p) | set(q)
    js = 0.0
    for k in keys:
        p_k = p.get(k, 0.0)
        q_k = q.get(k, 0.0)
        m_k = (p_k + q_k) / 2.0
        if m_k > 0:
            if p_k > 0:
                js += 0.5 * p_k * math.log(p_k / m_k)
            if q_k > 0:
                js += 0.5 * q_k * math.log(q_k / m_k)
    return js


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


def _year_stats(years: list[int]) -> dict[str, float]:
    if not years:
        return {}
    s = sorted(years)
    n = len(s)
    mid = n // 2
    median = s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
    return {
        "min":    float(s[0]),
        "max":    float(s[-1]),
        "mean":   sum(s) / n,
        "median": float(median),
    }


def _load_centroid(path: Path) -> list[float] | None:
    """Load a numpy embedding file and return its column-wise mean as a list."""
    if not path.exists():
        return None
    try:
        import numpy as np  # noqa: PLC0415
        embs = np.load(str(path))
        if embs.ndim != 2 or embs.shape[0] == 0:
            return None
        centroid = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 0:
            centroid = centroid / norm
        return centroid.tolist()
    except Exception:
        return None
