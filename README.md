# Neurohive — Neuroscience Knowledge Query Pipeline

**Author:** Ivan Debono

A retrieval-augmented generation (RAG) pipeline that takes a natural language question and returns a structured, evidence-grounded answer drawn exclusively from a curated neuroscience knowledge store. Every factual claim is traceable to a specific taxonomy concept or peer-reviewed paper excerpt. Sources that do not appear in the retrieved context are stripped before the response is returned.

The knowledge store is a live SQLite database. New papers, taxonomy nodes, and edges can be added incrementally at runtime without a full rebuild. A built-in drift detector monitors the corpus for distribution shift and raises warnings when the data departs significantly from a saved baseline.

---

## My Approach

The task is to build a system that gives *useful and grounded* answers about neuroscience using a fixed knowledge store: a taxonomy graph of 92 concepts connected by 130 typed edges, and 83 paper chunks from 24 peer-reviewed publications. The core challenge is retrieving the right context and ensuring the language model cannot go beyond it.

### 1. Retrieval

I use **BM25 Okapi** as the primary retrieval method. BM25 is well-suited here because the knowledge base is domain-specific: terminology like "voltage-gated channel kinetics" or "spike-timing-dependent plasticity" appears verbatim in both queries and documents. The BM25 implementation is pure Python (standard library only), so the pipeline runs with no retrieval dependency by default.

I run two independent retrieval passes — one over taxonomy nodes, one over paper chunks — so the system can find relevant concepts and relevant papers independently.

The retriever also exploits structure from the paper data. Paper chunks are indexed with their linked taxonomy node IDs plus the linked node names, types, and descriptions, so a query that matches taxonomy language can retrieve the right paper chunk even when the phrase is absent from the chunk text. Retrieved chunks then contribute their `taxonomy_node_ids` to graph expansion, meaning paper evidence can pull in the right taxonomy neighbourhood. Chunks from the same DOI receive a context boost that decays with `chunk_index` distance so adjacent passages are preferred over distant ones from the same paper.

For users who install the optional `sentence-transformers` extra and download the embedding model, retrieval upgrades automatically to **BM25 + dense cosine similarity**, fused via Reciprocal Rank Fusion (RRF). This handles semantic paraphrasing that BM25 misses. The switch is invisible — the pipeline detects the model directory at startup.

BM25 indices and optional dense embeddings are cached under the configured database directory during `--ingest`. Subsequent runs load valid caches when the indexed corpus hash, retrieval parameters, and model path still match. Cache filenames are read from `config.toml [retrieval]`, not hardcoded.

### 2. Graph Expansion

Pure retrieval on a small knowledge base risks missing essential context. If a query retrieves "Long-Term Potentiation", the answer improves significantly when the model also sees the parent node ("Synaptic Plasticity"), the governing theory ("Hebbian Learning Rule"), and closely related areas ("NMDA Receptors").

`smart_expand()` traverses the taxonomy graph one hop from the seed set, adding parents, theories, dimensions, and lateral neighbours. Six of the seven edge types are directed; `RELATED_TO` is the only bidirectional edge and is therefore traversed in both incoming and outgoing directions. A **confidence score** on each edge gates expansion: weak connections are excluded. Algorithmically inferred edges (`map_source="semantic"`) are held to a stricter threshold (`confidence_threshold + 0.1`) to offset their higher noise compared to manually curated canonical edges. Edges that pass the threshold but are not at full confidence are annotated in the context so the model can hedge appropriately.

### 3. Prompt Design and Grounding

The prompt is structured in three layers:

1. **System prompt** — instructs the model to use only the supplied context and to explicitly state when the context is insufficient rather than speculating.
2. **User prompt** — the query plus two clearly delimited sections: `TAXONOMY CONTEXT` (the expanded graph neighbourhood, formatted as a hierarchy with breadcrumbs, human-readable edge labels such as "explains" and "is related to", and `[inferred]` tags on algorithmically-derived edges) and `PAPER EVIDENCE` (chunks grouped by paper, in reading order, with linked concept names).
3. **Structured output** — generation runs either through the Anthropic API or a local Ollama model. The Anthropic backend uses `tool_choice` to force the model into a fixed JSON schema (`answer`: an array of paper/taxonomy triples). The Ollama backend requests JSON mode and validates the returned schema before the result enters verification.

### 4. Citation Verification

The prompt constraint reduces hallucinations but cannot eliminate them. After generation, `_verify()` programmatically checks every citation and taxonomy reference in the model's output against the *actually retrieved* context. Anything that cannot be traced is stripped and a warning is emitted. The final answer is guaranteed to contain only sources that retrieval found.

### Design Choices

- **No external retrieval library**: BM25 is implemented directly with standard-library Python. It precomputes document lengths, IDF values, and inverted postings, then persists those indices during ingestion.
- **Dynamic knowledge base**: the SQLite store supports incremental ingestion — new chunks, nodes, and edges can be added at runtime without a full rebuild. Each ingestion batch is recorded in a `batches` provenance table with a timestamp and source. Semantic deduplication (optional) prevents near-duplicate chunks from entering the store.
- **Data drift detection**: a `DriftDetector` snapshots five corpus metrics (volume, vocabulary JS divergence, type distributions, paper year distribution, embedding centroid distance) and compares them against a saved baseline. Thresholds are configurable under `config.toml [drift]`.
- **Query result cache**: a SQLite-backed cache stores answers keyed on query text, model, retrieval settings, and a corpus fingerprint. Adding new data changes the fingerprint and automatically invalidates stale entries.
- **Cross-encoder re-ranking**: the retriever can fetch an oversized candidate pool and re-rank it with an NLI cross-encoder before generation, improving precision at the cost of latency.
- **Config validation and typed access**: `config.toml` is validated at load time via Pydantic v2. `load_config()` returns a typed `Config` object, so every parameter is accessed as an attribute (`cfg.pipeline.node_top_k`, `cfg.anthropic.model`, etc.) with the correct Python type — no dict subscripting or manual casting anywhere in the codebase. Invalid values (e.g. `confidence_threshold` outside 0–1, or `vocab_warning_threshold ≥ vocab_alert_threshold`) raise a `ValueError` before any query runs.
- **Optional hybrid upgrade**: users who want stronger retrieval can `make setup-embeddings`. The default works without it.
- **Generation backend choice**: use Anthropic for stronger API-hosted generation, or pass `--ollama` to use a local open-source model through Ollama.
- **Configuration in one place**: `config.toml` is the source for model names, runtime paths, retrieval tuning, cache settings, and drift detection thresholds. Query-level settings can also be overridden per query via CLI flags.

---

## The Final Prompt

Every query is sent to the language model using this structure.

**System prompt** (sent to Anthropic or Ollama; Anthropic also uses prompt caching):

```
You are a closed-context neuroscience query system. The TAXONOMY CONTEXT and
PAPER EVIDENCE sections in the user message are your complete and only source
of knowledge. You have no other knowledge available.

Absolute rules — every rule is a hard constraint, not a guideline:
1. Use ONLY facts that appear verbatim or by direct implication in the provided
   TAXONOMY CONTEXT or PAPER EVIDENCE. Nothing else.
2. Do NOT draw on your training data, even for background context, definitions,
   or facts you are certain are correct. If it is not in the provided context,
   it does not exist for this query.
3. If the context is insufficient to answer the query fully, produce at least one
   taxonomy triple using the most relevant concept available; set the claim to
   state explicitly what is covered and what information is absent. Do not fill
   gaps with training knowledge under any circumstances.
4. Express every factual claim as a structured triple of one of two types:
   • Paper triple  (drawn from PAPER EVIDENCE):
       type: "paper", claim: one sentence, quote: verbatim passage, citation: "Surname et al., Year"
   • Taxonomy triple  (drawn from TAXONOMY CONTEXT):
       type: "taxonomy", claim: one sentence, concept: exact concept name
   Every factual sentence must map to exactly one triple.
5. Do not speculate, infer beyond what the context states, or add qualifications
   drawn from general neuroscience knowledge.
```

**User prompt** (filled in per query):

```
Query: {query}

---
TAXONOMY CONTEXT
[Research_area] Neural Signaling → Electrical Signaling → Action Potential Propagation
  The active spread of an action potential along the axon ...
  explains → Hodgkin Huxley Model [confidence: 93%]
    (Hodgkin-Huxley equations predict the shape and velocity of propagating action potentials)
  is related to → Saltatory Conduction [confidence: 88%]

[Theory] Hodgkin Huxley Model
  A conductance-based mathematical model describing action potential generation ...
...

---
PAPER EVIDENCE
[Catterall et al., 2017] "Voltage-Gated Sodium Channels" DOI:10.1085/...
  Concepts: Action Potential Propagation, Voltage Gated Channel Kinetics
  • Voltage-gated sodium channels initiate action potentials by opening ...
...

---
Answer the query using ONLY the context above. Express every factual claim
as a paper or taxonomy triple. Do not use any knowledge outside these two sections.
Return only a JSON object with one top-level key, "answer".
```

**How the context sections are built:**

`TAXONOMY CONTEXT` is assembled in three passes for each query:

1. **Retrieval** — BM25 (or BM25 + dense via RRF) scores every node by how closely its name, type, description, and attached edge notes match the query. Paper chunks are scored separately using title, text, `chunk_index`, explicit `taxonomy_node_ids`, and the names/types/descriptions of linked taxonomy nodes.
2. **Seed construction** — the graph seed set combines the top-k retrieved taxonomy nodes with every taxonomy node ID attached to the top-k retrieved paper chunks.
3. **Graph expansion** — `smart_expand()` traverses one hop from the seed set, adding: parent Pillars and Subpillars (via `HAS_SUBPILLAR` / `HAS_RESEARCH_AREA`), Theory nodes that explain the seed concept (incoming `EXPLAINS`), Dimension nodes that characterise it (outgoing `HAS_DIMENSION`), and lateral neighbours in both directions (bidirectional `RELATED_TO`). Edges are followed only if `confidence ≥ threshold`; algorithmically-inferred edges (`map_source="semantic"`) are held to a stricter bar (`threshold + 0.1`).
4. **Formatting** — the expanded set is sorted by type depth (Pillar → Subpillar → Research_area → Theory → Dimension) and each node is rendered with its full breadcrumb ancestry path, its description, and every outgoing edge to another in-context node. Edge types are mapped to natural-language phrases (`EXPLAINS →` becomes `explains →`; `HAS_DIMENSION →` becomes `is characterised by →`). Edges with `confidence < 1.0` show their score. Algorithmically-derived edges (`map_source="semantic"`) are tagged `[inferred]`; other non-canonical sources show their source name. Nodes with no edges in either direction are tagged `[no connections]`.

`PAPER EVIDENCE` is built from two sources, then deduplicated:

1. **Direct retrieval** — the top-k chunks from the BM25/RRF pass over the paper corpus.
2. **Node-linked harvest** — every chunk whose `taxonomy_node_ids` list overlaps the expanded node set is added. This surfaces papers relevant to the expanded context even if they did not rank highly in direct retrieval.

Within each paper, chunks are sorted by `chunk_index` to preserve reading order. During retrieval, chunks from the same DOI are context-boosted with a score floor that decays by `chunk_index` distance from the initially retrieved seed chunk, so nearby paper context is favoured.

**Raw model output schema** (enforced by Anthropic `tool_choice`; requested and validated for Ollama):

```json
{
  "answer": [
    {"type": "paper",    "claim": "one-sentence statement", "quote": "verbatim passage", "citation": "Surname et al., Year"},
    {"type": "taxonomy", "claim": "one-sentence statement", "concept": "exact concept name"}
  ]
}
```

Each element is either a **paper triple** (grounded in PAPER EVIDENCE) or a **taxonomy triple** (grounded in TAXONOMY CONTEXT). `_verify()` checks every triple against the retrieved context before the answer is assembled.

---

## Knowledge Store

The pipeline is built over two data sources in `data/raw/`:

**Taxonomy graph** — 92 neuroscience concept nodes across five types (Pillar, Subpillar, Research_area, Theory, Dimension), connected by 130 typed edges. Each edge carries a `confidence` score (0–1) and a `map_source` field (`canonical` or `semantic`).

![Taxonomy Graph](assets/taxonomy_graph.png)

*Node colours: blue = Pillar, purple = Subpillar, green = Research\_area, amber = Theory, red = Dimension. Dashed edges = RELATED\_TO (bidirectional); solid = directed.*

**Paper corpus** — 83 text chunks from 24 peer-reviewed publications (2001–2022), each pre-linked to one or more taxonomy nodes via `taxonomy_node_ids`.

---

## Installation

**Requirements:** Python 3.13+, [`uv`](https://docs.astral.sh/uv/)

### Choose a generation backend

**Anthropic API** is the default runtime backend:

```bash
git clone <repo-url>
cd neurohive
make setup
uv run python main.py --ingest   # build database + retrieval caches from data/raw/
```

It requires an API key:

```bash
# Option A: environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# Option B: .env file at the repo root (loaded automatically)
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

**Local Ollama** avoids paid API calls and uses a configured open-source model (`ministral-3:3b` by default):

```bash
git clone <repo-url>
cd neurohive
make setup-ollama              # installs Python deps and pulls the configured Ollama model
uv run python main.py --ingest # build database + retrieval caches
```

You must install and start Ollama first. After setup, use `--ollama` at runtime.

### Optional: hybrid retrieval (BM25 + dense)

```bash
make setup-embeddings   # installs sentence-transformers and downloads the model (~90 MB)
```

Hybrid mode activates automatically on the next run when the configured model exists under `[paths] models_dir`. Run `uv run python main.py --ingest` after downloading to precompute and cache embeddings.

### Optional: NLI verification and re-ranking

The NLI cross-encoder is used for two independent features: **entailment verification** (`--verify`) and **chunk re-ranking** (`--rerank`). Both require the same model download.

```bash
make setup-nli   # installs sentence-transformers and downloads the cross-encoder (~568 MB)
```

### Optional: install everything at once

```bash
make setup-embeddings-nli          # embedding + NLI models (Anthropic generation)
make setup-ollama-embeddings-nli   # Ollama + embedding + NLI models
```

---

## How to Run

### First-run database and retrieval cache setup

On the very first run the knowledge-base database is built automatically from the raw files in the configured data directory. Trigger this explicitly to rebuild the database and refresh retrieval caches after adding raw data, changing chunk taxonomy links, changing retrieval parameters, or downloading an embedding model:

```bash
uv run python main.py --ingest
```

### Single query

```bash
uv run python main.py "What is the role of voltage-gated sodium channels in action potential initiation?"
```

### Interactive REPL

```bash
uv run python main.py
# Type your question and press Enter. 'quit' or Ctrl-C to exit.
```

The REPL header shows the active retrieval mode and model:

```
Neurohive Query Pipeline  [hybrid (BM25 + dense, RRF) · claude-haiku-4-5-20251001]
Type your query and press Enter.  'quit' or Ctrl-C to exit.
```

### Show retrieved sources alongside the answer

```bash
uv run python main.py --show-sources "How does long-term potentiation relate to NMDA receptors?"
```

This prints the taxonomy nodes and paper chunks that were retrieved, so you can trace exactly where each part of the answer came from.

### Use a different model

```bash
uv run python main.py --model claude-sonnet-4-6 "Explain inhibitory interneuron diversity."
```

The `ANTHROPIC_MODEL` environment variable can also override the default model without editing the config file.

### Use local Ollama instead of Anthropic

Default model from `config.toml`:

```bash
uv run python main.py --ollama "Explain inhibitory interneuron diversity."
```

Override with a specific local model:

```bash
uv run python main.py --ollama llama3.1:8b "Explain inhibitory interneuron diversity."
```

### Query result cache

Repeated queries with identical settings return the cached answer instantly, with no LLM call. The cache is keyed on query text, model, retrieval settings, and a corpus fingerprint — so adding new data to the knowledge base automatically invalidates stale entries.

Enable the cache in `config.toml`:

```toml
[cache]
enabled     = true
ttl_seconds = 86400   # entries expire after 24 hours; 0 = never expire
db_file     = "query_cache.db"
```

Or disable it for a single run regardless of config:

```bash
uv run python main.py --no-cache "What is the role of dopamine in reward?"
```

When a cached answer is returned, `[cached]` is printed before the response.

### Cross-encoder re-ranking

Re-ranking retrieves a larger pool of candidate chunks (3× by default), scores each one with an NLI cross-encoder for entailment against the query, and keeps only the top-k by entailment score. This improves retrieval precision at the cost of added latency.

Requires the NLI model (`make setup-nli`), then:

```bash
uv run python main.py --rerank "What are the electrophysiological properties of pyramidal neurons?"
```

The pool size multiplier is configured in `config.toml [reranker] pool_multiplier` (default 3). Re-ranking and entailment verification (`--verify`) are independent features that can be combined:

```bash
uv run python main.py --rerank --verify "What are the electrophysiological properties of pyramidal neurons?"
```

### NLI entailment verification

After generation, each cited claim is checked against its source text using the NLI cross-encoder. Claims that fall below the entailment threshold emit a warning and are stripped from the final answer.

```bash
uv run python main.py --verify "What is the role of AMPA receptors in LTP?"
```

### Tune query-time parameters

Common settings can be overridden per query without editing the config file:

```bash
uv run python main.py --node-top-k 10 --chunk-top-k 8 --confidence-threshold 0.75 "query..."
uv run python main.py --entailment-threshold 0.6 --verify "query..."
```

To change permanent defaults, edit `config.toml`. Resolution order: `CLI flag > ANTHROPIC_MODEL env var > config.toml`.

### Debug mode

Prints the raw model triples before `_verify()` runs, so you can see what the model produced before citation stripping:

```bash
uv run python main.py --debug "query..."
```

### Query logging

Append a structured JSONL record for each query to `logs/YYYY-MM-DD.jsonl`:

```bash
uv run python main.py --log "What is spike-timing-dependent plasticity?"
```

Each log entry contains the timestamp, query, answer, citations, taxonomy concepts, raw triples, retrieval metadata, model name, and any verification warnings.

---

## Incremental Ingestion

Add new paper chunks or taxonomy nodes to the live database without a full rebuild. After any insertion the retrieval caches are invalidated; run `--ingest` to rebuild them before querying.

### From the CLI

```bash
# Add paper chunks (same schema as data/raw/paper_chunks.json)
uv run python main.py --add path/to/new_papers.json

# Add taxonomy nodes (same schema as data/raw/taxonomy_nodes.csv)
uv run python main.py --add path/to/new_nodes.csv

# Rebuild retrieval indices after ingestion
uv run python main.py --ingest
```

### From Python

```python
from neurohive.knowledge_base import KnowledgeBase
from neurohive.ingestor import IncrementalIngestor
from neurohive.entities import PaperChunk, Node, Edge

kb = KnowledgeBase("data")
ingestor = IncrementalIngestor(kb)

# Add paper chunks
chunks = PaperChunk.load_all("new_papers.json")
result = ingestor.add_chunks(chunks, source="new_papers.json")
print(result)
# Batch: batch_20260608T130000Z
#   Nodes:  +0 added, 0 skipped
#   Edges:  +0 added, 0 skipped
#   Chunks: +12 added, 2 skipped
#   Cache:  invalidated — run 'python main.py --ingest' to rebuild indices

# Add taxonomy nodes
nodes = Node.load_all("new_nodes.csv")
result = ingestor.add_nodes(nodes, source="new_nodes.csv")

# Add edges directly
edges = [Edge(from_id="A", to_id="B", relationship_type="RELATED_TO", confidence=0.85)]
result = ingestor.add_edges(edges, source="manual")

# List all recorded ingestion batches (newest first)
for batch in ingestor.list_batches():
    print(batch["batch_id"], batch["n_chunks"], "chunks")
```

**Duplicate detection** uses natural keys: `(doi, chunk_index)` for chunks, `id` for nodes, and `(from_id, to_id, relationship_type)` for edges. Duplicates are skipped with a warning.

**Semantic deduplication** (optional, requires the embedding model) additionally rejects chunks that are near-duplicates of anything already in the corpus, even when the natural key is new. Configured in `config.toml [ingestor]`:

```toml
[ingestor]
semantic_dedup_enabled   = true
semantic_dedup_threshold = 0.92   # cosine similarity; 1.0 = exact match only
```

If the embedding model is not present, semantic deduplication is silently skipped.

---

## Data Drift Detection

Monitor the corpus for distribution shift and raise warnings when the data departs from a saved baseline.

### Workflow

```bash
# Step 1: save the current state as a baseline (run once after initial ingestion)
uv run python main.py --snapshot-drift

# Step 2: add new data and rebuild indices
uv run python main.py --add new_papers.json
uv run python main.py --ingest

# Step 3: check for drift against the baseline
uv run python main.py --check-drift
```

Example output:

```
Drift status: WARNING  ⚠
  Baseline : 2026-06-01T09:00:00+00:00
  Current  : 2026-06-08T14:30:00+00:00

Findings:
  WARNING: chunks count changed by +18.1% (83 → 98)
  WARNING: vocabulary distribution shifted (JS=0.0712 ≥ 0.05)

Volume changes:
  nodes            92 →     92  (+0.0%)
  edges           130 →    130  (+0.0%)
  chunks           83 →     98  (+18.1%)
  papers           24 →     29  (+20.8%)

Vocabulary Jensen-Shannon divergence : 0.0712
```

### From Python

```python
from neurohive.knowledge_base import KnowledgeBase
from neurohive.drift import DriftDetector

kb = KnowledgeBase("data")
detector = DriftDetector(kb, cache_dir=kb.cache_dir)

# Save current state as baseline
detector.save_baseline()

# Later, after adding new data:
report = detector.check()
print(report)
# report.status is "ok", "warning", or "alert"

if report.status == "alert":
    # Investigate, then accept the new distribution as the next baseline
    detector.save_baseline()
```

### Drift metrics

| Metric | Method | Default thresholds |
|---|---|---|
| Volume | % change in node / edge / chunk / paper counts | 20% → warning, 50% → alert |
| Vocabulary | Jensen-Shannon divergence over token distributions | 0.05 → warning, 0.15 → alert |
| Type structure | New node or edge types not in baseline | any new type → warning |
| Paper recency | Median publication year shift | ≥ 5 years → warning |
| Embedding centroid | Cosine distance between corpus embedding means | 0.10 → warning, 0.25 → alert |

All thresholds are configurable in `config.toml [drift]`. Run `uv run python scripts/demo_drift.py` for a complete end-to-end walkthrough.

---

## All CLI Flags

| Flag | Default | Description |
|---|---|---|
| `query` (positional) | — | Question to answer; omit for interactive REPL mode |
| `--model MODEL` | config.toml `[anthropic] model` | Anthropic model ID (also reads `ANTHROPIC_MODEL` env var) |
| `--ollama [MODEL]` | config.toml `[ollama] model` | Use local Ollama instead of Anthropic; optionally pass an Ollama model name |
| `--node-top-k N` | config.toml `[pipeline] node_top_k` | Taxonomy nodes retrieved per query |
| `--chunk-top-k N` | config.toml `[pipeline] chunk_top_k` | Paper chunks retrieved per query |
| `--confidence-threshold F` | config.toml `[pipeline] confidence_threshold` | Minimum edge confidence for graph expansion (0.0–1.0) |
| `--embeddings-model MODEL` | config.toml `[embeddings] model` | HuggingFace model ID for dense retrieval |
| `--nli-model MODEL` | config.toml `[nli] model` | HuggingFace model ID for NLI verification / re-ranking |
| `--entailment-threshold F` | config.toml `[nli] entailment_threshold` | Minimum entailment probability for NLI verification (used with `--verify`) |
| `--show-sources` | off | Print retrieved nodes and chunks after the answer |
| `--verify` | off | Enable post-generation NLI entailment verification (requires `make setup-nli`) |
| `--rerank` | off | Re-rank retrieved chunks with the cross-encoder before generation (requires `make setup-nli`) |
| `--no-cache` | off | Disable the query result cache for this run |
| `--debug` | off | Print raw model triples before verification |
| `--log` | off | Append a JSONL record for each query to `[paths] logs_dir/YYYY-MM-DD.jsonl` |
| `--ingest` | — | Build (or rebuild) the database and retrieval caches, then exit |
| `--add FILE` | — | Incrementally add chunks (`.json`) or nodes (`.csv`) to the live database, then exit |
| `--snapshot-drift` | — | Save the current corpus state as the drift baseline, then exit |
| `--check-drift` | — | Compare the current corpus against the saved baseline and print a report, then exit |

---

## Python API

```python
from neurohive.pipeline import QueryPipeline

pipeline = QueryPipeline()
result = pipeline.run("What are the mechanisms of synaptic scaling?")

print(result.answer)           # assembled prose from verified triples
print(result.citation)         # ["Turrigiano et al., 2013"]
print(result.document)         # ["Synaptic_Scaling", "Homeostatic_Plasticity"]
print(result.answer_triples)   # raw verified triples
print(result.retrieval_mode)   # "hybrid (BM25 + dense, RRF)" or "bm25"
print(result.cached)           # True if answer came from the query cache

import json
print(json.dumps(result.as_dict(), indent=2))
```

Use Ollama:

```python
from neurohive.backends import OllamaBackend
from neurohive.pipeline import QueryPipeline

pipeline = QueryPipeline(backend=OllamaBackend())
result = pipeline.run("What are the mechanisms of synaptic scaling?")
```

Enable re-ranking and entailment verification:

```python
from pathlib import Path
from neurohive.pipeline import QueryPipeline

pipeline = QueryPipeline(
    rerank=True,
    verify_entailment=True,
    nli_model_dir=Path("models/nli-deberta-v3-small"),
    entailment_threshold=0.6,
)
result = pipeline.run("What is the role of AMPA receptors in LTP?")
```

Disable the query cache for a single call:

```python
pipeline = QueryPipeline(cache_enabled=False)
```

---

## Configuration

All tunable parameters live in `config.toml` at the repo root:

```toml
[paths]
data_dir   = "data"
env_file   = ".env"
logs_dir   = "logs"
models_dir = "models"

[pipeline]
node_top_k           = 6
chunk_top_k          = 6
confidence_threshold = 0.8

[retrieval]
bm25_k1          = 1.5
bm25_b           = 0.75
node_rrf_k       = 30
chunk_rrf_k      = 60
sibling_discount = 0.5

# Cache artifact filenames — used by IncrementalIngestor to know which files to invalidate.
bm25_cache_version = 1
bm25_meta_file     = "bm25_meta.json"
node_bm25_file     = "node_bm25.json"
chunk_bm25_file    = "chunk_bm25.json"
emb_meta_file      = "emb_meta.json"
node_emb_file      = "node_embs.npy"
chunk_emb_file     = "chunk_embs.npy"

[cache]
# SQLite-backed query result cache.
# The cache key covers query text, model, top-k settings, and a corpus fingerprint.
# Adding new data to the knowledge base changes the fingerprint and invalidates stale entries.
enabled     = false
ttl_seconds = 86400      # seconds before an entry expires; 0 = no expiry
db_file     = "query_cache.db"

[reranker]
# Cross-encoder re-ranking of retrieved chunks before generation.
# Requires the NLI model (make setup-nli). Enable per query with --rerank.
enabled         = false
pool_multiplier = 3      # fetch pool_multiplier × chunk_top_k candidates, keep top chunk_top_k

[ingestor]
# Semantic deduplication during incremental ingestion.
# Requires the embedding model (make setup-embeddings).
# Chunks whose cosine similarity to any existing chunk meets the threshold are skipped,
# even if their (doi, chunk_index) key is new.
semantic_dedup_enabled   = true
semantic_dedup_threshold = 0.92

[drift]
baseline_file           = "drift_baseline.json"
vocab_warning_threshold = 0.05
vocab_alert_threshold   = 0.15
volume_warning_pct      = 0.20
volume_alert_pct        = 0.50
embedding_warning_dist  = 0.10
embedding_alert_dist    = 0.25

[anthropic]
model = "claude-haiku-4-5-20251001"

[ollama]
model = "ministral-3:3b-instruct-2512-q4_K_M"
host  = "http://localhost:11434"

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"

[nli]
model                = "cross-encoder/nli-deberta-v3-small"
entailment_threshold = 0.5
```

The config is validated at load time using Pydantic. Invalid values (e.g. `confidence_threshold` outside 0–1, or `vocab_warning_threshold ≥ vocab_alert_threshold`) raise a `ValueError` with a clear message before any query runs. `config.toml` is the **sole source of truth** — no defaults exist anywhere in the Python code.

`load_config()` returns a typed `Config` instance, so parameters are available as attributes:

```python
from neurohive.config import load_config

cfg = load_config()
print(cfg.pipeline.node_top_k)          # int
print(cfg.pipeline.confidence_threshold) # float
print(cfg.anthropic.model)              # str
print(cfg.cache.enabled)                # bool
```

---

## Project Structure

```
neurohive/
├── assets/
│   └── taxonomy_graph.png          Taxonomy graph visualisation
│
├── data/
│   ├── raw/
│   │   ├── taxonomy_nodes.csv      92 neuroscience concept nodes
│   │   ├── taxonomy_edges.csv      130 typed relationships between nodes
│   │   ├── paper_chunks.json       83 excerpts from 24 peer-reviewed papers
│   │   └── README.md               Data schema reference
│   └── database/
│       ├── neurohive.db            SQLite knowledge base (auto-generated; gitignored)
│       ├── bm25_meta.json          BM25 cache metadata (generated by --ingest)
│       ├── node_bm25.json          Taxonomy BM25 index cache
│       ├── chunk_bm25.json         Paper chunk BM25 index cache
│       ├── emb_meta.json           Embedding cache metadata (hybrid retrieval only)
│       ├── node_embs.npy           Taxonomy dense embeddings
│       ├── chunk_embs.npy          Paper chunk dense embeddings
│       ├── query_cache.db          Query result cache (created when cache.enabled = true)
│       └── drift_baseline.json     Drift baseline snapshot (created by --snapshot-drift)
│
├── neurohive/
│   ├── entities/
│   │   ├── node.py                 Node dataclass
│   │   ├── edge.py                 Edge dataclass
│   │   └── paper_chunk.py          PaperChunk dataclass
│   ├── knowledge_base.py           SQLite-backed store; schema migration; graph expansion
│   ├── ingestor.py                 IncrementalIngestor: add chunks/nodes/edges at runtime
│   ├── drift.py                    DriftDetector: snapshot and compare corpus statistics
│   ├── cache.py                    QueryCache: SQLite-backed query result cache
│   ├── schema.py                   Pydantic v2 Config class + sub-configs; returned by load_config()
│   ├── retrieval.py                BM25 Okapi + optional hybrid RRF retriever
│   ├── backends.py                 AnthropicBackend and OllamaBackend
│   ├── pipeline.py                 QueryPipeline: end-to-end orchestration + _verify()
│   └── config.py                   load_config(): reads config.toml and returns a typed Config instance
│
├── scripts/
│   ├── demo_drift.py               End-to-end walkthrough: ingest → baseline → drift check
│   ├── setup_ollama_model.py       Pulls the configured local Ollama model
│   ├── download_model.py           Downloads the configured embedding model
│   └── download_nli_model.py       Downloads the configured NLI cross-encoder model
│
├── tests/
│   ├── conftest.py                 Shared fixtures and MockBackend
│   ├── test_backends.py
│   ├── test_models.py
│   ├── test_config.py
│   ├── test_knowledge_base.py
│   ├── test_retrieval.py
│   ├── test_pipeline.py
│   └── test_drift.py               47 tests for DriftDetector
│
├── logs/                           Per-day JSONL query logs (gitignored; created by --log)
│
├── main.py                         CLI entry point
├── config.toml                     All tunable parameters (single source of truth)
├── pyproject.toml
└── Makefile                        Run `make help` to see all targets
```

---

## Tests

```bash
uv run pytest
```

Run a focused subset:

```bash
uv run pytest tests/test_retrieval.py tests/test_pipeline.py
uv run pytest tests/test_drift.py -v
```
