# Data Reference

## How the files relate

- `taxonomy_edges.csv` links nodes to nodes: `from_id` and `to_id` both refer to `id` in `taxonomy_nodes.csv`
- `paper_chunks.json` links chunks to nodes: each chunk's `taxonomy_node_ids` is a list of `id` values from `taxonomy_nodes.csv`
- Nodes are organized in a hierarchy: `Pillar → Subpillar → Research_area`, connected by `HAS_SUBPILLAR` and `HAS_RESEARCH_AREA` edges. `Dimension` nodes attach at multiple levels (Pillar or Subpillar) via `HAS_DIMENSION` edges. `Theory` nodes attach at the Pillar level via `HAS_THEORY` edges and link down to `Research_area` nodes via `EXPLAINS` edges. `RELATED_TO` edges create lateral connections between `Research_area` nodes (and occasionally to `Dimension` nodes). `IS_DERIVED_FROM` edges mark `Research_area` nodes derived from another `Research_area`.

---

## taxonomy_nodes.csv

| Column | Type | Description |
|--------|------|-------------|
| id | string | Unique node identifier |
| name | string | Concept name |
| type | string | One of: Pillar, Subpillar, Research_area, Dimension, Theory |
| description | string | Full definition of the concept |
| source | string | Provenance of the node |

## taxonomy_edges.csv

| Column | Type | Description |
|--------|------|-------------|
| from_id | string | Source node ID — must exist in `taxonomy_nodes.csv` |
| to_id | string | Target node ID — must exist in `taxonomy_nodes.csv` |
| relationship_type | string | One of: HAS_SUBPILLAR, HAS_RESEARCH_AREA, HAS_DIMENSION, HAS_THEORY, EXPLAINS, RELATED_TO, IS_DERIVED_FROM |
| confidence | float | Relationship confidence score, 0.0–1.0 |
| map_source | string | How the edge was created: canonical or semantic |
| notes | string | Optional context |

## paper_chunks.json

A list of objects, each representing a chunk of text from an academic paper.

| Field | Type | Description |
|-------|------|-------------|
| id | integer | Unique chunk identifier |
| doi | string | Paper DOI |
| title | string | Paper title |
| authors | string | Semicolon-separated author list |
| year | integer | Publication year |
| chunk_index | integer | Position of this chunk within the paper (0-indexed) |
| text | string | Chunk text content |
| taxonomy_node_ids | list[string] | IDs of taxonomy nodes this chunk is linked to — each value is an `id` from `taxonomy_nodes.csv` |
