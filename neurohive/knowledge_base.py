"""Loads the three data files and builds in-memory indices for fast lookup."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from neurohive.models import Edge, Node, PaperChunk

# Six of the seven edge types are directed (source → target is meaningful).
# RELATED_TO is the only bidirectional edge and must be traversed in both
# incoming and outgoing directions during graph expansion.
_BIDIRECTIONAL_EDGE_TYPES = frozenset({"RELATED_TO"})

# Strict Pillar → Subpillar → Research_area spine, used only for ancestor
# walks (breadcrumbs). Subset of the six directed edge types.
_HIERARCHY_EDGE_TYPES = frozenset({"HAS_SUBPILLAR", "HAS_RESEARCH_AREA"})


class KnowledgeBase:
    """
    In-memory store for nodes, edges, and paper chunks with pre-built indices.

    Indices built at construction time
    -----------------------------------
    nodes_by_id      : dict[str, Node]
    outgoing         : node_id → edges where that node is `from_id`
    incoming         : node_id → edges where that node is `to_id`
    node_to_chunks   : node_id → chunks that list that node in taxonomy_node_ids
    chunks_by_id     : chunk.id → PaperChunk

    Parameters
    ----------
    data_dir : Directory containing the three data files.
    """

    def __init__(self, data_dir: Path | str):
        data_dir = Path(data_dir)

        nodes = Node.load_all(data_dir / "taxonomy_nodes.csv")
        edges = Edge.load_all(data_dir / "taxonomy_edges.csv")
        chunks = PaperChunk.load_all(data_dir / "paper_chunks.json")

        self.nodes_by_id: dict[str, Node] = {n.id: n for n in nodes}
        self.nodes: list[Node] = nodes
        self.edges: list[Edge] = edges
        self.chunks: list[PaperChunk] = chunks

        # Adjacency indices
        self.outgoing: dict[str, list[Edge]] = defaultdict(list)
        self.incoming: dict[str, list[Edge]] = defaultdict(list)
        for edge in edges:
            self.outgoing[edge.from_id].append(edge)
            self.incoming[edge.to_id].append(edge)

        # Chunk indices
        self.chunks_by_id: dict[int, PaperChunk] = {c.id: c for c in chunks}
        self.node_to_chunks: dict[str, list[PaperChunk]] = defaultdict(list)
        for chunk in chunks:
            for nid in chunk.taxonomy_node_ids:
                self.node_to_chunks[nid].append(chunk)

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def ancestors(self, node_id: str) -> list[Node]:
        """Walk up HAS_SUBPILLAR / HAS_RESEARCH_AREA edges to the root."""
        result: list[Node] = []
        current = node_id
        visited: set[str] = set()
        while current and current not in visited:
            visited.add(current)
            parent_edge = next(
                (e for e in self.incoming.get(current, []) if e.relationship_type in _HIERARCHY_EDGE_TYPES),
                None,
            )
            if parent_edge is None:
                break
            parent = self.nodes_by_id.get(parent_edge.from_id)
            if parent:
                result.append(parent)
            current = parent_edge.from_id
        return result

    def breadcrumb(self, node_id: str) -> str:
        """Return a human-readable path, e.g. 'Neural Signaling → Electrical Signaling → Action Potential'."""
        node = self.nodes_by_id.get(node_id)
        if not node:
            return node_id
        parts = [a.name.replace("_", " ") for a in reversed(self.ancestors(node_id))] + [node.name.replace("_", " ")]
        return " → ".join(parts)

    def smart_expand(
        self,
        node_ids: set[str],
        confidence_threshold: float = 0.8,
    ) -> set[str]:
        """
        Expand a seed set of node IDs with semantically useful neighbours.

        Traversal rules (one hop, all seven relationship types)
        -------------------------------------------------------
        Six edge types are directed; one (RELATED_TO) is bidirectional.

        incoming HAS_SUBPILLAR / HAS_RESEARCH_AREA  → parent Pillar/Subpillar
        incoming EXPLAINS                            → Theory that explains seed
        incoming HAS_THEORY                          → parent Pillar of a Theory seed
        incoming IS_DERIVED_FROM                     → concepts derived from seed
        incoming RELATED_TO                          → lateral neighbours (bidirectional)
        outgoing HAS_THEORY                          → Theory owned by seed Pillar
        outgoing HAS_DIMENSION                       → Dimension nodes of seed
        outgoing RELATED_TO                          → lateral neighbours (bidirectional)
        outgoing IS_DERIVED_FROM                     → source concept seed derives from

        Note: Theory nodes sit outside the HAS_SUBPILLAR/HAS_RESEARCH_AREA hierarchy,
        so their parent Pillar is reached via the reverse of HAS_THEORY.  This is
        handled by adding an incoming HAS_THEORY traversal when the seed is a Theory.

        Only edges with ``edge.confidence >= confidence_threshold`` are followed.
        Set ``confidence_threshold=0.0`` to follow all edges unconditionally.

        Parameters
        ----------
        node_ids             : Seed node IDs from BM25 / hybrid retrieval.
        confidence_threshold : Minimum confidence to follow an edge (default 0.8).
                               Hierarchy edges are always 1.0 and unaffected.
        """
        expanded = set(node_ids)

        for nid in node_ids:
            for edge in self.incoming.get(nid, []):
                threshold = (
                    min(confidence_threshold + 0.1, 1.0)
                    if edge.map_source == "semantic"
                    else confidence_threshold
                )
                if edge.confidence < threshold:
                    continue
                rt = edge.relationship_type
                candidate = self.nodes_by_id.get(edge.from_id)
                if candidate is None:
                    continue
                # Walk up the taxonomy hierarchy (Subpillar/Pillar)
                if rt in _HIERARCHY_EDGE_TYPES and candidate.type in ("Pillar", "Subpillar"):
                    expanded.add(edge.from_id)
                # Theory that explains this Research_area
                elif rt == "EXPLAINS" and candidate.type == "Theory":
                    expanded.add(edge.from_id)
                # Parent Pillar of a Theory (Theory nodes are not in the
                # HAS_SUBPILLAR/HAS_RESEARCH_AREA hierarchy, so ancestors()
                # won't find their Pillar — use HAS_THEORY incoming instead)
                elif rt == "HAS_THEORY" and candidate.type == "Pillar":
                    expanded.add(edge.from_id)
                # Concepts derived FROM this node (seed=LTP → add STDP)
                elif rt == "IS_DERIVED_FROM" and candidate.type == "Research_area":
                    expanded.add(edge.from_id)
                # RELATED_TO is bidirectional — also traverse incoming direction
                elif rt in _BIDIRECTIONAL_EDGE_TYPES:
                    expanded.add(edge.from_id)

            for edge in self.outgoing.get(nid, []):
                threshold = (
                    min(confidence_threshold + 0.1, 1.0)
                    if edge.map_source == "semantic"
                    else confidence_threshold
                )
                if edge.confidence < threshold:
                    continue
                rt = edge.relationship_type
                candidate = self.nodes_by_id.get(edge.to_id)
                if candidate is None:
                    continue
                # Theories owned by this Pillar (Pillar → Theory)
                if rt == "HAS_THEORY" and candidate.type == "Theory":
                    expanded.add(edge.to_id)
                # Measurable dimensions of this node
                elif rt == "HAS_DIMENSION" and candidate.type == "Dimension":
                    expanded.add(edge.to_id)
                # Lateral Research_area neighbours (RELATED_TO is bidirectional)
                elif rt in _BIDIRECTIONAL_EDGE_TYPES:
                    expanded.add(edge.to_id)
                # Source concept this node derives from (seed=STDP → add LTP)
                elif rt == "IS_DERIVED_FROM" and candidate.type == "Research_area":
                    expanded.add(edge.to_id)

        return expanded

    def chunks_for_nodes(self, node_ids: set[str]) -> list[PaperChunk]:
        """Return deduplicated chunks linked to any of the given node IDs."""
        seen: set[int] = set()
        result: list[PaperChunk] = []
        for nid in node_ids:
            for chunk in self.node_to_chunks.get(nid, []):
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    result.append(chunk)
        return result

    def draw_graph(
        self,
        highlight: set[str] | None = None,
        figsize: tuple[int, int] = (20, 20),
        seed: int = 42,
        save_path: str | Path | None = None,
    ) -> None:
        """
        Draw the taxonomy graph with a spring (force-directed) layout.

        Parameters
        ----------
        highlight  : Optional set of node IDs to ring in white (e.g. expanded set).
        figsize    : Figure size in inches.
        seed       : Random seed for layout reproducibility.
        save_path  : If given, save the figure to this path instead of showing it.

        Requires ``networkx`` and ``matplotlib`` (both available after ``make setup-dev``).
        """
        try:
            import networkx as nx                   # noqa: PLC0415
            import matplotlib.pyplot as plt        # noqa: PLC0415
            import matplotlib.patches as mpatches  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "draw_graph requires networkx and matplotlib.\n"
                "Run:  make setup-dev"
            ) from exc

        NODE_COLOR = {
            "Pillar":        "#3B82F6",
            "Subpillar":     "#8B5CF6",
            "Research_area": "#10B981",
            "Theory":        "#F59E0B",
            "Dimension":     "#EF4444",
        }
        EDGE_COLOR = {
            "HAS_SUBPILLAR":     "#93C5FD",
            "HAS_RESEARCH_AREA": "#C4B5FD",
            "HAS_THEORY":        "#FCD34D",
            "EXPLAINS":          "#FDE68A",
            "HAS_DIMENSION":     "#FCA5A5",
            "RELATED_TO":        "#6EE7B7",
            "IS_DERIVED_FROM":   "#A5F3FC",
        }
        NODE_SIZE = {
            "Pillar": 600, "Subpillar": 350,
            "Research_area": 160, "Theory": 300, "Dimension": 240,
        }
        FONT_SIZE = {
            "Pillar": 6.5, "Subpillar": 5.5,
            "Research_area": 4.2, "Theory": 5.5, "Dimension": 5.0,
        }

        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.id)
        for edge in self.edges:
            if edge.from_id in self.nodes_by_id and edge.to_id in self.nodes_by_id:
                G.add_edge(edge.from_id, edge.to_id, rel=edge.relationship_type)

        pos = nx.spring_layout(G, k=2.8, iterations=300, seed=seed)

        node_list = list(G.nodes())
        labels    = {n: self.nodes_by_id[n].name.replace("_", " ") for n in node_list}

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#0F172A")
        fig.patch.set_facecolor("#0F172A")

        for rel, ec in EDGE_COLOR.items():
            sub   = [(u, v) for u, v, d in G.edges(data=True) if d.get("rel") == rel]
            style = "dashed" if rel == "RELATED_TO" else "solid"
            alpha = 0.30 if rel in ("RELATED_TO", "HAS_RESEARCH_AREA") else 0.50
            nx.draw_networkx_edges(
                G, pos, edgelist=sub, ax=ax,
                edge_color=ec, alpha=alpha, width=0.8,
                arrows=True, arrowsize=6,
                connectionstyle="arc3,rad=0.06", style=style,
            )

        for ntype, color in NODE_COLOR.items():
            nlist = [n for n in node_list if self.nodes_by_id[n].type == ntype]
            ec    = "white" if (highlight and any(n in highlight for n in nlist)) else color
            nx.draw_networkx_nodes(
                G, pos, nodelist=nlist, ax=ax,
                node_color=color, node_size=NODE_SIZE[ntype],
                alpha=0.92, linewidths=1.0,
                edgecolors=["white" if (highlight and n in highlight) else color for n in nlist],
            )
            nx.draw_networkx_labels(
                G, pos, labels={n: labels[n] for n in nlist}, ax=ax,
                font_size=FONT_SIZE[ntype], font_color="white", font_weight="bold",
            )

        nleg = [mpatches.Patch(color=c, label=t.replace("_", " ")) for t, c in NODE_COLOR.items()]
        eleg = [mpatches.Patch(color=c, label=r.replace("_", " ").title()) for r, c in EDGE_COLOR.items()]
        ax.legend(
            handles=nleg + eleg, loc="lower left", fontsize=8.5,
            framealpha=0.4, facecolor="#1E293B", edgecolor="#475569",
            labelcolor="white", ncol=2,
        )
        ax.set_title(
            f"Neuroscience Taxonomy Graph  ({len(self.nodes)} nodes · {len(self.edges)} edges)",
            color="white", fontsize=15, pad=14, fontweight="bold",
        )
        ax.axis("off")
        plt.tight_layout(pad=1.2)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
        else:
            plt.show()
