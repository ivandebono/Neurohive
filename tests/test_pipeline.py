"""Tests for QueryPipeline: normalisation, verification, context formatting."""

from __future__ import annotations

import pytest

from neurohive.entities import Node, PaperChunk
from neurohive.pipeline import QueryPipeline


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------

class TestNormalise:
    def test_strips_punctuation_and_spaces(self):
        assert QueryPipeline._normalise("Smith et al., 2020") == "smithetal2020"

    def test_lowercases(self):
        assert QueryPipeline._normalise("ACTION_POTENTIAL") == "actionpotential"

    def test_empty_string(self):
        assert QueryPipeline._normalise("") == ""


class TestNormaliseText:
    def test_strips_punctuation_preserves_words(self):
        result = QueryPipeline._normalise_text("Hello, world!")
        assert result == "hello world"

    def test_collapses_whitespace(self):
        assert QueryPipeline._normalise_text("  foo   bar  ") == "foo bar"

    def test_empty_string(self):
        assert QueryPipeline._normalise_text("") == ""


# ---------------------------------------------------------------------------
# _assemble_answer
# ---------------------------------------------------------------------------

class TestAssembleAnswer:
    def test_empty_triples_returns_error_message(self):
        msg = QueryPipeline._assemble_answer([])
        assert len(msg) > 0
        assert "removed" in msg.lower() or "empty" in msg.lower() or "All" in msg

    def test_paper_triple_formatted_with_citation(self):
        triples = [{"type": "paper", "claim": "X activates Y.", "citation": "Smith et al., 2020", "quote": "q"}]
        result = QueryPipeline._assemble_answer(triples)
        assert "X activates Y." in result
        assert "Smith et al., 2020" in result

    def test_taxonomy_triple_formatted_with_concept(self):
        triples = [{"type": "taxonomy", "claim": "LTP is important.", "concept": "Long_Term_Potentiation"}]
        result = QueryPipeline._assemble_answer(triples)
        assert "LTP is important." in result
        assert "Long_Term_Potentiation" in result

    def test_multiple_triples_joined(self):
        triples = [
            {"type": "paper",    "claim": "A.",  "citation": "Smith et al., 2020", "quote": "q"},
            {"type": "taxonomy", "claim": "B.",  "concept": "Concept_X"},
        ]
        result = QueryPipeline._assemble_answer(triples)
        assert "A." in result and "B." in result


# ---------------------------------------------------------------------------
# _verify
# ---------------------------------------------------------------------------

def _make_chunk(citation_str: str, text: str = "Some source text.") -> PaperChunk:
    surname, year = citation_str.split(" et al., ")
    return PaperChunk(
        id=1, doi="10.1/x", title="T",
        authors=f"{surname}, A",
        year=int(year), chunk_index=0,
        text=text,
        taxonomy_node_ids=(),
    )


class TestVerify:
    def test_matched_paper_citation_kept(self, pipeline):
        chunk = _make_chunk("Smith et al., 2020")
        structured = {"answer": [
            {"type": "paper", "claim": "X.", "citation": "Smith et al., 2020", "quote": "q"}
        ]}
        result = pipeline._verify(structured, [], [chunk])
        assert len(result["citation"]) == 1
        assert result["citation"][0] == "Smith et al., 2020"

    def test_unmatched_paper_citation_dropped(self, pipeline):
        structured = {"answer": [
            {"type": "paper", "claim": "X.", "citation": "Ghost et al., 1999", "quote": "q"}
        ]}
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = pipeline._verify(structured, [], [])
        assert result["citation"] == []
        assert result["answer_triples"] == []

    def test_matched_taxonomy_concept_kept(self, pipeline):
        node = Node(id="n1", name="Action_Potential", type="Research_area", description="d")
        structured = {"answer": [
            {"type": "taxonomy", "claim": "AP fires.", "concept": "Action_Potential"}
        ]}
        result = pipeline._verify(structured, [node], [])
        assert result["document"] == ["Action_Potential"]

    def test_unmatched_taxonomy_concept_dropped(self, pipeline):
        structured = {"answer": [
            {"type": "taxonomy", "claim": "X.", "concept": "Nonexistent_Concept"}
        ]}
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = pipeline._verify(structured, [], [])
        assert result["document"] == []

    def test_quote_mismatch_warns_but_keeps_triple(self, pipeline):
        chunk = _make_chunk("Smith et al., 2020", text="Actual source text here.")
        structured = {"answer": [
            {"type": "paper", "claim": "X.", "citation": "Smith et al., 2020",
             "quote": "this quote does not appear in the text"}
        ]}
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = pipeline._verify(structured, [], [chunk])
        # Triple kept despite quote mismatch
        assert len(result["citation"]) == 1
        assert any("quote" in str(w.message).lower() or "verbatim" in str(w.message).lower()
                   for w in caught)

    def test_citation_deduplicated(self, pipeline):
        chunk = _make_chunk("Smith et al., 2020")
        structured = {"answer": [
            {"type": "paper", "claim": "A.", "citation": "Smith et al., 2020", "quote": "q"},
            {"type": "paper", "claim": "B.", "citation": "Smith et al., 2020", "quote": "q"},
        ]}
        result = pipeline._verify(structured, [], [chunk])
        assert result["citation"].count("Smith et al., 2020") == 1


# ---------------------------------------------------------------------------
# _format_taxonomy_context
# ---------------------------------------------------------------------------

class TestFormatTaxonomyContext:
    def test_isolated_node_shows_no_connections_label(self, pipeline):
        # Retrieve the LONE node from the test KB (it has no edges)
        lone = pipeline.kb.nodes_by_id["LONE"]
        ctx = pipeline._format_taxonomy_context([lone])
        assert "[no connections]" in ctx

    def test_connected_node_does_not_show_no_connections(self, pipeline):
        p001 = pipeline.kb.nodes_by_id["P001"]
        sp001 = pipeline.kb.nodes_by_id["SP001"]
        ctx = pipeline._format_taxonomy_context([p001, sp001])
        # P001 has outgoing edges; the label should not appear for it
        assert "[no connections]" not in ctx

    def test_breadcrumb_path_included(self, pipeline):
        ra = pipeline.kb.nodes_by_id["RA001"]
        sp = pipeline.kb.nodes_by_id["SP001"]
        p  = pipeline.kb.nodes_by_id["P001"]
        ctx = pipeline._format_taxonomy_context([ra, sp, p])
        assert "Neural Signaling" in ctx
        assert "→" in ctx

    def test_inferred_tag_on_semantic_edge(self, pipeline):
        # TH001 → RA001 is a semantic EXPLAINS edge; should show [inferred]
        th = pipeline.kb.nodes_by_id["TH001"]
        ra = pipeline.kb.nodes_by_id["RA001"]
        ctx = pipeline._format_taxonomy_context([th, ra])
        assert "[inferred]" in ctx

    def test_hidden_edges_count_shown(self, pipeline):
        # Render only P001 without its neighbours — should show "further connections"
        p001 = pipeline.kb.nodes_by_id["P001"]
        ctx = pipeline._format_taxonomy_context([p001])
        assert "further connection" in ctx


class TestRetrieveAndExpand:
    def test_chunk_taxonomy_links_seed_graph_expansion(self, pipeline):
        nodes, chunks = pipeline._retrieve_and_expand("channel kinetics mathematically")
        node_ids = {n.id for n in nodes}
        chunk_node_ids = {
            nid
            for chunk in chunks
            for nid in chunk.taxonomy_node_ids
        }
        assert "TH001" in chunk_node_ids
        assert "TH001" in node_ids
