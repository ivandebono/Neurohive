"""
Shared prompt fragments, base class, and helper functions for generation backends.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a closed-context neuroscience query system. The TAXONOMY CONTEXT and \
PAPER EVIDENCE sections in the user message are your complete and only source \
of knowledge. You have no other knowledge available.

Before producing answer triples, fill the "reasoning" field with 2–4 sentences \
identifying: (a) the 1–3 taxonomy concepts most directly relevant to the query, \
(b) the 1–3 papers providing the strongest evidence, and (c) any aspect of the \
query the context does not cover. This reasoning is internal — use it to focus \
your answer, not as output shown to the user.

Absolute rules — every rule is a hard constraint, not a guideline:
1. Use ONLY facts that appear verbatim or by direct implication in the provided \
   TAXONOMY CONTEXT or PAPER EVIDENCE. Nothing else.
2. Do NOT draw on your training data, even for background context, definitions, \
   or facts you are certain are correct. If it is not in the provided context, \
   it does not exist for this query.
3. If the context is insufficient to answer the query fully, produce at least one \
   taxonomy triple using the most relevant concept available; set the claim to state \
   explicitly what is covered and what information is absent. Do not fill gaps with \
   training knowledge under any circumstances.
4. Express every factual claim as a structured triple of one of two types:
   • Paper triple  (drawn from PAPER EVIDENCE):
       type:     "paper"
       claim:    your one-sentence statement derived from the source
       quote:    the verbatim passage from the paper that supports it
       citation: "Surname et al., Year" exactly as shown in PAPER EVIDENCE
   • Taxonomy triple  (drawn from TAXONOMY CONTEXT):
       type:     "taxonomy"
       claim:    your one-sentence statement derived from the source
       concept:  the exact concept name as it appears in TAXONOMY CONTEXT
   Every factual sentence must map to exactly one triple. The "answer" field \
   is an ordered array of these triples — it is not free-form prose.
5. Do not speculate, infer beyond what the context states, or add qualifications \
   drawn from general neuroscience knowledge.
6. Taxonomy nodes are tagged [direct] (scored by retrieval for this query) or \
   [expanded] (added by graph traversal for broader context). Prefer [direct] \
   nodes as primary evidence; use [expanded] nodes only as supporting context. \
   Edges marked [confidence: X%] represent weaker or inferred connections — \
   qualify claims that rely on such edges: write "may be related to" rather than \
   "is related to".
"""


def _user_prompt(query: str, taxonomy_ctx: str, paper_ctx: str) -> str:
    return (
        f"Query: {query}\n\n"
        f"---\nTAXONOMY CONTEXT\n{taxonomy_ctx}\n\n"
        f"---\nPAPER EVIDENCE\n{paper_ctx}\n\n"
        f"---\n"
        f"Query (restated): {query}\n\n"
        "Answer the query using ONLY the context above. "
        "Fill the reasoning field first, then express every factual claim as a "
        "paper or taxonomy triple. "
        "Do not use any knowledge outside these two sections. "
        "Return only a JSON object with keys \"reasoning\" and \"answer\"."
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class GenerationBackend(ABC):
    """Abstract generation backend.  Subclasses must implement ``generate``."""

    @abstractmethod
    def generate(self, query: str, taxonomy_ctx: str, paper_ctx: str) -> dict:
        """
        Return ``{"answer": list[dict]}`` where each dict is a paper or taxonomy
        triple.  Must never raise on a well-formed call; instead return a list
        containing a single error triple with ``"type": "taxonomy"`` and a
        ``"claim"`` describing what went wrong.
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable model identifier for display purposes."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _error_response(message: str) -> dict:
    return {
        "answer": [{
            "type": "taxonomy",
            "claim": message,
            "concept": "Generation_Backend",
        }]
    }


def _validate_structured_response(data: object) -> dict:
    if not isinstance(data, dict) or not isinstance(data.get("answer"), list):
        return _error_response("Generation backend returned invalid JSON schema.")
    answer: list[dict] = []
    for item in data["answer"]:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "paper" and all(k in item for k in ("claim", "quote", "citation")):
            answer.append({
                "type": "paper",
                "claim": str(item["claim"]),
                "quote": str(item["quote"]),
                "citation": str(item["citation"]),
            })
        elif item.get("type") == "taxonomy" and all(k in item for k in ("claim", "concept")):
            answer.append({
                "type": "taxonomy",
                "claim": str(item["claim"]),
                "concept": str(item["concept"]),
            })
    if not answer:
        return _error_response("Generation backend returned no valid grounded triples.")
    return {"answer": answer}


def _parse_json_object(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])
