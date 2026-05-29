"""
Generation backend for QueryPipeline.

AnthropicBackend  (default)
    Uses the Anthropic Messages API with forced tool use (tool_choice) to
    guarantee a valid JSON response matching the output schema.
    Requires: ANTHROPIC_API_KEY environment variable.

The backend returns::

    {"answer": list[dict]}   # ordered list of paper/taxonomy triple dicts

Model selection
---------------
The model name is configured in config.toml under [anthropic].
Passing a model string directly to the constructor overrides config.toml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Shared prompt fragments
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a closed-context neuroscience query system. The TAXONOMY CONTEXT and \
PAPER EVIDENCE sections in the user message are your complete and only source \
of knowledge. You have no other knowledge available.

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
"""


def _user_prompt(query: str, taxonomy_ctx: str, paper_ctx: str) -> str:
    return (
        f"Query: {query}\n\n"
        f"---\nTAXONOMY CONTEXT\n{taxonomy_ctx}\n\n"
        f"---\nPAPER EVIDENCE\n{paper_ctx}\n\n"
        "---\n"
        "Answer the query using ONLY the context above. "
        "Express every factual claim as a paper or taxonomy triple. "
        "Do not use any knowledge outside these two sections."
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
# Anthropic backend  (tool_choice → guaranteed schema)
# ---------------------------------------------------------------------------

_ANTHROPIC_TOOL: dict = {
    "name": "format_response",
    "description": (
        "Return an answer derived exclusively from the TAXONOMY CONTEXT and "
        "PAPER EVIDENCE supplied in the user message. Every claim must be "
        "expressed as a structured triple traceable to those two sections. "
        "Do not use training knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "array",
                "description": (
                    "Ordered list of grounded claim triples. "
                    "Each triple is either a paper triple or a taxonomy triple."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["paper", "taxonomy"],
                            "description": (
                                "'paper' for claims from PAPER EVIDENCE; "
                                "'taxonomy' for claims from TAXONOMY CONTEXT."
                            ),
                        },
                        "claim": {
                            "type": "string",
                            "description": "Your one-sentence statement derived from the source.",
                        },
                        "quote": {
                            "type": "string",
                            "description": (
                                "Verbatim passage from the paper that supports the claim. "
                                "Required for paper triples; omit for taxonomy triples."
                            ),
                        },
                        "citation": {
                            "type": "string",
                            "description": (
                                "'Surname et al., Year' exactly as shown in PAPER EVIDENCE. "
                                "Required for paper triples; omit for taxonomy triples."
                            ),
                        },
                        "concept": {
                            "type": "string",
                            "description": (
                                "Exact concept name from TAXONOMY CONTEXT. "
                                "Required for taxonomy triples; omit for paper triples."
                            ),
                        },
                    },
                    "required": ["type", "claim"],
                },
            },
        },
        "required": ["answer"],
    },
}


class AnthropicBackend(GenerationBackend):
    """
    Generates responses via the Anthropic Messages API.

    Uses ``tool_choice`` to force a single tool-use block, guaranteeing that
    the response is always a valid JSON object matching the output schema.

    Parameters
    ----------
    model : Anthropic model ID.  Defaults to the value in config.toml
            ([anthropic] model).
    """

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # noqa: PLC0415
        if model is None:
            from neurohive.config import load_config  # noqa: PLC0415
            model = load_config()["anthropic"].get("model")
        if not model:
            raise ValueError(
                "No Anthropic model specified. Set [anthropic] model in config.toml."
            )
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model

    def generate(self, query: str, taxonomy_ctx: str, paper_ctx: str) -> dict:
        user_msg = _user_prompt(query, taxonomy_ctx, paper_ctx)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[_ANTHROPIC_TOOL],
            tool_choice={"type": "tool", "name": "format_response"},
            messages=[{"role": "user", "content": user_msg}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return tool_block.input  # already a dict
