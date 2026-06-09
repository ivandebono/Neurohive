"""
Anthropic Messages API backend.

Uses ``tool_choice`` to force a single tool-use block, guaranteeing that the
response is always a valid JSON object matching the output schema.

Requires: ANTHROPIC_API_KEY environment variable.
Model is configured in config.toml under [anthropic] model.
"""

from __future__ import annotations

from neurohive.backends.base import GenerationBackend, _SYSTEM_PROMPT, _user_prompt


_ANTHROPIC_TOOL: dict = {
    "name": "format_response",
    "description": (
        "Return an answer derived exclusively from the TAXONOMY CONTEXT and "
        "PAPER EVIDENCE supplied in the user message. Fill \"reasoning\" first "
        "to identify the most relevant concepts and papers, then express every "
        "claim as a structured triple traceable to those two sections. "
        "Do not use training knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "2–4 sentences of pre-answer analysis: (a) the 1–3 taxonomy "
                    "concepts most directly relevant to the query, (b) the 1–3 "
                    "papers providing the strongest evidence, and (c) any aspect "
                    "of the query not covered by the provided context."
                ),
            },
            "answer": {
                "type": "array",
                "description": (
                    "Ordered list of grounded claim triples. "
                    "Each triple is either a paper triple or a taxonomy triple."
                ),
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "description": "A claim grounded in PAPER EVIDENCE.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "const": "paper",
                                    "description": "Use for claims from PAPER EVIDENCE.",
                                },
                                "claim": {
                                    "type": "string",
                                    "description": "Your one-sentence statement derived from the source.",
                                },
                                "quote": {
                                    "type": "string",
                                    "description": "Verbatim passage from the paper that supports the claim.",
                                },
                                "citation": {
                                    "type": "string",
                                    "description": "'Surname et al., Year' exactly as shown in PAPER EVIDENCE.",
                                },
                            },
                            "required": ["type", "claim", "quote", "citation"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "description": "A claim grounded in TAXONOMY CONTEXT.",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "const": "taxonomy",
                                    "description": "Use for claims from TAXONOMY CONTEXT.",
                                },
                                "claim": {
                                    "type": "string",
                                    "description": "Your one-sentence statement derived from the source.",
                                },
                                "concept": {
                                    "type": "string",
                                    "description": "Exact concept name from TAXONOMY CONTEXT.",
                                },
                            },
                            "required": ["type", "claim", "concept"],
                            "additionalProperties": False,
                        },
                    ],
                },
            },
        },
        "required": ["reasoning", "answer"],
    },
}


class AnthropicBackend(GenerationBackend):
    """
    Generates responses via the Anthropic Messages API.

    Parameters
    ----------
    model : Anthropic model ID.  Defaults to the value in config.toml
            ([anthropic] model).
    """

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # noqa: PLC0415
        if model is None:
            from neurohive.config import load_config  # noqa: PLC0415
            model = load_config().anthropic.model
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
