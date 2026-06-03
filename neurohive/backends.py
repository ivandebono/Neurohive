"""
Generation backends for QueryPipeline.

AnthropicBackend
    Uses the Anthropic Messages API with forced tool use (tool_choice) to
    guarantee a valid JSON response matching the output schema.
    Requires: ANTHROPIC_API_KEY environment variable.

OllamaBackend
    Uses a local Ollama server and an open-source model. It requests JSON
    output and validates the returned shape before handing it to the pipeline.

The backend returns::

    {"answer": list[dict]}   # ordered list of paper/taxonomy triple dicts

Model selection
---------------
Model names are configured in config.toml under [anthropic] and [ollama].
Passing a model string directly to the constructor overrides config.toml.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from urllib import error, request


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
        "Do not use any knowledge outside these two sections. "
        "Return only a JSON object with one top-level key, \"answer\"."
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


# ---------------------------------------------------------------------------
# Ollama backend  (local open-source model)
# ---------------------------------------------------------------------------

class OllamaBackend(GenerationBackend):
    """
    Generates responses through a local Ollama server.

    Parameters
    ----------
    model : Ollama model name. Defaults to config.toml ([ollama] model).
    host  : Ollama base URL. Defaults to config.toml ([ollama] host).
    """

    def __init__(self, model: str | None = None, host: str | None = None) -> None:
        if model is None or host is None:
            from neurohive.config import load_config  # noqa: PLC0415
            cfg = load_config()["ollama"]
            model = model or cfg["model"]
            host = host or cfg["host"]
        self._model = model
        self._host = host.rstrip("/")

    @property
    def model_id(self) -> str:
        return f"ollama/{self._model}"

    def generate(self, query: str, taxonomy_ctx: str, paper_ctx: str) -> dict:
        user_msg = _user_prompt(query, taxonomy_ctx, paper_ctx)
        payload = {
            "model": self._model,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "options": {
                "temperature": 0,
            },
        }
        req = request.Request(
            f"{self._host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            content = raw.get("message", {}).get("content", "")
            return _validate_structured_response(_parse_json_object(content))
        except (OSError, error.URLError, json.JSONDecodeError, KeyError, TypeError) as exc:
            return _error_response(
                "Ollama generation failed. Ensure Ollama is running and the model "
                f"'{self._model}' is pulled. Error: {exc}"
            )
