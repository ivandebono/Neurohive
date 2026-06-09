"""
Ollama backend for local open-source model inference.

Uses a local Ollama server and requests JSON output, validating the returned
shape before handing it to the pipeline.

Model and host are configured in config.toml under [ollama].
"""

from __future__ import annotations

import json
from urllib import error, request

from neurohive.backends.base import (
    GenerationBackend,
    _SYSTEM_PROMPT,
    _error_response,
    _parse_json_object,
    _user_prompt,
    _validate_structured_response,
)


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
            ollama_cfg = load_config().ollama
            model = model or ollama_cfg.model
            host = host or ollama_cfg.host
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
