"""Tests for generation backends."""

from __future__ import annotations

import json

from neurohive.backends import OllamaBackend


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_ollama_backend_parses_structured_json(monkeypatch):
    def fake_urlopen(req, timeout):
        content = json.dumps({
            "answer": [{
                "type": "taxonomy",
                "claim": "The context covers action potentials.",
                "concept": "Action_Potential",
            }]
        })
        return _FakeResponse({"message": {"content": content}})

    monkeypatch.setattr("neurohive.backends.request.urlopen", fake_urlopen)

    backend = OllamaBackend(model="test-model", host="http://localhost:11434")
    result = backend.generate("q", "taxonomy", "paper")

    assert result["answer"][0]["type"] == "taxonomy"
    assert result["answer"][0]["concept"] == "Action_Potential"
    assert backend.model_id == "ollama/test-model"


def test_ollama_backend_returns_error_triple_on_failure(monkeypatch):
    def fake_urlopen(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr("neurohive.backends.request.urlopen", fake_urlopen)

    backend = OllamaBackend(model="test-model", host="http://localhost:11434")
    result = backend.generate("q", "taxonomy", "paper")

    assert result["answer"][0]["type"] == "taxonomy"
    assert "Ollama generation failed" in result["answer"][0]["claim"]
