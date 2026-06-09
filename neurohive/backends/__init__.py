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

from neurohive.backends.base import GenerationBackend
from neurohive.backends.anthropic import AnthropicBackend
from neurohive.backends.ollama import OllamaBackend

__all__ = ["GenerationBackend", "AnthropicBackend", "OllamaBackend"]
