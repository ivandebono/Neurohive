.PHONY: help setup setup-ollama setup-embeddings setup-nli setup-embeddings-nli setup-ollama-embeddings-nli download-model download-nli-model pull-ollama-model graph clean

## Show this help message
help:
	@echo "Neurohive -- available targets"
	@echo ""
	@echo "  Setup"
	@echo "    setup               Install core dependencies (Anthropic API or Ollama)"
	@echo "    setup-ollama        Install core deps + pull configured local Ollama model"
	@echo "    setup-embeddings    Install sentence-transformers + download embedding model"
	@echo "    setup-nli           Install NLI deps + download cross-encoder model"
	@echo "    setup-embeddings-nli Install both embedding and NLI models in one step"
	@echo "    setup-ollama-embeddings-nli Install Ollama model + embedding and NLI models"
	@echo "                        NLI verification is OFF by default; pass --verify to enable"
	@echo ""
	@echo "  Models"
	@echo "    pull-ollama-model   Pull local Ollama generation model from config.toml"
	@echo "    download-model      Download embedding model from config.toml (needed for hybrid retrieval)"
	@echo "    download-nli-model  Install NLI deps + download cross-encoder model from config.toml"
	@echo ""
	@echo "  Other"
	@echo "    clean               Remove .venv, downloaded models/, and generated database"
	@echo "    help                Show this message"
	@echo ""
	@echo "  Quick start"
	@echo "    Anthropic:"
	@echo "      1. make setup"
	@echo "      2. echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env"
	@echo "      3. uv run python main.py \"What is synaptic plasticity?\""
	@echo "    Ollama:"
	@echo "      1. make setup-ollama"
	@echo "      2. uv run python main.py --ollama \"What is synaptic plasticity?\""

## Install core dependencies (generation can use Anthropic API or local Ollama)
setup:
	uv sync --no-dev
	uv pip install -e .

## Install core dependencies and pull the configured local Ollama generation model
setup-ollama: setup pull-ollama-model

## Pull local Ollama generation model from config.toml
pull-ollama-model:
	uv run python scripts/setup_ollama_model.py

## Add sentence-transformers for hybrid BM25 + dense retrieval, then download the model
setup-embeddings:
	uv sync --extra embeddings
	uv pip install -e .
	uv run python scripts/download_model.py

## Install both sentence-transformers and NLI deps, then download both models
setup-embeddings-nli:
	uv sync --extra embeddings --extra nli
	uv pip install -e .
	uv run python scripts/download_model.py
	uv run python scripts/download_nli_model.py

## Install local Ollama generation model plus embedding and NLI models
setup-ollama-embeddings-nli:
	uv sync --extra embeddings --extra nli
	uv pip install -e .
	uv run python scripts/setup_ollama_model.py
	uv run python scripts/download_model.py
	uv run python scripts/download_nli_model.py

## Download embedding model from config.toml and enable hybrid retrieval automatically
download-model:
	uv run python scripts/download_model.py

## Install NLI deps + download cross-encoder model (NLI verification is OFF by default; pass --verify to enable)
setup-nli: download-nli-model

## Install NLI dependencies and download the cross-encoder model from config.toml
download-nli-model:
	uv sync --extra nli
	uv pip install -e .
	uv run python scripts/download_nli_model.py

## Open the taxonomy graph visualisation
graph:
	@case "$$(uname -s)" in \
	  Darwin) open assets/taxonomy_graph.png ;; \
	  Linux)  xdg-open assets/taxonomy_graph.png ;; \
	  *)      echo "Open assets/taxonomy_graph.png manually" ;; \
	esac

## Remove .venv, downloaded models/, and generated database
clean:
	rm -rf .venv
	rm -rf models/
	rm -rf data/database/
