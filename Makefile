.PHONY: help setup setup-embeddings setup-nli setup-embeddings-nli download-model download-nli-model graph clean

## Show this help message
help:
	@echo "Neurohive -- available targets"
	@echo ""
	@echo "  Setup"
	@echo "    setup               Install core dependencies (Anthropic API)"
	@echo "    setup-embeddings    Install sentence-transformers + download embedding model"
	@echo "    setup-nli           Install NLI deps + download cross-encoder model"
	@echo "    setup-embeddings-nli Install both embedding and NLI models in one step"
	@echo "                        NLI verification is OFF by default; pass --verify to enable"
	@echo ""
	@echo "  Models"
	@echo "    download-model      Download embedding model from config.toml (needed for hybrid retrieval)"
	@echo "    download-nli-model  Install NLI deps + download cross-encoder model from config.toml"
	@echo ""
	@echo "  Other"
	@echo "    clean               Remove .venv, downloaded models/, and generated database"
	@echo "    help                Show this message"
	@echo ""
	@echo "  Quick start"
	@echo "    1. make setup"
	@echo "    2. echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env"
	@echo "    3. uv run python main.py \"What is synaptic plasticity?\""

## Install core dependencies (Anthropic API backend only)
setup:
	uv sync --no-dev
	uv pip install -e .

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
