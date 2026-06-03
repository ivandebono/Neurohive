#!/usr/bin/env python3
"""
Download the sentence-transformers embedding model for hybrid BM25 + dense retrieval.

The model ID is read from config.toml ([embeddings] model). The model is saved
under config.toml [paths] models_dir, where the Retriever picks it up
automatically.

Usage
-----
    uv run python scripts/download_model.py
"""

from pathlib import Path

from neurohive.config import load_config

_REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    cfg = load_config()
    hf_model_id: str = cfg["embeddings"]["model"]

    # Derive the local directory name from the model ID (strip the org prefix).
    model_name = hf_model_id.split("/")[-1]
    models_dir = Path(cfg["paths"]["models_dir"])
    if not models_dir.is_absolute():
        models_dir = _REPO_ROOT / models_dir
    model_dir = models_dir / model_name

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"Model already present at {model_dir}")
        print("Hybrid retrieval is active. Nothing to do.")
        return

    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    except ImportError:
        print("sentence-transformers is not installed.")
        print("Run:  uv sync --extra embeddings")
        raise SystemExit(1)

    print(f"Downloading {hf_model_id} …")
    model = SentenceTransformer(hf_model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    print(f"Saved to {model_dir}")
    print("Hybrid retrieval (BM25 + dense, RRF) will now be used automatically.")


if __name__ == "__main__":
    main()
