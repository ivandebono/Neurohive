#!/usr/bin/env python3
"""
Download the NLI cross-encoder model for post-generation entailment verification.

The model ID is read from config.toml ([nli] model). The model is saved under
config.toml [paths] models_dir, where the pipeline picks it up automatically
when --verify is passed.

Usage
-----
    uv run python scripts/download_nli_model.py
"""

from pathlib import Path

from neurohive.config import load_config

_REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    cfg = load_config()
    hf_model_id: str = cfg["nli"].get("model", "cross-encoder/nli-deberta-v3-small")

    model_name = hf_model_id.split("/")[-1]
    models_dir = Path(cfg["paths"]["models_dir"])
    if not models_dir.is_absolute():
        models_dir = _REPO_ROOT / models_dir
    model_dir = models_dir / model_name

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"NLI model already present at {model_dir}")
        print("Entailment checking is ready. Nothing to do.")
        return

    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415
    except ImportError:
        print("sentence-transformers is not installed.")
        print("Run:  uv sync --extra nli")
        raise SystemExit(1)

    print(f"Downloading {hf_model_id} (~568 MB for nli-deberta-v3-small) …")
    model = CrossEncoder(hf_model_id, num_labels=3)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    print(f"Saved to {model_dir}")
    print("Pass --verify to enable NLI verification at runtime.")


if __name__ == "__main__":
    main()
