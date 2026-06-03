#!/usr/bin/env python3
"""
Pull the configured local Ollama generation model.

The model name is read from config.toml ([ollama] model). Ollama itself must be
installed and the local Ollama service should be running.

Usage
-----
    uv run python scripts/setup_ollama_model.py
"""

from __future__ import annotations

import shutil
import subprocess

from neurohive.config import load_config


def main() -> None:
    cfg = load_config()
    model = cfg["ollama"]["model"]

    if shutil.which("ollama") is None:
        print("Ollama CLI not found.")
        print("Install Ollama from https://ollama.com, start it, then rerun this command.")
        raise SystemExit(1)

    print(f"Pulling Ollama model: {model}")
    try:
        subprocess.run(["ollama", "pull", model], check=True)
    except subprocess.CalledProcessError as exc:
        print("Failed to pull the Ollama model.")
        print("Check that Ollama is installed and running, then retry.")
        raise SystemExit(exc.returncode) from exc
    print(f"Ollama model ready: {model}")


if __name__ == "__main__":
    main()
