"""Read API keys from a local .env file and pick provider defaults.

Keys never go in the code or in the notebook: put them in `.env` at the repository root
(ignored by git), for example

    GEMINI_API_KEY=...

With only GEMINI_API_KEY set, the teacher and the judge default to Gemma 4 31B through
the Gemini API's OpenAI-compatible endpoint. TEACHER_* / JUDGE_* variables override it.
"""

from __future__ import annotations

import os
from pathlib import Path

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_GEMMA_MODEL = "gemma-4-31b-it"


def load_env(path: str | Path = ".env") -> None:
    """Set variables from KEY=VALUE lines, without overriding ones already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def provider(prefix: str) -> tuple[str | None, str | None, str]:
    """Return (model, base_url, key_env) for prefix 'TEACHER' or 'JUDGE'."""
    model = os.environ.get(f"{prefix}_MODEL")
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    key_env = f"{prefix}_API_KEY"
    if not os.environ.get(key_env) and os.environ.get("GEMINI_API_KEY"):
        key_env = "GEMINI_API_KEY"
        model = model or GEMINI_GEMMA_MODEL
        base_url = base_url or GEMINI_OPENAI_URL
    return model, base_url, key_env
