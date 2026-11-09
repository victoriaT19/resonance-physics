"""Publish the Gradio demo as a Hugging Face Space (ZeroGPU when available).

Uploads app.py, the resonance package and a Space README; the model weights are not
uploaded: the Space downloads Gemma 4 E2B and the adapter from the Hub at start-up.

Usage (HF_TOKEN with write access in the environment or in .env):
  python scripts/deploy_space.py --repo vtaua/resonance-physics-demo
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPACE_README = """---
title: Resonance — tutor de física
emoji: ⚛️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: gemma
short_description: Gemma 4 E2B physics tutor in two levels, with simulations
models:
- {adapter}
- {base}
---

# Resonance: tutor de física em dois níveis

Gemma 4 E2B com o adaptador LoRA [{adapter}](https://huggingface.co/{adapter}): responde em
português, primeiro com a intuição e depois com a derivação. Ao lado do chat, três
simulações (oscilador amortecido, pacote de onda, órbita kepleriana) com parâmetros
ajustáveis.

Um modelo de 2B parâmetros efetivos ainda erra física: confira as derivações. Código,
dados e avaliação: [github.com/victoriaT19/resonance-physics](https://github.com/victoriaT19/resonance-physics).
"""

SPACE_REQUIREMENTS = """torch
transformers>=5.5
peft>=0.19
accelerate>=1.0
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
"""


def main() -> None:
    from huggingface_hub import HfApi

    from resonance.env import load_env
    from resonance.model import ADAPTER, BASE_MODEL

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="vtaua/resonance-physics-demo")
    p.add_argument("--hardware", default="zero-a10g",
                   help="zero-a10g (ZeroGPU) or cpu-basic; ZeroGPU may need a PRO account")
    p.add_argument("--gradio-version", default=None)
    args = p.parse_args()

    load_env(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set (environment or .env)")

    import gradio

    gradio_version = args.gradio_version or gradio.__version__
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        shutil.copy(ROOT / "app.py", out / "app.py")
        shutil.copytree(ROOT / "resonance", out / "resonance",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (out / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
        (out / "README.md").write_text(SPACE_README.format(
            gradio_version=gradio_version, adapter=ADAPTER, base=BASE_MODEL), encoding="utf-8")

        # Creating a Gradio Space through the API can be refused for free accounts (HTTP 402)
        # while the website still allows it; create it on the website first in that case.
        if not api.repo_exists(args.repo, repo_type="space"):
            api.create_repo(args.repo, repo_type="space", space_sdk="gradio")
        try:
            api.request_space_hardware(args.repo, args.hardware)
            print(f"hardware requested: {args.hardware}")
        except Exception as exc:  # ZeroGPU can require a PRO subscription
            print(f"could not request {args.hardware} ({type(exc).__name__}: {exc}); "
                  "the Space keeps its current hardware")
        api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="space",
                          commit_message="Deploy the Resonance demo")
    print(f"https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()
