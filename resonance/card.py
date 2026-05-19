"""Write the Hugging Face model card from measured results, and optionally upload.

Every number in the card is read from files the pipeline wrote: the dataset stats
(data.py), the training log (train.py) and the evaluation table (evaluate.py).

Usage:
  python -m resonance.card --adapter outputs/adapter --data data/generated --eval results/v2
  python -m resonance.card ... --push vtaua/resonance-gemma4-physics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CARD = """---
base_model: {base_model}
library_name: peft
pipeline_tag: text-generation
language:
- pt
license: gemma
tags:
- lora
- qlora
- unsloth
- physics
- education
{dataset_meta}---

# Resonance: Gemma 4 E2B physics tutor (LoRA)

A LoRA adapter for [{base_model}](https://huggingface.co/{base_model}) that answers physics
questions in Portuguese in two levels: **Nível 1 — Intuição** (an intuitive explanation)
and **Nível 2 — Formal** (the derivation, with equations in LaTeX).

Code, data pipeline and evaluation: [github.com/victoriaT19/resonance-physics](https://github.com/victoriaT19/resonance-physics).

## How to use

```python
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel

base = "{base_model}"
tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForImageTextToText.from_pretrained(base, dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "{repo}")

SYSTEM_PROMPT = {system_repr}
messages = [
    {{"role": "system", "content": SYSTEM_PROMPT}},
    {{"role": "user", "content": "Derive o período de um pêndulo simples."}},
]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=1024)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The model was trained with this system prompt (`resonance/prompts.py`); use the same one.

## Training data

{data_section}

## Training

{train_section}

## Evaluation

20 held-out questions written by hand (`data/eval_questions.jsonl`), each with a reference
answer and its central equation. Topics of these questions were removed from the
training data. Base and fine-tuned answers come from the same weights, with the adapter
off and on, with greedy decoding.

{eval_section}

*Key-fact recall* checks whether the central equation appears in any of its usual
spellings; it does not check the derivation. {judge_note}

## Limitations

- A 2B-effective model makes physics mistakes, even after fine-tuning. Check every
  derivation before relying on it.
- The training answers were written by a larger model and filtered automatically{verify_note};
  they were not reviewed by a person.
- 20 evaluation questions are enough to see large effects only.
- Portuguese only.
"""


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_card(adapter: Path, data: Path | None, eval_dir: Path | None, repo: str,
               dataset_repo: str | None, verified: bool) -> str:
    log = _load(adapter / "training_log.json") or {}
    stats = _load(data / "stats.json") if data else None
    summary = _load(eval_dir / "summary.json") if eval_dir else None
    table = (eval_dir / "table.md").read_text(encoding="utf-8") if eval_dir and (eval_dir / "table.md").exists() else None

    if stats:
        src = ", ".join(f"{v} from {k}" for k, v in stats.get("by_source", {}).items())
        data_section = (
            f"{stats['accepted']} question/answer pairs ({stats['train']} train, "
            f"{stats['validation']} validation; {src}). Questions come from a syllabus of "
            f"physics topics and from the core concepts of recent arXiv abstracts. Answers were "
            f"written by `{stats['teacher']}` and kept only if they passed automatic format "
            f"checks (both levels, closed LaTeX, no truncation)."
        )
        if dataset_repo:
            data_section += f" Dataset: [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo})."
    else:
        data_section = "See `resonance/data.py`."

    if log:
        train_section = (
            f"QLoRA with Unsloth on one {log.get('gpu', 'GPU')}: rank {log['lora_r']}, alpha "
            f"{log['lora_alpha']}, on the language model's attention and MLP layers only. "
            f"{log['epochs']} epochs, {log['total_steps']} optimizer steps (effective batch "
            f"{log['effective_batch']}), learning rate {log['learning_rate']} with "
            f"{log['warmup_steps']} warm-up steps and cosine decay. The loss covers only the "
            f"answers ({log['answer_tokens']:,} tokens). Final training loss "
            f"{log['train_loss']:.3f}; best validation loss {log['best_eval_loss']:.3f}."
        )
    else:
        train_section = "See `resonance/train.py`."

    eval_section = table or "Not run yet."
    judge_note = ""
    if summary and summary.get("judge_model"):
        judge_note = (f"The judge grade is given by `{summary['judge_model']}`, which sees the "
                      f"reference answer.")

    dataset_meta = f"datasets:\n- {dataset_repo}\n" if dataset_repo else ""
    from .prompts import SYSTEM_PROMPT

    return CARD.format(
        base_model=log.get("base_model", "unsloth/gemma-4-E2B-it"), repo=repo,
        system_repr=repr(SYSTEM_PROMPT),
        dataset_meta=dataset_meta, data_section=data_section, train_section=train_section,
        eval_section=eval_section, judge_note=judge_note,
        verify_note=" and reviewed by the same model for physics errors" if verified else "",
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="outputs/adapter")
    p.add_argument("--data", default="data/generated")
    p.add_argument("--eval", default=None, help="evaluation folder with summary.json and table.md")
    p.add_argument("--repo", default="vtaua/resonance-gemma4-physics")
    p.add_argument("--dataset-repo", default=None)
    p.add_argument("--verified", action="store_true", help="the dataset was built with --verify")
    p.add_argument("--push", action="store_true", help="upload the adapter folder to --repo")
    args = p.parse_args(argv)

    adapter = Path(args.adapter)
    card = build_card(adapter, Path(args.data) if args.data else None,
                      Path(args.eval) if args.eval else None, args.repo,
                      args.dataset_repo, args.verified)
    (adapter / "README.md").write_text(card, encoding="utf-8")
    print(f"Model card written to {adapter.name}/README.md")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, exist_ok=True)
        api.upload_folder(folder_path=str(adapter), repo_id=args.repo,
                          commit_message="Resonance v2: chat-template fix, teacher data, evaluation",
                          ignore_patterns=["checkpoint-*", "*.pt"])
        print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
