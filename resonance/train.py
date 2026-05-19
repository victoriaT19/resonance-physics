"""QLoRA fine-tuning of Gemma 4 E2B with Unsloth (needs a CUDA GPU, e.g. a free T4).

What changed from the first version:
  * conversations are rendered with the tokenizer's chat template (Gemma 4 uses
    `<|turn>`/`<turn|>`; the old hand-written `<start_of_turn>` markers were plain text);
  * the loss covers only the answers, not the system prompt and the question;
  * LoRA goes on the language model only (the old adapter also covered the vision and
    audio towers, which text training never uses);
  * warm-up, evaluation and checkpoints are sized from the dataset, so they actually run.

Usage:
  python -m resonance.train --data data/generated --out outputs/adapter
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .data import load_jsonl
from .formatting import IGNORE_INDEX, tokenize_example
from .prompts import text_tokenizer

BASE_MODEL = "unsloth/gemma-4-E2B-it"


def tokenize_split(tokenizer, samples: list[dict], max_length: int) -> tuple[list[dict], int]:
    rows, dropped = [], 0
    for s in samples:
        msgs = s["messages"]
        row = tokenize_example(tokenizer, msgs[:-1], msgs[-1]["content"], max_length)
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    return rows, dropped


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/generated")
    p.add_argument("--out", default="outputs/adapter")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--warmup-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    # Unsloth must be imported before transformers to patch it.
    import torch
    from unsloth import FastModel
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    model, processor = FastModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_length,
        load_in_4bit=True,
    )
    tokenizer = text_tokenizer(processor)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    data = Path(args.data)
    train_rows, train_dropped = tokenize_split(tokenizer, load_jsonl(data / "train.jsonl"), args.max_length)
    val_rows, val_dropped = tokenize_split(tokenizer, load_jsonl(data / "validation.jsonl"), args.max_length)
    print(f"train: {len(train_rows)} samples ({train_dropped} too long) | "
          f"validation: {len(val_rows)} ({val_dropped} too long)")
    n_label_tokens = sum(sum(1 for t in r["labels"] if t != IGNORE_INDEX) for r in train_rows)
    print(f"answer tokens trained on: {n_label_tokens:,}")

    steps_per_epoch = math.ceil(len(train_rows) / (args.batch_size * args.grad_accum))
    total_steps = math.ceil(steps_per_epoch * args.epochs)
    warmup_steps = max(1, round(args.warmup_fraction * total_steps))
    print(f"{steps_per_epoch} optimizer steps per epoch, {total_steps} in total, {warmup_steps} warm-up")

    bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(Path(args.out).parent / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=0.3,
        optim="adamw_8bit",
        bf16=bf16,
        fp16=not bf16,
        logging_steps=max(1, steps_per_epoch // 10),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX),
    )
    result = trainer.train()
    final_eval = trainer.evaluate()

    out = Path(args.out)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    log = {
        "base_model": args.base_model,
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "dropped_too_long": train_dropped + val_dropped,
        "answer_tokens": n_label_tokens,
        "epochs": args.epochs,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "learning_rate": args.lr,
        "effective_batch": args.batch_size * args.grad_accum,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_loss": result.training_loss,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "best_eval_loss": final_eval.get("eval_loss"),
        "gpu": torch.cuda.get_device_name(0),
        "history": [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h],
    }
    (out / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Adapter saved to {out}; best validation loss {log['best_eval_loss']:.4f}")


if __name__ == "__main__":
    main()
