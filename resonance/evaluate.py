"""Compare the base model and the fine-tuned model on held-out physics questions.

Both models answer the same 20 questions (data/eval_questions.jsonl) with greedy
decoding, from one load: the base answers come from the same weights with the adapter
switched off. Each answer gets:
  * format checks (both levels, no template leaks, no self-introduction, closed LaTeX,
    not cut off by the token limit);
  * key-fact recall: whether the central equation appears, in any of its usual spellings;
  * optionally, a 1-5 correctness grade from an LLM judge that sees a reference answer.

Usage:
  python -m resonance.evaluate --adapter outputs/adapter --out results/v2
  python -m resonance.evaluate --adapter outputs/adapter --out results/v2 --judge-model <model>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from .data import load_jsonl
from .quality import check_answer, key_fact_hits

JUDGE_PROMPT = """Você avalia respostas de um tutor de física.

Pergunta: {question}

Resposta de referência (correta): {reference}

Resposta do tutor:
{answer}

Dê uma nota de 1 a 5 para a CORREÇÃO FÍSICA da resposta do tutor, comparando com a
referência. Ignore estilo e tamanho.
5 = correta e completa; 4 = correta com omissões pequenas; 3 = ideia certa com um erro
de equação, sinal ou fator; 2 = erros importantes; 1 = errada ou sem relação.
Responda exatamente neste formato:
NOTA: <1-5>
MOTIVO: <uma frase>"""


def score_answer(item: dict, text: str, truncated: bool) -> dict:
    checks = check_answer(text, truncated=truncated)
    hits = key_fact_hits(text, item["key_facts"])
    return {**checks, "key_fact_recall": sum(hits) / len(hits)}


def parse_judge(text: str) -> tuple[int | None, str]:
    # Tolerates Markdown such as "**NOTA:** 4" or "**NOTA**: 4".
    m = re.search(r"NOTA\W{0,4}([1-5])", text, flags=re.IGNORECASE)
    reason = re.search(r"MOTIVO\W{0,4}(.+)", text, flags=re.IGNORECASE)
    return (int(m.group(1)) if m else None), (reason.group(1).strip() if reason else text.strip())


def judge_rows(judge, items: list[dict], rows: list[dict], model_name: str, workers: int = 8) -> None:
    """Add a judge grade to each row, in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    by_id = {it["id"]: it for it in items}

    def one(row):
        item = by_id[row["id"]]
        text, _ = judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(
            question=item["question"], reference=item["reference"], answer=row["answer"])}],
            max_tokens=600)
        grade, reason = parse_judge(text)
        row["judge"] = {"model": model_name, "grade": grade, "reason": reason}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, rows))


def write_report(out: Path, results: dict[str, list[dict]], meta: dict) -> str:
    summaries = {label: summarize(rows) for label, rows in results.items()}
    comparisons = {}
    if "base" in results and "tuned" in results:
        comparisons["Key-fact recall"] = paired(results["base"], results["tuned"],
                                                lambda r: r["scores"]["key_fact_recall"])
        comparisons["Format checks passed"] = paired(
            results["base"], results["tuned"],
            lambda r: sum(r["scores"][k] for k in ("both_levels", "no_artifacts", "no_self_intro",
                                                   "balanced_math", "complete")))
        if any(r.get("judge", {}).get("grade") for r in results["tuned"]):
            comparisons["Judge grade"] = paired(results["base"], results["tuned"],
                                                lambda r: r.get("judge", {}).get("grade"))
    summary = {**meta, "summaries": summaries, "comparisons": comparisons}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table = markdown_table(summaries, comparisons)
    (out / "table.md").write_text(table, encoding="utf-8")
    return table


def rejudge(args) -> None:
    """Grade saved answers again (no GPU): python -m resonance.evaluate --rejudge results/v2"""
    from .data import Teacher

    out = Path(args.rejudge)
    items = load_jsonl(args.questions)
    judge = Teacher(model=args.judge_model, base_url=args.judge_base_url,
                    api_key=os.environ.get(args.judge_key_env), temperature=0.0, thinking="minimal")
    results = {}
    for label in ("base", "tuned"):
        path = out / f"answers_{label}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        judge_rows(judge, items, rows, args.judge_model)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        missing = sum(1 for r in rows if r["judge"]["grade"] is None)
        print(f"{label}: judged {len(rows)}, unparsed {missing}")
        results[label] = rows
    old = json.loads((out / "summary.json").read_text(encoding="utf-8")) if (out / "summary.json").exists() else {}
    meta = {k: v for k, v in old.items() if k not in ("summaries", "comparisons")}
    meta["judge_model"] = args.judge_model
    print(write_report(out, results, meta))


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test p-value (ties excluded)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    keys = ["ok", "both_levels", "no_artifacts", "no_self_intro", "balanced_math", "complete"]
    out = {k: sum(r["scores"][k] for r in rows) / n for k in keys}
    out["key_fact_recall"] = sum(r["scores"]["key_fact_recall"] for r in rows) / n
    out["mean_words"] = sum(r["scores"]["words"] for r in rows) / n
    grades = [r["judge"]["grade"] for r in rows if r.get("judge", {}).get("grade")]
    if grades:
        out["judge_mean"] = sum(grades) / len(grades)
        out["judge_n"] = len(grades)
    return out


def paired(base: list[dict], tuned: list[dict], metric) -> dict:
    wins = losses = ties = 0
    for b, t in zip(base, tuned):
        vb, vt = metric(b), metric(t)
        if vb is None or vt is None:
            continue
        if vt > vb:
            wins += 1
        elif vt < vb:
            losses += 1
        else:
            ties += 1
    return {"tuned_better": wins, "base_better": losses, "ties": ties,
            "sign_test_p": round(sign_test(wins, losses), 4)}


def markdown_table(summaries: dict[str, dict], comparisons: dict[str, dict]) -> str:
    labels = list(summaries)
    rows = [
        ("Answers passing every format check", "ok", "{:.0%}"),
        ("Both levels present", "both_levels", "{:.0%}"),
        ("No template leaks", "no_artifacts", "{:.0%}"),
        ("No self-introduction", "no_self_intro", "{:.0%}"),
        ("Closed LaTeX delimiters", "balanced_math", "{:.0%}"),
        ("Finished before the token limit", "complete", "{:.0%}"),
        ("Key-fact recall", "key_fact_recall", "{:.0%}"),
        ("Judge grade (1-5)", "judge_mean", "{:.2f}"),
        ("Mean length (words)", "mean_words", "{:.0f}"),
    ]
    lines = ["| Metric | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for title, key, fmt in rows:
        if not any(key in summaries[lb] for lb in labels):
            continue
        cells = [fmt.format(summaries[lb][key]) if key in summaries[lb] else "-" for lb in labels]
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    if comparisons:
        lines += ["", "| Paired, per question | tuned better | base better | ties | sign test p |",
                  "|---|---|---|---|---|"]
        for name, c in comparisons.items():
            lines.append(f"| {name} | {c['tuned_better']} | {c['base_better']} | {c['ties']} | {c['sign_test_p']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from .env import load_env, provider

    load_env()
    j_model, j_url, j_key_env = provider("JUDGE")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", default="outputs/adapter",
                   help="local folder or Hub id of the LoRA adapter")
    p.add_argument("--base-model", default=None)
    p.add_argument("--questions", default="data/eval_questions.jsonl")
    p.add_argument("--out", default="results/latest")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--judge-model", default=j_model)
    p.add_argument("--judge-base-url", default=j_url)
    p.add_argument("--judge-key-env", default=j_key_env)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--rejudge", default=None,
                   help="grade the saved answers in this folder again, without a GPU")
    args = p.parse_args(argv)

    if args.rejudge:
        rejudge(args)
        return

    from .model import BASE_MODEL, GenerationConfig, Tutor

    items = load_jsonl(args.questions)
    tutor = Tutor(base_model=args.base_model or BASE_MODEL, adapter=args.adapter,
                  load_in_4bit=args.load_in_4bit)
    cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, temperature=0.0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [("tuned", True)] if args.skip_base else [("base", False), ("tuned", True)]
    results: dict[str, list[dict]] = {}
    for label, use_adapter in runs:
        rows = []
        for i, item in enumerate(items):
            gen = tutor.generate(item["question"], cfg=cfg, use_adapter=use_adapter)
            row = {"id": item["id"], "question": item["question"], "answer": gen.text,
                   "new_tokens": gen.new_tokens, "truncated": gen.truncated,
                   "scores": score_answer(item, gen.text, gen.truncated)}
            rows.append(row)
            print(f"[{label} {i + 1}/{len(items)}] {item['id']}: ok={row['scores']['ok']} "
                  f"facts={row['scores']['key_fact_recall']:.0%} tokens={gen.new_tokens}")
        results[label] = rows

    # The judge runs after generation, in parallel, so its API latency does not hold the GPU.
    if args.no_judge:
        args.judge_model = None
    if args.judge_model:
        from .data import Teacher

        judge = Teacher(model=args.judge_model, base_url=args.judge_base_url,
                        api_key=os.environ.get(args.judge_key_env), temperature=0.0,
                        thinking="minimal")
        for rows in results.values():
            judge_rows(judge, items, rows, args.judge_model)

    for label, rows in results.items():
        with open(out / f"answers_{label}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "adapter": args.adapter if not os.path.isdir(args.adapter) else Path(args.adapter).name,
        "base_model": args.base_model or BASE_MODEL,
        "questions": len(items),
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens},
        "judge_model": args.judge_model,
    }
    print(write_report(out, results, meta))


if __name__ == "__main__":
    main()
