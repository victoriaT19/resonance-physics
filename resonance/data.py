"""Build the training set: physics questions answered by a stronger teacher model.

The first version had the base model answer its own questions, so fine-tuning could only
copy its format and its mistakes. Here a larger model (the teacher) writes the answers
through any OpenAI-compatible API, every answer passes the format checks in quality.py,
and an optional second pass asks the teacher to review each answer for physics errors.

Questions come from two sources:
  * topics: a syllabus of physics topics (data/topics.txt);
  * arxiv: the core concept behind recent arXiv abstracts in five categories.

Anything that overlaps the evaluation questions is dropped, so the evaluation stays
held out.

Usage:
  python -m resonance.data build --out data/generated
  python -m resonance.data push --data data/generated --repo <user>/resonance-physics-qa
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .prompts import SYSTEM_PROMPT
from .quality import acceptable_training_answer

ARXIV_CATEGORIES = [
    "physics.optics",
    "physics.class-ph",
    "astro-ph.SR",
    "astro-ph.GA",
    "physics.ed-ph",
]

QUESTION_STYLES = [
    "uma pergunta conceitual (por que / como acontece)",
    "um pedido de derivação de uma equação ou resultado",
    "uma pergunta com um cálculo numérico simples e valores realistas",
    "uma pergunta que compara dois fenômenos ou dois regimes",
]

TOPIC_QUESTIONS_PROMPT = """Você escreve perguntas de física para estudantes de graduação.
Tópico: {topic}

Escreva {n} perguntas diferentes, em português, sobre esse tópico. Varie o tipo:
{styles}
Cada pergunta deve ser autocontida, ter uma resposta bem estabelecida em livros-texto e
caber em uma ou duas frases. Responda só com as perguntas, uma por linha, sem numeração."""

ARXIV_QUESTION_PROMPT = """Abaixo está o resumo de um artigo de física.

Título: {title}
Resumo: {abstract}

Identifique o conceito físico fundamental, de nível de graduação, de que esse artigo
depende (por exemplo, difração, equilíbrio hidrostático, ressonância). Escreva UMA
pergunta, em português, que um estudante faria sobre esse conceito. A pergunta não deve
citar o artigo e deve ter resposta bem estabelecida em livros-texto. Responda só com a
pergunta."""

ANSWER_GUIDE = """

Regras adicionais para esta resposta:
- Use exatamente os cabeçalhos **Nível 1 — Intuição:** e **Nível 2 — Formal:**.
- Seja fisicamente correto. Defina cada símbolo na primeira vez que aparecer.
- No Nível 2, chegue ao resultado principal e destaque-o.
- Entre 150 e 600 palavras no total."""

REVIEW_PROMPT = """Você é um professor de física revisando a resposta de um tutor.

Pergunta: {question}

Resposta do tutor:
{answer}

A resposta tem algum erro de física, equação errada, sinal ou fator errado, ou conclusão
incorreta? Ignore estilo. Responda apenas "OK" ou "ERRO: <motivo em uma frase>"."""


# ── Teacher client ──────────────────────────────────────────────────────────

_THINKING = [
    re.compile(r"<\|channel>thought.*?<channel\|>", re.DOTALL),
    re.compile(r"<thought>.*?</thought>", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL),
]


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks some providers return inside the message text."""
    for pat in _THINKING:
        text = pat.sub("", text)
    return text.strip()


class Teacher:
    """A chat model behind an OpenAI-compatible endpoint, with retries. Thread-safe."""

    def __init__(self, model: str, base_url: str | None, api_key: str | None,
                 temperature: float = 0.4, min_interval: float = 0.0,
                 thinking: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self.model = model
        self.temperature = temperature
        self.min_interval = min_interval
        self._last_call = 0.0
        self._lock = threading.Lock()
        # Gemini API: Gemma 4 thinks by default, which multiplies latency. "minimal" turns
        # it down; the OpenAI `reasoning_effort` field is rejected for Gemma.
        self.extra_body = None
        if thinking and base_url and "generativelanguage.googleapis.com" in base_url:
            self.extra_body = {"extra_body": {"google": {"thinking_config": {"thinking_level": thinking}}}}

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()

    def chat(self, messages: list[dict], max_tokens: int = 2048,
             temperature: float | None = None) -> tuple[str, str | None]:
        """Return (text, finish_reason)."""
        delay = 5.0
        for attempt in range(6):
            self._throttle()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature if temperature is None else temperature,
                    extra_body=self.extra_body,
                )
                choice = resp.choices[0]
                return strip_thinking(choice.message.content or ""), choice.finish_reason
            except Exception as exc:  # rate limits and transient server errors
                if attempt == 5:
                    raise
                print(f"  teacher error ({type(exc).__name__}: {exc}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("unreachable")


# ── Held-out filter ─────────────────────────────────────────────────────────


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 3}


class HeldOutFilter:
    """Rejects questions that touch an evaluation question's topic."""

    def __init__(self, eval_items: list[dict], max_jaccard: float = 0.4):
        self.terms = sorted({t.lower() for item in eval_items for t in item.get("exclude_terms", [])})
        self.eval_tokens = [_tokens(item["question"]) for item in eval_items]
        self.max_jaccard = max_jaccard

    def reason(self, text: str) -> str | None:
        low = text.lower()
        for term in self.terms:
            if term in low:
                return f"eval_term:{term}"
        toks = _tokens(text)
        for et in self.eval_tokens:
            union = toks | et
            if union and len(toks & et) / len(union) > self.max_jaccard:
                return "eval_overlap"
        return None


# ── Question sources ────────────────────────────────────────────────────────


def load_topics(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _clean_question(line: str) -> str:
    line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip().strip('"').strip()
    return line


def topic_questions(teacher: Teacher, topic: str, n: int, rng: random.Random) -> list[str]:
    styles = "\n".join(f"- {s}" for s in rng.sample(QUESTION_STYLES, k=len(QUESTION_STYLES)))
    text, _ = teacher.chat(
        [{"role": "user", "content": TOPIC_QUESTIONS_PROMPT.format(topic=topic, n=n, styles=styles)}],
        max_tokens=600,
        temperature=0.9,
    )
    qs = [_clean_question(ln) for ln in text.splitlines()]
    return [q for q in qs if 15 <= len(q) <= 400 and q.endswith(("?", "."))][:n]


def fetch_arxiv(per_category: int, seed: int) -> list[dict]:
    import arxiv

    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    papers, seen = [], set()
    for cat in ARXIV_CATEGORIES:
        search = arxiv.Search(query=f"cat:{cat}", max_results=per_category,
                              sort_by=arxiv.SortCriterion.SubmittedDate)
        for r in client.results(search):
            pid = r.entry_id.split("/")[-1]
            if pid in seen:
                continue
            abstract = " ".join(r.summary.split())
            if len(abstract.split()) < 60:
                continue
            seen.add(pid)
            papers.append({"id": pid, "title": " ".join(r.title.split()),
                           "abstract": abstract, "category": cat})
    random.Random(seed).shuffle(papers)
    return papers


def arxiv_question(teacher: Teacher, paper: dict) -> str:
    text, _ = teacher.chat(
        [{"role": "user", "content": ARXIV_QUESTION_PROMPT.format(
            title=paper["title"], abstract=paper["abstract"][:2500])}],
        max_tokens=200,
        temperature=0.7,
    )
    lines = [_clean_question(ln) for ln in text.splitlines() if ln.strip()]
    return lines[0] if lines else ""


# ── Answers ─────────────────────────────────────────────────────────────────


def teacher_answer(teacher: Teacher, question: str) -> tuple[str, str | None]:
    return teacher.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT + ANSWER_GUIDE},
            {"role": "user", "content": question},
        ],
        max_tokens=3072,
    )


def review_answer(teacher: Teacher, question: str, answer: str) -> tuple[bool, str]:
    text, _ = teacher.chat(
        [{"role": "user", "content": REVIEW_PROMPT.format(question=question, answer=answer)}],
        max_tokens=120,
        temperature=0.0,
    )
    verdict = text.strip()
    return verdict.upper().startswith("OK"), verdict


def make_sample(question: str, answer: str, metadata: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "metadata": metadata,
    }


# ── Build ───────────────────────────────────────────────────────────────────


def build(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "cache.jsonl"

    eval_items = load_jsonl(args.eval_questions)
    held_out = HeldOutFilter(eval_items)
    teacher = Teacher(
        model=args.teacher_model,
        base_url=args.teacher_base_url,
        api_key=os.environ.get(args.teacher_key_env),
        min_interval=args.min_interval,
        thinking=args.thinking,
    )

    # The cache holds every attempt (accepted or not), so a build can resume.
    done: dict[str, dict] = {}
    if cache_path.exists():
        for rec in load_jsonl(cache_path):
            done[rec["key"]] = rec
    cache = open(cache_path, "a", encoding="utf-8")
    lock = threading.Lock()

    def record(rec: dict) -> None:
        with lock:
            done[rec["key"]] = rec
            cache.write(json.dumps(rec, ensure_ascii=False) + "\n")
            cache.flush()

    def run_parallel(label: str, jobs: list, fn) -> None:
        """Run fn(job) on a thread pool; a job that keeps failing is skipped (retried next run)."""
        if not jobs:
            return
        t0, finished = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(fn, job): job for job in jobs}
            for fut in as_completed(futures):
                finished += 1
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  {label}: job failed after retries ({type(exc).__name__}: {exc})")
                if finished % 20 == 0 or finished == len(jobs):
                    rate = finished / max(time.time() - t0, 1e-9) * 60
                    print(f"  {label}: {finished}/{len(jobs)} ({rate:.1f}/min)", flush=True)

    def process(job: tuple[str, str, dict]) -> None:
        key, question, metadata = job
        rec = {"key": key, "question": question, "metadata": metadata}
        why = held_out.reason(question)
        if why:
            rec.update(status="rejected", reason=why)
            record(rec)
            return
        answer, finish = teacher_answer(teacher, question)
        ok, why = acceptable_training_answer(answer, finish)
        rec["answer"] = answer
        if ok and args.verify:
            ok, verdict = review_answer(teacher, question, answer)
            rec["review"] = verdict
            why = "ok" if ok else "review"
        rec.update(status="accepted" if ok else "rejected", reason=why)
        record(rec)

    # 1. Questions from the syllabus
    topics = load_topics(args.topics)
    kept_topics = [t for t in topics if not held_out.reason(t)]
    if args.max_topics:
        kept_topics = kept_topics[: args.max_topics]
    print(f"Topics: {len(kept_topics)} of {len(topics)} (others overlap the evaluation set)")

    def ask_topic(topic: str) -> None:
        rng = random.Random(f"{args.seed}:{topic}")
        questions = topic_questions(teacher, topic, args.per_topic, rng)
        record({"key": f"topicq:{topic}", "status": "questions", "questions": questions})

    run_parallel("topic questions", [t for t in kept_topics if f"topicq:{t}" not in done], ask_topic)

    # 2. Questions from arXiv abstracts
    papers = fetch_arxiv(args.arxiv_per_category, args.seed) if args.arxiv_per_category > 0 else []
    if papers:
        print(f"arXiv: {len(papers)} papers")

    def ask_paper(paper: dict) -> None:
        q = arxiv_question(teacher, paper)
        record({"key": f"arxivq:{paper['id']}", "status": "questions", "questions": [q]})

    run_parallel("arXiv questions", [pp for pp in papers if f"arxivq:{pp['id']}" not in done], ask_paper)

    # 3. Answers (and reviews)
    jobs = []
    for topic in kept_topics:
        for qi, q in enumerate(done.get(f"topicq:{topic}", {}).get("questions", [])):
            jobs.append((f"topic:{topic}:{qi}", q, {"source": "topic", "topic": topic}))
    for paper in papers:
        qs = done.get(f"arxivq:{paper['id']}", {}).get("questions", [])
        if qs and qs[0]:
            jobs.append((f"arxiv:{paper['id']}", qs[0], {
                "source": "arxiv", "arxiv_id": paper["id"],
                "title": paper["title"], "category": paper["category"],
            }))
    todo = [j for j in jobs if j[0] not in done]
    print(f"Answers: {len(jobs)} questions, {len(todo)} still to answer")
    run_parallel("answers", todo, process)
    cache.close()

    write_splits(done, out, args.eval_fraction, args.seed, args.teacher_model)


def write_splits(done: dict[str, dict], out: Path, eval_fraction: float, seed: int,
                 teacher_model: str) -> None:
    accepted = [r for r in done.values() if r.get("status") == "accepted"]
    rejected = Counter(r["reason"].split(":")[0] for r in done.values() if r.get("status") == "rejected")

    # Deduplicate identical questions, keeping the first.
    seen, samples = set(), []
    for r in accepted:
        norm = " ".join(r["question"].lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        meta = dict(r["metadata"], teacher=teacher_model)
        samples.append(make_sample(r["question"], r["answer"], meta))

    random.Random(seed).shuffle(samples)
    n_eval = max(1, int(len(samples) * eval_fraction)) if samples else 0
    splits = {"validation": samples[:n_eval], "train": samples[n_eval:]}
    for name, data in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "teacher": teacher_model,
        "accepted": len(samples),
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "by_source": dict(Counter(s["metadata"]["source"] for s in samples)),
        "rejected": dict(rejected),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def push(args) -> None:
    from datasets import load_dataset

    files = {"train": str(Path(args.data) / "train.jsonl"),
             "validation": str(Path(args.data) / "validation.jsonl")}
    ds = load_dataset("json", data_files=files)
    ds.push_to_hub(args.repo, token=os.environ.get("HF_TOKEN"), private=args.private)
    print(f"Pushed to https://huggingface.co/datasets/{args.repo}")


def main(argv: list[str] | None = None) -> None:
    from .env import load_env, provider

    load_env()
    t_model, t_url, t_key_env = provider("TEACHER")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate and filter the training set")
    b.add_argument("--out", default="data/generated")
    b.add_argument("--topics", default="data/topics.txt")
    b.add_argument("--eval-questions", default="data/eval_questions.jsonl")
    b.add_argument("--per-topic", type=int, default=4)
    b.add_argument("--max-topics", type=int, default=0, help="use only the first N topics (smoke test)")
    b.add_argument("--arxiv-per-category", type=int, default=30)
    b.add_argument("--eval-fraction", type=float, default=0.1)
    b.add_argument("--teacher-model", default=t_model, required=t_model is None)
    b.add_argument("--teacher-base-url", default=t_url)
    b.add_argument("--teacher-key-env", default=t_key_env,
                   help="name of the environment variable that holds the API key")
    b.add_argument("--min-interval", type=float, default=0.0,
                   help="minimum seconds between teacher calls (free-tier rate limits)")
    b.add_argument("--workers", type=int, default=8, help="parallel teacher calls")
    b.add_argument("--thinking", default="minimal",
                   help="Gemini API thinking level for Gemma 4 (minimal or high)")
    b.add_argument("--verify", action="store_true",
                   help="ask the teacher to review every accepted answer for physics errors")
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=build)

    q = sub.add_parser("push", help="upload train/validation to the Hugging Face Hub")
    q.add_argument("--data", default="data/generated")
    q.add_argument("--repo", required=True)
    q.add_argument("--private", action="store_true")
    q.set_defaults(func=push)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
