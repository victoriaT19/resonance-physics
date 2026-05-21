import json
from pathlib import Path

from resonance.data import HeldOutFilter, _clean_question, load_jsonl, load_topics, write_splits
from resonance.evaluate import markdown_table, paired, parse_judge, sign_test, summarize

ROOT = Path(__file__).resolve().parents[1]


def eval_items():
    return load_jsonl(ROOT / "data/eval_questions.jsonl")


def test_held_out_filter_blocks_eval_topics():
    f = HeldOutFilter(eval_items())
    assert f.reason("Qual é o período de um pêndulo físico?").startswith("eval_term")
    assert f.reason("Derive a lei de Snell usando o princípio de Fermat.") is not None
    assert f.reason("Explique o efeito Compton e derive a variação do comprimento de onda.") is None


def test_some_topics_survive_and_some_are_filtered():
    f = HeldOutFilter(eval_items())
    topics = load_topics(ROOT / "data/topics.txt")
    kept = [t for t in topics if not f.reason(t)]
    assert len(topics) > 100
    assert 0.7 * len(topics) < len(kept) < len(topics)


def test_clean_question():
    assert _clean_question("1. O que é entropia?") == "O que é entropia?"
    assert _clean_question("- \"Por que o céu é azul?\"") == "Por que o céu é azul?"


def test_write_splits(tmp_path):
    done = {}
    for i in range(20):
        done[f"k{i}"] = {"key": f"k{i}", "status": "accepted", "reason": "ok",
                         "question": f"Pergunta {i}?", "answer": "resposta",
                         "metadata": {"source": "topic"}}
    done["dup"] = dict(done["k0"], key="dup")
    done["bad"] = {"key": "bad", "status": "rejected", "reason": "eval_term:snell"}
    write_splits(done, tmp_path, 0.1, 0, "teacher-x")
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["accepted"] == 20  # the duplicate question is dropped
    assert stats["train"] == 18 and stats["validation"] == 2
    assert stats["rejected"] == {"eval_term": 1}
    row = json.loads((tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["metadata"]["teacher"] == "teacher-x"


def test_parse_judge():
    assert parse_judge("NOTA: 4\nMOTIVO: faltou o fator 2.") == (4, "faltou o fator 2.")
    assert parse_judge("não sei")[0] is None
    assert parse_judge("**NOTA:** 3\n**MOTIVO:** faltou n")[0] == 3
    assert parse_judge("**NOTA**: 5")[0] == 5


def test_sign_test():
    assert sign_test(0, 0) == 1.0
    assert sign_test(10, 0) < 0.01
    assert sign_test(5, 5) == 1.0


def _row(ok, facts, grade=None):
    keys = ["ok", "both_levels", "no_artifacts", "no_self_intro", "balanced_math", "complete"]
    r = {"scores": {**{k: ok for k in keys}, "key_fact_recall": facts, "words": 100}}
    if grade:
        r["judge"] = {"grade": grade}
    return r


def test_summary_and_paired_table():
    base = [_row(False, 0.0, 2), _row(True, 1.0, 4)]
    tuned = [_row(True, 1.0, 4), _row(True, 1.0, 4)]
    s = {"base": summarize(base), "tuned": summarize(tuned)}
    assert s["tuned"]["ok"] == 1.0 and s["base"]["ok"] == 0.5
    c = {"Key-fact recall": paired(base, tuned, lambda r: r["scores"]["key_fact_recall"])}
    assert c["Key-fact recall"]["tuned_better"] == 1 and c["Key-fact recall"]["ties"] == 1
    table = markdown_table(s, c)
    assert "| Key-fact recall | 50% | 100% |" in table
    assert "Judge grade" in table


def test_strip_thinking():
    from resonance.data import strip_thinking

    assert strip_thinking("<thought>hmm</thought>\n**Nível 1**") == "**Nível 1**"
    assert strip_thinking("<|channel>thought\nx<channel|>Resposta") == "Resposta"


def test_provider_defaults_to_gemma_on_gemini(monkeypatch):
    from resonance.env import GEMINI_GEMMA_MODEL, GEMINI_OPENAI_URL, provider

    for k in ["TEACHER_MODEL", "TEACHER_BASE_URL", "TEACHER_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert provider("TEACHER") == (GEMINI_GEMMA_MODEL, GEMINI_OPENAI_URL, "GEMINI_API_KEY")
    monkeypatch.setenv("TEACHER_API_KEY", "y")
    monkeypatch.setenv("TEACHER_MODEL", "other")
    assert provider("TEACHER") == ("other", None, "TEACHER_API_KEY")


def test_load_env_does_not_override(tmp_path, monkeypatch):
    from resonance.env import load_env

    f = tmp_path / ".env"
    f.write_text("# comment\nA_KEY=from_file\nB_KEY='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("A_KEY", "already")
    monkeypatch.delenv("B_KEY", raising=False)
    load_env(f)
    import os
    assert os.environ["A_KEY"] == "already" and os.environ["B_KEY"] == "quoted"
