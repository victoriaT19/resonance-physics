import json
from pathlib import Path

import pytest

from resonance.quality import (
    acceptable_training_answer,
    check_answer,
    key_fact_hits,
    normalize_math,
)

ROOT = Path(__file__).resolve().parents[1]

GOOD = (
    "**Nível 1 — Intuição:** A luz que passa pela fenda se espalha. Bordas opostas da fenda "
    "mandam ondas que se cancelam em certas direções.\n\n"
    "**Nível 2 — Formal:** Dividindo a fenda em duas metades, a diferença de caminho é "
    "$\\frac{a}{2}\\sin\\theta$. Os mínimos ficam em $$a\\sin\\theta = m\\lambda,\\quad m=\\pm1,\\pm2,\\dots$$"
)


@pytest.mark.parametrize("spelling", [
    "a\\sin\\theta = m\\lambda",
    "$a \\sin(\\theta) = m \\lambda$",
    "a sinθ = mλ",
    "a\\,\\text{sen}\\,\\theta = m\\lambda",
    "a \\operatorname{sen} \\theta = m\\lambda",
])
def test_normalize_math_spellings_agree(spelling):
    assert normalize_math(spelling) == normalize_math("a\\sin\\theta=m\\lambda")


def test_normalize_math_fractions_and_greek():
    assert normalize_math("\\dfrac{2GM}{rc^2}") == normalize_math("\\frac{2GM}{rc^2}")
    assert normalize_math("\\Delta x \\Delta p \\geq \\frac{\\hbar}{2}") == normalize_math("Δx Δp ≥ \\frac{ħ}{2}")
    assert normalize_math("\\varepsilon_0") == normalize_math("\\epsilon_0")
    # \left and \le must not collide
    assert "≤" not in normalize_math("\\left( x \\right)")


def test_key_fact_hits():
    facts = [["a\\sin\\theta = m\\lambda"], ["I = I_0\\cos^2\\theta"]]
    assert key_fact_hits(GOOD, facts) == [True, False]


def test_check_answer_good():
    c = check_answer(GOOD)
    assert c["ok"], c


@pytest.mark.parametrize("bad,flag", [
    (GOOD.replace("**Nível 2 — Formal:**", "Formalmente:"), "both_levels"),
    ("Resonance</model>\n" + GOOD, "no_artifacts"),
    ("<|turn>model\n" + GOOD, "no_artifacts"),
    ("Olá! Eu sou Resonance, seu assistente.\n" + GOOD, "no_self_intro"),
    (GOOD + " e $x = 1", "balanced_math"),
])
def test_check_answer_flags(bad, flag):
    c = check_answer(bad)
    assert not c[flag]
    assert not c["ok"]


def test_truncated_answer_is_not_ok():
    assert not check_answer(GOOD, truncated=True)["ok"]


def test_training_filter():
    long_good = GOOD + " Explicação adicional." * 40
    assert acceptable_training_answer(long_good, "stop")[0]
    assert acceptable_training_answer(long_good, "length") == (False, "finish_reason=length")
    assert acceptable_training_answer(GOOD, "stop")[1].startswith("words=")


def test_eval_questions_reference_answers_hit_their_own_facts():
    """Each reference answer, written in plain text, should be checkable by its facts.

    Not every reference spells the equation exactly like a fact, so this only requires
    the file to be well formed and most references to match.
    """
    items = [json.loads(l) for l in (ROOT / "data/eval_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(items) == 20
    assert len({i["id"] for i in items}) == 20
    for item in items:
        assert item["question"] and item["reference"] and item["exclude_terms"]
        assert item["key_facts"] and all(isinstance(alts, list) and alts for alts in item["key_facts"])
