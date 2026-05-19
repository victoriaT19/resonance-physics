"""Chat formatting against the real Gemma 4 tokenizer (downloads ~30 MB on first run)."""

import pytest

from resonance.formatting import IGNORE_INDEX, render_example, tokenize_example
from resonance.prompts import SYSTEM_PROMPT, build_messages, render_prompt

MODEL = "unsloth/gemma-4-E2B-it"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:  # offline
        pytest.skip(f"tokenizer not available: {exc}")


ANSWER = "**Nível 1 — Intuição:** a luz se espalha.\n\n**Nível 2 — Formal:** $a\\sin\\theta = m\\lambda$."


def test_prompt_uses_gemma4_turn_markers(tok):
    prompt = render_prompt(tok, build_messages("O que é difração?"))
    assert prompt.startswith("<bos><|turn>system\n")
    assert SYSTEM_PROMPT.strip() in prompt
    assert prompt.endswith("<|turn>model\n")
    assert "<start_of_turn>" not in prompt


def test_old_markers_are_not_special_tokens(tok):
    """The first version wrote <start_of_turn> by hand; Gemma 4 splits it into plain text."""
    ids = tok("<start_of_turn>", add_special_tokens=False)["input_ids"]
    assert len(ids) > 1
    assert len(tok("<|turn>", add_special_tokens=False)["input_ids"]) == 1


def test_history_is_rendered_in_order(tok):
    msgs = build_messages("E no vácuo?", history=[
        {"role": "user", "content": "O que é a velocidade da luz?"},
        {"role": "assistant", "content": "Cerca de 3e8 m/s."},
    ])
    prompt = render_prompt(tok, msgs)
    assert prompt.index("velocidade da luz") < prompt.index("<|turn>model\nCerca") < prompt.index("E no vácuo?")


def test_labels_cover_only_the_answer(tok):
    msgs = build_messages("O que é difração?")
    prompt_text, full_text = render_example(tok, msgs, ANSWER)
    assert full_text.startswith(prompt_text)
    row = tokenize_example(tok, msgs, ANSWER, max_length=4096)
    labels, ids = row["labels"], row["input_ids"]
    assert len(labels) == len(ids) == len(row["attention_mask"])
    n_prompt = len(tok(prompt_text, add_special_tokens=False)["input_ids"])
    assert all(t == IGNORE_INDEX for t in labels[:n_prompt])
    trained = [t for t in labels if t != IGNORE_INDEX]
    decoded = tok.decode(trained)
    assert decoded.startswith("**Nível 1")
    # the end-of-turn marker is trained, so the model learns to stop
    assert tok.convert_tokens_to_ids("<turn|>") in trained
    # exactly one <bos>
    assert ids.count(tok.bos_token_id) == 1


def test_too_long_samples_are_dropped(tok):
    assert tokenize_example(tok, build_messages("x"), ANSWER, max_length=10) is None
