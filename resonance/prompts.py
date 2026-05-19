"""Prompt and chat formatting shared by data generation, training, evaluation and the app.

Every stage renders conversations with the tokenizer's own chat template. Gemma 4 marks
turns with `<|turn>` and `<turn|>`; hand-written Gemma 2/3 markers such as
`<start_of_turn>` are plain text to its tokenizer, so they must never be hard-coded.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Você é Resonance, um tutor de física. Responda em português, sempre em dois níveis:\n"
    "**Nível 1 — Intuição:** explicação acessível, com uma analogia ou imagem física e sem "
    "formalismo pesado.\n"
    "**Nível 2 — Formal:** a derivação ou o tratamento matemático, passo a passo, com as "
    "equações em LaTeX entre $...$ ou $$...$$, terminando no resultado principal.\n"
    "Não se apresente e não repita a pergunta. Se a pergunta tiver uma premissa errada, "
    "corrija-a antes de responder."
)

LEVEL1_HEADER = "Nível 1"
LEVEL2_HEADER = "Nível 2"


def build_messages(
    question: str,
    history: list[dict] | None = None,
    system: str = SYSTEM_PROMPT,
) -> list[dict]:
    """Return the chat messages for one question, after any earlier turns.

    `history` uses the OpenAI/Gradio "messages" shape: [{"role": ..., "content": ...}].
    """
    messages = [{"role": "system", "content": system}]
    for turn in history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """Render messages up to the start of the model's turn."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def text_tokenizer(processor_or_tokenizer):
    """Gemma 4 loads as a multimodal processor; the text tokenizer lives inside it."""
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
