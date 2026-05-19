"""Turn chat samples into token ids and labels for supervised fine-tuning.

The loss covers only the assistant's answer and its end-of-turn marker. The prompt part
(system and user turns) is masked with -100 so the model is not trained to reproduce it.
"""

from __future__ import annotations

IGNORE_INDEX = -100


class TemplateMismatch(ValueError):
    """The rendered prompt is not a prefix of the rendered conversation."""


def render_example(tokenizer, prompt_messages: list[dict], answer: str) -> tuple[str, str]:
    """Return (prompt_text, full_text) rendered with the tokenizer's chat template."""
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": answer}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise TemplateMismatch(
            "The chat template renders the prompt differently with and without the answer; "
            "the prompt/answer boundary cannot be found."
        )
    return prompt_text, full_text


def tokenize_example(
    tokenizer, prompt_messages: list[dict], answer: str, max_length: int
) -> dict | None:
    """Tokenize one sample. Returns None when it does not fit in `max_length` tokens.

    Samples are dropped rather than truncated: a cut answer would teach the model to stop
    mid-derivation, or never to emit its end-of-turn marker.
    """
    prompt_text, full_text = render_example(tokenizer, prompt_messages, answer)
    # The prompt and the answer are tokenized separately, as at inference time, where the
    # prompt is tokenized alone and the model continues from its last token. The template
    # already writes <bos>, so special tokens are not added again.
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + answer_ids
    if len(full_ids) > max_length:
        return None
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
