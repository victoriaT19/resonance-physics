# Data

| File | What |
|---|---|
| `eval_questions.jsonl` | 20 held-out evaluation questions, written by hand |
| `topics.txt` | syllabus of physics topics the teacher writes training questions about |
| `generated/` | output of `python -m resonance.data build` (not committed except `stats.json`) |

## Evaluation questions

Classic questions from optics, mechanics, electromagnetism, modern physics and
astrophysics. Each record has:

- `question`: the prompt, in Portuguese;
- `reference`: a short correct answer, shown to the LLM judge;
- `key_facts`: the central equation, as a list of acceptable spellings (LaTeX or plain);
  an answer "recalls" the fact if any spelling appears after normalization
  (`resonance/quality.py:normalize_math`);
- `exclude_terms`: words that, if present in a training question, remove it from the
  training set, so these topics stay unseen.

## Training set

Built by `resonance/data.py`:

1. **Questions.** For each topic in `topics.txt`, the teacher writes four questions of
   different kinds (conceptual, derivation, numerical, comparison). For recent arXiv
   abstracts in five categories, the teacher writes one question about the
   undergraduate-level concept the paper relies on.
2. **Held-out filter.** Questions that contain an evaluation `exclude_term`, or share
   more than 40% of their words with an evaluation question, are dropped.
3. **Answers.** The teacher (by default Gemma 4 31B, `gemma-4-31b-it`, through the Gemini
   API with its thinking level set to minimal; 12 calls in parallel, with retries on the
   free tier's frequent 500/503 errors) answers with the same system prompt the model is trained
   with, plus rules on correctness and length.
4. **Checks.** An answer is kept only if it has both level headers, closed LaTeX, no
   template leaks or self-introduction, 120 to 900 words, and was not cut off by the
   token limit. With `--verify`, the teacher also reviews each answer for physics errors.
5. **Split.** Duplicates removed, 90% train and 10% validation.

`generated/stats.json` records how many answers were kept and why the others were
dropped. Each sample:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "**Nível 1 — Intuição:** ... **Nível 2 — Formal:** ..."}
  ],
  "metadata": {"source": "topic", "topic": "efeito Compton", "teacher": "..."}
}
```

## What the first version did

The dataset of the first version (May 2026) had about 85 training samples, and their
answers were written by the base model itself, with a prompt in the wrong chat format.
Fine-tuning on it could only reinforce the base model's style and its mistakes. It is
not used anymore.
