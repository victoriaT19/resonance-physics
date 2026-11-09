# Resonance

**A physics tutor fine-tuned from Gemma 4 E2B that answers in two levels: intuition first, then the derivation.** Built for the Gemma 4 Good Hackathon (Kaggle, 2026) and rebuilt afterwards, once its first version turned out to be broken in ways the original benchmark did not show.

- **Live demo** (Hugging Face Space, ZeroGPU): [huggingface.co/spaces/vtaua/resonance-physics-demo](https://huggingface.co/spaces/vtaua/resonance-physics-demo)
- Model (LoRA adapter): [huggingface.co/vtaua/resonance-gemma4-physics](https://huggingface.co/vtaua/resonance-gemma4-physics)
- Pipeline notebook (Colab or Kaggle GPU): [`notebooks/resonance_pipeline.ipynb`](notebooks/resonance_pipeline.ipynb)

```text
Pergunta: Derive o período de um pêndulo simples para pequenas oscilações.

**Nível 1 — Intuição:** ...uma explicação com uma imagem física...
**Nível 2 — Formal:** ...a derivação, com as equações em LaTeX, até T = 2π√(L/g)
```

## What was wrong with the first version

The first version (May 2026) trained, published an adapter and served a demo. Looking at it again, almost nothing worked as described:

| Problem | Effect | Fix |
|---|---|---|
| Prompts were written by hand with Gemma 2/3 markers (`<start_of_turn>`). Gemma 4 uses `<\|turn>`/`<turn\|>`, so its tokenizer read those markers as ordinary text. | Training, data generation and the demo all used a chat format the model was never trained on. Answers leaked `Resonance</model>`, introduced themselves and did not stop. | Every stage renders with `tokenizer.apply_chat_template` (`resonance/prompts.py`); a test checks the markers against the real tokenizer. |
| The demo loaded the base model and never the adapter. | The demo showed the base model. | `resonance/model.py` loads the base model plus the adapter, with a `--base-only` flag for comparison. |
| The "benchmark" ran only the base model, on 5 questions, cut at 350 tokens. | There was no base vs. fine-tuned comparison. | `resonance/evaluate.py`: 20 held-out questions, both models, automatic checks and an optional LLM judge. |
| The training answers were written by the base model itself, about 85 of them. | Fine-tuning could only copy the base model's format and mistakes. | A stronger teacher model writes the answers, and automatic checks filter them (`resonance/data.py`). |
| With ~32 optimizer steps, warm-up was 50 steps and evaluation/checkpoints every 50. | The learning rate never reached its peak; validation and "best checkpoint" never ran. | Warm-up and evaluation are sized from the dataset (`resonance/train.py`). |
| LoRA targets matched every `q_proj`… in the model. | 296 of the adapter's 786 tensors went to the vision and audio towers, which text training never uses. | LoRA on the language model only. |
| The loss covered the system prompt and the question. | Part of the training signal went to copying the prompt. | Prompt tokens are masked; only the answer and its end-of-turn marker are trained. |
| `repetition_penalty=1.3`. | It penalizes the symbols LaTeX must repeat (`$`, `\frac`, subscripts) and garbles equations. | 1.0 for evaluation, 1.0 and temperature 0.3 in the demo. |
| "SymPy verification" ran `sympify` on the model's output. | It verified nothing (it only simplified), failed on real LaTeX, and `sympify` evaluates its input with `eval`: unsafe on a public demo. | Removed. |
| The plot was picked by keyword counts over question and answer; "onda" matched almost every answer. The wave-packet spectrum had the wrong k axis. Parameters could not be changed. | Wrong plots, a spectrum that did not peak at k₀, and a new plot only by asking a new question. | Word-boundary keywords weighted toward the question, a dropdown to switch, live sliders per simulation with derived numbers, and simulations checked against theory in tests. |

## How it works now

```text
data/topics.txt ─┐                        ┌─► data/generated/{train,validation}.jsonl
arXiv abstracts ─┴─► teacher model ─► checks + held-out filter
                                                        │
                       resonance/train.py (QLoRA, Unsloth, T4) ◄┘
                                │
                         outputs/adapter ─► resonance/evaluate.py (base vs tuned, 20 questions)
                                │                        │
                                └──► resonance/card.py ◄─┘ ─► Hugging Face Hub
```

1. **Data** (`resonance/data.py`). Questions come from a syllabus of about 120 topics and from the core concept of recent arXiv abstracts in five categories. A teacher model, behind any OpenAI-compatible API, answers each one with the same system prompt the tutor is trained on. An answer is kept only if it has both levels, closed LaTeX, no template leaks or self-introduction, a sensible length, and was not cut off; `--verify` adds a second pass where the teacher reviews it for physics errors. Questions that touch an evaluation topic are dropped. Details: [`data/README.md`](data/README.md).
2. **Training** (`resonance/train.py`). QLoRA (rank 16) with Unsloth on the language model's attention and MLP layers, loss on the answers only, learning rate 5e-5 for one epoch with warm-up and cosine decay (a stronger setting made the physics worse; see Results), validation at the end of each epoch.
3. **Evaluation** (`resonance/evaluate.py`). The base model and the fine-tuned model answer 20 hand-written questions with greedy decoding, from one load (the adapter switched off and on). Each answer is checked for format, for the central equation of the question ("key-fact recall", matched across LaTeX and Unicode spellings) and, optionally, graded 1-5 by an LLM judge that sees a reference answer. Results are compared question by question with a sign test.
4. **Model card** (`resonance/card.py`). Written from the measured numbers and uploaded with the adapter.
5. **Demo** (`app.py`). Gradio chat with streaming, LaTeX rendering and conversation history, next to three simulations that the app picks from the question (and fills with any numbers in it, such as "excentricidade 0,7"). Each has its own sliders; moving one recomputes the plot and a panel of derived quantities in about 0.1 s, without calling the model again:

   | Simulation | Parameters | Shows |
   |---|---|---|
   | Damped harmonic oscillator (exact solution) | ω, γ, x₀, v₀ | x(t), envelope, energy, FFT; regime, damped frequency (theory vs. FFT peak), period, Q factor, decay time |
   | Free Gaussian wave packet (FFT propagation, ħ = m = 1) | k₀, σ, t | ψ(x, 0) and ψ(x, t), momentum spectrum; σₓσₖ = 1/2 at t = 0, width vs. σ√(1 + (t/2σ²)²), centre vs. group velocity × t |
   | Kepler orbit (Kepler's equation solved by Newton's method) | e, a, star mass | orbit with positions at equal time steps, r(t) and v(t) in km/s; perihelion, aphelion, period from the third law, speeds |

   The simulations are tested against their closed forms (ODE residual, energy conservation, spreading law, equal areas, Earth's 1-year period and ~30 km/s).

## Results

Two runs on the same 330 teacher-written samples (Colab T4), on the 20 held-out questions:

| Metric | base | v2: lr 2e-4, 2 epochs | v3: lr 5e-5, 1 epoch |
|---|---|---|---|
| Answers passing every format check | 90% | 100% | 100% |
| Finished before the token limit | 90% | 100% | 100% |
| Key-fact recall | 80% | 65% | 85% |
| Judge grade (1-5, Gemma 4 31B with a reference answer) | 3.30 | **2.30** | **3.20** |
| Mean length (words) | 443 | 349 | 347 |

- **v2 made the physics worse.** Per question, the judge preferred the base model 13 times and v2 4 times (3 ties; sign test p = 0.049). v2 always answered in two levels, but wrote confident derivations that went wrong: Snell's law without refractive indices, a Newtonian gravitational redshift. Training a 2B model hard on a 31B model's derivations taught it the style of a derivation more than its content. v2 also had the *lower* validation loss (0.665 against 0.884): the validation set is teacher text too, so a lower loss means closer imitation of the teacher, not more correct physics. Only the held-out evaluation with a judge exposed the problem.
- **v3, trained gently, kept the base model's physics** (6 wins each, 8 ties; p = 1.0) while always using the two-level format, always finishing and writing about 22% fewer words. It does not make the model more correct: with a judge grade near 3 of 5, both still make mistakes. **v3 is the adapter published on the Hub**, and the training defaults in `resonance/train.py` reproduce it.

This matches what Gudibande et al. found for chat models: imitating a stronger model's outputs transfers its style much more than its factual accuracy ([The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717), 2023).

Full numbers, raw answers and grades: [`docs/test-log.md`](docs/test-log.md), [`results/v2/`](results/v2/), [`results/v3/`](results/v3/).

## Run

```bash
pip install -r requirements.txt

python app.py                          # Gemma 4 E2B + adapter from the Hub (GPU, or slow on CPU)
python app.py --base-only              # the base model
python app.py --adapter outputs/adapter --load-in-4bit --share
python app.py --dry-run                # no model, to try the interface
python scripts/deploy_space.py         # publish to a Hugging Face Space (create it on the website first)

pytest                                 # 60 tests; the formatting tests download the Gemma 4 tokenizer
```

Full pipeline. The dataset needs no GPU; put `GEMINI_API_KEY=...` in a `.env` file at the repository root (ignored by git) and the teacher defaults to Gemma 4 31B (`gemma-4-31b-it`) through the Gemini API, so the training answers come from a larger Gemma under the Gemma terms. Any other OpenAI-compatible model works through `TEACHER_MODEL`, `TEACHER_BASE_URL` and `TEACHER_API_KEY`.

```bash
pip install -r requirements-train.txt
python -m resonance.data build --out data/generated --verify --workers 12
```

Training and evaluation on a GPU ([`notebooks/resonance_pipeline.ipynb`](notebooks/resonance_pipeline.ipynb) runs them on a free Colab T4):

```bash
python -m resonance.train --data data/generated --out outputs/adapter
python -m resonance.evaluate --adapter outputs/adapter --out results/v2 --load-in-4bit
python -m resonance.card --adapter outputs/adapter --data data/generated --eval results/v2 --push
```

## Layout

```text
app.py                     Gradio demo
resonance/
  prompts.py               system prompt, chat rendering
  formatting.py            token ids and answer-only labels for training
  quality.py               format checks, LaTeX normalization, key-fact matching
  data.py                  teacher-written training set
  train.py                 QLoRA fine-tuning (Unsloth)
  model.py                 inference: base + adapter, streaming
  evaluate.py              base vs fine-tuned evaluation
  card.py                  model card from measured results
  sims.py                  simulations shown next to the chat
data/                      evaluation questions, topic list, generated data
notebooks/                 pipeline notebook for Colab/Kaggle
tests/                     pytest suite
docs/test-log.md           what was tested, when, and the results
```

## Limits

- A model with 2B effective parameters makes physics mistakes even after fine-tuning; the demo says so.
- Training answers are written and filtered by machines, not reviewed by a person.
- 20 evaluation questions detect only large differences; the sign test makes that explicit.
- Portuguese only.
