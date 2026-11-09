# Test log

## 2026-11-08: v2 rewrite, checks without a GPU

**Environment:** Windows 11, CPU only, Python 3.13.0, torch 2.15.0+cpu, transformers
5.22.2, peft 0.21.2, gradio 6.31.0. The full Gemma 4 E2B weights (10.3 GB) were not
downloaded, so nothing here measures answer quality.

### Unit tests: 60 passed

`pytest -v`, raw output in [`results/2026-11-08-pytest.txt`](results/2026-11-08-pytest.txt).

| File | Tests | What they check |
|---|---|---|
| `test_formatting.py` | 5 | against the real Gemma 4 tokenizer: prompts start with `<bos><\|turn>system` and end with `<\|turn>model\n`; `<start_of_turn>` (used by v1) is split into several plain-text tokens while `<\|turn>` is one token; history renders in order; labels mask the whole prompt, train the answer and the `<turn\|>` marker, and the sequence has one `<bos>`; samples longer than the limit are dropped |
| `test_quality.py` | 16 | LaTeX/Unicode/Portuguese spellings of a formula normalize to the same string; key-fact matching; each format check (levels, template leaks, self-introduction, LaTeX delimiters, truncation); training-answer filter; the 20 evaluation questions are well formed |
| `test_sims.py` | 29 | oscillator: the closed-form solution satisfies x'' + γx' + ω²x = 0 and the initial conditions in all three regimes, regime labels, FFT peak within 0.06 Hz of √(ω²−γ²/4)/2π, energy conserved without damping (0.1%); wave packet: σₓ = σ (1%), σₓσₖ = 0.5 (3%), peak at k₀, centre at k₀t (±0.02) and width σ√(1+(t/2σ²)²) (1%) for three (σ, t); Kepler: equation solved to 1e-10 at e = 0.9, r = a(1−e²)/(1+e cos θ), equal areas per time step (2%), v_peri/v_aph = (1+e)/(1−e), Earth: 1 year and ~30 km/s; simulation choice and parameter extraction; each render returns a figure and numbers in under 1 s |
| `test_data.py` | 10 | held-out filter blocks evaluation topics and keeps others; 70-100% of the syllabus topics survive it; question cleanup; train/validation split, deduplication and rejection stats; judge parsing; sign test; summary and comparison table; .env loading and Gemini/Gemma provider defaults; removal of reasoning blocks from teacher output |

### Inference path on a tiny random Gemma 4

A Gemma 4 with the real architecture and tokenizer but 26M random parameters
(4 text layers, KV sharing on the last two), with a random LoRA adapter on the language
model's attention and MLP layers, loaded through `resonance.model.Tutor` on CPU:

| Check | Result |
|---|---|
| Model class loaded by `AutoModelForImageTextToText` and `AutoModelForCausalLM` for the real E2B config | `Gemma4ForConditionalGeneration` for both |
| Stop token | `<turn\|>` (id 106) |
| Adapter on vs off (`disable_adapter`), same prompt, greedy | different outputs |
| Greedy decoding twice | identical outputs |
| `truncated` flag when the 12-token limit is hit | `True` |
| Streaming with a two-turn history | 9 incremental chunks |

### v1 adapter on the Hub, inspected

From the safetensors header of `vtaua/resonance-gemma4-physics` (May 2026): 786 LoRA
tensors, 490 on the language model, 224 on the vision tower and 72 on the audio tower.
Against the transformers 5.22.2 module tree, 40 of its 393 target modules do not
exist (`k_proj`/`v_proj` of the 20 KV-shared layers).

### Demo in the browser

`python app.py` with the tiny model: the page loads; a question streams an answer into
the chat; at the end the app selects "Órbita kepleriana" for
"Calcule a órbita de um planeta com excentricidade 0,7…" and draws it with e = 0.7
taken from the question. Plots rendered separately and checked by eye: wave-packet
spectrum peaks at k₀ = 5 with σₓσₖ = 0.500; oscillator FFT peaks at the theoretical
1.00 Hz.

### Live simulation controls

Median time to recompute one simulation (figure + numbers), 5 runs on this CPU:
oscillator 145 ms, wave packet 113 ms, orbit 128 ms.

In the browser (`python app.py --dry-run`): the orbit question selects "Órbita kepleriana",
shows only its three sliders and fills e = 0.7 from the question; the results panel reads
perihelion 0.3 AU, aphelion 1.7 AU, period 1 year, 70.9 / 12.5 km/s. Moving `a` to 5 AU
redraws the plot and updates the panel (period 11.2 years = 5^1.5) without a new answer
from the model.

### Training set

`python -m resonance.data build --out data/generated --per-topic 4 --arxiv-per-category 30 --verify --workers 12`,
teacher `gemma-4-31b-it` through the Gemini API (free tier, thinking level minimal).

Latency measured before the run: with default thinking one answer took 105 s and
returned a `<thought>` block; `reasoning_effort` was rejected (HTTP 400); with
`thinking_level: minimal` a one-sentence answer took 57 s; 8 parallel calls took 13-49 s
each with no errors for `gemma-4-31b-it`, and 1-3 s with 3 of 8 failing (HTTP 500) for
`gemma-4-26b-a4b-it`.

| | count |
|---|---|
| Topics used (6 of 105 overlap the evaluation set) | 99 |
| arXiv papers with a question | 139 |
| Questions answered | 485 |
| Accepted | **330** (297 train, 33 validation; 243 topic, 87 arXiv) |
| Rejected: hit the 3,072-token limit | 75 |
| Rejected: question touches an evaluation topic | 40 |
| Rejected: teacher's review found a physics error | 33 |
| Rejected: provider content filter | 3 |
| Rejected: other format checks | 4 |

Throughput about 8 answers (plus reviews) per minute with 12 workers; the free tier
returned frequent HTTP 500/503 errors, all recovered by retries.

Checked afterwards with the Gemma 4 tokenizer: rendered samples have a median of 780
tokens and a maximum of 1,076 (none above the 2,048 training limit); no training or
validation question matches an evaluation exclusion term; every answer passes the
format checks. `cache.jsonl` (every attempt, with the review verdicts) is kept out of
the repository.

### Training run v2 (Colab, Tesla T4)

`notebooks/resonance_pipeline.ipynb`, Unsloth 2026.11.3, transformers 5.17.0, torch
2.14.1+cu130. The T4 has no bfloat16; Unsloth ran Gemma 4 in float32 ("float16 won't
work"). Raw log: [`results/v2/training_log.json`](../results/v2/training_log.json).

| | value |
|---|---|
| Samples | 297 train, 33 validation (none dropped for length) |
| Answer tokens trained on | 183,778 |
| Steps | 76 (38 per epoch, 4 warm-up), effective batch 8, lr 2e-4 cosine |
| Trainable parameters | 24,158,208 of 5,128,455,712 (0.47%) |
| Training loss | 1.379 at the first log, 0.623 at the last, 0.739 mean |
| Validation loss | 0.675 after epoch 1, **0.665** after epoch 2 (kept) |
| Runtime | 742 s |

### Evaluation v2: base vs fine-tuned

20 held-out questions, greedy decoding, 1,024 new tokens, base model loaded in 4-bit with
the adapter off and on (about 1 h on the T4, judge calls included). Raw answers and
grades: [`results/v2/`](../results/v2/).

The judge returned an empty text on all 40 answers in the Colab run: with Gemma 4's
default thinking, the 200-token budget was spent on reasoning. Fixed (`thinking_level:
minimal`, 600 tokens, a parser that accepts `**NOTA:** 4`) and the saved answers were
graded again locally with `python -m resonance.evaluate --rejudge results/v2`; all 40
grades parsed.

| Metric | base | tuned |
|---|---|---|
| Answers passing every format check | 90% | 100% |
| Finished before the token limit | 90% | 100% |
| Both levels / no leaks / no self-intro / closed LaTeX | 100% | 100% |
| Key-fact recall | 80% | 65% |
| Judge grade (1-5, gemma-4-31b-it with the reference answer) | **3.30** | **2.30** |
| Mean length (words) | 443 | 349 |

| Paired, per question | tuned better | base better | ties | sign test p |
|---|---|---|---|---|
| Key-fact recall | 1 | 4 | 15 | 0.375 |
| Format checks passed | 2 | 0 | 18 | 0.5 |
| Judge grade | 4 | 13 | 3 | **0.049** |

**Result: the fine-tuned adapter answers in the right format every time but is less
correct than the base model.** The four key-fact losses are real errors, read by hand:
Snell's law derived without the refractive indices, a Newtonian "derivation" of the
gravitational redshift, orbital speed instead of Kepler's third law, wrong double-slit
geometry. The tuned model also produced a Wien temperature off by six orders of
magnitude. It improved on the pendulum, Faraday's law, the oscillator and the cyclotron
radius. The base model itself averages 3.3 of 5, so neither is reliable.

Not published (v3 below was published instead).

### Training run v3: conservative

Same data and Colab session; learning rate 5e-5 (4× lower) and 1 epoch, to move the
model less from what it already knows. 38 steps, 362 s; training loss 1.045 mean,
validation loss **0.884** (v2 reached 0.665). Raw log:
[`results/v3/training_log.json`](../results/v3/training_log.json).

Evaluated only the fine-tuned model (greedy decoding is deterministic, so the base
answers from v2 were reused), then graded base and v3 locally with the fixed judge
(`--rejudge results/v3`); all 40 grades parsed. The base model's mean grade came out at
3.30 again.

| Metric | base | v2 (lr 2e-4, 2 epochs) | v3 (lr 5e-5, 1 epoch) |
|---|---|---|---|
| Answers passing every format check | 90% | 100% | 100% |
| Finished before the token limit | 90% | 100% | 100% |
| Key-fact recall | 80% | 65% | 85% |
| Judge grade (1-5) | 3.30 | 2.30 | **3.20** |
| Mean length (words) | 443 | 349 | 347 |

| v3 vs base, per question | tuned better | base better | ties | sign test p |
|---|---|---|---|---|
| Key-fact recall | 3 | 2 | 15 | 1.0 |
| Format checks passed | 2 | 0 | 18 | 0.5 |
| Judge grade | 6 | 6 | 8 | 1.0 |

**Result: v3 keeps the base model's physics (no measurable difference in the judge's
grades) while always answering in the two-level format, finishing, and using about 22%
fewer words.** It does not make the model more correct.

## 2026-11-09: demo on a Hugging Face Space

- Creating a Gradio Space through the API with a free account was refused (HTTP 402: "hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription"). Creating it on the website worked, with ZeroGPU; `scripts/deploy_space.py` then uploaded the files.
- First start failed with `RuntimeError: No CUDA GPUs are available`: PEFT loads adapter weights onto "cuda" by default, and a ZeroGPU Space has no GPU outside `@spaces.GPU`. Fixed by loading the adapter on the CPU (`torch_device="cpu"`) before moving the model.
- Second start: running on `zero-a10g` about 1.5 min after upload. A question about the simple pendulum streamed a two-level answer with rendered LaTeX; the header read "Gemma 4 E2B + Resonance LoRA · ZeroGPU".
- Locally, before deploying: the generation path ran with and without a stand-in `spaces` module (the decorated generator was called once and streamed 1,025 chunks on the tiny random Gemma 4); 60 tests passed.
