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

### Not done

- Dataset generation (needs a teacher API key), training and evaluation (need a GPU).
  The README's Results section stays empty until they run.
