"""Resonance: Gradio demo.

Runs Gemma 4 E2B with the Resonance LoRA adapter. Works on a GPU or, slowly, on a CPU.

  python app.py                     # adapter from the Hub
  python app.py --adapter outputs/adapter
  python app.py --base-only         # the base model, for comparison
  python app.py --share             # public gradio.live link
  python app.py --dry-run           # no model: echoes the question (to try the interface)
"""

from __future__ import annotations

import argparse
import os

import gradio as gr

from resonance import sims
from resonance.model import ADAPTER, BASE_MODEL, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--base-model", default=os.environ.get("RESONANCE_BASE", BASE_MODEL))
parser.add_argument("--adapter", default=os.environ.get("RESONANCE_ADAPTER", ADAPTER))
parser.add_argument("--base-only", action="store_true")
parser.add_argument("--load-in-4bit", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--share", action="store_true")
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
args, _ = parser.parse_known_args()

MAX_HISTORY_TURNS = 4

if args.dry_run:
    tutor = None
    MODEL_LABEL = "modo de teste: sem modelo"
else:
    from resonance.model import Tutor

    adapter = None if args.base_only else args.adapter
    print(f"Loading {args.base_model}" + (f" + {adapter}" if adapter else ""))
    tutor = Tutor(base_model=args.base_model, adapter=adapter, load_in_4bit=args.load_in_4bit)
    MODEL_LABEL = "Gemma 4 E2B" + (" + Resonance LoRA" if adapter else " (base)") + f" · {tutor.device}"


def answer_stream(question: str, history: list[dict]):
    if tutor is None:
        reply = (
            "**Nível 1 — Intuição:** (modo de teste, sem modelo carregado) Você perguntou: "
            f"{question}\n\n**Nível 2 — Formal:** $E = mc^2$"
        )
        yield reply
        return
    yield from tutor.stream(question, history[-2 * MAX_HISTORY_TURNS:],
                            GenerationConfig(max_new_tokens=1024, temperature=0.3))


# Slider order shared by every event handler.
PARAM_KEYS = ["omega", "gamma", "x0", "v0", "k0", "sigma", "t", "eccentricity", "semi_major", "mass"]


def _params(values) -> dict:
    return dict(zip(PARAM_KEYS, values))


def redraw(sim, *values):
    """Recompute the simulation from the sliders. Takes milliseconds; the model is not called."""
    return sims.render(None if sim == "nenhuma" else sim, _params(values))


def show_controls(sim):
    return [gr.update(visible=sim == name) for name in ("oscilador", "pacote", "orbita")]


def respond(question, history):
    history = history or []
    n_out = 2 + len(PARAM_KEYS)
    if not question or not question.strip():
        yield (history, "") + (gr.update(),) * n_out
        return
    question = question.strip()
    past = list(history)
    history = past + [{"role": "user", "content": question},
                      {"role": "assistant", "content": ""}]
    text = ""
    for text in answer_stream(question, past):
        history[-1]["content"] = text
        yield (history, "") + (gr.update(),) * n_out

    # After the answer: pick a simulation and copy any numbers from the question into the
    # sliders. Changing them triggers `redraw`, which draws the plot.
    sim = sims.suggest_simulation(question, text)
    found = sims.extract_parameters(question)
    if "eccentricity" in found:
        found["eccentricity"] = min(found["eccentricity"], 0.95)
    slider_updates = tuple(gr.update(value=found[k]) if k in found else gr.update() for k in PARAM_KEYS)
    yield (history, "", gr.update(value=sim) if sim else gr.update(), gr.update()) + slider_updates


EXAMPLES = [
    "Derive o período de um pêndulo simples para pequenas oscilações.",
    "Calcule a órbita de um planeta com excentricidade 0,7 e explique a segunda lei de Kepler.",
    "Por que um pacote de onda mais estreito se espalha mais rápido?",
    "Como um oscilador amortecido perde energia ao longo do tempo?",
    "Explique a lei de Snell a partir do princípio de Fermat.",
    "Por que o céu é azul?",
]

CSS = """
.title h1 {font-size: 2.2rem; font-weight: 800; text-align: center; margin: 0;
  background: linear-gradient(135deg, #58a6ff, #56d364); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;}
.title p {text-align: center; opacity: .7; font-size: .8rem; letter-spacing: .05em; margin-top: .3rem;}
"""

LATEX = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
]

D = sims.DEFAULTS

with gr.Blocks(title="Resonance: physics tutor") as demo:
    gr.HTML(
        '<div class="title"><h1>Resonance</h1>'
        f"<p>TUTOR DE FÍSICA EM DOIS NÍVEIS · {MODEL_LABEL.upper()}</p></div>"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=560, latex_delimiters=LATEX, buttons=["copy"],
                                 label="Resonance")
            with gr.Row():
                user_in = gr.Textbox(placeholder="Pergunte sobre física: ondas, óptica, "
                                     "eletromagnetismo, mecânica, astrofísica...",
                                     show_label=False, lines=2, scale=5)
                with gr.Column(scale=1, min_width=90):
                    send = gr.Button("Enviar", variant="primary")
                    clear = gr.Button("Limpar")
            gr.Examples(EXAMPLES, inputs=user_in, label="Experimente")
        with gr.Column(scale=3):
            sim_choice = gr.Dropdown(
                choices=[("Nenhuma", "nenhuma")] + [(v, k) for k, v in sims.SIMULATIONS.items()],
                value="nenhuma", label="Simulação",
                info="Escolhida pela pergunta; troque à vontade. Mexer nos parâmetros "
                     "atualiza o gráfico e os números na hora, sem gerar a resposta de novo.")
            plot = gr.Plot(label="Visualização", format="png")
            with gr.Group(visible=False) as osc_box:
                with gr.Row():
                    omega = gr.Slider(0.5, 20, value=D["omega"], step=0.01, label="ω: frequência natural (rad/s)")
                    gamma = gr.Slider(0, 10, value=D["gamma"], step=0.01, label="γ: amortecimento (1/s)")
                with gr.Row():
                    x0 = gr.Slider(-2, 2, value=D["x0"], step=0.05, label="x₀: posição inicial")
                    v0 = gr.Slider(-20, 20, value=D["v0"], step=0.1, label="v₀: velocidade inicial")
            with gr.Group(visible=False) as pkt_box:
                with gr.Row():
                    k0 = gr.Slider(0, 10, value=D["k0"], step=0.1, label="k₀: número de onda central")
                    sigma = gr.Slider(0.2, 3, value=D["sigma"], step=0.05, label="σ: largura inicial")
                t = gr.Slider(0, 5, value=D["t"], step=0.05, label="t: tempo (evolução livre)")
            with gr.Group(visible=False) as orb_box:
                with gr.Row():
                    ecc = gr.Slider(0, 0.95, value=D["eccentricity"], step=0.01, label="e: excentricidade")
                    a = gr.Slider(0.2, 40, value=D["semi_major"], step=0.1, label="a: semieixo maior (UA)")
                mass = gr.Slider(0.1, 10, value=D["mass"], step=0.1, label="M: massa da estrela (massas solares)")
            results = gr.Markdown()
    gr.Markdown(
        "Respostas geradas por um modelo de 2B parâmetros efetivos: confira as derivações. "
        "[Código e avaliação](https://github.com/victoriaT19/resonance-physics)"
    )

    sliders = [omega, gamma, x0, v0, k0, sigma, t, ecc, a, mass]
    assert len(sliders) == len(PARAM_KEYS)

    chat_outputs = [chatbot, user_in, sim_choice, plot] + sliders
    send.click(respond, [user_in, chatbot], chat_outputs)
    user_in.submit(respond, [user_in, chatbot], chat_outputs)
    clear.click(lambda: ([], "nenhuma"), outputs=[chatbot, sim_choice])

    sim_choice.change(show_controls, sim_choice, [osc_box, pkt_box, orb_box])
    for comp in [sim_choice] + sliders:
        comp.change(redraw, [sim_choice] + sliders, [plot, results],
                    trigger_mode="always_last", show_progress="hidden")


if __name__ == "__main__":
    demo.queue().launch(server_name=os.environ.get("HOST", "127.0.0.1"),
                        server_port=args.port, share=args.share, show_error=True,
                        css=CSS, theme=gr.themes.Soft())
