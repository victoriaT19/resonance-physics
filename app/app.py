"""
app.py
Interface Gradio do Resonance — Physics AI.
Roda localmente com Gemma 4 E2B-IT via Unsloth.

Uso:
    python app.py
    python app.py --share    # link público
    python app.py --port 8080
"""

import argparse
import torch
import gradio as gr
from tools import detect_and_plot, extract_and_verify

# ── Argumentos ────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true")
parser.add_argument("--port",  type=int, default=7860)
parser.add_argument("--model", default="unsloth/gemma-4-E2B-it")
args, _ = parser.parse_known_args()

# ── Carrega modelo ────────────────────────────────────────────────────────────

print(f"Carregando modelo: {args.model}")
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name     = args.model,
    max_seq_length = 1024,
    load_in_4bit   = True,
)
FastModel.for_inference(model)
END_OF_TURN_ID = tokenizer.tokenizer.convert_tokens_to_ids("<end_of_turn>")
print("✅ Modelo carregado")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM = (
    "Você é Resonance, assistente científico especializado em física. "
    "Responda sempre em dois níveis:\n"
    "**Nível 1 — Intuição:** explicação acessível sem formalismo excessivo.\n"
    "**Nível 2 — Formal:** tratamento matemático completo com equações.\n"
    "Não se apresente. Vá direto à resposta."
)

# ── Inferência ────────────────────────────────────────────────────────────────

def generate(question: str, max_new_tokens: int = 500) -> str:
    prompt = (
        f"<bos><start_of_turn>user\n"
        f"{SYSTEM}\n\n"
        f"{question}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs = tokenizer.tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens     = max_new_tokens,
            do_sample          = True,
            temperature        = 0.3,
            repetition_penalty = 1.3,
            eos_token_id       = [tokenizer.tokenizer.eos_token_id, END_OF_TURN_ID],
            pad_token_id       = tokenizer.tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

# ── Interface ─────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400&family=Syne:wght@700;800&display=swap');
:root{--bg:#0d1117;--bg2:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;--blue:#58a6ff;--green:#56d364;}
body,.gradio-container{background:var(--bg)!important;font-family:'JetBrains Mono',monospace!important;color:var(--text)!important;}
.title h1{font-family:'Syne',sans-serif!important;font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,#58a6ff,#56d364);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent;text-align:center;margin:0;}
.title p{color:var(--dim);text-align:center;font-size:.8rem;letter-spacing:.06em;margin-top:.3rem;}
#chatbot{background:var(--bg2)!important;border:1px solid var(--border)!important;min-height:460px;}
#send{background:var(--blue)!important;color:#0d1117!important;font-weight:700!important;border:none!important;}
#plot{background:var(--bg2)!important;border:1px solid var(--border)!important;}
"""

EXAMPLES = [
    "Derive a equação de onda para um oscilador harmônico e mostre o espectro de frequências.",
    "O que é um pacote de onda quântico? Mostre a evolução e o espectro de momento.",
    "Calcule a órbita de um planeta com excentricidade 0.7 e explique as leis de Kepler.",
    "Derive as equações de Maxwell no vácuo e mostre como implicam a velocidade da luz c.",
    "Explique a lei de Snell a partir do princípio de Fermat.",
    "Como calcular o redshift gravitacional de uma estrela de nêutrons?",
    "Como o princípio da incerteza de Heisenberg emerge da mecânica ondulatória?",
    "Derive a relação de dispersão para ondas eletromagnéticas no vácuo.",
]


def respond(msg, history):
    if not msg.strip():
        return history, history, None, ""

    resp = generate(msg, max_new_tokens=500)

    # Verificação simbólica com SymPy
    sympy_block = extract_and_verify(resp)
    if sympy_block:
        resp += sympy_block

    history   = history + [[msg, resp]]
    plot_path = detect_and_plot(msg, resp)

    return history, history, plot_path, ""


with gr.Blocks(css=CSS, title="Resonance — Physics AI") as demo:
    gr.HTML(
        '<div class="title">'
        "<h1>Resonance</h1>"
        "<p>GEMMA 4 E2B-IT · PHYSICS REASONING · SYMPY · SCIPY · MATPLOTLIB</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id          = "chatbot",
                height           = 480,
                show_copy_button = True,
                render_markdown  = True,
                label            = "",
            )
            with gr.Row():
                user_in = gr.Textbox(
                    placeholder = "Pergunte sobre física — ondas, óptica, eletromagnetismo, astronomia...",
                    show_label  = False,
                    lines       = 2,
                    scale       = 5,
                )
                with gr.Column(scale=1, min_width=80):
                    send_btn  = gr.Button("Enviar", elem_id="send", variant="primary")
                    clear_btn = gr.Button("Limpar")

        with gr.Column(scale=2):
            plot_out = gr.Image(
                elem_id = "plot",
                label   = "Visualização Científica",
                height  = 400,
            )

    gr.Examples(EXAMPLES, inputs=user_in, label="Experimente")
    history_st = gr.State([])

    send_btn.click(respond,  [user_in, history_st], [chatbot, history_st, plot_out, user_in])
    user_in.submit(respond,  [user_in, history_st], [chatbot, history_st, plot_out, user_in])
    clear_btn.click(lambda: ([], [], None), outputs=[chatbot, history_st, plot_out])


if __name__ == "__main__":
    demo.launch(
        server_port = args.port,
        share       = args.share,
        show_error  = True,
    )
