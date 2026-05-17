"""
tools.py
Ferramentas científicas do Resonance: SymPy, SciPy, Matplotlib.
"""

import io
import re
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq


# ── Estilo visual ─────────────────────────────────────────────────────────────

BLUE = "#58a6ff"
RED  = "#f78166"
BG   = "#0d1117"
GRID = "#21262d"

def _style(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
    if title: ax.set_title(title, color="#e6edf3", pad=8)
    if xl:    ax.set_xlabel(xl, color="#8b949e")
    if yl:    ax.set_ylabel(yl, color="#8b949e")


# ── SymPy ─────────────────────────────────────────────────────────────────────

def sympy_solve(expression: str, operation: str, variable: str = "x") -> str:
    """
    Resolve expressão simbólica com SymPy.
    Operações: simplify, diff, integrate, expand, factor, solve
    """
    try:
        x, t, k, omega = sp.symbols("x t k omega", real=True)
        lmap = {
            "x": x, "t": t, "k": k, "omega": omega,
            "cos": sp.cos, "sin": sp.sin, "exp": sp.exp,
            "sqrt": sp.sqrt, "pi": sp.pi,
        }
        var  = lmap.get(variable, sp.Symbol(variable, real=True))
        expr = sp.sympify(expression, locals=lmap)
        ops  = {
            "simplify":  sp.simplify,
            "diff":      lambda e: sp.diff(e, var),
            "integrate": lambda e: sp.integrate(e, var),
            "expand":    sp.expand,
            "factor":    sp.factor,
            "solve":     lambda e: sp.solve(e, var),
        }
        res = ops.get(operation, sp.simplify)(expr)
        return f"**Resultado:** `{res}`\n\n**LaTeX:** ${sp.latex(res)}$"
    except Exception as e:
        return f"Erro SymPy: {e}"


def extract_and_verify(text: str) -> str:
    """
    Extrai expressões LaTeX do texto e verifica simbolicamente com SymPy.
    Retorna bloco de verificação ou string vazia se nada encontrado.
    """
    x, t, k, omega = sp.symbols("x t k omega", real=True)
    lmap = {
        "x": x, "t": t, "k": k, "omega": omega,
        "cos": sp.cos, "sin": sp.sin, "exp": sp.exp,
        "sqrt": sp.sqrt, "pi": sp.pi,
    }

    results = []
    seen    = set()

    for match in re.findall(r"\$([^$]{3,60})\$", text):
        if match in seen:
            continue
        seen.add(match)
        try:
            expr       = sp.sympify(match, locals=lmap)
            simplified = sp.simplify(expr)
            if str(simplified) != str(expr):
                results.append(
                    f"- `{match}` → ${sp.latex(simplified)}$"
                )
        except Exception:
            continue

    if not results:
        return ""
    return "\n\n---\n**🔢 Verificação simbólica (SymPy):**\n" + "\n".join(results[:3])


# ── SciPy + Matplotlib ────────────────────────────────────────────────────────

def plot_oscillator(omega: float = 2 * np.pi) -> str:
    """Simula e plota oscilador harmônico + espectro FFT."""
    t_eval = np.linspace(0, 10, 1000)
    sol    = solve_ivp(
        lambda t, y: [y[1], -omega**2 * y[0]],
        (0, 10), [1, 0], t_eval=t_eval
    )
    freqs  = fftfreq(1000, d=10 / 1000)
    spec   = np.abs(fft(sol.y[0]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    axes[0].plot(sol.t, sol.y[0], color=BLUE, lw=1.8)
    _style(axes[0], "Oscilador Harmônico", "t (s)", "x(t)")
    axes[1].fill_between(freqs[:500], spec[:500], alpha=0.3, color=RED)
    axes[1].plot(freqs[:500], spec[:500], color=RED, lw=1.5)
    _style(axes[1], "Espectro (FFT)", "Hz", "Amplitude")
    plt.tight_layout(pad=1.5)
    return _save(fig)


def plot_wave_packet(k0: float = 5.0, sigma: float = 1.0) -> str:
    """Plota pacote de onda gaussiano + espectro de momento."""
    x_arr = np.linspace(-10, 10, 1000)
    psi   = np.exp(-x_arr**2 / (2 * sigma**2)) * np.cos(k0 * x_arr)
    spec  = np.abs(fft(psi))
    freqs = fftfreq(1000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    axes[0].fill_between(x_arr, psi, alpha=0.2, color=BLUE)
    axes[0].plot(x_arr, psi, color=BLUE, lw=2)
    _style(axes[0], f"Pacote de Onda (k₀={k0}, σ={sigma})", "x", "ψ(x)")
    axes[1].plot(freqs[:500], spec[:500], color=RED, lw=1.5)
    _style(axes[1], "Espectro do Pacote", "k", "Amplitude")
    plt.tight_layout(pad=1.5)
    return _save(fig)


def plot_orbital(eccentricity: float = 0.5, semi_major: float = 1.0) -> str:
    """Plota órbita kepleriana + distância ao foco."""
    e  = min(max(eccentricity, 0.0), 0.99)  # Limita entre 0 e 0.99
    a  = semi_major
    th = np.linspace(0, 2 * np.pi, 1000)
    r  = a * (1 - e**2) / (1 + e * np.cos(th))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    axes[0].plot(r * np.cos(th), r * np.sin(th), color=BLUE, lw=2)
    axes[0].scatter([0], [0], color=RED, s=120, zorder=5, label="Estrela")
    axes[0].set_aspect("equal")
    axes[0].legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _style(axes[0], f"Órbita Kepleriana (e={e})", "x (UA)", "y (UA)")
    axes[1].plot(th * 180 / np.pi, r, color=RED, lw=2)
    _style(axes[1], "Distância ao foco", "θ (°)", "r (UA)")
    plt.tight_layout(pad=1.5)
    return _save(fig)


def detect_and_plot(question: str, response: str):
    """
    Detecta automaticamente qual simulação gerar baseado no contexto.
    Retorna path do arquivo de imagem ou None.
    """
    text   = (question + " " + response).lower()
    scores = {"oscillator": 0, "wave_packet": 0, "orbital": 0}

    osc_words   = ["oscilador", "harmônico", "frequência", "período",
                   "oscillator", "vibração", "ressonância", "mola"]
    wave_words  = ["pacote", "difração", "interferência", "onda",
                   "wave", "quântico", "dispersão"]
    orbit_words = ["órbita", "orbital", "kepleriana", "planeta",
                   "elipse", "excentricidade", "kepler", "gravitação"]

    for w in osc_words:   scores["oscillator"]  += text.count(w)
    for w in wave_words:  scores["wave_packet"]  += text.count(w)
    for w in orbit_words: scores["orbital"]      += text.count(w)

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None

    if best == "oscillator":
        match = re.search(r"omega\s*[=:]\s*([\d.]+)", question.lower())
        omega = float(match.group(1)) if match else 2 * np.pi
        return plot_oscillator(omega=omega)

    elif best == "wave_packet":
        match = re.search(r"k[₀0]?\s*[=:]\s*([\d.]+)", question.lower())
        k0    = float(match.group(1)) if match else 5.0
        return plot_wave_packet(k0=k0)

    elif best == "orbital":
        match = re.search(r"excentricidade\s+(?:de\s+)?(\d+\.?\d*)", question.lower())
        if match:
            val = float(match.group(1))
            ecc = val if val < 1.0 else val / 10
        else:
            ecc = 0.5
        return plot_orbital(eccentricity=ecc)

    return None


# ── Utilitário ────────────────────────────────────────────────────────────────

def _save(fig) -> str:
    """Salva figura em arquivo temporário e retorna o path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return tmp.name
