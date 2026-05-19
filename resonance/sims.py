"""Physics simulations shown next to the chat.

Each simulation is a closed-form or FFT computation that takes milliseconds, so the app
recomputes the plot and the numbers every time a parameter changes, without asking the
model again. `render(name, params)` returns (figure, results_markdown).
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE = "#58a6ff"
RED = "#f78166"
GREEN = "#56d364"
YELLOW = "#e3b341"
BG = "#0d1117"
GRID = "#21262d"
TEXT = "#e6edf3"
DIM = "#8b949e"


def _style(ax, title="", xl="", yl=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")
    ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=TEXT, pad=8, fontsize=10)
    if xl:
        ax.set_xlabel(xl, color=DIM)
    if yl:
        ax.set_ylabel(yl, color=DIM)


def _legend(ax, **kwargs):
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8, **kwargs)


def _fmt(x: float, digits: int = 3) -> str:
    """Number in Brazilian notation (decimal comma)."""
    if x is None or not np.isfinite(x):
        return "∞"
    return f"{x:.{digits}g}".replace(".", ",")


# ── Damped harmonic oscillator ──────────────────────────────────────────────


def oscillator_regime(omega: float, gamma: float) -> str:
    disc = (gamma / 2) ** 2 - omega**2
    if abs(disc) < 1e-9 * max(omega**2, 1.0):
        return "criticamente amortecido"
    return "subamortecido" if disc < 0 else "superamortecido"


def oscillator_peak_hz(omega: float, gamma: float) -> float:
    """Frequency of the damped oscillation, in Hz (0 when it does not oscillate)."""
    wd2 = omega**2 - (gamma / 2) ** 2
    return float(np.sqrt(wd2) / (2 * np.pi)) if wd2 > 1e-12 else 0.0


def oscillator_solution(t: np.ndarray, omega: float, gamma: float, x0: float, v0: float) -> np.ndarray:
    """Exact solution of x'' + γx' + ω²x = 0 with x(0)=x0, x'(0)=v0."""
    half = gamma / 2
    disc = half**2 - omega**2
    if abs(disc) < 1e-9 * max(omega**2, 1.0):  # critical
        return np.exp(-half * t) * (x0 + (v0 + half * x0) * t)
    if disc < 0:  # under-damped
        wd = np.sqrt(-disc)
        return np.exp(-half * t) * (x0 * np.cos(wd * t) + (v0 + half * x0) / wd * np.sin(wd * t))
    root = np.sqrt(disc)  # over-damped
    r1, r2 = -half + root, -half - root
    a = (v0 - r2 * x0) / (r1 - r2)
    return a * np.exp(r1 * t) + (x0 - a) * np.exp(r2 * t)


def oscillator(omega: float, gamma: float, x0: float = 1.0, v0: float = 0.0,
               t_max: float = 20.0, n: int = 4096) -> dict:
    t = np.linspace(0, t_max, n)
    x = oscillator_solution(t, omega, gamma, x0, v0)
    v = np.gradient(x, t)
    energy = 0.5 * v**2 + 0.5 * omega**2 * x**2
    freqs = np.fft.rfftfreq(n, d=t[1] - t[0])
    amp = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(n)))
    f_meas = float(freqs[np.argmax(amp[1:]) + 1])
    return {"t": t, "x": x, "energy": energy, "freqs": freqs, "amp": amp,
            "f_theory": oscillator_peak_hz(omega, gamma), "f_measured": f_meas,
            "regime": oscillator_regime(omega, gamma)}


def plot_oscillator(omega: float = 2 * np.pi, gamma: float = 0.3, x0: float = 1.0, v0: float = 0.0):
    s = oscillator(omega, gamma, x0, v0)
    t, x = s["t"], s["x"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    axes[0].plot(t, x, color=BLUE, lw=1.5, label="x(t)")
    if gamma > 0 and s["regime"] == "subamortecido":
        amp0 = np.sqrt(x0**2 + ((v0 + gamma * x0 / 2) / (2 * np.pi * s["f_theory"])) ** 2)
        env = amp0 * np.exp(-gamma * t / 2)
        axes[0].plot(t, env, color=DIM, lw=0.8, ls="--", label=r"envoltória $\propto e^{-\gamma t/2}$")
        axes[0].plot(t, -env, color=DIM, lw=0.8, ls="--")
    e0 = s["energy"][0] or 1.0
    ax_e = axes[0].twinx()
    ax_e.plot(t, s["energy"] / e0, color=YELLOW, lw=1, alpha=0.7)
    ax_e.set_ylim(0, 1.05)
    ax_e.tick_params(colors=YELLOW)
    ax_e.set_ylabel("E / E₀", color=YELLOW)
    _legend(axes[0], loc="upper right")
    _style(axes[0], f"Oscilador {s['regime']} (ω = {_fmt(omega)} rad/s, γ = {_fmt(gamma)} s⁻¹)",
           "t (s)", "x(t)")

    f_max = max(3 * omega / (2 * np.pi), 1.0)
    sel = s["freqs"] <= f_max
    axes[1].fill_between(s["freqs"][sel], s["amp"][sel], alpha=0.3, color=RED)
    axes[1].plot(s["freqs"][sel], s["amp"][sel], color=RED, lw=1.4)
    if s["f_theory"] > 0:
        axes[1].axvline(s["f_theory"], color=GREEN, lw=1, ls="--",
                        label=f"teoria: {_fmt(s['f_theory'])} Hz")
        _legend(axes[1])
    _style(axes[1], "Espectro (FFT)", "f (Hz)", "amplitude")
    fig.tight_layout(pad=1.5)

    q = omega / gamma if gamma > 0 else float("inf")
    tau = 2 / gamma if gamma > 0 else float("inf")
    period = 1 / s["f_theory"] if s["f_theory"] > 0 else float("inf")
    results = (
        f"**Regime:** {s['regime']}  \n"
        f"**Frequência amortecida:** teoria √(ω² − γ²/4)/2π = {_fmt(s['f_theory'])} Hz · "
        f"pico da FFT = {_fmt(s['f_measured'])} Hz  \n"
        f"**Período:** {_fmt(period)} s · **fator de qualidade** Q = ω/γ = {_fmt(q)}  \n"
        f"**Tempo de decaimento da amplitude** 2/γ = {_fmt(tau)} s · "
        f"**energia restante** em t = 20 s: {_fmt(100 * s['energy'][-1] / e0, 2)} %"
    )
    return fig, results


# ── Free Gaussian wave packet ───────────────────────────────────────────────


def wave_packet(k0: float = 5.0, sigma: float = 1.0, t: float = 0.0,
                n: int = 8192, half_width: float = 160.0):
    """Free Gaussian packet in units ħ = m = 1.

    ψ(x, 0) = exp(-x²/(4σ²)) e^{ik₀x}, so |ψ|² has standard deviation σ and |φ(k)|²
    has 1/(2σ): σₓσₖ = 1/2, the minimum allowed by Δx Δp ≥ ħ/2. Each plane wave evolves
    as e^{-ik²t/2}; the packet moves at the group velocity k₀ and spreads as
    σ(t) = σ √(1 + (t / 2σ²)²).
    """
    x = np.linspace(-half_width, half_width, n, endpoint=False)
    dx = x[1] - x[0]
    psi0 = np.exp(-x**2 / (4 * sigma**2)) * np.exp(1j * k0 * x)
    k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
    phi = np.fft.fft(psi0)
    psi_t = np.fft.ifft(phi * np.exp(-1j * k**2 * t / 2))
    order = np.argsort(k)
    return x, psi0, psi_t, k[order], phi[order]


def spread(values: np.ndarray, weights: np.ndarray) -> float:
    w = weights / weights.sum()
    mean = np.sum(values * w)
    return float(np.sqrt(np.sum((values - mean) ** 2 * w)))


def mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / weights.sum())


def wave_packet_width_theory(sigma: float, t: float) -> float:
    return float(sigma * np.sqrt(1 + (t / (2 * sigma**2)) ** 2))


def plot_wave_packet(k0: float = 5.0, sigma: float = 1.0, t: float = 0.0):
    x, psi0, psi_t, k, phi = wave_packet(k0, sigma, t)
    p0, pt = np.abs(psi0) ** 2, np.abs(psi_t) ** 2
    sx0, sxt = spread(x, p0), spread(x, pt)
    sk = spread(k, np.abs(phi) ** 2)
    xc = mean(x, pt)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    lo = min(-4 * sigma, xc - 5 * sxt)
    hi = max(4 * sigma, xc + 5 * sxt)
    sel = (x >= lo) & (x <= hi)
    axes[0].plot(x[sel], np.abs(psi0[sel]), color=DIM, lw=1, ls="--", label="|ψ(x, 0)|")
    axes[0].plot(x[sel], psi_t.real[sel], color=BLUE, lw=1.2, alpha=0.8, label=f"Re ψ(x, t={_fmt(t)})")
    axes[0].plot(x[sel], np.abs(psi_t[sel]), color=GREEN, lw=1.6, label=f"|ψ(x, t={_fmt(t)})|")
    _legend(axes[0], loc="upper right")
    _style(axes[0], f"Pacote livre (k₀ = {_fmt(k0)}, σ = {_fmt(sigma)}, t = {_fmt(t)})", "x", "ψ")

    ksel = np.abs(k - k0) <= 4 / sigma + 1
    axes[1].plot(k[ksel], np.abs(phi[ksel]) / np.abs(phi).max(), color=RED, lw=1.5)
    axes[1].axvline(k0, color=GREEN, lw=1, ls="--", label=f"k₀ = {_fmt(k0)}")
    _legend(axes[1])
    _style(axes[1], "Espectro de momento (não muda com t)", "k", "|φ(k)| (normalizado)")
    fig.tight_layout(pad=1.5)

    results = (
        f"**Em t = 0:** σₓ = {_fmt(sx0)}, σₖ = {_fmt(sk)}, σₓ·σₖ = {_fmt(sx0 * sk)} "
        f"(mínimo de Heisenberg: 0,5)  \n"
        f"**Em t = {_fmt(t)}:** largura σₓ = {_fmt(sxt)} (teoria σ√(1 + (t/2σ²)²) = "
        f"{_fmt(wave_packet_width_theory(sigma, t))}); centro ⟨x⟩ = {_fmt(xc)} "
        f"(velocidade de grupo × t = {_fmt(k0 * t)})  \n"
        f"Unidades com ħ = m = 1. Pacotes mais estreitos (σ pequeno) espalham mais rápido."
    )
    return fig, results


# ── Kepler orbit ────────────────────────────────────────────────────────────

AU_M = 1.495978707e11
GM_SUN = 1.32712440018e20  # m³/s²
YEAR_S = 365.25 * 86400


def solve_kepler(mean_anomaly: np.ndarray, e: float) -> np.ndarray:
    """Eccentric anomaly E from M = E − e sin E (Newton's method)."""
    E = np.where(e < 0.8, mean_anomaly, np.pi * np.ones_like(mean_anomaly))
    for _ in range(50):
        E = E - (E - e * np.sin(E) - mean_anomaly) / (1 - e * np.cos(E))
    return E


def kepler_orbit(eccentricity: float, semi_major: float = 1.0, n: int = 1000):
    """Positions sampled uniformly in time over one period.

    Returns t (fraction of the period), r (AU), true anomaly θ and speed in units GM = 1.
    """
    e = float(np.clip(eccentricity, 0.0, 0.99))
    a = float(semi_major)
    M = np.linspace(0, 2 * np.pi, n)
    E = solve_kepler(M, e)
    th = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    v = np.sqrt(2 / r - 1 / a)
    return M / (2 * np.pi), r, th, v


def plot_orbital(eccentricity: float = 0.5, semi_major: float = 1.0, mass: float = 1.0):
    e = float(np.clip(eccentricity, 0.0, 0.99))
    a = float(semi_major)
    frac, r, th, _ = kepler_orbit(e, a)
    gm = GM_SUN * mass
    v_kms = np.sqrt(gm * (2 / (r * AU_M) - 1 / (a * AU_M))) / 1e3
    period_yr = np.sqrt(a**3 / mass)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BG)
    xs, ys = r * np.cos(th), r * np.sin(th)
    axes[0].plot(xs, ys, color=BLUE, lw=1.8, label="órbita")
    marks = np.linspace(0, len(frac) - 1, 13).astype(int)[:-1]
    axes[0].scatter(xs[marks], ys[marks], color=YELLOW, s=14, zorder=4,
                    label="posições a cada 1/12 do período")
    axes[0].scatter([0], [0], color=RED, s=100, zorder=5, label="estrela (foco)")
    y_max = float(np.max(np.abs(ys)))
    axes[0].set_ylim(-1.8 * y_max, 1.2 * y_max)
    axes[0].set_aspect("equal", adjustable="box")
    _legend(axes[0], loc="lower center", ncol=2)
    _style(axes[0], f"Órbita (e = {_fmt(e)}, a = {_fmt(a)} UA, M = {_fmt(mass)} M☉)", "x (UA)", "y (UA)")

    ax2 = axes[1]
    ax2.plot(frac * period_yr, r, color=RED, lw=1.8, label="r (UA)")
    ax2.set_ylabel("r (UA)", color=RED)
    ax_v = ax2.twinx()
    ax_v.plot(frac * period_yr, v_kms, color=GREEN, lw=1.8)
    ax_v.set_ylabel("v (km/s)", color=GREEN)
    ax_v.tick_params(colors=GREEN)
    _style(ax2, "Distância e velocidade ao longo de um período", "t (anos)", "")
    ax2.set_ylabel("r (UA)", color=RED)
    fig.tight_layout(pad=1.5)

    rp, ra = a * (1 - e), a * (1 + e)
    results = (
        f"**Periélio** a(1 − e) = {_fmt(rp)} UA · **afélio** a(1 + e) = {_fmt(ra)} UA · "
        f"**semieixo menor** a√(1 − e²) = {_fmt(a * np.sqrt(1 - e**2))} UA  \n"
        f"**Período** (3ª lei de Kepler, T² = a³/M) = {_fmt(period_yr)} anos  \n"
        f"**Velocidade** no periélio = {_fmt(v_kms.max())} km/s · no afélio = {_fmt(v_kms.min())} km/s · "
        f"razão (1 + e)/(1 − e) = {_fmt((1 + e) / (1 - e))}  \n"
        f"Os pontos amarelos estão igualmente espaçados no tempo: mais separados perto do "
        f"periélio, onde o planeta é mais rápido (2ª lei, áreas iguais em tempos iguais)."
    )
    return fig, results


# ── Selection ───────────────────────────────────────────────────────────────

SIMULATIONS = {
    "oscilador": "Oscilador harmônico amortecido",
    "pacote": "Pacote de onda livre",
    "orbita": "Órbita kepleriana",
}

DEFAULTS = {
    "omega": 2 * np.pi, "gamma": 0.3, "x0": 1.0, "v0": 0.0,
    "k0": 5.0, "sigma": 1.0, "t": 0.0,
    "eccentricity": 0.5, "semi_major": 1.0, "mass": 1.0,
}

_KEYWORDS = {
    "oscilador": [r"oscilador", r"harmônic\w*", r"massa[- ]mola", r"\bmola\b",
                  r"pêndulo", r"amortec\w*", r"ressonância", r"vibraç\w*", r"oscilaç\w*"],
    "pacote": [r"pacote de onda", r"pacote gaussiano", r"incerteza", r"heisenberg",
               r"de broglie", r"função de onda", r"velocidade de grupo", r"dispersão"],
    "orbita": [r"órbita\w*", r"orbital", r"kepler\w*", r"planeta\w*", r"excentricidade",
               r"periélio", r"afélio", r"satélite\w*", r"gravitaç\w*"],
}


def suggest_simulation(question: str, answer: str = "") -> str | None:
    """Pick the simulation that matches the conversation, or None.

    The question counts three times as much as the answer, so a passing mention in a
    long answer does not override what was asked.
    """
    scores = {}
    for name, patterns in _KEYWORDS.items():
        q = sum(len(re.findall(p, question, flags=re.IGNORECASE)) for p in patterns)
        a = sum(len(re.findall(p, answer, flags=re.IGNORECASE)) for p in patterns)
        scores[name] = 3 * q + a
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else None


def extract_parameters(question: str) -> dict:
    """Numbers the user wrote in the question, such as 'excentricidade 0,7' or 'ω = 3'."""
    params = {}
    text = question.lower().replace(",", ".")
    patterns = {
        "eccentricity": r"excentricidade\s*(?:de\s*|=\s*)?(\d*\.?\d+)",
        "omega": r"(?:ω|omega)\s*(?:=|de)\s*(\d*\.?\d+)",
        "gamma": r"(?:γ|gamma)\s*(?:=|de)\s*(\d*\.?\d+)",
        "k0": r"k[₀0]\s*(?:=|de)\s*(\d*\.?\d+)",
        "semi_major": r"semieixo maior\s*(?:de\s*|=\s*)?(\d*\.?\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            params[key] = float(m.group(1))
    return params


def render(name: str | None, params: dict | None = None):
    """Return (figure, results markdown) for a simulation, or (None, "") for none."""
    p = {**DEFAULTS, **(params or {})}
    if name == "oscilador":
        return plot_oscillator(p["omega"], p["gamma"], p["x0"], p["v0"])
    if name == "pacote":
        return plot_wave_packet(p["k0"], p["sigma"], p["t"])
    if name == "orbita":
        return plot_orbital(p["eccentricity"], p["semi_major"], p["mass"])
    return None, ""
