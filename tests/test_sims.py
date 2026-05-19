import time

import numpy as np
import pytest

from resonance import sims


# ── Oscillator ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("omega,gamma,x0,v0", [
    (2.0, 0.5, 1.0, 0.0),   # under-damped
    (2.0, 4.0, 1.0, 0.5),   # critical (γ = 2ω)
    (2.0, 7.0, -0.5, 2.0),  # over-damped
])
def test_oscillator_solution_satisfies_ode_and_initial_conditions(omega, gamma, x0, v0):
    t = np.linspace(0, 5, 20001)
    x = sims.oscillator_solution(t, omega, gamma, x0, v0)
    v = np.gradient(x, t)
    acc = np.gradient(v, t)
    residual = acc + gamma * v + omega**2 * x
    assert x[0] == pytest.approx(x0, abs=1e-12)
    assert v[0] == pytest.approx(v0, abs=1e-2)
    assert np.max(np.abs(residual[5:-5])) < 1e-3 * max(1.0, omega**2)


def test_oscillator_regimes():
    assert sims.oscillator_regime(2.0, 0.5) == "subamortecido"
    assert sims.oscillator_regime(2.0, 4.0) == "criticamente amortecido"
    assert sims.oscillator_regime(2.0, 7.0) == "superamortecido"
    assert sims.oscillator_peak_hz(1.0, 3.0) == 0.0


def test_oscillator_spectrum_peaks_at_theory():
    s = sims.oscillator(2 * np.pi * 1.5, 0.2)
    assert s["f_measured"] == pytest.approx(s["f_theory"], abs=0.06)


def test_undamped_oscillator_conserves_energy():
    s = sims.oscillator(3.0, 0.0, 1.0, 0.0)
    e = s["energy"][5:-5]
    assert np.ptp(e) / e.mean() < 1e-3


# ── Wave packet ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_wave_packet_is_minimum_uncertainty(sigma):
    x, psi0, _, k, phi = sims.wave_packet(k0=5.0, sigma=sigma)
    sx = sims.spread(x, np.abs(psi0) ** 2)
    sk = sims.spread(k, np.abs(phi) ** 2)
    assert sx == pytest.approx(sigma, rel=0.01)
    assert sx * sk == pytest.approx(0.5, rel=0.03)
    assert k[np.argmax(np.abs(phi))] == pytest.approx(5.0, abs=0.05)


@pytest.mark.parametrize("sigma,t", [(1.0, 2.0), (0.5, 3.0), (0.2, 5.0)])
def test_wave_packet_moves_at_group_velocity_and_spreads(sigma, t):
    k0 = 4.0
    x, _, psi_t, _, _ = sims.wave_packet(k0=k0, sigma=sigma, t=t)
    p = np.abs(psi_t) ** 2
    assert sims.mean(x, p) == pytest.approx(k0 * t, abs=0.02)
    assert sims.spread(x, p) == pytest.approx(sims.wave_packet_width_theory(sigma, t), rel=0.01)


# ── Kepler orbit ────────────────────────────────────────────────────────────

def test_kepler_equation_solver():
    e = 0.9
    M = np.linspace(0, 2 * np.pi, 500)
    E = sims.solve_kepler(M, e)
    assert np.max(np.abs(E - e * np.sin(E) - M)) < 1e-10


def test_kepler_orbit_geometry_and_second_law():
    frac, r, th, v = sims.kepler_orbit(0.5, 2.0)
    assert r.min() == pytest.approx(2.0 * (1 - 0.5), rel=1e-6)
    assert r.max() == pytest.approx(2.0 * (1 + 0.5), rel=1e-4)
    # Orbit equation r = a(1-e²)/(1+e cos θ)
    assert np.allclose(r, 2.0 * (1 - 0.25) / (1 + 0.5 * np.cos(th)), rtol=1e-8)
    # Samples are uniform in time, so equal areas per step: r² dθ/dt is constant
    dth = np.diff(np.unwrap(th))
    areal = (r[:-1] ** 2) * dth
    assert np.ptp(areal) / areal.mean() < 0.02
    assert v.max() / v.min() == pytest.approx(1.5 / 0.5, rel=1e-3)


def test_earth_orbit_numbers():
    _, text = sims.plot_orbital(0.0167, 1.0, 1.0)
    assert "Período** (3ª lei de Kepler, T² = a³/M) = 1 anos" in text
    assert "30," in text  # ~30 km/s orbital speed


# ── Selection and rendering ─────────────────────────────────────────────────

@pytest.mark.parametrize("question,expected", [
    ("Derive o período de um pêndulo simples.", "oscilador"),
    ("Calcule a órbita de um planeta com excentricidade 0,7.", "orbita"),
    ("Como o princípio da incerteza aparece em um pacote de onda?", "pacote"),
    ("Explique as equações de Maxwell.", None),
    ("Por que o céu é azul?", None),
])
def test_suggest_simulation(question, expected):
    assert sims.suggest_simulation(question) == expected


def test_answer_mentions_do_not_override_the_question():
    answer = "Uma onda eletromagnética... uma onda... a onda se propaga como uma onda."
    assert sims.suggest_simulation("Explique as equações de Maxwell.", answer) is None


def test_extract_parameters():
    p = sims.extract_parameters("órbita com excentricidade 0,7, semieixo maior de 5 e ω = 3")
    assert p == {"eccentricity": 0.7, "omega": 3.0, "semi_major": 5.0}


@pytest.mark.parametrize("name", ["oscilador", "pacote", "orbita"])
def test_render_returns_figure_and_results(name):
    import matplotlib.pyplot as plt

    fig, text = sims.render(name, {})
    assert fig is not None and text.startswith("**")
    plt.close(fig)


def test_render_none():
    assert sims.render(None) == (None, "")


@pytest.mark.parametrize("name,params", [
    ("oscilador", {"omega": 12.0, "gamma": 2.0}),
    ("pacote", {"sigma": 0.3, "t": 4.0}),
    ("orbita", {"eccentricity": 0.9, "semi_major": 30.0}),
])
def test_render_is_fast_enough_for_live_sliders(name, params):
    import matplotlib.pyplot as plt

    sims.render(name, params)  # warm-up
    start = time.perf_counter()
    fig, _ = sims.render(name, params)
    elapsed = time.perf_counter() - start
    plt.close(fig)
    assert elapsed < 1.0
