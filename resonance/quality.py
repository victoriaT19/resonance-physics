"""Automatic checks on a generated answer.

Used twice: to filter teacher answers before they enter the training set, and to score
model answers during evaluation. The checks are cheap and deterministic; physical
correctness is judged separately (key facts here, an optional LLM judge in evaluate.py).
"""

from __future__ import annotations

import re
import unicodedata

from .prompts import LEVEL1_HEADER, LEVEL2_HEADER

# Text that means the answer leaked template markup or ignored the system prompt.
ARTIFACT_PATTERNS = [
    r"<\|?turn\|?>",
    r"<start_of_turn>|<end_of_turn>",
    r"</?model>",
    r"<\|?channel\|?>",
    r"\[SYSTEM\]",
]
SELF_INTRO_PATTERNS = [
    r"\beu sou (o |a )?resonance\b",
    r"\bsou (o |a )?resonance\b",
    r"^\s*\**\s*resonance\s*\**\s*:",
    r"^\s*olá[!,.]",
]

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "nabla": "∇", "partial": "∂",
    "hbar": "ħ", "infty": "∞",
}
_SYMBOLS = {
    "geq": "≥", "ge": "≥", "leq": "≤", "le": "≤", "approx": "≈", "propto": "∝",
    "times": "×", "neq": "≠", "max": "max", "min": "min",
}
# Commands whose name carries no meaning for matching (formatting or spacing).
_DROP_COMMANDS = (
    "mathbf", "mathrm", "mathit", "boldsymbol", "vec", "text", "textbf", "textrm",
    "operatorname", "displaystyle", "left", "right", "big", "Big", "bigg", "Bigg",
    "cdot", "quad", "qquad", "frac", "sqrt", "sin", "cos", "tan", "ln", "log", "exp",
)
_KEEP_LETTERS = {"frac", "sqrt", "sin", "cos", "tan", "ln", "log", "exp"}


def normalize_math(text: str) -> str:
    """Collapse LaTeX and Unicode spellings of a formula into one comparable string.

    `a \\sin\\theta = m\\lambda`, `$a\\sin(\\theta)=m\\lambda$` and `a sinθ = mλ` all become
    `asinθ=mλ`. Function names such as `sin` and `sqrt` are kept as letters, so the
    result is only meant for substring matching against facts normalized the same way.
    """
    s = unicodedata.normalize("NFC", text)
    s = re.sub(r"\\[dt]frac(?![A-Za-z])", r"\\frac", s)
    # \Theta and \theta both map to θ (case is dropped at the end anyway).
    for name, sym in sorted(_GREEK.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(r"\\" + name + r"(?![A-Za-z])", sym, s, flags=re.IGNORECASE)
    for name, sym in sorted(_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(r"\\" + name + r"(?![A-Za-z])", sym, s)
    # Keep the letters of function names (sin, sqrt, frac) but drop the backslash.
    s = re.sub(r"\\(" + "|".join(_DROP_COMMANDS) + r")(?![A-Za-z])",
               lambda m: m.group(1) if m.group(1) in _KEEP_LETTERS else "",
               s)
    s = re.sub(r"\\[,;:! ]", "", s)
    s = s.replace("\\", "")
    s = s.replace("·", "").replace("⋅", "").replace("*", "")
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("Φ", "φ").replace("ϕ", "φ").replace("ϑ", "θ")
    s = re.sub(r"[\s${}()\[\]]", "", s)
    # Portuguese writes the sine as "sen"; match it with "sin".
    return s.lower().replace("sen", "sin")


def has_both_levels(text: str) -> bool:
    t = text.lower()
    i1 = t.find(LEVEL1_HEADER.lower())
    i2 = t.find(LEVEL2_HEADER.lower())
    return i1 != -1 and i2 != -1 and i1 < i2


def has_artifacts(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in ARTIFACT_PATTERNS)


def has_self_intro(text: str) -> bool:
    return any(
        re.search(p, text, flags=re.IGNORECASE | re.MULTILINE) for p in SELF_INTRO_PATTERNS
    )


def balanced_math(text: str) -> bool:
    """Every $ and $$ delimiter is closed."""
    without_display = text.replace("$$", "")
    return text.count("$$") % 2 == 0 and without_display.count("$") % 2 == 0


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))


def key_fact_hits(text: str, key_facts: list[list[str]]) -> list[bool]:
    """For each fact (a list of acceptable spellings), whether any spelling appears."""
    norm = normalize_math(text)
    return [any(normalize_math(alt) in norm for alt in alts) for alts in key_facts]


def check_answer(text: str, truncated: bool = False) -> dict:
    """All format checks for one answer, plus an overall `ok` flag."""
    result = {
        "both_levels": has_both_levels(text),
        "no_artifacts": not has_artifacts(text),
        "no_self_intro": not has_self_intro(text),
        "balanced_math": balanced_math(text),
        "complete": not truncated,
        "words": word_count(text),
    }
    result["ok"] = all(
        result[k] for k in ("both_levels", "no_artifacts", "no_self_intro", "balanced_math", "complete")
    )
    return result


def acceptable_training_answer(text: str, finish_reason: str | None,
                               min_words: int = 120, max_words: int = 900) -> tuple[bool, str]:
    """Filter for teacher answers. Returns (accepted, reason)."""
    if finish_reason not in (None, "stop"):
        return False, f"finish_reason={finish_reason}"
    checks = check_answer(text)
    for key in ("both_levels", "no_artifacts", "no_self_intro", "balanced_math"):
        if not checks[key]:
            return False, key
    if not (min_words <= checks["words"] <= max_words):
        return False, f"words={checks['words']}"
    return True, "ok"
