# SPDX-License-Identifier: Apache-2.0
"""Cloth accuracy: ``alloy.Scene(cloth_accuracy="default" | "super" | "fast")`` (SCR-WP2-13; ``liquid_accuracy``'s
words, and "fast").

The router (``RouterConfig.cloth_accuracy``) tags every CLOTH_VBD decision with ``EFFECT[setting]``, and the scene's
VBD entry (``alloy.scene.entries.solver_kwargs``) then runs ``KWARGS[setting]``: SolverVBDAlloy's adaptive initial
guess and Chebyshev-accelerated iterations (alloy/solvers/vbd/accel.py), at DEFAULT_ITERATIONS by default and
SUPER_ITERATIONS for "super". "fast" runs VBD's own solve (its inertial start, no Chebyshev, FAST_ITERATIONS = 10:
the behaviour before the switch). The entry is shared, so soft solids in a scene with cloth step with the same
setting. A VBD entry without cloth (soft solids only) carries no tag and is unchanged.

Why (rulings 2026-11-08 10:15Z and 11:08Z; docs/notes/wp2-results.md, SCR-WP2-13): at VBD's own 10 iterations cloth
is far from converged. A 30 cm disc draped over an 18 cm stand stretches up to 10 % and hangs too low, at drape
coefficient 0.19 against its converged 0.74 (1 stays flat, 0 hangs straight down). The default (WP11's pick, with
its self-contact rule) reaches 0.728 at 0.36 % stretch; "super" reaches 0.732 at 0.34 %, at about 1.4x the default's
cloth cost. "fast" (ruling 2026-11-08 15:58Z) is for runs whose cloth shape is not scored: the same drape at DC 0.19
and 12 % stretch, at about 0.4x the default's cloth cost on an L4 and 0.25x on a CPU. The default stays 40; bench
headlines never use "fast". Health keeps reporting, but 12 % is under CLOTH_STRETCH's 50 % warning, so a fast drape
reads OK. ``ACCURACY`` and ``COST`` hold the measured numbers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CLOTH_ACCURACY = ("default", "super", "fast")
DEFAULT_EFFECT = "vbd:cloth_accuracy=default"
SUPER_EFFECT = "vbd:cloth_accuracy=super"
FAST_EFFECT = "vbd:cloth_accuracy=fast"
EFFECT: Mapping[str, str] = {"default": DEFAULT_EFFECT, "super": SUPER_EFFECT, "fast": FAST_EFFECT}
DEFAULT_ITERATIONS = 40
SUPER_ITERATIONS = 60
FAST_ITERATIONS = 10                  # VBD's own (the capability default)
DEFAULT_KWARGS: Mapping[str, Any] = {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9,
                                     "alloy_chebyshev_hold_self_contact": True, "iterations": DEFAULT_ITERATIONS}
SUPER_KWARGS: Mapping[str, Any] = {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9,
                                   "alloy_chebyshev_hold_self_contact": True, "iterations": SUPER_ITERATIONS}
FAST_KWARGS: Mapping[str, Any] = {"alloy_initial_guess": "inertial", "alloy_chebyshev_rho": None,
                                  "iterations": FAST_ITERATIONS}
KWARGS: Mapping[str, Mapping[str, Any]] = {"default": DEFAULT_KWARGS, "super": SUPER_KWARGS, "fast": FAST_KWARGS}
#: measured on a Cusick-style drape with self-contact (64 worlds, one L4, the 2-3 s window, 16 substeps, WP11's
#: self-contact rule, on this flip's own tree; converged reference: 200-400 iterations, drape coefficient
#: 0.739-0.741). The rows are in docs/notes/wp2-results.md (SCR-WP2-13).
ACCURACY: Mapping[str, str] = {
    "default": "a draped sheet stretches up to 0.36 % and hangs at drape coefficient 0.728 (converged 0.74)",
    "super": "a draped sheet stretches up to 0.34 % and hangs at drape coefficient 0.732 (converged 0.74)",
    "fast": "the cloth solver's own 10 iterations, far from converged: a draped sheet stretches up to 12 % and hangs "
            "at drape coefficient 0.19 (converged 0.74)",
}
#: the cloth solver's time, setting / default, on the same drape (L4 sim-s per wall s: default 1.888, super 1.308;
#: fast 5.01, main d7503c225 before the switch with the self-contact rule)
COST: Mapping[str, float] = {"default": 1.0, "super": 1.888 / 1.308, "fast": 1.888 / 5.01}
NOTE = ("super: cloth solved closer to convergence at about 1.4x the default's cloth cost; fast: the cloth solver's "
        "own 10 iterations at about 0.4x (CPU 0.25x), far from converged (a drape stretches 12 %)")


def checked(value: Any, error: type = ValueError) -> str:
    """``value`` when it is one of CLOTH_ACCURACY, else ``error`` naming the choices: ValueError, or the API's
    SceneError (message, fix, field="cloth_accuracy")."""
    if isinstance(value, str) and value in CLOTH_ACCURACY:
        return value
    msg = f"cloth_accuracy must be one of {list(CLOTH_ACCURACY)}, got {value!r}"
    if error is ValueError:
        raise ValueError(msg)
    raise error(msg, f"pass cloth_accuracy='default', 'super' or 'fast' ({NOTE})", field="cloth_accuracy")


def scene_property(error: type) -> property:
    """``Scene.cloth_accuracy``: "default", "super" or "fast", checked when set; ``error`` is the API's SceneError
    (its message, fix and field)."""

    def get(self) -> str:
        return self._cloth_accuracy

    def put(self, value: str) -> None:
        self._cloth_accuracy = checked(value, error)

    return property(get, put, doc=f'"default", "super" or "fast" ({NOTE}); set it before building. No cloth: no '
                                  'change.')


def tagged(effects: set[str]) -> str | None:
    """The setting a VBD entry's decisions carry (the most accurate wins: "super", then "default", then "fast"), or
    None without cloth."""
    for acc in ("super", "default", "fast"):
        if EFFECT[acc] in effects:
            return acc
    return None


def entry_kwargs(kw: Mapping[str, Any], effects: set[str]) -> dict[str, Any]:
    """The VBD entry's kwargs with KWARGS[setting] applied when a decision carries a cloth_accuracy effect. An
    iteration count already above the setting's (the "accurate" cloth profile's) is kept."""
    out = dict(kw)
    acc = tagged(effects)
    if acc is None:
        return out
    setting = KWARGS[acc]
    iterations = max(int(out.get("iterations", 0) or 0), int(setting["iterations"]))
    out.update(setting)
    out["iterations"] = iterations
    return out


__all__ = ["ACCURACY", "CLOTH_ACCURACY", "COST", "DEFAULT_EFFECT", "DEFAULT_ITERATIONS", "DEFAULT_KWARGS", "EFFECT",
           "FAST_EFFECT", "FAST_ITERATIONS", "FAST_KWARGS", "KWARGS", "NOTE", "SUPER_EFFECT", "SUPER_ITERATIONS",
           "SUPER_KWARGS", "checked", "entry_kwargs", "scene_property", "tagged"]
