# SPDX-License-Identifier: Apache-2.0
"""One build-time line on CPU speed (GEO P6, 2026-11-11): a towel ran about 20x slower than real time on a Mac CPU
and nothing said to expect it. ``alloy.build`` ends with ``after_build``; ``after_build(sim, None)`` returns ``sim``."""

from __future__ import annotations

from typing import Any

# Routes that step many times slower than real time on a CPU (GEO P6, 2026-11-11, physics' condition): cloth, liquid,
# sand and paste (MPM), soft bodies. Rigid bodies and robots run about real time there, so they get no line.
_CPU_SLOW_WORDS = (("cloth", "cloth"), ("liquid", "liquid"), ("soft", "soft bodies"), ("mpm", "sand or paste"))
_noted = [False]                 # one line per process


def cpu_note(sim: Any, device: str | None) -> bool:
    """When no GPU was found (``device`` not given, the build on the CPU) and the scene holds cloth, liquid, sand or
    paste, or soft bodies: one line per process saying they run many times slower than real time here and how to
    estimate the wall time. Never raises; True when the line was written. Example: ``cpu_note(sim, None)``."""
    if _noted[0] or device is not None or getattr(sim, "device", None) != "cpu":
        return False
    try:
        routes = [str(row.get("route", "")) for row in sim.routes().values()]
    except Exception:  # noqa: BLE001 - a hint must never stop a build
        return False
    kinds = [next(word for key, word in _CPU_SLOW_WORDS if key in r) for r in routes
             if any(key in r for key, _ in _CPU_SLOW_WORDS)]
    if not kinds:
        return False
    _noted[0] = True
    from alloy.api._say import warn  # noqa: PLC0415

    kinds = list(dict.fromkeys(kinds))
    named = kinds[0] if len(kinds) == 1 else ", ".join(kinds[:-1]) + " and " + kinds[-1]
    warn("no NVIDIA GPU found: this scene runs on the CPU, where scenes with %s run many times slower than real "
         "time (a small cloth about 20x slower on a laptop); alloy.estimate(scene, worlds, seconds) predicts the wall "
         "time, and build(..., device='cpu') keeps this line quiet", named)
    return True


def after_build(sim: Any, device: str | None) -> Any:
    """``cpu_note(sim, device)``, then ``sim``: the last step of ``alloy.build``."""
    cpu_note(sim, device)
    return sim
