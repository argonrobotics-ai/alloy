# SPDX-License-Identifier: Apache-2.0
"""Health events, maximum penetration and energy drift of the timed run, as findings.

- ``health``: worlds the health layer made INVALID (non-finite state, capacity overflow, deep penetration, speed,
  far from the origin, joint violation), grouped by what tripped them first, with the objects involved; and a run that
  stopped with an exception.
- ``penetration``: the deepest contact over the run (health record after every period, plus the bench monitors'
  per-step maximum on Newton contacts), attributed to an object pair by a geometric re-collide, with the depth at the
  end of the run. ``sim.health()`` reports that end depth (its latest check), so a landing that sinks 3 mm for a moment
  and rests at 0.01 mm reads 3 mm here and 0.01 mm there; the finding gives both.
- ``energy``: unexplained energy gain of the passive objects over E_scale (bench monitors, D20).

Limits are the severity table's (SPEC §4.3, ``alloy.core.defaults``).

Example::

    >>> found, metrics = health_findings(res, index)        # doctest: +SKIP
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alloy.check.advice import mm, worlds_text
from alloy.check.contacts import GROUND
from alloy.check.objects import ObjectIndex
from alloy.check.run import RunResult
from alloy.check.report import Finding
from alloy.core.defaults import ENERGY_JUMP_REL, ENERGY_S1_REL, PEN_LIMIT_M, PEN_S1_M

log = logging.getLogger("alloy.check")

PAIR_PEN_LIMIT_M = 5e-4
ENERGY_WARN_REL = 0.05       # cumulative unexplained gain that warns (S1 errors at ENERGY_S1_REL)

_FIX = {
    "NONFINITE": ("the state became NaN or infinite (the solve diverged)",
                  "fix overlaps at load first; then look for very stiff contacts or very light parts next to heavy "
                  "ones (mass ratio > 1000), or rerun with profile='precision'"),
    "OVERFLOW": ("the contact or constraint capacity overflowed, so contacts were dropped",
                 "build with alloy.build(scene, contact_capacity=n) and check that simulation with sim.check() "
                 "(alloy.check(scene) runs the default capacity), or simplify collision shapes (fewer convex parts "
                 "or triangles) or reduce clutter per world"),
    "PEN_FAIL": ("a contact went deeper than 10 mm or half the thinner part",
                 "stiffen the contact (contact_stiffness of the materials or their pair card), thicken the thin "
                 "part, or remove the start overlap"),
    "PAIR_PEN_FAIL": ("a precision pair went past its clearance",
                      "check the precision feature's clearance and loads.contact_speed on both cards"),
    "SPEED_FAIL": ("a body went faster than 20 m/s (it was launched)",
                   "fix overlaps at load first; then lower contact_stiffness or control_dt"),
    "FAR_FROM_ORIGIN": ("a body left the workspace (fell off or was launched)",
                        "add a floor or a support under it, and fix overlaps at load"),
    "JOINT_FAIL": ("a joint or weld pulled apart past its limit",
                   "check attachment anchors and that the load on the joint is within its design force"),
    "ENERGY_JUMP": ("energy jumped in one step",
                    "fix overlaps at load first; then lower contact_stiffness or control_dt"),
}
_GROUPS = (("NONFINITE", ("NONFINITE_Q", "NONFINITE_QD", "NONFINITE_PARTICLE")),
           ("OVERFLOW", ("CONTACT_OVERFLOW", "ROW_OVERFLOW")), ("PEN_FAIL", ("PEN_FAIL",)),
           ("PAIR_PEN_FAIL", ("PAIR_PEN_FAIL",)), ("SPEED_FAIL", ("SPEED_FAIL",)),
           ("FAR_FROM_ORIGIN", ("FAR_FROM_ORIGIN",)), ("JOINT_FAIL", ("JOINT_FAIL",)),
           ("ENERGY_JUMP", ("ENERGY_JUMP",)))


def health_findings(res: RunResult, index: ObjectIndex) -> tuple[list[Finding], dict[str, Any]]:
    """Findings for INVALID worlds (grouped by first-failure cause) and for a run that stopped early.

    Example::

        >>> health_findings(RunResult(), index)
        ([], {'invalid_worlds': None})
    """
    out: list[Finding] = []
    if res.graph_error is not None:
        out.append(Finding("health", "info", f"the run could not be captured into a CUDA graph ({res.graph_error}); "
                           "it ran again without graphs, which is slower",
                           "nothing to change in the scene: the log's traceback names the part that reads the host "
                           "inside a step, for its owner to fix"))
    if res.error is not None:
        out.append(Finding("health", "error", f"the simulation stopped at t={res.error_t:.3g} s: {res.error}",
                           _run_error_fix(res.error), value=res.error_t, unit="s"))
    h = res.health
    if h is None:
        return out, {"invalid_worlds": None}
    invalid = list(h.invalid_worlds)
    ff = getattr(h, "first_failure", None)
    for key, flags in _GROUPS:
        worlds = [w for w in invalid if _first_flags(h, ff, w) & set(flags)]
        if worlds:
            out.append(_invalid_finding(key, worlds, h, ff, index))
    degraded = _degraded_flags(h)
    if "UNCONVERGED" in degraded:
        ws = degraded["UNCONVERGED"]
        out.append(Finding("health", "info", f"the solver hit its iteration cap in some steps "
                           f"{worlds_text(ws, index.world_count)}", "usually harmless at rest; if results matter, "
                           "rerun with profile='precision'", worlds=tuple(ws)))
    metrics = {"invalid_worlds": invalid, "degraded_flags": {k: len(v) for k, v in degraded.items()}}
    return out, metrics


OOM_WORDS = ("failed to allocate", "out of memory", "cudaerrormemoryallocation")


def _run_error_fix(error: str) -> str:
    if any(w in error.lower() for w in OOM_WORDS):
        return ("pass fewer worlds (worlds= / CLI --worlds; a quarter of these) or free GPU memory: the run needed "
                "more device memory than was free")
    return "fix overlaps at load and card errors first; if it persists, report the scene with this message"


def _first_flags(h: Any, ff: Any, w: int) -> set[str]:
    from alloy.precision.types import HealthFlags  # noqa: PLC0415

    bits = int(ff.flags[w]) if ff is not None and int(ff.flags[w]) else int(h.flags[w])
    return {f.name for f in HealthFlags if bits & int(f)}


def _degraded_flags(h: Any) -> dict[str, list[int]]:
    from alloy.precision.types import HealthFlags  # noqa: PLC0415

    out: dict[str, list[int]] = {}
    for w, bits in enumerate(np.asarray(h.flags, np.int64)):
        for f in HealthFlags:
            if int(bits) & int(f):
                out.setdefault(f.name, []).append(w)
    return out


def _invalid_finding(key: str, worlds: list[int], h: Any, ff: Any, index: ObjectIndex) -> Finding:
    what, fix = _FIX[key]
    w = worlds[0]
    info = ff.world(w) if ff is not None else None
    objs = _failure_objects(info, index)
    when = f" at t={info['time_s']:.3g} s" if info and np.isfinite(info["time_s"]) else ""
    who = f" ({', '.join(objs)})" if objs else ""
    detail = _detail(key, info)
    msg = f"{len(worlds)} world(s) failed: {what}{who}, first in world {w}{when}{detail}"
    return Finding("health", "error", msg, fix, objects=tuple(objs), worlds=tuple(worlds))


def _failure_objects(info: dict | None, index: ObjectIndex) -> list[str]:
    if not info:
        return []
    names = [index.name_of("shape", info["shape_a"]), index.name_of("shape", info["shape_b"]),
             index.name_of("particle", info["particle"]), index.name_of("body", info["body"])]
    out: list[str] = []
    for n in names:
        if n and n != GROUND and n not in out:
            out.append(n)
    return out


def _detail(key: str, info: dict | None) -> str:
    if not info:
        return ""
    if key == "PEN_FAIL" and np.isfinite(info["pen_m"]):
        return f": depth {mm(info['pen_m'])}"
    if key == "SPEED_FAIL":
        return f": {info['speed_mps']:.3g} m/s"
    if key == "PAIR_PEN_FAIL":
        return f": depth {mm(info['pair_pen_m'])}"
    if key == "JOINT_FAIL":
        return f": {mm(info['joint_err_m'])}, {info['joint_err_rad']:.3g} rad"
    return ""


def penetration_findings(res: RunResult, index: ObjectIndex) -> tuple[list[Finding], dict[str, Any]]:
    """The deepest contact of the run against PEN_LIMIT_M (warn) and PEN_S1_M (error); None when unmeasured.
    ``end_pen_m`` is that world's depth at the end of the run, the value ``sim.health()`` then reports.

    Example::

        >>> penetration_findings(RunResult(), index)
        ([], {'max_pen_m': None})
    """
    pen = _pen_per_world(res)
    if pen is None:
        return [], {"max_pen_m": None}
    w = int(np.nanargmax(pen))
    d = float(pen[w])
    metrics = {"max_pen_m": d, "world": w, "end_pen_m": _pen_end(res, w),
               "max_pair_pen_m": float(np.max(res.pair_pen_max))}
    out = []
    if d > PEN_LIMIT_M:
        out.append(_pen_finding(res, index, pen, w, d))
    pp = float(np.max(res.pair_pen_max)) if res.pair_pen_max is not None else 0.0
    if pp > PAIR_PEN_LIMIT_M:
        out.append(Finding("penetration", "warn",
                           f"a precision pair penetrates {mm(pp)} (limit {mm(PAIR_PEN_LIMIT_M)})",
                           "check the precision features' clearance and loads.contact_speed on both cards",
                           value=pp, limit=PAIR_PEN_LIMIT_M, unit="m"))
    return out, metrics


def _pen_per_world(res: RunResult) -> np.ndarray | None:
    pen = res.pen_max.copy() if res.pen_max is not None else None
    mon = (res.monitors or {}).get("max_pen")
    if mon is not None:
        mon = np.where(np.asarray(mon) > 0, np.asarray(mon, np.float64), np.nan)
        pen = mon if pen is None else np.fmax(pen, mon)
    if pen is None or not np.isfinite(pen).any():
        return None
    return pen


def _pen_finding(res: RunResult, index: ObjectIndex, pen: np.ndarray, w: int, d: float) -> Finding:
    level = "error" if d > PEN_S1_M else "warn"
    over = [int(k) for k in np.flatnonzero(pen > PEN_LIMIT_M)]
    t = float(res.pen_t[w]) if res.pen_t is not None and np.isfinite(res.pen_t[w]) else None
    pair = max(res.attributions, key=lambda a: a[1].depth)[1] if res.attributions else None
    when = f" at t={t:.3g} s" if t is not None else ""
    where = f"world {w}; {worlds_text(over, index.world_count) or 'one world'} past {mm(PEN_LIMIT_M)}"
    worlds = tuple(over) if len(over) < index.world_count else ()
    if pair is not None and pair.a == pair.b:
        return Finding("penetration", "info", f"deepest contact {mm(d)}{when} is inside {pair.a}, between its own "
                       f"parts ({where})", f"nothing, unless {pair.a}'s own parts should not touch: its collision "
                       f"filters (robot file or card) decide", objects=(pair.a,), value=d, limit=PEN_LIMIT_M,
                       unit="m", worlds=worlds)
    who = f" between {pair.a} and {pair.b}" if pair is not None else ""
    objs = tuple(n for n in (pair.a, pair.b) if n != GROUND) if pair is not None else ()
    end = _pen_end(res, w)
    msg = f"deepest contact {mm(d)}{who}{when} ({where})"
    if end is not None:
        msg += f"; {mm(end)} at the end of the run, the depth sim.health() reports"
    stiffen = f"stiffen the contact{who}: raise contact_stiffness of their materials or add a pair card"
    if end is not None and end <= PEN_LIMIT_M:     # deep for a moment only: a landing or a push
        what = next((n for n in objs if index.motion.get(n) != "static"), "the object")
        fix = (f"it sank for a moment (a landing or a push), not at rest: start {what} on its support (the "
               f"rest_drift finding gives the settled pose) or drop it from lower, or {stiffen}")
    else:
        fix = (f"{stiffen}, thicken a thin part, or lower the load (a resting contact should stay under "
               f"{mm(PEN_LIMIT_M)})")
    return Finding("penetration", level, msg, fix, objects=objs, value=d,
                   limit=PEN_S1_M if level == "error" else PEN_LIMIT_M, unit="m", worlds=worlds)


def _pen_end(res: RunResult, w: int) -> float | None:
    end = res.pen_end[w] if res.pen_end is not None and w < len(res.pen_end) else np.nan
    return float(end) if np.isfinite(end) else None


def energy_findings(res: RunResult, index: ObjectIndex) -> tuple[list[Finding], dict[str, Any]]:
    """Unexplained energy gain over the run (fraction of E_scale): cumulative > ENERGY_S1_REL (50 %) errors, > 5 %
    warns; a single-step jump over ENERGY_JUMP_REL (1 %, e.g. a touchdown on a tiny object) is info.

    Example::

        >>> energy_findings(RunResult(), index)
        ([], {})
    """
    m = res.monitors
    if not m or "e_gain_cum" not in m:
        return [], {}
    cum, step = np.asarray(m["e_gain_cum"]), np.asarray(m["e_gain"])
    w = int(np.argmax(cum))
    metrics = {"gain_cum_rel": float(cum.max()), "gain_step_rel": float(step.max()),
               "e_scale_j": float(res.e_scale[w]) if res.e_scale is not None else None}
    fix = ("something adds energy: fix overlaps at load first (they release stored energy), then look for contacts "
           "too stiff for the time step or a joint fighting a collision; rerun with profile='precision' to compare")
    if cum[w] > ENERGY_WARN_REL:
        level, limit = ("error", ENERGY_S1_REL) if cum[w] > ENERGY_S1_REL else ("warn", ENERGY_WARN_REL)
        return [Finding("energy", level, f"the passive objects gained {cum[w] * 100:.0f} % of the scene's energy "
                        f"scale without a cause over the run (worst world {w})", fix, value=float(cum[w]),
                        limit=limit, unit="E_scale")], metrics
    ws = int(np.argmax(step))
    if step[ws] > ENERGY_JUMP_REL:
        return [Finding("energy", "info", f"energy jumped by {step[ws] * 100:.1f} % of the scene's energy scale in "
                        f"one step (worst world {ws}), with no lasting gain", "nothing, unless it repeats in longer "
                        "runs; " + fix, value=float(step[ws]), limit=ENERGY_JUMP_REL, unit="E_scale")], metrics
    return [], metrics


__all__ = ["energy_findings", "health_findings", "penetration_findings"]
