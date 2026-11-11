# SPDX-License-Identifier: Apache-2.0
"""Health in plain words: per-world severity and flags from the engine's health monitor, each with a fix.

Example::

    h = sim.health()
    for w in h["invalid_worlds"]:
        print(w, h["worlds"][w]["problem"], "->", h["worlds"][w]["fixes"])
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from alloy.api import errors as _errors

SEVERITY = ("OK", "DEGRADED", "INVALID")
#: flags that inform but ask for no change (a run with only these is fine)
INFO_FLAGS = frozenset({"FALLBACK_USED", "DET_DOWNGRADED", "FAR_FROM_ORIGIN"})   # = the ladder's notices (its law)

#: health flag name -> (what it means, what to change)
REMEDIES: dict[str, tuple[str, str]] = {
    "NONFINITE_Q": ("a position became NaN/inf", "lower Scene(dt=...), check masses and stiffness are realistic, "
                    "and remove initial overlaps"),
    "NONFINITE_QD": ("a velocity became NaN/inf", "lower Scene(dt=...) or soften the contact (lower stiffness)"),
    "NONFINITE_PARTICLE": ("a cloth/liquid particle became NaN/inf", "lower Scene(dt=...) or use a finer "
                           "resolution; check the sheet is not created inside another object"),
    "CONTACT_OVERFLOW": ("more contacts than the buffer holds (results are not trustworthy)",
                         "pass alloy.build(scene, contact_capacity=n) with the n the console line names (MuJoCo's "
                         "contacts per world; default 8 per collision shape, at least 64), or use fewer touching "
                         "objects per world, or coarser meshes (collision='convex')"),
    "ROW_OVERFLOW": ("more constraint rows than the solver holds (with contact_capacity=n alone: this world holds "
                     "more than 2 n contacts)", "pass a larger alloy.build(scene, constraint_rows=m) if you set it, "
                     "else a larger contact_capacity=n (rows are sized for 2 n contacts plus the scene's joint "
                     "limits, equalities and friction loss), or use fewer touching objects per world"),
    "UNCONVERGED": ("the solver did not converge", "lower Scene(dt=...) or build with profile='precision'"),
    "PEN_WARN": ("objects overlap by more than 1 mm", "lower Scene(dt=...), or use profile='precision' for tight fits"),
    "PEN_FAIL": ("objects overlap by more than 1 cm (or half a thin part)", "remove initial overlaps (start objects "
                 "apart), lower Scene(dt=...), or thicken thin parts"),
    "PAIR_PEN_FAIL": ("a precision pair went past its clearance gate", "lower Scene(dt=...) or slow the insertion"),
    "PAIR_PEN_WARN": ("a precision pair is close to its gate", "lower Scene(dt=...) if this persists"),
    "JOINT_FAIL": ("a joint came apart", "lower joint gains / Scene(dt=...), or check joint limits in the robot file"),
    "ENERGY_JUMP": ("energy rose with nothing adding it (a jump in one step, or a steady gain)",
                    "remove initial overlaps and lower Scene(dt=...); a mesh object that never comes to rest: try "
                    "collision='convex' or 'decompose'"),
    "SPEED_FAIL": ("something moved faster than 20 m/s", "remove initial overlaps and huge forces; lower gains"),
    "FAR_FROM_ORIGIN": ("an object flew far from the origin", "check it is supported (ground?) and not launched"),
    "FALLBACK_USED": ("the engine used a fallback path", "nothing to change unless results look wrong"),
    "DET_DOWNGRADED": ("bitwise determinism was not available", "nothing to change unless you need repeat runs"),
    "LATCHED": ("the world was frozen at its last good state", "fix the first problem listed for this world"),
    "RUNG_PINNED": ("a solver asked for a smaller step than it was pinned to", "lower Scene(dt=...)"),
    "STRAIN_RANGE": ("a soft part was squeezed or stretched past the soft-body model's range (over 60 % strain); "
                     "motion past it is approximate", "press with an edge rather than a flat face, or use less force"),
    "START_PEN": ("objects overlapped at the start", "move them apart before the run (sim.settle does not undo an "
                  "overlap); a library object stands on pos by default, but with anchor='center' or 'origin' it can "
                  "reach into what is beside it: leave room or use anchor='base'"),
    "CLOTH_STRETCH": ("a cloth stretched far past its rest size (over 50 %; over 500 % the sheet blew up)",
                      "lower Scene(dt=...) or give the cloth more mass= or thickness=; keep its mesh edges (size / "
                      "(resolution - 1)) longer than its thickness, since shorter ones start the sheet in contact with "
                      "itself: lower resolution= or pass self_contact=False"),
    "SELF_CONTACT_OVERFLOW": ("a cloth or soft body touched itself or another one in more places than its contact "
                              "buffers hold; the extra contacts were dropped, so those surfaces can pass into each "
                              "other",
                              "stack fewer cloth layers (deep piles of folded sheets are the likely case); pass "
                              "self_contact=False only for a sheet that never folds and is the scene's only cloth or "
                              "soft body"),
    "FAST_CONTACT": ("a body closed on a static part by more than 5 mm in one physics step, and the part is thinner "
                     "than twice that travel: the body can pass through it (the flag clears after the approach)",
                     "step rigid bodies finer so a body moves under 5 mm per step: scene.rigid_step = 0.0003125 (a rigid-only "
                     "scene) or Scene(dt=0.0003); or make the static part thicker than twice that travel; or slow the "
                     "body (docs/known_gaps/rigid-tunnelling.md)"),
    "TUNNEL": ("a body passed through a static shape (the world is frozen before it)",
               "step rigid bodies finer (scene.rigid_step = 0.0003125 in a rigid-only scene, or Scene(dt=0.0003)), "
               "slow the body, or make the parts thicker; no travel per step is safe for every part (a pin of radius "
               "1.5 mm passed a 5 mm pane at 1.4 mm per step), so measure your own part at your speed and read "
               "sim.check() (docs/known_gaps/rigid-tunnelling.md)"),
    "SPIN": ("a body turned more than half a radian in one physics step, so contacts can add energy and fling it",
             "make the thin part heavier or thicker, or step rigid bodies finer: sim.routes() names the scene's rigid "
             "step; scene.rigid_step = 0.000625 pins a finer one, or Scene(dt=0.001) steps them at 1 ms"),
}


def _flag_names(bits: int) -> list[str]:
    from alloy.precision.types import HealthFlags  # noqa: PLC0415

    return [f.name for f in HealthFlags if bits & int(f)]


def _finite(value: Any) -> float | None:
    """``value`` as a float, or None when it is NaN or inf (JSON-ready rows)."""
    v = float(value)
    return v if math.isfinite(v) else None


def _world(report: Any, w: int) -> dict[str, Any]:
    sev = int(report.severity[w])
    flags = _flag_names(int(report.flags[w]))
    pen = float(report.max_pen_m[w])
    peak = getattr(report, "peak_pen_m", None)       # the deepest contact since the build or the last reset
    row: dict[str, Any] = {"severity": SEVERITY[min(sev, 2)], "flags": flags,
                           "max_penetration_m": pen if math.isfinite(pen) else None,
                           "peak_penetration_m": _finite(peak[w]) if peak is not None and math.isfinite(pen) else None,
                           "fixes": [REMEDIES[f][1] for f in flags if f in REMEDIES and f != "LATCHED"
                                     and f not in INFO_FLAGS]}
    if getattr(report, "seen_flags", None) is not None:      # every flag raised since the reset, gone or not
        row["seen_flags"] = _flag_names(int(report.seen_flags[w]) | int(report.flags[w]))
    cloth = _cloth_stretch(report, w)
    if cloth is not None:
        row["cloth_stretch"] = cloth
    gain = _energy_gain(report, w)
    if gain is not None:
        row["energy_gain"] = gain
    from alloy.precision.self_contact import CROSS_FIX, cross_row, cross_text  # noqa: PLC0415

    cross = cross_row(report, w)
    if cross is not None:
        row["self_contact"] = cross
    if sev >= 2:
        row["problem"] = report.failure_text(w) if cloth is None else f"{_what(report, w, 'CLOTH_STRETCH')}; " \
            f"{report.failure_text(w)}"
        if "TUNNEL" in flags and _tunnel_note(report, w, "TUNNEL"):
            row["problem"] = f"{_tunnel_note(report, w, 'TUNNEL')}; {row['problem']}"
        if cross is not None and cross["level"] == "fail":
            row["problem"] = f"{cross_text(cross)}; {row['problem']}"
            row["fixes"].insert(0, CROSS_FIX)
        row["first_bad_step"] = int(report.first_bad_step[w])
    elif set(flags) - INFO_FLAGS:                  # a notice alone is not a problem: the world is OK
        row["problem"] = "; ".join(_what(report, w, f) for f in flags if f in REMEDIES and f not in INFO_FLAGS)
    start = _start_overlap(report, w)
    if start is not None:
        row["start_overlap"] = start
    return row


def _name(names: tuple[str, ...], i: int, kind: str) -> str:
    return names[i] if 0 <= i < len(names) and names[i] else (f"{kind} {i}" if i >= 0 else "a static part")


def _start_overlap(report: Any, w: int) -> dict[str, Any] | None:
    """SCR-WP5-14: the deepest start overlap of world ``w`` past pen_warn (None: none)."""
    level = getattr(report, "start_level", None)
    if level is None or int(level[w]) == 0:
        return None
    a, b = (int(x) for x in report.start_shapes[w])
    names = report.shape_names
    out = {"depth_mm": round(float(report.start_pen_m[w]) * 1e3, 2), "a": _name(names, a, "shape"),
           "b": _name(names, b, "shape"), "level": "fail" if int(level[w]) == 2 else "warn"}
    others = [p for p in (getattr(report, "start_pairs", None) or {}).get(w, []) if {p[0], p[1]} != {a, b}]
    if others:                                    # WP13: fixing one pair at a time hides the rest
        out["more"] = [{"a": _name(names, pa, "shape"), "b": _name(names, pb, "shape"), "depth_mm": round(d * 1e3, 2)}
                       for pa, pb, d in others[:2]]
        out["more_count"] = len(others)
    return out


def _overlap_text(s: dict[str, Any]) -> str:
    """``"a and b overlapped by 44.84 mm at the start; also c and d (35.1 mm) and 2 more pairs over 1 cm"``."""
    text = f"{s['a']} and {s['b']} overlapped by {s['depth_mm']} mm at the start"
    more = s.get("more", [])
    if more:
        text += "; also " + ", ".join(f"{m['a']} and {m['b']} ({m['depth_mm']} mm)" for m in more)
        rest = s["more_count"] - len(more)
        text += f" and {rest} more pair{'s' if rest > 1 else ''} over 1 cm" if rest else ""
    return text


def _cloth_stretch(report: Any, w: int) -> dict[str, Any] | None:
    """SCR-WP5-17: a health row's ``cloth_stretch``: world ``w``'s largest cloth stretch over the run (its peak, not
    the latest check), present only past cloth_stretch_warn (None: none)."""
    s = getattr(report, "cloth_stretch", None)
    if s is None or not float(s[w]) > report.cloth_stretch_warn:
        return None
    names = getattr(report, "cloth_names", ()) or ()
    return {"percent": round(float(s[w]) * 100.0, 1), "cloth": names[w] if w < len(names) and names[w] else "a cloth",
            "level": "fail" if float(s[w]) > report.cloth_stretch_fail else "warn"}


def _energy_gain(report: Any, w: int) -> dict[str, Any] | None:
    """PHYSA-1: a health row's ``energy_gain``: the most energy world ``w`` gained over its lowest earlier level with
    nothing adding it, in % of its energy scale, present only past the limit (None: none, or energy not checked)."""
    g = getattr(report, "energy_gain", None)
    if g is None or not float(g[w]) > report.energy_gain_limit:
        return None
    return {"percent": round(float(g[w]) * 100.0, 1), "limit_percent": round(report.energy_gain_limit * 100.0, 1)}


def _gain_text(g: dict[str, Any]) -> str:
    return (f"the objects gained energy with nothing adding it: {g['percent']:,.0f} % of the scene's energy scale "
            f"(limit {g['limit_percent']:g} %)")


def _what(report: Any, w: int, flag: str) -> str:
    """A flag's problem text, with the bodies and numbers where health keeps them (START_PEN, SPIN)."""
    gain = _energy_gain(report, w) if flag == "ENERGY_JUMP" else None
    if gain is not None:
        return _gain_text(gain)
    start = _start_overlap(report, w) if flag == "START_PEN" else None
    if start is not None:
        return _overlap_text(start)
    cloth = _cloth_stretch(report, w) if flag == "CLOTH_STRETCH" else None
    if cloth is not None:
        limit = report.cloth_stretch_fail if cloth["level"] == "fail" else report.cloth_stretch_warn
        return (f"{cloth['cloth']} stretched {cloth['percent']:,.0f} % past its rest size (limit {limit * 100:g} %)"
                + (": it blew up" if cloth["level"] == "fail" else ""))
    note = _tunnel_note(report, w, flag)
    if note:
        return note
    if flag == "SPIN" and getattr(report, "spin_body", None) is not None and int(report.spin_body[w]) >= 0:
        body = _name(report.body_names, int(report.spin_body[w]), "body")
        return (f"{body} turned {float(report.spin_rad[w]):.2f} rad in one physics step (limit "
                f"{report.spin_limit:g} rad), so contacts can add energy and fling it")
    return REMEDIES[flag][0]


def _tunnel_note(report: Any, w: int, flag: str) -> str:
    """GAP-E2: the named text of FAST_CONTACT / TUNNEL ('' where the API has no names)."""
    texts = getattr(report, "tunnel_text" if flag == "TUNNEL" else "fast_text", ()) if flag in (
        "TUNNEL", "FAST_CONTACT") else ()
    return texts[w] if w < len(texts) else ""


def _with_tunnel_texts(report: Any, engine: Any) -> Any:
    """``report`` with the per-world texts of FAST_CONTACT / TUNNEL (``health_tunnel.TunnelWatch.notes``)."""
    from alloy.precision.health_tunnel import tunnel_texts  # noqa: PLC0415

    return tunnel_texts(report, engine)


def tunnel_problems(h: dict[str, Any]) -> list[dict[str, Any]]:
    """``sim.check()`` problems for worlds with TUNNEL (GAP-E2): each makes the check not ok, since
    ``alloy.check`` does not know the flag. Example: ``tunnel_problems(sim.health())`` -> ``[{"world": 3, "check":
    "tunnel", "what": "\"ball\" passed through \"plate\"", ...}]``."""
    out = []
    for w, row in enumerate(h.get("worlds", [])):
        if "TUNNEL" in row.get("flags", []):
            out.append({"world": w, "level": "fail", "check": "tunnel", "severity": row["severity"],
                        "what": row.get("problem") or REMEDIES["TUNNEL"][0], "fix": REMEDIES["TUNNEL"][1]})
    return out


def start_problems(h: dict[str, Any]) -> list[dict[str, Any]]:
    """``sim.check()`` problems for start overlaps past pen_fail (SCR-WP5-14): each makes the check not ok.

    Example: ``start_problems(sim.health())`` -> ``[{"world": 0, "level": "fail", "check": "start_overlap",
    "shapes": ["block", "bin_wall0"], "what": "block and bin_wall0 overlapped by 17.3 mm at the start", ...}]``."""
    out = []
    for w, row in enumerate(h.get("worlds", [])):
        s = row.get("start_overlap")
        if s is not None and s["level"] == "fail":
            out.append({"world": w, "level": "fail", "check": "start_overlap", "severity": row["severity"],
                        "shapes": [s["a"], s["b"]],
                        "what": _overlap_text(s),
                        "fix": REMEDIES["START_PEN"][1]})
    return out


def cloth_problems(h: dict[str, Any], listed: list[dict[str, Any]] = ()) -> list[dict[str, Any]]:
    """``sim.check()`` problems for cloths stretched past cloth_stretch_fail (SCR-WP5-17), each making the check not
    ok, except worlds a problem in ``listed`` already names a stretch for. WP2's blown drape read ok in check_sim.

    Example: ``cloth_problems(sim.health())`` -> ``[{"world": 0, "check": "cloth_stretch", "level": "fail",
    "what": "sheet stretched 525 % past its rest size: it blew up", ...}]``."""
    out = []
    for w, row in enumerate(h.get("worlds", [])):
        c = row.get("cloth_stretch")
        if c is None or c["level"] != "fail" or any(
                p.get("world") == w and "stretched" in str(p.get("what", "")) for p in listed):
            continue
        out.append({"world": w, "level": "fail", "check": "cloth_stretch", "severity": row["severity"],
                    "what": f"{c['cloth']} stretched {c['percent']:,.0f} % past its rest size: it blew up",
                    "fix": REMEDIES["CLOTH_STRETCH"][1]})
    return out


def gain_problems(h: dict[str, Any]) -> list[dict[str, Any]]:
    """``sim.check()`` problems for worlds whose energy rose past the gain limit with nothing adding it (PHYSA-1):
    each makes the check not ok (INVALID worlds are problems already). A free mesh body that pumped itself 77 mm up
    read ok before.

    Example: ``gain_problems(sim.health())`` -> ``[{"world": 0, "check": "energy_gain", "level": "fail",
    "what": "the objects gained energy with nothing adding it: 44 % of the scene's energy scale (limit 10 %)", ...}]``
    """
    return [{"world": w, "level": "fail", "check": "energy_gain", "severity": row["severity"],
             "what": _gain_text(row["energy_gain"]), "fix": REMEDIES["ENERGY_JUMP"][1]}
            for w, row in enumerate(h.get("worlds", []))
            if row.get("energy_gain") is not None and row["severity"] != "INVALID"]


def _with_starts(problems: list[dict[str, Any]], fails: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each start overlap that no problem of its world already names as one (first: it is the cause), then
    ``problems``; a penetration problem between the same two shapes points at the start overlap (WP13: its
    stiffness advice sends users the wrong way when the overlap was placed)."""
    def names(p: dict[str, Any], f: dict[str, Any]) -> bool:
        return all(n in str(p.get("what", "")) for n in f["shapes"])

    def named(p: dict[str, Any], f: dict[str, Any]) -> bool:
        return p.get("world") == f["world"] and "start" in str(p.get("what", "")) and names(p, f)

    out = [f for f in fails if not any(named(p, f) for p in problems)]
    for p in problems:
        cause = next((f for f in fails if p.get("check") == "penetration" and names(p, f)), None)
        out.append(p if cause is None else {**p, "fix": "caused by the start overlap of these two shapes: "
                                            + REMEDIES["START_PEN"][1]})
    return out


def poll(engine: Any) -> Any:
    """The engine's ``HealthReport`` without raising (profile "precision" raises on INVALID worlds when stepping;
    reading health afterwards must still work), or None without a monitor. Example: ``poll(sim.engine)``."""
    monitor = getattr(engine, "health", None)
    if monitor is None:
        return None
    from alloy.core.errors import PrecisionError  # noqa: PLC0415

    try:
        report = monitor.poll()
    except PrecisionError as exc:
        report = getattr(exc, "report", None)
    return _with_tunnel_texts(_with_cloth_names(report, engine), engine)


def _with_cloth_names(report: Any, engine: Any) -> Any:
    """``report`` with ``cloth_names``: each world's most stretched cloth, by the name it was added with."""
    particle = getattr(report, "cloth_particle", None)
    built = getattr(engine, "built", None)
    if particle is None or built is None:
        return report
    import dataclasses  # noqa: PLC0415

    from alloy.api.names import NameIndex  # noqa: PLC0415

    idx = getattr(engine, "_health_name_index", None)
    if idx is None:                              # built once per engine (OSS-3)
        idx = NameIndex.from_built(built)
        try:
            engine._health_name_index = idx
        except AttributeError:
            pass
    stride = max(int(idx.stride.get("particle", 0)), 1)

    def name(w: int, p: int) -> str:
        t = p - w * stride if p >= w * stride else p
        return next((n for n, r in idx.particles.items() if t in r), "") if p >= 0 else ""
    return dataclasses.replace(report, cloth_names=tuple(name(w, int(p)) for w, p in enumerate(particle)))


def retired(engine: Any) -> list[int] | None:
    """Liquid particles each world has retired (left the SPH escape box; frozen where they left, no longer
    simulated), or None without a liquid solver. Example: ``retired(sim.engine)`` -> ``[0, 3]``."""
    from alloy.api._liquid import liquid_solvers  # noqa: PLC0415

    counts = None
    for solver in liquid_solvers(engine):
        escaped = [int(x) for x in getattr(solver.report(), "escaped", [])]
        counts = escaped if counts is None else [a + b for a, b in zip(counts, escaped)]
    return counts


def add_retired(result: dict[str, Any], counts: list[int] | None) -> dict[str, Any]:
    """Put ``retired_particles`` (per world) into a health / check result, and a warning per world that has any."""
    result["retired_particles"] = counts
    for w, n in enumerate(counts or []):
        if n and "problems" in result:
            result["problems"].append({"world": w, "level": "warn", "check": "retired", "severity": "INFO",
                                       "what": f"{n} liquid particles left the scene and were retired (frozen where "
                                               "they left, no longer simulated; '<liquid>.active' is False for them)",
                                       "fix": "contain the liquid: walls high enough, a pour aimed into the "
                                              "container (Scene.add_pour), no block dropped from above"})
    return result


def refines(engine: Any) -> list[int] | None:
    """Per world: control periods the coupling re-ran at a finer exchange step for that world's escalation, that
    period only (``alloy.coupling.ladder``), since the build (``sim.reset()`` does not zero them), or None without a
    coupled stepper. Example: ``refines(sim.engine)``."""
    fn = getattr(getattr(engine, "stepper", None), "period_refines", None)
    return [int(x) for x in fn()] if callable(fn) else None


def add_refines(result: dict[str, Any], counts: list[int] | None) -> dict[str, Any]:
    """Put ``refined_periods`` (per world, ``refines``) into a health result. A refined period re-ran that world's
    period at a finer step; the other worlds kept their first run. Informational: not a problem."""
    result["refined_periods"] = counts
    return result


def summary(report: Any, worlds: int) -> dict[str, Any]:
    """JSON-ready health of every world from a ``HealthReport`` (None: no monitor, nothing measured).

    Example: ``summary(sim.engine.report()["health"], sim.worlds)["ok"]``."""
    if report is None:
        return {"ok": True, "monitored": False, "invalid_worlds": [], "degraded_worlds": [], "worlds": []}
    rows = [_world(report, w) for w in range(worlds)]
    invalid = [w for w, r in enumerate(rows) if r["severity"] == "INVALID"]
    degraded = [w for w, r in enumerate(rows) if r["severity"] == "DEGRADED"]
    return {"ok": not invalid, "monitored": True, "invalid_worlds": invalid, "degraded_worlds": degraded,
            "worlds": rows}


def warn_downgraded(engine: Any, profile: str) -> bool:
    """Profile "precision" was chosen and runs will not repeat bit for bit here (the effective determinism is below
    the requested one: every GPU run with rigid contact, robots or cloth): warn once at the build, with the reason
    and the page. Other profiles get the DET_DOWNGRADED notice in ``sim.health()`` only (ruling 2026-11-10 03:22Z).
    Once per ``Sim``. Returns True when it warned. Example: ``warn_downgraded(sim.engine, "precision")``."""
    monitor = getattr(engine, "health", None)
    asked = getattr(getattr(monitor, "config", None), "determinism", None)
    got = getattr(monitor, "det_mode", None)
    if profile != "precision" or asked is None or got is None or int(got) >= min(int(asked), 1):   # 1: RUN_TO_RUN
        return False
    from alloy.api._say import warn  # noqa: PLC0415

    reasons = "; ".join(getattr(monitor, "det_reasons", ())) or "no reason given"
    warn('profile="precision": runs on this device will not repeat bit for bit (%s). Results are checked as usual; '
         'for the same bits run with device="cpu". See docs/known_gaps/determinism-gpu-rigid.md', reasons)
    return True


def seen_warnings(h: dict[str, Any]) -> list[dict[str, Any]]:
    """One warning per flag a world raised earlier in the run that its latest check no longer shows (a contact that
    was too deep for a few steps, a solver that did not converge): ``sim.check()`` lists them, ``ok`` is unchanged.
    A flag raised in a pass the engine then re-ran (a coupled period repeated at a finer step) counts too.
    Example: ``[{"world": 0, "what": "PEN_WARN earlier in this run: ...", "fix": ..., "level": "warn"}]``."""
    out = []
    for w, row in enumerate(h["worlds"]):
        gone = [f for f in row.get("seen_flags", ()) if f not in row["flags"] and f in REMEDIES
                and f not in INFO_FLAGS and f != "LATCHED"]
        out += [{"world": w, "what": f"{f} earlier in this run: {REMEDIES[f][0]}", "fix": REMEDIES[f][1],
                 "check": "health", "level": "warn", "objects": [], "flag": f} for f in gone]
    return out


def check(sim: Any) -> dict[str, Any]:
    """``sim.check()``: ``alloy.check.check_sim(sim)`` when that package exists, else health-based problems.

    Example: ``{"ok": False, "source": "health", "problems": [{"world": 3, "what": ..., "fix": ...}]}``."""
    try:
        from alloy.check import check_sim  # noqa: PLC0415 - optional package (physics-check tool)
    except ImportError:
        check_sim = None
    if check_sim is not None:
        with _errors.reading(sim.engine):                    # profile "precision": check reads, it does not raise
            checked = check_sim(sim)
        result = add_retired(checked, retired(sim.engine))
        h = sim.health()
        listed = result.get("problems", [])
        extra, blown, gains = start_problems(h), cloth_problems(h, listed), gain_problems(h)
        from alloy.precision.self_contact import cross_problems  # noqa: PLC0415

        blown = blown + tunnel_problems(h) + cross_problems(h, listed)
        if extra or blown or gains:
            listed = [p for p in listed                               # the gain replaces alloy.check's energy warning
                      if not (gains and p.get("check") == "energy" and p.get("level") == "warn")]
            result["problems"] = _with_starts(listed, extra) + blown + gains
            result["ok"] = False
        result["problems"] = result.get("problems", []) + seen_warnings(h)
        return result
    h = sim.health()
    problems, gains = [], gain_problems(h)
    gained = {g["world"] for g in gains}
    for w, row in enumerate(h["worlds"]):
        if row["severity"] == "OK" or (row["severity"] == "DEGRADED" and set(row["flags"]) <= INFO_FLAGS):
            continue
        if w in gained and set(row["flags"]) <= INFO_FLAGS | {"ENERGY_JUMP"}:
            continue                              # its gain row (check and level keys, as on the other path) says it
        problems.append({"world": w, "severity": row["severity"], "what": row.get("problem", ""),
                         "fix": "; ".join(row["fixes"]) or "see the flags"})
    problems += gains + seen_warnings(h)
    fails = start_problems(h)
    problems += [p for p in tunnel_problems(h) if not any(q["world"] == p["world"] for q in problems)]
    ok = h["ok"] and not fails and not gains
    return add_retired({"ok": ok, "source": "health", "monitored": h["monitored"],
                        "problems": _with_starts(problems, fails),
                        "invalid_worlds": h["invalid_worlds"]}, h.get("retired_particles"))
