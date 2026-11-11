# SPDX-License-Identifier: Apache-2.0
"""Checks only soft items need, on a short run of the composed scene: the liquid or grains stay in their container
(and a pour ends up in it), the amount simulated matches the amount asked, piles keep to the table, cloth does not
overstretch, cables keep their length, and everything stays in the workspace and settles.

They run inside ``alloy.check`` as an extra run check (``soft_check``), on the same run as the standard checks
(cards, overlap, rest drift, penetration, health, energy): ``report.checks["soft"]`` holds the verdict and
``report.metrics["soft"]`` the numbers per soft item.

Example::

    >>> report = alloy.check.check(card, extra=[soft_check(watches, names, table)])   # doctest: +SKIP
    >>> report.metrics["soft"]["scene_water_1"]["in_container"]                      # doctest: +SKIP
    1.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.compose import geom

log = logging.getLogger("alloy.compose")

IN_WARN, IN_ERROR = 0.98, 0.90          # share of a fill or pour inside its container at the end
AMOUNT_WARN = 0.15                      # |simulated / asked - 1| above this: warn
STRETCH_WARN, STRETCH_ERROR = 0.05, 0.20
LENGTH_WARN, LENGTH_ERROR = 0.02, 0.10
LATE_WARN_M = 5e-3                      # m any particle or link moves over the last half: still moving
WALL_TOL = 3e-3                         # m around a container's cavity that still counts as inside
WORK_XY, WORK_BELOW, WORK_ABOVE = 0.3, 0.8, 1.0      # m around the table's outline
MIN_REST_S = 0.5                        # s: shorter (or budget-cut) runs do not judge whether things came to rest


@dataclass
class Watch:
    """What to check for one soft item: ``kind`` ("fill", "pour", "pile", "cloth", "cable"), the engine ``name``
    of its card (None for a pour's emitted liquid), its container's layout ``key`` and cavity, and ``expect``
    (asked volume m^3, cable length m, angle of repose rad)."""

    kind: str
    name: str | None
    key: str | None = None
    interior: Any = None
    rest: tuple[float, float, float, float] | None = None
    expect: dict[str, float] = field(default_factory=dict)


def soft_check(watches: list[Watch], names: dict[str, str], table: np.ndarray, top: float = 0.0,
               poses: dict[str, tuple] | None = None) -> Any:
    """An ``alloy.check`` extra run check (``check(..., extra=[soft_check(...)])``) named "soft": the findings and
    metrics of ``soft_findings`` on the check's own run. ``poses``: engine name -> placement pose, read for static
    containers (they have no body).

    Example: ``alloy.check.check(card, extra=[soft_check(watches, names, table)])``."""
    def soft(res: Any, built: Any, index: Any) -> tuple[list[Any], dict[str, Any]]:
        from alloy.check.report import Finding  # noqa: PLC0415

        found, metrics = soft_findings(res, built, index, watches, names, table, top, poses)
        return [Finding("soft", f["level"], f["message"], f["fix"], objects=tuple(f["objects"])) for f in found], \
            metrics

    soft.check_name = "soft"                                       # type: ignore[attr-defined]
    return soft


def _finding(level: str, message: str, fix: str, name: str | None) -> dict[str, Any]:
    return {"level": level, "message": message, "fix": fix, "objects": [name] if name else []}


def soft_findings(res: Any, built: Any, index: Any, watches: list[Watch], names: dict[str, str],
                  table: np.ndarray, top: float = 0.0, poses: dict[str, tuple] | None = None
                  ) -> tuple[list[dict], dict[str, Any]]:
    """Findings (report-format dicts) and metrics for each watch from the run ``res`` (``names``: engine name ->
    layout key; ``table``: the support's outline, world xy; ``top``: its top height)."""
    if not res.samples:
        return [], {}
    engine = {k: n for n, k in names.items()}
    start, mid, end = res.samples[0], res.samples[len(res.samples) // 2], res.samples[-1]
    judge_rest = not res.truncated and res.seconds_run >= MIN_REST_S
    found: list[dict] = []
    metrics: dict[str, Any] = {}
    for w in watches:
        label = w.name or f"pour into {w.key}"
        try:
            m, f = _one(w, built, index, (engine, poses or {}), start, mid, end, table, top)
            if not judge_rest:                     # a run cut short says nothing about coming to rest
                f = [x for x in f if "still moves" not in x[1]]
        except Exception as exc:  # noqa: BLE001 - a probe that cannot read the run reports it, never hides it
            log.warning("compose soft probe %s: %s", label, exc)
            m, f = {"error": f"{type(exc).__name__}: {exc}"}, []
        metrics[label] = m
        found += [_finding(lvl, msg, fix, w.name) for lvl, msg, fix in f]
    return found, metrics


def _one(w: Watch, built, index, names, start, mid, end, table, top) -> tuple[dict, list]:
    if w.kind == "cable":
        return _cable(w, built, index, start, mid, end, table, top)
    parts, pw = _particles(w, built, index, start, end)
    m: dict[str, Any] = {"particles": int(len(parts))}
    out: list[tuple[str, str, str]] = []
    if not len(parts):
        if w.kind == "pour":           # a card emitter has no liquid of its own: alloy.check runs no pour
            return m, [("info", f"the pour into {w.key} was not run: alloy.check does not run a card's pour",
                        "alloy.load(path) pours it (Scene.add_pour)")]
        return m, [("error", f"{w.name or 'the item'}: no particles in the run", "check the item's route")]
    x0, xm, xe = start.particle_q[parts], mid.particle_q[parts], end.particle_q[parts]
    finite = np.isfinite(xe).all(axis=1)
    m["finite"] = round(float(finite.mean()), 4)
    out += _workspace(w, xe[finite], table, top, m)
    late = np.linalg.norm(xe - xm, axis=1)
    m["late_move_max_mm"] = round(float(np.nanmax(late)) * 1e3, 2) if late.size else 0.0
    if w.kind in ("fill", "pour"):
        out += _inside(w, built, index, names, end, xe, pw, m)
    if w.kind in ("fill", "pile"):
        out += _amount(w, built, parts, pw, m)
    if w.kind == "pile":
        _angle(xe[finite], m, w.expect.get("angle"))
    if w.kind == "cloth":
        out += _stretch(w, built, parts, x0, xe, m)
    if w.kind != "pour" and m["late_move_max_mm"] > LATE_WARN_M * 1e3:
        out.append(("warn", f"{w.name} still moves {m['late_move_max_mm']:.1f} mm over the second half of the run",
                    "run longer (seconds=...) or check its support"))
    return m, out


def _particles(w: Watch, built, index, start, end) -> tuple[np.ndarray, np.ndarray]:
    pw = built.model.particle_world.numpy() if built.model.particle_count else np.zeros(0, np.int64)
    if w.name is not None:
        parts = np.flatnonzero(index.particle_obj == index.names.index(w.name))
    else:                                            # a pour: particles no object owns that the run brought in
        free = np.flatnonzero(index.particle_obj < 0)
        moved = np.isfinite(end.particle_q[free]).all(axis=1) & (np.linalg.norm(
            end.particle_q[free] - start.particle_q[free], axis=1) > 1e-6)
        parts = free[moved]
    return parts, pw[parts]


def _workspace(w: Watch, x: np.ndarray, table: np.ndarray, top: float, m: dict) -> list:
    lo, hi = table.min(axis=0) - WORK_XY, table.max(axis=0) + WORK_XY
    ok = (x[:, 0] >= lo[0]) & (x[:, 0] <= hi[0]) & (x[:, 1] >= lo[1]) & (x[:, 1] <= hi[1]) & \
        (x[:, 2] >= top - WORK_BELOW) & (x[:, 2] <= top + WORK_ABOVE)
    m["in_workspace"] = round(float(ok.mean()), 4) if len(x) else 1.0
    if m["in_workspace"] < 1.0:
        return [("error", f"{w.name or 'the pour'}: {100 * (1 - m['in_workspace']):.1f} % of it left the workspace",
                 "a blow-up or a spill: check the health findings, or start it lower / slower")]
    return []


def _container_poses(w: Watch, built, index, names, end) -> list[tuple[int, np.ndarray]]:
    """(world, pose x y z qx qy qz qw) of the watched container in each world: its body, or its placement when
    static (held still)."""
    engine, poses = names
    try:
        bodies = index.bodies_of(engine[w.key])
    except ValueError:                                       # a static object has no entry among the bodies
        bodies = np.zeros(0, int)
    if len(bodies):
        bw = built.model.body_world.numpy()[bodies]
        return [(int(k), np.asarray(end.body_q[b])) for b, k in zip(bodies, bw)]
    pose = poses.get(engine[w.key])
    return [] if pose is None else [(k, np.asarray(pose, float)) for k in range(built.world_count)]


def _inside(w: Watch, built, index, names, end, xe, pw, m) -> list:
    """Share inside the container's cavity at the end, per world (the worst world counts)."""
    rest = geom.matrix_from_quat(w.rest)
    shares = []
    for k, q in _container_poses(w, built, index, names, end):
        sel = pw == k
        if not sel.any():
            continue
        rot = geom.matrix_from_quat(tuple(q[3:7]))
        local = (xe[sel] - q[None, :3]) @ rot @ rest.T
        shares.append(float(_in_cavity(w.interior, local).mean()))
    share = min(shares) if shares else 0.0
    m["in_container"] = round(share, 4)
    what = f"{w.name}" if w.name else f"the liquid poured into {w.key}"
    if share < IN_ERROR:
        return [("error", f"{what}: only {share:.0%} is inside {w.key} at the end (worst world)",
                 "less liquid, a slower pour, or a larger container")]
    if share < IN_WARN:
        return [("warn", f"{what}: {share:.1%} is inside {w.key} at the end", "check for splashes or a tilted "
                 "container")]
    return []


def _in_cavity(it, p: np.ndarray) -> np.ndarray:
    hull = geom.hull2d(it.xy)
    inside_xy = geom.depths(hull, p[:, :2]) >= -WALL_TOL
    return inside_xy & (p[:, 2] >= it.floor - WALL_TOL) & (p[:, 2] <= it.rim + WALL_TOL)


def _amount(w: Watch, built, parts: np.ndarray, pw: np.ndarray, m: dict) -> list:
    """Volume the particles stand for (their mass over the material density, worst world) against the ask."""
    v, rho = w.expect.get("volume"), w.expect.get("density")
    if not v or not rho or not len(parts):
        return []
    mass = built.model.particle_mass.numpy()[parts].astype(np.float64)
    vols = [float(mass[pw == k].sum()) / rho for k in np.unique(pw)]
    sim = max(vols, key=lambda x: abs(x / v - 1.0))
    err = sim / v - 1.0
    m.update(volume_ml_simulated=round(sim * 1e6, 2), volume_ml_asked=round(v * 1e6, 2), amount_error=round(err, 4))
    if abs(err) > AMOUNT_WARN:
        return [("warn", f"{w.name}: the particles stand for {sim * 1e6:.0f} ml, {v * 1e6:.0f} ml asked ({err:+.0%}: "
                 "the particle lattice is coarse for this amount)", "ask for more, or accept the simulated amount "
                 "(metrics list it)")]
    return []


def _angle(x: np.ndarray, m: dict, expect: float | None) -> None:
    """Settled pile: height over the 90th-percentile base radius, as an angle (the sourced one beside it)."""
    if len(x) < 10:
        return
    c = x[:, :2].mean(axis=0)
    r = np.linalg.norm(x[:, :2] - c, axis=1)
    h = float(np.percentile(x[:, 2], 99) - np.percentile(x[:, 2], 1))
    base = float(np.percentile(r, 90))
    m["settled_angle_deg"] = round(math.degrees(math.atan2(h, max(base, 1e-9))), 1)
    if expect is not None:
        m["angle_of_repose_deg"] = round(math.degrees(expect), 1)


def _stretch(w: Watch, built, parts: np.ndarray, x0: np.ndarray, xe: np.ndarray, m: dict) -> list:
    """Largest edge strain of the cloth's triangles at the end against its start (the start is the flat rest)."""
    tris = built.model.tri_indices.numpy() if built.model.tri_count else np.zeros((0, 3), np.int64)
    pos = {int(p): i for i, p in enumerate(parts)}
    mine = np.array([t for t in tris if int(t[0]) in pos and int(t[1]) in pos and int(t[2]) in pos])
    if not len(mine):
        return []
    idx = np.vectorize(pos.get)(mine)
    worst = 0.0
    for a, b in ((0, 1), (1, 2), (2, 0)):
        l0 = np.linalg.norm(x0[idx[:, a]] - x0[idx[:, b]], axis=1)
        le = np.linalg.norm(xe[idx[:, a]] - xe[idx[:, b]], axis=1)
        worst = max(worst, float(np.nanmax(le / np.maximum(l0, 1e-12) - 1.0)))
    m["max_strain"] = round(worst, 4)
    if worst > STRETCH_ERROR:
        return [("error", f"{w.name} is stretched {worst:.0%} at the end", "a blow-up or a snag: check the health "
                 "findings; lay it flatter or over a rounder object")]
    if worst > STRETCH_WARN:
        return [("warn", f"{w.name} is stretched {worst:.1%} at the end (cloth at the default resolution is "
                 "unconverged, XB4-5)", "accept it, or use a finer cloth")]
    return []


def _cable(w: Watch, built, index, start, mid, end, table, top) -> tuple[dict, list]:
    bodies = index.bodies_of(w.name)
    bw = built.model.body_world.numpy()[bodies]
    out: list = []
    m: dict[str, Any] = {"links": int(len(bodies) // max(1, built.world_count))}
    ratios, late = [], []
    for k in np.unique(bw):
        b = bodies[bw == k]
        p0, pe, pm = start.body_q[b, :3], end.body_q[b, :3], mid.body_q[b, :3]
        l0 = float(np.linalg.norm(np.diff(p0, axis=0), axis=1).sum())
        le = float(np.linalg.norm(np.diff(pe, axis=0), axis=1).sum())
        ratios.append(le / max(l0, 1e-12) - 1.0)
        late.append(float(np.max(np.linalg.norm(pe - pm, axis=1))))
        out += _workspace(w, pe, table, top, m) if not out else []
    err = max(ratios, key=abs) if ratios else 0.0
    m["length_change"] = round(err, 4)
    m["late_move_max_mm"] = round(max(late) * 1e3, 2) if late else 0.0
    if abs(err) > LENGTH_ERROR:
        out.append(("error", f"{w.name} changed length by {err:+.1%}", "a joint pulled apart: check the health "
                    "findings and the attachments"))
    elif abs(err) > LENGTH_WARN:
        out.append(("warn", f"{w.name} changed length by {err:+.1%}", "check its attachments"))
    if m["late_move_max_mm"] > LATE_WARN_M * 1e3:
        out.append(("warn", f"{w.name} still moves {m['late_move_max_mm']:.1f} mm over the second half of the run",
                    "run longer (seconds=...)"))
    return m, out


__all__ = ["Watch", "soft_check", "soft_findings"]
