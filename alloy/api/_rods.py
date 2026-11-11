# SPDX-License-Identifier: Apache-2.0
"""``Scene.add_rod`` (SCR-WP9-ROD): an elastic rod as a curve object card, its clamped ends, and its centre line.

**The card.** A rod is a `curve` object card: a centre-line polyline and a radius, with the material's E and nu, or
a card's measured bend and twist stiffness. The router reads it as a rod.
- `DEFAULT_ROUTE` None leaves the route to the router: CHAIN until SCR-WP9-ROD's flip.
- "rod_vbd" writes a route override to ROD_VBD, Newton's Cosserat rod on `SolverVBDRod`.
- `segments` becomes the card's `features.chain_links`.

**Clamps** live on the item, not the card, since a card field would change every card's hash. A clamped rod asks for
ROD_VBD in its card (OSS-3's review): after assembly, `clamp_ends` makes the end segments kinematic in every world,
so the rod solve holds them exactly where they were built. Opt-in (S4, 2026-11-09): `route="chain"` with
clamp="start" keeps it on CHAIN instead. The scene's router is wrapped (`with_clamps`) so the rod's decision carries
`ROD_CLAMP_NUMBER`, and the chain emitter welds link 0 to the world, as it does for a tree weld's root. Beside a robot
that is far cheaper: the chain steps inside MuJoCo, with no exchange clock. Saved cards (`scene.save()`) do not keep a
clamp (it warns); Scene JSON does.

**`"<rod>.points"`** is the centre line, (worlds, segments + 1, 3). Each node is a segment's end point: its capsule
shape's axis ends in the body frame, ordered along the rod at build. This works on both routes. CHAIN's capsules end
one radius short of their link's ends, so its caps meet at the joints, and their axis ends are extended by the
radius. ROD_VBD's capsules span their segment.

**`"<rod>.twist"`** is each joint's twist, (worlds, segments - 1) in radians: the rotation of segment i + 1 relative
to segment i about segment i's axis (a swing-twist split of the relative rotation). Its sum is the end-to-end twist.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.api import _args as A
from alloy.api.errors import SceneError

log = logging.getLogger("alloy.api")
ROD_MATERIAL = "cable_usb_composite"
DEFAULT_ROUTE: str | None = None
CLAMPS = ("start", "end", "both")
ROD_ROUTES = ("chain", "rod_vbd")


def centre_line(points: Any, length: float | None, direction: Sequence[float]) -> np.ndarray:
    """(n, 3) polyline relative to the item's pos: `points` as given, else `length` along `direction` from 0."""
    if points is not None:
        if length is not None:
            raise SceneError("both points and length given", "pass the rod's centre line as points=[...] or "
                             "length= with direction=", field="length")
        pts = np.asarray(points, np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2 or not np.isfinite(pts).all():
            raise SceneError(f"points must be at least two finite (x, y, z) points, got shape {pts.shape}",
                             "pass points=[(0, 0, 0), (0.5, 0, 0)]", field="points")
        if np.any(np.linalg.norm(np.diff(pts, axis=0), axis=1) <= 0.0):
            raise SceneError("points has two equal neighbours", "drop the repeated point", field="points")
        return pts
    if length is None:
        raise SceneError("a rod needs points or a length", "pass length=0.5 (m) or points=[...]", field="length")
    d = np.asarray(direction, np.float64)
    if d.shape != (3,) or not np.linalg.norm(d) > 0.0:
        raise SceneError(f"direction must be a nonzero (x, y, z), got {direction!r}", "pass direction=(1, 0, 0)",
                         field="direction")
    L = A.positive(length, "length", "m")
    return np.stack([np.zeros(3), L * d / np.linalg.norm(d)])


def spec(points: Any, radius: float, length: float | None, direction: Sequence[float], segments: int | None,
         clamp: str | None = None, route: str | None = None) -> tuple[dict[str, Any], dict[str, Any], float]:
    """(card shape, card extras, volume) of a rod item; `route` "chain" or "rod_vbd" asks for that route."""
    r = A.positive(radius, "radius", "m")
    pts = centre_line(points, length, direction)
    arc = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    extra: dict[str, Any] = {}
    if segments is not None:
        if isinstance(segments, bool) or int(segments) != segments or int(segments) < 3:
            raise SceneError(f"segments must be a whole number >= 3, got {segments!r}", "pass segments=20, or drop "
                             "it to size them from the rod's own sag", field="segments")
        extra["features"] = {"chain_links": int(segments)}
    if checked_route(route, checked_clamp(clamp)) is not None:
        extra["route"] = {"route": route, "reason": f"alloy.Scene.add_rod(route={route!r})"}
    elif clamp is not None:
        extra["route"] = {"route": "rod_vbd",
                          "reason": "alloy.Scene.add_rod(clamp=...): the elastic rod holds an end (route='chain' "
                                    "with clamp='start' welds the chain's first link instead)"}
    elif DEFAULT_ROUTE is not None:
        extra["route"] = {"route": DEFAULT_ROUTE, "reason": "alloy.Scene.add_rod"}
    shape = {"curve": {"points": [[float(v) for v in p] for p in pts], "radius": r}}
    return shape, extra, math.pi * r * r * arc


def refuse_per_world(name: str) -> None:
    """A rod starts at one pos in every world (per-world starts and randomized x / y / z are not supported).

    Example: ``refuse_per_world("cable")`` raises SceneError(field="pos")."""
    raise SceneError(f"rod {name!r}: per-world start positions are not supported for rods",
                     "build the rod at one pos for every world", field="pos")


def warn_unsaved_clamps(items: Mapping[str, Mapping[str, Any]]) -> None:
    """scene.save() writes cards, which carry no clamp: say so once.

    Example: ``warn_unsaved_clamps(scene.items)`` logs "scene.save(): the cards do not keep clamp= for cable; ..."."""
    lost = sorted(n for n, it in items.items() if it["kind"] == "rod" and it.get("rod", {}).get("clamp"))
    if lost:
        log.warning("scene.save(): the cards do not keep clamp= for %s; a scene loaded from them has those ends "
                    "free. Keep clamps with scene.to_json() / Scene.from_dict.", ", ".join(lost))


def checked_clamp(clamp: str | None) -> str | None:
    if clamp is not None and clamp not in CLAMPS:
        raise SceneError(f"clamp must be one of {CLAMPS} or None, got {clamp!r}", "pass clamp='start' to hold the "
                         "rod's first end where it is built", field="clamp")
    return clamp


def checked_route(route: str | None, clamp: str | None) -> str | None:
    """`route` for add_rod: None (the router's choice), "chain" or "rod_vbd"; the chain holds only its start."""
    if route is not None and route not in ROD_ROUTES:
        raise SceneError(f"route must be one of {ROD_ROUTES} or None, got {route!r}", "pass route='chain' or "
                         "route='rod_vbd', or drop it to let the router choose", field="route")
    if route == "chain" and clamp not in (None, "start"):
        raise SceneError(f"clamp={clamp!r} on the chain route: the chain holds its first end only", "pass "
                         "clamp='start' with the points in reverse to hold the other end, or route='rod_vbd'",
                         field="clamp")
    return route


def chain_clamped(items: Mapping[str, Mapping[str, Any]]) -> frozenset[str]:
    """Names of the rod items asking for the chain route with clamp="start"."""
    return frozenset(n for n, it in items.items() if it["kind"] == "rod" and it["rod"].get("clamp") == "start"
                     and it["rod"].get("route") == "chain")


class ClampRouter:
    """A router that marks each chain-clamped rod's decision with ``ROD_CLAMP_NUMBER`` (module docstring).
    Every other decision is the wrapped router's, unchanged."""

    def __init__(self, router: Any, names: frozenset[str]):
        self.router, self.names = router, names

    def route(self, obj: Any, ctx: Any) -> Any:
        from dataclasses import replace  # noqa: PLC0415

        from alloy.scene.emitters.chain import ROD_CLAMP_NUMBER  # noqa: PLC0415 - built scenes only

        d = self.router.route(obj, ctx)
        if obj.card.name not in self.names or str(d.route) != "chain":
            return d
        return replace(d, numbers={**d.numbers, ROD_CLAMP_NUMBER: 1.0},
                       reasons=(*d.reasons, "add_rod(clamp='start', route='chain'): link 0 welded to the world"))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)


def with_clamps(router: Any, items: Mapping[str, Mapping[str, Any]]) -> Any:
    """`router` itself, or a ClampRouter around it when some rod asks for a clamped chain."""
    names = chain_clamped(items)
    return ClampRouter(router, names) if names else router


def clamp_ends(built: Any, items: Mapping[str, Mapping[str, Any]]) -> None:
    """Make each clamped rod's end segments kinematic in every world (module docstring). A chain-clamped rod's link 0
    is welded by its emitter instead."""
    chained = chain_clamped(items)
    clamped = {n: it["rod"]["clamp"] for n, it in items.items()
               if it["kind"] == "rod" and it["rod"].get("clamp") and n not in chained}
    for name in chained:
        if str(built.handles[name].decision.route) != "chain":
            raise SceneError(f"rod {name!r}: route='chain' was asked for, but it routed "
                             f"{built.handles[name].decision.route}", "drop route=, or use route='rod_vbd'",
                             field="route")
    if not clamped:
        return
    from alloy._vendor import newton  # noqa: PLC0415 - built scenes only

    model = built.model
    flags = model.body_flags.numpy()
    for name, clamp in clamped.items():
        h = built.handles[name]
        if str(h.decision.route) != "rod_vbd":
            raise SceneError(f"rod {name!r}: clamp needs the elastic-rod route (rod_vbd), it routed {h.decision.route}",
                             "drop clamp=, or remove the route override that sends this rod elsewhere", field="clamp")
        ends = {"start": (0,), "end": (-1,), "both": (0, -1)}[clamp]
        body = np.arange(h.template.body.start, h.template.body.stop)
        stride = len(h.world_stride.body)
        for e in ends:
            rows = body[e] + stride * np.arange(int(built.world_count))
            flags[rows] |= int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)


@dataclass(frozen=True)
class RodLine:
    """Template body ids of a rod's segments in order, and each segment's start and end in its body frame."""

    bodies: np.ndarray
    start: np.ndarray
    end: np.ndarray


def rod_lines(built: Any) -> dict[str, RodLine]:
    """Centre-line geometry of every rod object (CHAIN or ROD_VBD) in a built scene."""
    m = built.model
    out: dict[str, RodLine] = {}
    rods = {n: h for n, h in built.handles.items() if str(h.decision.route) in ROD_ROUTES}
    if not rods:
        return out
    sb, st, ss = m.shape_body.numpy(), m.shape_transform.numpy(), m.shape_scale.numpy()
    bq = m.body_q.numpy()
    for name, h in rods.items():
        bodies = np.arange(h.template.body.start, h.template.body.stop)
        cap = str(h.decision.route) == "chain"                                 # caps inside the link
        ends = np.stack([_axis_ends(b, sb, st, ss, cap) for b in bodies])       # (N, 2, 3) body frame
        world = np.stack([[_apply(bq[b], e) for e in ends[i]] for i, b in enumerate(bodies)])
        out[name] = _ordered(bodies, ends, world)
    return out


def plastic_strain(engine: Any, rows: np.ndarray) -> np.ndarray | None:
    """(worlds, N - 1): each joint's largest equivalent plastic strain over its fibres (alloy.solvers.rod.plastic);
    None when no solver steps this rod with yielding joints. `rows` as for `points` (global body rows per segment)."""
    from alloy.solvers.rod.topology import ROD  # noqa: PLC0415

    for solver, g2l in _rod_solvers(engine):
        plastic = getattr(solver, "rod_plastic", None)
        if plastic is None:
            continue
        m = solver.model
        jc, jt = m.joint_child.numpy(), m.joint_type.numpy()
        by_child = {int(c): j for j, c in enumerate(jc) if int(jt[j]) == ROD}
        local = rows if g2l is None else g2l[rows]
        joints = np.array([[by_child.get(int(b), -1) for b in local[k]] for k in range(1, len(rows))])
        if (joints < 0).any():
            continue
        return plastic.ebar_max.numpy()[joints].T.copy()
    return None


def _rod_solvers(engine: Any):
    """(solver, global -> local body map or None) for the engine's VBD solvers: a coupled stepper's entries, else the
    single entry's solver over the whole model."""
    stepper = getattr(engine, "stepper", None)
    entries = getattr(stepper, "_entries", None)
    if entries:
        for e in entries.values():
            g2l = getattr(e, "body_global_to_local", None)
            yield e.solver, (g2l.numpy() if g2l is not None else None)
        return
    solver = getattr(stepper, "solver", None)
    if solver is not None:
        yield solver, None


def twist(line: RodLine, body_q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(worlds, N - 1) joint twists (module docstring). `rows` as for `points`."""
    axis = line.end - line.start
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)                   # (N, 3) body frame
    out = []
    for k in range(len(line.bodies) - 1):
        qa, qb = body_q[rows[k], 3:7].astype(np.float64), body_q[rows[k + 1], 3:7].astype(np.float64)
        rel = _quat_mul(_quat_conj(qa), qb)
        tw = 2.0 * np.arctan2(rel[:, :3] @ axis[k], rel[:, 3])
        out.append(np.where(tw > np.pi, tw - 2.0 * np.pi, np.where(tw < -np.pi, tw + 2.0 * np.pi, tw)))
    return np.stack(out, axis=1) if out else np.zeros((rows.shape[1], 0))


def points(line: RodLine, body_q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(worlds, N + 1, 3) nodes: every segment's start, then the last segment's end. `rows[k]`: segment k's body row
    in each world, (N, worlds)."""
    starts = np.stack([_apply_many(body_q[rows[k]], line.start[k]) for k in range(len(line.bodies))], axis=1)
    last = _apply_many(body_q[rows[-1]], line.end[-1])[:, None, :]
    return np.concatenate([starts, last], axis=1)


def _axis_ends(body: int, shape_body: np.ndarray, shape_tf: np.ndarray, scale: np.ndarray,
               caps_inside: bool) -> np.ndarray:
    """The body's segment ends (2, 3) in the body frame: shape transform x (0, 0, +-h), h the capsule's half height
    (plus its radius when the caps lie inside the segment)."""
    s = int(np.flatnonzero(shape_body == body)[0])
    hh = float(scale[s][1]) + (float(scale[s][0]) if caps_inside else 0.0)
    return np.stack([_apply(shape_tf[s], (0.0, 0.0, -hh)), _apply(shape_tf[s], (0.0, 0.0, hh))])


def _ordered(bodies: np.ndarray, ends: np.ndarray, world: np.ndarray) -> RodLine:
    """Pick each segment's start as the end nearer the previous segment (the first: farther from the second)."""
    n = len(bodies)
    start, end = np.empty((n, 3)), np.empty((n, 3))
    for i in range(n):
        ref = world[i - 1].mean(axis=0) if i > 0 else (world[1].mean(axis=0) if n > 1 else world[0][0])
        near = int(np.argmin(np.linalg.norm(world[i] - ref, axis=1)))
        first = near if i > 0 else 1 - near
        start[i], end[i] = ends[i][first], ends[i][1 - first]
    return RodLine(bodies, start, end)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.concatenate([-q[..., :3], q[..., 3:4]], axis=-1)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products of quaternions (..., 4) x y z w."""
    av, aw, bv, bw = a[..., :3], a[..., 3:4], b[..., :3], b[..., 3:4]
    return np.concatenate([aw * bv + bw * av + np.cross(av, bv), aw * bw - np.sum(av * bv, -1, keepdims=True)], -1)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate v by quaternions q (..., 4) x y z w."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _apply(tf: Any, p: Any) -> np.ndarray:
    tf = np.asarray(tf, np.float64)
    return tf[:3] + _quat_rotate(tf[3:7], np.asarray(p, np.float64))


def _apply_many(tfs: np.ndarray, p: np.ndarray) -> np.ndarray:
    tfs = np.asarray(tfs, np.float64)
    return tfs[:, :3] + _quat_rotate(tfs[:, 3:7], np.broadcast_to(np.asarray(p, np.float64), (len(tfs), 3)))


__all__ = ["CLAMPS", "ClampRouter", "DEFAULT_ROUTE", "ROD_MATERIAL", "ROD_ROUTES", "RodLine", "centre_line",
           "chain_clamped", "checked_clamp", "checked_route", "clamp_ends", "with_clamps",
           "plastic_strain", "points", "refuse_per_world", "rod_lines", "spec", "twist", "warn_unsaved_clamps"]
