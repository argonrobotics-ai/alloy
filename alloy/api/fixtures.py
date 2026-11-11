# SPDX-License-Identifier: Apache-2.0
"""Static fixtures built from convex boxes: open boxes (bins, square cups), round cups and blocks with a hole.

MuJoCo collides each box exactly (a single non-convex mesh would collide as its convex hull, filling the hollow), so
these are the fast, robust way to model containers and sockets. Each fixture is several static objects named
``<name>_floor``, ``<name>_wall<k>``; ``pos`` is the centre of the fixture's bottom face. Used by ``alloy.Scene``.
A kinematic fixture (``kinematic=True``) is one kinematic body carrying every part's box (``FixtureRouter`` marks the
parts, ``alloy.scene.emitters.rigid.emit_part`` emits them): its state is ``<name>`` (the floor's frame; also
``<name>_floor``), and ``sim.drive(name)`` moves that one body.

Example::

    scene.add_open_cylinder("cup", inner_radius=0.04, depth=0.10, wall=0.004)     # a cup standing at the origin
    scene.add_block_with_hole("block", size=(0.1, 0.1, 0.05), hole=(0.0102, 0.0102), depth=0.03)
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy.api import _args as A
from alloy.api.errors import SceneError

Part = tuple[str, tuple[float, float, float], tuple[float, float, float], float]   # suffix, size, centre, yaw

# A scene with a precise fixture (add_block_with_hole) steps its rigid bodies at PRECISE_DT or finer and gives stiff
# materials PRECISE_K. At the default steps (1-1.25 ms) MuJoCo's contact stiffness is capped by the step, and a 10 mm
# peg landing on a hole floor at 0.8 m/s sinks 1.0 mm; at 0.25 ms with 2e7 N/m it sinks 0.27 mm (T3, CPU).
PRECISE_DT = 2.5e-4        # s
PRECISE_K = 2.0e7          # N/m per material (the engine default is 2e6, series 1e6 for a pair)
PRECISE_MIN_E = 1.0e9      # Pa: plastics, wood and metals get PRECISE_K; rubbers and foams keep their own


def precise_substeps(scene: Any) -> int:
    """Engine periods per control period: 1, or with a precise fixture in ``scene`` the smallest power of 2 that
    brings ``scene.dt`` to ``PRECISE_DT`` or below. Example: ``precise_substeps(scene)`` -> ``4`` at dt 0.001."""
    if not any(item.get("precise") for item in scene.items.values()):
        return 1
    return 2 ** max(0, math.ceil(math.log2(scene.dt / PRECISE_DT - 1e-9)))


# Static fixtures: walls run the full outer length and from z = 0 (through the floor), overlapping the floor and each
# other, so no part's face lies on another part's face inside the solid. A wall standing on the floor puts its bottom
# face (normal down) on the floor's top; a particle solver that takes the nearest collider face (MPM's grid
# colliders) then pushes grains in the wall's foot down through the floor (dry sand fell out of an open-box tray,
# 3 of 3 runs). Rigid contacts see the same inner and outer faces either way: static parts never collide with each
# other. Kinematic fixtures keep abutting parts (``overlap=False``): when their parts were separate moving bodies,
# overlapping ones read as 4 mm penetrations from the first step (health latched the world). They are one body now
# (parts of one body never collide) and keep the same boxes, so tilts and pours from them are unchanged.


def open_box_parts(inner: Sequence[float], wall: float, overlap: bool = True) -> list[Part]:
    """Floor and four walls of an open-top box: cavity ``inner`` (x, y, depth), floor top at z = ``wall``;
    ``overlap``: walls through the floor and full length (static), else walls standing on it (kinematic).

    Example: ``open_box_parts((0.08, 0.08, 0.1), 0.004)``."""
    ix, iy, depth = A.sizes(inner, 3, "inner")
    w = A.positive(wall, "wall", "m")
    ox, oy = ix + 2 * w, iy + 2 * w
    z0, ly = (0.0, ox) if overlap else (w, ix)                   # wall bottom, y-wall length
    h, zc = w + depth - z0, (z0 + w + depth) / 2
    return [("floor", (ox, oy, w), (0.0, 0.0, w / 2), 0.0),
            ("wall0", (w, oy, h), ((ix + w) / 2, 0.0, zc), 0.0),
            ("wall1", (w, oy, h), (-(ix + w) / 2, 0.0, zc), 0.0),
            ("wall2", (ly, w, h), (0.0, (iy + w) / 2, zc), 0.0),
            ("wall3", (ly, w, h), (0.0, -(iy + w) / 2, zc), 0.0)]


def open_cylinder_parts(inner_radius: float, depth: float, wall: float, segments: int = 24,
                        overlap: bool = True) -> list[Part]:
    """A square floor plate and ``segments`` wall slabs whose inner faces are tangent to ``inner_radius``;
    ``overlap``: slabs from z = 0 through the plate (static), else standing on it (kinematic).

    Example: ``open_cylinder_parts(0.04, 0.10, 0.004)``."""
    r = A.positive(inner_radius, "inner_radius", "m")
    d, w = A.positive(depth, "depth", "m"), A.positive(wall, "wall", "m")
    if not isinstance(segments, int) or segments < 8:
        raise SceneError(f"segments must be an int >= 8, got {segments!r}", "pass segments=24", field="segments")
    outer = (r + w) / math.cos(math.pi / segments)          # covers the corners between slabs
    chord = 2.0 * outer * math.tan(math.pi / segments)
    parts: list[Part] = [("floor", (2 * outer, 2 * outer, w), (0.0, 0.0, w / 2), 0.0)]
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        c = r + w / 2
        z0 = 0.0 if overlap else w
        parts.append((f"wall{k}", (w, chord, w + d - z0), (c * math.cos(a), c * math.sin(a), (z0 + w + d) / 2), a))
    return parts


def block_with_hole_parts(size: Sequence[float], hole: Sequence[float], depth: float,
                          overlap: bool = True) -> list[Part]:
    """A block of full ``size`` with a rectangular blind hole (x, y) of ``depth`` in its top, as five boxes;
    ``overlap``: walls from z = 0 through the floor and full length (static), else on the floor (kinematic).

    Example: ``block_with_hole_parts((0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03)``."""
    sx, sy, sz = A.sizes(size, 3, "size")
    hx, hy = A.sizes(hole, 2, "hole")
    d = A.positive(depth, "depth", "m")
    if not (hx < sx and hy < sy and d < sz):
        raise SceneError(f"hole {hole} x {depth} m does not fit in the block {size}", "make the hole smaller than "
                         "the block (and the depth less than its height)", field="hole")
    floor = sz - d
    side_x, side_y = (sx - hx) / 2, (sy - hy) / 2
    z0, ly = (0.0, sx) if overlap else (floor, hx)               # wall bottom, y-wall length
    h, zc = sz - z0, (z0 + sz) / 2
    return [("floor", (sx, sy, floor), (0.0, 0.0, floor / 2), 0.0),
            ("wall0", (side_x, sy, h), ((hx + side_x) / 2, 0.0, zc), 0.0),
            ("wall1", (side_x, sy, h), (-(hx + side_x) / 2, 0.0, zc), 0.0),
            ("wall2", (ly, side_y, h), (0.0, (hy + side_y) / 2, zc), 0.0),
            ("wall3", (ly, side_y, h), (0.0, -(hy + side_y) / 2, zc), 0.0)]


def place(scene: Any, name: str, parts: list[Part], pos: Any, yaw: float, material: str,
          kinematic: bool = False, precise: bool = False, opening: float | None = None) -> list[str]:
    """Add the parts as static (or kinematic) boxes around ``pos`` (bottom centre) turned by ``yaw``; returns their
    names. Each part records its fixture; a kinematic fixture's parts are one body (``FixtureRouter``) that
    ``sim.drive(name)`` moves, and it takes ``pos`` as one row per world (each part follows); a static one is the
    same in every world. ``opening``: the
    cavity's narrower inner width [m], which a build checks against the liquid's particle size
    (``_liquid_accuracy``)."""
    from alloy.api import _worlds  # noqa: PLC0415

    A.check_name(name, list(scene.items))
    if _worlds.per_world(pos) and not kinematic:
        _worlds.refuse_static(name, "a static fixture")
    first, rows = A.positions(pos, scene.worlds)
    starts = rows if rows is not None else [first]
    c, s = math.cos(yaw), math.sin(yaw)
    names = []
    for suffix, size, (px, py, pz), part_yaw in parts:
        centre = [[x0 + c * px - s * py, y0 + s * px + c * py, z0 + pz] for x0, y0, z0 in starts]
        part = scene.add_box(f"{name}_{suffix}", size=size, pos=centre if len(centre) > 1 else centre[0],
                             rpy=(0.0, 0.0, yaw + part_yaw),
                             material=material, static=not kinematic, kinematic=kinematic)
        scene.items[part]["fixture"] = name
        if opening is not None:
            scene.items[part]["opening"] = float(opening)
        if precise:
            scene.items[part]["precise"] = True
        names.append(part)
    return names

def motion(name: str, static: bool | None, kinematic: bool) -> None:
    """Check a fixture's ``static`` / ``kinematic`` keywords: a fixture is static unless kinematic, so ``static=True``
    changes nothing (the docs list it for every ``add_*``; agent trial r2, R2-D1). Example: ``motion("bin", True,
    False)``."""
    if static is None or (static and not kinematic):
        return
    if static and kinematic:
        raise SceneError(f"fixture {name!r} cannot be both static and kinematic", "pass one of static=True (never "
                         "moves) or kinematic=True (moved by sim.drive / set_pose)", field="kinematic")
    raise SceneError(f"fixture {name!r}: static=False, but a fixture is static or kinematic; its parts are never "
                     "free bodies", "pass kinematic=True to move it (sim.drive), or build a free container with "
                     "add_mesh(..., vertices=, faces=) from alloy.api.meshes", field="static")


def kinematic_groups(items: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """{fixture: its part names, root (the floor) first} for every kinematic fixture in a Scene's ``items``.
    Example: ``kinematic_groups(scene.items)`` -> ``{"jug": ["jug_floor", "jug_wall0", ...]}``."""
    out: dict[str, list[str]] = {}
    for n, it in items.items():
        if it.get("fixture") and (it.get("card_extra") or {}).get("motion") == "kinematic":
            out.setdefault(str(it["fixture"]), []).append(n)
    return out


class FixtureRouter:
    """A router that marks each kinematic fixture's parts with ``FIXTURE_GROUP`` (and its root with
    ``FIXTURE_ROOT``), so the rigid emitter makes the fixture one kinematic body (module docstring). Every other
    decision is the wrapped router's, unchanged. Example: ``FixtureRouter(router, kinematic_groups(scene.items))``."""

    def __init__(self, router: Any, groups: Mapping[str, Sequence[str]]):
        self.router = router
        self.parts = {p: (k + 1, name, i == 0) for k, (name, parts) in enumerate(groups.items())
                      for i, p in enumerate(parts)}

    def route(self, obj: Any, ctx: Any) -> Any:
        from dataclasses import replace  # noqa: PLC0415

        from alloy.scene.emitters.rigid import FIXTURE_GROUP, FIXTURE_ROOT  # noqa: PLC0415 - built scenes only

        d = self.router.route(obj, ctx)
        hit = self.parts.get(obj.card.name)
        if hit is None:
            return d
        group, name, root = hit
        numbers = {**d.numbers, FIXTURE_GROUP: float(group), **({FIXTURE_ROOT: 1.0} if root else {})}
        why = (f"kinematic fixture {name!r}: one body, " + ("this part's, carrying the other parts" if root else
               "this part a shape on it"))
        return replace(d, numbers=numbers, reasons=(*d.reasons, why))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)


def with_bodies(router: Any, items: Mapping[str, Mapping[str, Any]]) -> Any:
    """``router`` itself, or a FixtureRouter around it when the scene has a kinematic fixture.
    Example: ``router = with_bodies(router, scene.items)``."""
    groups = kinematic_groups(items)
    return FixtureRouter(router, groups) if groups else router


def check_riders(items: Mapping[str, Mapping[str, Any]], attachments: Sequence[Mapping[str, Any]] = ()) -> None:
    """BuildError for what a rider (a kinematic fixture's part other than its floor) cannot take now that it is a shape
    on the floor's body: a randomized start position (randomize the floor: it moves the whole fixture) or an
    attachment endpoint (attach to the floor: the same rigid body). A density draw on any part, the floor included, is
    refused too: the one body carries every part's mass, and a draw scales one part's object only (WP2's review).
    Example: ``check_riders(scene.items)``."""
    from alloy.api import _random  # noqa: PLC0415
    from alloy.api.errors import BuildError  # noqa: PLC0415

    groups = kinematic_groups(items)
    for fixture, parts in groups.items():
        for name in parts:
            rho = _random.card_path(name, "density")
            if any(r.get("field") == rho for r in items[name].get("randomize") or ()):
                raise BuildError(f"{name!r} is a part of the kinematic fixture {fixture!r}, which is one body: its "
                                 "density cannot be randomized part by part", "drop the density draw (a kinematic "
                                 "fixture moves as driven; its mass only sets how its weld yields)", field=name)
    riders = {p: (f, parts[0]) for f, parts in groups.items() for p in parts[1:]}
    for name, (fixture, root) in riders.items():
        axes = {_random.card_path(name, ax) for ax in ("x", "y", "z")}
        if any(r.get("field") in axes for r in items[name].get("randomize") or ()):
            raise BuildError(f"{name!r} is a part of the kinematic fixture {fixture!r}, which is one body: its start "
                             "cannot be randomized alone", f"randomize {root!r} instead (it moves the whole fixture)",
                             field=name)
    for att in attachments:
        for end in ((att.get("a"), att.get("b")) if isinstance(att, Mapping) else (att.a, att.b)):
            obj = str(end or "").partition(":")[0]
            if obj in riders:
                raise BuildError(f"attachment endpoint {end!r} is a part of the kinematic fixture {riders[obj][0]!r}, "
                                 "which is one body", f"attach to {riders[obj][1]!r}: the body that carries every part",
                                 field="attachments")


def own_rows(per_world_pos: Mapping[str, Any], items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per-world start rows less a kinematic fixture's non-root parts: they ride the root's body, whose row moves
    them all. Example: ``own_rows(materialized.per_world_pos, scene.items)``."""
    riders = {p for parts in kinematic_groups(items).values() for p in parts[1:]}
    return {n: r for n, r in per_world_pos.items() if n not in riders}


class FixtureMixin:
    """``alloy.Scene`` methods that add static (or kinematic) fixtures built from boxes."""

    def add_open_box(self, name: str, inner: Sequence[float], wall: float = 0.004, *, pos: Any = (0.0, 0.0, 0.0),
                     yaw: float = 0.0, material: str = "abs",
                     kinematic: bool = False, static: bool | None = None) -> list[str]:
        """A static open-top box (bin, square cup) from five boxes: cavity ``inner`` (x, y, depth) in m, walls and
        floor ``wall`` thick; ``pos`` is the centre of its bottom face (the cavity floor is ``wall`` higher).
        Returns the part names ``<name>_floor``, ``<name>_wall0..3``. ``kinematic=True``: one body that
        ``sim.drive(name, ...)`` moves (tilt a container to pour from it). ``static=True`` is accepted (fixtures are
        static unless kinematic), as on every ``add_*``.
        A kinematic one is one body: read ``"<name>.pos"`` / ``"<name>.quat_xyzw"`` (its floor's frame; also
        ``"<name>_floor"``); the other parts have no state of their own. It makes no contacts with static objects or
        other kinematic ones (neither can push the other), only with moving bodies, cloth and liquid.

        Example: ``scene.add_open_box("bin", inner=(0.08, 0.08, 0.10), wall=0.004)``."""
        motion(name, static, kinematic)
        return place(self, name, open_box_parts(inner, wall, overlap=not kinematic), pos, yaw, material, kinematic,
                     opening=min(float(inner[0]), float(inner[1])))

    def add_open_cylinder(self, name: str, inner_radius: float, depth: float, wall: float = 0.004, *,
                          pos: Any = (0.0, 0.0, 0.0), segments: int = 24, material: str = "abs",
                          kinematic: bool = False, static: bool | None = None) -> list[str]:
        """A static round cup from a floor plate and ``segments`` wall slabs (inner faces tangent to
        ``inner_radius``); ``pos`` is the centre of its bottom face. Returns the part names. ``static=True`` is
        accepted (as on every ``add_*``); ``kinematic=True`` makes it one body that ``sim.drive(name, ...)`` moves.
        A kinematic one is one body: read ``"<name>.pos"`` / ``"<name>.quat_xyzw"`` (its floor's frame; also
        ``"<name>_floor"``); the other parts have no state of their own. It makes no contacts with static objects or
        other kinematic ones (neither can push the other), only with moving bodies, cloth and liquid.

        Example: ``scene.add_open_cylinder("cup", inner_radius=0.04, depth=0.10, wall=0.004)`` (about 500 ml)."""
        motion(name, static, kinematic)
        parts = open_cylinder_parts(inner_radius, depth, wall, segments, overlap=not kinematic)
        return place(self, name, parts, pos, 0.0, material, kinematic, opening=2.0 * float(inner_radius))

    def add_block_with_hole(self, name: str, size: Sequence[float], hole: Sequence[float], depth: float, *,
                            pos: Any = (0.0, 0.0, 0.0), yaw: float = 0.0, material: str = "abs",
                            kinematic: bool = False, precise: bool = True,
                            static: bool | None = None) -> list[str]:
        """A static block of full ``size`` with a rectangular blind hole ``hole`` (x, y) and ``depth`` in its top,
        centred; ``pos`` is the centre of its bottom face. Exact box contacts: a 10 mm peg drops into a 10.2 mm hole.

        ``precise=True`` (default) makes the whole scene a precise-fit scene: rigid bodies step at 0.25 ms or finer
        (``sim.dt`` is unchanged; the engine takes several periods per ``step()``) and plastics, wood and metals
        get a machined part's contact stiffness (2e7 N/m). Why: at the default 1-1.25 ms step the contact stiffness
        is capped by the step, and a peg landing on the hole floor at 0.8 m/s sinks 1 mm into it; precise, 0.27 mm.
        It costs 4-8x more rigid steps; ``precise=False`` keeps the default step. ``static=True`` is accepted.
        A kinematic one is one body: read ``"<name>.pos"`` / ``"<name>.quat_xyzw"`` (its floor's frame; also
        ``"<name>_floor"``); the other parts have no state of their own. It makes no contacts with static objects or
        other kinematic ones (neither can push the other), only with moving bodies, cloth and liquid.

        Example: ``scene.add_block_with_hole("block", (0.1, 0.1, 0.05), hole=(0.0102, 0.0102), depth=0.03)``."""
        motion(name, static, kinematic)
        return place(self, name, block_with_hole_parts(size, hole, depth, overlap=not kinematic), pos, yaw, material,
                     kinematic, precise=bool(precise), opening=min(float(hole[0]), float(hole[1])))
