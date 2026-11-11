# SPDX-License-Identifier: Apache-2.0
"""Route pairs Alloy cannot step together yet, refused at build with the pair and the nearest supported alternative.

WP13's material-combination sweep (2026-11-07, [internal note] notes/combo-matrix-1107.md) found pairs that built and then
failed (F1: the first step raised a solver's own NotImplementedError or ValueError, with no Alloy class and no fix) or
ran wrong without a word (F4: cloth fell through film paper). ``UNSUPPORTED`` is the one list of them:
``alloy.integration.wiring.wire`` calls ``check_built`` before it wires any part, with or without plug-ins
(``alloy.build`` raises BuildError before a step), and WP13's tests/integration/test_combo_matrix.py pins it.
``FILLED_SOFT_CONTAINER`` (F12) is about where objects start, not their routes: an SPH liquid that starts inside a
free cloth or soft body (a soft cup filled at build). The cup alone, water beside it and water poured into it ran, and
cloth beside a liquid works, so ``alloy.scene.entries.make_entries`` calls ``check_objects`` at assembly with the
template builder's particle positions.

An SPH liquid beside an MPM paste or powder steps since SCR-WP12-MIX (the MPM particles are a particle wall of the
liquid, ``alloy.solvers.sph.particle_walls``; the liquid's force reaches MPM, ``alloy.integration.mpm_proxy``). Two
cases stay refused: a second liquid on the MPM route (``liquid_mpm``: the wall would hold it apart from the SPH liquid
as a solid), and a liquid that starts inside the MPM material (``PREWETTED``, checked with the starting positions like
``FILLED_SOFT_CONTAINER``).

A pair is refused when its two routes sit in two different entries of one scene, whether or not the objects touch:
the coupler pairs whole entries, so the failing path runs either way (a rigid hub with SPH as its one-way follower
does not couple SPH to the MPM entry at all, which is no better). Routes inside one entry are that solver's business
(cloth and a soft body in one VBD entry work).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from alloy.core.errors import RouteError
from alloy.route.types import Route

__all__ = ["FILLED_SOFT_CONTAINER", "MPM_ROUTES", "PREWETTED", "ROD_BESIDE_FILM", "ROUTE_WORDS", "UNSUPPORTED",
           "Unsupported", "UnsupportedPairError", "check_built", "check_entries", "check_objects", "held_by",
           "unsupported"]

MPM_ROUTES = frozenset({Route.GRAIN_MPM, Route.PASTE_MPM, Route.LIQUID_MPM})
VBD_SURFACES = frozenset({Route.CLOTH_VBD, Route.SOFT_VBD})
_RIGID = "make the grains or paste rigid (rigid=True on add_*, or a rigid card)"
_CHAIN = ("keep the rod on the chain route (an unclamped rod of at most 40 segments with an elastic material, or "
          "clamp=\"start\", route=\"chain\")")
_NO_STEP = "their two solvers cannot exchange contacts yet, so the simulation would stop at its first step"
# What a route is, in the words the guides use; the message adds the route name sim.routes() prints.
ROUTE_WORDS: dict[Route, str] = {
    Route.CLOTH_VBD: "cloth", Route.SOFT_VBD: "soft body", Route.ROD_VBD: "elastic rod",
    Route.CLOTH_FILM: "film sheet", Route.GRAIN_MPM: "granular material", Route.PASTE_MPM: "paste",
    Route.LIQUID_MPM: "liquid on the paste solver", Route.LIQUID_SPH: "liquid",
}


@dataclass(frozen=True)
class Unsupported:
    """Any route of ``a`` in one entry with any route of ``b`` in another: ``what`` would happen and the nearest
    supported ``alternative``, both in a user's words (they reach the error), and the sweep ``finding`` it comes
    from (for the tests and this file, not the message). The engine cause, where one is known: in a comment."""

    a: frozenset[Route]
    b: frozenset[Route]
    what: str
    alternative: str
    finding: str

    def matches(self, x: Route, y: Route) -> bool:
        return (x in self.a and y in self.b) or (x in self.b and y in self.a)


# The engine cause of "_NO_STEP": the cloth / rod solver's proxy-particle coupling takes no full-surface soft
# contacts (SolverVBD raised NotImplementedError at the first step).
UNSUPPORTED: tuple[Unsupported, ...] = (
    Unsupported(VBD_SURFACES, MPM_ROUTES, _NO_STEP,
                f"{_RIGID}; a paper or foil sheet on the film route runs beside sand or paste, slowly (8.3 s per "
                "10 ms control step on the CPU); a soft body beside them has no other way but rigid=True",
                "F1: cloth+mpm, mpm+soft"),
    Unsupported(frozenset({Route.ROD_VBD}), MPM_ROUTES, _NO_STEP,
                f"{_RIGID}; or {_CHAIN}, which steps beside sand and paste with the rigid bodies",
                "audit COUPLE-2: wire+sand, wire+dough"),
    Unsupported(frozenset({Route.SOFT_VBD}), frozenset({Route.CLOTH_FILM}), _NO_STEP,
                "rigid=True on the soft body; or a paper material that is not a film sheet (such as paper_a4), "
                "which steps in the soft body's solver (untested beside a soft body, and slow: 0.2 s took 31 s on "
                "the CPU)",
                "F1: paper+soft"),
    Unsupported(frozenset({Route.CLOTH_VBD}), frozenset({Route.CLOTH_FILM}),
                "the two kinds of sheet do not touch each other, so the cloth would fall through the film sheet",
                "a paper material that is not a film sheet (such as paper_a4), which steps with the cloth (untested, "
                "and slow: 0.2 s took 31 s on the CPU)",
                "F4: paper+cloth"),
    Unsupported(frozenset({Route.LIQUID_SPH}), frozenset({Route.LIQUID_MPM}),
                "the second liquid would be a solid wall to the first, so the two would never mix",
                "both liquids on SPH, the default liquid route (no route liquid_mpm on the second liquid's card)",
                "MIX R1: liquid+liquid_mpm"),
)

# Refused only when the pair runs on the proxy coupler (lagged / staggered), where the same NotImplementedError
# raised; a pair the strong (ADMM) coupler takes steps (``_check_rod_beside_film``).
ROD_BESIDE_FILM = Unsupported(
    frozenset({Route.ROD_VBD}), frozenset({Route.CLOTH_FILM}), _NO_STEP,
    f"{_CHAIN}; or a paper material that is not a film sheet (such as paper_a4), which steps in the rod's solver",
    "audit COUPLE-2: wire+paper, wire+metal sheet")


PREWETTED = Unsupported(
    frozenset({Route.LIQUID_SPH}), MPM_ROUTES,
    "a liquid that starts inside a paste or powder is pushed out of it at the first step, not mixed into it: the "
    "paste or powder is a solid wall to the liquid, so there is no soaking or wetting",
    "start the liquid beside the paste or powder, or pour it on top (Scene.add_pour)", "MIX R3: liquid inside MPM")


FILLED_SOFT_CONTAINER = Unsupported(
    VBD_SURFACES, frozenset({Route.LIQUID_SPH}),
    "a liquid that starts inside a free cloth or soft container blew up within 0.4 ms of the first step (a soft "
    "plastic tumbler filled with water at build; the cup alone, water beside it and water poured into it ran)",
    "make the container static (static=True), or pour the liquid in with Scene.add_pour (16 ml poured into the free "
    "cup over 0.8 s ran on the CPU; larger volumes untried); if the liquid is not really in it, start the liquid "
    "outside the soft object's box or with no part of it underneath",
    "F12: liquid in a free soft cup")


class UnsupportedPairError(RouteError):
    """Two routes of one scene that Alloy cannot step together (``UNSUPPORTED``): ``pair`` (the two routes) and
    ``fix`` (the nearest supported alternative)."""

    def __init__(self, message: str, *, pair: tuple[Route, Route], fix: str) -> None:
        super().__init__(message)
        self.pair, self.fix = pair, fix


def unsupported(x: Route, y: Route) -> Unsupported | None:
    """The registry entry refusing routes ``x`` and ``y`` in two entries, or None."""
    return next((u for u in UNSUPPORTED if u.matches(x, y)), None)


def _side(route: Route, objects: Iterable[str]) -> str:
    """``wire (elastic rod) on route rod_vbd``: the objects, what they are, and the route ``sim.routes()`` prints."""
    return f"{_names(objects)} ({ROUTE_WORDS.get(route, 'object')}) on route {route}"


def _names(objects: Iterable[str], most: int = 4) -> str:
    names = list(objects)
    return ", ".join(names[:most]) + (f" and {len(names) - most} more" if len(names) > most else "")


def check_entries(entries: Sequence[Any]) -> None:
    """Raise ``UnsupportedPairError`` for the first two entries (``name``, ``routes``, ``objects``, as
    ``coupler_select.EntryMass``) holding an unsupported route pair; the message names both routes, their entries'
    objects, what would happen and the alternative."""
    for i, e in enumerate(entries):
        for f in entries[i + 1:]:
            for x in sorted(e.routes, key=str):
                for y in sorted(f.routes, key=str):
                    u = unsupported(x, y)
                    if u is not None:
                        raise UnsupportedPairError(
                            f"{_side(x, e.objects)} and {_side(y, f.objects)} are not supported in one scene yet: "
                            f"{u.what}. Nearest supported: {u.alternative}", pair=(x, y), fix=u.alternative)


def check_built(built: Any) -> None:
    """``check_entries`` over a built scene's solver entries, from its route decisions (each object's ``route`` and
    ``entry``; static objects have no entry). A built scene without a route report (hand-made rigs) has nothing to
    check."""
    entries: dict[str, SimpleNamespace] = {}
    decisions = getattr(getattr(built, "report", None), "decisions", None) or {}
    for name, d in decisions.items():
        if not getattr(d, "entry", None):
            continue
        e = entries.setdefault(d.entry, SimpleNamespace(name=d.entry, routes=set(), objects=[]))
        e.routes.add(d.route)
        e.objects.append(name)
    check_entries(list(entries.values()))
    _check_rod_beside_film(built, entries)


def _check_rod_beside_film(built: Any, entries: dict[str, SimpleNamespace]) -> None:
    """``ROD_BESIDE_FILM``: refused only when the two entries pair on the proxy coupler (lagged or staggered); on
    ADMM the pair steps."""
    u = ROD_BESIDE_FILM
    for pair in getattr(getattr(built, "coupler", None), "pairs", ()) or ():
        e, f = entries.get(pair.source), entries.get(pair.destination)
        if e is None or f is None or pair.mode == "admm":
            continue
        for x, y in ((e, f), (f, e)):
            if x.routes & u.a and y.routes & u.b:
                raise UnsupportedPairError(
                    f"{_side(Route.ROD_VBD, x.objects)} and {_side(Route.CLOTH_FILM, y.objects)} are not supported "
                    f"in this scene yet: {u.what}. Nearest supported: {u.alternative}",
                    pair=(Route.ROD_VBD, Route.CLOTH_FILM), fix=u.alternative)


def _free(t: Any) -> bool:
    card = getattr(t.obj, "card", None)
    return card is not None and card.motion == "dynamic"


def _active_points(t: Any, q: np.ndarray, flags: np.ndarray | None) -> np.ndarray:
    r = t.ranges.particle
    x = q[r.start:r.stop]
    if flags is None:
        return x
    from alloy._vendor.newton import ParticleFlags  # noqa: PLC0415

    return x[(flags[r.start:r.stop] & int(ParticleFlags.ACTIVE)) != 0]


def held_by(shell: np.ndarray, x: np.ndarray) -> np.ndarray:
    """bool per point of ``x``: inside the box of the ``shell`` particles with a shell particle under it, closer in x
    and y than the shell's mean spacing (sqrt(box area / count)). A liquid in a cup has the cup's floor under it; one
    under a draped towel or beside a bag has none, and a flat sheet's box has no inside."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    lo, hi = shell.min(axis=0), shell.max(axis=0)
    held = np.all((x > lo) & (x < hi), axis=1)
    if not held.any():
        return held
    idx = np.flatnonzero(held)
    h = max(float(np.sqrt((hi[0] - lo[0]) * (hi[1] - lo[1]) / len(shell))), 1e-6)
    near = cKDTree(x[idx, :2]).sparse_distance_matrix(cKDTree(shell[:, :2]), h, output_type="ndarray")
    under = shell[near["j"], 2] < x[idx[near["i"]], 2]                  # (liquid i, shell j) pairs, j below i
    held[idx] = np.bincount(near["i"][under], minlength=len(idx)) > 0
    return held


def check_objects(objs: Sequence[Any], builder: Any = None) -> None:
    """Raise ``UnsupportedPairError`` for a liquid's start position: ``FILLED_SOFT_CONTAINER`` (an SPH liquid with an
    active particle held by a free (dynamic) cloth or soft body: ``held_by``, in its particles' box with a part of it
    underneath) or ``PREWETTED`` (an active SPH particle within touching distance, r_liquid + r_mpm, of an MPM
    particle), as placed in the template ``builder`` (``particle_q``, ``particle_flags``, ``particle_radius``).
    ``objs`` are the scene's template objects (``name``, ``decision.route``, ``obj.card``, ``ranges.particle``). Liquid
    still to be poured is not active. The template is world 0 as placed: with per-world start positions (``pos`` of
    shape (worlds, 3)) other worlds may differ."""
    liquids = [t for t in objs if t.decision.route == Route.LIQUID_SPH and t.ranges.particle]
    if builder is None or not liquids:
        return
    q = np.asarray(builder.particle_q, dtype=np.float64).reshape(-1, 3)
    flags = np.asarray(builder.particle_flags, dtype=np.int64) if len(builder.particle_flags) else None
    points = {t.name: _active_points(t, q, flags) for t in liquids}
    _check_filled_soft(objs, liquids, points, q)
    _check_prewetted(objs, liquids, points, q, np.asarray(getattr(builder, "particle_radius", ()), dtype=np.float64))


def _check_filled_soft(objs: Sequence[Any], liquids: Sequence[Any], points: dict, q: np.ndarray) -> None:
    rule = FILLED_SOFT_CONTAINER
    for s in (t for t in objs if t.decision.route in rule.a and _free(t) and t.ranges.particle):
        p = _active_points(s, q, None)
        for t in liquids:
            held = int(held_by(p, points[t.name]).sum())
            if held:
                x, y = s.decision.route, t.decision.route
                raise UnsupportedPairError(
                    f"{t.name} ({y}) starts inside {s.name} ({x}, free): {held} liquid particles in its box with a "
                    f"part of it underneath. Not supported in one scene yet: {rule.what}. Nearest supported: "
                    f"{rule.alternative}", pair=(x, y), fix=rule.alternative)


def _radius(t: Any, radius: np.ndarray) -> float:
    r = t.ranges.particle
    return float(radius[r.start:r.stop].max()) if radius.size >= r.stop else 0.0


def _check_prewetted(objs: Sequence[Any], liquids: Sequence[Any], points: dict, q: np.ndarray,
                     radius: np.ndarray) -> None:
    from scipy.spatial import cKDTree  # noqa: PLC0415

    rule = PREWETTED
    for m in (t for t in objs if t.decision.route in rule.b and t.ranges.particle):
        tree = cKDTree(q[m.ranges.particle.start:m.ranges.particle.stop])
        for t in liquids:
            x = points[t.name]
            reach = _radius(t, radius) + _radius(m, radius)
            if not len(x) or reach <= 0.0:
                continue
            wet = int(np.isfinite(tree.query(x, distance_upper_bound=reach)[0]).sum())
            if wet:
                a, b = t.decision.route, m.decision.route
                raise UnsupportedPairError(
                    f"{t.name} ({a}) starts inside {m.name} ({b}): {wet} liquid particles within {reach * 1e3:.3g} mm "
                    f"of its particles. Not supported in one scene yet: {rule.what}. Nearest supported: "
                    f"{rule.alternative}", pair=(a, b), fix=rule.alternative)
