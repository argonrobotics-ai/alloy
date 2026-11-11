# SPDX-License-Identifier: Apache-2.0
"""Overlap at load: shapes that interpenetrate in the initial state, per object pair, with the pose edit that
separates them ("mug overlaps table by 3.2 mm at start: lift mug by 4 mm").

Levels (D3 limits, ``alloy.core.defaults``): deeper than ``PEN_LIMIT_M`` (1 mm) is an error, the solver would push the
objects apart in the first steps; deeper than ``OVERLAP_WARN_M`` (0.1 mm) warns; shallower is a resting contact.
Randomized placements are checked in every world, and a finding names the worlds that overlap. Overlaps the
simulation never acts on are info only: two parts of one object (an articulated robot's own links, whose collision
filters live in its robot file), two objects that are both static or kinematic, and two attached objects.

Example::

    >>> found, metrics = overlap_findings(built, index)        # doctest: +SKIP
    >>> found[0].fix
    'lift peg_d12 by 4 mm (pose z 0.04 -> 0.044 m)'
"""

from __future__ import annotations

import logging
from typing import Any

from alloy.check.advice import Move, mm, separate, worlds_text
from alloy.check.contacts import GROUND, ContactProbe, PairDepth
from alloy.check.objects import ObjectIndex
from alloy.check.report import Finding
from alloy.core.defaults import PEN_LIMIT_M

log = logging.getLogger("alloy.check")

OVERLAP_WARN_M = 1.0e-4


def overlap_findings(built: Any, index: ObjectIndex, state: Any = None, probe: ContactProbe | None = None,
                     when: str = "at start") -> tuple[list[Finding], dict[str, Any]]:
    """Findings and metrics for the overlaps of ``state`` (default: the model's initial state); ``when`` words the
    time in messages ("at start", "now").

    Example::

        >>> found, metrics = overlap_findings(built, index)  # doctest: +SKIP
        >>> metrics["max_overlap_m"] < 1e-4
        True
    """
    if state is None:
        state = initial_state(built.model)
    probe = probe if probe is not None else ContactProbe(built)
    rows = probe.depths(state, index, limits=(OVERLAP_WARN_M, PEN_LIMIT_M))
    found = [f for f in (_finding(p, index, when) for p in rows) if f is not None] + probe_findings(probe, "overlap")
    metrics = {"max_overlap_m": rows[0].depth if rows else 0.0,
               "pairs": [{"a": p.a, "b": p.b, "depth_m": p.depth, "world": p.world} for p in rows[:20]]}
    log.info("check overlap: %d touching pairs, deepest %s", len(rows), mm(rows[0].depth) if rows else "none")
    return found, metrics


def probe_findings(probe: ContactProbe, check: str = "overlap") -> list[Finding]:
    """One warning, once per probe, when the check's own contact probe lost contacts it could not make room for
    (``probe.dropped``): the pair depths of ``check`` are incomplete. The simulation's contacts are not affected.

    Example::

        >>> probe_findings(ContactProbe(built))  # doctest: +SKIP
        []
    """
    n = probe.dropped
    if n is None or probe.reported:
        return []
    probe.reported = True
    worlds = max(int(probe.model.world_count or 0), 1)
    what = (f"{n['triangle_pairs']:,} mesh triangle pairs for {n['triangle_slots']:,} slots"
            if n["triangle_pairs"] > n["triangle_slots"] else
            f"{n['rigid_contacts']:,} contacts for {n['rigid_slots']:,} slots"
            if n["rigid_contacts"] > n["rigid_slots"] else "contact candidates past its reducer")
    fewer = f"check fewer worlds (worlds={max(worlds // 4, 1)})" if worlds > 1 else "check fewer objects at a time"
    return [Finding(check, "warn", f"pair depths are incomplete: the check's own contact probe dropped contacts "
                    f"({what}, {worlds} worlds); the simulation's contacts are not affected",
                    f"{fewer}, or give the scanned meshes a simpler collision shape (collision='convex')",
                    value=float(max(n["triangle_pairs"], n["rigid_contacts"])), unit="contacts")]


def initial_state(model: Any) -> Any:
    """``model.state()`` with forward kinematics applied (as Simulation starts).

    Example::

        >>> state = initial_state(built.model)  # doctest: +SKIP
        >>> len(state.body_q) == built.model.body_count
        True
    """
    from alloy._vendor import newton  # noqa: PLC0415

    state = model.state()
    if model.joint_count:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


def _finding(p: PairDepth, index: ObjectIndex, when: str = "at start") -> Finding | None:
    level = "error" if p.depth > PEN_LIMIT_M else "warn" if p.depth > OVERLAP_WARN_M else None
    if level is None:
        return None
    limit = PEN_LIMIT_M if level == "error" else OVERLAP_WARN_M
    worlds = p.worlds_past(limit)
    where = worlds_text(worlds, index.world_count)
    label = {n: f"particles of {n}" if n == p.particles else n for n in (p.a, p.b)}
    msg = f"{label[p.a]} and {label[p.b]} overlap by {mm(p.depth)} {when}" + (f" {where}" if where else "")
    passive = _passive_reason(p, index)
    if passive is not None:
        return Finding("overlap", "info", f"{msg} ({passive[0]})", passive[1],
                       objects=tuple(dict.fromkeys(n for n in (p.a, p.b) if n != GROUND)), value=p.depth,
                       limit=limit, unit="m")
    move = separate(index, p.a, p.b, p.normal, p.depth)
    fix = _fix_text(move, p, index, worlds)
    objs = tuple(n for n in (p.a, p.b) if n != GROUND)
    return Finding("overlap", level, msg, fix, objects=objs, value=p.depth, limit=limit, unit="m",
                   worlds=tuple(worlds) if len(worlds) < index.world_count else (),
                   suggestion=move.suggestion(index) if move is not None else None)


def _passive_reason(p: PairDepth, index: ObjectIndex) -> tuple[str, str] | None:
    """(why the simulation does not act on this overlap, what to do) or None when it does."""
    if p.a == p.b:
        return (f"two parts of {p.a}", f"nothing, unless {p.a}'s own parts should collide: its collision filters "
                f"(robot file or card) decide")
    if not any(index.movable(n) or index.motion.get(n) == "robot" for n in (p.a, p.b)):
        return ("both static or kinematic, no contact force", "nothing for the simulation; move one if the scene "
                "should look right")
    if index.is_attached(p.a, p.b):
        return ("attached to each other", "nothing, unless the attachment anchor is wrong")
    return None


def _fix_text(move: Move | None, p: PairDepth, index: ObjectIndex, worlds: list[int]) -> str:
    if move is None:
        return (f"move {p.a} or {p.b} apart by {mm(p.depth)} or more (both are static or kinematic, so the check "
                f"cannot pick one)")
    text = move.verb()
    if move.pose is not None:
        old = index.poses[move.name]
        axis = max(range(3), key=lambda i: abs(move.direction[i]))
        text += f" (pose {'xyz'[axis]} {old[axis]:.4g} -> {move.pose[axis]:.4g} m)"
    some = 0 < len(worlds) < index.world_count
    if some and move.name in index.placement:
        text += (f"; only some worlds overlap, so a randomize range on {move.name} reaches into {_other(move, p)}: "
                 f"narrow it instead if the nominal pose is right")
    return text


def _other(move: Move, p: PairDepth) -> str:
    return p.b if move.name == p.a else p.a


__all__ = ["OVERLAP_WARN_M", "initial_state", "overlap_findings"]
