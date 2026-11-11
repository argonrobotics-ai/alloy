# SPDX-License-Identifier: Apache-2.0
"""Jittered start placements: each rigid group of free objects gets an offset within +-``pos`` in x and y and a yaw
within +-``yaw``, drawn until the moved start state passes two checks (Newton's contact probe, ``alloy.check``):

- no new overlap: every owner pair with a moved object overlaps at most as much as at the start, plus 0.1 mm
  (``alloy.check``'s overlap warning level);
- still supported: an object that rested on something outside its group (a table, a bin floor, the ground) still
  touches it. An object pushed over a table edge, or off a narrow stand, is redrawn.

Free objects touching a robot at the start (held in a gripper) stay where they are.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.scene.variants.base import Base, groups, touching
from alloy.scene.variants.xform import Move

log = logging.getLogger("alloy.scene")

OVERLAP_TOL = 1.0e-4    # [m] alloy.check.overlap.OVERLAP_WARN_M
REST_COS = 0.7          # |contact normal z| above this: the two rest on each other (not side by side)


@dataclass
class Placer:
    """Draws and checks the moves of ``groups`` on ``base``'s start state."""

    base: Base
    groups: list[list[str]]
    pos: float
    yaw: float
    base_depth: dict[tuple[str, str], float] = field(default_factory=dict)
    supports: dict[str, set[str]] = field(default_factory=dict)
    state: Any = None

    @classmethod
    def make(cls, base: Base, free: list[str], pos: float, yaw_deg: float) -> Placer:
        """A placer for the free objects ``free`` (``base.free_objects``)."""
        state = base.state()
        contacts = base.probe.collide(state)
        touch = touching(base, state, contacts)
        robots = _robots(base)
        held = {a if b in robots else b for a, b, _ in touch if {a, b} & robots} & set(free)
        if held:
            log.info("variants: %s touch a robot at the start and stay where they are", sorted(held))
        movable = [n for n in free if n not in held]
        placer = cls(base, groups(movable, base), float(pos), math.radians(float(yaw_deg)), state=state)
        placer.base_depth = _depths(base, state, contacts)
        placer.supports = placer._supports(touch)
        return placer

    def draw(self, rng: np.random.Generator) -> list[Move]:
        """One move per group: offset uniform in +-pos (x, y), yaw uniform in +-yaw, about the group's centre."""
        out = []
        for g in self.groups:
            dx, dy = rng.uniform(-self.pos, self.pos, 2) if self.pos > 0 else (0.0, 0.0)
            yaw = float(rng.uniform(-self.yaw, self.yaw)) if self.yaw > 0 else 0.0
            center = np.mean([self.base.poses[n][:3] for n in g], axis=0)
            out.append(Move(tuple(g), (float(dx), float(dy)), yaw, tuple(float(c) for c in center)))
        return out

    def problem(self, moves: list[Move]) -> str | None:
        """Why the moved start state is refused (module docstring), or None when it passes."""
        state = self.state_for(moves)
        contacts = self.base.probe.collide(state)
        moved = {n for mv in moves for n in mv.names}
        for (a, b), depth in _depths(self.base, state, contacts).items():
            allowed = self.base_depth.get((a, b), 0.0) + OVERLAP_TOL
            if (a in moved or b in moved) and depth > allowed:
                return f"{a} overlaps {b} by {depth * 1e3:.2f} mm"
        resting = _resting(touching(self.base, state, contacts))
        for name in sorted(moved):
            lost = self.supports.get(name, set()) - resting.get(name, set())
            if lost:
                return f"{name} no longer rests on {', '.join(sorted(lost))}"
        return None

    def place(self, rng: np.random.Generator, max_tries: int) -> tuple[list[Move], int]:
        """Moves that pass ``problem`` and the number of draws taken; raises SceneError after ``max_tries``."""
        why = None
        for tries in range(1, max_tries + 1):
            moves = self.draw(rng)
            why = self.problem(moves)
            if why is None:
                return moves, tries
        from alloy.api.errors import SceneError  # noqa: PLC0415

        raise SceneError(f"no placement within +-{self.pos * 1e3:.1f} mm / +-{math.degrees(self.yaw):.1f} deg in "
                         f"{max_tries} draws (last: {why})", "lower pos / yaw_deg, raise max_tries, or give the "
                         "objects room in the base scene", field="pos")

    def state_for(self, moves: list[Move]) -> Any:
        """The start state with ``moves`` applied (one shared buffer: read it before the next call)."""
        bq, pq = self.base.body_q.copy(), self.base.particle_q.copy()
        idx = self.base.index
        for mv in moves:
            for name in mv.names:
                i = idx.names.index(name)
                b = np.flatnonzero(idx.body_obj == i)
                if b.size:
                    bq[b, :3], bq[b, 3:7] = mv.points(bq[b, :3]), mv.quats(bq[b, 3:7])
                p = np.flatnonzero(idx.particle_obj == i)
                if p.size:
                    pq[p] = mv.points(pq[p])
        if len(bq):
            self.state.body_q.assign(bq.astype(np.float32))
        if len(pq):
            self.state.particle_q.assign(pq.astype(np.float32))
        return self.state

    def _supports(self, touch: list[tuple[str, str, np.ndarray]]) -> dict[str, set[str]]:
        group_of = {n: k for k, g in enumerate(self.groups) for n in g}
        out: dict[str, set[str]] = {}
        for name, others in _resting(touch).items():
            if name in group_of:
                out[name] = {o for o in others if group_of.get(o, -1) != group_of[name]}
        return out


def _robots(base: Base) -> set[str]:
    return {n for n, m in base.index.motion.items() if m == "robot"}


def _depths(base: Base, state: Any, contacts: Any) -> dict[tuple[str, str], float]:
    from alloy.check.contacts import depths  # noqa: PLC0415

    return {(p.a, p.b): p.depth for p in depths(base.built.model, state, base.index, (), contacts)}


def _resting(touch: list[tuple[str, str, np.ndarray]]) -> dict[str, set[str]]:
    """{owner: owners it rests on or carries} from touching rows with a near-vertical normal."""
    out: dict[str, set[str]] = {}
    for a, b, n in touch:
        if abs(float(n[2])) >= REST_COS * float(np.linalg.norm(n)):
            out.setdefault(a, set()).add(b)
            out.setdefault(b, set()).add(a)
    return out


def edit_poses(scene: Any, moves: list[Move], base: Base) -> None:
    """Write ``moves`` into ``scene``'s items: the start pose (or the scene card's placement pose) and every
    per-world start position."""
    for mv in moves:
        for name in mv.names:
            item = scene.items[name]
            new = mv.pose(base.poses[name])
            if item.get("raw_placement") is not None:
                item["raw_placement"]["pose"] = new
            if item.get("pose") is not None:
                item["pose"] = new
            if item.get("per_world_pos"):
                item["per_world_pos"] = mv.points(np.asarray(item["per_world_pos"], np.float64)).tolist()


__all__ = ["OVERLAP_TOL", "Placer", "edit_poses"]
