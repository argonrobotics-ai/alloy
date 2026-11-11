# SPDX-License-Identifier: Apache-2.0
"""Geometric contact depths of one state, per object pair: Newton's collision pipeline run once on the host's
request (not the solver's own contacts), so the depths are the shapes' true overlap at that instant.

``ContactProbe(built).depths(state, index)`` returns ``PairDepth`` rows: for every pair of owners that touch, the
deepest penetration, the world it is in, the contact normal there (from the first owner toward the second) and the
worlds past a limit. Rigid shape pairs and particle-vs-shape (soft) contacts are both covered; particles of
contact-free solvers (SPH liquids, which keep their own walls) get no soft contacts, as in the simulation.

Example::

    >>> rows = ContactProbe(built).depths(sim.state, index)          # doctest: +SKIP
    >>> rows[0].a, rows[0].b, round(rows[0].depth * 1e3, 1)
    ('mug', 'table', 3.2)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.check.objects import ObjectIndex

log = logging.getLogger("alloy.check")

BUDGET_MIN_WORLDS = 16   # from this many worlds a rigid-only scene's probe is sized from MuJoCo's capacities
TRIANGLE_PAIRS_PER_CONTACT = 2  # triangle-pair slots per contact slot (G1 with hands used 45 pairs for 31 contacts)

COUNT_MARGIN = 1.25      # a counted rebuild's slots over the count the overflowing probe reported
COUNTED_REBUILDS = 3     # rebuilds from reported counts per probe (a count can stop short of the need: the stages
                         # before a full buffer cap what reaches it, and more triangle pairs make more contacts)
MODEL_CAPACITY = ("rigid_contact_max", "soft_contact_max", "_contact_capacity_initialized")  # restored after a build
GROUND = "ground"     # owner label of shapes no placement owns (a ground plane, a world-attached shape)


@dataclass
class PairDepth:
    """Deepest contact between owners ``a`` and ``b`` (``a`` sorted first), over all worlds.

    Example::

        >>> p = PairDepth("mug", "table", 0.003, 1, (0.0, 0.0, 1.0))
        >>> p.a, p.b, p.depth
        ('mug', 'table', 0.003)
    """

    a: str
    b: str
    depth: float                    # [m] > 0 = overlap
    world: int                      # world of the deepest contact
    normal: tuple[float, float, float]   # unit, world frame, from a toward b at the deepest contact
    worlds_over: dict[float, list[int]] = field(default_factory=dict)   # limit -> worlds deeper than it
    particles: str | None = None    # for a particle-vs-shape contact: the owner of the particles (a or b)

    def worlds_past(self, limit: float) -> list[int]:
        """Worlds whose contact is deeper than ``limit`` (empty when ``limit`` was not asked for in ``depths``).

        Example::

            >>> p = PairDepth("mug", "table", 0.003, 1, (0.0, 0.0, 1.0), {1e-4: [1, 3]})
            >>> p.worlds_past(1e-4), p.worlds_past(5e-3)
            ([1, 3], [])
        """
        return self.worlds_over.get(limit, [])


class ContactProbe:
    """A Newton collision pipeline (``alloy.scene.steppers.NewtonContacts``) sized once for ``built`` and reused for
    every probe; contact-free (SPH) particles get no soft candidates, as in the wiring's fallback buffer.

    Example::

        >>> probe = ContactProbe(built)  # doctest: +SKIP
        >>> probe.depths(sim.state, index)[0].b
        'table'
    """

    def __init__(self, built: Any) -> None:
        self.model = built.model
        self.built = built
        self.fell_back = False          # True once a sized probe overflowed and was rebuilt at Newton's own size
        self.overflowed = False         # True once the last rebuild overflowed as well (warned once)
        self.counted = 0                # rebuilds sized from the counts an overflow reported (COUNTED_REBUILDS)
        self.dropped: dict[str, int] | None = None   # ``need()`` at the last overflow nothing fixed: depths incomplete
        self.reported = False           # the check's finding for ``dropped`` was made (alloy.check.overlap)
        self.layer = self._make_layer(probe_budget(self.model))
        self.contacts = self.layer.contacts()

    def _make_layer(self, budget: Any) -> Any:
        from alloy.integration.stepper import dynamic_entries  # noqa: PLC0415
        from alloy.integration.wiring import contact_free_particles  # noqa: PLC0415
        from alloy.scene.steppers import NewtonContacts  # noqa: PLC0415

        free = contact_free_particles(dynamic_entries(self.built))
        n = int(self.model.particle_count or 0)
        particles = None if free.size == 0 else np.setdiff1d(np.arange(n), free)
        keep = {k: getattr(self.model, k, None) for k in MODEL_CAPACITY}
        try:    # the pipeline publishes its capacity on the shared model: a later (re)build would size itself from it
            return NewtonContacts(self.model, particles=particles, budget=budget)
        finally:
            for k, v in keep.items():
                setattr(self.model, k, v)

    def collide(self, state: Any) -> Any:
        """Run the collision pipeline once at ``state`` and return its contacts buffer.

        Example::

            >>> contacts = probe.collide(sim.state)  # doctest: +SKIP
            >>> contacts.rigid_contact_count.numpy().shape
            (1,)
        """
        self.layer.reset_overflow()                         # the counter is sticky: read this collide's losses only
        self.layer.collide(state, self.contacts, dt=1e-3)
        if self._lost():
            self._grow(state)
        return self.contacts

    def _lost(self) -> bool:
        return int(self.layer.overflow.numpy()[0]) > 0

    def _grow(self, state: Any) -> None:
        """Contacts were lost: Newton's own size, then sizes from the counts it reports; else say so (``dropped``)."""
        if not self.fell_back:
            self._rebuild_unsized(state)
        while self._lost() and self.counted < COUNTED_REBUILDS and self._rebuild_counted(state):
            pass
        if not self._lost():
            return
        self.dropped = self.need()
        if not self.overflowed:                             # say it once; the check adds a finding (overlap.py)
            self.overflowed = True
            msg = (f"the contact probe of sim.check() still overflows ({_need_text(self.dropped)}): some pair depths "
                   "are missing from this check (the simulation's own contacts are not affected); check fewer worlds")
            log.warning(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def need(self) -> dict[str, int]:
        """What the last collide asked of the probe's buffers and what they hold: ``triangle_pairs`` /
        ``triangle_slots``, ``rigid_contacts`` / ``rigid_slots`` (all worlds together), and ``reducer`` (1: the
        contact reducer dropped candidates).

        Example::

            >>> n = probe.need()  # doctest: +SKIP
            >>> n["rigid_contacts"] <= n["rigid_slots"]
            True
        """
        from alloy.coupling.proxy_contacts import _triangle_capacity  # noqa: PLC0415

        pipeline, c = self.layer.pipeline, self.contacts
        tri = getattr(getattr(pipeline, "narrow_phase", None), "triangle_pairs_count", None)
        counters = getattr(c, "contact_counters", None)
        return {"triangle_pairs": 0 if tri is None else int(tri.numpy()[0]),
                "triangle_slots": _triangle_capacity(pipeline),
                "rigid_contacts": int(c.rigid_contact_count.numpy()[0]), "rigid_slots": int(c.rigid_contact_max),
                "reducer": 0 if counters is None else int(counters.numpy()[2] > 0)}

    def _rebuild_counted(self, state: Any) -> bool:
        """Rebuild with slots for the counts the overflowing collide reported (rigid-only scenes: a budget would
        change a soft scene's candidates). False when there is nothing to size from or the build fails."""
        from alloy.scene.steppers import ContactBudget  # noqa: PLC0415

        n, worlds = self.need(), max(int(self.model.world_count or 0), 1)
        tri = _grown(n["triangle_pairs"], n["triangle_slots"], bool(n["reducer"]))
        rigid = _grown(n["rigid_contacts"], n["rigid_slots"], False)
        if int(self.model.particle_count or 0) or (tri == n["triangle_slots"] and rigid == n["rigid_slots"]):
            return False
        budget = ContactBudget(triangle_pairs=-(-tri // worlds) if tri else None, rigid_contacts=-(-rigid // worlds))
        self.counted += 1
        try:
            layer = self._make_layer(budget)
        except Exception as exc:  # noqa: BLE001 - device memory: keep the probe that exists and report its loss
            log.warning("check probe: no rebuild for %s (%s: %s)", _need_text(n), type(exc).__name__,
                        str(exc).strip().splitlines()[-1][:160] if str(exc).strip() else "")
            self.counted = COUNTED_REBUILDS
            return False
        log.info("check probe: rebuilt for %s", _need_text(n))
        self.layer, self.contacts = layer, layer.contacts()
        self.layer.collide(state, self.contacts, dt=1e-3)
        return True

    def _rebuild_unsized(self, state: Any) -> None:
        """The sized buffers overflowed (contacts or triangle pairs lost): fail closed, redo at Newton's own size."""
        msg = ("the contact probe of sim.check() overflowed its buffers (sized from MuJoCo's contact capacity) and "
               "was rebuilt at Newton's automatic size, which needs much more device memory with many worlds; "
               "nothing to change in the scene; if the device runs out of memory, check fewer worlds")
        log.warning(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        self.fell_back = True
        self.layer = self._make_layer(None)
        self.contacts = self.layer.contacts()
        self.layer.reset_overflow()
        self.layer.collide(state, self.contacts, dt=1e-3)

    def depths(self, state: Any, index: ObjectIndex, limits: tuple[float, ...] = ()) -> list[PairDepth]:
        """Deepest overlap per owner pair at ``state``; see ``depths``.

        Example::

            >>> rows = probe.depths(sim.state, index, limits=(1e-4,))  # doctest: +SKIP
            >>> rows[0].depth > 0
            True
        """
        return depths(self.model, state, index, limits, self.collide(state))


def _grown(count: int, slots: int, reducer: bool) -> int:
    """Slots for a buffer that held ``slots`` and was asked for ``count``: when it was over, the count plus
    COUNT_MARGIN and at least twice the slots; twice the slots when only the reducer (sized from the triangle pairs)
    dropped; else unchanged."""
    if count > slots:
        return max(int(np.ceil(count * COUNT_MARGIN)), 2 * slots)
    return 2 * slots if reducer else slots


def _need_text(n: dict[str, int]) -> str:
    """``need()`` in words, the buffers over their size only (or the reducer)."""
    parts = [f"{n[k + suffix]:,} {label} for {n[k + '_slots']:,} slots"
             for k, suffix, label in (("triangle", "_pairs", "triangle pairs"), ("rigid", "_contacts", "contacts"))
             if n[k + suffix] > n[k + "_slots"]]
    if n["reducer"] and not parts:
        parts.append("the contact reducer dropped candidates")
    return ", ".join(parts) or "contacts lost"


def probe_budget(model: Any) -> Any:
    """Per-world capacities for the probe of a rigid-only scene with ``BUDGET_MIN_WORLDS`` or more worlds (MuJoCo's
    own contacts per world, ``default_capacities``); None keeps Newton's automatic size (small or soft scenes: its
    soft candidates would change with a budget). Newton's size is 144,458 contact slots per world for a G1 with
    hands, 21.7 MiB of GPU memory per world.

    Example::

        >>> probe_budget(built.model).rigid_contacts  # doctest: +SKIP
        2768
    """
    from alloy.scene.entries import default_capacities  # noqa: PLC0415
    from alloy.scene.steppers import ContactBudget  # noqa: PLC0415

    worlds = int(model.world_count or 0)
    if worlds < BUDGET_MIN_WORLDS or int(model.particle_count or 0):
        return None
    sw = model.shape_world.numpy()
    per_world = int(np.count_nonzero(sw >= 0)) // worlds + int(np.count_nonzero(sw < 0))
    contacts = default_capacities(per_world)[0]
    return ContactBudget(triangle_pairs=TRIANGLE_PAIRS_PER_CONTACT * contacts, rigid_contacts=contacts)


def depths(model: Any, state: Any, index: ObjectIndex, limits: tuple[float, ...], contacts: Any) -> list[PairDepth]:
    """Deepest overlap per owner pair of ``contacts`` (collided at ``state``), positive depths only, deepest first.

    Example::

        >>> rows = depths(built.model, state, index, (), contacts)  # doctest: +SKIP
        >>> rows[0].depth > 0
        True
    """
    c = contacts
    body_q = state.body_q.numpy() if model.body_count else np.zeros((0, 7))
    rows = _rigid_rows(model, c, body_q, index) + _soft_rows(model, c, state, body_q, index)
    pairs: dict[tuple[str, str], PairDepth] = {}
    for a, b, depth, world, normal, owner in rows:
        if a > b:
            a, b, normal = b, a, -normal
        p = pairs.get((a, b))
        if p is None or depth > p.depth:
            keep = p.worlds_over if p is not None else {}
            pairs[(a, b)] = p = PairDepth(a, b, float(depth), int(world), _unit(normal), keep, owner)
        for lim in limits:
            if depth > lim and world not in p.worlds_over.setdefault(lim, []):
                p.worlds_over[lim].append(int(world))
    out = sorted(pairs.values(), key=lambda p: -p.depth)
    log.debug("check contacts: %d rows, %d touching pairs", len(rows), len(out))
    return out


def _owner(index: ObjectIndex, kind: str, i: int) -> str:
    return index.name_of(kind, i) or GROUND


def _rigid_rows(model: Any, c: Any, body_q: np.ndarray, index: ObjectIndex) -> list[tuple]:
    n = int(c.rigid_contact_count.numpy()[0]) if c.rigid_contact_max else 0
    if n == 0:
        return []
    s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
    p0, p1 = c.rigid_contact_point0.numpy()[:n], c.rigid_contact_point1.numpy()[:n]
    nrm = c.rigid_contact_normal.numpy()[:n]
    m0, m1 = c.rigid_contact_margin0.numpy()[:n], c.rigid_contact_margin1.numpy()[:n]
    shape_body, shape_world = model.shape_body.numpy(), model.shape_world.numpy()
    x0 = _to_world(body_q, shape_body[s0], p0)
    x1 = _to_world(body_q, shape_body[s1], p1)
    depth = -(np.einsum("ij,ij->i", nrm, x1 - x0) - (m0 + m1))
    out = []
    for k in np.flatnonzero(depth > 0.0):
        w = max(int(shape_world[s0[k]]), int(shape_world[s1[k]]))
        out.append((_owner(index, "shape", int(s0[k])), _owner(index, "shape", int(s1[k])), float(depth[k]), w,
                    nrm[k].astype(np.float64), None))
    return out


def _soft_rows(model: Any, c: Any, state: Any, body_q: np.ndarray, index: ObjectIndex) -> list[tuple]:
    n = int(c.soft_contact_count.numpy()[0]) if getattr(c, "soft_contact_max", 0) else 0
    if n == 0 or not model.particle_count:
        return []
    part, shp = c.soft_contact_particle.numpy()[:n], c.soft_contact_shape.numpy()[:n]
    ok = (part >= 0) & (shp >= 0)
    part, shp = part[ok], shp[ok]
    pos, nrm = c.soft_contact_body_pos.numpy()[:n][ok], c.soft_contact_normal.numpy()[:n][ok]
    xs = _to_world(body_q, model.shape_body.numpy()[shp], pos)
    pq, rad = state.particle_q.numpy()[part], model.particle_radius.numpy()[part]
    depth = -(np.einsum("ij,ij->i", nrm, pq - xs) - rad)
    pw = model.particle_world.numpy()
    out = []
    for k in np.flatnonzero(depth > 0.0):
        owner = _owner(index, "particle", int(part[k]))
        out.append((_owner(index, "shape", int(shp[k])), owner, float(depth[k]), int(pw[part[k]]),
                    nrm[k].astype(np.float64), owner))
    return out


def _to_world(body_q: np.ndarray, bodies: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Body-frame points to world (body -1: already world)."""
    out = np.asarray(points, np.float64).copy()
    has = bodies >= 0
    if not np.any(has):
        return out
    q = body_q[bodies[has]].astype(np.float64)
    out[has] = q[:, :3] + _rotate(q[:, 3:7], out[has])
    return out


def _rotate(quat_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    xyz, w = quat_xyzw[:, :3], quat_xyzw[:, 3:4]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _unit(v: np.ndarray) -> tuple[float, float, float]:
    n = float(np.linalg.norm(v))
    u = v / n if n > 0 else np.array([0.0, 0.0, 1.0])
    return (float(u[0]), float(u[1]), float(u[2]))


__all__ = ["GROUND", "ContactProbe", "PairDepth", "depths"]
