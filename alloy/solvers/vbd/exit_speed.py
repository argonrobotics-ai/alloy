# SPDX-License-Identifier: Apache-2.0
"""Exit-speed cap for cloth particles leaving a body contact (SCR-WP11-KICK; SC4's ClothFold pinch INVALID, 2026-11-10).

What goes wrong. VBD solves a step implicitly. A particle that starts a step inside a body record's contact distance is
put back near the surface within the step, and its velocity is then read off the step's displacement, v = (x - x_start)
/ dt. So the depth it started inside becomes speed: it leaves at about depth / dt whatever the stiffness. A 60 mg
particle 0.12 mm inside a static box at dt 20 us leaves at 3.4 m/s on make_vbd's log barrier (5.2 m/s on Newton's
linear penalty), 0.5 mm deep at 18.7 m/s, 0.8 mm at 31.6 m/s ([internal note] notes/wp11-kick/exit_probe.py). Contact damping
acts only while a particle approaches. In ClothFold's coupled steps (19.5-39 us) a towel corner particle starting a step
inside a closing finger's record left at 14 m/s and a neighbour at 33 m/s: SPEED_FAIL froze about 40 % of the worlds at
the pinch (SC4's probe dumps, [internal note] notes/sc4-kick). Rigid engines bound the same effect (PhysX's maximum
depenetration velocity, Bullet's split impulse).

The cap (split impulse). After the step, take a record whose contact point started the step inside its contact
distance by a depth d0: the particle's radius plus the shape's margin, at the body's pose over the step, as Newton
measures it. The corners of edge and face records count too. The contact point's outward motion along the record's
normal, relative to the body point, up to d0, is the depth the step resolved. That motion must not become speed: past
``v_cap``, its share is taken off the particle's outward normal velocity.
- Particle record: the cut is resolved / dt - v_cap.
- Edge / face record: the corner's share of the push, w_k / m_k / sum_j w_j^2 / m_j.
- The cut never takes the particle's outward normal speed relative to the body below v_cap.
- Its position stays as solved: the penetration is still resolved; only the speed the resolution gave it goes. Its
  tangential velocity is untouched.
- A particle that was already leaving keeps its speed: the start depth bounds the cut, so a resting particle that a
  step lifts off at 2 m/s loses at most its resting depth / dt.
- The step's contacts, friction and force harvest are untouched. The coupler's record-momentum feedback reads the
  capped velocity, so a robot body gets back what the particle took; a body VBD integrates itself keeps the push it
  got in the step.
- The cut never takes more than the particle's own outward speed: a particle resting on a pad that pulls away
  faster than the cap is not pulled after it.
- A particle in several such records is cut by the record that cuts most (ties: the lowest record), so the result
  does not depend on record order, and a particle started deeper in one finger's record and thrown out of the other
  finger's is capped (AUDIT-STACK 2026-11-10).
- Records that start the step outside their contact distance are left alone: an impact still stops in one step.
On ``exit_probe.py``'s 60 mg particle 0.5 mm deep (dt 20 us), the cap leaves it at 1.0 m/s instead of 18.7 m/s; the
next step resolves the rest of the depth at that speed.

Cost: it takes energy out of particles only (the elastic energy of a depth the step should not have started with).
Not covered: a particle squeezed sideways out of a wedge between two records leaves along the wedge, not along either
normal; its tangential speed is not capped.

Opt-in: ``body_particle_exit_speed`` in the VBD entry's kwargs (``make_vbd``): a cap in m/s, or True for
``DEFAULT_SPEED``. Or ``ALLOY_EXIT_SPEED=<m/s>`` in the environment, for every VBD entry of a process (the kwarg wins).
The cap keeps no state across steps, so there is nothing to snapshot and nothing for coupled passes. Off, nothing here
runs. ``ExitSpeedCap.capped`` counts the particles cut so far.
"""

from __future__ import annotations

import os
from typing import Any

import warp as wp

from alloy.solvers.film.grip import ContactView, _body_velocity, _pose
from alloy.solvers.vbd.records import RecordViews
from alloy.solvers.vbd.stick import NO_OWNER, _live, _weight

EXIT_KEY = "body_particle_exit_speed"
EXIT_ENV = "ALLOY_EXIT_SPEED"     # a cap in m/s for every VBD entry of a process ("" or "0": off; the kwarg wins)
DEFAULT_SPEED = 1.0               # m/s, for ``body_particle_exit_speed=True``
CUT_UNIT = 1.0e-6                 # m/s: the cut quantum of the owner choice
_MAX_UNITS = 2.0e9


@wp.func
def _start_depth(cv: ContactView, bary: wp.array[wp.vec3], c: int, x0: wp.array[wp.vec3]) -> float:
    """How far record c's contact point started the step inside its contact distance (<= 0: outside), as Newton's
    particle iteration measures it (rigid_vbd_kernels._eval_soft_ef_contact): the corners' barycentric mix, their
    largest radius plus the shape's margin, against the body point at the body's pose over the step."""
    ids = cv.corners[c]
    pt = x0[ids[0]]
    r = cv.radius[ids[0]]
    if ids[1] >= 0:
        b = bary[c]
        pt = b[0] * x0[ids[0]] + b[1] * x0[ids[1]]
        r = wp.max(r, cv.radius[ids[1]])
        if ids[2] >= 0:
            pt = pt + b[2] * x0[ids[2]]
            r = wp.max(r, cv.radius[ids[2]])
    s = cv.shape[c]
    margin = float(0.0)
    if cv.shape_margin.shape[0] > 0:
        margin = cv.shape_margin[s]
    bx = wp.transform_point(_pose(cv.body_q, cv.shape_body[s]), cv.body_pos[c])
    return r + margin - wp.dot(cv.normal[c], pt - bx)


@wp.struct
class CutInputs:
    """The step's start and end positions, the particles' inverse masses and the cap."""
    x0: wp.array[wp.vec3]
    x1: wp.array[wp.vec3]
    inv_mass: wp.array[float]
    v_cap: float


@wp.func
def _units(cut: float) -> int:
    return int(wp.min(cut / CUT_UNIT, _MAX_UNITS)) + 1


@wp.func
def _particle(cv: ContactView, bary: wp.array[wp.vec3], i: int) -> int:
    """Pair i = (record i // 3, corner i % 3): the corner's particle, or -1 (a dead record or a padding corner)."""
    c = i // 3
    k = i - 3 * c
    if not _live(cv, c) or _weight(cv, bary, c, k) <= 0.0:
        return -1
    return cv.corners[c][k]


@wp.func
def _share(cv: ContactView, bary: wp.array[wp.vec3], c: int, k: int, inv_mass: wp.array[float]) -> float:
    """Corner k's share of a normal displacement of record c's contact point: 1 for a particle record; for an edge /
    face record w_k / m_k / sum_j w_j^2 / m_j, how VBD splits a push on the contact point among the corners (a light
    corner moves more, up to 1.6x for a corner particle beside an interior one)."""
    ids = cv.corners[c]
    if ids[1] < 0:
        return 1.0
    den = float(0.0)
    for j in range(3):
        w = _weight(cv, bary, c, j)
        if w > 0.0:
            den = den + w * w * inv_mass[ids[j]]
    if den <= 0.0:
        return 0.0
    return _weight(cv, bary, c, k) * inv_mass[ids[k]] / den


@wp.func
def _moved(cv: ContactView, bary: wp.array[wp.vec3], c: int, x0: wp.array[wp.vec3], x1: wp.array[wp.vec3]) -> wp.vec3:
    """The step's displacement of record c's contact point (the corners' barycentric mix)."""
    ids = cv.corners[c]
    if ids[1] < 0:
        return x1[ids[0]] - x0[ids[0]]
    d = wp.vec3(0.0)
    for k in range(3):
        w = _weight(cv, bary, c, k)
        if w > 0.0:
            d = d + w * (x1[ids[k]] - x0[ids[k]])
    return d


@wp.func
def _cut(cv: ContactView, bary: wp.array[wp.vec3], ci: CutInputs, qd: wp.array[wp.vec3], i: int) -> float:
    """Pair i's cut of its particle's outward normal velocity (<= 0: none). The record's contact point moved out along
    the normal, relative to the body point, by up to its start depth: that is the depth the step resolved. The cut is
    the particle's share of that speed past v_cap. It never takes the particle's outward speed relative to the body
    below v_cap, and never takes more than its own outward speed: a particle resting on a pad that pulls away faster
    than the cap is not pulled after it (AUDIT-STACK 2026-11-10)."""
    c = i // 3
    depth = _start_depth(cv, bary, c, ci.x0)
    if depth <= 0.0:
        return 0.0
    k = i - 3 * c
    p = cv.corners[c][k]
    n = cv.normal[c]
    bv = _body_velocity(cv, c)
    resolved = wp.clamp(wp.dot(n, _moved(cv, bary, c, ci.x0, ci.x1) - bv * cv.dt), 0.0, depth)
    cut = wp.min(_share(cv, bary, c, k, ci.inv_mass) * resolved / cv.dt, wp.dot(n, qd[p] - bv)) - ci.v_cap
    return wp.min(cut, wp.max(wp.dot(n, qd[p]), 0.0))


@wp.kernel(enable_backward=False)
def largest(cv: ContactView, bary: wp.array[wp.vec3], ci: CutInputs, qd: wp.array[wp.vec3], best: wp.array[wp.int32]):
    """Per particle: the largest positive cut of its records, in CUT_UNIT (0: none)."""
    i = wp.tid()
    p = _particle(cv, bary, i)
    if p < 0:
        return
    cut = _cut(cv, bary, ci, qd, i)
    if cut > 0.0:
        wp.atomic_max(best, p, _units(cut))


@wp.kernel(enable_backward=False)
def choose(cv: ContactView, bary: wp.array[wp.vec3], ci: CutInputs, qd: wp.array[wp.vec3], best: wp.array[wp.int32],
           owner: wp.array[wp.int32]):
    """Per particle: the lowest pair with its largest cut owns it."""
    i = wp.tid()
    p = _particle(cv, bary, i)
    if p < 0:
        return
    cut = _cut(cv, bary, ci, qd, i)
    if cut > 0.0 and _units(cut) == best[p]:
        wp.atomic_min(owner, p, i)


@wp.kernel(enable_backward=False)
def cap_exit(cv: ContactView, bary: wp.array[wp.vec3], ci: CutInputs, owner: wp.array[wp.int32], qd: wp.array[wp.vec3],
             capped: wp.array[wp.int32]):
    """The owning pair cuts its particle's outward normal velocity (``_cut``)."""
    i = wp.tid()
    p = _particle(cv, bary, i)
    if p < 0 or owner[p] != i:
        return
    cut = _cut(cv, bary, ci, qd, i)
    if cut > 0.0:
        qd[p] = qd[p] - cut * cv.normal[i // 3]
        wp.atomic_add(capped, 0, 1)


class ExitSpeedCap:
    """The cap of one VBD solver (module docstring)."""

    def __init__(self, solver: Any, v_cap: float) -> None:
        m, dev = solver.model, solver.model.device
        n = max(int(m.particle_count), 1)
        self.solver, self.device, self.v_cap = solver, dev, float(v_cap)
        self.x0 = wp.zeros(n, dtype=wp.vec3, device=dev)        # the step's start positions
        self.best = wp.zeros(n, dtype=wp.int32, device=dev)
        self.owner = wp.full(n, NO_OWNER, dtype=wp.int32, device=dev)
        self.capped = wp.zeros(1, dtype=wp.int32, device=dev)
        self.views = RecordViews(solver)

    def begin(self, state_in: Any, contacts: Any, dt: float) -> bool:
        """Before the step: keep its start positions and body poses. False: nothing to cap."""
        if (contacts is None or contacts.soft_contact_max == 0 or dt <= 0.0 or self.solver.model.particle_count == 0
                or state_in.particle_qd is None):
            return False
        wp.copy(self.x0, state_in.particle_q)
        self.views.keep_start()
        return True

    def end(self, state_in: Any, state_out: Any, contacts: Any, dt: float) -> None:
        """After the step: cap the exit speeds (module docstring)."""
        cv = self.views.view(contacts, state_in, state_out, dt, after=True)
        bary, dim = self.views.bary(contacts), 3 * contacts.soft_contact_max
        ci = CutInputs()
        ci.x0, ci.x1, ci.inv_mass, ci.v_cap = self.x0, state_out.particle_q, self.solver.model.particle_inv_mass, \
            self.v_cap
        qd = state_out.particle_qd
        self.best.zero_()
        self.owner.fill_(NO_OWNER)
        wp.launch(largest, dim=dim, inputs=[cv, bary, ci, qd], outputs=[self.best], device=self.device)
        wp.launch(choose, dim=dim, inputs=[cv, bary, ci, qd, self.best], outputs=[self.owner], device=self.device)
        wp.launch(cap_exit, dim=dim, inputs=[cv, bary, ci, self.owner], outputs=[qd, self.capped], device=self.device)


class ExitSpeedMixin:
    """The exit-speed cap on a SolverVBD class (module docstring). Mixed in front of the entry's class."""

    alloy_exit_speed = DEFAULT_SPEED

    def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(model, *args, **kwargs)
        self.alloy_exit = ExitSpeedCap(self, self.alloy_exit_speed)

    def step(self, state_in, state_out, control, contacts, dt, **kwargs):
        live = self.alloy_exit.begin(state_in, contacts, dt)
        out = super().step(state_in, state_out, control, contacts, dt, **kwargs)
        if live:
            self.alloy_exit.end(state_in, state_out, contacts, dt)
        return out


def with_exit_speed(cls: type, v_cap: float) -> type:
    """``cls`` with `ExitSpeedMixin` in front, capping at ``v_cap`` m/s."""
    return type(f"{cls.__name__}ExitSpeed", (ExitSpeedMixin, cls), {"alloy_exit_speed": float(v_cap)})


def pop_exit_speed(kw: dict[str, Any]) -> float | None:
    """Pop the cap from a VBD entry's kwargs (``EXIT_ENV`` when the kwarg is absent): m/s, or None when off."""
    value = kw.pop(EXIT_KEY, None)
    if value is None:
        value = os.environ.get(EXIT_ENV, "").strip() or None
    if value is None or value is False:
        return None
    speed = DEFAULT_SPEED if value is True else float(value)
    if speed < 0.0:
        raise ValueError(f"{EXIT_KEY} must be a speed >= 0 m/s (or True for {DEFAULT_SPEED}), got {value!r}")
    return speed if speed > 0.0 else None


__all__ = ["DEFAULT_SPEED", "EXIT_ENV", "EXIT_KEY", "ExitSpeedCap", "ExitSpeedMixin", "pop_exit_speed",
           "with_exit_speed"]
