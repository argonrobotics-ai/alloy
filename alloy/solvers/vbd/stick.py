# SPDX-License-Identifier: Apache-2.0
"""Stick friction for cloth held by rigid contacts (SCR-WP11-STICK; user ruling 2026-11-10 04:35Z, cloth grasp).

Newton's VBD friction is IPC's smoothed Coulomb law on the slip u within one step, measured from the step's start
(``particle_q_prev`` and the body point's motion over the step): f = mu f_n f1(|u| / eps_u), eps_u = friction_epsilon
dt, f1(x) = 2x - x^2 below 1. A vertex held against a steady tangential load T < mu f_n therefore slides by the u
where f = T in every step, a creep of v = friction_epsilon (1 - sqrt(1 - T / (mu f_n))) whatever the step: on the
route's friction_epsilon 1e-2, 2.9 mm/s at half the Coulomb limit, so a pinched cloth walks out of the fingers.

The stick: each particle keeps up to ``SLOTS`` anchors keyed by the shape it touches, as the elastic pre-slip s
(tangential, in the shape's body frame) it carried at the end of its last step in contact. A step's friction slip is
then u + s, the slip from where the vertex stuck, not from the step's start, and at the step's end s takes the new
slip clipped to eps_u (the anchor moves only by the slip past eps_u). Below the Coulomb limit a held vertex sits at
s = the pre-slip of its load, with no drift; above it, it slides at mu f_n as before (kinetic friction is unchanged:
|s| = eps_u, so f1 is saturated from the step's start). The hysteretic hinge's sliding rest (``hysteresis``) is the
same pattern on bending.

How: Newton's body-particle terms read the body point's velocity as (X bx - X_prev bx) / dt + X cvel, cvel the
record's ``soft_contact_body_vel`` (body frame). ``begin`` subtracts X^-1 s / dt from cvel for each record with an
anchor, so every reader of the record sees the slip u + s: the particle solve, the rigid side, and the coupler's
proxy harvest after the step (``coupling_harvest_proxy_wrenches``, ``alloy.coupling.record_momentum``), which is why
the shift stays on the record after the step. The next ``begin`` takes it off again when the record was not
refreshed in between (its velocity still holds the value written here, bit for bit; a refreshed record starts from
the collision pipeline's value). ``end`` reads the converged slip and stores the new anchors. No vendored kernel
changes. Only particle records (p, -1, -1) take part: the edge / face records of full-surface contacts keep Newton's
friction. A particle touching more than ``SLOTS`` shapes keeps Newton's friction on the extra ones (which ones depends
on the record order on a GPU).

Who writes the records (the take-off's invariant; AUDIT-STACK review 2026-11-10): the collision pipeline (a refresh),
``alloy.coupling.contact_order`` (a permutation of all record arrays right after a collide), the cutting plug-ins
(``topology/blade_contact``, ``vnode_cut``: they set a rewritten record's velocity to 0, so it carries no shift that
step) and this module. Nothing in Alloy snapshots or restores the record arrays; code that did would have to restore
``added`` and ``written`` with them. One solver per record buffer: in a coupled scene each entry steps on its own
filtered buffer (``SolverCoupledProxy._contacts_for_entry``), and ``begin`` raises when a second stick solver is
handed a buffer another one shifts. Readers of the shifted velocity inside the step (the momentum coarse contact
step, the rod solver, the coupler's harvest after it) see the stick's slip, as intended; film entries (SolverFilm)
never carry the stick.

Coupled passes: a coupler can step the entry several times for one exchange from the same input (the lagged loop's
two passes, a residual-gated loop's k passes, a restart). Substeps chain (a step's anchors feed the next), passes must
not: ``coupling_notify_input_state_update`` keeps the anchors the exchange started with and puts them back on an
iteration restart, as Newton keeps the exchange's input particle pose (``_coupling_particle_q_prev_snapshot``).
Without that the second pass would count the interval's slip twice (WP4 review 2026-11-10).

Opt-in: ``body_particle_friction_stick=True`` in the VBD entry's kwargs (``make_vbd``), or ``ALLOY_FRICTION_STICK=1``
in the environment for every VBD entry of a process. State for snapshots:
``_stick_shape`` and ``_stick_slip`` (particle x slot; ``arrays()``; ``alloy.coupling.adapters`` restores them per
world). Off, nothing here runs.
"""

from __future__ import annotations

import os
import weakref
from typing import Any

import warp as wp

from alloy.solvers.film.grip import ContactView, _body_velocity, _contact_force, _pose
from alloy.solvers.vbd.records import RecordViews

STICK_KEY = "body_particle_friction_stick"
STICK_ENV = "ALLOY_FRICTION_STICK"   # "1": the same choice for every VBD entry of a process (the kwarg wins)
SLOTS = 3                    # anchors per particle (a pinch between two fingers, on a table)
KEEP_S = 0.02                # an anchor outlives its contact this long [s] (a chattering finger keeps it)
NO_SHAPE = -1
NO_OWNER = 2**31 - 1
_BUFFER_OWNERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()   # record buffer ptr -> stick


@wp.func
def _tangent(n: wp.vec3, v: wp.vec3) -> wp.vec3:
    return v - n * wp.dot(n, v)


@wp.func
def _slot(shape_of: wp.array2d[wp.int32], p: int, s: int) -> int:
    for j in range(SLOTS):
        if shape_of[p, j] == s:
            return j
    return -1


@wp.func
def _same(a: wp.vec3, b: wp.vec3) -> bool:
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def _live(cv: ContactView, c: int) -> bool:
    return c < wp.min(cv.count[0], cv.max_count)


@wp.func
def _weight(cv: ContactView, bary: wp.array[wp.vec3], c: int, k: int) -> float:
    """Corner k's share of record c's contact point: 1 for a particle record's particle; the barycentric weight of
    an edge / face record's corner (full-surface contacts); 0 for a padding corner."""
    ids = cv.corners[c]
    if ids[1] < 0:
        if k == 0:
            return 1.0
        return 0.0
    if ids[k] < 0:
        return 0.0
    return bary[c][k]


@wp.func
def _pressing(cv: ContactView, bary: wp.array[wp.vec3], c: int, x: wp.array[wp.vec3]) -> bool:
    """Record c presses at x: Newton's penetration test at its contact point (the corners' barycentric mix)."""
    ids = cv.corners[c]
    if ids[1] < 0:
        return _contact_force(cv, c, ids[0], x) > 0.0
    b = bary[c]
    pt = b[0] * x[ids[0]] + b[1] * x[ids[1]]
    r = wp.max(cv.radius[ids[0]], cv.radius[ids[1]])
    if ids[2] >= 0:
        pt = pt + b[2] * x[ids[2]]
        r = wp.max(r, cv.radius[ids[2]])
    s = cv.shape[c]
    margin = float(0.0)
    if cv.shape_margin.shape[0] > 0:
        margin = cv.shape_margin[s]
    bx = wp.transform_point(_pose(cv.body_q, cv.shape_body[s]), cv.body_pos[c])
    return wp.dot(cv.normal[c], pt - bx) < r + margin


@wp.kernel(enable_backward=False)
def apply_stick(cv: ContactView, bary: wp.array[wp.vec3], shape_of: wp.array2d[wp.int32],
                slip: wp.array2d[wp.vec4], added: wp.array[wp.vec3], written: wp.array[wp.vec3],
                body_vel: wp.array[wp.vec3]):
    """Before the step: take off the last shift if record c was not refreshed since, then shift its body-point
    velocity by -s / dt (body frame), s the pre-slip of its contact point: its particle's anchor on the record's
    shape, or for an edge / face record the corners' anchors mixed by barycentric weight. Its slip reads u + s."""
    c = wp.tid()
    v = body_vel[c]
    if _same(v, written[c]):
        v = v - added[c]
    a = wp.vec3(0.0)
    if _live(cv, c):
        s = cv.shape[c]
        X = _pose(cv.body_q, cv.shape_body[s])
        ids = cv.corners[c]
        sw = wp.vec3(0.0)
        for k in range(3):
            w = _weight(cv, bary, c, k)
            if w > 0.0:
                j = _slot(shape_of, ids[k], s)
                if j >= 0:
                    q = slip[ids[k], j]
                    sw = sw + w * wp.transform_vector(X, wp.vec3(q[0], q[1], q[2]))
        sw = _tangent(cv.normal[c], sw)
        a = -wp.transform_vector(wp.transform_inverse(X), sw) / cv.dt
    body_vel[c] = v + a
    written[c] = v + a
    added[c] = a


@wp.func
def _vertex_slip(cv: ContactView, added: wp.array[wp.vec3], shape_of: wp.array2d[wp.int32],
                 slip: wp.array2d[wp.vec4], c: int, p: int, x: wp.array[wp.vec3]) -> wp.vec3:
    """Vertex p's converged slip from its anchor on record c's shape, clipped to eps_u, in the body frame: its own
    motion against the record's body point (the record's shift taken off) plus its own carried pre-slip."""
    s = cv.shape[c]
    X = _pose(cv.body_q, cv.shape_body[s])
    n = cv.normal[c]
    bv = _body_velocity(cv, c) - wp.transform_vector(X, added[c])
    u = _tangent(n, (x[p] - cv.anchor[p]) - bv * cv.dt)
    j = _slot(shape_of, p, s)
    if j >= 0:
        q = slip[p, j]
        u = u + _tangent(n, wp.transform_vector(X, wp.vec3(q[0], q[1], q[2])))
    length = wp.length(u)
    if length > cv.eps_u:
        u = u * (cv.eps_u / length)
    return wp.transform_vector(wp.transform_inverse(X), u)


@wp.func
def _corner_of(cv: ContactView, bary: wp.array[wp.vec3], t: int, x: wp.array[wp.vec3]) -> int:
    """Thread t = 3 c + k: record c's corner k when it has a share of a pressing record, else -1."""
    c = t / 3
    k = t - 3 * c
    if not _live(cv, c) or _weight(cv, bary, c, k) <= 0.0 or not _pressing(cv, bary, c, x):
        return -1
    return cv.corners[c][k]


@wp.func
def _key(cv: ContactView, t: int) -> int:
    """Owner key of thread t: particle records before edge / face records, then record order."""
    c = t / 3
    if cv.corners[c][1] < 0:
        return c
    return cv.max_count + t


@wp.kernel(enable_backward=False)
def carry_aged(shape_of: wp.array2d[wp.int32], slip: wp.array2d[wp.vec4], dt: float, keep: float,
               next_shape: wp.array2d[wp.int32], next_slip: wp.array2d[wp.vec4]):
    """First after the step: every anchor goes on, one step older, until it is ``keep`` seconds past its last
    pressing contact; ``keep_matched`` then renews those whose contact still presses."""
    p = wp.tid()
    for j in range(SLOTS):
        if shape_of[p, j] != NO_SHAPE:
            k = slip[p, j]
            age = k[3] + dt
            if age <= keep:
                next_shape[p, j] = shape_of[p, j]
                next_slip[p, j] = wp.vec4(k[0], k[1], k[2], age)


@wp.kernel(enable_backward=False)
def keep_matched(cv: ContactView, bary: wp.array[wp.vec3], x: wp.array[wp.vec3], shape_of: wp.array2d[wp.int32],
                 next_shape: wp.array2d[wp.int32], owner: wp.array2d[wp.int32]):
    """After the step, a pressing corner whose shape has an anchor keeps its slot; the lowest key owns it."""
    t = wp.tid()
    p = _corner_of(cv, bary, t, x)
    if p < 0:
        return
    s = cv.shape[t / 3]
    j = _slot(shape_of, p, s)
    if j >= 0:
        next_shape[p, j] = s
        wp.atomic_min(owner, p, j, _key(cv, t))


@wp.kernel(enable_backward=False)
def claim_new(cv: ContactView, bary: wp.array[wp.vec3], x: wp.array[wp.vec3], shape_of: wp.array2d[wp.int32],
              next_shape: wp.array2d[wp.int32], owner: wp.array2d[wp.int32]):
    """After ``keep_matched``: a pressing corner whose shape has no anchor takes a free slot (compare-and-swap), or
    the slot another record of the same particle and shape just took; the lowest key owns it."""
    t = wp.tid()
    p = _corner_of(cv, bary, t, x)
    if p < 0:
        return
    s = cv.shape[t / 3]
    if _slot(shape_of, p, s) >= 0:
        return
    for j in range(SLOTS):
        old = wp.atomic_cas(next_shape, p, j, NO_SHAPE, s)
        if old == NO_SHAPE or old == s:
            wp.atomic_min(owner, p, j, _key(cv, t))
            return


@wp.kernel(enable_backward=False)
def write_slips(cv: ContactView, bary: wp.array[wp.vec3], x: wp.array[wp.vec3], added: wp.array[wp.vec3],
                shape_of: wp.array2d[wp.int32], slip: wp.array2d[wp.vec4], next_shape: wp.array2d[wp.int32],
                owner: wp.array2d[wp.int32], next_slip: wp.array2d[wp.vec4]):
    """Each slot's owner writes its vertex's converged slip (one writer per slot: the same bits on every run)."""
    t = wp.tid()
    p = _corner_of(cv, bary, t, x)
    if p < 0:
        return
    j = _slot(next_shape, p, cv.shape[t / 3])
    if j >= 0 and owner[p, j] == _key(cv, t):
        u = _vertex_slip(cv, added, shape_of, slip, t / 3, p, x)
        next_slip[p, j] = wp.vec4(u[0], u[1], u[2], 0.0)


@wp.kernel(enable_backward=False)
def clear_worlds(particle_world: wp.array[wp.int32], world_mask: wp.array[wp.bool],
                 shape_of: wp.array2d[wp.int32], slip: wp.array2d[wp.vec4]):
    p = wp.tid()
    w = particle_world[p]
    if w >= 0 and w < world_mask.shape[0] and world_mask[w]:
        for j in range(SLOTS):
            shape_of[p, j] = NO_SHAPE
            slip[p, j] = wp.vec4(0.0)


def _empty(dtype: Any, device: Any) -> wp.array:
    return wp.zeros(1, dtype=dtype, device=device)


class FrictionStick:
    """The anchors of one VBD solver (module docstring)."""

    def __init__(self, solver: Any) -> None:
        m, dev = solver.model, solver.model.device
        self.solver, self.device = solver, dev
        n = max(int(m.particle_count), 1)
        self.shape_of = wp.full((n, SLOTS), NO_SHAPE, dtype=wp.int32, device=dev)
        self.slip = wp.zeros((n, SLOTS), dtype=wp.vec4, device=dev)
        self.next_shape = wp.full((n, SLOTS), NO_SHAPE, dtype=wp.int32, device=dev)
        self.next_slip = wp.zeros((n, SLOTS), dtype=wp.vec4, device=dev)
        self.owner = wp.full((n, SLOTS), NO_OWNER, dtype=wp.int32, device=dev)
        self.shape_in = wp.full((n, SLOTS), NO_SHAPE, dtype=wp.int32, device=dev)    # at the exchange's input
        self.slip_in = wp.zeros((n, SLOTS), dtype=wp.vec4, device=dev)
        self.added, self.written = _empty(wp.vec3, dev), _empty(wp.vec3, dev)   # per record: the shift, the value
        self.views = RecordViews(solver)
        pw = getattr(m, "particle_world", None)
        self.particle_world = pw if pw is not None else wp.zeros(max(int(m.particle_count), 1), dtype=wp.int32,
                                                                   device=dev)
        self._view: ContactView | None = None
        self._state_in: Any = None
        self._dt = 0.0
        self._contacts: Any = None

    def arrays(self) -> dict[str, wp.array]:
        """Per particle x slot state for snapshots: the anchors' shapes and pre-slips."""
        return {"_stick_shape": self.shape_of, "_stick_slip": self.slip}

    def _make_view(self, contacts: Any, state_in: Any, state_out: Any, dt: float, after: bool = False) -> ContactView:
        return self.views.view(contacts, state_in, state_out, dt, after)

    def begin(self, state_in: Any, state_out: Any, contacts: Any, dt: float) -> bool:
        """Before the step: shift the records' body velocities by the anchors' pre-slips. False: no contacts."""
        self._view = None
        if contacts is None or contacts.soft_contact_max == 0 or dt <= 0.0 or self.solver.model.particle_count == 0:
            return False
        if self.added.shape[0] != contacts.soft_contact_max:
            if wp.get_device(self.device).is_capturing:
                raise RuntimeError("body_particle_friction_stick: the contact buffer changed size inside a captured "
                                   "graph; run the first step with contacts eagerly (stick.py)")
            self.added = wp.zeros(contacts.soft_contact_max, dtype=wp.vec3, device=self.device)
            self.written = wp.full(contacts.soft_contact_max, wp.vec3(float("nan")), dtype=wp.vec3,
                                   device=self.device)      # NaN never equals a velocity: nothing to take off
        self._own(contacts.soft_contact_body_vel)
        self.views.keep_start()                                     # the step's start pose (records.py)
        self._view, self._contacts = self._make_view(contacts, state_in, state_out, dt), contacts
        self._state_in, self._dt = state_in, float(dt)
        wp.launch(apply_stick, dim=contacts.soft_contact_max,
                  inputs=[self._view, self._bary(contacts), self.shape_of, self.slip],
                  outputs=[self.added, self.written, contacts.soft_contact_body_vel], device=self.device)
        return True

    def _bary(self, contacts: Any) -> wp.array:
        return self.views.bary(contacts)

    def _own(self, body_vel: wp.array) -> None:
        """One stick per record buffer (module docstring): a second one would read the first one's shift as fresh."""
        owner = _BUFFER_OWNERS.get(body_vel.ptr)
        if owner is None:
            _BUFFER_OWNERS[body_vel.ptr] = self
        elif owner is not self:
            raise RuntimeError("body_particle_friction_stick: two VBD solvers with the stick share one Contacts "
                               "buffer; give each solver its own contacts (stick.py)")

    def end(self, state_out: Any) -> None:
        """After the step: the converged slips become the anchors (the records keep their shift: module doc)."""
        contacts = self._contacts
        if self._view is None:
            return
        cv = self._make_view(contacts, self._state_in, state_out, self._dt, after=True)
        dim, x = contacts.soft_contact_max, state_out.particle_q
        self.next_shape.fill_(NO_SHAPE)
        self.next_slip.zero_()
        self.owner.fill_(NO_OWNER)
        wp.launch(carry_aged, dim=self.shape_of.shape[0], inputs=[self.shape_of, self.slip, cv.dt, KEEP_S],
                  outputs=[self.next_shape, self.next_slip], device=self.device)
        bary = self._bary(contacts)
        for kernel in (keep_matched, claim_new):
            wp.launch(kernel, dim=3 * dim, inputs=[cv, bary, x, self.shape_of], outputs=[self.next_shape, self.owner],
                      device=self.device)
        wp.launch(write_slips, dim=3 * dim, inputs=[cv, bary, x, self.added, self.shape_of, self.slip,
                                                    self.next_shape, self.owner], outputs=[self.next_slip],
                  device=self.device)
        wp.copy(self.shape_of, self.next_shape)
        wp.copy(self.slip, self.next_slip)
        self._view = None

    def on_input(self, iteration_restart: bool, dt: float) -> None:
        """The coupler's input notification: keep the anchors at a new exchange's input, put them back on a restart
        (module docstring, coupled passes). A restart with dt 0 is a reset's, which ``reset`` already did."""
        if dt <= 0.0:
            return
        if iteration_restart:
            wp.copy(self.shape_of, self.shape_in)
            wp.copy(self.slip, self.slip_in)
        else:
            wp.copy(self.shape_in, self.shape_of)
            wp.copy(self.slip_in, self.slip)

    def reset(self, world_mask: wp.array | None = None) -> None:
        for shape_of, slip in ((self.shape_of, self.slip), (self.shape_in, self.slip_in)):
            if world_mask is None:
                shape_of.fill_(NO_SHAPE)
                slip.zero_()
                continue
            wp.launch(clear_worlds, dim=shape_of.shape[0], inputs=[self.particle_world, world_mask, shape_of, slip],
                      device=self.device)


class StickMixin:
    """Stick friction on a SolverVBD class (module docstring). Mixed in front of the entry's class."""

    def __init__(self, model: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(model, *args, **kwargs)
        self.alloy_stick = FrictionStick(self)
        for key, arr in self.alloy_stick.arrays().items():       # snapshot names (module docstring)
            setattr(self, key, arr)

    def step(self, state_in, state_out, control, contacts, dt, **kwargs):
        live = self.alloy_stick.begin(state_in, state_out, contacts, dt)
        try:
            return super().step(state_in, state_out, control, contacts, dt, **kwargs)
        finally:
            if live:
                self.alloy_stick.end(state_out)

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask, flags)
        self.alloy_stick.reset(world_mask)

    def coupling_notify_input_state_update(self, state, flags, *, iteration_restart: bool = False, dt: float = 0.0):
        self.alloy_stick.on_input(iteration_restart, dt)
        return super().coupling_notify_input_state_update(state, flags, iteration_restart=iteration_restart, dt=dt)


def with_stick(cls: type) -> type:
    """``cls`` with `StickMixin` in front."""
    return type(f"{cls.__name__}Stick", (StickMixin, cls), {})


def pop_stick(kw: dict[str, Any]) -> bool:
    """Pop the stick option from a VBD entry's kwargs (``STICK_ENV`` when the kwarg is absent)."""
    value = kw.pop(STICK_KEY, None)
    if value is None:
        value = os.environ.get(STICK_ENV, "0") not in ("", "0", "false", "False")
    return bool(value)


__all__ = ["SLOTS", "STICK_ENV", "STICK_KEY", "FrictionStick", "StickMixin", "pop_stick", "with_stick"]
