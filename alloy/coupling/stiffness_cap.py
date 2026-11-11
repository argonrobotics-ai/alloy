# SPDX-License-Identifier: Apache-2.0
"""Exchange stiffness cap for lagged proxy bodies on a VBD destination (SCR-SC4-4d, WP10).

A proxy body's contact force reaches its source one exchange late. That explicit exchange is stable only while the
stiffness it sees, K = the summed stiffness of the body's touching records, stays below ``c · m_eff / dt_x²``
(m_eff: the source body's mass along its most mobile axis, ``alloy.coupling.axis_mass``, not the heavier isotropic
proxy mass; dt_x: the exchange interval). Full-surface contact breaks this. C16's film puts ~7,400 records
of 80.5 kN/m on one YAM finger, so K is about 6e8 N/m, ~100 x over the bound. Every burst of records engaging then
gives a one-exchange force spike (100-885 N against a 1-80 N squeeze) that throws the finger open (run P,
notes/wp10-runs/c16/fsP_kick_4w.trace.jsonl).

``ExchangeStiffnessCap`` caps K per proxy body. Right after SolverVBD seeds a body's records (material ke and
penalty), it sums the ke of the records within ``ENGAGE_SPEED · dt_x`` of touching that VBD gives force (Newton's own
gate, ``_soft_contact_force_enabled``: not SC4-6's muted flat edges, rows without an ACTIVE particle, or rows the
mesh filter drops). If the sum exceeds the cap, it
scales all of that body's records by cap / sum. The solve, the AVBD ramp ceiling and the coupling harvest all read
the scaled arrays, so the destination stays implicit and only the lagged stiffness changes.

**Opt-in, off by default** (coordinator 2026-11-06): ``SchedulerPolicy.exchange_stiffness_c`` (c) and
``exchange_damping_c`` (c_d) are None unless a scene sets them. Measured costs (docs/notes/wp10-sc4-4d.md):
- c 0.1 moves the film double-layer pinch gap from 1.41 to 1.22 mm (out of the 1.4-1.6 mm band); c >= 2 is
  film-neutral (1.40-1.41 mm). On C16 (with c_d 0.5) no c held the straps to 9.1 s: c 0.1 and c 0.3 each latched
  4 of 4 worlds.
- c_d 0.5 (beside slew c_s 0.1, which barely engaged) cuts C7's one-pass finger spikes from 605-707 N to 73-83 N
  (WP4, SCR-SC4-7) and keeps the light-box watchdog fixture at its 4.17 ms clock.
- But c_d removes contact dissipation that resting contact needs. In WP12's box under a towel
  (tests/coupling/test_rollback_bitwise.py) it engages every period (kd 7-24 x the bound), both worlds sit at WARN,
  and the coupler's interface work turns from -0.8 to -1.3 mJ per period into +0.5 mJ per period. With it on, a
  refined re-run's kept world also leaves the reference (P+3) when the re-run world needs more passes.
- The kd bound c_d m / dt_x is the explicit-damping limit; VBD's contact damping sits inside the destination's
  implicit solve, so that bound is conservative wherever the exchange itself is not what goes unstable.
- Sums are integer (``QUANT`` per N/m), so the scale does not depend on the atomic order and runs stay deterministic.
- One-way pairs get no cap (no feedback). Kinematic proxies (mass 0 in the view) get no cap.

Newton private names (pinned in ``alloy.core.newton_compat.PRIVATE_HOOKS`` by WP0): SolverVBD
``_refresh_body_particle_contact_state``, ``_particle_contact_adjacency_initialized``,
``body_particle_contact_material_ke``, ``body_particle_contact_penalty_k``; soft_contacts_mesh
``_soft_contact_force_enabled``, ``_make_soft_contact_filter_data``, ``_SoftContactFilterData``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import CouplingError
from alloy._vendor.newton._src.geometry.soft_contacts_mesh import (  # PRIVATE_HOOKS: VBD's own per-record force gate
    _make_soft_contact_filter_data,
    _soft_contact_force_enabled,
    _SoftContactFilterData,
)

log = logging.getLogger("alloy.coupling")

QUANT = 1.0e3          # fixed-point units per N/m in the per-body sums (1 mN/m resolution; int64 holds 9e15 N/m)
ENGAGE_SPEED = 0.5     # [m/s] a record counts when it can touch within one exchange at this approach speed
_HOOK = "_refresh_body_particle_contact_state"
_ARRAYS = ("body_particle_contact_material_ke", "body_particle_contact_penalty_k",
           "body_particle_contact_material_kd")


@wp.func
def record_gap(corners: wp.vec3i, bary: wp.vec3, q: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
               x_body: wp.transform, body_pos: wp.vec3, normal: wp.vec3, margin: float) -> float:
    """Distance from a soft record (vertex, edge or face point) to its rigid point, less radius and margin: < 0 is
    penetration (the geometry of Newton's _eval_soft_ef_contact)."""
    x = bary[0] * q[corners[0]]
    r = radius[corners[0]]
    if corners[1] >= 0:
        x += bary[1] * q[corners[1]]
        r = wp.max(r, radius[corners[1]])
    if corners[2] >= 0:
        x += bary[2] * q[corners[2]]
        r = wp.max(r, radius[corners[2]])
    return wp.dot(normal, x - wp.transform_point(x_body, body_pos)) - (r + margin)


@wp.kernel(enable_backward=False)
def sum_near_record_ke(count: wp.array(dtype=wp.int32), shape: wp.array(dtype=wp.int32),
                       shape_body: wp.array(dtype=wp.int32), shape_margin: wp.array(dtype=float),
                       capped: wp.array(dtype=wp.int32), corners: wp.array(dtype=wp.vec3i),
                       bary: wp.array(dtype=wp.vec3), q: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
                       body_q: wp.array(dtype=wp.transform), body_pos: wp.array(dtype=wp.vec3),
                       normal: wp.array(dtype=wp.vec3), ke: wp.array(dtype=float), kd: wp.array(dtype=float),
                       reach: float, gate: _SoftContactFilterData, total: wp.array2d(dtype=wp.int64)):
    """Per capped body: the summed ke (column 0) and kd (column 1) of its near records that VBD gives force."""
    i = wp.tid()
    if i >= count[0] or shape[i] < 0 or corners[i][0] < 0:
        return
    if not _soft_contact_force_enabled(i, gate):        # muted (SC4-6 flat edges), inactive or filtered: no force
        return
    b = shape_body[shape[i]]
    if b < 0 or capped[b] == 0:
        return
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[shape[i]]
    if record_gap(corners[i], bary[i], q, radius, body_q[b], body_pos[i], normal[i], margin) < reach:
        wp.atomic_add(total, b, 0, wp.int64(ke[i] * QUANT + 0.5))
        wp.atomic_add(total, b, 1, wp.int64(kd[i] * QUANT + 0.5))


@wp.func
def cap_factor(total: wp.int64, bound: float) -> float:
    """bound / sum when the sum exceeds a positive bound, else 1 (bound <= 0: that cap is off)."""
    k = float(total) / QUANT
    if bound <= 0.0 or k <= bound:
        return 1.0
    return bound / k


@wp.kernel(enable_backward=False)
def scale_body_records(count: wp.array(dtype=wp.int32), shape: wp.array(dtype=wp.int32),
                       shape_body: wp.array(dtype=wp.int32), total: wp.array2d(dtype=wp.int64),
                       body_mass: wp.array(dtype=float), k_per_kg: float, d_per_kg: float,
                       ke: wp.array(dtype=float), penalty: wp.array(dtype=float), kd: wp.array(dtype=float),
                       peak: wp.array2d(dtype=float)):
    """Scale every record of a body whose near-record ke sum exceeds k_per_kg · m (ke and penalty), and whose kd sum
    exceeds d_per_kg · m (kd); peak holds the largest sum / cap of each."""
    i = wp.tid()
    if i >= count[0] or shape[i] < 0:
        return
    b = shape_body[shape[i]]
    if b < 0 or body_mass[b] <= 0.0:                     # kinematic (mass 0): no lagged dynamics
        return
    fk = cap_factor(total[b, 0], k_per_kg * body_mass[b])
    fd = cap_factor(total[b, 1], d_per_kg * body_mass[b])
    if fk < 1.0:
        ke[i] = ke[i] * fk
        penalty[i] = penalty[i] * fk
        wp.atomic_max(peak, b, 0, 1.0 / fk)
    if fd < 1.0:
        kd[i] = kd[i] * fd
        wp.atomic_max(peak, b, 1, 1.0 / fd)


def _check_vbd(solver: Any) -> None:
    missing = [n for n in (_HOOK, "_particle_contact_adjacency_initialized", *_ARRAYS) if not hasattr(solver, n)]
    if missing:
        raise CouplingError(f"exchange stiffness cap: {type(solver).__name__} lacks Newton private {missing} "
                            "(PRIVATE_HOOKS; a Newton bump changed SolverVBD's contact seeding)")


class ExchangeStiffnessCap:
    """The cap on one VBD destination solver: wraps its ``step`` (to keep the input state, whose positions decide
    which records are near) and its record seeding (to scale them). ``dt_x`` is set by the scheduler per exchange.
    ``c`` bounds the near-record ke sum at c · m / dt_x², ``c_d`` the kd sum at c_d · m / dt_x; None: that cap off."""

    def __init__(self, solver: Any, proxy_bodies: np.ndarray, c: float | None, c_d: float | None = None,
                 engage_speed: float = ENGAGE_SPEED) -> None:
        _check_vbd(solver)
        if (c is not None and c <= 0.0) or (c_d is not None and c_d <= 0.0) or (c is None and c_d is None):
            raise CouplingError(f"exchange stiffness cap: c and c_d must be > 0 or None, not both None "
                                f"(got {c}, {c_d})")
        self.solver, self.c, self.c_d, self.engage_speed = solver, c, c_d, float(engage_speed)
        self.dt_x: float | None = None
        m, dev = solver.model, solver.model.device
        mask = np.zeros(int(m.body_count), np.int32)
        mask[np.asarray(proxy_bodies, np.int64)] = 1
        mask[m.body_mass.numpy() <= 0.0] = 0                 # kinematic proxies: nothing to destabilise
        self.capped = wp.array(mask, dtype=wp.int32, device=dev)
        self.cap_mass = wp.clone(m.body_mass)                 # the mass each body's cap reads (``use_masses``)
        self.total = wp.zeros((int(m.body_count), 2), dtype=wp.int64, device=dev)
        self.peak = wp.zeros((int(m.body_count), 2), dtype=float, device=dev)
        self._state: Any = None
        self._wrap()

    def _wrap(self) -> None:
        s = self.solver
        step, seed = s.step, getattr(s, _HOOK)

        def step_keeping_state(state_in: Any, *args: Any, **kw: Any) -> Any:
            self._state = state_in
            return step(state_in, *args, **kw)

        def seed_then_cap(contacts: Any, refresh: bool) -> None:
            seeds = self._seeds(contacts, refresh)
            seed(contacts, refresh)
            if seeds:
                self.apply(contacts)

        s.step = step_keeping_state
        setattr(s, _HOOK, seed_then_cap)

    def _seeds(self, contacts: Any, refresh: bool) -> bool:
        """Whether SolverVBD's seeding call will re-seed the record arrays (its own refresh condition)."""
        s = self.solver
        if contacts is None or s.model.particle_count == 0 or contacts.soft_contact_max == 0:
            return False
        return bool(refresh or s.body_particle_contact_penalty_k.shape[0] < contacts.soft_contact_max
                    or not s._particle_contact_adjacency_initialized)

    def use_masses(self, bodies: np.ndarray, masses: np.ndarray) -> None:
        """Cap these destination-view bodies at ``masses`` (their source bodies' most mobile axis, ``axis_mass``)
        instead of the installed proxy mass."""
        host = self.cap_mass.numpy().copy()
        host[np.asarray(bodies, np.int64)] = np.asarray(masses, np.float64)
        self.cap_mass.assign(host.astype(np.float32))

    def apply(self, contacts: Any) -> None:
        """Scale the just-seeded records of every capped body over c · m_eff / dt_x² (ke) and c_d · m_eff / dt_x
        (kd)."""
        if self._state is None or not self.dt_x:
            return
        s, m = self.solver, self.solver.model
        n, dev, dt = contacts.soft_contact_max, m.device, self.dt_x
        self.total.zero_()
        wp.launch(sum_near_record_ke, dim=n, device=dev,
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_shape, m.shape_body, m.shape_margin,
                          self.capped, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
                          self._state.particle_q, m.particle_radius, self._state.body_q,
                          contacts.soft_contact_body_pos, contacts.soft_contact_normal,
                          s.body_particle_contact_material_ke, s.body_particle_contact_material_kd,
                          self.engage_speed * dt,
                          _make_soft_contact_filter_data(m, contacts, self._state.particle_q, self._state.body_q),
                          self.total])
        wp.launch(scale_body_records, dim=n, device=dev,
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_shape, m.shape_body, self.total,
                          self.cap_mass, (self.c or 0.0) / (dt * dt), (self.c_d or 0.0) / dt,
                          s.body_particle_contact_material_ke, s.body_particle_contact_penalty_k,
                          s.body_particle_contact_material_kd, self.peak])

    def peak_ratio(self, reset: bool = True) -> np.ndarray:
        """Per body, the largest near-record sum / cap since the last read, columns (ke, kd); 0: never capped."""
        out = self.peak.numpy().copy()
        if reset:
            self.peak.zero_()
        return out


def bind_axis_masses(caps: list[ExchangeStiffnessCap], sets: list[Any]) -> None:
    """Each cap reads its proxies' axis masses (``EndpointSet.m_axis``) from the body endpoint sets into it."""
    for cap in caps:
        for es in sets:
            if es.kind == "body" and es.dst_name == getattr(cap, "destination", None) and es.m_axis is not None:
                cap.use_masses(es.proxy_local.numpy(), es.m_axis)


def install_stiffness_caps(sched: Any, c: float | None, c_d: float | None = None) -> list[ExchangeStiffnessCap]:
    """One cap per VBD destination of a coupled (not one-way) body-proxy mapping: the policy's ``(c, c_d)`` when either
    is set, else the destination solver's own default ``alloy_exchange_caps`` (SolverFilm: c 4, c_d 0.5, coordinator
    2026-11-08 15:16Z); none where both are None."""
    bodies: dict[str, list[np.ndarray]] = {}
    for mp in getattr(sched, "_proxy_mappings", ()):
        if (mp.src_name, mp.dst_name) in sched._one_way_routes or mp.proxy_ids_local is None:
            continue
        if isinstance(sched._entries[mp.dst_name].solver, newton.solvers.SolverVBD):
            bodies.setdefault(mp.dst_name, []).append(mp.proxy_ids_local.numpy())
    caps = []
    for name, ids in sorted(bodies.items()):
        solver = sched._entries[name].solver
        ce, cde = (c, c_d) if (c is not None or c_d is not None) else (getattr(solver, "alloy_exchange_caps", None)
                                                                     or (None, None))
        if ce is None and cde is None:
            continue
        cap = ExchangeStiffnessCap(solver, np.concatenate(ids), ce, cde)
        cap.destination = name
        solver.alloy_exchange_cap_installed = (ce, cde)     # sim.routes() names it (the solver's notes)
        caps.append(cap)
        log.info("exchange stiffness cap on %s: c %s, c_d %s, %d proxy bodies", name, ce, cde,
                 int(cap.capped.numpy().sum()))
    return caps
