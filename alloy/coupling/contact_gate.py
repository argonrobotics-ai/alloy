# SPDX-License-Identifier: Apache-2.0
"""Contact-aware exchange clock (mixed-speed lever #2, [internal note] notes/mixed-speed-1107.md).

The interface bound ``0.5 · sqrt(m_eff / k_c)`` (and the opt-in damping bound) holds a light proxy's held contact
force stable, and today it takes the lightest proxy of every coupled pair whether or not it touches anything: a
0.1 kg Panda link on a 2e6 N/m cloth contact sets 128 exchanges per 10 ms for the whole scene while the towel is
still falling half a metre away. ``ContactGate`` keeps a pair's bound only while the pair is in or near contact:

- once per control period, at its start (host), each proxy (source body or particle mirrored into the destination) is
  tested against its destination in its world: the proxy's bounding sphere (its shapes' collision radii about the
  body origin; a particle's radius) against the box around the destination's own particles and bodies. It is *near*
  when the gap is at most ``gate_margin + 2 T (v_proxy + v_destination) + g T²`` (T = control_dt): what the pair
  can close within the period, twice over, plus a static margin. The sphere about the body origin is conservative:
  a long link arms several cm early (a towel falling onto a swinging Panda arms from its first period);
- a coupled pair (a source -> destination route) is armed whole when any of its movable proxies is near, and stays
  armed ``gate_hold_periods`` periods after that; the clock bound is the minimum over armed proxies (``bound``), so
  a pair in or near contact runs exactly today's clock (its lightest proxy's bound) and the rest of the scene keeps
  the step its other limits allow. When the armed set changes, the clock goes straight to what the new bound
  needs (the hold is the hysteresis), and when that is finer the entries' substep rungs follow at once
  (``ClockController.decide(jump=)``). Whole pairs, not single proxies: VBD's own step follows the exchange step,
  so a towel draping over a heavy link at that link's coarser bound drifted 42 mm against a 15 mm run-to-run floor
  (L4, robot_cloth_drop);
- proxies that cannot hold the clock (infinite own bound: a fixed body such as a table in the robot's file, or a
  contact-free pair) are always armed and never arm their pair: a towel resting on the table leaves the clock free;
- safety net: after every exchange a kernel flags any proxy that is not armed and carries a coupling force (it
  touched within the period). The coupler then asks for the whole period again with their pairs armed
  (``rerun_all``), so every exchange with a contact force on a proxy runs at today's clock;
- a released proxy adds nothing to its world's coupling residuals (``residuals.ResidualInputs.gated``; speed-now S2,
  WP10 2026-11-09): it cannot touch within the period, so its lag behind its source changes no result, while a
  lagged pair's velocity residual is that lag (a dt_x, the same in every pass). At the released clock it held every
  world to k_max passes: Panda + towel + water (64 worlds, L4) ran 8 passes per 1.25 ms exchange, 0.012-0.035 m/s
  against tol 0.01 with no force on any proxy, and a period took 65 s against today's clock's 17.5 s (6.8 s after).

Pairs that state an ``interface_stiffness`` (an always-on interface, D17) are never gated. Before the first period
(no state seen yet) every proxy is armed. Opt-in: ``ClockPolicy.contact_gate``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling.clocks import damping_bound, interface_bound
from alloy.coupling.endpoints import clock_masses

log = logging.getLogger("alloy.coupling")

__all__ = ["ContactGate", "GatedEndpoints", "gate_for"]

GRAVITY = 9.81                    # [m/s²], the acceleration term of the travel margin
BIG = 1.0e30


@wp.kernel(enable_backward=False)
def particle_box_kernel(ids: wp.array[wp.int32], q: wp.array[wp.vec3], qd: wp.array[wp.vec3], radius: wp.array[float],
                        world: wp.array[wp.int32], lo: wp.array2d[float], hi: wp.array2d[float],
                        vmax: wp.array[float], count: wp.array[wp.int32]):
    """Per world: the box around the destination's particles (grown by their radii) and their largest speed."""
    i = wp.tid()
    p = ids[i]
    w = world[p]
    if w < 0:
        return
    x = q[p]
    r = radius[p]
    for k in range(3):
        wp.atomic_min(lo, w, k, x[k] - r)
        wp.atomic_max(hi, w, k, x[k] + r)
    wp.atomic_max(vmax, w, wp.length(qd[p]))
    wp.atomic_add(count, w, 1)


@wp.kernel(enable_backward=False)
def body_box_kernel(ids: wp.array[wp.int32], body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                    radius: wp.array[float], world: wp.array[wp.int32], lo: wp.array2d[float],
                    hi: wp.array2d[float], vmax: wp.array[float], count: wp.array[wp.int32]):
    """Per world: grow the box by the destination's own bodies (bounding spheres about their origins)."""
    i = wp.tid()
    b = ids[i]
    w = world[b]
    if w < 0:
        return
    x = wp.transform_get_translation(body_q[b])
    r = radius[i]
    qd = body_qd[b]
    for k in range(3):
        wp.atomic_min(lo, w, k, x[k] - r)
        wp.atomic_max(hi, w, k, x[k] + r)
    v = wp.vec3(qd[0], qd[1], qd[2])
    om = wp.vec3(qd[3], qd[4], qd[5])
    wp.atomic_max(vmax, w, wp.length(v) + 2.0 * wp.length(om) * r)
    wp.atomic_add(count, w, 1)


@wp.kernel(enable_backward=False)
def near_kernel(is_body: wp.array[wp.int32], item: wp.array[wp.int32], world: wp.array[wp.int32],
                radius: wp.array[float], dest: wp.array[wp.int32], body_q: wp.array[wp.transform],
                body_qd: wp.array[wp.spatial_vector], particle_q: wp.array[wp.vec3], particle_qd: wp.array[wp.vec3],
                lo: wp.array3d[float], hi: wp.array3d[float], vmax: wp.array2d[float], count: wp.array2d[wp.int32],
                margin: float, horizon: float, near: wp.array[wp.int32]):
    """One proxy: 1 when the gap between its bounding sphere and its destination's box in its world is within the
    margin (static + twice the travel over ``horizon``, plus g horizon²); 0 when that world's destination is empty."""
    e = wp.tid()
    w = world[e]
    d = dest[e]
    r = radius[e]
    near[e] = 0
    if w < 0 or count[d, w] == 0:
        return
    c = wp.vec3(0.0)
    speed = float(0.0)
    if is_body[e] != 0:
        c = wp.transform_get_translation(body_q[item[e]])
        qd = body_qd[item[e]]
        speed = wp.length(wp.vec3(qd[0], qd[1], qd[2])) + 2.0 * wp.length(wp.vec3(qd[3], qd[4], qd[5])) * r
    else:
        c = particle_q[item[e]]
        speed = wp.length(particle_qd[item[e]])
    gap2 = float(0.0)
    for k in range(3):
        g = wp.max(wp.max(lo[d, w, k] - c[k], c[k] - hi[d, w, k]), 0.0)
        gap2 = gap2 + g * g
    gap = wp.sqrt(gap2) - r
    reach = margin + 2.0 * horizon * (speed + vmax[d, w]) + 9.81 * horizon * horizon
    if gap <= reach:
        near[e] = 1


@wp.kernel(enable_backward=False)
def touched_body_kernel(armed: wp.array[wp.int32], gate_index: wp.array[wp.int32], proxy: wp.array[wp.int32],
                        forces: wp.array[wp.spatial_vector], touched: wp.array[wp.int32]):
    """After an exchange: a proxy body that is not armed and carries a coupling force touched within the period."""
    i = wp.tid()
    e = gate_index[i]
    if armed[e] == 0 and wp.length_sq(forces[proxy[i]]) > 0.0:
        touched[e] = 1


@wp.kernel(enable_backward=False)
def touched_particle_kernel(armed: wp.array[wp.int32], gate_index: wp.array[wp.int32], proxy: wp.array[wp.int32],
                            forces: wp.array[wp.vec3], touched: wp.array[wp.int32]):
    i = wp.tid()
    e = gate_index[i]
    if armed[e] == 0 and wp.length_sq(forces[proxy[i]]) > 0.0:
        touched[e] = 1


@dataclass
class GatedEndpoints:
    """Host description of the gated proxies (one row per proxy endpoint) and their destinations."""

    is_body: np.ndarray             # (E,) int32
    item: np.ndarray                # (E,) int32 parent-model body or particle id of the proxy's source
    world: np.ndarray               # (E,) int32
    radius: np.ndarray              # (E,) float32 bounding radius [m]
    dest: np.ndarray                # (E,) int32 index into ``dest_names``
    bound_s: np.ndarray             # (E,) float64 the proxy's own clock bound [s] (stiffness and damping)
    pair: np.ndarray                # (E,) int32 the proxy's coupled pair (route); a pair is armed whole
    dest_names: list[str]
    dest_particles: list[np.ndarray]    # per destination: parent particle ids it owns
    dest_bodies: list[np.ndarray]       # per destination: parent body ids it owns
    dest_body_radius: list[np.ndarray]
    fixed_bound_s: float = math.inf     # always-on pairs (stated interface stiffness): never gated
    routes: list[tuple[str, str]] = field(default_factory=list)      # per pair index: (source, destination)


def body_radii(model: Any) -> np.ndarray:
    """(B,) bounding radius of each body's shapes about the body origin [m] (0 for a body without shapes)."""
    out = np.zeros(int(model.body_count), dtype=np.float64)
    if not model.shape_count:
        return out.astype(np.float32)
    body = model.shape_body.numpy()
    xf = model.shape_transform.numpy()
    rad = model.shape_collision_radius.numpy()
    for s in range(int(model.shape_count)):
        b = int(body[s])
        if b >= 0:
            out[b] = max(out[b], float(np.linalg.norm(xf[s][:3])) + float(rad[s]))
    return out.astype(np.float32)


class ContactGate:
    """Per-period arming of proxies near their destination, the gated clock bound, and the touched-unarmed flags."""

    def __init__(self, model: Any, ends: GatedEndpoints, sets: list[Any], margin: float, hold_periods: int) -> None:
        self.model, self.ends, self.margin, self.hold_periods = model, ends, float(margin), int(hold_periods)
        dev, n = model.device, len(ends.item)
        self.count = n
        self.armed_host = np.ones(n, dtype=bool)            # before the first period: everything armed
        self.always = ~np.isfinite(ends.bound_s)            # cannot hold the clock: always armed, never decides
        self._pair_count = int(ends.pair.max()) + 1 if n else 0
        self.hold = np.full(n, self.hold_periods, dtype=np.int64)
        self.armed = wp.array(np.ones(n, np.int32), dtype=wp.int32, device=dev)
        self.touched = wp.zeros(max(n, 1), dtype=wp.int32, device=dev)
        self.near = wp.zeros(max(n, 1), dtype=wp.int32, device=dev)
        self._dev_ends = {k: wp.array(getattr(ends, k).astype(np.int32 if k != "radius" else np.float32),
                                      device=dev) for k in ("is_body", "item", "world", "radius", "dest")}
        w, nd = max(int(model.world_count), 1), max(len(ends.dest_names), 1)
        self._lo = wp.zeros((nd, w, 3), dtype=float, device=dev)
        self._hi = wp.zeros((nd, w, 3), dtype=float, device=dev)
        self._vmax = wp.zeros((nd, w), dtype=float, device=dev)
        self._count = wp.zeros((nd, w), dtype=wp.int32, device=dev)
        self._dst = [(wp.array(p.astype(np.int32), device=dev), wp.array(b.astype(np.int32), device=dev),
                      wp.array(r.astype(np.float32), device=dev))
                     for p, b, r in zip(ends.dest_particles, ends.dest_bodies, ends.dest_body_radius)]
        self.set_index: dict[int, wp.array] = {}          # id(endpoint set) -> its endpoints' gate entries
        self._touch = self._touch_launches(sets)
        self.reruns = 0
        self.seen_state = False

    def _touch_launches(self, sets: list[Any]) -> list[tuple[Any, wp.array, wp.array, Any]]:
        """Per endpoint set that is gated: (kernel, gate index per endpoint, proxy index, the mapping)."""
        out, start = [], 0
        for es in sets:
            n = int(es.count)
            idx = wp.array(np.arange(start, start + n, dtype=np.int32), device=self.model.device)
            kernel = touched_body_kernel if es.kind == "body" else touched_particle_kernel
            out.append((kernel, idx, es.proxy_global, es.mapping))
            self.set_index[id(es)] = idx
            start += n
        return out

    # ------------------------------------------------------------------ host, once per period
    def evaluate(self, state: Any, control_dt: float) -> bool:
        """Arm the pairs in or near contact at the period start ``state``; returns True when the armed set changed
        (the clock then goes straight to what the new bound needs: ``ClockController.decide(jump=)``)."""
        if self.count == 0:
            return False
        near = self._whole_pairs(self._near(state, control_dt))
        self.hold = np.where(near, self.hold_periods, self.hold - 1)
        armed = near | (self.hold > 0)
        changed = bool(np.any(self.armed_host != armed))
        self._set_armed(armed)
        self.seen_state = True
        return changed

    def _near(self, state: Any, control_dt: float) -> np.ndarray:
        dev = self.model.device
        self._lo.fill_(BIG)
        self._hi.fill_(-BIG)
        self._vmax.zero_()
        self._count.zero_()
        for d, (pids, bids, brad) in enumerate(self._dst):
            lo, hi, vmax, cnt = self._lo[d], self._hi[d], self._vmax[d], self._count[d]
            if pids.shape[0] and state.particle_q is not None:
                wp.launch(particle_box_kernel, dim=pids.shape[0], device=dev,
                          inputs=[pids, state.particle_q, state.particle_qd, self.model.particle_radius,
                                  self.model.particle_world, lo, hi, vmax, cnt])
            if bids.shape[0] and state.body_q is not None:
                wp.launch(body_box_kernel, dim=bids.shape[0], device=dev,
                          inputs=[bids, state.body_q, state.body_qd, brad, self.model.body_world, lo, hi, vmax, cnt])
        e = self._dev_ends
        empty_t = wp.empty(0, dtype=wp.transform, device=dev)
        empty_s = wp.empty(0, dtype=wp.spatial_vector, device=dev)
        empty_v = wp.empty(0, dtype=wp.vec3, device=dev)
        wp.launch(near_kernel, dim=self.count, device=dev,
                  inputs=[e["is_body"], e["item"], e["world"], e["radius"], e["dest"],
                          state.body_q if state.body_q is not None else empty_t,
                          state.body_qd if state.body_qd is not None else empty_s,
                          state.particle_q if state.particle_q is not None else empty_v,
                          state.particle_qd if state.particle_qd is not None else empty_v,
                          self._lo, self._hi, self._vmax, self._count, self.margin, float(control_dt), self.near])
        return self.near.numpy()[: self.count].astype(bool)

    def _whole_pairs(self, flags: np.ndarray) -> np.ndarray:
        """Every proxy of each pair with a flagged movable proxy, and the always-armed ones."""
        hit = np.zeros(max(self._pair_count, 1), dtype=bool)
        np.logical_or.at(hit, self.ends.pair, flags & ~self.always)
        return self.always | hit[self.ends.pair]

    def _set_armed(self, armed: np.ndarray) -> None:
        self.armed_host = armed.copy()
        self.armed.assign(armed.astype(np.int32))

    def bound(self) -> float:
        """The clock bound [s]: the minimum over armed proxies, and the always-on pairs."""
        own = self.ends.bound_s[self.armed_host]
        return float(min(self.ends.fixed_bound_s, float(own.min()) if own.size else math.inf))

    # ------------------------------------------------------------------ device, every exchange
    def after_exchange(self) -> None:
        for kernel, idx, proxy, mapping in self._touch:
            if idx.shape[0]:
                wp.launch(kernel, dim=idx.shape[0], device=self.model.device,
                          inputs=[self.armed, idx, proxy, mapping.coupling_forces, self.touched])

    # ------------------------------------------------------------------ host, per attempt
    def begin_attempt(self) -> None:
        self.touched.zero_()

    def touched_unarmed(self) -> np.ndarray:
        return self.touched.numpy()[: self.count].astype(bool) if self.count else np.zeros(0, dtype=bool)

    def accept(self) -> bool:
        """Arm the pairs whose proxies touched unarmed (and hold them); True when there were any (re-run the period)."""
        t = self.touched_unarmed()
        if not t.any():
            return False
        arm = self._whole_pairs(t)
        self.hold = np.where(arm, self.hold_periods, self.hold)
        self._set_armed(self.armed_host | arm)
        self.reruns += 1
        log.info("coupling clock: %d unarmed proxy(ies) touched within the period; re-run with their pairs armed",
                 int(t.sum()))
        return True

    def describe(self) -> dict[str, Any]:
        """What the gate does to this scene's clock (for routes and estimates): the gated pairs and the known gap."""
        return {"exchange_clock": "contact gate (opt-in ClockPolicy.contact_gate): a pair's light-proxy bound holds "
                                  "only while the pair is in or near contact",
                "gated_pairs": [f"{s} -> {d}" for s, d in self.ends.routes],
                "gated_entries": sorted({name for route in self.ends.routes for name in route}),
                "known_gap": "docs/known_gaps/coupling-contact-gate-settling.md"}

    def notes(self) -> dict[str, Any]:
        return {"gate_rerun": bool(self.touched_unarmed().any()), "gate_armed": int(self.armed_host.sum()),
                "gate_proxies": int(self.count), "gate_reruns": int(self.reruns)}


def endpoint_bounds(masses: np.ndarray, k_c: float | None, c_d: float | None, safety: float,
                    omega_dt_max: float) -> np.ndarray:
    """(E,) each proxy's own clock bound [s] from its mass: the stiffness bound, and the damping bound when on."""
    out = np.full(masses.shape, math.inf)
    for i, m in enumerate(masses):
        if not m > 0.0:
            continue
        b = interface_bound(k_c, float(m), safety, omega_dt_max) if k_c else math.inf
        if c_d:
            b = min(b, damping_bound(c_d, float(m), safety, omega_dt_max))
        out[i] = b
    return out


def _append_rows(rows: dict[str, list], es: Any, bounds: np.ndarray, radii: np.ndarray, prad: np.ndarray,
                 dest: int, pair: int) -> None:
    """One endpoint set's rows: its proxies' source items, worlds, bounding radii, destination, bounds and pair."""
    items = es.src_global.numpy().astype(np.int64)
    body = es.kind == "body"
    rows["is_body"].append(np.full(len(items), int(body)))
    rows["item"].append(items)
    rows["world"].append(es.world.numpy())
    rows["radius"].append(radii[items] if body else prad[items])
    rows["dest"].append(np.full(len(items), dest))
    rows["bound_s"].append(bounds)
    rows["pair"].append(np.full(len(items), pair))


def gate_for(sched: Any) -> ContactGate | None:
    """The gate for a proxy ``Scheduler`` (None unless ``ClockPolicy.contact_gate`` and coupled endpoint sets exist).
    Each pair's terms come from the hooks today's bound reads: ``sched._pair_k_c(pair)`` and, with
    ``ClockPolicy.damping_bound``, ``sched._pair_c_d(pair)`` (None: no such term)."""
    clock = sched._alloy_policy.clock
    if not getattr(clock, "contact_gate", False) or not sched._sets:
        return None
    model = sched.model
    radii = body_radii(model)
    prad = model.particle_radius.numpy() if model.particle_count else np.zeros(0, np.float32)
    by_route = {(p.source, p.destination): p for p in sched._alloy_pairs}
    rows: dict[str, list] = {k: [] for k in ("is_body", "item", "world", "radius", "dest", "bound_s", "pair")}
    names: list[str] = []
    routes: list[tuple[str, str]] = []
    fixed, gated = math.inf, []
    for es in sched._sets:
        route = (es.src_name, es.dst_name)
        pair = by_route[route]
        c_d = sched._pair_c_d(pair) if clock.damping_bound else None
        masses = clock_masses(es)                         # 0 (no bound) for a proxy that cannot touch
        bounds = endpoint_bounds(masses, sched._pair_k_c(pair), c_d, clock.safety, clock.omega_dt_max)
        if getattr(pair, "interface_stiffness", None):     # an always-on interface (D17): never gated
            fixed = min(fixed, float(bounds.min()) if bounds.size else math.inf)
            continue
        names += [] if es.dst_name in names else [es.dst_name]
        routes += [] if route in routes else [route]
        _append_rows(rows, es, bounds, radii, prad, names.index(es.dst_name), routes.index(route))
        gated.append(es)
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rows.items()}
    owned = [sched._alloy_entries[n].entry for n in names]
    bodies = [np.asarray(e.bodies, dtype=np.int64) for e in owned]
    ends = GatedEndpoints(cat["is_body"], cat["item"], cat["world"], cat["radius"], cat["dest"],
                          cat["bound_s"].astype(np.float64), cat["pair"].astype(np.int64), names,
                          [np.asarray(e.particles, dtype=np.int64) for e in owned], bodies,
                          [radii[b] for b in bodies], fixed_bound_s=fixed, routes=routes)
    log.info("contact gate: %d proxies in %d pairs gated against %s; always-on bound %.3g s", len(ends.item),
             len(routes), names, fixed)
    return ContactGate(model, ends, gated, clock.gate_margin, clock.gate_hold_periods)
