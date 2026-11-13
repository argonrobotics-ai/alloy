# SPDX-License-Identifier: Apache-2.0
"""Exchange-step bound from the proxy contact band (R2-E3, coordinator 2026-11-07).

A proxy pipeline re-collides once per exchange: Newton's ``SolverCoupledProxy`` counts ``collide_interval`` in outer
coupled steps, and the destination's substeps reuse that contact set. A particle that crosses a large part of the
pipeline's detection band (``soft_contact_gap``) relative to the rigid feature it touches, within one exchange, is
solved against a stale contact frame.

The trial agent's B3 fold ([internal note], R2-E3) hit this. A 4 mm sheet pinched by two
1 kg pads whipped at 2-3 m/s at the top of its carry. A corner vertex beside a pad then grew 2.5 -> 2.8 -> 6.2 ->
10.7 -> 23 m/s over five exchanges at dt_x 0.3125 ms (SPEED_FAIL). That is 0.26-0.31 of the 3 mm full-surface band
per exchange.
- The same world stayed healthy with dt_x held at 0.156 ms (0.16 of the band).
- Eight VBD substeps per exchange did not help, since they reuse the exchange's contacts.
- The GPU run failed the same way in 56 of 64 worlds at one period, where worlds 0-63 move nearly in lockstep; the
  CPU failed in about 1 world of 14.

So the clock keeps ``dt_x <= BAND_TRAVEL * band / v``, per pipeline, over the last period and every world (the clock
is shared). ``BAND_TRAVEL`` is 1/6: C7's full-surface design moves under 0.5 mm of the 3 mm band per exchange. ``v`` is
the larger of two speeds:
- the largest speed of a soft-contact record's particles relative to the body point it touches (a fast pad on a
  still sheet). The body point's velocity is the body's own motion at that point. Newton's record velocity
  (``soft_contact_body_vel``) is only a shape's surface velocity in its body frame, zero for rigid primitives. Until
  the R2-E3 follow-up the bound read that alone, so a box falling onto a sheet at 4.4 m/s read about 1 m/s;
- the destination's fastest particle, in any exchange where its pipeline holds a record.
The second one is the one that matters here. Bounding by records alone engaged at B3 world 7's first whip and got past
it. But it relaxed, and the second whip (2-3 m/s, carried by particles not yet in a record) blew up the corner on
entering the band. That happens within five exchanges, too fast for a per-period clock. A sheet that is falling with
nothing in any band gives no bound. The level rises at once and comes back down only after the clock's quiet periods
(``ClockPolicy.downshift_quiet_periods``).

Graph-safe: two kernels per built pipeline with fixed launch dims (``soft_contact_max``, the view's particles). The
host reads the maxima
once per period, in the coupler's report.

**Re-run** (``BandRerun``, R2-E3 follow-up, coordinator 2026-11-08 09:16Z). The bound reaches the clock one period
late, so a contact that speeds up inside a period runs that period at the old step. B3 at N220 with the fabric's
bending (WP11's L4 run, world 32 of 64): period 591 ran at dt_x 0.156 ms, set from 2.7 m/s in period 590, and a contact
inside it reached 21 m/s, which needs 0.024 ms. The world hit SPEED_FAIL at its end. When a period's own bound is
under ``dt_x / ClockPolicy.band_rerun`` (3 by default: a contact crossed more than half its band in one exchange at the
default ``band_travel``), the period is re-run, every world, from its start. The re-run's clock takes the measured
bound, the attempt's escalations are dropped, and its own report decides. At most ``RERUNS_PER_PERIOD`` per period,
and none when the bound is finer than the clock's finest step (control_dt / 2^j_max). A re-run there cannot reach the
bound and costs up to 2^j_max exchanges. Such a contact speed (over 25 m/s at a 3 mm band) is a blow-up for the health
monitor (SPEED_FAIL > 20 m/s), not an impact a finer clock resolves. Example: a cloth started inside a box under a
one-record contact budget (test_wiring_fallback) has bound 6 µs against a finest step of 19.5 µs. Its 512-exchange
re-run took that CPU test past 900 s (gate runner, 2026-11-08 12:34Z). Those are counted as ``band.unreachable``.
Each firing is counted (``band_reruns``, report notes ``band.reruns``). The
re-run's finer clock is the whole scene's, so one world's whip changes every world's steps in that period (known gap
coupling-shared-exchange-clock, WP3).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.coupling")

BAND_TRAVEL = 1.0 / 6.0       # largest fraction of a pipeline's detection band a contact may cross per exchange
V_FLOOR = 1.0e-6              # [m/s] below this a pipeline sets no bound
BAND_RERUN = 3.0              # re-run a period whose own bound is under dt_x / BAND_RERUN (ClockPolicy.band_rerun)
RERUNS_PER_PERIOD = 1         # band re-runs per control period
NO_ROWS = -1                  # kernel ``base`` without per-world rows (``ContactBandBound.keep_world_rows`` off)


@wp.func
def skipped_particle(p: int, world: wp.array[wp.int32], skip: wp.array[wp.int32], worlds: int) -> bool:
    """A particle of a world whose contacts set no bound (``ContactBandBound.park``). ``worlds`` 0: no world map."""
    if worlds == 0:
        return False
    w = world[p]
    if w < 0 or w >= skip.shape[0]:
        return False
    return skip[w] != 0


@wp.kernel(enable_backward=False)
def sheet_speed_kernel(count: wp.array[wp.int32], particle_qd: wp.array[wp.vec3], particle_mass: wp.array[float],
                       slot: int, world: wp.array[wp.int32], skip: wp.array[wp.int32], worlds: int, base: int,
                       vmax: wp.array[float]):
    """vmax[slot] = the fastest particle with mass, when the pipeline holds at least one record this exchange.
    One world signature for both band kernels: ``world`` the particle -> world map (``worlds`` 0: none), ``skip`` the
    parked worlds, whose particles set no bound, ``base`` >= 0 the first slot of this route's per-world row (NO_ROWS:
    none): the same speed then also goes to the particle's own world, vmax[base + world]."""
    i = wp.tid()
    if count[0] <= 0 or particle_mass[i] <= 0.0 or skipped_particle(i, world, skip, worlds):
        return
    s = wp.length(particle_qd[i])
    wp.atomic_max(vmax, slot, s)
    if base >= 0 and worlds > 0:
        w = world[i]
        if w >= 0 and w < worlds:
            wp.atomic_max(vmax, base + w, s)


@wp.kernel(enable_backward=False)
def band_speed_kernel(count: wp.array[wp.int32], cap: int, indices: wp.array[wp.vec3i], shape: wp.array[wp.int32],
                      body_pos: wp.array[wp.vec3], body_vel: wp.array[wp.vec3], shape_body: wp.array[wp.int32],
                      local_to_global: wp.array[wp.int32], body_q: wp.array[wp.transform],
                      body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], bodies: int,
                      particle_qd: wp.array[wp.vec3], slot: int, world: wp.array[wp.int32],
                      skip: wp.array[wp.int32], worlds: int, base: int, vmax: wp.array[float]):
    """vmax[slot] = max over the valid soft-contact records of |qd(particle) - v(body point)| over their particles
    (one for a vertex record, two for an edge, three for a face). The body point moves with its body, v + w x r about
    the centre of mass, plus the record's own surface velocity (body frame), as the solvers read it
    (semi_implicit/kernels_contact.py). The body's twist is the scene's at the exchange start (``body_q``/``body_qd``
    global, through the entry's ``local_to_global``): a lagged proxy restarts there every exchange, so that is how fast
    it sweeps the band; the destination's own end-of-exchange proxy has already been slowed by the contact. ``bodies``
    0: no body state, only the record's velocity. ``world``, ``skip``, ``worlds``, ``base`` as in
    ``sheet_speed_kernel``: a parked world's particles are left out, and with rows each particle's relative speed also
    goes to its own world, vmax[base + world], and vmax[base + 2 worlds + world] is set to 1 (it holds a record)."""
    i = wp.tid()
    if i >= wp.min(count[0], cap):
        return
    idx = indices[i]
    v = body_vel[i]
    if bodies != 0:
        b = int(-1)
        if shape[i] >= 0:
            b = shape_body[shape[i]]
        if b >= 0:
            b = local_to_global[b]
            X = body_q[b]
            r = wp.transform_point(X, body_pos[i]) - wp.transform_point(X, body_com[b])
            qd = body_qd[b]
            v = wp.spatial_top(qd) + wp.transform_vector(X, body_vel[i]) + wp.cross(wp.spatial_bottom(qd), r)
    s = float(0.0)
    for k in range(3):
        p = idx[k]
        if p >= 0 and not skipped_particle(p, world, skip, worlds):
            sp = wp.length(particle_qd[p] - v)
            s = wp.max(s, sp)
            if base >= 0 and worlds > 0:
                w = world[p]
                if w >= 0 and w < worlds:
                    wp.atomic_max(vmax, base + w, sp)
                    wp.atomic_max(vmax, base + 2 * worlds + w, 1.0)
    wp.atomic_max(vmax, slot, s)


def band_bound(gap: float, v_rel: float, travel: float = BAND_TRAVEL) -> float:
    """``travel·gap/v_rel`` [s]; inf for a pipeline at rest or without a band."""
    if not (v_rel > V_FLOOR) or not (gap > 0.0):
        return math.inf
    return travel * gap / v_rel


class ContactBandBound:
    """Per proxy pipeline (deduplicated factories) and its destination entry: the largest relative contact speed of
    the period on the device, and the exchange-step bound on the host."""

    def __init__(self, routes: Sequence[tuple[Any, str]], device: Any, worlds: int = 1) -> None:
        seen: set[int] = set()
        self.routes: list[tuple[Any, str]] = []
        for factory, destination in routes:
            if id(factory) not in seen:
                seen.add(id(factory))
                self.routes.append((factory, destination))
        self.vmax = wp.zeros(max(len(self.routes), 1), dtype=float, device=device)
        self._none = (wp.zeros(1, dtype=wp.int32, device=device), wp.zeros(1, dtype=wp.transform, device=device),
                      wp.zeros(1, dtype=wp.spatial_vector, device=device), wp.zeros(1, dtype=wp.vec3, device=device))
        self.last_speed = np.zeros(len(self.routes))        # [m/s] the last period's maxima (report notes)
        self.last_bound = math.inf                          # [s]
        self.skip = wp.zeros(max(int(worlds), 1), dtype=wp.int32, device=device)   # 1: a parked world (``park``)
        self.per_world = False                              # keep per-world rows when asked (keep_world_rows)
        self.worlds = 0                                     # per-world rows (ensure_worlds); 0: only the maxima
        self.last_world = np.zeros((len(self.routes), 3, 0))    # [route, (record, sheet, has record), world]
        self.last_travel: float | None = BAND_TRAVEL

    def park(self, worlds: np.ndarray | None) -> None:
        """``worlds`` (bool [W]; None: none) are parked (``parked.py``): their particles and records set no bound."""
        if worlds is None:
            self.skip.zero_()
        else:
            self.skip.assign(np.asarray(worlds, dtype=np.int32))

    def keep_world_rows(self, on: bool = True) -> None:
        """Ask for (or drop) the per-world rows; they appear at the next ``ensure_worlds``. Off (the default) the
        array and the kernels' writes are the shared maxima alone."""
        self.per_world = bool(on)

    def ensure_worlds(self, worlds: int) -> None:
        """Host, before an attempt (never inside a captured period): keep per-world rows behind the shared maxima, in
        the same array (so every ``vmax.zero_()`` clears them too). Layout: [routes] the shared maxima, then per route
        three rows of ``worlds``: record speed, sheet speed, has-a-record. The shared maxima are written as before."""
        worlds = int(worlds) if self.per_world else 0
        if worlds == self.worlds or not self.routes:
            return
        n = len(self.routes)
        grown = np.zeros(n * (1 + 3 * worlds), dtype=np.float32)
        grown[:n] = self.vmax.numpy()[:n]
        self.vmax = wp.array(grown, dtype=float, device=self.vmax.device)
        self.worlds = worlds
        self.last_world = np.zeros((n, 3, worlds))

    def _world_inputs(self, views: tuple[Any, Any], slot: int, row: int, count: int) -> list[Any]:
        """The band kernels' world inputs (world, skip, worlds, base) for route ``slot``'s per-world row ``row`` (0
        record, 1 sheet). The particle -> world map is the destination entry's view's, else the pipeline's view's
        (``views``); ``worlds`` 0 when neither covers ``count`` particles: then nothing is skipped and no row is
        written. ``base`` is NO_ROWS without per-world rows."""
        world = next((m for m in (getattr(v, "particle_world", None) for v in views)
                      if m is not None and int(m.shape[0]) >= count), None)
        if world is None:
            return [self._none[0], self.skip, 0, NO_ROWS]
        if not self.worlds:
            return [world, self.skip, int(self.skip.shape[0]), NO_ROWS]
        return [world, self.skip, self.worlds, len(self.routes) + (3 * slot + row) * self.worlds]

    def accumulate(self, entries: Mapping[str, Any], scene: Any = None, model: Any = None) -> None:
        """After an exchange: fold each pipeline's records against its destination's end-of-exchange particle
        velocities and the bodies' twists in ``scene`` (the exchange's input state; ``model`` the scene's model)."""
        for slot, (factory, destination) in enumerate(self.routes):
            entry = entries[destination]
            state, mass = entry.state_1, entry.solver.model.particle_mass
            for view, _pipeline, contacts, _excluded in factory.built:
                if contacts is None or state.particle_qd is None or not int(contacts.soft_contact_max):
                    continue
                count = max(int(state.particle_qd.shape[0]), int(mass.shape[0]) if mass is not None else 0)
                views = (getattr(entry, "view", None), view)
                wp.launch(band_speed_kernel, dim=int(contacts.soft_contact_max), device=self.vmax.device,
                          inputs=[contacts.soft_contact_count, int(contacts.soft_contact_max),
                                  contacts.soft_contact_indices, *self._bodies(view, entry, contacts, scene, model),
                                  state.particle_qd, slot,
                                  *self._world_inputs(views, slot, 0, count)],
                          outputs=[self.vmax])
                if mass is not None and int(mass.shape[0]):
                    wp.launch(sheet_speed_kernel, dim=int(mass.shape[0]), device=self.vmax.device,
                              inputs=[contacts.soft_contact_count, state.particle_qd, mass, slot,
                                      *self._world_inputs(views, slot, 1, count)],
                              outputs=[self.vmax])

    def _bodies(self, view: Any, entry: Any, contacts: Any, scene: Any, model: Any) -> list[Any]:
        """The record and body inputs of ``band_speed_kernel``: the view's shape -> body map, the entry's body map to
        the scene, and the scene's body poses, twists and centres of mass, else placeholders with ``bodies`` 0."""
        shape_body, l2g = getattr(view, "shape_body", None), getattr(entry, "body_local_to_global", None)
        body_q, body_qd = getattr(scene, "body_q", None), getattr(scene, "body_qd", None)
        com, shape = getattr(model, "body_com", None), getattr(contacts, "soft_contact_shape", None)
        pos = getattr(contacts, "soft_contact_body_pos", None)
        arrays = (shape_body, l2g, body_q, body_qd, com, shape, pos)
        if not all(a is not None and int(a.shape[0]) for a in arrays):
            i32, q, qd, v3 = self._none
            return [shape if shape is not None else i32, pos if pos is not None else v3, contacts.soft_contact_body_vel,
                    i32, i32, q, qd, v3, 0]
        return [shape, pos, contacts.soft_contact_body_vel, shape_body, l2g, body_q, body_qd, com, 1]

    def end_period(self, travel: float | None = BAND_TRAVEL) -> float:
        """Host, once per period (outside capture): the bound from the period's maxima [s] (inf when ``travel`` is
        None: the policy turned it off); clears them."""
        if not self.routes:
            return math.inf
        host = self.vmax.numpy()
        self.last_speed = host[: len(self.routes)].copy()
        self.last_world = host[len(self.routes):].astype(np.float64).reshape(len(self.routes), 3, self.worlds)
        self.last_travel = travel
        self.vmax.zero_()
        if travel is None:
            self.last_bound = math.inf
            return self.last_bound
        bounds = [band_bound(_gap(factory), float(v), travel)
                  for (factory, _d), v in zip(self.routes, self.last_speed, strict=True)]
        self.last_bound = min(bounds, default=math.inf)
        return self.last_bound

    def world_speed(self, need: bool = True) -> np.ndarray:
        """[route, world] m/s, the last period's per-world speeds (zeros without rows). ``need``: a world's own need,
        its records' relative speed and, when it held a record in the period, its fastest particle. Not ``need``: the
        shared rule world by world (the sheet counts whenever any world of the pipeline holds a record), whose maximum
        over worlds is ``last_speed`` bit for bit when every particle has a world."""
        record, sheet, has = self.last_world[:, 0], self.last_world[:, 1], self.last_world[:, 2]
        return np.maximum(record, np.where(has > 0.0, sheet, 0.0) if need else sheet)

    def world_bounds(self, need: bool = True) -> np.ndarray | None:
        """(worlds,) s: each world's band bound from ``world_speed`` (the last read's ``travel``); None without
        per-world rows."""
        if not self.routes or not self.worlds:
            return None
        out = np.full(self.worlds, math.inf)
        if self.last_travel is None:
            return out
        for (factory, _d), speeds in zip(self.routes, self.world_speed(need), strict=True):
            gap = _gap(factory)
            out = np.minimum(out, [band_bound(gap, float(np.float32(v)), self.last_travel) for v in speeds])
        return out


class BandRerun:
    """Host state of the band re-run (module docstring, Re-run). ``enabled`` once the driver re-runs whole periods
    (``IntegratedSimulation``: ``SchedulerMixin.allow_capped_captures``); ``ratio`` None turns it off."""

    def __init__(self, ratio: float | None = BAND_RERUN, per_period: int = RERUNS_PER_PERIOD) -> None:
        self.ratio, self.per_period = ratio, int(per_period)
        self.enabled = False
        self.total = 0              # firings over the run (report notes "reruns")
        self.this_period = 0
        self.wanted = False         # the attempt just reported asks for a re-run
        self.pending = False        # the next attempt is a band re-run
        self.unreachable = 0        # attempts outran by a bound finer than the finest step: no re-run (docstring)
        self.attempts = 0           # attempts begun (begin_attempt): read_band acts once per attempt

    @property
    def on(self) -> bool:
        return self.enabled and self.ratio is not None

    def new_period(self) -> None:
        self.this_period = 0

    def judge(self, dt_x: float | None, bound: float, finest: float) -> bool:
        """After an attempt at ``dt_x``: True when its own band ``bound`` is under dt_x / ratio, the period has a re-run
        left and the clock can step at the bound (``bound`` at least ``finest``, so dt_x is above it too)."""
        outran = bool(self.on and dt_x is not None and self.this_period < self.per_period
                      and bound < dt_x / float(self.ratio))
        self.wanted = outran and bound >= finest * (1.0 - 1.0e-9)
        if outran and not self.wanted:
            self.unreachable += 1
            (log.warning if self.unreachable == 1 else log.debug)(
                "coupling: contacts outran dt_x %.3g s with a band bound of %.3g s, finer than the finest step %.3g s: "
                "no band re-run (it cannot reach the bound; such a speed is the health monitor's), %d so far",
                dt_x, bound, finest, self.unreachable)
        return self.wanted

    def accept(self, speeds: np.ndarray, dt_x: float | None, bound: float) -> None:
        """The driver rolled the period back for this re-run (``speeds``: the attempt's per-pipeline maxima [m/s])."""
        self.wanted, self.pending = False, True
        self.total += 1
        self.this_period += 1
        log.info("coupling: band re-run %d: contacts at %s m/s outran dt_x %.3g s (bound %.3g s); the period re-runs "
                 "at the bound", self.total, np.round(np.asarray(speeds, float), 3).tolist(), dt_x or math.nan, bound)

    def begin_attempt(self) -> bool:
        """True when this attempt is a band re-run (clears the mark); counts the attempt."""
        self.attempts += 1
        pending, self.pending = self.pending, False
        return pending

    def notes(self) -> dict[str, Any]:
        """For the report's ``band`` notes: this attempt asks for a re-run; firings so far; unreachable bounds so far."""
        return {"rerun": self.wanted, "reruns": self.total, "unreachable": self.unreachable}

    def whole(self) -> dict[str, Any]:
        """For the report's notes: ``rerun_all`` when this attempt asks for a re-run (every world goes back)."""
        return {"rerun_all": True} if self.wanted else {}


def read_band(sched: Any) -> float:
    """The coupler report's R2-E3 step, once per attempt (``BandRerun.attempts``): the band bound for the next dt_x
    from the attempt's maxima, and the band re-run's verdict. ``report`` runs again for every ``sim.report()`` /
    ``sim.check()`` / ``alloy.check`` between periods; a repeat returns the bound already read (the maxima were
    cleared by the first read, so reading again would leave the next period an empty band: inf, the clock unbound)."""
    band, rerun, clock = sched._band, sched._band_rerun, sched._alloy_policy.clock
    if getattr(band, "read_attempt", None) == rerun.attempts:
        return float(sched._band_bound)
    band.read_attempt = rerun.attempts
    bound = band.end_period(clock.band_travel)
    rerun.judge(sched._dt_x, bound, sched._control_dt / 2 ** clock.j_max)
    return bound


def _gap(factory: Any) -> float:
    """The pipeline's detection band [m]: the smallest ``soft_contact_gap`` of what it built."""
    gaps = [float(getattr(p, "soft_contact_gap", 0.0) or 0.0) for _v, p, _c, _e in factory.built]
    gaps = [g for g in gaps if g > 0.0]
    return min(gaps) if gaps else 0.0


__all__ = ["BAND_RERUN", "BAND_TRAVEL", "NO_ROWS", "RERUNS_PER_PERIOD", "BandRerun", "ContactBandBound", "band_bound",
           "band_speed_kernel", "read_band", "sheet_speed_kernel"]
