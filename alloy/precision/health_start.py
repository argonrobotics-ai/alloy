# SPDX-License-Identifier: Apache-2.0
"""Start overlaps and fast spins in health (SCR-WP5-14 and its addendum, 2026-11-08).

**Start overlaps (START_PEN; WP13's find).** Bodies placed overlapping (YCB cans 16-22 mm inside a bin wall) were
never judged: health measures contact depth on each check step's end state, after the solver has pushed the overlap
out, and on the MuJoCo route its depths raise no PEN flag at all (SCR-SC0-1). At the first ``update`` (eager in
IntegratedSimulation: the first period of every graph key runs uncaptured) the START state's contacts are judged once,
with the plain rule (depth against pen_warn / pen_fail, no thin-part rule). The judge reads the host. A caller that
captures its first period into a CUDA graph (WP10's SliceSim, CUDA error 906 on rc-1109) gets no START_PEN for that
run, one warning, and no host read; a later update does not judge a state that is no longer the start:
- Newton's contacts, computed on ``state_in``: depth = -(n . (X_b p1 - X_a p0) - margin0 - margin1);
- each SolverMuJoCo's own contacts (``d.contact.dist``), collided at the start of that first step.
Shapes marked by ``HealthMonitor.relax_thin_part_rule`` (cutters: their thickness set to inf after the build) are left
out; a plane or height field, inf thick by nature, is judged (OSS-18's agent smoke 11-09: a prism started 28 mm into
the ground raised no START_PEN, since the ground's own inf read as a cutter). Per world the deepest overlap and its
two shapes are kept for the whole run (``depth``, ``shapes``):
- deeper than pen_warn (1 mm): START_PEN on every check (DEGRADED; the world keeps running);
- deeper than pen_fail (1 cm): ``sim.check()`` is not ok (``alloy.api.health``); INVALID only when
  ``CheckPolicy.start_pen_invalid`` (the precision preset): PEN_FAIL at the first check, then the latch.

**Fast spin (SPIN; WP3's marker report, notes/wp3-marker/REPORT.md).** A thin body spun by contacts to 0.4-0.7 rad
per step gains energy from contacts linearised over that rotation and is flung about 15 ms later. At each check with a
step length, any dynamic body turning more than ``CheckPolicy.spin_rad_per_step`` (0.5 rad) in one step (|omega| dt)
raises SPIN (DEGRADED) for that check. The run's fastest spin per world (rad per step, body) is kept. Bodies with no
inverse mass (kinematic, static) and bodies a rig marks with ``relax_spin`` (a wheel, a spindle) are left out. Argmax
by packed (value bits, index) keys: the same body every run.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import warp as wp

from alloy.precision.types import HealthFlags, StepHealth

log = logging.getLogger("alloy.precision")

FL_START_PEN = wp.constant(wp.uint32(int(HealthFlags.START_PEN)))
FL_SPIN = wp.constant(wp.uint32(int(HealthFlags.SPIN)))
SEV_DEGRADED = wp.constant(1)


@wp.kernel
def _spin_keys(body_qd: wp.array[wp.spatial_vector], body_world: wp.array[wp.int32], judged: wp.array[wp.int32],
               dt: float, keys: wp.array[wp.uint64]):
    """keys[w] = max over the world's judged bodies of (bits(|omega| dt) << 32 | body)."""
    b = wp.tid()
    w = body_world[b]
    if w < 0 or judged[b] == 0:
        return
    rad = wp.length(wp.spatial_bottom(body_qd[b])) * dt
    if wp.isfinite(rad) and rad > 0.0:
        wp.atomic_max(keys, w, (wp.uint64(wp.cast(rad, wp.uint32)) << wp.uint64(32)) | wp.uint64(b))


@wp.kernel
def _apply(keys: wp.array[wp.uint64], limit: float, start: wp.array[wp.int32], peak: wp.array[wp.float32],
           peak_body: wp.array[wp.int32], record: wp.array[StepHealth]):
    """After ``classify``: this check's SPIN and the sticky START_PEN into the record (at least DEGRADED); the run's
    fastest spin kept; this check's keys cleared."""
    w = wp.tid()
    key = keys[w]
    keys[w] = wp.uint64(0)
    rad = wp.cast(wp.uint32(key >> wp.uint64(32)), wp.float32)
    h = record[w]
    flags = wp.uint32(0)
    if key != wp.uint64(0):
        if rad > peak[w]:
            peak[w] = rad
            peak_body[w] = wp.int32(key & wp.uint64(0xFFFFFFFF))
        if rad > limit:
            flags = flags | FL_SPIN
    if start[w] != 0:
        flags = flags | FL_START_PEN
    if flags != wp.uint32(0):
        h.flags = h.flags | flags
        if h.severity < SEV_DEGRADED:
            h.severity = SEV_DEGRADED
        record[w] = h


@wp.kernel
def _start_fail(start: wp.array[wp.int32], pen_fail: wp.array[wp.int32]):
    """Precision preset: a start overlap past pen_fail is PEN_FAIL at the first check (INVALID, latched)."""
    w = wp.tid()
    if start[w] == 2:
        pen_fail[w] = 3                      # at the first check (health_pen: above NATIVE_HARD), no dwell


def capturing(device: Any) -> bool:
    """True while a CUDA graph is being captured on ``device``: no host reads then."""
    return bool(wp.get_device(device).is_capturing)


def _sphere_only(model: Any, nb: int) -> np.ndarray:
    """[nb] bodies whose collision shapes are all spheres (at least one): turning one leaves its contacts as they
    were, so a fast roll (a 5 mm marble at 2.5 m/s: 0.625 rad per 1.25 ms step) is no SPIN."""
    if not nb or not int(getattr(model, "shape_count", 0)):
        return np.zeros(nb, bool)
    from alloy._vendor.newton import GeoType, ShapeFlags  # noqa: PLC0415 - keeps this module's import light

    body, typ, flags = model.shape_body.numpy(), model.shape_type.numpy(), model.shape_flags.numpy()
    col = (body >= 0) & ((flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0)
    n_col = np.bincount(body[col], minlength=nb)[:nb]
    n_sph = np.bincount(body[col & (typ == int(GeoType.SPHERE))], minlength=nb)[:nb]
    return (n_col > 0) & (n_col == n_sph)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` [N, 3] by unit quaternions ``q`` [N, 4] (x, y, z, w)."""
    u, s = q[:, :3], q[:, 3:4]
    return 2.0 * np.sum(u * v, axis=1, keepdims=True) * u + (s * s - np.sum(u * u, axis=1, keepdims=True)) * v \
        + 2.0 * s * np.cross(u, v)


class StartChecks:
    """START_PEN (once, at the first update) and SPIN (each check) for a ``HealthMonitor`` (see the module doc)."""

    def __init__(self, model: Any, cfg: Any, thresholds: Any, world_count: int, device: Any,
                 thick: wp.array | None = None):
        self.model, self.W, self.device = model, int(world_count), device
        built = thick.numpy() if thick is not None and thick.shape[0] else np.zeros(0, np.float32)
        self._inf_built = np.isinf(built)                       # planes, height fields: inf thick, never cutters
        self.warn, self.fail = float(thresholds.pen_warn_m), float(thresholds.pen_fail_m)
        self.invalid = bool(getattr(cfg, "start_pen_invalid", False))
        self.spin_limit = float(getattr(cfg, "spin_rad_per_step", 0.5))
        self.pending = True
        self.skipped = False                                   # the first update ran under capture: not judged
        self.depth = np.full(self.W, np.nan)                   # [W] m, deepest start overlap; NaN: none measured
        self.shapes = np.full((self.W, 2), -1, np.int32)       # [W, 2] its two shapes
        self.fails: dict[int, dict[tuple[int, int], float]] = {}   # world -> {pair: depth} past pen_fail (WP13)
        nb = int(model.body_count)
        inv = model.body_inv_mass.numpy() if nb else np.zeros(0)
        judged = (np.asarray(inv) > 0) & ~_sphere_only(model, nb)      # WP3: a turning sphere keeps its contacts
        self._judged = wp.array(judged.astype(np.int32), dtype=wp.int32, device=device)
        self.rigid_substeps: Callable[[], int] | None = None   # set by wiring: rigid steps per exchange (WP3)
        self._keys = wp.zeros(self.W, dtype=wp.uint64, device=device)
        self._start = wp.zeros(self.W, dtype=wp.int32, device=device)      # 0 none, 1 > warn, 2 > fail
        self.peak = wp.zeros(self.W, dtype=wp.float32, device=device)      # [W] rad per step, the run's fastest
        self.peak_body = wp.full(self.W, -1, dtype=wp.int32, device=device)

    def relax_spin(self, bodies: Sequence[int]) -> None:
        """Bodies that spin by design (a wheel, a spindle): never SPIN."""
        judged = self._judged.numpy()
        judged[np.asarray(list(bodies), dtype=np.int64)] = 0
        self._judged.assign(judged)

    # ----- START_PEN -----
    def first(self, state_in: Any, contacts: Any, mjw_sources: Sequence[Any], thick: wp.array, acc: Any) -> None:
        """Judge the start state's contacts (eager, once). ``thick``: health's per-shape thickness (inf since the build:
        a cutter);
        ``acc``: this check's accumulators (None on a step that is not a check step: no precision PEN_FAIL then)."""
        self.pending = False
        if capturing(self.device):                  # a caller captured its first period: judging reads the host
            self.skipped = True
            log.warning("health: start overlaps (START_PEN) not judged for this run: its first period was captured "
                        "into a CUDA graph; run the first period eagerly to judge them")
            return
        relaxed = np.isinf(thick.numpy()) if thick.shape[0] else np.zeros(0, bool)
        if self._inf_built.shape == relaxed.shape:              # cutters only: inf set after the build
            relaxed &= ~self._inf_built
        if contacts is not None and getattr(contacts, "rigid_contact_max", 0):
            self._note(*self._newton(state_in, contacts), relaxed)
        for src in mjw_sources:
            self._note(*self._mujoco(src), relaxed)
        level = np.where(self.depth > self.fail, 2, np.where(self.depth > self.warn, 1, 0)).astype(np.int32)
        self._start.assign(level)
        if self.invalid and acc is not None and (level == 2).any():
            wp.launch(_start_fail, dim=self.W, inputs=[self._start, acc.pen_fail], device=self.device)

    def _newton(self, state: Any, c: Any) -> tuple[np.ndarray, ...]:
        n = min(int(c.rigid_contact_count.numpy()[0]), int(c.rigid_contact_shape0.shape[0]))  # overflow
        s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
        keep = (s0 >= 0) & (s1 >= 0)
        s0, s1 = s0[keep], s1[keep]
        body = self.model.shape_body.numpy()
        q = state.body_q.numpy() if state.body_q is not None and self.model.body_count else np.zeros((0, 7))

        def world_point(shapes: np.ndarray, p: np.ndarray) -> np.ndarray:
            b = body[shapes]
            out = p.astype(np.float64).copy()
            dyn = b >= 0
            if dyn.any():
                X = q[b[dyn]].astype(np.float64)
                out[dyn] = X[:, :3] + _rotate(X[:, 3:], p[dyn].astype(np.float64))
            return out

        pa = world_point(s0, c.rigid_contact_point0.numpy()[:n][keep])
        pb = world_point(s1, c.rigid_contact_point1.numpy()[:n][keep])
        nrm = c.rigid_contact_normal.numpy()[:n][keep].astype(np.float64)
        margin = c.rigid_contact_margin0.numpy()[:n][keep] + c.rigid_contact_margin1.numpy()[:n][keep]
        depth = -(np.sum(nrm * (pb - pa), axis=1) - margin)
        sw = self.model.shape_world.numpy()
        w = np.where(sw[s0] >= 0, sw[s0], sw[s1])
        return w, depth, s0, s1

    def _mujoco(self, src: Any) -> tuple[np.ndarray, ...]:
        d = src.data
        n = min(int(d.nacon.numpy()[0]), int(d.contact.dist.shape[0]))    # past the buffer on overflow
        w = d.contact.worldid.numpy()[:n]
        g = d.contact.geom.numpy()[:n]
        depth = -d.contact.dist.numpy()[:n].astype(np.float64)
        gs = src.geom_shape.numpy()
        if gs.size == 0:
            return w, depth, np.full(n, -1, np.int32), np.full(n, -1, np.int32)
        row = gs[w % gs.shape[0]]
        ga, gb = g[:, 0], g[:, 1]
        sa = np.where((ga >= 0) & (ga < gs.shape[1]), row[np.arange(n), np.clip(ga, 0, gs.shape[1] - 1)], -1)
        sb = np.where((gb >= 0) & (gb < gs.shape[1]), row[np.arange(n), np.clip(gb, 0, gs.shape[1] - 1)], -1)
        return w, depth, sa, sb

    def _note(self, w: np.ndarray, depth: np.ndarray, sa: np.ndarray, sb: np.ndarray, relaxed: np.ndarray) -> None:
        cut = np.zeros(len(depth), bool)
        if relaxed.size:
            cut = (sa >= 0) & relaxed[np.clip(sa, 0, relaxed.size - 1)] | \
                  (sb >= 0) & relaxed[np.clip(sb, 0, relaxed.size - 1)]
        idx = np.flatnonzero((w >= 0) & (w < self.W) & np.isfinite(depth) & ~cut)
        if idx.size == 0:
            return
        idx = idx[np.lexsort((idx, -depth[idx], w[idx]))]     # per world: deepest first, ties by contact index
        for k in idx[depth[idx] > self.fail].tolist():          # every pair past pen_fail, deepest per pair
            pair = (min(int(sa[k]), int(sb[k])), max(int(sa[k]), int(sb[k])))
            pairs = self.fails.setdefault(int(w[k]), {})
            pairs[pair] = max(pairs.get(pair, 0.0), float(depth[k]))
        worlds, first = np.unique(w[idx], return_index=True)
        best = idx[first]
        for wk, k in zip(worlds.tolist(), best.tolist()):       # one per world
            if not float(depth[k]) <= self.depth[wk]:           # NaN (unset) or deeper than another source's
                self.depth[wk], self.shapes[wk] = float(depth[k]), (int(sa[k]), int(sb[k]))

    # ----- SPIN + sticky START_PEN, each check -----
    def apply(self, state_out: Any, body_world: wp.array, dt: float | None, record: wp.array) -> None:
        if dt is not None and self.model.body_count and state_out.body_qd is not None:
            k = max(int(self.rigid_substeps()), 1) if self.rigid_substeps is not None else 1     # dt: one exchange
            wp.launch(_spin_keys, dim=self.model.body_count,
                      inputs=[state_out.body_qd, body_world, self._judged, float(dt) / k, self._keys],
                      device=self.device)
        wp.launch(_apply, dim=self.W, inputs=[self._keys, self.spin_limit, self._start, self.peak, self.peak_body,
                                              record], device=self.device)

    def report_fields(self) -> dict[str, Any]:
        """For ``HealthReport``: the start overlaps and the run's fastest spins (host copies)."""
        m = self.model
        pairs = {w: sorted(((a, b, d) for (a, b), d in ps.items()), key=lambda x: -x[2])
                 for w, ps in self.fails.items()}
        return {"start_pen_m": self.depth.copy(), "start_shapes": self.shapes.copy(),
                "start_level": self._start.numpy(), "start_pairs": pairs,
                "spin_limit": self.spin_limit,
                "spin_rad": self.peak.numpy().astype(np.float64), "spin_body": self.peak_body.numpy().copy(),
                "shape_names": tuple(getattr(m, "shape_label", None) or ()),
                "body_names": tuple(getattr(m, "body_label", None) or ())}


__all__ = ["StartChecks"]
