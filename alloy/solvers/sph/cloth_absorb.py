# SPDX-License-Identifier: Apache-2.0
"""Cloth takes up liquid (opt-in): resting liquid particles are taken whole into a per-vertex water mass.

Each absorbing cloth vertex v holds a water mass w_v [kg] up to its capacity c A_v (c in kg/m^2, A_v the vertex's
share of the sheet's rest area). Once per SPH step, after the particles moved:

1. Candidates. A fluid particle whose boundary slots hold a face of an absorbing cloth component, and whose distance
   to that face's surface (the face moved to the step's end) is within ``reach`` (one particle radius), is in
   contact. It becomes a candidate after one step in contact (``touch``), so a fast stream hands its load over
   through the wall forces first. Its candidate face is the nearest qualifying one, if that face may take water this
   step: the faces are coloured so that no two faces of one colour share a vertex, and one colour per step takes
   water, in turn (a per-world step count). So each vertex gets at most one claim per step, and a claim is refused
   only when it does not fit that vertex. (Without the colours, a sheet covered in water claims every vertex from up
   to six faces at once, more than any vertex can take, and refuses everything forever.) The lowest particle index
   wins each face (atomic min): at most one particle per face per step.
2. Claims. Each winner claims its mass on the face's three vertices, split by barycentric weight in int64 fixed point
   (the three shares sum exactly to the particle's mass).
3. Decision. A vertex is over-full when its water plus this step's claims passes its capacity, or the claims pass
   its uptake allowance. The allowance fills at q A_v per second (the rate limit) up to one particle's mass and is
   spent by uptake, so whole particles are taken at q A_v on average. (A per-step limit, q A_v dt, is far below one
   particle's mass at SPH steps: 0.1-0.7 mg against 64 mg at 4 mm spacing, so nothing would ever be taken.) Every
   claim touching an over-full vertex is refused. The decision reads w before any
   update, and integer sums do not depend on order, so the result is the same on the CPU and the GPU and from run
   to run.
4. Uptake. An accepted particle's shares go to w, and to the exchange's per-vertex accumulators M (mass) and
   P (momentum, b m v at uptake), all int64 fixed point. The particle is retired as ABSORBED: ACTIVE cleared, phase
   -4, velocity zero. Its mass is counted per world in ``soaked_mass`` (fixed point) and ``soaked_count``.
   At the exchange end the coupling side (alloy.coupling.liquid_cloth_absorb) merges M and P into the vertex's
   velocity once, (m v + P) / (m + M), writes every mass from w and the phases, and zeroes M and P.

5. Wicking (``wicking`` > 0). At each exchange end the coupling side spreads water along the sheet's edges,
   dw_i/dt = D sum_j k_ij (c_j - c_i), with c = w / A the water per area and k_ij the cotangent weights of the rest
   mesh (the 5-point Laplacian on a regular sheet), over the simulated time since the last exchange end (``elapsed``,
   a per-world device count). Each edge's flux is limited to its share of the sender's water and of the receiver's
   room, so w stays in [0, capacity]. It is applied in int64 from a copy of w, so the order does not matter, and the
   receiver takes it at the sender's velocity through M and P (the sender keeps its velocity).

Books: initial + in = active + out + soaked, exactly in particles. ``pass_start`` keeps the state across the coupling
passes that re-run a step from one start (flags and phase too: a scene without emitters keeps no snapshot of them),
and ``state_arrays`` / ``state_worlds`` list it, with world maps, for the period rollback and frozen worlds.
All arrays are allocated at install, and no step reads the host: graph-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor.newton import ParticleFlags

from .bodies import IMP_SCALE, _fx, _ufx
from .boundary import NB
from .pressure import Particles, is_fluid
from .soft_walls import SoftArrays, soft_closest

ABSORBED = -4  # phase of a particle taken up by cloth (config.py: -2 retired, -3 held)
W_SCALE_HOST = float(2**40)  # water mass fixed point [1/kg]: 9.1e-13 kg resolution
W_SCALE = wp.constant(W_SCALE_HOST)
NO_WINNER = 2**31 - 1
_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@dataclass(frozen=True)
class AbsorbSpec:
    """A fabric's water uptake: capacity [kg/m^2], uptake rate limit [kg/m^2/s], in-plane wicking [m^2/s]."""

    capacity: float
    rate: float
    wicking: float = 0.0

    def __post_init__(self) -> None:
        if not (self.capacity > 0.0 and self.rate > 0.0 and self.wicking >= 0.0):
            raise ValueError(f"absorb needs capacity > 0, rate > 0 and wicking >= 0, got {self}")


@wp.func
def _share(m: float, b: wp.vec3, a: int) -> wp.int64:
    """Vertex a's fixed-point share of mass m at barycentric b; the three shares sum to round(m W_SCALE)."""
    s0 = _fx(m * b[0], W_SCALE)
    s1 = _fx(m * b[1], W_SCALE)
    if a == 0:
        return s0
    if a == 1:
        return s1
    return _fx(m, W_SCALE) - s0 - s1


@wp.kernel(enable_backward=False)
def candidate_kernel(p: Particles, slot_shape: wp.array2d[wp.int32], slot_face: wp.array2d[wp.int32],
                     soft: SoftArrays, absorbs: wp.array[wp.int32], reach: float, dt: float,
                     touch: wp.array[wp.int8], color: wp.array[wp.int32], ncolor: int, tick: wp.array[wp.int32],
                     cand_face: wp.array[wp.int32], cand_bary: wp.array[wp.vec3], win: wp.array[wp.int32],
                     seen: wp.array[wp.int32]):
    """Step 1 (module doc): contact, the one-step gate, the nearest absorbing face, the face's lowest index."""
    i = wp.tid()
    cand_face[i] = -1
    if not is_fluid(p, i):
        touch[i] = wp.int8(0)
        return
    best = int(-1)
    best_d = reach
    best_b = wp.vec3(0.0, 0.0, 0.0)
    for k in range(NB):
        s = slot_shape[i, k]
        f = slot_face[i, k]
        if s < soft.base or f < 0:
            continue
        if absorbs[s - soft.base] == 0:
            continue
        phi, n, xb, vb, b, dist = soft_closest(soft, f, p.q[i], dt)
        if phi <= best_d:
            best = f
            best_d = phi
            best_b = b
    if best < 0:
        touch[i] = wp.int8(0)
        return
    wp.atomic_add(seen, 0, 1)
    if int(touch[i]) < 1:
        touch[i] = wp.int8(1)
        return
    if color[best] != tick[p.world[i]] % ncolor:
        return
    cand_face[i] = best
    cand_bary[i] = best_b
    wp.atomic_add(seen, 1, 1)
    wp.atomic_min(win, best, i)


@wp.kernel(enable_backward=False)
def claim_kernel(p: Particles, soft: SoftArrays, cand_face: wp.array[wp.int32], cand_bary: wp.array[wp.vec3],
                 win: wp.array[wp.int32], claim: wp.array[wp.int64]):
    """Step 2: each face's winner claims its mass on the three vertices."""
    i = wp.tid()
    f = cand_face[i]
    if f < 0 or win[f] != i:
        return
    t = soft.tri[f]
    for a in range(3):
        wp.atomic_add(claim, t[a], _share(float(p.mass[i]), cand_bary[i], a))


@wp.kernel(enable_backward=False)
def elapsed_kernel(dt: float, elapsed: wp.array[wp.float32]):
    w = wp.tid()
    elapsed[w] = elapsed[w] + dt


@wp.kernel(enable_backward=False)
def wick_kernel(edges: wp.array[wp.vec2i], kappa: wp.array[wp.float32], deg: wp.array[wp.int32],
                area: wp.array[wp.float32], vworld: wp.array[wp.int32], elapsed: wp.array[wp.float32],
                w0: wp.array[wp.int64], cap: wp.array[wp.int64], kglobal: wp.array[wp.int32], qd: wp.array[wp.vec3],
                water: wp.array[wp.int64], acc_m: wp.array[wp.int64], acc_p: wp.array2d[wp.int64]):
    """One edge's wicking flux over the elapsed time (module doc, step 5), from the copy w0."""
    e = wp.tid()
    i = edges[e][0]
    j = edges[e][1]
    t = elapsed[vworld[i]]
    ci = _ufx(w0[i], W_SCALE) / area[i]
    cj = _ufx(w0[j], W_SCALE) / area[j]
    F = kappa[e] * t * (ci - cj)
    src = i
    dst = j
    if F < 0.0:
        src = j
        dst = i
        F = -F
    q = _fx(F, W_SCALE)
    q = wp.min(q, w0[src] // wp.int64(deg[src]))
    q = wp.min(q, (cap[dst] - w0[dst]) // wp.int64(deg[dst]))
    if q <= wp.int64(0):
        return
    wp.atomic_add(water, src, -q)
    wp.atomic_add(water, dst, q)
    wp.atomic_add(acc_m, dst, q)
    g = kglobal[src]
    if g >= 0:
        v = qd[g]
        m = _ufx(q, W_SCALE)
        for d in range(3):
            wp.atomic_add(acc_p, dst, d, _fx(m * v[d], IMP_SCALE))


@wp.kernel(enable_backward=False)
def zero_elapsed_kernel(elapsed: wp.array[wp.float32]):
    elapsed[wp.tid()] = 0.0


@wp.kernel(enable_backward=False)
def tick_kernel(tick: wp.array[wp.int32]):
    w = wp.tid()
    tick[w] = tick[w] + 1


@wp.kernel(enable_backward=False)
def refill_kernel(rate: wp.array[wp.float32], dt: float, burst: wp.int64, allow: wp.array[wp.int64]):
    """Each step: the vertex's uptake allowance grows by rate x dt, up to ``burst`` (one particle's mass)."""
    v = wp.tid()
    if rate[v] > 0.0:
        allow[v] = wp.min(allow[v] + _fx(float(rate[v]) * dt, W_SCALE), burst)


@wp.kernel(enable_backward=False)
def decide_kernel(soft: SoftArrays, cand_face: wp.array[wp.int32], win: wp.array[wp.int32],
                  claim: wp.array[wp.int64], water: wp.array[wp.int64], cap: wp.array[wp.int64],
                  allow: wp.array[wp.int64], accept: wp.array[wp.int8], seen: wp.array[wp.int32]):
    """Step 3: accept a winner only if none of its vertices is over-full (reads water before any update)."""
    i = wp.tid()
    accept[i] = wp.int8(0)
    f = cand_face[i]
    if f < 0 or win[f] != i:
        return
    t = soft.tri[f]
    wp.atomic_add(seen, 2, 1)
    for a in range(3):
        v = t[a]
        if water[v] + claim[v] > cap[v] or claim[v] > allow[v]:
            return
    accept[i] = wp.int8(1)
    wp.atomic_add(seen, 3, 1)


@wp.kernel(enable_backward=False)
def uptake_kernel(p: Particles, soft: SoftArrays, cand_face: wp.array[wp.int32], cand_bary: wp.array[wp.vec3],
                  accept: wp.array[wp.int8], water: wp.array[wp.int64], allow: wp.array[wp.int64],
                  acc_m: wp.array[wp.int64], acc_p: wp.array2d[wp.int64], soaked_mass: wp.array[wp.int64],
                  soaked_count: wp.array[wp.int32]):
    """Step 4: water and the exchange's mass / momentum accumulators to the vertices, the particle retired."""
    i = wp.tid()
    if accept[i] == wp.int8(0):
        return
    f = cand_face[i]
    t = soft.tri[f]
    b = cand_bary[i]
    m = float(p.mass[i])
    v = p.qd[i]
    for a in range(3):
        s = _share(m, b, a)
        wp.atomic_add(water, t[a], s)
        wp.atomic_add(allow, t[a], -s)
        wp.atomic_add(acc_m, t[a], s)
        for d in range(3):
            wp.atomic_add(acc_p, t[a], d, _fx(m * b[a] * v[d], IMP_SCALE))
    w = p.world[i]
    wp.atomic_add(soaked_mass, w, _fx(m, W_SCALE))
    wp.atomic_add(soaked_count, w, 1)
    p.flags[i] = p.flags[i] & ~_ACTIVE
    p.phase[i] = wp.int8(ABSORBED)
    p.qd[i] = wp.vec3(0.0, 0.0, 0.0)


def face_colors(tri: np.ndarray) -> np.ndarray:
    """(F,) colours, greedy in face order: no two faces that share a vertex have the same colour."""
    nv = int(tri.max()) + 1 if len(tri) else 0
    taken: list[set[int]] = [set() for _ in range(nv)]
    out = np.zeros(len(tri), np.int32)
    for f, t in enumerate(tri):
        used = taken[t[0]] | taken[t[1]] | taken[t[2]]
        c = 0
        while c in used:
            c += 1
        out[f] = c
        for v in t:
            taken[v].add(c)
    return out


def cot_edges(x: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique edges (E, 2) of the faces and their cotangent weights k_ij = (cot a + cot b) / 2 (a, b: the angles
    opposite the edge), clamped at 0 (an obtuse angle would make the edge pull water uphill)."""
    acc: dict[tuple[int, int], float] = {}
    for t in tri:
        for o in range(3):
            a, b, c = int(t[o]), int(t[(o + 1) % 3]), int(t[(o + 2) % 3])
            u, v = x[b] - x[a], x[c] - x[a]
            sin = np.linalg.norm(np.cross(u, v))
            cot = float(np.dot(u, v) / sin) if sin > 0 else 0.0
            key = (min(b, c), max(b, c))
            acc[key] = acc.get(key, 0.0) + 0.5 * cot
    if not acc:
        return np.zeros((0, 2), np.int32), np.zeros(0, np.float32)
    keys = sorted(acc)
    return np.asarray(keys, np.int32), np.maximum(np.asarray([acc[k] for k in keys], np.float32), 0.0)


def vertex_areas(x: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """(V,) each vertex's share of the faces' area: a third of every incident face's area."""
    a = 0.5 * np.linalg.norm(np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]]), axis=1)
    out = np.zeros(len(x))
    for c in range(3):
        np.add.at(out, tri[:, c], a / 3.0)
    return out


@wp.kernel(enable_backward=False)
def reset_kernel(vworld: wp.array[wp.int32], pworld: wp.array[wp.int32], mask: wp.array[wp.bool], nv: int,
                 water: wp.array[wp.int64], allow: wp.array[wp.int64], acc_m: wp.array[wp.int64],
                 acc_p: wp.array2d[wp.int64], touch: wp.array[wp.int8]):
    """Zero the uptake state of the masked worlds: vertex arrays for i < nv, the touch counter for every particle."""
    i = wp.tid()
    if i < nv and mask[vworld[i]]:
        water[i] = wp.int64(0)
        allow[i] = wp.int64(0)
        acc_m[i] = wp.int64(0)
        for d in range(3):
            acc_p[i, d] = wp.int64(0)
    if i < touch.shape[0] and mask[pworld[i]]:
        touch[i] = wp.int8(0)


class ClothAbsorb:
    """Uptake state and launches for one SPH solver's soft walls (module doc).

    ``capacity`` and ``rate`` are per compact soft vertex, in kg and kg/s: 0 on vertices that take no water
    (pinned, or of a component that does not absorb). ``absorbs`` flags the absorbing components. ``flags`` and
    ``phase`` are the solver's particle flag and phase arrays (pass snapshots)."""

    def __init__(self, soft_walls, absorbs: np.ndarray, capacity: np.ndarray, rate: np.ndarray, reach: float,
                 model, phase: wp.array, worlds: int, burst: float, wicking: np.ndarray | None = None) -> None:
        dev = soft_walls.device
        V, F, P = int(soft_walls.vid.shape[0]), int(soft_walls.count), max(int(model.particle_count), 1)
        self.soft_walls, self.reach, self.device, self.worlds = soft_walls, float(reach), dev, max(int(worlds), 1)
        self.flags, self.phase, self.pworld = model.particle_flags, phase, model.particle_world
        self.vworld = wp.array(model.particle_world.numpy()[soft_walls.vid.numpy()].astype(np.int32),
                               dtype=wp.int32, device=dev)
        self.absorbs = wp.array(np.asarray(absorbs, np.int32), dtype=wp.int32, device=dev)
        colors = face_colors(np.asarray([tuple(t) for t in soft_walls.arrays.tri.numpy()], np.int64))
        self.ncolor = int(colors.max()) + 1 if len(colors) else 1
        self.color = wp.array(colors, dtype=wp.int32, device=dev)
        self.tick = wp.zeros(self.worlds, dtype=wp.int32, device=dev)   # steps taken per world (colour in turn)
        self._init_wicking(model, soft_walls, capacity, wicking)
        self.cap = wp.array(np.round(np.asarray(capacity, np.float64) * W_SCALE_HOST).astype(np.int64),
                            dtype=wp.int64, device=dev)
        self.rate = wp.array(np.asarray(rate, np.float32), dtype=wp.float32, device=dev)
        self.water = wp.zeros(V, dtype=wp.int64, device=dev)
        self.allow = wp.zeros(V, dtype=wp.int64, device=dev)        # uptake allowance (module doc, step 3)
        self.burst = int(round(float(burst) * W_SCALE_HOST))
        self.seen = wp.zeros(4, dtype=wp.int32, device=dev)        # contact, candidate, winner, accepted (all passes)
        self.acc_m = wp.zeros(V, dtype=wp.int64, device=dev)
        self.acc_p = wp.zeros((V, 3), dtype=wp.int64, device=dev)
        self.claim = wp.zeros(V, dtype=wp.int64, device=dev)
        self.win = wp.full(F, NO_WINNER, dtype=wp.int32, device=dev)
        self.touch = wp.zeros(P, dtype=wp.int8, device=dev)
        self.accept = wp.zeros(P, dtype=wp.int8, device=dev)
        self.cand_face = wp.full(P, -1, dtype=wp.int32, device=dev)
        self.cand_bary = wp.zeros(P, dtype=wp.vec3, device=dev)
        self.soaked_mass = wp.zeros(self.worlds, dtype=wp.int64, device=dev)
        self.soaked_count = wp.zeros(self.worlds, dtype=wp.int32, device=dev)
        self._saved = {k: wp.empty_like(a) for k, a in self._pass_state().items()}

    def _init_wicking(self, model, soft_walls, capacity: np.ndarray, wicking: np.ndarray | None) -> None:
        """Edges between vertices that both take water, weighted D x the cotangent weight (``wicking``: D per
        vertex; an edge takes the smaller of its two ends)."""
        dev, vid = self.device, soft_walls.vid.numpy()
        tri = np.asarray([tuple(t) for t in soft_walls.arrays.tri.numpy()], np.int64).reshape(-1, 3)
        x = model.particle_q.numpy()[vid].astype(np.float64)
        edges, k = cot_edges(x, tri)
        D = np.zeros(len(vid)) if wicking is None else np.asarray(wicking, np.float64)
        cap = np.asarray(capacity, np.float64)
        if len(edges):
            a, b = edges[:, 0], edges[:, 1]
            k = k * np.minimum(D[a], D[b])
            keep = (cap[a] > 0) & (cap[b] > 0) & (k > 0)
            edges, k = edges[keep], k[keep].astype(np.float32)
        deg = np.bincount(edges.ravel(), minlength=len(vid)).astype(np.int32)
        self.n_edges = len(edges)
        pad = (lambda arr, shape: arr if len(arr) else np.zeros(shape, arr.dtype))
        self.edges = wp.array(pad(edges, (1, 2)), dtype=wp.vec2i, device=dev)
        self.kappa = wp.array(pad(k, (1,)), dtype=wp.float32, device=dev)
        self.deg = wp.array(np.maximum(deg, 1), dtype=wp.int32, device=dev)
        self.area = wp.array(np.maximum(vertex_areas(x, tri), 1e-12).astype(np.float32), dtype=wp.float32, device=dev)
        self.elapsed = wp.zeros(self.worlds, dtype=wp.float32, device=dev)
        self.w0 = wp.zeros(len(vid), dtype=wp.int64, device=dev)

    def wick(self, kglobal: wp.array, qd: wp.array) -> None:
        """The exchange-end wicking (module doc, step 5); device launches only."""
        if self.n_edges == 0:
            return
        wp.copy(self.w0, self.water)
        wp.launch(wick_kernel, dim=self.n_edges, device=self.device,
                  inputs=[self.edges, self.kappa, self.deg, self.area, self.vworld, self.elapsed, self.w0, self.cap,
                          kglobal, qd, self.water, self.acc_m, self.acc_p])
        wp.launch(zero_elapsed_kernel, dim=self.worlds, device=self.device, inputs=[self.elapsed])

    def step(self, p: Particles, slots, n: int, dt: float) -> None:
        """One SPH step's uptake (module doc, steps 1-4); call after the particles moved, before flags mirror."""
        if n == 0:
            return
        sw, dev = self.soft_walls, self.device
        self.win.fill_(NO_WINNER)
        self.claim.zero_()
        wp.launch(tick_kernel, dim=self.worlds, device=dev, inputs=[self.tick])
        if self.n_edges:
            wp.launch(elapsed_kernel, dim=self.worlds, device=dev, inputs=[float(dt), self.elapsed])
        wp.launch(refill_kernel, dim=self.rate.shape[0], device=dev,
                  inputs=[self.rate, float(dt), wp.int64(self.burst), self.allow])
        wp.launch(candidate_kernel, dim=n, device=dev,
                  inputs=[p, slots.shape, slots.face, sw.arrays, self.absorbs, self.reach, float(dt), self.touch,
                          self.color, self.ncolor, self.tick, self.cand_face, self.cand_bary, self.win, self.seen])
        wp.launch(claim_kernel, dim=n, device=dev,
                  inputs=[p, sw.arrays, self.cand_face, self.cand_bary, self.win, self.claim])
        wp.launch(decide_kernel, dim=n, device=dev,
                  inputs=[sw.arrays, self.cand_face, self.win, self.claim, self.water, self.cap, self.allow,
                          self.accept, self.seen])
        wp.launch(uptake_kernel, dim=n, device=dev,
                  inputs=[p, sw.arrays, self.cand_face, self.cand_bary, self.accept, self.water, self.allow,
                          self.acc_m, self.acc_p, self.soaked_mass, self.soaked_count])

    # ------------------------------------------------------------------ restart passes, rollback, reset
    def _pass_state(self) -> dict[str, wp.array]:
        return {"water": self.water, "allow": self.allow, "tick": self.tick, "elapsed": self.elapsed,
                "acc_m": self.acc_m, "acc_p": self.acc_p, "touch": self.touch, "soaked_mass": self.soaked_mass,
                "soaked_count": self.soaked_count, "flags": self.flags, "phase": self.phase}

    def pass_start(self, restart: bool) -> None:
        """A coupling pass starts: a restart restores the exchange's first-pass state, else it is saved."""
        for k, a in self._pass_state().items():
            if restart:
                wp.copy(a, self._saved[k])
            else:
                wp.copy(self._saved[k], a)

    def state_arrays(self) -> dict[str, wp.array]:
        """Arrays for the period rollback and frozen worlds (the solver's state adapter); phase and flags are the
        solver's own entries."""
        return {"sph.absorb_water": self.water, "sph.absorb_allow": self.allow, "sph.absorb_m": self.acc_m,
                "sph.absorb_p": self.acc_p, "sph.absorb_touch": self.touch, "sph.absorb_tick": self.tick,
                "sph.absorb_soaked_mass": self.soaked_mass, "sph.absorb_soaked_count": self.soaked_count,
                "sph.absorb_elapsed": self.elapsed}

    def state_worlds(self) -> dict[str, wp.array]:
        """Item -> world maps of the per-item arrays above (the per-world counters are world-major)."""
        return {"sph.absorb_water": self.vworld, "sph.absorb_allow": self.vworld, "sph.absorb_m": self.vworld,
                "sph.absorb_p": self.vworld, "sph.absorb_touch": self.pworld}

    def reset(self, world_mask: wp.array | None) -> None:
        """Zero the uptake state of the masked worlds (all without a mask); the counters too."""
        mask = world_mask if world_mask is not None else wp.full(self.worlds, True, dtype=wp.bool,
                                                                    device=self.device)
        n = max(int(self.water.shape[0]), int(self.touch.shape[0]))
        wp.launch(reset_kernel, dim=n, device=self.device,
                  inputs=[self.vworld, self.pworld, mask, int(self.water.shape[0]), self.water, self.allow,
                          self.acc_m, self.acc_p, self.touch])
        m = mask.numpy()[: self.worlds].astype(bool)
        for arr in (self.soaked_mass, self.soaked_count, self.tick, self.elapsed):
            a = arr.numpy()
            a[: self.worlds][m] = 0
            arr.assign(a)

    def diagnostics(self) -> dict[str, int]:
        """Host: cumulative counts over every step and pass run so far (re-run passes count again)."""
        c = self.seen.numpy()
        return {"contacts": int(c[0]), "candidates": int(c[1]), "winners": int(c[2]), "accepted": int(c[3])}

    def water_kg(self) -> np.ndarray:
        """Host: water per compact soft vertex [kg]."""
        return self.water.numpy().astype(np.float64) / W_SCALE_HOST

    def soaked(self) -> tuple[np.ndarray, np.ndarray]:
        """Host: (absorbed mass [kg], absorbed particle count) per world."""
        return self.soaked_mass.numpy().astype(np.float64) / W_SCALE_HOST, self.soaked_count.numpy().copy()


__all__ = ["ABSORBED", "AbsorbSpec", "ClothAbsorb", "W_SCALE_HOST", "candidate_kernel", "claim_kernel",
           "cot_edges", "decide_kernel", "elapsed_kernel", "face_colors", "refill_kernel", "reset_kernel",
           "tick_kernel", "uptake_kernel", "vertex_areas", "wick_kernel", "zero_elapsed_kernel"]
