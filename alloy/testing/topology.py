# SPDX-License-Identifier: Apache-2.0
"""Test helpers for alloy.topology (WP10): scenes, VBD array capture, the TOP-3 cut fuzz and a twin runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.topology.meshes import grid_tet_mesh, grid_tri_mesh, tet_volumes, tri_areas  # noqa: F401  (re-export)


def vbd_topology_arrays(model: newton.Model, solver: newton.solvers.SolverVBD | None = None) -> dict[str, np.ndarray]:
    """Host copies of every array that encodes VBD adjacency and colouring (compared bitwise by the tests)."""
    out: dict[str, np.ndarray] = {}
    adj = getattr(model, "soft_mesh_adjacency_device", None)
    for owner, name in ((adj, "adjacency"), (getattr(solver, "particle_adjacency", None), "solver_adjacency")):
        if owner is None:
            continue
        for attr in sorted(vars(owner) if hasattr(owner, "__dict__") else dir(owner)):
            val = getattr(owner, attr, None)
            if isinstance(val, wp.array):
                out[f"{name}.{attr}"] = val.numpy().copy()
    for k, grp in enumerate(model.particle_color_groups or []):
        out[f"color_group.{k}"] = grp.numpy().copy()
    if getattr(model, "particle_colors", None) is not None:
        out["particle_colors"] = model.particle_colors.numpy().copy()
    for attr in ("tri_indices", "tet_indices", "edge_indices", "spring_indices"):
        arr = getattr(model, attr, None)
        if arr is not None:
            out[attr] = arr.numpy().copy()
    return out


def clamp_particles(model: newton.Model, ids: np.ndarray) -> None:
    """Zero mass and inverse mass so VBD keeps these particles where the test puts them."""
    for name in ("particle_mass", "particle_inv_mass"):
        arr = getattr(model, name).numpy()
        arr[ids] = 0.0
        getattr(model, name).assign(arr)


def cloth_kwargs(*, tri_ke: float = 1.0e3, tri_ka: float = 0.0, tri_kd: float = 0.0, density: float = 0.2,
                 edge_ke: float = 0.0, pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    return dict(pos=wp.vec3(*pos), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0), density=density,
                tri_ke=tri_ke, tri_ka=tri_ka, tri_kd=tri_kd, edge_ke=edge_ke, edge_kd=0.0, particle_radius=1.0e-3)


def soft_kwargs(*, k_mu: float = 2.0e5, k_lambda: float = 2.0e5, k_damp: float = 0.0, density: float = 1000.0,
                pos: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> dict:
    return dict(pos=wp.vec3(*pos), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0), density=density,
                k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp, particle_radius=1.0e-3)


@dataclass
class BlockScene:
    """A pre-split tet block clamped at x = 0, with one blade body per world (kinematic; posed by the test)."""

    model: newton.Model
    blades: list
    blade_bodies: np.ndarray        # [W] body index of the blade in each world
    clamp: np.ndarray               # clamped particle ids
    size: tuple[float, float, float]
    bonds_per_world: int


def presplit_block(device: str, *, cells: tuple[int, int, int] = (6, 4, 4), cell: float = 5.0e-3,
                   plan=None, worlds: int = 1, failure=None, gravity: float = -9.81,
                   k_mu: float = 2.0e5, k_lambda: float = 2.0e5) -> BlockScene:
    from alloy.topology.emit import add_presplit_soft
    from alloy.topology.presplit import SplitPlan
    from alloy.topology.types import Blade, TopologyBudget

    t = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    v, tets = grid_tet_mesh(*cells, cell)
    plan = plan or SplitPlan(spacing=2 * cell, axes=(0, 1))
    rec = add_presplit_soft(t, v, tets, plan, soft_kwargs(k_mu=k_mu, k_lambda=k_lambda),
                            budget=TopologyBudget(spare_vertices=100_000), failure=failure)
    blade = t.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
    t.add_shape_box(blade, hx=0.0005, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    b.replicate(t, worlds)
    b.color()
    m = b.finalize(device=device)
    x0 = m.particle_q.numpy()
    clamp = np.flatnonzero(np.isclose(x0[:, 0], 0.0))
    clamp_particles(m, clamp)
    bodies = np.flatnonzero(m.body_world.numpy() >= 0)
    blade_bodies = bodies[np.argsort(m.body_world.numpy()[bodies], kind="stable")]
    blades = [Blade(body=int(bb), shape=-1, edge_a=(0.0, -0.05, -0.02), edge_b=(0.0, 0.05, -0.02))
              for bb in blade_bodies]
    size = tuple(float(n * cell) for n in cells)
    return BlockScene(m, blades, blade_bodies, clamp, size, len(rec.bonds))


def blade_pose(p: np.ndarray, edge_dir: np.ndarray, motion_dir: np.ndarray) -> np.ndarray:
    """(x, y, z, qx, qy, qz, qw): body y along edge_dir, body -z (edge side) along motion_dir."""
    y = edge_dir / np.linalg.norm(edge_dir)
    z = -motion_dir + np.dot(motion_dir, y) * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    q = wp.quat_from_matrix(wp.mat33(*np.stack([x, y, z], axis=1).ravel().tolist()))
    return np.array([*p, q[0], q[1], q[2], q[3]], dtype=np.float32)


# ---------------------------------------------------------------------------------------------------------------
# TOP-3 cut fuzz (shared by test_cut_fuzz and test_fixed_arrays; see test_cut_fuzz.py for the protocol).

FUZZ_WORLDS, FUZZ_STEPS, FUZZ_DT = 16, 10, 1.0e-3
FUZZ_CELLS = (4, 3, 3)
# Cut parameters of the fuzz blades. The fuzz blade sweeps ~4 mm per 1 ms step, so a bond sees it for about one
# step; softness 8 (a fast-cutting blade) takes most touched bonds to zero stiffness in that step, which drives the
# cut -> commit -> cap path hard. cut_spring_ke is the Blade default (docs/notes/cleanroom-cut-log.md D9).
FUZZ_BLADE: dict[str, float] = {"softness": 8.0, "cut_spring_ke": 500.0}


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def random_stroke(rng: np.random.Generator, size: tuple[float, float, float]) -> np.ndarray:
    """[FUZZ_STEPS, 7] blade poses: a random line through the block, random edge direction."""
    lo, hi = np.zeros(3), np.asarray(size)
    centre = rng.uniform(lo, hi)
    motion = _unit(rng)
    edge = _unit(rng)
    edge -= np.dot(edge, motion) * motion
    edge /= max(np.linalg.norm(edge), 1e-9)
    reach = 0.6 * np.linalg.norm(hi - lo)
    ts = np.linspace(-1.0, 1.0, FUZZ_STEPS)
    return np.stack([blade_pose(centre + t * reach * motion, edge, motion) for t in ts])


@dataclass
class _FuzzRig:
    scene: BlockScene
    ctl: object
    solver: newton.solvers.SolverVBD
    snap: object
    states: list
    control: newton.Control
    world_bonds: list


def _fuzz_rig(device: str):
    from alloy.cards.types import Failure
    from alloy.core.quantity import Q
    from alloy.core.snapshot import WorldSnapshot
    from alloy.topology.bonds import bonds_from_model
    from alloy.topology.controller import TopologyController
    from alloy.topology.types import Blade, TopologyBudget

    fail = Failure(fracture_strain=Q(0.03, "model", note="TOP-3 fuzz value"))
    sc = presplit_block(device, worlds=FUZZ_WORLDS, cells=FUZZ_CELLS, failure=fail, k_mu=5.0e4, k_lambda=5.0e4)
    m = sc.model
    blades = [Blade(b.body, b.shape, b.edge_a, b.edge_b, **FUZZ_BLADE) for b in sc.blades]
    bonds = bonds_from_model(m)
    ctl = TopologyController(m, bonds, TopologyBudget(max_events_per_step=4), blades)
    s0, s1 = m.state(), m.state()
    snap = WorldSnapshot(m, s0, extra=ctl.state_arrays())
    snap.save(s0)
    return _FuzzRig(sc, ctl, newton.solvers.SolverVBD(m, iterations=4), snap, [s0, s1], m.control(),
                    np.split(np.arange(bonds.count), FUZZ_WORLDS))


def _fuzz_round(rig: _FuzzRig, rng: np.random.Generator, stats: dict[str, float]) -> None:
    from alloy.topology.cut import PHASE_CUTTING

    s0, s1 = rig.states
    rig.snap.restore(s0)
    pre = np.concatenate([rng.choice(ids, size=rng.integers(0, int(0.4 * len(ids)) + 1), replace=False)
                          for ids in rig.world_bonds])
    rig.ctl.preset_broken(pre)
    paths = [random_stroke(rng, rig.scene.size) for _ in range(FUZZ_WORLDS)]
    for k in range(FUZZ_STEPS):
        bq = s0.body_q.numpy()
        bq[rig.scene.blade_bodies] = np.stack([p[k] for p in paths])
        s0.body_q.assign(bq)
        rig.ctl.pre_step(s0, None, FUZZ_DT)
        rig.solver.step(s0, s1, rig.control, None, FUZZ_DT)
        rig.ctl.apply(s1)
        s0, s1 = s1, s0
    rig.states = [s0, s1]
    led = rig.ctl.ledger()
    q = s0.particle_q.numpy().reshape(FUZZ_WORLDS, -1, 3)
    for key, val in (("preset", len(pre)), ("entered_cutting", int((rig.ctl.phase.numpy() == PHASE_CUTTING).sum())),
                     ("blade_cuts", int(rig.ctl.cut_done.numpy().sum())), ("events", int(led.events.sum())),
                     ("overflow", int(led.overflow.sum())), ("strokes", FUZZ_WORLDS),
                     ("nan_strokes", int((~np.isfinite(q).all(axis=(1, 2))).sum()))):
        stats[key] = stats.get(key, 0) + val


def run_cut_fuzz(device: str, strokes: int, seed: int = 0) -> dict[str, float]:
    """Run the TOP-3 protocol; returns counts (strokes, NaN strokes, events, overflow, blade cuts, ...)."""
    rig = _fuzz_rig(device)
    before = vbd_topology_arrays(rig.scene.model, rig.solver)
    rng = np.random.default_rng(seed)
    stats: dict[str, float] = {}
    for _ in range(int(np.ceil(strokes / FUZZ_WORLDS))):
        _fuzz_round(rig, rng, stats)
    after = vbd_topology_arrays(rig.scene.model, rig.solver)
    stats["vbd_arrays_unchanged"] = float(before.keys() == after.keys() and
                                          all(np.array_equal(before[k], after[k]) for k in before))
    return stats


# ---------------------------------------------------------------------------------------------------------------
# A minimal TwinLog and runner for SimTwins (WP6's run_law / DeviceTwinLog replace these once they land).


class ArrayTwinLog:
    """TwinLog (alloy.bench.types) backed by [W, T] device arrays, sampling every ``every`` steps."""

    def __init__(self, n_worlds: int, dt: float, steps: int, every: int = 1, device: str = "cpu") -> None:
        self.n_worlds, self.dt, self._every, self._device = n_worlds, dt, every, device
        self._cols = (steps + every - 1) // every
        self._channels: dict[str, wp.array] = {}

    def column(self, step: int) -> int:
        return step // self._every if step % self._every == 0 and step // self._every < self._cols else -1

    def times(self) -> np.ndarray:
        return np.arange(self._cols) * self._every * self.dt

    def channel(self, name: str, dtype=wp.float32) -> wp.array:
        if name not in self._channels:
            self._channels[name] = wp.zeros((self.n_worlds, self._cols), dtype=dtype, device=self._device)
        return self._channels[name]

    def accumulator(self, name: str, dtype=wp.float64) -> wp.array:
        key = f"acc:{name}"
        if key not in self._channels:
            self._channels[key] = wp.zeros(self.n_worlds, dtype=dtype, device=self._device)
        return self._channels[key]

    def series(self, name: str) -> np.ndarray:
        return self._channels[name].numpy()


def run_twin(twin, cond, samples: list[dict], *, dt: float, steps: int, device: str) -> dict[str, np.ndarray]:
    """One world per sample: build, make solver and pipeline, then drive/collide/step/record; returns scalars."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for sample in samples:
        b.begin_world()
        twin.build_world(b, cond, sample)
        b.end_world()
    b.color()
    m = b.finalize(device=device)
    solver = twin.make_solver(m, "vbd")
    pipe = twin.make_pipeline(m, solver)
    contacts = pipe.contacts()
    log = ArrayTwinLog(len(samples), dt, steps, device=device)
    s0, s1, ctrl = m.state(), m.state(), m.control()
    for k in range(steps):
        twin.drive(k, k * dt, s0, ctrl)
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, dt)
        twin.record(k, s1, contacts, None, log)
        s0, s1 = s1, s0
    return twin.scalars(log)


# ---------------------------------------------------------------------------------------------------------------
# SCR-SC5-2 seam press (C9-like): three slices are cut apart, pulled away, then pressed back onto their former
# partners by a kinematic plate with a per-world small sideways shift and tilt (less than a triangle).

PRESS_CELLS, PRESS_CELL, PRESS_DT = (9, 4, 4), 5.0e-3, 1.0e-3
PRESS_SEAMS = (0.0151, 0.0301)                  # [m] seam planes x = const: three 15 mm slices
PRESS_MARGIN, PRESS_GAP, PRESS_RADIUS = 5.0e-4, 2.5e-4, 2.0e-4   # VBD self-contact; C9's rest-shape radius


@wp.kernel
def _piece_query(mesh: wp.uint64, pts: wp.array[wp.vec3], depth: wp.array[float], dist: wp.array[float]):
    """Per point: depth inside the closed surface (0 outside) and distance to it (0 inside, 1 m when no hit)."""
    i = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 1.0, 0.5)
    inside, outside = float(0.0), float(1.0)
    if q.result:
        outside = wp.length(wp.mesh_eval_position(mesh, q.face, q.u, q.v) - pts[i])
        if q.sign < 0.0:
            inside, outside = outside, float(0.0)
    depth[i] = inside
    dist[i] = outside


def _components(elems: np.ndarray, n: int) -> np.ndarray:
    """Smallest vertex id of each vertex's connected component over ``elems`` (pre-split pieces own their copies)."""
    label = np.arange(n)
    e = np.concatenate([elems[:, [0, k]] for k in range(1, elems.shape[1])])
    while True:
        lo = np.minimum(label[e[:, 0]], label[e[:, 1]])
        new = label.copy()
        np.minimum.at(new, e[:, 0], lo)
        np.minimum.at(new, e[:, 1], lo)
        new = new[new]
        if np.array_equal(new, label):
            break
        label = new
    return label


def surface_pieces(tris: np.ndarray, n: int) -> list[np.ndarray]:
    """Triangles of each connected surface, ordered by smallest vertex id."""
    tl = _components(tris, n)[tris[:, 0]]
    return [tris[tl == r] for r in np.unique(tl)]


def piece_rank(model: newton.Model) -> np.ndarray:
    """[particles] rank of each particle's tet piece along x within its world (0 = smallest mean x)."""
    x = model.particle_q.numpy()
    label = _components(model.tet_indices.numpy(), model.particle_count)
    world = model.particle_world.numpy()
    rank = np.zeros(model.particle_count, np.int64)
    for w in np.unique(world):
        mine = world == w
        roots = np.unique(label[mine])
        order = np.argsort([x[label == r, 0].mean() for r in roots])
        for k, r in enumerate(roots[order]):
            rank[label == r] = k
    return rank


def piece_overlap(q: np.ndarray, tris_a: np.ndarray, tris_b: np.ndarray) -> float:
    """Deepest vertex of piece b inside piece a's closed surface [m] (Warp winding number, on the CPU)."""
    return float(_query_piece(q, tris_a, tris_b)[0].max())


def _query_piece(q: np.ndarray, tris_a: np.ndarray, tris_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = wp.Mesh(points=wp.array(q, dtype=wp.vec3, device="cpu"),
                   indices=wp.array(tris_a.ravel().astype(np.int32), dtype=wp.int32, device="cpu"))
    vb = np.unique(tris_b)
    depth, dist = (wp.zeros(vb.size, dtype=float, device="cpu") for _ in range(2))
    wp.launch(_piece_query, dim=vb.size, inputs=[mesh.id, wp.array(q[vb], dtype=wp.vec3, device="cpu"), depth, dist],
              device="cpu")
    return depth.numpy(), dist.numpy()


def piece_distance(q: np.ndarray, tris_a: np.ndarray, tris_b: np.ndarray) -> float:
    """Closest vertex of piece b to piece a's surface [m]; 0 if any is inside."""
    return float(_query_piece(q, tris_a, tris_b)[1].min())


@dataclass
class _PressRig:
    sc: BlockScene
    vbd: newton.solvers.SolverVBD
    ctl: object
    pipe: newton.CollisionPipeline
    pieces: list[list[np.ndarray]]      # per world: surface triangles of slices 1..3 (by x)
    slice_of: np.ndarray                # [particles] slice index 0..2


def _press_rig(device: str, worlds: int, attach: bool) -> _PressRig:
    from alloy.topology.bonds import bonds_from_model
    from alloy.topology.controller import TopologyController
    from alloy.topology.presplit import SplitPlan
    from alloy.topology.types import TopologyBudget

    plan = SplitPlan(planes=tuple(((1.0, 0.0, 0.0), x) for x in PRESS_SEAMS))
    sc = presplit_block(device, cells=PRESS_CELLS, cell=PRESS_CELL, plan=plan, worlds=worlds, gravity=0.0)
    m = sc.model
    m.soft_contact_ke = 1.0e4
    vbd = newton.solvers.SolverVBD(m, iterations=10, particle_enable_self_contact=True,
                                   particle_self_contact_margin=PRESS_MARGIN, particle_self_contact_gap=PRESS_GAP,
                                   particle_rest_shape_contact_exclusion_radius=0.0 if attach else PRESS_RADIUS)
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=10_000), sc.blades)
    if attach:
        ctl.attach_self_contact(vbd)
    x0 = m.particle_q.numpy()
    per = m.particle_count // worlds
    tris = m.tri_indices.numpy()
    pcs = surface_pieces(tris, m.particle_count)
    pieces = [sorted((p for p in pcs if p[0, 0] // per == w), key=lambda p: x0[np.unique(p), 0].mean())
              for w in range(worlds)]
    slice_of = piece_rank(m)
    return _PressRig(sc, vbd, ctl, newton.CollisionPipeline(m), pieces, slice_of)


def _plate_poses(rig: _PressRig, x: np.ndarray, shift: np.ndarray, yaw: np.ndarray, bq: np.ndarray) -> np.ndarray:
    for w, body in enumerate(rig.sc.blade_bodies):
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw[w]))
        bq[body] = [x[w], 0.01 + shift[w, 0], 0.01 + shift[w, 1], q[0], q[1], q[2], q[3]]
    return bq


def _press_overlap(rig: _PressRig, q: np.ndarray, overlap: np.ndarray) -> None:
    for w, p in enumerate(rig.pieces):
        deepest = max(piece_overlap(q, p[i], p[j]) for i in range(3) for j in range(3) if i != j)
        overlap[w] = max(overlap[w], deepest)


def _closest_approach(rig: _PressRig, q: np.ndarray) -> np.ndarray:
    """Per world, smallest distance from a slice's surface vertices to the next slice's surface [m]
    (0 if inside; small and positive when the faces are pressed into contact)."""
    out = np.zeros(len(rig.pieces))
    for w, p in enumerate(rig.pieces):
        out[w] = min(piece_distance(q, p[i], p[j]) for i, j in ((0, 1), (1, 0), (1, 2), (2, 1)))
    return out


def run_seam_press(device: str, *, worlds: int = 16, attach: bool = True, seed: int = 0, open_steps: int = 40,
                   press_steps: int = 260, hold_steps: int = 60, squeeze: float = 3.0e-3) -> dict:
    """Cut both seams (``request_break``), pull slices 2 and 3 away (0.1 / 0.2 m/s for ``open_steps``), then
    press them back with the plate until it is ``squeeze`` past the block's rest length; hold. Returns per world
    the deepest overlap of any slice into another during press and hold [m], the closest approach of neighbouring
    slices at the end [m] (small: the faces were pressed into contact), plus twin bookkeeping."""
    rig = _press_rig(device, worlds, attach)
    m, ctl = rig.sc.model, rig.ctl
    s0, s1, c = m.state(), m.state(), m.control()
    contacts = rig.pipe.contacts()
    ctl.request_break(np.arange(rig.sc.bonds_per_world))
    rng = np.random.default_rng(seed)   # per world plate (y, z) shift [m] and yaw [rad], well under a 5 mm triangle
    shift, yaw = rng.uniform(-1.0e-3, 1.0e-3, (worlds, 2)), rng.uniform(-np.radians(3.0), np.radians(3.0), worlds)
    length = PRESS_CELLS[0] * PRESS_CELL
    s0.particle_qd.assign(np.stack([0.1 * rig.slice_of, 0 * rig.slice_of, 0 * rig.slice_of], 1).astype(np.float32))
    bq, overlap = s0.body_q.numpy(), np.zeros(worlds)
    x, start = np.full(worlds, length + 0.02), None
    for k in range(open_steps + press_steps + hold_steps):
        if k == open_steps:
            s0.particle_qd.zero_()
            start = s0.particle_q.numpy()[:, 0].reshape(worlds, -1).max(axis=1) + 1.0e-3
        if start is not None:
            x = start + (length + 0.5e-3 - squeeze - start) * min(k - open_steps, press_steps) / press_steps
        s0.body_q.assign(_plate_poses(rig, x, shift, yaw, bq))
        ctl.pre_step(s0, None, PRESS_DT)
        rig.pipe.collide(s0, contacts)
        rig.vbd.step(s0, s1, c, contacts, PRESS_DT)
        s1.body_q.assign(bq)
        ctl.apply(s1)
        s0, s1 = s1, s0
        if k >= open_steps and (k - open_steps) % 20 == 19:
            _press_overlap(rig, s0.particle_q.numpy(), overlap)
    released = ctl.state_arrays().get("topology.contact_released")
    return {"overlap": overlap, "gap": _closest_approach(rig, s0.particle_q.numpy()),
            "broken": int((ctl.bonds.intact.numpy() == 0).sum()), "bonds": ctl.bonds.count,
            "released": int(released.numpy().sum()) if released is not None else 0,
            "finite": bool(np.isfinite(s0.particle_q.numpy()).all())}


# ---------------------------------------------------------------------------------------------------------------
# SCR-SC5-3 cut stroke through a TET_CUT seam with real soft contacts (WP5's cut-front PEN tests, test_cut_front).
# STROKE_KE: cutting twins reach 0.69 mm into the blade (over the 0.5 mm thin-part limit) and clear before their bond
# breaks; all other contacts stay within 0.40 mm (Newton's default 1e3 leaves just-cut twins 0.55 mm deep).
STROKE_TOP, STROKE_EDGE, STROKE_SPEED, STROKE_KE = 0.015, 0.02, 0.05, 3.0e3   # [m], [m], [m/s], [N/m]


@dataclass
class CutStrokeRig:
    sc: BlockScene
    ctl: object                          # TopologyController
    vbd: newton.solvers.SolverVBD
    pipe: newton.CollisionPipeline
    contacts: object                     # newton.Contacts, refilled by each step's collide
    states: list
    blade_contact: object                # BladeContact (SCR-SC5-1, as the scene wiring adds it) or None


def cut_stroke_rig(device: str, *, worlds: int = 1, side_aware: bool = True) -> CutStrokeRig:
    """Per world: a 20 x 15 x 15 mm presplit block (TET_CUT seams x = 10 mm, y = 10 mm; 1 mm particles) clamped at
    x = 0, no gravity, a 1 mm thick blade box on its own body; BladeContact puts cutting twins on their own face."""
    from alloy.topology.blade_contact import BladeContact
    from alloy.topology.bonds import bonds_from_model
    from alloy.topology.controller import TopologyController
    from alloy.topology.types import TopologyBudget
    sc = presplit_block(device, worlds=worlds, cells=(4, 3, 3), gravity=0.0, k_mu=5e4, k_lambda=5e4)
    m = sc.model
    m.soft_contact_ke = STROKE_KE
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(max_events_per_step=64), sc.blades)
    pipe, vbd = newton.CollisionPipeline(m), newton.solvers.SolverVBD(m, iterations=4)
    bc = BladeContact(m, ctl.bonds, sc.blades) if side_aware else None
    return CutStrokeRig(sc, ctl, vbd, pipe, pipe.contacts(), [m.state(), m.state()], bc)


def run_cut_stroke(rig: CutStrokeRig, edge_z: np.ndarray, *, x: float = 0.01, dt: float = 1.0e-3,
                   observer=None) -> None:
    """Blade edge along y at (x, ``edge_z[k]``) (x = 10 mm: the seam), body_qd its travel since the last step. Per
    step: pose, ``pre_step``, collide, blade contact, VBD, ``observer(k, state_in, state_out, contacts)`` (health),
    apply, swap. To leave particles inside the blade, pin them (``clamp_particles``) and give one z (a teleport)."""
    (s0, s1), ctl, c = rig.states, rig.ctl, rig.sc.model.control()
    edge, down = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])
    for k, z in enumerate(edge_z):
        bq, bqd = s0.body_q.numpy(), s0.body_qd.numpy()
        bq[rig.sc.blade_bodies] = blade_pose(np.array([x, 0.0075, z + STROKE_EDGE]), edge, down)
        bqd[rig.sc.blade_bodies] = (0.0, 0.0, (z - edge_z[k - 1]) / dt if k else 0.0, 0.0, 0.0, 0.0)
        s0.body_q.assign(bq)
        s0.body_qd.assign(bqd)
        ctl.pre_step(s0, rig.contacts, dt)
        rig.pipe.collide(s0, rig.contacts)
        if rig.blade_contact is not None:
            rig.blade_contact.adjust_contacts(s0, rig.contacts)
        rig.vbd.step(s0, s1, c, rig.contacts, dt)
        s1.body_q.assign(bq)
        if observer is not None:
            observer(k, s0, s1, rig.contacts)
        ctl.apply(s1)
        s0, s1 = s1, s0
    rig.states = [s0, s1]
