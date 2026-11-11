# SPDX-License-Identifier: Apache-2.0
"""SB-7 bully target ``cloth_box_press`` (WP11 for WP7's cloth class, 2026-11-05).

The sheet of ``alloy.testing.scenes.box_on_cloth`` (8 x 8 cells, 0.4 m, pinned on two edges at z 0.5, VBD) under a
2 kg box that rides a MuJoCo carriage (slides x, y, z), driven by joint position targets as ``sph_pour``'s cup is.
Default coupled route, as the scene's: SolverCoupledProxy, MuJoCo -> VBD, lagged proxy (Newton's pipeline off in the
loop: ``collide=False``; the proxy has its own; MuJoCo runs its own collision, which finds nothing: the box is the
only rigid shape). The carriage force is capped (MuJoCo actuator force range) at what
the pinned sheet carries at the bottom of the legal range, so a legal press cannot drive the box through by
construction.

Actions (``joint_target_q``, 3 dims): the box centre's x, y within +-``reach`` of the sheet centre (over the pinned
span) and its z from ``lift`` above the sheet to ``press_depth`` below its rest plane (box bottom face), rate-limited
to ``speed``.

S1 rows (``PressSim.bully_monitors``, rows registered by WP7), per world, the episode's peak count, checked once per
control period on the device:
- ``cloth_pass_through``: cloth particles more than their radius inside the box, or inside its footprint above its
  top face (the sheet crossed the box);
- ``cloth_crossings``: edge-triangle crossings between non-adjacent elements of the sheet (``alloy.clips.fold_checks``'
  segment test over every pair of the grid).
Registered with ``alloy.bully.targets`` on import of ``alloy.solvers.vbd``.
"""

from __future__ import annotations

from dataclasses import dataclass

from alloy._vendor import newton
import numpy as np
import warp as wp

TARGET_NAME = "cloth_box_press"


@dataclass(frozen=True)
class PressSpec:
    cells: int = 8
    size: float = 0.4                # [m] sheet side
    sheet_z: float = 0.5             # [m] sheet rest plane
    half: float = 0.05               # [m] box half extent
    box_mass: float = 2.0            # [kg]
    particle_radius: float = 0.01    # [m]
    reach: float = 0.12              # +- [m] box x, y about the sheet centre
    lift: float = 0.02               # [m] box bottom above the sheet at the start and at the top of the range
    press_depth: float = 0.10        # [m] box bottom below the rest plane at the bottom of the range
    speed: float = 0.5               # [m/s] commanded rate limit
    slide_ke: float = 5.0e3          # [N/m] carriage position gains
    slide_kd: float = 200.0          # [N s/m]
    force_xy: float = 100.0          # [N] carriage force caps; force_z from the calibration (wp11-press/calib.py):
    force_z: float = 150.0           # deepest particle centre in the box over six extreme legal paths, 2.5 s: 3.2 mm
                                     # at 150 N, 5.8 at 200 N, pass-through at 300 N (S1 line: the 10 mm radius)
    carriage_mass: float = 0.1       # [kg] per slide link
    vbd_iterations: int = 10         # box_on_cloth's coupled default
    control_dt: float = 0.02
    substeps: int = 32               # 0.625 ms (alloy.testing.scenes.CLOTH_DT)


def press_template(spec: PressSpec) -> newton.ModelBuilder:
    """One world: the pinned sheet, and the box on a three-slide carriage (x, y, z) starting ``lift`` above it."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    cell = spec.size / spec.cells
    b.add_cloth_grid(pos=wp.vec3(-0.5 * spec.size, -0.5 * spec.size, spec.sheet_z), rot=wp.quat_identity(),
                     vel=wp.vec3(0.0), dim_x=spec.cells, dim_y=spec.cells, cell_x=cell, cell_y=cell, mass=0.02,
                     fix_left=True, fix_right=True, tri_ke=1.0e4, tri_ka=1.0e4, edge_ke=1.0e-2,
                     particle_radius=spec.particle_radius, label="cloth")
    origin = wp.transform((0.0, 0.0, spec.sheet_z + spec.half + spec.lift), wp.quat_identity())
    inertia = wp.mat33(np.eye(3) * 1.0e-4 * spec.carriage_mass)
    ranges = ((-spec.reach, spec.reach), (-spec.reach, spec.reach), (-(spec.lift + spec.press_depth), 0.0))
    caps = (spec.force_xy, spec.force_xy, spec.force_z)
    joints, parent = [], -1
    for k, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        if k < 2:
            link = b.add_link(xform=origin, mass=spec.carriage_mass, inertia=inertia, label=f"carriage_{'xy'[k]}")
        else:
            link = b.add_link(xform=origin, label="box")
        frame = origin if parent < 0 else wp.transform_identity()
        joints.append(b.add_joint_prismatic(parent, link, parent_xform=frame, axis=axis, target_ke=spec.slide_ke,
                                            target_kd=spec.slide_kd, limit_lower=ranges[k][0],
                                            limit_upper=ranges[k][1], effort_limit=caps[k]))
        parent = link
    b.add_articulation(joints)
    density = spec.box_mass / (8.0 * spec.half**3)
    b.add_shape_box(parent, hx=spec.half, hy=spec.half, hz=spec.half,
                    cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.5))
    return b


PARALLEL_RTOL = 1.0e-6               # |det| below this x |d||e1||e2|: the edge lies in the triangle's plane


@wp.func
def _f64(a: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]))


@wp.func
def _segment_crosses_triangle(p: wp.vec3, q: wp.vec3, v0: wp.vec3, v1: wp.vec3, v2: wp.vec3) -> bool:
    """Moller-Trumbore (fold_checks._seg_tri): the open segment p->q crosses the triangle. In float64 on the float32
    positions, and an edge (near-)parallel to the triangle's plane never counts: next to a pinned edge, an edge
    (free, pinned) and a triangle (pinned, free, pinned) whose free particles sag alike are coplanar to ~1e-8, the edge
    touching the plane only at its pinned end (t = 1, u = 0); float32 with CUDA's fused multiply-adds turned that 0/0
    into a crossing (WP7's 11 SB-7 trips in 300k, all this pair; CPU and float64 count 0)."""
    a = _f64(p)
    o = _f64(v0)
    d = _f64(q) - a
    e1 = _f64(v1) - o
    e2 = _f64(v2) - o
    h = wp.cross(d, e2)
    det = wp.dot(e1, h)
    if wp.abs(det) <= wp.float64(PARALLEL_RTOL) * wp.length(d) * wp.length(e1) * wp.length(e2):
        return False
    inv = wp.float64(1.0) / det
    s = a - o
    u = inv * wp.dot(s, h)
    qv = wp.cross(s, e1)
    v = inv * wp.dot(qv, d)
    t = inv * wp.dot(e2, qv)
    eps = wp.float64(1.0e-9)
    one = wp.float64(1.0)
    return u > eps and v > eps and u + v < one - eps and t > eps and t < one - eps


@wp.kernel
def count_pass_through(q: wp.array[wp.vec3], particle_world: wp.array[wp.int32], radius: wp.array[float],
                       body_q: wp.array[wp.transform], box_of_world: wp.array[wp.int32], half: float,
                       count: wp.array[wp.int32]):
    """Per world: particles more than their radius inside the box, or inside its footprint above its top face."""
    i = wp.tid()
    w = particle_world[i]
    if w < 0:
        return
    p = wp.transform_point(wp.transform_inverse(body_q[box_of_world[w]]), q[i])
    inside = wp.max(wp.max(wp.abs(p[0]), wp.abs(p[1])), wp.abs(p[2])) - half  # < 0 inside the box
    footprint = wp.abs(p[0]) < half and wp.abs(p[1]) < half
    if inside < -radius[i] or (footprint and p[2] > half):
        wp.atomic_add(count, w, 1)


@wp.kernel
def count_crossings(q: wp.array[wp.vec3], world_start: wp.array[wp.int32], edges: wp.array2d[wp.int32],
                    tris: wp.array2d[wp.int32], pair_edge: wp.array[wp.int32], pair_tri: wp.array[wp.int32],
                    count: wp.array[wp.int32]):
    """Per world: crossings of the sheet's (edge, triangle) pairs that share no vertex (local indices)."""
    w, k = wp.tid()
    s = world_start[w]
    e, t = pair_edge[k], pair_tri[k]
    if _segment_crosses_triangle(q[s + edges[e, 0]], q[s + edges[e, 1]], q[s + tris[t, 0]], q[s + tris[t, 1]],
                                 q[s + tris[t, 2]]):
        wp.atomic_add(count, w, 1)


def sheet_pairs(model: newton.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(world_start [W], edges [E, 2], tris [T, 3], pair_edge [P], pair_tri [P]) of one world's sheet, local indices.

    Worlds are equal, contiguous particle blocks (``ModelBuilder.replicate``); the pairs are every edge-triangle pair
    sharing no vertex."""
    world = model.particle_world.numpy()
    w_count = int(model.world_count)
    start = np.searchsorted(world, np.arange(w_count)).astype(np.int32)
    n = int(np.sum(world == 0))
    if not np.array_equal(world, np.repeat(np.arange(w_count), n)):
        raise ValueError("cloth_box_press: particles are not one equal contiguous block per world")
    tri = model.tri_indices.numpy()
    tris = tri[world[tri[:, 0]] == 0].astype(np.int32)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    shared = (edges[:, None, :, None] == tris[None, :, None, :]).any(axis=(2, 3))
    pair_edge, pair_tri = np.nonzero(~shared)
    return start, edges.astype(np.int32), tris, pair_edge.astype(np.int32), pair_tri.astype(np.int32)


def make_press_sim(spec: PressSpec, worlds: int, device: str):
    """PressSim over SolverCoupledProxy(MuJoCo carriage -> VBD sheet, lagged) for ``worlds`` copies."""
    from alloy._vendor.newton.solvers import SolverMuJoCo, SolverVBD

    from alloy.core.newton_compat import SolverCoupledProxy

    scene = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(press_template(spec), copy_count=worlds)
    scene.color()
    m = scene.finalize(device=device)
    m.soft_contact_ke, m.soft_contact_mu = 1.0e4, 0.5                  # box_on_cloth's contact
    boxes = [i for i, lbl in enumerate(m.body_label) if str(lbl).endswith("box")]
    world_of_box = m.body_world.numpy()[boxes]
    order = np.argsort(world_of_box)
    if not np.array_equal(world_of_box[order], np.arange(worlds)):
        raise ValueError(f"cloth_box_press: expected one box per world, got worlds {world_of_box.tolist()}")
    solver = SolverCoupledProxy(
        model=m,
        entries=[
            SolverCoupledProxy.Entry(name="mjc", bodies=list(range(m.body_count)), joints=list(range(m.joint_count)),
                                     solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=True)),
            SolverCoupledProxy.Entry(name="vbd", particles=list(range(m.particle_count)),
                                     solver=lambda v: SolverVBD(model=v, iterations=spec.vbd_iterations)),
        ],
        coupling=SolverCoupledProxy.Config(proxies=[SolverCoupledProxy.Proxy(
            source="mjc", destination="vbd", bodies=boxes, mode="lagged",
            collision_pipeline=lambda v: newton.CollisionPipeline(v), collide_interval=1)], iterations=2),
    )
    from .press_sim import PressSim

    return PressSim(m, solver, np.asarray(boxes)[order], spec.half, control_dt=spec.control_dt, substeps=spec.substeps,
                    collide=False)


def cloth_box_press(spec: PressSpec | None = None):
    """The registered SB-7 cloth target factory."""
    from alloy.bully.types import ActionSpace, BullyTarget

    spec = spec or PressSpec()
    rate = 1.0 / spec.control_dt
    low = np.array([-spec.reach, -spec.reach, -(spec.lift + spec.press_depth)])
    high = np.array([spec.reach, spec.reach, 0.0])
    actions = ActionSpace(low=low, high=high, rate_hz=rate, max_delta_per_tick=np.full(3, spec.speed / rate),
                          initial=np.zeros(3))
    return BullyTarget(name=TARGET_NAME, scene=None,
                       make_sim=lambda worlds, device: make_press_sim(spec, worlds, device), actions=actions,
                       action_kind="joint_target_q", action_index=(0, 1, 2))


def register() -> None:
    try:
        from alloy.bully.targets import register_target
    except ImportError:  # bully package absent in a trimmed install
        return
    register_target(TARGET_NAME, cloth_box_press)


__all__ = ["TARGET_NAME", "PressSpec", "cloth_box_press", "count_crossings", "count_pass_through", "make_press_sim",
           "press_template", "register", "sheet_pairs"]
