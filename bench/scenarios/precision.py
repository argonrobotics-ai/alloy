# SPDX-License-Identifier: Apache-2.0
"""Precision contact family (bench/SCENARIOS.md section 1.2): SDF and hydroelastic pairs, sub-mm clearance.

    python -m bench.run --family precision --list
    python -m bench.run --family precision --device cpu --ladder 1 --horizon 0.05      # Mac CPU smoke
    python -m bench.run --family precision --scenario P01_peg_hole_0p2 --config sdf_mujoco --device cuda:0

Scenarios (laws in each class docstring, notes in bench/notes/precision.md):
  P01_peg_hole_0p2     9.80 mm peg into a 10.00 mm hole through a compliant holder (A8, CON-46, bench.md #9)
  P02_nut_bolt         nut run down a bolt by a velocity servo under 5 N axial load (A9 thread law)
  P03_rj45_plug        scripted RJ45 insertion and 5 N pull-out (latch as an equivalent cantilever)
  P04_hertz_sphere     R = 10 mm sphere: load ramp to 50 N and impacts (Hertz / configured pressure law, A7)
  P05_thin_wall_tunnel capsule fired into 2 and 5 mm free walls at 5-50 m/s (tunnelling, restitution, momentum)

Every driven body is moved by a Warp kernel that writes body_f from a device step counter (no host sync inside
step(), so CUDA graphs work). Per-step events (peak force, contact start/end, tunnelling) are kept in device
arrays; trajectories of the first worlds go to a device trace that finish() analyses.

Device notes (Newton ac7f7830):
- Mesh and primitive texture SDFs need CUDA (newton@ac7f7830:newton/_src/geometry/sdf_utils.py:451-458,
  "SDF.create_from_mesh requires a CUDA device"). On CPU the sdf_* configs run mesh-mesh contact through Newton's
  BVH fallback
  (newton/_src/geometry/sdf_contact.py:1431-1438) and record metrics.sdf_backend = "bvh_fallback_cpu".
  hydro_* configs raise on CPU (hydroelastic contact needs the texture SDF).
- SolverSteel solve_mode="matrix_free" needs CUDA; on CPU the steel configs use "split" (recorded in metrics).
"""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench.harness import Built, Config, Scenario

log = logging.getLogger("alloy.bench.precision")

SDF_CACHE_DIR = Path.home() / ".cache" / "alloy" / "sdf"
TRACE_W = 18  # trace row: t, p(3), q(4), v(3), w(3), aux(4)
FACTORY_GIT = "https://github.com/isaac-sim/IsaacGymEnvs.git"
FACTORY_DIR = "assets/factory/mesh/factory_nut_bolt"


# =============================================================================================================
# Shared device kernels
# =============================================================================================================
@wp.func
def _ease(x: float) -> float:
    """Smoothstep on [0, 1] (loads are eased in, SCENARIOS.md conventions)."""
    y = wp.clamp(x, 0.0, 1.0)
    return y * y * (3.0 - 2.0 * y)


@wp.kernel
def _tick(step: wp.array(dtype=int)):
    step[0] = step[0] + 1


@wp.kernel
def _reducer_watch(dropped: wp.array(dtype=wp.int32), insert_fail: wp.array(dtype=wp.int32),
                   out: wp.array(dtype=wp.int32)):
    """Latch the mesh contact reducer's per-pass loss counters (Newton resets them every collide and only
    wp.printf's a warning): out = (passes with a loss, max candidates dropped, max hashtable insert failures)."""
    n = dropped[0]
    f = insert_fail[0]
    if n > 0 or f > 0:
        out[0] = out[0] + 1
    out[1] = wp.max(out[1], n)
    out[2] = wp.max(out[2], f)


@wp.kernel
def _trace_kernel(
    step: wp.array(dtype=int),
    h: float,
    every: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    bodies: wp.array(dtype=int),
    worlds: wp.array(dtype=int),
    aux: wp.array(dtype=wp.vec4),
    trace: wp.array3d(dtype=float),
):
    """Trace r, row k: body bodies[r] (world worlds[r]) after substep (k + 1) * every (step not yet ticked)."""
    w = wp.tid()
    s = step[0] + 1
    if s % every != 0:
        return
    k = s / every - 1
    if k >= trace.shape[1]:
        return
    b = bodies[w]
    X = body_q[b]
    p = wp.transform_get_translation(X)
    q = wp.transform_get_rotation(X)
    v = wp.spatial_top(body_qd[b])
    om = wp.spatial_bottom(body_qd[b])
    a = aux[worlds[w]]
    trace[w, k, 0] = float(s) * h
    for i in range(3):
        trace[w, k, 1 + i] = p[i]
        trace[w, k, 8 + i] = v[i]
        trace[w, k, 11 + i] = om[i]
    for i in range(4):
        trace[w, k, 4 + i] = q[i]
        trace[w, k, 14 + i] = a[i]


@wp.func
def _rotvec(q_target: wp.quat, q: wp.quat) -> wp.vec3:
    """Rotation vector (world frame) that turns q into q_target, shortest way."""
    dq = q_target * wp.quat_inverse(q)
    if dq[3] < 0.0:
        dq = wp.quat(-dq[0], -dq[1], -dq[2], -dq[3])
    v = wp.vec3(dq[0], dq[1], dq[2])
    s = wp.length(v)
    if s < 1.0e-12:
        return 2.0 * v
    ang = 2.0 * wp.atan2(s, dq[3])
    return v * (ang / s)


# =============================================================================================================
# Shared host helpers
# =============================================================================================================
def _is_cuda(device: str) -> bool:
    return wp.get_device(device).is_cuda


def _gravity(model: newton.Model) -> np.ndarray:
    return np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)[0]


def _world_blocks(nworld: int, nvariant: int) -> list[tuple[int, int, int]]:
    """Contiguous blocks (variant, first_world, count). Geometry variants tile in blocks (one replicate per
    variant); the remaining conditions tile as (w - first_world) % n inside each block."""
    out = []
    for g in range(nvariant):
        a, b = -(-g * nworld // nvariant), -(-(g + 1) * nworld // nvariant)  # ceil: world 0 is variant 0
        if b > a:
            out.append((g, a, b - a))
    return out


def _replicate_blocks(main: newton.ModelBuilder, variants: list[newton.ModelBuilder], nworld: int,
                      local_bodies: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate each geometry variant over its block of worlds.

    Returns (variant_of_world, sub_index_of_world, body_ids[nworld, nlocal]) where body_ids maps each variant's
    listed local body indices to global body indices.
    """
    variant = np.zeros(nworld, np.int32)
    sub = np.zeros(nworld, np.int32)
    nloc = len(local_bodies[0])
    ids = np.zeros((nworld, nloc), np.int32)
    for g, first, count in _world_blocks(nworld, len(variants)):
        b0 = main.body_count
        nb = variants[g].body_count
        main.replicate(variants[g], copy_count=count)
        for k in range(count):
            w = first + k
            variant[w], sub[w] = g, k
            ids[w] = [b0 + k * nb + j for j in local_bodies[g]]
    return variant, sub, ids


def _set_free_bodies(model: newton.Model, bodies: np.ndarray, q7: np.ndarray | None, qd6: np.ndarray | None) -> None:
    """Write free-joint coordinates (p, quat xyzw) and velocities (v_com, omega, world) for the listed bodies."""
    child = model.joint_child.numpy()
    jtype = model.joint_type.numpy()
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    joint_of = {int(c): j for j, c in enumerate(child) if jtype[j] == int(newton.JointType.FREE)}
    jq, jqd = model.joint_q.numpy(), model.joint_qd.numpy()
    for i, b in enumerate(bodies):
        j = joint_of[int(b)]
        if q7 is not None:
            jq[q_start[j]:q_start[j] + 7] = q7[i]
        if qd6 is not None:
            jqd[qd_start[j]:qd_start[j] + 6] = qd6[i]
    model.joint_q.assign(jq)
    model.joint_qd.assign(jqd)


def _states(model: newton.Model) -> tuple[Any, Any, Any]:
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)
    return s0, s1, control


def _quat_tilt_y(theta: float) -> np.ndarray:
    """xyzw quaternion of a rotation by theta about +y."""
    return np.array([0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)])


def _quat_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) of xyzw quaternions (..., 4)."""
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def _trace_rows(b: Built) -> np.ndarray:
    """Trace as [ntrace, nrec, TRACE_W] with unwritten rows (t < 0) dropped per world (returns list-like array)."""
    return b.extra["trace"].numpy()


def _new_trace(ntrace: int, nrec: int, device: str) -> wp.array:
    return wp.full((ntrace, nrec, TRACE_W), -1.0, dtype=float, device=device)


# ---- solvers and contact pipeline ---------------------------------------------------------------------------
def make_solver(cfg: Config, model: newton.Model, contacts_per_world: int, device: str) -> tuple[Any, dict]:
    """Solver factory for this family. Returns (solver, notes) where notes go to rec['metrics']."""
    S, p = newton.solvers, dict(cfg.params)
    notes: dict[str, Any] = {}
    if cfg.solver == "mujoco":
        return S.SolverMuJoCo(
            model, use_mujoco_contacts=False, disable_sensors=True, solver="newton",
            integrator=p.get("integrator", "implicitfast"), cone=p.get("cone", "elliptic"),
            njmax=p.get("njmax_mult", 3) * contacts_per_world, nconmax=contacts_per_world,
            iterations=p.get("iterations", 15), ls_iterations=p.get("ls_iterations", 100),
            impratio=p.get("impratio", 1.0),
        ), notes
    if cfg.solver == "kamino":
        kc = S.SolverKamino.Config.from_model(model, dynamics_solver=p.get("dynamics_solver", "dvi"),
                                              sparse_dynamics=True, sparse_jacobian=True)
        kc.dvi.max_alternating_iterations = p.get("max_alternating_iterations", 4)
        return S.SolverKamino(model, config=kc), notes
    if cfg.solver == "xpbd":
        return S.SolverXPBD(model, iterations=p.get("iterations", 10),
                            rigid_contact_relaxation=p.get("rigid_contact_relaxation", 0.8),
                            enable_restitution=p.get("enable_restitution", False)), notes
    if cfg.solver == "steel":
        mode = p.get("solve_mode", "matrix_free")
        if mode == "matrix_free" and not _is_cuda(device):
            mode = "split"
            notes["steel_solve_mode"] = "split (matrix_free needs CUDA)"
        else:
            notes["steel_solve_mode"] = mode
        restitution = p.get("enable_restitution", True)
        if restitution and mode != "matrix_free":
            restitution = False  # SolverSteel: "Contact restitution requires solve_mode='matrix_free'"
            notes["steel_restitution"] = "off (split mode cannot do restitution)"
        return S.SolverSteel(model, solve_mode=mode, iterations=p.get("iterations", 12),
                             enable_restitution=restitution), notes
    if cfg.solver == "vbd":
        return S.SolverVBD(model, iterations=p.get("iterations", 12),
                           rigid_body_contact_buffer_size=p.get("contact_buffer", 256)), notes
    if cfg.solver == "semi_implicit":
        return S.SolverSemiImplicit(model), notes
    raise ValueError(f"unknown solver {cfg.solver!r}")


def make_pipeline(model: newton.Model, contacts_per_world: int, nworld: int, hydro: Any = None,
                  **kw: Any) -> tuple[newton.CollisionPipeline, Any]:
    """Collision pipeline with explicit capacity (SCENARIOS.md: contact capacity is set explicitly)."""
    cmax = contacts_per_world * nworld
    model.rigid_contact_max = cmax  # Kamino sizes from the model (example_nut_bolt_sdf.py:150)
    broad_phase = kw.pop("broad_phase", "sap")
    pipe = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=cmax, broad_phase=broad_phase,
                                    sdf_hydroelastic_config=hydro, **kw)
    return pipe, pipe.contacts()


def _tri_pairs_kw(cfg: Config, nworld: int) -> dict:
    """Explicit mesh triangle-pair / contact-reducer capacity (SCENARIOS.md: capacity is set explicitly).
    Newton's automatic estimate (128 per mesh pair, 1e6 minimum) overflowed on L4 at P01 n=4096 and P02
    n>=1024 (2026-11-01), dropping contacts silently."""
    tpw = cfg.params.get("tri_pairs_per_world")
    return {} if tpw is None else {"max_triangle_pairs": max(1_000_000, int(tpw) * nworld)}


def _register(cfg: Config, builder: newton.ModelBuilder) -> None:
    if cfg.solver == "kamino":
        newton.solvers.SolverKamino.register_custom_attributes(builder)


def _backend(cfg: Config) -> str:
    """Contact backend from the config name prefix: sdf, hydro, point/primitive, or plain."""
    return cfg.params.get("backend", cfg.name.split("_")[0])


def _require_cuda_for_hydro(cfg: Config, device: str) -> None:
    if _backend(cfg) == "hydro" and not _is_cuda(device):
        raise RuntimeError("hydroelastic contact needs the CUDA texture SDF (newton@ac7f7830 "
                           "geometry/sdf_utils.py:451-458: SDF.create_from_mesh requires a CUDA device); "
                           "run on cuda:0")


def _attach_sdf(mesh: newton.Mesh, device: str, max_resolution: int, band: tuple[float, float],
                margin: float) -> str:
    """Bake the texture SDF on CUDA; on CPU leave the mesh bare so Newton uses its BVH fallback."""
    if not _is_cuda(device):
        log.debug("no CUDA on %s: mesh with %d vertices uses the BVH SDF fallback", device, len(mesh.vertices))
        return "bvh_fallback_cpu"
    SDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mesh.build_sdf(max_resolution=max_resolution, narrow_band_range=band, margin=margin, device=device,
                   cache_dir=SDF_CACHE_DIR)
    return f"texture_sdf_r{max_resolution}"


# ---- procedural meshes (watertight, outward normals; volumes are checked in tests) ---------------------------
def _ring(r: np.ndarray | float, z: np.ndarray | float, n: int, square: float | None = None) -> np.ndarray:
    th = 2.0 * np.pi * np.arange(n) / n
    c, s = np.cos(th), np.sin(th)
    if square is not None:
        r = square / np.maximum(np.abs(c), np.abs(s))
    r = np.broadcast_to(r, th.shape)
    return np.stack([r * c, r * s, np.broadcast_to(z, th.shape)], -1)


def _band(a0: int, b0: int, n: int) -> list[tuple[int, int, int]]:
    """Triangles joining ring a (vertices a0..a0+n-1) to ring b; normal = tangent(ccw) x (b - a)."""
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(a0 + i, a0 + j, b0 + i), (a0 + j, b0 + j, b0 + i)]
    return tris


def _fan(ring0: int, centre: int, n: int, up: bool) -> list[tuple[int, int, int]]:
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((ring0 + i, ring0 + j, centre) if up else (ring0 + i, centre, ring0 + j))
    return tris


class _MeshAcc:
    """Vertex/triangle accumulator for the procedural meshes."""

    def __init__(self) -> None:
        self.v: list[np.ndarray] = []
        self.n = 0
        self.t: list[tuple[int, int, int]] = []

    def add(self, pts: np.ndarray) -> int:
        pts = np.atleast_2d(pts)
        first = self.n
        self.v.append(pts)
        self.n += len(pts)
        return first

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.v).astype(np.float32), np.asarray(self.t, np.int32).reshape(-1)


def mesh_signed_volume(v: np.ndarray, idx: np.ndarray) -> float:
    t = v[idx.reshape(-1, 3)].astype(np.float64)
    return float(np.sum(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2]))) / 6.0)


def mesh_cylinder(radius: float, half_height: float, n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder along z, centred at the origin."""
    m = _MeshAcc()
    bot = m.add(_ring(radius, -half_height, n))
    top = m.add(_ring(radius, half_height, n))
    cb = m.add(np.array([0.0, 0.0, -half_height]))
    ct = m.add(np.array([0.0, 0.0, half_height]))
    m.t += _band(bot, top, n) + _fan(top, ct, n, up=True) + _fan(bot, cb, n, up=False)
    return m.arrays()


def mesh_block_with_hole(half: float, height: float, hole_r: float, depth: float, chamfer: float,
                         n: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Square block (top face at z = 0, bottom at -height) with a blind round hole of radius hole_r and the given
    depth, optionally chamfered chamfer x 45 deg at the entry. n must be a multiple of 8 (square corners exact)."""
    assert n % 8 == 0
    m = _MeshAcc()
    ot = m.add(_ring(0.0, 0.0, n, square=half))
    it = m.add(_ring(hole_r + chamfer, 0.0, n))
    ic = m.add(_ring(hole_r, -chamfer, n)) if chamfer > 0 else it
    ib = m.add(_ring(hole_r, -depth, n))
    ob = m.add(_ring(0.0, -height, n, square=half))
    chb = m.add(np.array([0.0, 0.0, -depth]))
    cbb = m.add(np.array([0.0, 0.0, -height]))
    m.t += _band(ot, it, n)  # top annulus, +z
    if chamfer > 0:
        m.t += _band(it, ic, n)  # chamfer, up and in
    m.t += _band(ic, ib, n)  # hole wall, toward the axis
    m.t += _fan(ib, chb, n, up=True)  # hole bottom, +z
    m.t += _band(ob, ot, n)  # outer side, outward
    m.t += _fan(ob, cbb, n, up=False)  # block bottom, -z
    return m.arrays()


def iso_thread_radius(z_minus_helix: np.ndarray, pitch: float, d_major: float) -> np.ndarray:
    """ISO 68-1 basic external profile radius at axial phase u = z - p*theta/(2 pi) (right-hand helix).

    Crest flat p/8 at d/2, root flat p/4 at d1/2 = d/2 - 5H/8 (H = 0.866 p), 60 deg flanks."""
    H = math.sqrt(3.0) / 2.0 * pitch
    r_maj, r_min = d_major / 2.0, d_major / 2.0 - 5.0 * H / 8.0
    s = np.abs(np.mod(z_minus_helix / pitch + 0.5, 1.0) - 0.5) * pitch  # axial distance to nearest crest centre
    flank = (s - pitch / 16.0) / (5.0 * pitch / 16.0)
    return np.where(s <= pitch / 16.0, r_maj, np.where(s >= 3.0 * pitch / 8.0, r_min,
                                                       r_maj - flank * (r_maj - r_min)))


def mesh_thread_bolt(d: float, pitch: float, length: float, n: int = 128, rows_per_pitch: int = 24
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Threaded shank z in [0, length], world-fixed helix phase (z - p theta / 2 pi)."""
    m = _MeshAcc()
    nz = int(round(length / pitch * rows_per_pitch))
    th = 2.0 * np.pi * np.arange(n) / n
    rows = []
    for j in range(nz + 1):
        z = length * j / nz
        r = iso_thread_radius(z - pitch * th / (2 * np.pi), pitch, d)
        rows.append(m.add(np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], -1)))
    cb = m.add(np.array([0.0, 0.0, 0.0]))
    ct = m.add(np.array([0.0, 0.0, length]))
    for j in range(nz):
        m.t += _band(rows[j], rows[j + 1], n)
    m.t += _fan(rows[0], cb, n, up=False) + _fan(rows[-1], ct, n, up=True)
    return m.arrays()


def mesh_thread_nut(d: float, pitch: float, z0: float, height: float, outer_r: float, clearance: float,
                    n: int = 128, rows_per_pitch: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Internal ISO thread (same helix phase as mesh_thread_bolt, radially offset by clearance) between z0 and
    z0 + height, round outside of radius outer_r. Vertices are in world coordinates."""
    m = _MeshAcc()
    nz = max(2, int(round(height / pitch * rows_per_pitch)))
    th = 2.0 * np.pi * np.arange(n) / n
    inner = []
    for j in range(nz + 1):
        z = z0 + height * j / nz
        r = iso_thread_radius(z - pitch * th / (2 * np.pi), pitch, d) + clearance
        inner.append(m.add(np.stack([r * np.cos(th), r * np.sin(th), np.full(n, z)], -1)))
    ob = m.add(_ring(outer_r, z0, n))
    ot = m.add(_ring(outer_r, z0 + height, n))
    for j in range(nz):
        m.t += _band(inner[j + 1], inner[j], n)  # inner wall, toward the axis
    m.t += _band(ob, ot, n)  # outer wall
    m.t += _band(ot, inner[-1], n)  # top annulus, +z
    m.t += _band(inner[0], ob, n)  # bottom annulus, -z
    return m.arrays()


@lru_cache(maxsize=4)
def _factory_dir() -> Path:
    import alloy._vendor.newton.examples; from alloy._vendor import newton

    return Path(newton.examples.download_external_git_folder(FACTORY_GIT, FACTORY_DIR))


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    tm = trimesh.load(str(path), force="mesh")
    return np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32).reshape(-1)


# =============================================================================================================
# Base scenario: kernel-driven substeps, contact capacity bookkeeping
# =============================================================================================================
class _PrecisionScenario(Scenario):
    """Shared stepping: clear forces, drive kernel, collide (every collide_every substeps), solve, swap, post
    kernels (trace, events), tick the device step counter."""

    family = "precision"
    has_contacts = True
    ladder = (1, 16, 256, 1024)

    # ---- hooks for subclasses ----------------------------------------------------------------------------
    def drive(self, b: Built, h: float) -> None:
        """Write body_f for this substep (state_0 is the substep input)."""

    def post(self, b: Built, h: float) -> None:
        """Per-substep device bookkeeping on the substep output (now state_0)."""
        tr = b.extra.get("trace")
        if tr is not None:
            n = tr.shape[0]
            wp.launch(_trace_kernel, dim=n, inputs=[b.extra["step"], h, b.extra["trace_every"], b.state_0.body_q,
                                                    b.state_0.body_qd, b.extra["trace_body"], b.extra["trace_world"],
                                                    b.extra["aux"], tr])

    # ---- stepping ----------------------------------------------------------------------------------------
    def step(self, b: Built, config: Config) -> None:
        h = config.dt / config.substeps
        every = int(config.params.get("collide_every", 1))
        for i in range(config.substeps):
            b.state_0.clear_forces()
            self.drive(b, h)
            if b.pipeline is not None and i % every == 0:
                self.collide(b, h * every)
            b.solver.step(b.state_0, b.state_1, b.control, b.contacts, h)
            b.state_0, b.state_1 = b.state_1, b.state_0
            self.post(b, h)
            wp.launch(_tick, dim=1, inputs=[b.extra["step"]])

    def collide(self, b: Built, horizon: float) -> None:
        """Collision pass; a speculative pipeline needs the collision-update horizon [s]."""
        if b.extra.get("speculative"):
            b.pipeline.collide(b.state_0, b.contacts, dt=horizon)
        else:
            b.pipeline.collide(b.state_0, b.contacts)
        self._watch_reducer(b)

    def _watch_reducer(self, b: Built) -> None:
        """Count contact candidates the mesh contact reducer dropped (first call, in warm-up, allocates)."""
        if "red_watch" not in b.extra:
            red = getattr(getattr(b.pipeline, "narrow_phase", None), "global_contact_reducer", None)
            b.extra["red_watch"] = None if red is None else (red, wp.zeros(3, dtype=wp.int32, device=b.model.device))
        w = b.extra["red_watch"]
        if w is not None:
            wp.launch(_reducer_watch, dim=1, inputs=[w[0].buffer_overflows, w[0].ht_insert_failures, w[1]])

    # ---- measurement -------------------------------------------------------------------------------------
    def measure(self, b: Built, t: float, rec: dict) -> None:
        super().measure(b, t, rec)
        if b.contacts is not None and getattr(b.contacts, "rigid_contact_count", None) is not None:
            n = int(b.contacts.rigid_contact_count.numpy()[0])
            b.extra["max_contacts"] = max(b.extra.get("max_contacts", 0), n)

    def finish_common(self, b: Built, rec: dict) -> None:
        met = rec["metrics"]
        met.update(b.extra.get("notes", {}))
        if b.contacts is not None:
            cmax = int(b.contacts.rigid_contact_max)
            seen = int(b.extra.get("max_contacts", 0))
            met["rigid_contact_max"] = cmax
            met["rigid_contacts_peak"] = seen
            met["contact_overflow"] = max(0, seen - cmax)
        w = b.extra.get("red_watch")
        if w is not None:
            passes, dropped, fails = (int(x) for x in w[1].numpy())
            met["max_triangle_pairs"] = int(w[0].position_depth.shape[0] - 1)
            met["reduction_overflow_passes"] = passes
            met["reduction_dropped_max"] = dropped
            met["reduction_insert_failures_max"] = fails
            if passes:
                rec["notes"] += (f"mesh contact reducer dropped up to {dropped} candidates in {passes} collide "
                                 f"passes (max_triangle_pairs too small): contacts were lost; ")
        mj = getattr(b.solver, "mjw_data", None)
        if mj is not None and hasattr(mj, "nacon"):
            met["mjw_nacon_end"] = int(np.max(mj.nacon.numpy()))
            met["mjw_naconmax"] = int(getattr(mj, "naconmax", -1))

    def base_extra(self, device: str, nworld: int, ntrace: int, nrec: int, every: int,
                   trace_body: np.ndarray, trace_world: np.ndarray | None = None) -> dict:
        """Shared device state. Trace r follows body trace_body[r] of world trace_world[r] (default r)."""
        tw = np.arange(ntrace) if trace_world is None else np.asarray(trace_world)[:ntrace]
        return {
            "trace_world": wp.array(tw.astype(np.int32), dtype=int, device=device),
            "trace_world_np": tw,
            "step": wp.zeros(1, dtype=int, device=device),
            "trace": _new_trace(ntrace, nrec, device) if ntrace > 0 else None,
            "trace_every": every,
            "trace_body": wp.array(trace_body[:ntrace].astype(np.int32), dtype=int, device=device),
            "aux": wp.zeros(nworld, dtype=wp.vec4, device=device),
            "notes": {},
        }


def _nrec(horizon_s: float, h: float, every: int) -> int:
    return int(math.ceil(horizon_s / (h * every))) + 2


# =============================================================================================================
# P05 thin-wall tunnelling
# =============================================================================================================
@wp.kernel
def _p05_post(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    cap: wp.array(dtype=int),
    wall: wp.array(dtype=int),
    reach: float,
    half_t: wp.array(dtype=float),
    tunnel: wp.array(dtype=int),
    max_pen: wp.array(dtype=float),
):
    """Tunnelling = capsule centre ever beyond the wall mid-plane; penetration = front overlap of the wall's
    back face while the capsule is still on the near side."""
    w = wp.tid()
    xc = wp.transform_get_translation(body_q[cap[w]])[0]
    xw = wp.transform_get_translation(body_q[wall[w]])[0]
    if xc > xw:
        tunnel[w] = 1
    elif tunnel[w] == 0:
        pen = xc + reach - (xw - half_t[w])
        if pen > max_pen[w]:
            max_pen[w] = pen


class ThinWallTunnel(_PrecisionScenario):
    """P05: capsule (r 5 mm, 30 mm long, steel) fired along +x into a free steel wall (100 x 100 mm, 2 or 5 mm
    thick) at v in {5, 10, 20, 50} m/s; gravity off. Conditions: thickness tiles in world blocks, speed w % 4.

    Laws: zero tunnelling. Newton's restitution law for the separation speed v2' - v1' = e v (e = configured
    restitution), within 5 %. Momentum m1 v = m1 v1' + m2 v2' (no external x impulse), and the wall velocity
    v2' = (1 + e) m1 v / (m1 + m2) (wall_vel_err_pct). MuJoCo has no restitution coefficient, so its e is whatever
    its contact damping gives; the row records it against the configured e.
    """

    id = "P05_thin_wall_tunnel"
    title = "Capsule at 5-50 m/s into 2/5 mm free walls: tunnelling, restitution, momentum"
    horizon_s = 0.1
    accuracy_unit = "%"
    accuracy_ref = "separation speed = e v (Newton restitution), within 5 %; zero tunnelling; momentum conserved"
    SPEEDS = (5.0, 10.0, 20.0, 50.0)
    THICK = (0.002, 0.005)
    R, HH, GAP0, E = 0.005, 0.010, 0.02, 0.5
    configs = (
        Config("xpbd_adaptive", "xpbd", dt=1 / 60, substeps=20,
               # nxn broad phase: SAP allocates temporaries per call, which a CUDA conditional graph body (the
               # scheduler's while loop) rejects ("unsupported operation (memory allocation)", L4 2026-11-01)
               params={"iterations": 10, "enable_restitution": True, "adaptive": True, "spec_gap_max": 0.2,
                       "broad_phase": "nxn"}),
        Config("xpbd_fixed", "xpbd", dt=1 / 60, substeps=20,
               params={"iterations": 10, "enable_restitution": True, "collide_every": 20, "spec_gap_max": 0.2}),
        Config("mujoco", "mujoco", dt=1 / 60, substeps=20, params={}),
        Config("kamino", "kamino", dt=1 / 60, substeps=20, params={}),
        Config("steel", "steel", dt=1 / 60, substeps=20, params={"enable_restitution": True}),
    )

    def _variant(self, thick: float, cfg: Config) -> newton.ModelBuilder:
        sb = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        stiff = {"ke": 1.0e7, "kd": 1.0e4}  # nut/bolt example values; MuJoCo clamps the time constant to 2 h
        wall_cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, restitution=self.E, gap=0.01, **stiff)
        cap_cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.3, restitution=self.E, gap=0.0, **stiff)
        wall = sb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), label="wall")
        sb.add_shape_box(wall, hx=thick / 2, hy=0.05, hz=0.05, cfg=wall_cfg)
        x0 = -(thick / 2 + self.GAP0 + self.R + self.HH)
        cap = sb.add_body(xform=wp.transform(wp.vec3(x0, 0.0, 0.0), wp.quat_identity()), label="capsule")
        sb.add_shape_capsule(cap, xform=wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0),
                                                                                             0.5 * math.pi)),
                             radius=self.R, half_height=self.HH, cfg=cap_cfg)
        return sb

    def build(self, nworld: int, device: str, config: Config) -> Built:
        variants = [self._variant(t, config) for t in self.THICK]
        main = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _register(config, main)
        var, sub, ids = _replicate_blocks(main, variants, nworld, [[0, 1]] * len(variants))
        model = main.finalize(device=device)
        speed = np.array([self.SPEEDS[s % len(self.SPEEDS)] for s in sub])
        qd = np.zeros((nworld, 6))
        qd[:, 0] = speed
        _set_free_bodies(model, ids[:, 1], None, qd)
        cpw = 64
        kw = {}
        if "spec_gap_max" in config.params:
            kw["speculative_contact_gap_max"] = config.params["spec_gap_max"]
        if "broad_phase" in config.params:
            kw["broad_phase"] = config.params["broad_phase"]
        pipe, contacts = make_pipeline(model, cpw, nworld, **kw)
        solver, notes = make_solver(config, model, cpw, device)
        s0, s1, control = _states(model)
        h = config.dt / config.substeps
        ex = self.base_extra(device, nworld, 0, 0, 1, ids[:, 1])
        mass = model.body_mass.numpy()
        thick = np.array([self.THICK[g] for g in var])
        ex.update(notes=notes, speculative="spec_gap_max" in config.params,
                  cap=wp.array(ids[:, 1], dtype=int, device=device),
                  wall=wp.array(ids[:, 0], dtype=int, device=device), ids=ids, speed=speed, thick=thick,
                  half_t=wp.array(thick / 2, dtype=float, device=device),
                  tunnel=wp.zeros(nworld, dtype=int, device=device),
                  max_pen=wp.zeros(nworld, dtype=float, device=device),
                  m_cap=mass[ids[:, 1]], m_wall=mass[ids[:, 0]], var=var, sub=sub)
        b = Built(model, solver, s0, s1, control, contacts, pipe, nworld=nworld, extra=ex)
        if config.params.get("adaptive"):
            b.extra["scheduler"] = newton.CollisionSubstepScheduler(
                pipe, (s0, s1), collision_callback=lambda st, cdt: pipe.collide(st, contacts, dt=cdt),
                substep_callback=lambda si, so, dt: self._adaptive_substep(b, si, so, dt),
                frame_dt=config.dt, substeps=config.substeps)
            pipe.collide(s0, contacts, dt=config.dt)  # preallocate lazy storage (example_adaptive_collision.py)
        return b

    def _adaptive_substep(self, b: Built, s_in: Any, s_out: Any, dt: float) -> None:
        s_in.clear_forces()
        b.solver.step(s_in, s_out, b.control, b.contacts, dt)
        wp.launch(_p05_post, dim=b.nworld, inputs=[s_out.body_q, s_out.body_qd, b.extra["cap"], b.extra["wall"],
                                                   self.R + self.HH, b.extra["half_t"], b.extra["tunnel"],
                                                   b.extra["max_pen"]])

    def step(self, b: Built, config: Config) -> None:
        sch = b.extra.get("scheduler")
        if sch is not None:
            sch.step()  # frame begins and ends in states[0] (substeps even), so b.state_0 stays valid
            return
        super().step(b, config)

    def post(self, b: Built, h: float) -> None:
        wp.launch(_p05_post, dim=b.nworld, inputs=[b.state_0.body_q, b.state_0.body_qd, b.extra["cap"],
                                                   b.extra["wall"], self.R + self.HH, b.extra["half_t"],
                                                   b.extra["tunnel"], b.extra["max_pen"]])

    def finish(self, b: Built, rec: dict) -> None:
        self.finish_common(b, rec)
        ex = b.extra
        qd = b.state_0.body_qd.numpy().astype(np.float64)
        v1 = qd[ex["ids"][:, 1], 0]
        v2 = qd[ex["ids"][:, 0], 0]
        m1, m2, v = ex["m_cap"], ex["m_wall"], ex["speed"]
        tunnel = ex["tunnel"].numpy().astype(bool)
        hit = (v2 > 1e-6) | tunnel
        sep = v2 - v1
        rebound_err = np.abs(sep - self.E * v) / (self.E * v) * 100.0
        mom_err = np.abs(m1 * v1 + m2 * v2 - m1 * v) / (m1 * v) * 100.0
        v2_law = (1 + self.E) * m1 * v / (m1 + m2)
        wall_err = np.abs(v2 - v2_law) / v2_law * 100.0
        pen = ex["max_pen"].numpy() * 1e3
        ok = hit & ~tunnel
        per = []
        for w in range(b.nworld):
            per.append({"world": w, "thick_mm": ex["thick"][w] * 1e3, "v": v[w], "tunnel": bool(tunnel[w]),
                        "hit": bool(hit[w]), "e_meas": float(sep[w] / v[w]), "rebound_err_pct": float(rebound_err[w]),
                        "momentum_err_pct": float(mom_err[w]), "wall_vel_err_pct": float(wall_err[w]),
                        "max_pen_mm": float(pen[w])})
        worst = float(np.max(rebound_err[ok])) if ok.any() else None
        met = rec["metrics"]
        met.update(
            tunnel_rate=float(tunnel.mean()), max_pen_mm=float(pen.max()),
            rebound_err_pct=worst, momentum_err_pct=float(np.max(mom_err[ok])) if ok.any() else None,
            wall_vel_err_pct=float(np.max(wall_err[ok])) if ok.any() else None,
            e_configured=self.E, conditions_covered=bool(b.nworld >= len(self.THICK) * len(self.SPEEDS)),
            per_condition=per[: 2 * len(self.SPEEDS) * len(self.THICK)],
        )
        met["law_pass"] = bool(not tunnel.any() and worst is not None and worst <= 5.0
                               and met["momentum_err_pct"] is not None and met["momentum_err_pct"] <= 5.0)
        rec["accuracy_error"] = worst
        if not hit.all():
            rec["notes"] += f"{int((~hit).sum())} world(s) without impact by the horizon; "



# =============================================================================================================
# P04 Hertz sphere
# =============================================================================================================
@wp.kernel
def _p04_drive(
    step: wp.array(dtype=int),
    h: float,
    bodies: wp.array(dtype=int),
    cond: wp.array(dtype=int),
    f_max: float,
    t_ramp: float,
    body_f: wp.array(dtype=wp.spatial_vector),
    aux: wp.array(dtype=wp.vec4),
):
    """Condition 0: eased push F(t) = F_max * ease(t / t_ramp) down on the sphere; impact worlds: no force."""
    w = wp.tid()
    t = float(step[0]) * h
    f = 0.0
    if cond[w] == 0:
        f = f_max * _ease(t / t_ramp)
    b = bodies[w]
    body_f[b] = body_f[b] + wp.spatial_vector(wp.vec3(0.0, 0.0, -f), wp.vec3(0.0))
    aux[w] = wp.vec4(f, 0.0, 0.0, 0.0)


@wp.kernel
def _p04_events(
    step: wp.array(dtype=int),
    h: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    bodies: wp.array(dtype=int),
    radius: float,
    z_top: float,
    t_on: wp.array(dtype=float),
    t_off: wp.array(dtype=float),
    v_off: wp.array(dtype=float),
    d_max: wp.array(dtype=float),
):
    """First contact interval of each world, with overlap delta = R - (z - z_top) > 0."""
    w = wp.tid()
    t = float(step[0] + 1) * h
    b = bodies[w]
    d = radius - (wp.transform_get_translation(body_q[b])[2] - z_top)
    if t_off[w] >= 0.0:
        return
    if d > 0.0:
        if t_on[w] < 0.0:
            t_on[w] = t
        if d > d_max[w]:
            d_max[w] = d
    elif t_on[w] >= 0.0:
        t_off[w] = t
        v_off[w] = wp.spatial_top(body_qd[b])[2]


def power_law_contact_time(m: float, K: float, n: float, v: float) -> float:
    """Duration of an elastic impact with F = K delta^n at approach speed v (energy integral, closed form).

    t_c = 2 delta_max / v * sqrt(pi) Gamma(1 + 1/(n+1)) / Gamma(1/2 + 1/(n+1)),
    delta_max = ((n+1) m v^2 / (2K))^(1/(n+1)).
    n = 1.5 with K = (4/3) E* sqrt(R) gives Hertz t_c = 2.87 (m^2 / (R E*^2 v))^(1/5); n = 1 gives pi sqrt(m / K)."""
    a = 1.0 / (n + 1.0)
    dmax = ((n + 1.0) * m * v * v / (2.0 * K)) ** a
    return 2.0 * dmax / v * math.sqrt(math.pi) * math.gamma(1.0 + a) / math.gamma(0.5 + a)


def mujoco_impedance(r: np.ndarray, solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)) -> np.ndarray:
    """MuJoCo solimp impedance d(r) (MuJoCo docs, Modeling > Solver parameters)."""
    d0, dw, width, mid, pw = solimp
    x = np.clip(np.abs(r) / width, 0.0, 1.0)
    y = np.where(x <= mid, x ** pw / mid ** (pw - 1.0), 1.0 - (1.0 - x) ** pw / (1.0 - mid) ** (pw - 1.0))
    return d0 + y * (dw - d0)


class HertzSphere(_PrecisionScenario):
    """P04: steel sphere R = 10 mm on a static box (top at z = 0). World condition 0: eased push 0 -> 50 N over
    0.5 s, then hold; conditions 1-3: impacts at v = 0.1, 0.5, 1.0 m/s from a 0.2 mm gap.

    Laws (A7): a fitted exponent n of F = K delta^n over delta/R in [0.005, 0.05] and the prefactor K, plus the
    impact duration t_c = power_law_contact_time(m, K, n, v) (Hertz: t_c = 2.87 (m^2/(R E*^2 v))^(1/5)), each per
    the configured contact law:
      hydro_*  default linear pressure field p = kh * depth on both bodies: F = pi k_eff R delta^2, n = 2,
               k_eff = kh_s kh_b / (kh_s + kh_b)   (derived in bench/notes/precision.md)
      point_*  MuJoCo soft contact with Newton's legacy ke/kd -> solref mapping (convert_solref(ke, kd, 1, 1)):
               at rest F = m ke delta d(delta)^2 / (dmax^2 (1 - d(delta))), d = solimp impedance (n ~ 1.1);
               impacts follow the constraint oscillator (omega = sqrt(ke) d/dmax), see _mujoco_impact
      xpbd     rigid (no compliance law): only penetration and contact duration are reported
    Reference Hertz for the same sphere with E* = 30 MPa is reported in metrics (hertz_* keys).
    accuracy_error = max(prefactor_err_pct, tc_err_pct over speeds).
    """

    id = "P04_hertz_sphere"
    title = "Sphere R=10 mm: 0-50 N load ramp and 0.1-1 m/s impacts vs Hertz / configured pressure law"
    horizon_s = 0.6
    accuracy_unit = "%"
    accuracy_ref = "F = K delta^n fit over delta/R in [0.005, 0.05] and t_c of the configured contact law (A7)"
    R, RHO, F_MAX, T_RAMP, GAP0 = 0.010, 7800.0, 50.0, 0.5, 0.0002
    SPEEDS = (0.1, 0.5, 1.0)
    KH_SPHERE, KH_BOX = 1.0e10, 1.0e12
    KE, KD = 2.0e5, 90.0
    E_STAR_REF = 30.0e6
    configs = (
        Config("hydro_mujoco", "mujoco", dt=5e-5, substeps=1, params={"backend": "hydro", "sdf_res": 64}),
        Config("hydro_kamino", "kamino", dt=5e-5, substeps=1, params={"backend": "hydro", "sdf_res": 64}),
        Config("point_mujoco", "mujoco", dt=5e-5, substeps=1, params={"backend": "point"}),
        Config("xpbd", "xpbd", dt=5e-5, substeps=1, params={"backend": "point", "iterations": 10}),
    )

    def _shape_cfgs(self, cfg: Config) -> tuple[Any, Any]:
        base = newton.ModelBuilder.ShapeConfig(density=self.RHO, mu=0.3, ke=self.KE, kd=self.KD, gap=0.001,
                                               mu_torsional=0.0, mu_rolling=0.0)
        if _backend(cfg) != "hydro":
            return base, base
        res = int(cfg.params.get("sdf_res", 64))
        sph = replace(base, is_hydroelastic=True, kh=self.KH_SPHERE, sdf_max_resolution=res,
                      sdf_narrow_band_range=(-0.002, 0.002))
        box = replace(base, is_hydroelastic=True, kh=self.KH_BOX, sdf_max_resolution=res,
                      sdf_narrow_band_range=(-0.002, 0.002))
        return sph, box

    def build(self, nworld: int, device: str, config: Config) -> Built:
        _require_cuda_for_hydro(config, device)
        sph_cfg, box_cfg = self._shape_cfgs(config)
        sb = newton.ModelBuilder()
        sb.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.005), wp.quat_identity()), hx=0.03, hy=0.03,
                         hz=0.005, cfg=box_cfg)
        ball = sb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, self.R), wp.quat_identity()), label="ball")
        sb.add_shape_sphere(ball, radius=self.R, cfg=sph_cfg)
        main = newton.ModelBuilder()
        _register(config, main)
        _, sub, ids = _replicate_blocks(main, [sb], nworld, [[0]])
        model = main.finalize(device=device)
        ncond = 1 + len(self.SPEEDS)
        cond = sub % ncond
        q7 = np.zeros((nworld, 7))
        q7[:, 6] = 1.0
        qd6 = np.zeros((nworld, 6))
        for w in range(nworld):
            if cond[w] == 0:
                q7[w, 2] = self.R
            else:
                q7[w, 2] = self.R + self.GAP0
                qd6[w, 2] = -self.SPEEDS[cond[w] - 1]
        _set_free_bodies(model, ids[:, 0], q7, qd6)
        cpw = 64
        hydro = None
        if _backend(config) == "hydro":
            from alloy._vendor.newton.geometry import HydroelasticSDF

            hydro = HydroelasticSDF.Config()
        pipe, contacts = make_pipeline(model, cpw, nworld, hydro=hydro)
        solver, notes = make_solver(config, model, cpw, device)
        s0, s1, control = _states(model)
        h = config.dt / config.substeps
        ntrace = min(nworld, ncond)
        every = max(1, int(round(0.5e-3 / h)))
        ex = self.base_extra(device, nworld, ntrace, _nrec(self.horizon_s, h, every), every, ids[:, 0])
        ex.update(notes=notes, ball=wp.array(ids[:, 0], dtype=int, device=device),
                  cond=wp.array(cond.astype(np.int32), dtype=int, device=device), cond_np=cond,
                  t_on=wp.full(nworld, -1.0, dtype=float, device=device),
                  t_off=wp.full(nworld, -1.0, dtype=float, device=device),
                  v_off=wp.zeros(nworld, dtype=float, device=device),
                  d_max=wp.zeros(nworld, dtype=float, device=device),
                  mass=float(model.body_mass.numpy()[ids[0, 0]]), g=float(np.linalg.norm(_gravity(model))),
                  h=h, cfg=config)
        return Built(model, solver, s0, s1, control, contacts, pipe, nworld=nworld, extra=ex)

    def drive(self, b: Built, h: float) -> None:
        wp.launch(_p04_drive, dim=b.nworld, inputs=[b.extra["step"], h, b.extra["ball"], b.extra["cond"],
                                                    self.F_MAX, self.T_RAMP, b.state_0.body_f, b.extra["aux"]])

    def post(self, b: Built, h: float) -> None:
        super().post(b, h)
        ex = b.extra
        wp.launch(_p04_events, dim=b.nworld, inputs=[ex["step"], h, b.state_0.body_q, b.state_0.body_qd, ex["ball"],
                                                     self.R, 0.0, ex["t_on"], ex["t_off"], ex["v_off"], ex["d_max"]])

    # ---- laws --------------------------------------------------------------------------------------------
    def law(self, cfg: Config, m: float) -> dict:
        be = _backend(cfg)
        if be == "hydro":
            k_eff = self.KH_SPHERE * self.KH_BOX / (self.KH_SPHERE + self.KH_BOX)
            return {"law_kind": "hydro_linear_pressure", "n": 2.0, "K": math.pi * k_eff * self.R}
        if cfg.solver == "mujoco":
            return {"law_kind": "mujoco_soft_penalty", "n": None, "K": None}
        return {"law_kind": "rigid", "n": None, "K": None}

    def _mujoco_force(self, m: float, delta: np.ndarray) -> np.ndarray:
        d = mujoco_impedance(delta)
        return m * self.KE * delta * d ** 2 / (0.95 ** 2 * (1.0 - d))

    def _mujoco_impact(self, delta_max: float) -> tuple[float, float]:
        """Impact duration and restitution of MuJoCo's soft contact: the constraint follows d * aref with
        k = d ke / dmax^2, b = kd / dmax (legacy unit-mass solref mapping), i.e. a damped oscillator with
        omega = sqrt(ke) d / dmax and 2 zeta omega = d kd / dmax; d is taken at half the peak overlap."""
        dmax = 0.95
        d = float(mujoco_impedance(np.array([0.5 * delta_max]))[0])
        om = math.sqrt(self.KE) * d / dmax
        zeta = d * self.KD / dmax / (2.0 * om)
        wd = om * math.sqrt(max(1e-12, 1.0 - zeta * zeta))
        return math.pi / wd, math.exp(-zeta * math.pi / math.sqrt(max(1e-12, 1.0 - zeta * zeta)))

    def finish(self, b: Built, rec: dict) -> None:
        self.finish_common(b, rec)
        ex, met = b.extra, rec["metrics"]
        m, g = ex["mass"], ex["g"]
        tr = _trace_rows(b)
        law = self.law(ex["cfg"], m)
        met.update(law)
        # Quasi-static branch (world 0 is condition 0).
        fit = {"exponent": None, "prefactor": None}
        rows = tr[0][tr[0, :, 0] >= 0] if tr.shape[0] > 0 else np.zeros((0, TRACE_W))
        if len(rows):
            delta = self.R - rows[:, 3]
            force = rows[:, 14] + m * g
            vz = np.abs(rows[:, 10])
            sel = (delta > 0.005 * self.R) & (delta < 0.05 * self.R) & (vz < 0.01)
            met["fit_points"] = int(sel.sum())
            met["delta_max_mm"] = float(delta.max() * 1e3)
            met["force_end_n"] = float(force[-1])
            if sel.sum() >= 5:
                n, logk = np.polyfit(np.log(delta[sel]), np.log(force[sel]), 1)
                fit = {"exponent": float(n), "prefactor": float(math.exp(logk))}
                if law["law_kind"] == "mujoco_soft_penalty":
                    ref = self._mujoco_force(m, delta[sel])
                    met["penalty_law_err_pct"] = float(np.max(np.abs(force[sel] - ref) / ref) * 100.0)
        met.update(fit)
        # A7 as written (Hertz exponent 1.5 +- 0.05): reported for every backend; the pass/fail below uses the
        # configured law, because no backend here is a Hertz material (hydro is linear-pressure, n = 2).
        met["hertz_exponent_pass"] = bool(fit["exponent"] is not None and abs(fit["exponent"] - 1.5) <= 0.05)
        err_terms = []
        if law["n"] is not None and fit["exponent"] is not None:
            met["exponent_err"] = abs(fit["exponent"] - law["n"])
            # prefactor compared at the fitted points through the law exponent: K_meas = mean(F / delta^n_law)
            dsel = self.R - rows[:, 3]
            fsel = rows[:, 14] + m * g
            sel = (dsel > 0.005 * self.R) & (dsel < 0.05 * self.R) & (np.abs(rows[:, 10]) < 0.01)
            k_meas = float(np.median(fsel[sel] / dsel[sel] ** law["n"]))
            met["prefactor_err_pct"] = abs(k_meas - law["K"]) / law["K"] * 100.0
            err_terms.append(met["prefactor_err_pct"])
        elif "penalty_law_err_pct" in met:
            err_terms.append(met["penalty_law_err_pct"])
        # Impacts.
        cond = ex["cond_np"]
        t_on, t_off = ex["t_on"].numpy(), ex["t_off"].numpy()
        v_off, d_max = ex["v_off"].numpy(), ex["d_max"].numpy()
        per = []
        for c, v in enumerate(self.SPEEDS, start=1):
            ws = np.where(cond == c)[0]
            if not len(ws):
                continue
            w = ws[0]
            tc = float(t_off[w] - t_on[w]) if t_off[w] >= 0 else None
            row = {"v": v, "tc_ms": None if tc is None else tc * 1e3, "delta_max_mm": float(d_max[w] * 1e3),
                   "e_meas": float(v_off[w] / v) if t_off[w] >= 0 else None}
            K, n = (law["K"], law["n"])
            tc_law = None
            if law["law_kind"] == "mujoco_soft_penalty":
                tc_law, e_law = self._mujoco_impact(float(d_max[w]))
                row["e_law"] = e_law
            elif K is not None:
                tc_law = power_law_contact_time(m, K, n, v)
            if tc_law is not None:
                row["tc_law_ms"] = tc_law * 1e3
                if tc is not None:
                    row["tc_err_pct"] = abs(tc - tc_law) / tc_law * 100.0
                    err_terms.append(row["tc_err_pct"])
            row["tc_hertz_ref_ms"] = power_law_contact_time(m, 4.0 / 3.0 * self.E_STAR_REF * math.sqrt(self.R),
                                                            1.5, v) * 1e3
            per.append(row)
        met["per_condition"] = per
        met["conditions_covered"] = bool(b.nworld >= 1 + len(self.SPEEDS))
        met["hertz_ref"] = {"E_star_pa": self.E_STAR_REF, "K": 4.0 / 3.0 * self.E_STAR_REF * math.sqrt(self.R),
                            "n": 1.5}
        rec["accuracy_error"] = float(max(err_terms)) if err_terms else None
        if law["law_kind"] == "rigid":
            met["law_pass"] = bool(float(d_max.max()) * 1e3 <= 0.02)  # non-penetration is the only law
            met["law_note"] = "rigid contact: no force law; pass = peak overlap <= 0.02 mm"
            return
        ok_exp = law["n"] is None or (fit["exponent"] is not None and abs(fit["exponent"] - law["n"]) <= 0.05)
        met["law_pass"] = bool(max(err_terms) <= 10.0 and ok_exp) if err_terms else None  # None: nothing measured



# =============================================================================================================
# P01 peg in hole, 0.2 mm diametral clearance
# =============================================================================================================
@wp.kernel
def _p01_drive(
    step: wp.array(dtype=int),
    h: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    bodies: wp.array(dtype=int),
    p0: wp.array(dtype=wp.vec3),
    q_t: wp.array(dtype=wp.quat),
    gravity: wp.vec3,
    speed: float,
    t1: float,
    f_push: float,
    t_ramp: float,
    k_lin: wp.vec3,
    c_lin: wp.vec3,
    k_ang: wp.vec3,
    c_ang: wp.vec3,
    body_f: wp.array(dtype=wp.spatial_vector),
    aux: wp.array(dtype=wp.vec4),
    peak: wp.array(dtype=float),
    lag: wp.array(dtype=float),
):
    """Compliant holder: 6-DOF spring-damper to a target pose that moves down at `speed` until t1, then a
    force-controlled push eased 0 -> f_push over t_ramp (lateral and angular springs stay). Gravity cancelled.
    Angular gains are per body axis (k_ang, c_ang in the body frame): the yaw inertia of a slim peg is ~7x smaller
    than its tilt inertia, and one isotropic damper made the explicit yaw loop unstable (c h / I_zz = 4 > 2)."""
    w = wp.tid()
    b = bodies[w]
    t = float(step[0]) * h
    X = body_q[b]
    p = wp.transform_get_translation(X)
    q = wp.transform_get_rotation(X)
    v = wp.spatial_top(body_qd[b])
    om = wp.spatial_bottom(body_qd[b])
    tgt = p0[w]
    tz = tgt[2] - speed * wp.min(t, t1)
    e = wp.vec3(tgt[0] - p[0], tgt[1] - p[1], tz - p[2])
    f = wp.cw_mul(k_lin, e) - wp.cw_mul(c_lin, v)
    phase = 0.0
    if t >= t1:
        phase = 1.0
        f = wp.vec3(f[0], f[1], -f_push * _ease((t - t1) / t_ramp) - c_lin[2] * v[2])
    else:
        wp.atomic_max(peak, w, wp.abs(f[2]))
        wp.atomic_max(lag, w, p[2] - tz)
    rv_b = wp.quat_rotate_inv(q, _rotvec(q_t[w], q))
    om_b = wp.quat_rotate_inv(q, om)
    tau = wp.quat_rotate(q, wp.cw_mul(k_ang, rv_b) - wp.cw_mul(c_ang, om_b))
    m = body_mass[b]
    body_f[b] = body_f[b] + wp.spatial_vector(f - m * gravity, tau)
    aux[w] = wp.vec4(f[0], f[1], f[2], phase)


def _a8_tilt_bound(d: float, D: float, l: np.ndarray) -> np.ndarray:
    """A8: largest tilt with d cos(theta) + l sin(theta) = D (two-point contact at engagement l), radians."""
    R = np.hypot(d, l)
    return np.arctan2(l, d) - np.arccos(np.clip(D / R, -1.0, 1.0))


class PegHole(_PrecisionScenario):
    """P01: d = 9.80 mm steel peg (30 mm long) into a D = 10.00 mm blind hole, 20 mm deep, in a 30 x 30 x 25 mm
    block; entry plain or chamfered 0.5 mm x 45 deg (geometry variant, world blocks). The peg hangs in a compliant
    holder (k_x = k_y = 1000 N/m, k_z = 5000 N/m, k_theta = 0.5 N m/rad, near-critical damping) whose target moves
    down at 1 mm/s from 1 mm above the hole to the hole bottom (21 s), then pushes with 50 N eased in over 0.5 s.
    Conditions inside a block: tilt {0, 0.5, 1, 2} deg (about the tip, holder keeps the tilt) x offset {0, 0.05,
    0.1} mm in x, i.e. world sub-index % 12.

    Laws: A8 clearance geometry, interference d cos(theta) + l sin(theta) - D <= 0 at every engagement l (headline
    accuracy_error = max interference [mm]; tilt_bound_err_deg = max excess tilt, pass <= 0.01 deg per
    docs/research/bench.md A8). Centred gap preserved: the
    tilt 0 / offset 0 world stays centred within 1 um (centred_dev_um). CON-46: under the 50 N push the tip goes
    < 0.05 mm below the hole floor (push_pen_mm). bench.md #9: penetration_p99_mm <= 0.02.
    primitive config: cylinder peg in a ring of 32 boxes (inscribed D = 10.00 mm) on a floor box; no chamfer.
    """

    id = "P01_peg_hole_0p2"
    title = "9.80 mm peg into 10.00 mm hole, 20 mm deep, compliant 1 mm/s insertion + 50 N push"
    horizon_s = 22.0
    accuracy_unit = "mm"
    accuracy_ref = "A8: d cos(theta) + l sin(theta) <= D (max interference); CON-46 push < 0.05 mm; p99 <= 0.02 mm"
    D_PEG, D_HOLE, DEPTH, L_PEG, CHAMFER = 0.0098, 0.0100, 0.020, 0.030, 0.0005
    BLOCK_HALF, BLOCK_H, H0, SPEED, F_PUSH, T_RAMP = 0.015, 0.025, 0.001, 0.001, 50.0, 0.5
    TILTS_DEG = (0.0, 0.5, 1.0, 2.0)
    OFFSETS = (0.0, 0.00005, 0.0001)
    K_LIN, K_ANG = (1000.0, 1000.0, 5000.0), 0.5
    MU = 0.2
    _sdf = {"sdf_res": 256, "tri_pairs_per_world": 1024}
    configs = (
        Config("sdf_mujoco", "mujoco", dt=1e-3, substeps=2, params={**_sdf}),
        Config("sdf_kamino", "kamino", dt=1e-3, substeps=2, params={**_sdf}),
        Config("sdf_xpbd", "xpbd", dt=1e-3, substeps=2, params={**_sdf, "iterations": 10}),
        Config("sdf_steel", "steel", dt=1e-3, substeps=2, params={**_sdf}),
        Config("hydro_mujoco", "mujoco", dt=1e-3, substeps=2, params={**_sdf, "kh": 1.0e12, "collide_every": 1}),
        Config("primitive", "mujoco", dt=1e-3, substeps=2, params={"backend": "primitive"}),
    )

    def t1(self) -> float:
        return (self.H0 + self.DEPTH) / self.SPEED

    def _shape_cfg(self, cfg: Config) -> Any:
        base = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=self.MU, ke=1.0e7, kd=1.0e4, gap=0.0003,
                                               margin=0.0, mu_torsional=0.0, mu_rolling=0.0)
        if _backend(cfg) == "hydro":
            return replace(base, is_hydroelastic=True, kh=float(cfg.params.get("kh", 1.0e12)), gap=0.0)
        return base

    def _variant(self, chamfer: float, cfg: Config, device: str, notes: dict) -> newton.ModelBuilder:
        sc = self._shape_cfg(cfg)
        sb = newton.ModelBuilder()
        if _backend(cfg) == "primitive":
            n, r_in, wall = 32, self.D_HOLE / 2, 0.002
            half_w = r_in * math.tan(math.pi / n) * 1.02
            for k in range(n):
                a = 2 * math.pi * k / n
                c = wp.vec3((r_in + wall / 2) * math.cos(a), (r_in + wall / 2) * math.sin(a), -self.DEPTH / 2)
                sb.add_shape_box(-1, xform=wp.transform(c, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a)),
                                 hx=wall / 2, hy=half_w, hz=self.DEPTH / 2, cfg=sc)
            sb.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -self.DEPTH - 0.0025), wp.quat_identity()),
                             hx=0.01, hy=0.01, hz=0.0025, cfg=sc)
            peg = sb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), label="peg")
            sb.add_shape_cylinder(peg, radius=self.D_PEG / 2, half_height=self.L_PEG / 2, cfg=sc)
            notes["sdf_backend"] = "primitive"
            return sb
        res = int(cfg.params.get("sdf_res", 256))
        band, margin = (-0.001, 0.001), 0.001
        hv, hi = mesh_block_with_hole(self.BLOCK_HALF, self.BLOCK_H, self.D_HOLE / 2, self.DEPTH, chamfer)
        hole = newton.Mesh(hv, hi)
        notes["sdf_backend"] = _attach_sdf(hole, device, res, band, margin)
        sb.add_shape_mesh(-1, mesh=hole, cfg=sc, label="hole_block")
        pv, pi = mesh_cylinder(self.D_PEG / 2, self.L_PEG / 2)
        pmesh = newton.Mesh(pv, pi)
        _attach_sdf(pmesh, device, res, band, margin)
        peg = sb.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), label="peg")
        sb.add_shape_mesh(peg, mesh=pmesh, cfg=sc)
        return sb

    def build(self, nworld: int, device: str, config: Config) -> Built:
        _require_cuda_for_hydro(config, device)
        notes: dict[str, Any] = {}
        chamfers = (0.0,) if _backend(config) == "primitive" else (0.0, self.CHAMFER)
        variants = [self._variant(c, config, device, notes) for c in chamfers]
        main = newton.ModelBuilder()
        _register(config, main)
        var, sub, ids = _replicate_blocks(main, variants, nworld, [[0]] * len(variants))
        model = main.finalize(device=device)
        nsub = len(self.TILTS_DEG) * len(self.OFFSETS)
        cidx = sub % nsub
        tilt = np.radians([self.TILTS_DEG[c // len(self.OFFSETS)] for c in cidx])
        offs = np.array([self.OFFSETS[c % len(self.OFFSETS)] for c in cidx])
        chamf = np.array([chamfers[g] for g in var])
        # tip centre at (offset, 0, H0); body centre = tip + R(tilt) * (0, 0, L/2)
        q7 = np.zeros((nworld, 7))
        quats = np.stack([_quat_tilt_y(t) for t in tilt])
        axis = _quat_mat(quats)[:, :, 2]
        q7[:, :3] = np.stack([offs, np.zeros(nworld), np.full(nworld, self.H0)], -1) + axis * self.L_PEG / 2
        q7[:, 3:] = quats
        _set_free_bodies(model, ids[:, 0], q7, np.zeros((nworld, 6)))
        hydro = None
        if _backend(config) == "hydro":
            from alloy._vendor.newton.geometry import HydroelasticSDF

            hydro = HydroelasticSDF.Config()
        cpw = int(config.params.get("contacts_per_world", 512))
        pipe, contacts = make_pipeline(model, cpw, nworld, hydro=hydro, **_tri_pairs_kw(config, nworld))
        solver, snotes = make_solver(config, model, cpw, device)
        notes.update(snotes)
        s0, s1, control = _states(model)
        h = config.dt / config.substeps
        ncond = len(chamfers) * nsub
        ntrace = min(nworld, ncond)
        every = max(1, int(round(0.02 / h)))  # 20 ms rows (0.02 mm of travel)
        ex = self.base_extra(device, nworld, ntrace, _nrec(self.horizon_s, h, every), every, ids[:, 0])
        m = model.body_mass.numpy()[ids[:, 0]]
        inertia = np.diag(model.body_inertia.numpy()[ids[0, 0]]).astype(np.float64)
        c_lin = [2.0 * math.sqrt(k * m[0]) for k in self.K_LIN]
        k_ang = [self.K_ANG, self.K_ANG, self.K_ANG * inertia[2] / inertia[0]]  # same natural frequency on yaw
        c_ang = [2.0 * math.sqrt(k * i) for k, i in zip(k_ang, inertia)]
        ex.update(notes=notes, cfg=config, peg=wp.array(ids[:, 0], dtype=int, device=device), ids=ids,
                  p0=wp.array(q7[:, :3], dtype=wp.vec3, device=device),
                  q_t=wp.array(quats, dtype=wp.quat, device=device),
                  gravity=wp.vec3(*_gravity(model)), c_lin=wp.vec3(*c_lin), k_ang=wp.vec3(*k_ang),
                  c_ang=wp.vec3(*c_ang),
                  peak=wp.zeros(nworld, dtype=float, device=device),
                  lag=wp.zeros(nworld, dtype=float, device=device),
                  tilt=tilt, offset=offs, chamfer=chamf, var=var, ncond=ncond)
        return Built(model, solver, s0, s1, control, contacts, pipe, nworld=nworld, extra=ex)

    def drive(self, b: Built, h: float) -> None:
        ex = b.extra
        wp.launch(_p01_drive, dim=b.nworld, inputs=[
            ex["step"], h, b.state_0.body_q, b.state_0.body_qd, b.model.body_mass, ex["peg"], ex["p0"], ex["q_t"],
            ex["gravity"], self.SPEED, self.t1(), self.F_PUSH, self.T_RAMP, wp.vec3(*self.K_LIN), ex["c_lin"],
            ex["k_ang"], ex["c_ang"], b.state_0.body_f, ex["aux"], ex["peak"], ex["lag"]])

    def _tip(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tip centre and peg axis for body rows (..., 7)."""
        axis = _quat_mat(q[..., 3:7])[..., :, 2]
        return q[..., :3] - axis * self.L_PEG / 2, axis

    def finish(self, b: Built, rec: dict) -> None:
        self.finish_common(b, rec)
        ex, met = b.extra, rec["metrics"]
        d, D = self.D_PEG, self.D_HOLE
        tr = _trace_rows(b)
        per, inter_max, excess_max, centred_dev, engaged = [], 0.0, 0.0, None, False
        for w in range(tr.shape[0]):
            rows = tr[w][tr[w, :, 0] >= 0]
            if not len(rows):
                continue
            pose = np.concatenate([rows[:, 1:4], rows[:, 4:8]], -1).astype(np.float64)
            tip, axis = self._tip(pose)
            theta = np.arccos(np.clip(axis[:, 2], -1.0, 1.0))
            l_eng = (-ex["chamfer"][w] - tip[:, 2]) / np.maximum(np.cos(theta), 1e-6)
            eng = l_eng > 0.0005
            engaged = engaged or bool(eng.any())
            inter = (d * np.cos(theta) + l_eng * np.sin(theta) - D)[eng] * 1e3 if eng.any() else np.zeros(1)
            excess = np.degrees(theta[eng] - _a8_tilt_bound(d, D, l_eng[eng])) if eng.any() else np.zeros(1)
            lat = np.hypot(tip[:, 0], tip[:, 1])
            row = {"world": w, "chamfer_mm": ex["chamfer"][w] * 1e3, "tilt_deg": float(np.degrees(ex["tilt"][w])),
                   "offset_mm": ex["offset"][w] * 1e3, "depth_reached_mm": float(-tip[:, 2].min() * 1e3),
                   "max_interference_mm": float(inter.max()), "max_tilt_excess_deg": float(excess.max()),
                   "peak_force_n": float(ex["peak"].numpy()[w])}
            inter_max, excess_max = max(inter_max, row["max_interference_mm"]), max(excess_max,
                                                                                   row["max_tilt_excess_deg"])
            if ex["tilt"][w] == 0.0 and ex["offset"][w] == 0.0:
                dev = float(lat[eng].max() * 1e6) if eng.any() else float(lat.max() * 1e6)
                row["centred_dev_um"] = dev
                centred_dev = dev if centred_dev is None else max(centred_dev, dev)
            push = rows[:, 17] > 0.5
            if push.any():
                row["push_pen_mm"] = float((-self.DEPTH - tip[push, 2].min()) * 1e3)
            per.append(row)
        q_end = b.state_0.body_q.numpy()[ex["ids"][:, 0]].astype(np.float64)
        tip_end, _ = self._tip(q_end)
        depth = -tip_end[:, 2]
        lag = ex["lag"].numpy()
        t_end = int(ex["step"].numpy()[0]) * (ex["cfg"].dt / ex["cfg"].substeps)
        target_depth = min(self.DEPTH, max(0.0, self.SPEED * t_end - self.H0))
        success = depth >= target_depth - 0.0005
        jam = lag > 0.002
        pushes = [r["push_pen_mm"] for r in per if "push_pen_mm" in r]
        met.update(
            success_rate=float(success.mean()), jam_rate=float(jam.mean()),
            depth_reached_mm=float(depth.mean() * 1e3), target_depth_mm=target_depth * 1e3,
            peak_force_n=float(ex["peak"].numpy().max()), tilt_bound_err_deg=float(max(0.0, excess_max)),
            max_interference_mm=float(inter_max), centred_dev_um=centred_dev,
            push_pen_mm=float(max(pushes)) if pushes else None,
            conditions_covered=bool(b.nworld >= ex["ncond"]), per_condition=per,
        )
        rec["accuracy_error"] = float(max(0.0, inter_max))
        pen = rec.get("penetration_p99_mm")
        met["law_pass"] = bool(inter_max <= 0.02 and excess_max <= 0.01 and (centred_dev is None or centred_dev <= 1.0)
                               and (not pushes or max(pushes) < 0.05) and (pen is None or pen <= 0.02))
        if not engaged:
            met["law_pass"] = None  # nothing to judge: no traced world entered the hole (short horizon)


# =============================================================================================================
# P02 nut run down a bolt (thread law A9)
# =============================================================================================================
@wp.kernel
def _p02_drive(
    step: wp.array(dtype=int),
    h: float,
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    bodies: wp.array(dtype=int),
    gravity: wp.vec3,
    f_axial: float,
    omega: float,
    t_ramp: float,
    kp: wp.array(dtype=float),
    ki: wp.array(dtype=float),
    integ: wp.array(dtype=float),
    body_f: wp.array(dtype=wp.spatial_vector),
    tau_out: wp.array(dtype=float),
):
    """Velocity servo on the nut's free revolute DOF (PI torque about +z, explicit and exactly known), target
    omega eased in over t_ramp; axial load f_axial along +z (opposing the advance) eased in the same way; gravity
    cancelled so the axial load is exactly f_axial."""
    w = wp.tid()
    b = bodies[w]
    s = _ease(float(step[0]) * h / t_ramp)
    err = omega * s - wp.spatial_bottom(body_qd[b])[2]
    integ[w] = integ[w] + err * h
    tau = kp[w] * err + ki[w] * integ[w]
    f = wp.vec3(0.0, 0.0, f_axial * s) - body_mass[b] * gravity
    body_f[b] = body_f[b] + wp.spatial_vector(f, wp.vec3(0.0, 0.0, tau))
    tau_out[w] = tau


@wp.kernel
def _p02_post(
    step: wp.array(dtype=int),
    h: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    bodies: wp.array(dtype=int),
    omega: float,
    t_meas: float,
    tau_in: wp.array(dtype=float),
    acc: wp.array2d(dtype=float),
    aux: wp.array(dtype=wp.vec4),
):
    """acc columns: 0 theta (unwrapped yaw), 1 int tau_dir dt over the window, 2 window time, 3 theta at window
    start, 4 z at window start, 5 max |omega - omega_t| / |omega_t| in the window, 6 omega at window start.
    tau_dir = torque along the direction of rotation."""
    w = wp.tid()
    t = float(step[0] + 1) * h
    b = bodies[w]
    wz = wp.spatial_bottom(body_qd[b])[2]
    z = wp.transform_get_translation(body_q[b])[2]
    acc[w, 0] = acc[w, 0] + wz * h
    sgn = wp.sign(omega)
    tau_dir = tau_in[w] * sgn
    phase = 0.0
    if t >= t_meas:
        phase = 1.0
        if acc[w, 2] == 0.0:
            acc[w, 3] = acc[w, 0]
            acc[w, 4] = z
            acc[w, 6] = wz
        acc[w, 1] = acc[w, 1] + tau_dir * h
        acc[w, 2] = acc[w, 2] + h
        acc[w, 5] = wp.max(acc[w, 5], wp.abs(wz - omega) / wp.abs(omega))
    aux[w] = wp.vec4(tau_dir, acc[w, 0], wz, phase)


def thread_torque_law(F: float, d2: float, p: float, mu: float, half_angle_deg: float = 30.0) -> float:
    """A9 power-screw torque to drive a nut against an axial load F (raising form):
    T = F (d2/2) (p + pi mu d2 sec a) / (pi d2 - mu p sec a), a = flank half-angle (30 deg for ISO)."""
    sec = 1.0 / math.cos(math.radians(half_angle_deg))
    return F * d2 / 2.0 * (p + math.pi * mu * d2 * sec) / (math.pi * d2 - mu * p * sec)


class NutBolt(_PrecisionScenario):
    """P02: nut on a static bolt, rotated by a velocity servo about the bolt axis (pi rad/s, eased in over 0.5 s)
    under a 5 N axial load that opposes the advance (gravity cancelled). The nut sits on a D6 joint with only
    axial translation and axial rotation free (a revolute + prismatic pair on the bolt axis), so the thread
    contact alone sets the advance. The servo is a PI torque written to body_f, so the applied torque is known
    exactly for every solver (a joint velocity drive would report solver-specific forces).

    Geometry variants (world blocks): procedural ISO 68-1 M10x1.5 (d2 = 9.026 mm, radial clearance 0.05 mm,
    8 mm nut), Factory m12_loose (p = 1.75 mm) and m20_loose (p = 2.5 mm) meshes from IsaacGymEnvs
    (assets/factory/yaml/factory_asset_info_nut_bolt.yaml:69,140). The Factory nut starts in its CAD pose raised by
    an integer number of pitches, so the thread phase is unchanged.

    Laws (A9): advance per revolution = p within 0.5 % (z vs unwrapped yaw fitted over the window), and the mean
    servo torque T = F (d2/2)(p + pi mu d2 sec 30)/(pi d2 - mu p sec 30) within 5 % (d2 = d - 0.649519 p, ISO 724;
    the Factory d2 is nominal, the loose meshes may differ). cross_thread = advance off by > 25 % or the servo
    lost the speed by > 50 %. accuracy_error = advance_err_pct (the geometric law); torque_err_pct is in metrics.
    """

    id = "P02_nut_bolt"
    title = "Nut run down a bolt by a pi rad/s servo under 5 N axial load: advance = p, power-screw torque"
    horizon_s = 3.0
    accuracy_unit = "%"
    accuracy_ref = ("A9: advance per rev = p within 0.5 %; "
                    "T = F d2/2 (p + pi mu d2 sec30)/(pi d2 - mu p sec30) within 5 %")
    OMEGA, F_AX, T_RAMP, T_MEAS, MU, ROTOR_I = -math.pi, 5.0, 0.5, 0.75, 0.1, 1.0e-5
    VARIANTS = ("iso_m10x1p5", "factory_m12_loose", "factory_m20_loose")
    FACTORY = {"factory_m12_loose": {"d": 0.012, "p": 0.00175, "raise_pitches": 4},
               "factory_m20_loose": {"d": 0.020, "p": 0.0025, "raise_pitches": 8}}
    ISO = {"d": 0.010, "p": 0.0015, "length": 0.020, "nut_z0": 0.010, "nut_h": 0.008, "nut_r": 0.008,
           "clearance": 0.00005}
    _m = {"mesh_n": 96, "rows_per_pitch": 16, "tri_pairs_per_world": 4096}
    configs = (
        Config("sdf_mujoco", "mujoco", dt=1 / 120, substeps=5, params={**_m, "sdf_res": 512}),
        Config("sdf_kamino", "kamino", dt=1 / 120, substeps=5, params={**_m, "sdf_res": 512}),
        Config("sdf_xpbd", "xpbd", dt=1 / 120, substeps=5, params={**_m, "sdf_res": 512, "iterations": 10}),
        Config("hydro_mujoco", "mujoco", dt=1 / 120, substeps=4,
               params={**_m, "sdf_res": 128, "kh": 1.0e11, "collide_every": 2}),
        Config("hydro_kamino", "kamino", dt=1 / 120, substeps=4,
               params={**_m, "sdf_res": 128, "kh": 1.0e11, "collide_every": 2}),
    )

    def _shape_cfg(self, cfg: Config) -> Any:
        base = newton.ModelBuilder.ShapeConfig(density=8000.0, mu=self.MU, ke=1.0e7, kd=1.0e4, gap=0.001,
                                               margin=0.0, mu_torsional=0.0, mu_rolling=0.0)
        if _backend(cfg) == "hydro":
            return replace(base, is_hydroelastic=True, kh=float(cfg.params.get("kh", 1.0e11)), gap=0.0)
        return base

    def _meshes(self, name: str, cfg: Config) -> tuple[tuple, tuple, float, dict]:
        """(bolt (v, i), nut (v, i), nut body z0, law geometry) with vertices in the bolt frame."""
        if name == "iso_m10x1p5":
            g, n, rpp = self.ISO, int(cfg.params["mesh_n"]), int(cfg.params["rows_per_pitch"])
            bolt = mesh_thread_bolt(g["d"], g["p"], g["length"], n=n, rows_per_pitch=rpp)
            nut = mesh_thread_nut(g["d"], g["p"], g["nut_z0"], g["nut_h"], g["nut_r"], g["clearance"], n=n,
                                  rows_per_pitch=rpp)
            return bolt, nut, 0.0, {"d": g["d"], "p": g["p"]}
        f = self.FACTORY[name]
        root = _factory_dir()
        size = name.split("_", 1)[1]
        bolt = load_obj(root / f"factory_bolt_{size}.obj")
        nut = load_obj(root / f"factory_nut_{size}_subdiv_3x.obj")
        return bolt, nut, f["raise_pitches"] * f["p"], {"d": f["d"], "p": f["p"]}

    def _variant(self, name: str, cfg: Config, device: str, notes: dict) -> tuple[newton.ModelBuilder, dict]:
        sc = self._shape_cfg(cfg)
        (bv, bi), (nv, ni), z0, geo = self._meshes(name, cfg)
        res = int(cfg.params.get("sdf_res", 512))
        band, margin = (-0.002, 0.002), 0.002
        sb = newton.ModelBuilder()
        bolt = newton.Mesh(bv, bi)
        notes["sdf_backend"] = _attach_sdf(bolt, device, res, band, margin)
        sb.add_shape_mesh(-1, mesh=bolt, cfg=sc, label="bolt")
        nut_mesh = newton.Mesh(nv, ni)
        _attach_sdf(nut_mesh, device, res, band, margin)
        X0 = wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity())
        nut = sb.add_link(xform=X0, label="nut")
        sb.add_shape_mesh(nut, mesh=nut_mesh, cfg=sc)
        # Drive rotor inertia (isotropic, so the tensor stays valid): a bare 9 g nut has I_zz ~ 4e-7 kg m^2, which
        # an explicit servo cannot hold against mN m thread torques without chatter.
        I = np.array(sb.body_inertia[nut], dtype=np.float64).reshape(3, 3) + np.eye(3) * self.ROTOR_I
        sb.body_inertia[nut] = wp.mat33(*I.astype(np.float32).ravel())
        sb.body_inv_inertia[nut] = wp.mat33(*np.linalg.inv(I).astype(np.float32).ravel())
        J = newton.ModelBuilder.JointDofConfig
        j = sb.add_joint_d6(-1, nut, linear_axes=[J(axis=(0.0, 0.0, 1.0))], angular_axes=[J(axis=(0.0, 0.0, 1.0))],
                            parent_xform=X0, child_xform=wp.transform_identity(), label="nut_axis")
        sb.add_articulation([j])
        geo.update(name=name, nverts_bolt=int(len(bv)), nverts_nut=int(len(nv)))
        return sb, geo

    def build(self, nworld: int, device: str, config: Config) -> Built:
        _require_cuda_for_hydro(config, device)
        notes: dict[str, Any] = {}
        names = list(config.params.get("variants", self.VARIANTS))
        built = [self._variant(nm, config, device, notes) for nm in names[:max(1, min(len(names), nworld))]]
        variants, geos = [v for v, _ in built], [g for _, g in built]
        main = newton.ModelBuilder()
        _register(config, main)
        var, sub, ids = _replicate_blocks(main, variants, nworld, [[0]] * len(variants))
        model = main.finalize(device=device)
        hydro = None
        if _backend(config) == "hydro":
            from alloy._vendor.newton.geometry import HydroelasticSDF

            hydro = HydroelasticSDF.Config(mc_edge_clamp_min=0.0)  # example_nut_bolt_hydro.py: clamp off on helices
        cpw = int(config.params.get("contacts_per_world", 1024))
        pipe, contacts = make_pipeline(model, cpw, nworld, hydro=hydro, **_tri_pairs_kw(config, nworld))
        solver, snotes = make_solver(config, model, cpw, device)
        notes.update(snotes)
        s0, s1, control = _states(model)
        h = config.dt / config.substeps
        izz = model.body_inertia.numpy()[ids[:, 0]][:, 2, 2].astype(np.float64)
        kp = 0.5 * izz / h  # discrete PI with a double pole at 0.75 per substep (critically damped, stable)
        ki = izz / (16.0 * h * h)
        ntrace = min(nworld, len(variants))
        every = max(1, int(round(0.005 / h)))
        trace_worlds = np.array([np.where(var == g)[0][0] for g in range(ntrace)])  # first world of each variant
        ex = self.base_extra(device, nworld, ntrace, _nrec(self.horizon_s, h, every), every,
                             ids[trace_worlds, 0], trace_worlds)
        ex.update(notes=notes, cfg=config, nut=wp.array(ids[:, 0], dtype=int, device=device), ids=ids, var=var,
                  izz=izz,
                  geos=geos, gravity=wp.vec3(*_gravity(model)), kp=wp.array(kp, dtype=float, device=device),
                  ki=wp.array(ki, dtype=float, device=device), integ=wp.zeros(nworld, dtype=float, device=device),
                  tau=wp.zeros(nworld, dtype=float, device=device),
                  acc=wp.zeros((nworld, 7), dtype=float, device=device))
        return Built(model, solver, s0, s1, control, contacts, pipe, nworld=nworld, extra=ex)

    def drive(self, b: Built, h: float) -> None:
        ex = b.extra
        wp.launch(_p02_drive, dim=b.nworld, inputs=[
            ex["step"], h, b.state_0.body_qd, b.model.body_mass, ex["nut"], ex["gravity"], self.F_AX, self.OMEGA,
            self.T_RAMP, ex["kp"], ex["ki"], ex["integ"], b.state_0.body_f, ex["tau"]])

    def post(self, b: Built, h: float) -> None:
        ex = b.extra
        wp.launch(_p02_post, dim=b.nworld, inputs=[ex["step"], h, b.state_0.body_q, b.state_0.body_qd, ex["nut"],
                                                   self.OMEGA, self.T_MEAS, ex["tau"], ex["acc"], ex["aux"]])
        super().post(b, h)  # trace rows carry aux = (tau_dir, theta, omega_z, phase)

    def finish(self, b: Built, rec: dict) -> None:
        self.finish_common(b, rec)
        ex, met = b.extra, rec["metrics"]
        acc = ex["acc"].numpy().astype(np.float64)
        z_end = b.state_0.body_q.numpy()[ex["ids"][:, 0], 2].astype(np.float64)
        tr = _trace_rows(b)
        row_of = {int(w): r for r, w in enumerate(ex["trace_world_np"])}
        per, adv_errs, tq_errs, cross = [], [], [], []
        for w in range(b.nworld):
            g = ex["geos"][ex["var"][w]]
            p, d = g["p"], g["d"]
            d2 = d - 0.649519 * p
            T_law = thread_torque_law(self.F_AX, d2, p, self.MU)
            row = {"world": w, "variant": g["name"], "p_mm": p * 1e3, "d2_mm": d2 * 1e3, "torque_law_nm": T_law}
            if acc[w, 2] > 0.0:
                dth = acc[w, 0] - acc[w, 3]
                adv = 2.0 * math.pi * (z_end[w] - acc[w, 4]) / dth if abs(dth) > 1e-6 else float("nan")
                if w in row_of:  # least-squares slope dz/dtheta over the traced window
                    r = row_of[w]
                    rows = tr[r][(tr[r, :, 0] >= 0) & (tr[r, :, 17] > 0.5)]
                    if len(rows) > 10 and np.ptp(rows[:, 15]) > 1e-3:
                        adv = 2.0 * math.pi * float(np.polyfit(rows[:, 15], rows[:, 3], 1)[0])
                # mean servo torque minus the rotor/nut inertia term I (w_end - w_start) / window
                w_end = float(b.state_0.body_qd.numpy()[ex["ids"][w, 0], 5])
                w_start = acc[w, 6]
                T = acc[w, 1] / acc[w, 2] - ex["izz"][w] * (w_end - w_start) * np.sign(self.OMEGA) / acc[w, 2]
                row.update(revs=float(abs(dth) / (2 * math.pi)), advance_mm=adv * 1e3, torque_nm=T,
                           speed_err_max=float(acc[w, 5]))
                row["advance_err_pct"] = abs(adv - p) / p * 100.0
                row["torque_err_pct"] = abs(T - T_law) / T_law * 100.0
                row["cross_thread"] = bool(not np.isfinite(adv) or row["advance_err_pct"] > 25.0
                                           or acc[w, 5] > 0.5)
                adv_errs.append(row["advance_err_pct"])
                tq_errs.append(row["torque_err_pct"])
                cross.append(row["cross_thread"])
            per.append(row)
        met.update(per_condition=per[: 3 * len(ex["geos"])], mu=self.MU, f_axial_n=self.F_AX, omega=self.OMEGA,
                   rotor_inertia=self.ROTOR_I,
                   conditions_covered=bool(len(ex["geos"]) == len(self.VARIANTS)),
                   variants=[{k: v for k, v in g.items()} for g in ex["geos"]],
                   advance_err_pct=float(np.nanmax(adv_errs)) if adv_errs else None,
                   torque_err_pct=float(np.nanmax(tq_errs)) if tq_errs else None,
                   cross_thread=bool(any(cross)) if cross else None)
        rec["accuracy_error"] = met["advance_err_pct"]
        pen = rec.get("penetration_p99_mm")
        met["law_pass"] = bool(adv_errs and met["advance_err_pct"] <= 0.5 and met["torque_err_pct"] <= 5.0
                               and not met["cross_thread"] and (pen is None or pen <= 0.02))
        if not adv_errs:
            met["law_pass"] = None
            rec["notes"] += f"horizon ended before the measurement window (t >= {self.T_MEAS} s); "


# =============================================================================================================
# P03 RJ45 plug insertion and pull-out (latch as an equivalent cantilever)
# =============================================================================================================
P03_ACC = 11  # acc columns, see _p03_post


@wp.kernel
def _p03_drive(
    step: wp.array(dtype=int),
    h: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    plug: wp.array(dtype=int),
    latch: wp.array(dtype=int),
    p0: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    speed: float,
    travel: float,
    t_acc: float,
    t_dwell: float,
    f_pull: float,
    t_ramp: float,
    k: float,
    c: float,
    body_f: wp.array(dtype=wp.spatial_vector),
    acc: wp.array2d(dtype=float),
    aux: wp.array(dtype=wp.vec4),
):
    """Insertion: stiff spring-damper drive to a target that accelerates uniformly to `speed` over t_acc, moves
    +y for `travel` in total, then holds for t_dwell. Pull-out: y drive replaced by a pure force -f_pull eased in
    over t_ramp (x, z springs stay). Gravity cancelled on plug and latch (as the example does). Accumulates the
    drive work and the peak insertion force (after t_acc, so the start-up inertia force is excluded)."""
    w = wp.tid()
    b = plug[w]
    t = float(step[0]) * h
    p = wp.transform_get_translation(body_q[b])
    v = wp.spatial_top(body_qd[b])
    tgt = p0[w]
    t_ins = travel / speed + 0.5 * t_acc
    s = speed * (t - 0.5 * t_acc)
    if t < t_acc:
        s = 0.5 * speed * t * t / t_acc
    yt = tgt[1] + wp.min(s, travel)
    vt = wp.where(t < t_acc, speed * t / t_acc, wp.where(s < travel, speed, 0.0))
    f = wp.vec3(k * (tgt[0] - p[0]) - c * v[0], k * (yt - p[1]) + c * (vt - v[1]), k * (tgt[2] - p[2]) - c * v[2])
    phase = 0.0
    if t >= t_ins + t_dwell:
        phase = 1.0
        f = wp.vec3(f[0], -f_pull * _ease((t - t_ins - t_dwell) / t_ramp), f[2])
    else:
        acc[w, 0] = acc[w, 0] + wp.dot(f, v) * h
        if t >= t_acc:
            acc[w, 5] = wp.max(acc[w, 5], f[1])
    body_f[b] = body_f[b] + wp.spatial_vector(f - body_mass[b] * gravity, wp.vec3(0.0))
    l = latch[w]
    body_f[l] = body_f[l] + wp.spatial_vector(-body_mass[l] * gravity, wp.vec3(0.0))
    aux[w] = wp.vec4(f[1], acc[w, 2], acc[w, 0], phase)


@wp.kernel
def _p03_post(
    step: wp.array(dtype=int),
    h: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_mass: wp.array(dtype=float),
    plug: wp.array(dtype=int),
    latch: wp.array(dtype=int),
    p0: wp.array(dtype=wp.vec3),
    hinge_axis: wp.vec3,
    kd_latch: float,
    t_pull: float,
    acc: wp.array2d(dtype=float),
):
    """acc: 0 drive work W, 1 latch damper loss E_d, 2 phi_min (inward deflection, rad), 3 W at phi_min,
    4 E_d at phi_min, 5 peak insertion force, 6 y at pull start (1e9 = unset), 7 max retreat during pull,
    8 max lateral drift (x, z) of the plug, 9 plug+latch kinetic energy at phi_min, 10 phi_max (outward)."""
    w = wp.tid()
    t = float(step[0] + 1) * h
    b = plug[w]
    l = latch[w]
    qp = wp.transform_get_rotation(body_q[b])
    ql = wp.transform_get_rotation(body_q[l])
    qr = wp.quat_inverse(qp) * ql
    phi = 2.0 * wp.atan2(wp.dot(wp.vec3(qr[0], qr[1], qr[2]), hinge_axis), qr[3])
    ax_w = wp.quat_rotate(qp, hinge_axis)
    wrel = wp.dot(wp.spatial_bottom(body_qd[l]) - wp.spatial_bottom(body_qd[b]), ax_w)
    acc[w, 1] = acc[w, 1] + kd_latch * wrel * wrel * h
    if phi < acc[w, 2]:
        vb = wp.spatial_top(body_qd[b])
        vl = wp.spatial_top(body_qd[l])
        acc[w, 2] = phi
        acc[w, 3] = acc[w, 0]
        acc[w, 4] = acc[w, 1]
        acc[w, 9] = 0.5 * body_mass[b] * wp.dot(vb, vb) + 0.5 * body_mass[l] * wp.dot(vl, vl)
    acc[w, 10] = wp.max(acc[w, 10], phi)
    p = wp.transform_get_translation(body_q[b])
    tgt = p0[w]
    acc[w, 8] = wp.max(acc[w, 8], wp.max(wp.abs(p[0] - tgt[0]), wp.abs(p[2] - tgt[2])))
    if t >= t_pull:
        if acc[w, 6] > 1.0e8:
            acc[w, 6] = p[1]
        acc[w, 7] = wp.max(acc[w, 7], acc[w, 6] - p[1])


class RJ45Plug(_PrecisionScenario):
    """P03: the example's RJ45 socket (static), plug (on a D6 joint: translation free, rotation locked) and latch
    (revolute to the plug, return spring k_theta = 0.15 N m/rad, damping 0.03, limits -0.2/+0.3 rad), all SDF
    meshes from newton/examples/assets/rj45_plug.usd with the example's contact material (mu 0, ke 1e5, kd 0,
    gap 2 mm, density 1e6). The cable is left out (it does not touch the latch). The gizmo is replaced by a
    scripted drive: the plug starts 25 mm out and is driven +y at 10 mm/s through a stiff holder (k = 5e4 N/m,
    critical damping, velocity feed-forward, 0.2 s uniform start-up) to the authored pose, dwells 0.3 s, then the
    y drive becomes a pure 5 N pull eased in over 0.5 s and held.

    Laws: the latch is a rigid lever on a torsional spring, so its equivalent cantilever has F = 3 EI delta / L^3
    with EI = k_theta L / 3 (F = k_theta delta / L^2, L = hinge-to-tip length from the latch mesh). With mu = 0 the
    contact does no dissipative work, so by virtual work the drive work up to the peak deflection equals the
    latch energy: W - E_damper - KE = 1/2 k_meas phi^2. latch_force_err_pct = |k_meas - k_theta| / k_theta (equal
    to the force error of F = 3 EI delta / L^3 at the peak deflection). locked = the plug retreats < 1 mm under
    the 5 N pull. Penetration p99 <= 0.05 mm. accuracy_error = latch_force_err_pct.
    """

    id = "P03_rj45_plug"
    title = "RJ45 plug scripted insertion at 10 mm/s and 5 N pull-out: latch cantilever law, lock, penetration"
    horizon_s = 4.0
    accuracy_unit = "%"
    accuracy_ref = ("latch F = 3 EI delta / L^3 (EI = k_theta L / 3) via virtual work, within 10 %; "
                    "locked at 5 N; p99 <= 0.05 mm")
    SPEED, TRAVEL, T_ACC, T_DWELL, F_PULL, T_RAMP = 0.010, 0.025, 0.2, 0.3, 5.0, 0.5
    K_DRIVE, LOCK_TOL = 5.0e4, 0.001
    LATCH_KE, LATCH_KD, LATCH_LO, LATCH_HI, LATCH_LIMIT_KD = 0.15, 0.03, -0.2, 0.3, 1.0e-4
    HINGE_AXIS = (-1.0, 0.0, 0.0)
    _c = {"sdf_res": 128}
    configs = (
        Config("vbd", "vbd", dt=1 / 60, substeps=6, params={**_c, "iterations": 12, "contact_buffer": 256}),
        Config("sdf_mujoco", "mujoco", dt=1 / 60, substeps=6, params={**_c}),
        Config("sdf_kamino", "kamino", dt=1 / 60, substeps=6, params={**_c}),
        Config("sdf_xpbd", "xpbd", dt=1 / 60, substeps=6, params={**_c, "iterations": 10}),
    )

    def _load(self, device: str, res: int, gap: float, notes: dict) -> dict:
        import alloy._vendor.newton.examples; from alloy._vendor import newton
        import alloy._vendor.newton.usd; from alloy._vendor import newton
        from pxr import Usd

        stage = Usd.Stage.Open(newton.examples.get_asset("rj45_plug.usd"))
        out = {}
        for key, path in (("socket", "/World/Socket"), ("plug", "/World/Plug"), ("latch", "/World/Latch")):
            prim = stage.GetPrimAtPath(path)
            um = newton.usd.get_mesh(prim, load_normals=True)
            pos = np.array(wp.transform_get_translation(newton.usd.get_transform(prim, local=False)), np.float64)
            v = np.asarray(um.vertices, np.float32)
            nrm = np.asarray(um.normals, np.float32) if um.normals is not None else None
            mesh = newton.Mesh(v, np.asarray(um.indices, np.int32), normals=nrm)
            notes["sdf_backend"] = _attach_sdf(mesh, device, res, (-2.0 * gap, 2.0 * gap), gap)
            out[key] = (mesh, pos, v)
        return out

    def build(self, nworld: int, device: str, config: Config) -> Built:
        notes: dict[str, Any] = {"cable": "omitted (does not touch the latch)"}
        sc = newton.ModelBuilder.ShapeConfig(mu=0.0, ke=1.0e5, kd=0.0, gap=0.002, density=1.0e6, mu_torsional=0.0,
                                             mu_rolling=0.0)
        parts = self._load(device, int(config.params.get("sdf_res", 128)), sc.gap, notes)
        (sm, sp, _), (pm, pp, _), (lm, lp, lv) = parts["socket"], parts["plug"], parts["latch"]
        sb = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        sb.add_shape_mesh(-1, mesh=sm, xform=wp.transform(wp.vec3(*sp), wp.quat_identity()), cfg=sc, label="socket")
        start = pp - np.array([0.0, self.TRAVEL, 0.0])
        plug = sb.add_link(xform=wp.transform(wp.vec3(*start), wp.quat_identity()), label="plug")
        sb.add_shape_mesh(plug, mesh=pm, cfg=sc)
        latch = sb.add_link(xform=wp.transform(wp.vec3(*(lp - pp + start)), wp.quat_identity()), label="latch")
        sb.add_shape_mesh(latch, mesh=lm, cfg=sc)
        J = newton.ModelBuilder.JointDofConfig
        d6 = sb.add_joint_d6(-1, plug, linear_axes=[J(axis=(1.0, 0.0, 0.0)), J(axis=(0.0, 1.0, 0.0)),
                                                    J(axis=(0.0, 0.0, 1.0))], angular_axes=None,
                             parent_xform=wp.transform(wp.vec3(*start), wp.quat_identity()),
                             child_xform=wp.transform_identity())
        rev = sb.add_joint_revolute(plug, latch, axis=self.HINGE_AXIS,
                                    parent_xform=wp.transform(wp.vec3(*(lp - pp)), wp.quat_identity()),
                                    child_xform=wp.transform_identity(), target_ke=self.LATCH_KE,
                                    target_kd=self.LATCH_KD, limit_lower=self.LATCH_LO, limit_upper=self.LATCH_HI,
                                    limit_kd=self.LATCH_LIMIT_KD, collision_filter_parent=True)
        sb.add_articulation([d6, rev])
        main = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _register(config, main)
        _, _, ids = _replicate_blocks(main, [sb], nworld, [[plug, latch]])
        if config.solver == "vbd":
            main.color()
        model = main.finalize(device=device)
        cpw = int(config.params.get("contacts_per_world", 1024))
        pipe, contacts = make_pipeline(model, cpw, nworld)
        solver, snotes = make_solver(config, model, cpw, device)
        notes.update(snotes)
        s0, s1, control = _states(model)
        h = config.dt / config.substeps
        mass = model.body_mass.numpy()
        m_plug = float(mass[ids[0, 0]])
        acc0 = np.zeros((nworld, P03_ACC))
        acc0[:, 6] = 1.0e9
        L = float(np.max(np.hypot(lv[:, 1], lv[:, 2])))  # hinge on the latch origin, axis along x
        ntrace = min(nworld, 1)
        every = max(1, int(round(0.005 / h)))
        ex = self.base_extra(device, nworld, ntrace, _nrec(self.horizon_s, h, every), every, ids[:, 0])
        ex.update(notes=notes, cfg=config, ids=ids, plug=wp.array(ids[:, 0], dtype=int, device=device),
                  latch=wp.array(ids[:, 1], dtype=int, device=device),
                  p0=wp.array(np.tile(start, (nworld, 1)), dtype=wp.vec3, device=device),
                  gravity=wp.vec3(*_gravity(model)), c_drive=2.0 * math.sqrt(self.K_DRIVE * m_plug),
                  acc=wp.array(acc0, dtype=float, device=device), latch_L=L, m_plug=m_plug,
                  m_latch=float(mass[ids[0, 1]]))
        return Built(model, solver, s0, s1, control, contacts, pipe, nworld=nworld, extra=ex)

    def t_pull(self) -> float:
        return self.TRAVEL / self.SPEED + 0.5 * self.T_ACC + self.T_DWELL

    def drive(self, b: Built, h: float) -> None:
        ex = b.extra
        wp.launch(_p03_drive, dim=b.nworld, inputs=[
            ex["step"], h, b.state_0.body_q, b.state_0.body_qd, b.model.body_mass, ex["plug"], ex["latch"], ex["p0"],
            ex["gravity"], self.SPEED, self.TRAVEL, self.T_ACC, self.T_DWELL, self.F_PULL, self.T_RAMP, self.K_DRIVE,
            ex["c_drive"], b.state_0.body_f, ex["acc"], ex["aux"]])

    def post(self, b: Built, h: float) -> None:
        ex = b.extra
        wp.launch(_p03_post, dim=b.nworld, inputs=[
            ex["step"], h, b.state_0.body_q, b.state_0.body_qd, b.model.body_mass, ex["plug"], ex["latch"], ex["p0"],
            wp.vec3(*self.HINGE_AXIS), self.LATCH_KD, self.t_pull(), ex["acc"]])
        super().post(b, h)

    def finish(self, b: Built, rec: dict) -> None:
        self.finish_common(b, rec)
        ex, met = b.extra, rec["metrics"]
        acc = ex["acc"].numpy().astype(np.float64)
        L = ex["latch_L"]
        t_end = int(ex["step"].numpy()[0]) * (ex["cfg"].dt / ex["cfg"].substeps)
        y_end = b.state_0.body_q.numpy()[ex["ids"][:, 0], 1].astype(np.float64)
        y_seat = ex["p0"].numpy()[:, 1] + self.TRAVEL
        per, errs = [], []
        for w in range(b.nworld):
            phi = acc[w, 2]
            row = {"world": w, "phi_min_rad": phi, "phi_max_rad": acc[w, 10], "peak_insert_force_n": acc[w, 5],
                   "lateral_drift_mm": acc[w, 8] * 1e3, "seat_err_mm": float((y_seat[w] - y_end[w]) * 1e3)}
            if phi < -0.01:
                work = acc[w, 3] - acc[w, 4] - acc[w, 9]
                k_meas = 2.0 * work / (phi * phi)
                row.update(latch_work_mj=work * 1e3, k_meas=k_meas,
                           latch_force_pred_n=self.LATCH_KE * abs(phi) / L,
                           latch_force_meas_n=k_meas * abs(phi) / L,
                           latch_force_err_pct=abs(k_meas - self.LATCH_KE) / self.LATCH_KE * 100.0,
                           limit_hit=bool(phi <= self.LATCH_LO + 0.005))
                errs.append(row["latch_force_err_pct"])
            if t_end >= self.t_pull() + self.T_RAMP:
                row["retreat_mm"] = acc[w, 7] * 1e3
                row["locked"] = bool(acc[w, 7] < self.LOCK_TOL)
            per.append(row)
        locked = [r["locked"] for r in per if "locked" in r]
        met.update(per_condition=per[:16], latch_L_mm=L * 1e3, latch_EI_equiv=self.LATCH_KE * L / 3.0,
                   m_plug_kg=ex["m_plug"], m_latch_kg=ex["m_latch"],
                   peak_insert_force_n=float(acc[:, 5].max()), lateral_drift_mm=float(acc[:, 8].max() * 1e3),
                   latch_force_err_pct=float(max(errs)) if errs else None,
                   locked=bool(all(locked)) if locked else None, conditions_covered=True)
        vbd = b.solver if ex["cfg"].solver == "vbd" else None
        if vbd is not None and hasattr(vbd, "body_body_contact_counts"):
            met["vbd_contact_overflow"] = int(np.sum(vbd.body_body_contact_counts.numpy()
                                                     > vbd.body_body_contact_buffer_pre_alloc))
        rec["accuracy_error"] = met["latch_force_err_pct"]
        pen = rec.get("penetration_p99_mm")
        met["law_pass"] = bool(errs and max(errs) <= 10.0 and met["locked"] and (pen is None or pen <= 0.05))
        if not locked:
            met["law_pass"] = None  # the pull-out phase was not reached (short horizon)
        if not errs:
            rec["notes"] += "latch never deflected inward by 0.01 rad (horizon too short or no engagement); "


SCENARIOS: list[Scenario] = [PegHole(), NutBolt(), RJ45Plug(), HertzSphere(), ThinWallTunnel()]
