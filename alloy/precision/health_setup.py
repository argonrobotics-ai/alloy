# SPDX-License-Identifier: Apache-2.0
"""Host-side tables the HealthMonitor builds once per Model (SPEC §3.7, §4.2 health row, D3, D20).

Shape thickness for the "half the thinner part" PEN_FAIL rule, the weld/connect table with per-constraint limits
from MuJoCo solref/solimp, per-world item lists for the energy sums and the E_scale floor.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.precision.chunk_sums import chunk_count, make_chunks
from alloy.precision.digest import _coord_world, world_items
from alloy.precision.health_physics import EQ_CONNECT, EQ_WELD, EnergyModel, WeldTable, WorldList
from alloy.precision.world_groups import group_by_world, world_reduce, world_sum

log = logging.getLogger("alloy.precision")

JOINT_LIMIT_FLOOR_M = 1e-3   # §4.2: max(1e-3 m, 3x solref steady state)
JOINT_LIMIT_FLOOR_RAD = 1e-3
SOLREF_DEFAULT = (0.02, 1.0)  # MuJoCo and Newton defaults (newton/_src/solvers/mujoco/utils.py:39-44)
SOLIMP_DEFAULT = (0.9, 0.95, 0.001, 0.5, 2.0)
G = newton.GeoType


def shape_thickness(model: newton.Model) -> np.ndarray:
    """Smallest full extent of each shape [m] (inf for planes, height fields and unknown geometry).

    Primitives are computed per type in one pass; meshes go through ``_mesh_thickness`` once per distinct
    (source, scale), so replicated worlds cost one lookup per shape."""
    n = model.shape_count
    out = np.full(n, np.inf, dtype=np.float32)
    if n == 0:
        return out
    types = model.shape_type.numpy()
    scale = model.shape_scale.numpy()[:, :3].astype(np.float32)
    for gtypes, half_axes in _PRIMITIVE_AXES:
        sel = np.isin(types, [int(t) for t in gtypes])
        out[sel] = 2.0 * scale[sel][:, half_axes].min(axis=1)
    cache: dict[tuple[int, bytes], float] = {}
    for s in np.flatnonzero(np.isin(types, [int(G.MESH), int(G.CONVEX_MESH)])):
        source = model.shape_source[s] if s < len(model.shape_source) else None
        key = (id(source), scale[s].tobytes())
        if key not in cache:
            cache[key] = _mesh_thickness(scale[s], source)
        out[s] = cache[key]
    return out


# geometry types -> scale components whose smallest one is the half thickness
_PRIMITIVE_AXES = (((G.SPHERE, G.CAPSULE), [0]), ((G.BOX, G.ELLIPSOID), [0, 1, 2]), ((G.CYLINDER, G.CONE), [0, 1]))


def _mesh_thickness(scale: np.ndarray, source: object | None) -> float:
    if source is None or not hasattr(source, "vertices"):
        return math.inf
    pts = np.asarray(source.vertices, dtype=np.float64) * np.asarray(scale[:3], dtype=np.float64)
    return float((pts.max(axis=0) - pts.min(axis=0)).min()) if len(pts) else math.inf


def coord_worlds(model: newton.Model) -> tuple[wp.array, wp.array]:
    """World id per ``joint_q`` and per ``joint_qd`` entry."""
    dev = model.device
    if model.joint_count == 0:
        empty = wp.zeros(0, dtype=wp.int32, device=dev)
        return empty, empty
    jw = model.joint_world.numpy()
    q = _coord_world(model.joint_q_start.numpy(), jw)
    qd = _coord_world(model.joint_qd_start.numpy(), jw)
    return wp.array(q.astype(np.int32), device=dev), wp.array(qd.astype(np.int32), device=dev)


def unjudged_world(particle_world: wp.array, skip: np.ndarray) -> wp.array:
    """``particle_world`` with -1 at ``skip``: the soft-contact check leaves those particles' rows out (their solver,
    SPH or MPM, handles shapes itself and never reads Newton's soft contacts)."""
    w = particle_world.numpy().copy()
    w[np.asarray(skip, np.int64)] = -1
    return wp.array(w, dtype=wp.int32, device=particle_world.device)


def world_list(item_world: np.ndarray, world_count: int, device, include_global: bool = True) -> WorldList:
    items = world_items(item_world, world_count, device, include_global)
    lst = WorldList()
    lst.begin, lst.end, lst.items = items.begin, items.end, items.items
    return lst


def energy_model(model: newton.Model) -> EnergyModel:
    W, dev = int(model.world_count), model.device
    em = EnergyModel()
    body_world = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
    part_world = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32)
    # global (-1) items are shared by all worlds and left out of per-world energies
    em.bodies = world_list(body_world, W, dev, include_global=False)
    em.particles = world_list(part_world, W, dev, include_global=False)
    em.body_mass = _or_empty(model.body_mass, wp.float32, dev)
    em.body_com = _or_empty(model.body_com, wp.vec3, dev)
    em.body_inertia = _or_empty(model.body_inertia, wp.mat33, dev)
    em.particle_mass = _or_empty(model.particle_mass, wp.float32, dev)
    em.particle_flags = _or_empty(model.particle_flags, wp.int32, dev)
    em.gravity = model.gravity
    em.body_active = wp.ones(max(model.body_count, 1), dtype=wp.int8, device=dev)
    _spring_terms(model, em)
    _energy_chunks(em, dev)
    return em


def _energy_chunks(em: EnergyModel, dev) -> None:
    """Chunks of each world's bodies-then-particles sequence and their partial-sum scratch (``chunk_sums``)."""
    counts = sum(lst.end.numpy().astype(np.int64) - lst.begin.numpy() for lst in (em.bodies, em.particles))
    em.chunks = make_chunks(counts, dev)
    em.partial = wp.zeros(max(chunk_count(em.chunks), 1), dtype=wp.float64, device=dev)


def _attr(model: newton.Model, name: str) -> np.ndarray | None:
    arr = getattr(getattr(model, "mujoco", None), name, None)
    return arr.numpy() if arr is not None and getattr(arr, "size", 0) else None


def _core(model: newton.Model, name: str) -> np.ndarray | None:
    arr = getattr(model, name, None)
    return arr.numpy() if arr is not None and getattr(arr, "size", 0) else None


def _springs_on(model: newton.Model) -> bool:
    """MuJoCo's spring flag (``mujoco:flag_spring``; absent = on): off means no passive springs act."""
    flag = _attr(model, "flag_spring")
    return flag is None or bool(flag.reshape(-1)[0])


def _joint_springs(model: newton.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(world, q index, k, q_rest) of every 1-dof joint dof with a passive stiffness > 0. Newton 1.8 keeps
    the springs in the core arrays: ``Model.joint_stiffness`` per dof and ``Model.joint_rest_q`` per joint coordinate,
    MuJoCo's springref - ref: the rest in the coordinates the energy reads ``joint_q`` in (the legacy
    ``mujoco.dof_springref`` raises for a model with a free, ball, distance or rod joint). Ball and free joints with
    stiffness are left out and logged: their spring is not 0.5 k (q - q_ref)^2. ROD joints (a VBD rod's) are left out
    quietly: they are not MuJoCo springs."""
    k, rest = _core(model, "joint_stiffness"), _core(model, "joint_rest_q")
    empty = (np.zeros(0, np.int32),) * 2 + (np.zeros(0, np.float32),) * 2
    if k is None or not model.joint_count or not np.any(k > 0):
        return empty
    rest = rest if rest is not None else np.zeros(model.joint_coord_count, np.float32)
    jtype, qs, qds = model.joint_type.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    jworld = model.joint_world.numpy()
    one_dof = (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC))
    quiet = {int(newton.JointType.ROD)} if hasattr(newton.JointType, "ROD") else set()
    rows = []
    for j in range(model.joint_count):
        for d in range(int(qds[j]), int(qds[j + 1])):
            if k[d] <= 0 or int(jtype[j]) in quiet:
                continue
            if int(jtype[j]) not in one_dof:
                log.warning("health energy: dof %d (joint %d, type %d) has passive stiffness but is not a 1-dof joint; "
                            "its spring energy is left out of D20", d, j, int(jtype[j]))
                continue
            q = int(qs[j]) + d - int(qds[j])
            rows.append((int(jworld[j]), q, float(k[d]), float(rest[q])))
    if not rows:
        return empty
    w, q, kk, r = map(np.array, zip(*rows))
    return w.astype(np.int32), q.astype(np.int32), kk.astype(np.float32), r.astype(np.float32)


def _fixed_tendons(model: newton.Model) -> dict[str, np.ndarray] | None:
    """Fixed tendons with stiffness > 0 (L = sum coef * q over 1-dof joints) and their spring deadband."""
    k = _attr(model, "tendon_stiffness")
    if k is None or not np.any(k > 0):
        return None
    ttype, world = _attr(model, "tendon_type"), _attr(model, "tendon_world")
    adr, num = _attr(model, "tendon_joint_adr"), _attr(model, "tendon_joint_num")
    joint, coef, spring = _attr(model, "tendon_joint"), _attr(model, "tendon_coef"), _attr(model, "tendon_springlength")
    if any(a is None for a in (adr, num, joint, coef, spring)):
        log.warning("health energy: tendon springs present but their joint coefficients are missing; left out of D20")
        return None
    qs = model.joint_q_start.numpy()
    keep = [t for t in range(k.shape[0]) if k[t] > 0 and (ttype is None or int(ttype[t]) == 0)]
    if len(keep) < int(np.count_nonzero(k > 0)):
        log.warning("health energy: %d spatial tendon spring(s) left out of D20 (only fixed tendons are summed)",
                    int(np.count_nonzero(k > 0)) - len(keep))
    sl = np.asarray(spring, np.float64).reshape(k.shape[0], -1)
    return {"world": (world.astype(np.int32) if world is not None else np.zeros(k.shape[0], np.int32))[keep],
            "adr": adr.astype(np.int32)[keep], "num": num.astype(np.int32)[keep], "k": k.astype(np.float32)[keep],
            "lo": sl[keep, 0].astype(np.float32), "hi": sl[keep, -1].astype(np.float32),
            "q": qs[joint.astype(np.int64)].astype(np.int32), "coef": coef.astype(np.float32)}


def _spring_terms(model: newton.Model, em: EnergyModel) -> None:
    W, dev = int(model.world_count), model.device
    on = _springs_on(model)
    if on:
        w, q, k, ref = _joint_springs(model)
    else:
        w, q, k, ref = np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.float32)
    em.springs = world_list(w, W, dev, include_global=False)
    em.spring_q = wp.array(q if q.size else np.zeros(1, np.int32), dtype=wp.int32, device=dev)
    em.spring_k = wp.array(k if k.size else np.zeros(1, np.float32), dtype=wp.float32, device=dev)
    em.spring_ref = wp.array(ref if ref.size else np.zeros(1, np.float32), dtype=wp.float32, device=dev)
    ten = _fixed_tendons(model) if on else None
    if ten is None:
        ten = {"world": np.zeros(0, np.int32), "adr": np.zeros(0, np.int32), "num": np.zeros(0, np.int32),
               "k": np.zeros(0, np.float32), "lo": np.zeros(0, np.float32), "hi": np.zeros(0, np.float32),
               "q": np.zeros(0, np.int32), "coef": np.zeros(0, np.float32)}
    em.tendons = world_list(ten["world"], W, dev, include_global=False)
    for name, dtype in (("adr", wp.int32), ("num", wp.int32), ("k", wp.float32), ("lo", wp.float32),
                        ("hi", wp.float32), ("q", wp.int32), ("coef", wp.float32)):
        a = ten[name]
        setattr(em, f"tendon_{name}", wp.array(a if a.size else np.zeros(1, a.dtype), dtype=dtype, device=dev))


def _or_empty(arr: wp.array | None, dtype, device) -> wp.array:
    return arr if arr is not None else wp.zeros(0, dtype=dtype, device=device)


def energy_scale_floor(model: newton.Model) -> np.ndarray:
    """m_tot·|g|·L_char per world, L_char = largest extent of the world's initial bodies, particles and shapes.

    Worlds without bodies or particles get 0. Linear in the item count (grouped by world, no per-world masks)."""
    W = int(model.world_count)
    out = np.zeros(W, dtype=np.float64)
    lo, hi, masses = _world_bounds_and_masses(model)
    has = np.isfinite(lo[:, 0])
    if not has.any():
        return out
    extent = (hi - lo).max(axis=1).astype(np.float64) + 2.0 * _shape_pad(model)
    out[has] = masses[has] * _gravity_norm(model, W)[has] * extent[has]
    return out


def _gravity_norm(model: newton.Model, W: int) -> np.ndarray:
    """|g| per world; worlds past the gravity rows use row 0."""
    g = np.asarray(model.gravity.numpy(), dtype=np.float64).reshape(-1, 3)
    norm = np.full(W, float(np.linalg.norm(g[0])) if len(g) else 0.0)
    k = min(W, len(g))
    norm[:k] = np.linalg.norm(g[:k], axis=1)
    return norm


def _world_bounds_and_masses(model: newton.Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-world min and max corner of the initial body and particle positions ([W, 3] float32, +inf/-inf for
    worlds with none) and total mass ([W] float64)."""
    W = int(model.world_count)
    lo = np.full((W, 3), np.inf, dtype=np.float32)
    hi = np.full((W, 3), -np.inf, dtype=np.float32)
    masses = np.zeros(W, dtype=np.float64)
    sources = []
    if model.body_count:
        sources.append((model.body_q.numpy()[:, :3], model.body_world.numpy(), model.body_mass.numpy()))
    if model.particle_count:
        sources.append((model.particle_q.numpy(), model.particle_world.numpy(), model.particle_mass.numpy()))
    for p, world, m in sources:
        groups = group_by_world(world, W)
        lo = np.minimum(lo, world_reduce(groups, p.astype(np.float32), np.minimum, np.inf))
        hi = np.maximum(hi, world_reduce(groups, p.astype(np.float32), np.maximum, -np.inf))
        masses += world_sum(world, m, W)
    return lo, hi, masses


def _shape_pad(model: newton.Model) -> np.ndarray:
    """Largest collision radius of each world's dynamic shapes (planes report a huge radius and are skipped)."""
    W = int(model.world_count)
    if not model.shape_count or model.shape_collision_radius is None:
        return np.zeros(W, dtype=np.float64)
    r, sw, sb = model.shape_collision_radius.numpy(), model.shape_world.numpy(), model.shape_body.numpy()
    keep = np.isfinite(r) & (sb >= 0)
    pad = world_reduce(group_by_world(np.where(keep, sw, -1), W), r, np.maximum, -np.inf).astype(np.float64)
    pad[~np.isfinite(pad)] = 0.0
    return pad


def steady_state_violation(solref: tuple[float, float], solimp: tuple[float, ...], accel: float) -> float:
    """MuJoCo soft-constraint sag r = (1 - d0)·a / (d0·k), k = 1/(dmax²·τ²·ζ²) (direct form: k = -solref[0]).

    Uses d0 (the impedance at zero violation), the smallest d over the sigmoid, so r is an upper bound. At the
    defaults (0.02, 1) / (0.9, 0.95) under 9.81 m/s²: k = 2770 s⁻², r = 0.39 mm."""
    d0, dmax = float(solimp[0]), float(solimp[1])
    tau, zeta = float(solref[0]), float(solref[1])
    k = -tau if tau < 0.0 else 1.0 / (dmax * dmax * tau * tau * zeta * zeta)
    if k <= 0.0 or d0 <= 0.0:
        return math.inf
    return (1.0 - d0) * accel / (d0 * k)


def _transform_np(x: np.ndarray) -> wp.transform:
    return wp.transform(wp.vec3(*x[:3]), wp.quat(*x[3:7]))


def _pose(body_q: np.ndarray, b: int) -> wp.transform:
    return _transform_np(body_q[b]) if b >= 0 else wp.transform_identity()


@dataclass(frozen=True)
class _BodyHost:
    """Host copies of the body tables the weld limits read (one device copy per table, not per row)."""

    q: np.ndarray
    mass: np.ndarray
    com: np.ndarray
    inertia: np.ndarray
    g: float  # |gravity| of world 0

    @classmethod
    def of(cls, model: newton.Model) -> _BodyHost:
        if not model.body_count:
            return cls(np.zeros((0, 7), np.float32), np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3)),
                       float(np.linalg.norm(model.gravity.numpy()[0])))
        return cls(model.body_q.numpy(), model.body_mass.numpy(), model.body_com.numpy(),
                   model.body_inertia.numpy(), float(np.linalg.norm(model.gravity.numpy()[0])))


@dataclass(frozen=True)
class WeldLimits:
    joint_limit_m: float | None
    joint_limit_rad: float | None
    design_force_n: float = 0.0


def weld_table(model: newton.Model, limits: WeldLimits) -> WeldTable | None:
    """Connect/weld rows of ``model.mujoco.equality_constraint_*`` with per-row violation limits; None if none."""
    mj = getattr(model, "mujoco", None)
    kinds = getattr(mj, "equality_constraint_type", None)
    if kinds is None or kinds.shape[0] == 0:
        return None
    rows = _weld_rows(model, mj, limits)
    dev = model.device
    t = WeldTable()
    t.kind = wp.array(rows["kind"], dtype=wp.int32, device=dev)
    t.body1 = wp.array(rows["b1"], dtype=wp.int32, device=dev)
    t.body2 = wp.array(rows["b2"], dtype=wp.int32, device=dev)
    t.anchor1 = wp.array(rows["a1"], dtype=wp.vec3, device=dev)
    t.anchor2 = wp.array(rows["a2"], dtype=wp.vec3, device=dev)
    t.rel = wp.array(rows["rel"], dtype=wp.transform, device=dev)
    t.lim_m = wp.array(rows["lim_m"], dtype=wp.float32, device=dev)
    t.lim_rad = wp.array(rows["lim_rad"], dtype=wp.float32, device=dev)
    t.world = wp.array(rows["world"], dtype=wp.int32, device=dev)
    return t


def _weld_rows(model: newton.Model, mj, limits: WeldLimits) -> dict[str, np.ndarray]:
    kinds = mj.equality_constraint_type.numpy()
    n = len(kinds)
    b1s, b2s = mj.equality_constraint_body1.numpy(), mj.equality_constraint_body2.numpy()
    anchors, relposes = mj.equality_constraint_anchor.numpy(), mj.equality_constraint_relpose.numpy()
    solref = mj.eq_solref.numpy() if hasattr(mj, "eq_solref") else np.tile(SOLREF_DEFAULT, (n, 1))
    solimp = mj.eq_solimp.numpy() if hasattr(mj, "eq_solimp") else np.tile(SOLIMP_DEFAULT, (n, 1))
    host = _BodyHost.of(model)
    rows = {k: [] for k in ("kind", "b1", "b2", "a1", "a2", "rel", "lim_m", "lim_rad")}
    for e in range(n):
        row = _weld_row(int(kinds[e]), int(b1s[e]), int(b2s[e]), anchors[e], relposes[e], host.q)
        lim_m, lim_rad = _weld_limits(host, row["b1"], row["b2"], row["a2"], solref[e], solimp[e], limits)
        row.update(lim_m=lim_m, lim_rad=lim_rad)
        for k in rows:
            rows[k].append(row[k])
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["world"] = mj.equality_constraint_world.numpy().astype(np.int32)
    return out


def _weld_row(kind: int, b1: int, b2: int, anchor: np.ndarray, relpose: np.ndarray, body_q: np.ndarray) -> dict:
    """Targets from the initial pose where MuJoCo infers them (zero-quaternion relpose sentinel, equality.py)."""
    import mujoco

    x1, x2 = _pose(body_q, b1), _pose(body_q, b2)
    a1 = wp.vec3(*anchor[:3])
    row = {"kind": 0, "b1": b1, "b2": b2, "a1": a1, "a2": wp.vec3(0.0, 0.0, 0.0), "rel": wp.transform_identity()}
    if kind == int(mujoco.mjtEq.mjEQ_CONNECT):
        row.update(kind=int(EQ_CONNECT), a2=wp.transform_point(wp.transform_inverse(x2), wp.transform_point(x1, a1)))
    elif kind == int(mujoco.mjtEq.mjEQ_WELD):
        sentinel = not np.any(relpose[3:7])
        rel = wp.transform_inverse(x1) * x2 if sentinel else _transform_np(relpose)
        row.update(kind=int(EQ_WELD), a2=a1, rel=rel)
    return row


def _weld_limits(host: _BodyHost, b1: int, b2: int, a2, solref, solimp, limits: WeldLimits) -> tuple[float, float]:
    """max(floor, 3x steady-state sag) for position and angle, load = |g| + design force / mass (D3, §4.2)."""
    body = b2 if b2 >= 0 else b1
    mass = float(host.mass[body]) if body >= 0 else 0.0
    accel = host.g + (limits.design_force_n / mass if mass > 0.0 else 0.0)
    sag = steady_state_violation(tuple(solref), tuple(solimp), accel)
    lim_m = limits.joint_limit_m if limits.joint_limit_m is not None else max(JOINT_LIMIT_FLOOR_M, 3.0 * sag)
    lever = float(np.linalg.norm(np.asarray(host.com[body]) - np.asarray(a2))) if body >= 0 else 0.0
    i_min = float(np.linalg.eigvalsh(host.inertia[body]).min()) if body >= 0 else 0.0
    alpha = mass * accel * lever / (i_min + mass * lever * lever) if mass > 0.0 and (i_min + lever) > 0.0 else 0.0
    sag_rad = steady_state_violation(tuple(solref), tuple(solimp), alpha)
    lim_rad = limits.joint_limit_rad if limits.joint_limit_rad is not None else max(JOINT_LIMIT_FLOOR_RAD,
                                                                                    3.0 * sag_rad)
    return lim_m, lim_rad
