# SPDX-License-Identifier: Apache-2.0
"""Scale and robustness family: determinism (S01), sleeping (S02), throughput and CUDA graphs (S03), memory and
contact capacity (S04). Spec: bench/SCENARIOS.md §1.6. Notes, smoke results and known problems: bench/notes/scale.md.

    python -m bench.run --family scale --device cpu --ladder 1 --horizon 0.1                   # Mac CPU smoke test
    python -m bench.run --family scale --scenario S01_determinism --device cuda:0 --ladder 1,64,4096
    python -m bench.run --family scale --scenario S03_throughput --device cuda:0 --ladder 1,16,256,1024,4096,8192,16384
    python -m bench.run --family scale --scenario S04_memory_capacity --device cuda:0           # ladder to OOM

Scenes are self-contained copies (no other family is imported): the 64-body mixed bin (pa/bench/tools/bin_scene.py),
newton/examples/assets/nv_ant.xml and nv_humanoid.xml, a Franka FR3 + 5 cm cube, a 32 x 32 cloth, a ~10k-particle
MPM block, plus for S02 the ant stacks of newton/examples/mujoco/example_mujoco_sleeping.py and the box towers and
thrown block of newton/examples/steel/example_steel_sleeping.py. Every scripted input (joint drives, kicks) runs from
device arrays indexed by a device step counter, so CUDA graphs stay on.

Cross-config and cross-rung laws (S01 D2 batch invariance, S02 speedup and awake deviation, S03 graph on/off, S04
memory fit and 0.25x capacity) read values an earlier config or rung of the same process left in _CACHE. Configs
are ordered so the reference runs first: run a scenario's configs and ladder in one process to get those laws.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench.harness import DEFAULT_LADDER, REPO, Built, Config, Scenario

NEWTON_ASSETS = Path(newton.__file__).resolve().parent / "examples" / "assets"
RESULTS = REPO / "bench" / "results" / "scale"
GAP = 0.005  # [m] shape gap for Newton-pipeline scenes; Newton's builder default rigid_gap is 0.1 m
NEVER = wp.constant(1 << 30)
L4_CAP_MB, H100_CAP_MB = 23034.0, 81559.0  # nvidia-smi memory.total [MiB] of an L4 and an H100 80GB
FALSE_SLEEP_V = 0.1  # [m/s] a body asleep while its sleeping-off twin moves faster than this is a false sleep
AWAKE_DEV_TOL_MM = 0.1
TWIN_AGREE_MM = 1.0  # S02: a false sleep is judged against the off twin only while the two runs agree this well
REF_WORLDS = 16  # S02 compares bodies of the first REF_WORLDS worlds against the sleeping-off run
EASE_S = 0.5  # drives ramp in over this time (SCENARIOS.md conventions, CON-51)
_CACHE: dict[tuple, Any] = {}
logging.getLogger("trimesh").setLevel(logging.CRITICAL)  # its scipy-less sparse fallback logs a traceback per mesh


# ---- graph-safe kernels -----------------------------------------------------------------------------------
@wp.kernel
def _k_count(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def _k_sine(counter: wp.array(dtype=wp.int32), dt: float, omega: float, base: wp.array(dtype=float),
            amp: wp.array(dtype=float), phase: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    t = float(counter[0]) * dt
    ease = wp.min(1.0, t / EASE_S)
    out[i] = base[i] + ease * amp[i] * wp.sin(omega * t + phase[i])


@wp.kernel
def _k_peak(src: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32)):
    i = wp.tid()
    wp.atomic_max(peak, 0, src[i])


@wp.kernel
def _k_sticky_or(src: wp.array(dtype=wp.int32), dst: wp.array(dtype=wp.int32)):
    i = wp.tid()
    dst[i] = dst[i] | src[i]


@wp.kernel
def _k_row_overflow(count: wp.array(dtype=wp.int32), size: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32),
                    dropped: wp.array(dtype=wp.int32)):
    i = wp.tid()
    c = count[i]
    wp.atomic_max(peak, 0, c)
    if c > size[i]:
        wp.atomic_add(dropped, 0, c - size[i])


@wp.kernel
def _k_drop2d(src: wp.array2d(dtype=wp.int32), peak: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.int32)):
    f, w = wp.tid()
    v = src[f, w]
    if v > 0:
        wp.atomic_max(peak, 0, v)
        wp.atomic_add(total, 0, v)


@wp.kernel
def _k_kick(counter: wp.array(dtype=wp.int32), kick_step: wp.array(dtype=wp.int32), bodies: wp.array(dtype=wp.int32),
            force: wp.spatial_vector, body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    if counter[0] == kick_step[0]:
        body = bodies[i]
        body_f[body] = body_f[body] + force


@wp.kernel
def _k_wake_bodies(counter: wp.array(dtype=wp.int32), kick_step: wp.array(dtype=wp.int32),
                   awake: wp.array(dtype=wp.bool), group: wp.array2d(dtype=wp.int32),
                   asleep_at_kick: wp.array(dtype=wp.int32), first_wake: wp.array(dtype=wp.int32)):
    """Steel: per world, was the watched body group asleep just before the kick, and the first step it is awake."""
    w = wp.tid()
    any_awake = int(0)
    for k in range(group.shape[1]):
        if awake[group[w, k]]:
            any_awake = 1
    step = counter[0]
    if step == kick_step[0] - 1:
        asleep_at_kick[w] = 1 - any_awake
    if step >= kick_step[0] and any_awake == 1 and first_wake[w] == NEVER:
        first_wake[w] = step


@wp.kernel
def _k_kick_sleeper(counter: wp.array(dtype=wp.int32), kick_step: wp.array(dtype=wp.int32),
                    tree_asleep: wp.array2d(dtype=wp.int32), tree_root: wp.array2d(dtype=wp.int32),
                    body_mass: wp.array(dtype=float), dv: float, h: float, chosen: wp.array(dtype=wp.int32),
                    kicked_at: wp.array(dtype=wp.int32), body_f: wp.array(dtype=wp.spatial_vector)):
    """MuJoCo: from kick_step on, the first step a world has a sleeping tree, give that tree's root body an upward
    dv (once per world) and remember the tree and the step."""
    w = wp.tid()
    if counter[0] < kick_step[0] or kicked_at[w] != NEVER:
        return
    for t in range(tree_root.shape[1]):
        b = tree_root[w, t]
        if b >= 0 and tree_asleep[w, t] >= 0:
            chosen[w] = t
            kicked_at[w] = counter[0]
            body_f[b] = body_f[b] + wp.spatial_vector(0.0, 0.0, body_mass[b] * dv / h, 0.0, 0.0, 0.0)
            return


@wp.kernel
def _k_wake_chosen(counter: wp.array(dtype=wp.int32), kicked_at: wp.array(dtype=wp.int32),
                   tree_asleep: wp.array2d(dtype=wp.int32), chosen: wp.array(dtype=wp.int32),
                   first_wake: wp.array(dtype=wp.int32)):
    """MuJoCo: first step at or after the kick at which the kicked tree is awake (tree_asleep < 0)."""
    w = wp.tid()
    t = chosen[w]
    if t < 0 or counter[0] < kicked_at[w]:
        return
    if tree_asleep[w, t] < 0 and first_wake[w] == NEVER:
        first_wake[w] = counter[0]


@wp.kernel
def _k_touch(counter: wp.array(dtype=wp.int32), kick_step: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
             shape0: wp.array(dtype=wp.int32), shape1: wp.array(dtype=wp.int32), point0: wp.array(dtype=wp.vec3),
             point1: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3), margin0: wp.array(dtype=float),
             margin1: wp.array(dtype=float), shape_body: wp.array(dtype=wp.int32), shape_role: wp.array(dtype=wp.int32),
             shape_world: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform), tol: float,
             first_touch: wp.array(dtype=wp.int32)):
    """First step after the kick at which a role-1 shape (block) touches a role-2 shape (target tower) within tol."""
    i = wp.tid()
    if i >= count[0] or counter[0] < kick_step[0]:
        return
    a = shape0[i]
    c = shape1[i]
    if a < 0 or c < 0:
        return
    if shape_role[a] * shape_role[c] != 2:
        return
    pa = point0[i]
    pc = point1[i]
    if shape_body[a] >= 0:
        pa = wp.transform_point(body_q[shape_body[a]], pa)
    if shape_body[c] >= 0:
        pc = wp.transform_point(body_q[shape_body[c]], pc)
    gap = wp.dot(normal[i], pc - pa) - (margin0[i] + margin1[i])
    if gap <= tol:
        wp.atomic_min(first_touch, shape_world[a], counter[0])


# ---- scenes -----------------------------------------------------------------------------------------------
@dataclass
class SceneInfo:
    """One world of a scene. contacts/njmax are explicit per-world capacities (Newton pipeline and MuJoCo)."""

    builder: newton.ModelBuilder
    kind: str = "rigid"  # rigid | cloth | mpm
    ground: bool = True
    drive: str = "none"  # none | joint_f | target (control.joint_target_q)
    drive_amp: float = 0.0  # [N m] or [rad]; eased in over EASE_S
    drive_dofs: int | None = None  # drive only the first n non-free dofs of each world (None = all)
    contacts: int = 0
    njmax: int = 0
    steel_rows: tuple[int, int] = (512, 32)  # Steel (free_body_constraint_max, articulation_constraint_max) per world
    meta: dict = field(default_factory=dict)


def _builder(solver: str, native_mujoco: bool = False) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    S = newton.solvers
    if solver == "mujoco":
        S.SolverMuJoCo.register_custom_attributes(b)
    elif solver == "kamino":
        S.SolverKamino.register_custom_attributes(b)
    elif solver == "implicit_mpm":
        S.SolverImplicitMPM.register_custom_attributes(b)
    b.default_shape_cfg.gap = 0.0 if native_mujoco else GAP
    b.rigid_gap = 0.0 if native_mujoco else GAP
    return b


def bin_mjcf(n: int = 64, half: float = 0.015, pitch: float = 0.04) -> str:
    """N mixed box/sphere/capsule free bodies over a walled bin (copy of pa/bench/tools/bin_scene.py build_xml)."""
    k = math.ceil(n ** (1 / 3)) if n >= 27 else math.ceil(math.sqrt(n))
    w = max(0.12, k * pitch / 2 + 0.03)
    wall_h = 0.05 + 0.6 * math.ceil(n / (k * k)) * pitch
    geoms = (f'<geom type="box" size="{half} {half * 0.8} {half * 0.6}"/>', f'<geom type="sphere" size="{half}"/>',
             f'<geom type="capsule" size="{half * 0.6} {half}"/>')
    bodies = []
    for i in range(n):
        ix, iy, iz = i % k, (i // k) % k, i // (k * k)
        x, y, z = (ix - (k - 1) / 2) * pitch, (iy - (k - 1) / 2) * pitch, 0.05 + iz * pitch
        rot = (0.13 * i) % 3.14
        bodies.append(f'<body name="o{i}" pos="{x:.4f} {y:.4f} {z:.4f}" euler="{rot:.3f} {0.7 * rot:.3f} 0">'
                      f'<freejoint/>{geoms[i % 3]}</body>')
    t = 0.01
    walls = (f'<geom type="box" size="{w + t} {w + t} {t}" pos="0 0 {-t}"/>'
             f'<geom type="box" size="{t} {w + t} {wall_h / 2}" pos="{w + t} 0 {wall_h / 2}"/>'
             f'<geom type="box" size="{t} {w + t} {wall_h / 2}" pos="{-w - t} 0 {wall_h / 2}"/>'
             f'<geom type="box" size="{w + t} {t} {wall_h / 2}" pos="0 {w + t} {wall_h / 2}"/>'
             f'<geom type="box" size="{w + t} {t} {wall_h / 2}" pos="0 {-w - t} {wall_h / 2}"/>')
    return (f'<mujoco model="bin_mixed_{n}"><option timestep="0.002" solver="Newton" iterations="50" '
            f'ls_iterations="50" cone="elliptic" integrator="implicitfast"/>'
            f'<default><geom density="700" friction="0.5 0.005 0.0001" solref="0.004 1"/></default>'
            f'<worldbody>{walls}{"".join(bodies)}</worldbody></mujoco>')


def scene_bin64(solver: str, native: bool) -> SceneInfo:
    b = _builder(solver, native)
    b.add_mjcf(bin_mjcf(64))
    return SceneInfo(b, ground=False, contacts=1024, njmax=4096, steel_rows=(1024, 32))  # 512 drops rows (S04)


def _robot(solver: str, native: bool, asset: str, z: float) -> newton.ModelBuilder:
    b = _builder(solver, native)
    b.add_mjcf(str(NEWTON_ASSETS / asset), ignore_names=["floor", "ground"], collapse_fixed_joints=False,
               parse_sites=False, xform=wp.transform((0.0, 0.0, z), wp.quat_identity()))
    return b


def scene_ant(solver: str, native: bool) -> SceneInfo:
    return SceneInfo(_robot(solver, native, "nv_ant.xml", 0.8), drive="joint_f", drive_amp=3.0, contacts=96, njmax=384,
                     steel_rows=(512, 128))


def scene_humanoid(solver: str, native: bool) -> SceneInfo:
    return SceneInfo(_robot(solver, native, "nv_humanoid.xml", 1.4), drive="joint_f", drive_amp=10.0, contacts=128,
                     njmax=512, steel_rows=(512, 128))


FRANKA_Q0 = (-3.68e-03, 2.39e-02, 3.68e-03, -2.3683, -1.29e-04, 2.3922, 0.7855)


def scene_franka_cube(solver: str, native: bool) -> SceneInfo:
    """FR3 + hand holding a pose with PD targets swept by a sine; a 5 cm cube on the floor under the hand."""
    b = _builder(solver, native)
    b.add_urdf(_franka_urdf_xml(), xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()), floating=False,
               enable_self_collisions=False, parse_visuals_as_colliders=False, joint_damping_mode="passive",
               collapse_fixed_joints=True)
    b.joint_q[:9] = [*FRANKA_Q0, 0.04, 0.04]
    b.joint_target_q[:9] = [*FRANKA_Q0, 0.04, 0.04]
    b.joint_target_ke[:9] = [600.0] * 7 + [100.0] * 2
    b.joint_target_kd[:9] = [60.0] * 7 + [10.0] * 2
    b.joint_armature[:9] = [0.1] * 9
    cube = b.add_body(xform=wp.transform((0.42, 0.0, 0.026), wp.quat_identity()), label="cube")
    b.add_shape_box(cube, hx=0.025, hy=0.025, hz=0.025)
    return SceneInfo(b, drive="target", drive_amp=0.3, drive_dofs=7, contacts=96, njmax=384, steel_rows=(512, 128))


def _franka_urdf_xml() -> str:
    """fr3_franka_hand.urdf without <visual> blocks (their .dae meshes need pycollada, absent from the venv) and with
    absolute collision-mesh paths, so it loads from a string."""
    root = newton.utils.download_asset("franka_emika_panda")
    text = re.sub(r"<visual\b.*?</visual>", "", (root / "urdf/fr3_franka_hand.urdf").read_text(), flags=re.S)
    return text.replace("package://franka_emika_panda/", f"{root}/")


def scene_cloth32(solver: str, native: bool) -> SceneInfo:
    """32 x 32 cloth pinned on one edge, draping onto the floor (newton/examples/cloth/example_cloth_hanging.py)."""
    b = newton.ModelBuilder()
    common = dict(pos=wp.vec3(0.0, 0.0, 2.5), rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
                  vel=wp.vec3(0.0, 0.0, 0.0), dim_x=32, dim_y=32, cell_x=0.1, cell_y=0.1, mass=0.1, fix_left=True,
                  edge_ke=1.0e1, edge_kd=0.0, particle_radius=0.05)
    if solver == "xpbd":
        b.add_cloth_grid(**common, add_springs=True, spring_ke=1.0e3, spring_kd=1.0)
    else:
        b.add_cloth_grid(**common, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e2)
    return SceneInfo(b, kind="cloth", contacts=0)


def scene_mpm10k(solver: str, native: bool, voxel: float = 0.02, side: float = 0.14) -> SceneInfo:
    """~10.6k granular MPM particles (22^3) dropped on the floor (newton/examples/mpm/example_mpm_granular.py)."""
    b = _builder("implicit_mpm")
    lo = np.array([-side / 2, -side / 2, 0.05])
    res = np.ceil(3 * side / voxel).astype(int) * np.ones(3, int)
    cell = side / res
    radius = 0.5 * float(np.max(cell))
    b.add_particle_grid(pos=wp.vec3(*lo), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=int(res[0]) + 1,
                        dim_y=int(res[1]) + 1, dim_z=int(res[2]) + 1, cell_x=float(cell[0]), cell_y=float(cell[1]),
                        cell_z=float(cell[2]), mass=float(np.prod(cell) * 1000.0), jitter=2.0 * radius,
                        radius_mean=radius)
    return SceneInfo(b, kind="mpm", meta={"voxel": voxel})


def _pendulum(b: newton.ModelBuilder, pivot: tuple[float, float, float], angle: float = 1.0) -> int:
    """A free-swinging hinge pendulum: a witness island that never sleeps. Returns the link body."""
    link = b.add_link(xform=wp.transform((pivot[0], pivot[1], pivot[2] - 0.4), wp.quat_identity()), label="witness")
    b.add_shape_box(link, hx=0.03, hy=0.03, hz=0.1)
    j = b.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Y,
                             parent_xform=wp.transform(pivot, wp.quat_identity()),
                             child_xform=wp.transform((0.0, 0.0, 0.4), wp.quat_identity()))
    b.add_articulation([j])
    b.joint_q[-1] = angle
    return link


ANT_STACK_SPACING = 3.5  # [m] upstream 2.5 starts with toppled feet of adjacent stacks 16 cm deep (notes / S02)


def scene_ant_stacks(solver: str, native: bool) -> SceneInfo:
    """Four stacks of five ants toppling into each other (example_mujoco_sleeping.py) plus a witness pendulum."""
    S = newton.solvers.SolverMuJoCo
    ant = _builder("mujoco", True)
    root = len(ant.body_mass)
    ant.add_mjcf(str(NEWTON_ASSETS / "nv_ant.xml"), ignore_names=["floor", "ground"], collapse_fixed_joints=False,
                 parse_sites=False)
    ant.custom_attributes["mujoco:sleep_policy"].values[root] = S.SleepPolicy.ALLOWED
    stack = _builder("mujoco", True)
    for i in range(5):
        level = max(0, i - 2)
        stack.add_builder(ant, xform=wp.transform((0.08 * level, 0.0, 1.0 + 1.25 * i), wp.quat_from_axis_angle(
            wp.vec3(0.0, 1.0, 0.0), math.radians(4.0) if level else 0.0)), label_prefix=f"ant_{i}_")
    b = _builder("mujoco", True)
    for s in range(4):
        x, y = (s % 2 - 0.5) * ANT_STACK_SPACING, (s // 2 - 0.5) * ANT_STACK_SPACING
        b.add_builder(stack, xform=wp.transform((x, y, 0.0), wp.quat_from_axis_angle(
            wp.vec3(0.0, 0.0, 1.0), math.atan2(-y, -x))), label_prefix=f"stack_{s}_")
    witness = _pendulum(b, (6.0, 0.0, 1.5))
    return SceneInfo(b, contacts=2048, njmax=4096, meta={"witness": witness})


TOWER_HEIGHTS, TOWER_DROPS, TARGET_TOWER = (6, 7, 5, 6), (0.0, 0.5, 1.0, 1.5), 1
KICK_VELOCITY = (0.0, 5.0, 3.0)
KICK_DV_ANT = 2.0  # [m/s] upward velocity change given to the kicked (sleeping) ant's torso


def scene_towers(solver: str, native: bool) -> SceneInfo:
    """Four box towers and a block thrown into tower 1 (example_steel_sleeping.py) plus a witness pendulum."""
    b = _builder(solver)
    b.default_shape_cfg.mu = 0.7
    tower_of_body, x0, h = [], -0.5 * (len(TOWER_HEIGHTS) - 1) * 1.2, 0.1
    for tower, (height, drop) in enumerate(zip(TOWER_HEIGHTS, TOWER_DROPS, strict=True)):
        for level in range(height):
            yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(12.0) * (level % 2))
            body = b.add_body(xform=wp.transform((x0 + tower * 1.2, 0.0, h + drop + level * (2 * h + 0.002)), yaw))
            b.add_shape_box(body, hx=h, hy=h, hz=h)
            tower_of_body.append(tower)
    block = b.add_body(xform=wp.transform((x0 + TARGET_TOWER * 1.2, -1.4, 0.12), wp.quat_identity()), label="block")
    b.add_shape_box(block, hx=0.12, hy=0.12, hz=0.12)
    tower_of_body.append(-1)
    witness = _pendulum(b, (4.0, 0.0, 1.5))
    tower_of_body.append(-2)
    meta = {"tower_of_body": np.asarray(tower_of_body), "block": block, "witness": witness}
    return SceneInfo(b, contacts=512, steel_rows=(1024, 64), meta=meta)


SCENES: dict[str, Callable[[str, bool], SceneInfo]] = {
    "bin64": scene_bin64, "ant": scene_ant, "humanoid": scene_humanoid, "franka_cube": scene_franka_cube,
    "cloth32": scene_cloth32, "mpm10k": scene_mpm10k, "ant_stacks": scene_ant_stacks, "towers": scene_towers,
}
HETERO = ("ant", "humanoid")  # world w holds HETERO[w % 2]


def build_model(scene: str, solver: str, nworld: int, device: str, native: bool) -> tuple[newton.Model, SceneInfo]:
    """Replicate one world nworld times (or alternate HETERO scenes), add the shared floor, finalize."""
    main = _builder(solver, native)
    if scene == "hetero":
        infos = [SCENES[s](solver, native) for s in HETERO]
        for w in range(nworld):
            main.add_world(builder=infos[w % len(infos)].builder)
        info = infos[1]  # the larger capacities cover both
        info.meta["world_amp"] = [infos[w % len(infos)].drive_amp for w in range(nworld)]
    else:
        info = SCENES[scene](solver, native)
        main.replicate(info.builder, copy_count=nworld)
    if info.ground:
        main.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5) if info.kind == "mpm" else None)
    if info.kind == "cloth" and solver == "vbd":
        main.color(include_bending=True)
    model = main.finalize(device=device)
    if info.kind == "cloth":
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1.0e2, 1.0e2, 1.0
    if info.kind == "mpm":
        model.mpm.friction.fill_(0.68)
    return model, info


# ---- solvers, pipelines, capacities -----------------------------------------------------------------------
def make_solver(model: newton.Model, solver: str, kw: dict, det: bool):
    """Solver factory; kw are the config's solver kwargs, det adds deterministic=RUN_TO_RUN where supported."""
    S, kw = newton.solvers, dict(kw)
    mode = wp.DeterministicMode.RUN_TO_RUN if det else None
    if solver == "mujoco":
        return S.SolverMuJoCo(model, deterministic=mode, **kw)
    if solver == "featherstone":
        return S.SolverFeatherstone(model, deterministic=mode, **kw)
    if solver == "xpbd":
        return S.SolverXPBD(model, deterministic=mode, **{"iterations": 10, **kw})
    if solver == "steel":
        kw.setdefault("solve_mode", "matrix_free" if model.device.is_cuda else "split")
        return S.SolverSteel(model, deterministic=mode, **kw)
    if solver == "vbd":
        return S.SolverVBD(model, deterministic=mode, **kw)
    if solver == "kamino":  # no deterministic switch; det mode only sorts the collision pipeline
        cfg = S.SolverKamino.Config.from_model(model, **{"dynamics_solver": "dvi", "sparse_dynamics": True,
                                                         "sparse_jacobian": True, **kw})
        return S.SolverKamino(model, config=cfg)
    if solver == "implicit_mpm":  # no deterministic switch
        cfg = S.SolverImplicitMPM.Config()
        for k, v in kw.items():
            setattr(cfg, k, v)
        cfg.separate_worlds = model.world_count > 1
        return S.SolverImplicitMPM(model, config=cfg)
    raise ValueError(f"unknown solver {solver!r}")


SOLVER_KW = {  # production-like settings shared by S01/S03 (the record shows them under params.kw)
    "mujoco": {"use_mujoco_contacts": True, "iterations": 20, "ls_iterations": 10, "cone": "elliptic"},
    "mujoco_nc": {"use_mujoco_contacts": False, "iterations": 20, "ls_iterations": 10, "cone": "elliptic"},
    "featherstone": {}, "xpbd": {"iterations": 10}, "kamino": {}, "steel": {},
    "vbd": {"iterations": 10, "particle_enable_self_contact": True, "particle_self_contact_margin": 0.02,
            "particle_self_contact_gap": 0.01},
    "implicit_mpm": {"voxel_size": 0.02, "max_iterations": 50, "tolerance": 1.0e-3},
}
NEED_FALLBACK = {"rigid": 400, "nacon": 300, "nefc": 1200, "vertex": 8, "edge": 8, "cells": 400,
                 "soft": 400}  # per world (vertex/edge: per item)


def _capacities(info: SceneInfo, p: dict, nworld: int, solver: str, device: str) -> tuple[dict, dict]:
    """Per-config (solver kwargs, CollisionPipeline kwargs): explicit capacities (S01-S03), Newton's defaults
    (S04 cap_default) or 0.25x the rows the default run needed (S04 cap_quarter)."""
    mode = p.get("cap", "explicit")
    if mode == "default":
        return {}, {}
    if mode == "explicit":
        kw = {"nconmax": info.contacts, "njmax": info.njmax} if solver == "mujoco" and info.contacts else {}
        if solver == "steel":
            kw = {"free_body_constraint_max": info.steel_rows[0], "articulation_constraint_max": info.steel_rows[1]}
        return kw, ({"rigid_contact_max": info.contacts * nworld} if info.contacts else {})
    need = {**NEED_FALLBACK, **_CACHE.get(("S04need", p["scene"], solver, device), {})}
    q = p.get("cap_fraction", 0.25)
    cap = lambda key, n=1: max(1, int(q * need[key] * n))  # noqa: E731
    if solver == "mujoco":  # per world
        return {"nconmax": cap("nacon"), "njmax": cap("nefc")}, {}
    if solver == "vbd":  # per vertex / per edge, plus the pipeline's particle-shape (soft) contacts
        return ({"particle_vertex_contact_buffer_size": cap("vertex"), "particle_edge_contact_buffer_size": cap("edge")},
                {"soft_contact_max": cap("soft", nworld)})
    if solver == "implicit_mpm":  # sparse-grid capacity is a total across worlds
        return {"max_active_cell_count": cap("cells", nworld)}, {}
    return {}, {"rigid_contact_max": cap("rigid", nworld)}


# ---- per-step extras: drives, kicks, capacity monitors ----------------------------------------------------
class StepExtras:
    """Device-side additions to every step: a step counter, joint drives, kicks and capacity/wake monitors."""

    def __init__(self, b: Built, info: SceneInfo, dt_outer: float):
        self.device = b.model.device
        self.counter = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.pre: list[tuple] = []  # launched before the substeps of an outer step
        self.on_substep0: list[tuple] = []  # launched after clear_forces on substep 0 (kicks)
        self.post_sub: list[tuple] = []  # launched after every solver.step
        self.post: list[tuple] = []  # launched after the outer step, before the counter increments
        self.step_kw: dict = {}
        self._add_drive(b, info, dt_outer)

    def _add_drive(self, b: Built, info: SceneInfo, dt: float) -> None:
        m = b.model
        if info.drive == "none" or m.joint_dof_count == 0:
            return
        amp, phase = drive_profile(m, info)
        out = b.control.joint_f if info.drive == "joint_f" else b.control.joint_target_q
        base = wp.zeros_like(out) if info.drive == "joint_f" else wp.clone(out)
        arrs = [wp.array(a, dtype=float, device=self.device) for a in (amp, phase)]
        self.pre.append((_k_sine, m.joint_dof_count, [self.counter, dt, 2.0 * math.pi * 0.8, base, arrs[0], arrs[1],
                                                      out]))

    def launch(self, items: list[tuple]) -> None:
        for kernel, dim, inputs in items:
            wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def end_step(self) -> None:
        self.launch(self.post)
        wp.launch(_k_count, dim=1, inputs=[self.counter], device=self.device)


def drive_profile(model: newton.Model, info: SceneInfo) -> tuple[np.ndarray, np.ndarray]:
    """Per-dof amplitude and phase. Free-joint dofs get 0; phases depend on the world-local dof index only, so
    world 0 sees the same drive at every world count."""
    jt, qd0 = model.joint_type.numpy(), model.joint_qd_start.numpy()
    jw = model.joint_world.numpy()
    amp, phase = np.zeros(model.joint_dof_count), np.zeros(model.joint_dof_count)
    world_amp = info.meta.get("world_amp")
    next_local: dict[int, int] = {}
    for j in range(model.joint_count):
        w = int(jw[j])
        for d in range(int(qd0[j]), int(qd0[j + 1])):
            local = next_local.get(w, 0)
            next_local[w] = local + 1
            if jt[j] == newton.JointType.FREE:
                continue
            driven = info.drive_dofs is None or local < info.drive_dofs
            amp[d] = (world_amp[w] if world_amp else info.drive_amp) if driven else 0.0
            phase[d] = 2.0 * math.pi * ((0.6180339887 * local) % 1.0)
    return amp, phase


class Monitor:
    """Capacity counters updated after every solver substep (graph safe) and their host-side report.

    Ground truth is the raw counter of each buffer (Newton's rigid contact counter, MuJoCo Warp's nacon/nefc and
    VBD's per-vertex/per-edge collision counts all keep counting past capacity). An engine signal is a flag the
    engine itself exposes: Newton pipeline counter > capacity (verify_buffers also prints), MuJoCo's OVERFLOW
    observable, Steel's constraint_overflow and Kamino's per-world warning flag (both also printed; Kamino caps its
    counter, so its row count is a lower bound). A drop with no such signal is a silent drop (S1,
    docs/research/bench.md §4.6). VBD's TriMeshCollisionDetector.resize_flags is recorded as a passive flag only:
    Newton resets it every detection pass and never reads or prints it, so a VBD drop still counts as silent.
    """

    MJ_NON_CAPACITY_BITS = 512 | 1024  # mujoco_warp.OverflowType ITERATIONS, LS_ITERATIONS

    def __init__(self, b: Built, x: StepExtras):
        self.x, dev, self.items = x, b.model.device, {}
        z = lambda n=1: wp.zeros(n, dtype=wp.int32, device=dev)  # noqa: E731
        c = b.contacts
        if c is not None and getattr(c, "rigid_contact_count", None) is not None and c.rigid_contact_max > 0:
            self.items["pipeline_rigid"] = {"cap": int(c.rigid_contact_max), "peak": z(), "signal": "self"}
            x.post_sub.append((_k_peak, 1, [c.rigid_contact_count, self.items["pipeline_rigid"]["peak"]]))
        if c is not None and getattr(c, "soft_contact_count", None) is not None and c.soft_contact_max > 0 \
                and b.model.particle_count:
            self.items["pipeline_soft"] = {"cap": int(c.soft_contact_max), "peak": z(), "signal": "self"}
            x.post_sub.append((_k_peak, 1, [c.soft_contact_count, self.items["pipeline_soft"]["peak"]]))
        red = getattr(c, "_reduction_overflow", None) if c is not None else None
        if red is not None:  # contact reduction dropped candidates (private flag, reset every collision pass)
            self.reduction_flag = z(1)
            x.post_sub.append((_k_sticky_or, 1, [red, self.reduction_flag]))
        drop = getattr(b.solver, "_row_dropped_all", None)  # Steel rows dropped per (family, world), per step
        if drop is not None and drop.ndim == 2:
            it = {"cap": None, "peak": z(), "total": z(), "signal": "steel"}
            self.items["steel_rows"] = it
            x.post_sub.append((_k_drop2d, drop.shape, [drop, it["peak"], it["total"]]))
        if isinstance(b.solver, newton.solvers.SolverMuJoCo) and not b.solver.use_mujoco_cpu:
            self._mujoco(b, x, z)
        det = getattr(b.solver, "trimesh_collision_detector", None)
        if det is not None:
            info = det.collision_info
            for name, cnt, size in (("vbd_vertex_tri", "vertex_colliding_triangles_count",
                                     "vertex_colliding_triangles_buffer_sizes"),
                                    ("vbd_edge_edge", "edge_colliding_edges_count",
                                     "edge_colliding_edges_buffer_sizes")):
                cnt_a, size_a = getattr(info, cnt, None), getattr(info, size, None)
                if cnt_a is None or size_a is None:
                    continue
                it = {"cap": int(size_a.numpy().min()), "peak": z(), "dropped": z(), "signal": None}
                self.items[name] = it
                x.post_sub.append((_k_row_overflow, cnt_a.shape[0], [cnt_a, size_a, it["peak"], it["dropped"]]))
            flags = getattr(det, "resize_flags", None)  # set on overflow, reset every detection pass, read by nobody
            if flags is not None:
                self.vbd_flags = z(flags.shape[0])
                x.post_sub.append((_k_sticky_or, flags.shape[0], [flags, self.vbd_flags]))
        kc = getattr(b.solver, "_contacts_kamino", None)  # Kamino's own per-world contact buffer (capped counter)
        if kc is not None and kc.model_max_contacts_host > 0:
            it = {"cap": int(min(kc.world_max_contacts_host)), "peak": z(), "signal": "kamino"}
            self.items["kamino_world_contacts"] = it
            x.post_sub.append((_k_peak, kc.world_active_contacts.shape[0], [kc.world_active_contacts, it["peak"]]))
            flag = getattr(kc.data, "_contact_overflow_warning_emitted", None)  # zeroed every step; printed once
            if flag is not None:
                self.kamino_flag = z(1)
                x.post_sub.append((_k_sticky_or, 1, [flag, self.kamino_flag]))

    def _mujoco(self, b: Built, x: StepExtras, z) -> None:
        s, d = b.solver, b.solver.mjw_data
        obs = s.observables({s.ObservableKind.OVERFLOW})
        x.step_kw["observables"] = obs
        self.mj_sticky = z(b.model.world_count)
        x.post_sub.append((_k_sticky_or, b.model.world_count, [obs.overflow, self.mj_sticky]))
        if getattr(s, "use_mujoco_contacts", True) and getattr(d, "nacon", None) is not None:
            self.items["mujoco_nacon"] = {"cap": int(d.naconmax), "peak": z(), "signal": "mj"}
            x.post_sub.append((_k_peak, 1, [d.nacon, self.items["mujoco_nacon"]["peak"]]))
        if getattr(d, "nefc", None) is not None:
            self.items["mujoco_nefc_per_world"] = {"cap": int(d.njmax), "peak": z(), "signal": "mj"}
            x.post_sub.append((_k_peak, d.nefc.shape[0], [d.nefc, self.items["mujoco_nefc_per_world"]["peak"]]))

    def report(self, b: Built) -> dict:
        bufs, signals = {}, {}
        if hasattr(self, "mj_sticky"):
            bits = int(np.bitwise_or.reduce(self.mj_sticky.numpy())) if b.model.world_count else 0
            signals["mujoco_overflow_bits"] = bits
            signals["mujoco_capacity_overflow"] = bool(bits & ~self.MJ_NON_CAPACITY_BITS)
        over = getattr(b.solver, "constraint_overflow", None)
        if over is not None:
            signals["steel_constraint_overflow"] = bool(np.any(over.numpy()))
        if hasattr(self, "kamino_flag"):
            signals["kamino_capacity_warning"] = bool(self.kamino_flag.numpy()[0])
        passive = {}
        if hasattr(self, "reduction_flag"):
            passive["pipeline_reduction_overflow"] = int(self.reduction_flag.numpy()[0])
        if hasattr(self, "vbd_flags"):
            passive["vbd_resize_flags"] = self.vbd_flags.numpy().tolist()  # [vertex, tri, edge, tri_tri] buffers
        total = 0
        for name, it in self.items.items():
            peak = int(it["peak"].numpy()[0])
            row = {"capacity": it["cap"], "peak": peak}
            if it["signal"] == "steel":  # peak = most rows one world dropped in one substep; total over substeps
                row.update(capacity=None, row_substeps_dropped=int(it["total"].numpy()[0]))
                lost = peak
            else:
                lost = int(it["dropped"].numpy()[0]) if "dropped" in it else max(0, peak - it["cap"])
            if it["signal"] == "kamino":  # counter is capped: the warning flag says rows were dropped, not how many
                lost = int(signals.get("kamino_capacity_warning", False))
                row["count_capped"] = True
            row["overflow_rows"] = lost
            if it["signal"] == "self":
                key = f"{name}_count_gt_capacity"
                signals[key] = signals.get(key, False) or lost > 0
            bufs[name] = row
            total += lost
        detected = any(bool(v) for k, v in signals.items() if k != "mujoco_overflow_bits")
        return {"capacity_buffers": bufs, "contact_overflow": total, "overflow_signals": signals,
                "passive_flags": passive, "overflow_detected": detected,
                "passive_flag_only": bool(total > 0 and not detected and any(any(v) for v in passive.values())),
                "silent_drop": bool(total > 0 and not detected)}


# ---- helpers ----------------------------------------------------------------------------------------------
def world_snapshot(model: newton.Model, state, world: int = 0) -> dict[str, np.ndarray]:
    """State arrays of one world (bodies and particles whose world index matches)."""
    out = {}
    if model.body_count:
        m = model.body_world.numpy() == world
        out["body_q"], out["body_qd"] = state.body_q.numpy()[m], state.body_qd.numpy()[m]
    if model.particle_count:
        m = model.particle_world.numpy() == world
        out["particle_q"], out["particle_qd"] = state.particle_q.numpy()[m], state.particle_qd.numpy()[m]
    return out


def positions(snap: dict[str, np.ndarray]) -> np.ndarray:
    parts = [snap["body_q"][:, :3]] if "body_q" in snap else []
    if "particle_q" in snap:
        parts.append(snap["particle_q"])
    return np.concatenate(parts).astype(np.float64) if parts else np.zeros((0, 3))


def max_ulp(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> int:
    """Largest ULP distance between matching float32 entries of two snapshots (0 = bitwise identical)."""
    worst = 0
    for k, x in a.items():
        y = b[k]
        if x.dtype != np.float32:
            d = np.abs(x.astype(np.float64) - y) / np.maximum(np.spacing(np.abs(x.astype(np.float64))), 1e-300)
            worst = max(worst, int(np.nanmax(d)) if d.size else 0)
            continue
        ix, iy = (v.view(np.int32).astype(np.int64) for v in (x, y))
        ix = np.where(ix < 0, -(ix & 0x7FFFFFFF), ix)
        iy = np.where(iy < 0, -(iy & 0x7FFFFFFF), iy)
        if ix.size:
            worst = max(worst, int(np.max(np.abs(ix - iy))))
    return worst


def w1_mm(pa: np.ndarray, pb: np.ndarray) -> float:
    """W1 between two equal-size point sets [mm] (GPUSimBench assignment W1). Exact with scipy for n <= 512,
    otherwise identity matching, an upper bound that is exact when per-item drift is far below item spacing."""
    d = np.linalg.norm(pa - pb, axis=1)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return float(d.mean() * 1e3) if d.size else 0.0
    if len(pa) > 512 or not len(pa):
        return float(d.mean() * 1e3) if d.size else 0.0
    cost = np.linalg.norm(pa[:, None] - pb[None], axis=2)
    r, c = linear_sum_assignment(cost)
    return float(cost[r, c].mean() * 1e3)


def array_census_mb(*roots: Any, max_depth: int = 6) -> float:
    """MiB of Warp arrays that own memory and are reachable from roots (Newton / MuJoCo Warp / Warp objects).

    Device independent, so the S04 memory law also runs on CPU. Views are skipped (no deleter); each owning
    allocation is counted once by pointer."""
    seen, ptrs, total = set(), set(), 0
    stack = [(r, 0) for r in roots if r is not None]
    while stack:
        obj, depth = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, wp.array):
            if obj.deleter is not None and obj.ptr and obj.ptr not in ptrs:
                ptrs.add(obj.ptr)
                total += int(obj.capacity)
            continue
        if depth >= max_depth or len(seen) > 400_000 or isinstance(obj, (str, bytes, np.ndarray)):
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            kids = obj
        elif isinstance(obj, dict):
            kids = obj.values()
        elif type(obj).__module__.split(".")[0] in ("newton", "mujoco_warp", "warp") and hasattr(obj, "__dict__"):
            kids = vars(obj).values()
        else:
            continue
        stack.extend((k, depth + 1) for k in kids)
    return total / 2**20


def _pool_mb(device: str) -> float | None:
    """Warp mempool bytes in use on a CUDA device [MiB]; None on CPU or with the mempool off."""
    dev = wp.get_device(device)
    if not dev.is_cuda or not wp.is_mempool_enabled(dev):
        return None
    return wp.get_mempool_used_mem_current(dev) / 2**20


def linear_fit(n: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least squares y = a + b n; returns (a, b, R^2)."""
    A = np.vstack([np.ones_like(n), n]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    ss_res, ss_tot = float(np.sum((y - (a + b * n)) ** 2)), float(np.sum((y - y.mean()) ** 2))
    return float(a), float(b), (1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def perturb_neighbours(model: newton.Model, state, amp: float = 0.05, seed: int = 1234) -> None:
    """Add uniform velocity noise (amp m/s or rad/s) to every world except world 0, then refresh body states."""
    if model.world_count <= 1:
        return
    rng = np.random.default_rng(seed)
    if model.joint_dof_count:
        qd0, jw = model.joint_qd_start.numpy(), model.joint_world.numpy()
        dof_world = np.repeat(jw, np.diff(qd0))
        qd = state.joint_qd.numpy()
        m = dof_world >= 1
        qd[m] += rng.uniform(-amp, amp, int(m.sum())).astype(qd.dtype)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    if model.particle_count:
        v = state.particle_qd.numpy()
        m = (model.particle_world.numpy() >= 1) & (model.particle_mass.numpy() > 0)
        v[m] += rng.uniform(-amp, amp, (int(m.sum()), 3)).astype(v.dtype)
        state.particle_qd.assign(v)


# ---- shared base ------------------------------------------------------------------------------------------
class ScaleBase(Scenario):
    """Builds any SCENES entry for any solver from Config.params and steps it with StepExtras and Monitor.

    params: scene, kw (solver kwargs), det (pipeline deterministic + solver RUN_TO_RUN), native (MuJoCo's own
    contacts), cap (explicit | default | quarter), perturb (S01 neighbour noise)."""

    family = "scale"
    has_contacts = True

    def build(self, nworld: int, device: str, config: Config) -> Built:
        p = config.params
        native = config.solver == "mujoco" and p.get("kw", {}).get("use_mujoco_contacts", False)
        model, info = build_model(p["scene"], config.solver, nworld, device, native)
        cap_kw, pipe_kw = _capacities(info, p, nworld, config.solver, device)
        rigid_max = pipe_kw.get("rigid_contact_max")
        det_pipe, det_solver = bool(p.get("det_pipe", p.get("det"))), bool(p.get("det_solver", p.get("det")))
        solver = make_solver(model, config.solver, {**p.get("kw", {}), **cap_kw}, det_solver)
        s0, s1, control = model.state(), model.state(), model.control()
        if model.joint_count:
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        pipeline = contacts = None
        if info.kind != "mpm" and not native:
            pipeline = newton.CollisionPipeline(model, deterministic=det_pipe, **pipe_kw)
            contacts = pipeline.contacts()
        if p.get("perturb"):
            perturb_neighbours(model, s0)
        b = Built(model, solver, s0, s1, control, contacts, pipeline, nworld=nworld,
                  extra={"config": config, "info": info, "device": device, "cap_kw": {**cap_kw, **pipe_kw},
                         "rigid_max": rigid_max})
        if info.kind == "mpm":
            if hasattr(s0, "mpm") and hasattr(s0.mpm, "particle_Jp"):
                s0.mpm.particle_Jp.fill_(1.0)
            self.guard(b, lambda: solver.project_outside(s0, s0, config.dt / config.substeps))
        x = b.extra["x"] = StepExtras(b, info, config.dt)
        b.extra["monitor"] = Monitor(b, x)
        self.extend(b, info, config)
        return b

    def extend(self, b: Built, info: SceneInfo, config: Config) -> None:
        """Scenario-specific device extras (S02 kicks and wake monitors)."""

    def guard(self, b: Built, fn: Callable[[], Any]) -> None:
        """Run fn; S04 overrides this to record an engine capacity exception as a detected overflow."""
        fn()

    def step(self, b: Built, config: Config) -> None:
        if b.extra.get("halted"):
            return
        self.guard(b, lambda: self._step(b, config))

    def _step(self, b: Built, config: Config) -> None:
        x: StepExtras = b.extra["x"]
        x.launch(x.pre)
        h = config.dt / config.substeps
        for k in range(config.substeps):
            self._substep(b, h, k, x)
        x.end_step()

    def _substep(self, b: Built, dt: float, k: int, x: StepExtras) -> None:
        b.state_0.clear_forces()
        if k == 0:
            x.launch(x.on_substep0)
        if b.extra["info"].kind == "mpm":
            b.solver.step(b.state_0, b.state_1, None, None, dt)
            b.solver.project_outside(b.state_1, b.state_1, dt)
        else:
            if b.pipeline is not None:
                b.pipeline.collide(b.state_0, b.contacts)
            b.solver.step(b.state_0, b.state_1, b.control, b.contacts, dt, **x.step_kw)
        b.state_0, b.state_1 = b.state_1, b.state_0
        x.launch(x.post_sub)

    def measure(self, b: Built, t: float, rec: dict) -> None:
        """Default penetration sample plus a capacity snapshot, so a run that later goes non-finite (the harness
        then skips finish()) still records whether rows were dropped before it blew up."""
        super().measure(b, t, rec)
        rec["metrics"].update(b.extra["monitor"].report(b), capacity_snapshot_t=t)

    def rollout(self, config: Config, nworld: int, device: str, steps: int,
                snap: Callable[[Built], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """A fresh eager run of the same config and nworld for `steps` outer steps; returns snap(final Built)."""
        with wp.ScopedDevice(device):
            b = self.build(nworld, device, config)
            for _ in range(steps):
                self.step(b, config)
            wp.synchronize_device(device)
            out = snap(b)
        del b
        gc.collect()
        return out

    def capacity_metrics(self, b: Built, rec: dict) -> dict:
        rep = b.extra["monitor"].report(b)
        rec["metrics"].update(rep, rigid_contact_max=b.extra["rigid_max"], capacity_kw=b.extra["cap_kw"])
        if rep["silent_drop"]:
            rec["metrics"]["severity"] = "S1"
        return rep


SUBSTEPS = {("ant", "featherstone"): 40, ("hetero", "featherstone"): 40}  # NaN at 600 Hz (notes §known problems)


def _cfg(name: str, solver: str, scene: str, kw_key: str | None = None, dt: float = 1 / 60, substeps: int = 10,
         **params: Any) -> Config:
    substeps = SUBSTEPS.get((scene, solver), substeps)
    graph = params.pop("cuda_graph", True)
    max_nworld = params.pop("max_nworld", None)
    kw = {**SOLVER_KW[kw_key or solver], **params.pop("kw", {})}
    return Config(name, solver, dt=dt, substeps=substeps, params={"scene": scene, "kw": kw, **params},
                  cuda_graph=graph, max_nworld=max_nworld)


# ---- S01 determinism --------------------------------------------------------------------------------------
S01_PLAN = {  # scene -> solvers (MuJoCo on Newton contacts so the pipeline's deterministic switch matters)
    "bin64": ("mujoco_nc", "xpbd", "kamino", "steel", "featherstone"),
    "ant": ("mujoco_nc", "featherstone", "xpbd", "kamino", "steel"),
    "humanoid": ("mujoco_nc", "featherstone", "xpbd", "kamino"),
    "franka_cube": ("mujoco_nc", "featherstone", "xpbd"),
    "cloth32": ("vbd", "xpbd"),
    "mpm10k": ("implicit_mpm",),
}
NO_DET_SWITCH = {"implicit_mpm"}


def _s01_configs() -> tuple[Config, ...]:
    out = []
    for scene, solvers in S01_PLAN.items():
        for key in solvers:
            solver = "mujoco" if key == "mujoco_nc" else key
            modes = [("det", True, True), ("nondet", False, False)]  # (label, pipeline, solver RUN_TO_RUN)
            if scene == "bin64" and solver != "kamino":  # contact-rich scene: the full pipeline x solver cross
                modes[1:1] = [("pipedet", True, False), ("solverdet", False, True)]
            if solver in NO_DET_SWITCH:
                modes = [("nondet", False, False)]
            for label, det_pipe, det_solver in modes:
                dt, sub = (1 / 120, 1) if scene == "mpm10k" else (1 / 60, 10)
                out.append(_cfg(f"{scene}_{key}_{label}", solver, scene, key, dt=dt, substeps=sub,
                                det=det_pipe and det_solver, det_pipe=det_pipe, det_solver=det_solver, perturb=True,
                                cuda_graph=scene != "mpm10k",
                                max_nworld=64 if scene == "mpm10k" else None))  # 4096 x 10.6k particles: skip
    return tuple(out)


class Determinism(ScaleBase):
    """D1 rerun ULP, D2 batch invariance of world 0 (neighbours perturbed), D3 rerun spread W1 (GPUSimBench Eq. 11).

    Modes: det = CollisionPipeline(deterministic=True) + solver deterministic=wp.DeterministicMode.RUN_TO_RUN;
    nondet = both off; on bin64 also pipedet = pipeline on, solver off and solverdet = pipeline off, solver on.
    Kamino and ImplicitMPM have no solver switch.
    finish() makes RERUNS fresh eager rollouts of the same config and nworld; D1 and D3 compare them. D2 compares
    world 0 of rerun 0 with the nworld = 1 rerun 0 of the same config in this process, so run the ladder in one
    process starting at 1. Neighbour worlds (>= 1) start with +-0.05 m/s (rad/s) velocity noise."""

    id = "S01_determinism"
    title = "Rerun ULP, batch invariance and rerun W1 per solver x deterministic mode"
    horizon_s = 100 / 60  # 1000 solver steps at 600 Hz
    ladder = (1, 64, 4096)
    accuracy_unit = "mm"
    accuracy_ref = "D1 0 ulp (CPU, det); D2 0 mm (det); D3 W1 noise floor (GPUSimBench arXiv:2607.13059 Eq. 11)"
    configs = _s01_configs()
    RERUNS = 5

    def _rollout(self, config: Config, nworld: int, device: str, steps: int) -> dict[str, np.ndarray]:
        return self.rollout(config, nworld, device, steps, lambda r: world_snapshot(r.model, r.state_0))

    def finish(self, b: Built, rec: dict) -> None:
        cfg, dev, steps, det = b.extra["config"], b.extra["device"], rec["steps"], bool(b.extra["config"].params["det"])
        self.capacity_metrics(b, rec)
        done_key = ("S01done", cfg.name, b.nworld, dev, steps)
        if done_key in _CACHE:  # the harness's repeat_identical run: its record is discarded, skip the reruns
            rec["metrics"]["note"] = "repeat run, reruns skipped"
            return
        _CACHE[done_key] = True
        main = world_snapshot(b.model, b.state_0)
        runs = [self._rollout(cfg, b.nworld, dev, steps) for _ in range(self.RERUNS)]
        ulp = max(max_ulp(runs[0], r) for r in runs[1:])
        ulp_q = max(max_ulp(_q_only(runs[0]), _q_only(r)) for r in runs[1:])  # positions only (no near-zero velocities)
        pos = [positions(r) for r in runs]
        rerun_max_mm = max(float(np.max(np.linalg.norm(p - pos[0], axis=1), initial=0.0)) * 1e3 for p in pos[1:])
        pairs = [(i, j) for i in range(len(pos)) for j in range(i + 1, len(pos))]
        w1 = [w1_mm(pos[i], pos[j]) for i, j in pairs]
        key = ("S01", cfg.name, dev, steps)
        if b.nworld == 1:
            _CACHE[key] = runs[0]
        ref = _CACHE.get(key)
        bi_mm = bi_ulp = None
        if ref is not None and b.nworld > 1:
            bi_mm = float(np.max(np.linalg.norm(positions(runs[0]) - positions(ref), axis=1), initial=0.0)) * 1e3
            bi_ulp = max_ulp(_q_only(ref), _q_only(runs[0]))
        elif b.nworld == 1:
            bi_mm, bi_ulp = 0.0, 0
        cpu = not wp.get_device(dev).is_cuda
        must_d1 = cpu or det
        d1 = (ulp == 0) if must_d1 else None
        d2 = (bi_mm == 0.0) if (det and bi_mm is not None) else None
        d3 = (max(w1) == 0.0) if must_d1 else None
        checks = [c for c in (d1, d2, d3) if c is not None]
        rec["accuracy_error"] = max(rerun_max_mm, bi_mm or 0.0)
        rec["metrics"].update(
            scene=cfg.params["scene"], det_mode=det, det_pipeline=bool(cfg.params.get("det_pipe", det)),
            det_solver=bool(cfg.params.get("det_solver", det)) and cfg.solver not in ("kamino", "implicit_mpm"),
            det_switch=cfg.solver not in NO_DET_SWITCH, reruns=self.RERUNS,
            max_ulp=ulp, max_ulp_positions=ulp_q, rerun_identical=ulp == 0, rerun_max_mm=rerun_max_mm,
            rerun_w1_mm=float(np.mean(w1)),
            rerun_w1_max_mm=float(np.max(w1)), batch_invariance_mm=bi_mm, batch_invariance_ulp=bi_ulp,
            batch_ref_available=ref is not None, main_vs_rerun_ulp=max_ulp(main, runs[0]),
            d1_pass=d1, d2_pass=d2, d3_pass=d3, law_pass=all(checks) if checks else None,
            w1_method="assignment" if _has_scipy() else "identity (scipy absent)")


def _q_only(snap: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pose arrays only: a ULP distance across zero (velocities near rest) is huge and says nothing physical."""
    return {k: v for k, v in snap.items() if k in ("body_q", "particle_q")}


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


# ---- S02 sleeping -----------------------------------------------------------------------------------------
MJ_SLEEP_KW = {"iterations": 10, "ls_iterations": 10, "jacobian": "sparse", "njmax": 4096, "nconmax": 2048,
               "update_data_interval": 0, "use_mujoco_contacts": True}
KICK_FRAC = {"towers": 0.65,  # the block is thrown at this fraction of the horizon (example: frame 300 of ~420)
             "ant_stacks": 0.5}  # from here on, each world kicks its first sleeping ant (first sleep ~5.7 s on CPU)


def _s02_configs() -> tuple[Config, ...]:
    # MuJoCo runs non-det: RUN_TO_RUN sizes per-thread records by nv^2/2 when the model has actuators (nv 281 here),
    # a 5.3 GB allocation that fails on an L4 (solver_mujoco.py _mujoco_warp_deterministic_max_records)
    mj = lambda name, **kw: Config(name, "mujoco", dt=1 / 60, substeps=5, cuda_graph=True,  # noqa: E731
                                   params={"scene": "ant_stacks", "det": False, "sleep": bool(kw),
                                           "kw": {**MJ_SLEEP_KW, **kw}})
    st = lambda name, **kw: Config(name, "steel", dt=1 / 60, substeps=4, cuda_graph=True,  # noqa: E731
                                   params={"scene": "towers", "det": True, "sleep": bool(kw),
                                           "kw": {"articulated_contact_response": "immediate", **kw}})
    return (mj("mujoco_ants_off"),
            mj("mujoco_ants_on", enable_sleeping=True, sleep_tolerance=0.05),
            mj("mujoco_ants_on_tol0p1", enable_sleeping=True, sleep_tolerance=0.005),
            st("steel_towers_off"),
            st("steel_towers_on", enable_sleeping=True),
            st("steel_towers_on_tol0p1", enable_sleeping=True, sleep_linear_threshold=0.005,
               sleep_angular_threshold=0.015))


class Sleeping(ScaleBase):
    """Sleeping on vs off: speedup, awake-body deviation, wake latency after a kick, false sleeps.

    The off config of each scene runs first and leaves its ms/step and per-sample poses and speeds (first
    REF_WORLDS worlds) in _CACHE. Steel runs det mode so off/on differ only by sleeping; MuJoCo cannot (see
    _s02_configs), so on GPU its deviations include run-to-run noise.
    The law value awake_dev_mm is the deviation of the witness pendulum (its own island, never asleep): the stacks
    and towers are chaotic once they topple or are hit, so their never-slept bodies (awake_dev_never_slept_mm)
    amplify any ULP-level change; presleep_dev_mm is the deviation of all bodies before the first sleep.
    Steel: the block is thrown into tower 1 at KICK_FRAC of the horizon; latency = first step any tower-1 body is
    awake minus first step the block touches tower 1 (contact gap <= 1 mm). MuJoCo: at the kick step the first
    sleeping ant of each world gets an upward KICK_DV_ANT on its torso; latency = first awake step minus the kick
    step. False sleeps: asleep while the off twin moves > 0.1 m/s and the twins still agree to 1 mm, or asleep
    while moving > 0.1 m/s itself. diverge_step_0p1mm is the first sample at which any body is 0.1 mm off."""

    id = "S02_sleeping"
    title = "Sleeping on/off: speedup, awake deviation, wake latency, false sleep"
    horizon_s = 8.0
    ladder = (1, 16, 256, 1024)
    sample_every = 6
    accuracy_unit = "mm"
    accuracy_ref = "awake-body deviation vs sleeping off <= 0.1 mm; wake <= 1 step; no false sleep"
    configs = _s02_configs()

    def extend(self, b: Built, info: SceneInfo, config: Config) -> None:
        m, x, dev, nw = b.model, b.extra["x"], b.model.device, b.nworld
        bpw = m.body_count // nw
        b.extra.update(bpw=bpw, sel=np.arange(min(nw, REF_WORLDS) * bpw), samples={}, ever_slept=None,
                       dev_never=0.0, dev_witness=0.0, dev_presleep=0.0, first_sleep_step=None, false_sleep=0,
                       false_sleep_own=0, asleep_frac=[], kick_step_host=None, dev_trace=[])
        kick = b.extra["kick_step"] = wp.array([NEVER], dtype=wp.int32, device=dev)
        b.extra["asleep_at_kick"] = wp.zeros(nw, dtype=wp.int32, device=dev)
        b.extra["first_wake"] = wp.full(nw, NEVER, dtype=wp.int32, device=dev)
        h = config.dt / config.substeps
        if config.params["scene"] == "towers":
            self._towers(b, info, x, kick, bpw, h)
        else:
            self._ants(b, info, x, kick, bpw, h)

    def _towers(self, b: Built, info: SceneInfo, x: StepExtras, kick, bpw: int, h: float) -> None:
        m, dev, nw = b.model, b.model.device, b.nworld
        tower = info.meta["tower_of_body"]
        block = info.meta["block"]
        mass = float(m.body_mass.numpy()[block])
        force = wp.spatial_vector(*(mass * np.asarray(KICK_VELOCITY) / h), 0.0, 0.0, 0.0)
        blocks = wp.array([w * bpw + block for w in range(nw)], dtype=wp.int32, device=dev)
        x.on_substep0.append((_k_kick, nw, [x.counter, kick, blocks, force, b.state_0.body_f]))
        b.extra["kick_bodies"] = blocks
        local = np.arange(m.body_count) % bpw
        role_body = np.where(local == block, 1, np.where(tower[local] == TARGET_TOWER, 2, 0))
        sb = m.shape_body.numpy()
        role = np.where(sb >= 0, role_body[np.maximum(sb, 0)], 0).astype(np.int32)
        b.extra["first_touch"] = wp.full(nw, NEVER, dtype=wp.int32, device=dev)
        c = b.contacts
        x.post.append((_k_touch, c.rigid_contact_max, [
            x.counter, kick, c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1,
            c.rigid_contact_point0, c.rigid_contact_point1, c.rigid_contact_normal, c.rigid_contact_margin0,
            c.rigid_contact_margin1, m.shape_body, wp.array(role, dtype=wp.int32, device=dev), m.shape_world,
            None, 1.0e-3, b.extra["first_touch"]]))
        b.extra["touch_item"] = x.post[-1]
        if b.extra["config"].params["sleep"]:
            kind = newton.solvers.SolverSteel.ObservableKind.BODY_AWAKE
            obs = b.solver.observables({kind})
            x.step_kw["observables"] = obs
            b.extra["awake_arr"] = obs.body_awake
            group = np.array([[w * bpw + i for i in np.flatnonzero(tower == TARGET_TOWER)] for w in range(nw)])
            x.post.append((_k_wake_bodies, nw, [x.counter, kick, obs.body_awake, wp.array(group, dtype=wp.int32,
                           device=dev), b.extra["asleep_at_kick"], b.extra["first_wake"]]))

    def _ants(self, b: Built, info: SceneInfo, x: StepExtras, kick, bpw: int, h: float) -> None:
        """Kick the first sleeping ant of each world (chosen on the device at the kick step) and time its wake-up."""
        m, dev, nw, s = b.model, b.model.device, b.nworld, b.solver
        mjm, n2n = s.mj_model, s.mjc_body_to_newton.numpy()
        treeid, parent = np.asarray(mjm.body_treeid), np.asarray(mjm.body_parentid)
        tree = np.full(m.body_count, -1, dtype=np.int32)
        for world_map in n2n:
            for mj_body, nb in enumerate(world_map):
                if nb >= 0:
                    tree[nb] = treeid[mj_body]
        b.extra["body_tree"] = tree
        if not s.enable_sleeping:
            return
        roots = [(int(treeid[k]), k) for k in range(mjm.nbody) if parent[k] == 0 and treeid[k] >= 0]
        tree_root = np.full((nw, int(mjm.ntree)), -1, dtype=np.int32)
        for w in range(nw):
            for t, mj_body in roots:
                nb = int(n2n[w % n2n.shape[0], mj_body])
                if nb >= 0 and nb % bpw != info.meta["witness"]:
                    tree_root[w, t] = nb
        chosen = b.extra["kicked_tree"] = wp.full(nw, -1, dtype=wp.int32, device=dev)
        kicked_at = b.extra["kicked_at"] = wp.full(nw, NEVER, dtype=wp.int32, device=dev)
        x.on_substep0.append((_k_kick_sleeper, nw, [x.counter, kick, s.mjw_data.tree_asleep,
                                                    wp.array(tree_root, dtype=wp.int32, device=dev), m.body_mass,
                                                    KICK_DV_ANT, h, chosen, kicked_at, b.state_0.body_f]))
        x.post.append((_k_wake_chosen, nw, [x.counter, kicked_at, s.mjw_data.tree_asleep, chosen,
                                            b.extra["first_wake"]]))

    def _substep(self, b: Built, dt: float, k: int, x: StepExtras) -> None:
        # kick and touch kernels hold state references that the ping-pong swap changes; refresh them
        for kernel, _dim, inputs in x.on_substep0:
            inputs[-1] = b.state_0.body_f
        super()._substep(b, dt, k, x)
        item = b.extra.get("touch_item")
        if item is not None:
            item[2][13] = b.state_0.body_q

    def on_start(self, b: Built, rec: dict) -> None:
        cfg = b.extra["config"]
        total = b.extra["total_steps"] = int(round(rec["horizon_s"] / cfg.dt))
        ks = max(int(KICK_FRAC[cfg.params["scene"]] * total), 3)
        b.extra["kick_step_host"] = ks
        b.extra["kick_step"].assign(np.array([ks], dtype=np.int32))

    def _awake(self, b: Built) -> np.ndarray | None:
        arr = b.extra.get("awake_arr")
        if arr is not None:
            return arr.numpy().astype(bool)
        s = b.solver
        if isinstance(s, newton.solvers.SolverMuJoCo) and s.enable_sleeping:
            ta, tree = s.mjw_data.tree_asleep.numpy(), b.extra["body_tree"]
            bw = b.model.body_world.numpy()
            return np.where(tree >= 0, ta[np.maximum(bw, 0), np.maximum(tree, 0)] < 0, True)
        return None

    def measure(self, b: Built, t: float, rec: dict) -> None:
        super().measure(b, t, rec)
        step = int(round(t / b.extra["config"].dt))
        sel, ex = b.extra["sel"], b.extra
        pos = b.state_0.body_q.numpy()[sel, :3].astype(np.float64)
        speed = np.linalg.norm(b.state_0.body_qd.numpy()[sel, :3], axis=1)
        if not ex["config"].params["sleep"]:
            ex["samples"][step] = (pos, speed)
            return
        awake = self._awake(b)
        aw = awake[sel] if awake is not None else np.ones(len(sel), bool)
        ex["ever_slept"] = ~aw if ex["ever_slept"] is None else (ex["ever_slept"] | ~aw)
        ex["asleep_frac"].append(float(1.0 - aw.mean()))
        ex["false_sleep_own"] += int(np.sum(~aw & (speed > FALSE_SLEEP_V)))
        if ex["first_sleep_step"] is None and ex["ever_slept"].any():
            ex["first_sleep_step"] = step
        ref = _CACHE.get(self._key(b))
        if ref is None or step not in ref["samples"]:
            return
        rpos, rspeed = ref["samples"][step]
        dev_mm = np.linalg.norm(pos - rpos, axis=1) * 1e3
        witness = (sel % ex["bpw"]) == ex["info"].meta["witness"]
        ex["dev_witness"] = max(ex["dev_witness"], float(dev_mm[witness].max()))
        never = aw & ~ex["ever_slept"]
        if never.any():
            ex["dev_never"] = max(ex["dev_never"], float(dev_mm[never].max()))
        if ex["first_sleep_step"] is None:
            ex["dev_presleep"] = max(ex["dev_presleep"], float(dev_mm.max()))
        ex["dev_trace"].append((step, float(dev_mm.max())))
        # the off twin's speed is a fair reference only while the two runs still agree
        ex["false_sleep"] += int(np.sum(~aw & (rspeed > FALSE_SLEEP_V) & (dev_mm <= TWIN_AGREE_MM)))

    def _key(self, b: Built) -> tuple:
        cfg = b.extra["config"]
        return ("S02", cfg.params["scene"], b.nworld, b.extra["device"], cfg.dt, cfg.substeps, b.extra["total_steps"])

    def finish(self, b: Built, rec: dict) -> None:
        self.capacity_metrics(b, rec)
        cfg = b.extra["config"]
        sleep = bool(cfg.params["sleep"])
        ks = b.extra["kick_step_host"]
        mt = rec["metrics"]
        mt.update(scene=cfg.params["scene"], sleeping=sleep, kick_step=ks, kick_reached=rec["steps"] > (ks or 0))
        if not sleep:
            _CACHE[self._key(b)] = {"ms": rec["ms_per_step"], "samples": b.extra["samples"]}
            mt.update(note="sleeping-off reference")
            return
        ref = _CACHE.get(self._key(b))
        first_wake = b.extra["first_wake"].numpy()
        asleep_at_kick = b.extra["asleep_at_kick"].numpy().astype(bool)
        # latency counts only worlds whose watched group was asleep just before the kick (else nothing to wake)
        if cfg.params["scene"] == "towers":
            touch = b.extra["first_touch"].numpy()
            ok = asleep_at_kick & (first_wake < NEVER) & (touch < NEVER)
            lat = (first_wake - touch)[ok]
        else:  # MuJoCo: a world is tested once its kick found a sleeping ant (kicked_at < NEVER)
            kicked_at = b.extra["kicked_at"].numpy() if "kicked_at" in b.extra else np.full(len(first_wake), NEVER)
            asleep_at_kick = kicked_at < NEVER
            ok = asleep_at_kick & (first_wake < NEVER)
            lat = (first_wake - kicked_at)[ok]
            mt.update(kick_steps=sorted(set(kicked_at[asleep_at_kick].tolist()))[:8])
        latency = int(lat.max()) if lat.size else None
        never_woke = int(np.sum(asleep_at_kick & (first_wake >= NEVER)))  # asleep at kick and never woke: a failure
        speedup = ref["ms"] / rec["ms_per_step"] if ref and ref["ms"] and rec["ms_per_step"] else None
        ex = b.extra
        dev = ex["dev_witness"] if ref else None
        wake_ok = None if not asleep_at_kick.any() else (never_woke == 0 and latency is not None and latency <= 1)
        checks = [dev <= AWAKE_DEV_TOL_MM if dev is not None else None, wake_ok,
                  (ex["false_sleep"] == 0 and ex["false_sleep_own"] == 0) if ref else None]
        checks = [c for c in checks if c is not None]
        rec["accuracy_error"] = dev
        mt.update(speedup=speedup, awake_dev_mm=dev, awake_dev_never_slept_mm=ex["dev_never"] if ref else None,
                  presleep_dev_mm=ex["dev_presleep"] if ref else None, first_sleep_step=ex["first_sleep_step"],
                  dev_first_sample_mm=ex["dev_trace"][0][1] if ex["dev_trace"] else None,
                  diverge_step_0p1mm=next((st for st, d in ex["dev_trace"] if d > AWAKE_DEV_TOL_MM), None),
                  wake_latency_steps=latency, woken_worlds=int(ok.sum()),
                  asleep_at_kick_worlds=int(asleep_at_kick.sum()),
                  never_woke_worlds=never_woke, wake_tested=bool(asleep_at_kick.any()), wake_pass=wake_ok,
                  false_sleep_count=ex["false_sleep"] if ref else None, false_sleep_own_speed=ex["false_sleep_own"],
                  asleep_frac_max=max(ex["asleep_frac"], default=0.0), ref_available=ref is not None,
                  law_pass=all(checks) if checks else None)


# ---- S03 throughput ---------------------------------------------------------------------------------------
S03_SCENES = ("ant", "humanoid", "franka_cube", "bin64", "hetero")
S03_SOLVERS = ("mujoco", "featherstone", "kamino", "xpbd", "steel")


class Throughput(ScaleBase):
    """World-count ladder per scene x solver with CUDA graphs on and off. The graph-on config of each pair runs
    first; the graph-off record carries graph_speedup, launch_overhead_ms and the on/off state comparison.

    Graph law, at the same nworld and horizon: on/off bitwise passes. Otherwise one more eager run of the graph-off
    config gives the run-to-run floor (eager_rerun_mm). If the two eager runs are bitwise, the graph changed the
    arithmetic: fail. Else pass when graph_diff_mm <= GRAPH_FLOOR_FACTOR x eager_rerun_mm (an order-of-magnitude
    test: one sample of each spread). The S01 floor for the scene is reported for reference (s01_noise_floor_mm)."""

    id = "S03_throughput"
    title = "Throughput ladder, graph on/off, heterogeneous worlds"
    horizon_s = 1.0
    ladder = DEFAULT_LADDER
    accuracy_unit = "mm"
    accuracy_ref = "graph on/off state: bitwise, or within 10x the same-horizon eager rerun spread"
    configs = tuple(_cfg(f"{sc}_{so}_{'graph' if g else 'nograph'}", so, sc, cuda_graph=g)
                    for sc in S03_SCENES for so in S03_SOLVERS for g in (True, False))
    GRAPH_FLOOR_FACTOR = 10.0

    def finish(self, b: Built, rec: dict) -> None:
        self.capacity_metrics(b, rec)
        cfg, dev = b.extra["config"], b.extra["device"]
        scene, mt = cfg.params["scene"], rec["metrics"]
        eff_key = ("S03eff", cfg.name, dev)
        if b.nworld == 1:
            _CACHE[eff_key] = rec["world_steps_per_s"]
        base = _CACHE.get(eff_key)
        mt.update(scene=scene, heterogeneous=scene == "hetero" and b.nworld >= 2,
                  scaling_efficiency=(rec["world_steps_per_s"] / (b.nworld * base))
                  if base and rec["world_steps_per_s"] else None,
                  world_substeps_per_s=rec["world_steps_per_s"] * cfg.substeps if rec["world_steps_per_s"] else None)
        ref = pa_reference(scene, cfg.solver, rec["device_name"], b.nworld)
        if ref is not None:
            mt.update(pa_ref=ref, pa_ratio=(mt["world_substeps_per_s"] / ref["world_steps_per_s"])
                      if mt["world_substeps_per_s"] else None)
        key = ("S03", scene, cfg.solver, b.nworld, dev)
        arrays = self.state_arrays(b)
        if cfg.cuda_graph:
            _CACHE[key] = {"ms": rec["ms_per_step"], "arrays": arrays, "steps": rec["steps"],
                           "graph": rec["cuda_graph"]}
            return
        on = _CACHE.pop(key, None)
        if on is None:  # graph-on rung missing (run alone, or the harness's repeat run of this config)
            mt.update(graph_ref_available=False)
            return
        mt.update(graph_ref_available=True, graph_was_replayed=on["graph"],
                  graph_speedup=(rec["ms_per_step"] / on["ms"]) if on["ms"] and rec["ms_per_step"] else None,
                  launch_overhead_ms=(rec["ms_per_step"] - on["ms"]) if on["ms"] and rec["ms_per_step"] else None,
                  s01_noise_floor_mm=s01_noise_floor_mm(scene, cfg.solver, rec["device_name"]))
        self._graph_law(b, rec, on, arrays)

    def _graph_law(self, b: Built, rec: dict, on: dict, arrays: dict[str, np.ndarray]) -> None:
        """Graph on vs off at the same horizon, judged against one more eager run (see the class docstring)."""
        cfg, mt = b.extra["config"], rec["metrics"]
        if on["steps"] != rec["steps"]:  # a budget cut one run short: nothing comparable
            mt.update(graph_bitwise=None, graph_diff_mm=None, law_pass=None, note="on/off step counts differ")
            return
        diff = _max_pos_diff_mm(on["arrays"], arrays)
        bitwise = _bitwise(on["arrays"], arrays)
        rec["accuracy_error"] = diff
        mt.update(graph_bitwise=bitwise, graph_diff_mm=diff)
        if bitwise:
            mt.update(law_pass=True)
            return
        try:  # the rerun holds a second model: at the top rungs it can run out of memory, which is recorded
            rerun = self.rollout(cfg, b.nworld, b.extra["device"], rec["steps"], self.state_arrays)
        except RuntimeError as exc:
            mt.update(eager_rerun_error=str(exc)[:300], law_pass=None)
            return
        eager_bitwise = _bitwise(rerun, arrays)
        floor = _max_pos_diff_mm(rerun, arrays)
        mt.update(eager_rerun_bitwise=eager_bitwise, eager_rerun_mm=floor,
                  graph_diff_over_floor=diff / floor if floor > 0 else None,
                  law_pass=(not eager_bitwise) and diff <= self.GRAPH_FLOOR_FACTOR * floor)


def _bitwise(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(v.view(np.uint8), b[k].view(np.uint8)) for k, v in a.items())


def _max_pos_diff_mm(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    worst = 0.0
    for k in ("body_q", "particle_q"):
        if k in a and k in b and a[k].size:
            worst = max(worst, float(np.max(np.abs(a[k][..., :3].astype(np.float64) - b[k][..., :3]))) * 1e3)
    return worst


# Earlier-study rows (pa/bench/results/newton_bin.csv, trace r5_bin50: 50 mixed bodies, dt 2 ms, 1 substep), copied
# so the lookup works on any VM: (solver, GPU, nworld) -> (solver steps per s summed over worlds, pa config).
PA_BIN50 = {
    ("mujoco", "NVIDIA L4", 1): (182.48, "newton_default"), ("mujoco", "NVIDIA L4", 64): (6521.89, "newton_default"),
    ("mujoco", "NVIDIA L4", 1024): (14541.3, "newton_default"), ("xpbd", "NVIDIA L4", 1): (3435.49, "newton_default"),
    ("xpbd", "NVIDIA L4", 64): (93110.8, "newton_default"), ("xpbd", "NVIDIA L4", 1024): (83522.8, "newton_default"),
    ("steel", "NVIDIA L4", 1): (446.656, "newton_default"), ("steel", "NVIDIA L4", 64): (23491.2, "newton_default"),
    ("steel", "NVIDIA L4", 1024): (50387.6, "newton_default"), ("kamino", "NVIDIA L4", 1): (614.532, "newton_default"),
    ("kamino", "NVIDIA L4", 64): (19515.8, "newton_default"), ("kamino", "NVIDIA L4", 1024): (27838.7, "newton_default"),
    ("xpbd", "NVIDIA H100 80GB HBM3", 1): (3686.22, "gap5mm_cap24"),
    ("xpbd", "NVIDIA H100 80GB HBM3", 64): (167319.0, "gap5mm_cap24"),
    ("xpbd", "NVIDIA H100 80GB HBM3", 1024): (831998.0, "gap5mm_cap24"),
    ("kamino", "NVIDIA H100 80GB HBM3", 1): (597.691, "gap5mm_cap24"),
    ("kamino", "NVIDIA H100 80GB HBM3", 64): (27653.5, "gap5mm_cap24"),
}


def pa_reference(scene: str, solver: str, device_name: str | None, nworld: int) -> dict | None:
    """Earlier-study throughput for the bin scene at the same solver, GPU and nworld (50 bodies vs our 64, dt 2 ms
    single step vs our 1/600 s substeps). Its rate counts solver steps, so compare it with world_substeps_per_s."""
    row = PA_BIN50.get((solver, device_name, nworld)) if scene == "bin64" else None
    if row is None:
        return None
    return {"source": f"pa/bench/results/newton_bin.csv r5_bin50 {row[1]}", "nworld": nworld,
            "world_steps_per_s": row[0], "dt": 0.002}


def s01_noise_floor_mm(scene: str, solver: str, device_name: str | None) -> float | None:
    """Largest S01 rerun spread (nondet mode) recorded for this scene/solver on this device, or None."""
    path = RESULTS / "S01_determinism.jsonl"
    if not path.exists():
        return None
    vals = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        m = r.get("metrics", {})
        if (r.get("status") == "ok" and r.get("solver") == solver and m.get("scene") == scene
                and not m.get("det_mode") and r.get("device_name") == device_name
                and m.get("rerun_max_mm") is not None):
            vals.append(m["rerun_max_mm"])
    return max(vals) if vals else None


# ---- S04 memory and capacity ------------------------------------------------------------------------------
S04_KW = {"mujoco": {"use_mujoco_contacts": True, "solver": "newton", "cone": "elliptic", "integrator": "implicitfast",
                     "iterations": 50, "ls_iterations": 50}}


def _s04_configs() -> tuple[Config, ...]:
    out = []
    for scene, solver in (("bin64", "mujoco"), ("bin64", "kamino"), ("bin64", "xpbd"), ("bin64", "steel"),
                          ("cloth32", "vbd"), ("mpm10k", "implicit_mpm")):
        for cap in ("default", "quarter"):
            dt, sub = (1 / 120, 1) if scene == "mpm10k" else (1 / 60, 10)
            out.append(_cfg(f"{scene}_{solver}_cap_{cap}", solver, scene, dt=dt, substeps=sub, cap=cap,
                            kw=S04_KW.get(solver, {}), cuda_graph=scene != "mpm10k"))
    return tuple(out)


class MemoryCapacity(ScaleBase):
    """Memory vs worlds (fit mem = a + b n, R^2 >= 0.99; max worlds on an L4 / H100) and contact capacity at Newton's
    default vs 0.25x the rows the default run needed. Memory is the harness's mempool growth on CUDA and a census of
    owning Warp arrays everywhere (alloc_mb). Overflow must be detected; a silent drop is severity S1."""

    id = "S04_memory_capacity"
    title = "Memory per world, max worlds, contact-capacity overflow detection"
    horizon_s = 1.0
    ladder = (1, 16, 256, 1024, 4096, 8192, 16384, 32768, 65536)
    accuracy_unit = "R2"
    accuracy_ref = "mem = a + b*nworld, R^2 >= 0.99; overflow detected (silent drop = S1)"
    configs = _s04_configs()

    def guard(self, b: Built, fn: Callable[[], Any]) -> None:
        """An engine exception about capacity (ImplicitMPM's sparse-grid rebuild check raises) is the engine
        detecting an overflow: record it, halt the run (later steps are no-ops) and let finish() report it."""
        try:
            fn()
        except RuntimeError as exc:
            if "capacity" not in str(exc):
                raise
            x = b.extra.get("x")
            step = int(x.counter.numpy()[0]) if x is not None else "project_outside"
            b.extra.update(halted=True, capacity_exception=str(exc)[:300], halted_at_step=step)

    def build(self, nworld: int, device: str, config: Config) -> Built:
        pool0 = _pool_mb(device)
        b = self._build_or_standin(nworld, device, config)
        b.extra["pool0_mb"] = pool0
        return b

    def _build_or_standin(self, nworld: int, device: str, config: Config) -> Built:
        """ImplicitMPM checks its sparse-grid capacity in the constructor, so 0.25x the occupied cells raises before
        a step. That is a detected overflow: rebuild with the default capacity only to get a record (no stepping,
        no memory point) and report the exception."""
        try:
            return super().build(nworld, device, config)
        except RuntimeError as exc:
            if "capacity" not in str(exc) or config.params.get("cap") != "quarter":
                raise
            kw, pipe_kw = _capacities(SceneInfo(newton.ModelBuilder()), config.params, nworld, config.solver, device)
            cap_kw = {**kw, **pipe_kw}
            b = super().build(nworld, device, replace(config, params={**config.params, "cap": "default"}))
            b.extra.update(config=config, cap_kw=cap_kw, halted=True, capacity_exception=str(exc)[:300],
                           halted_at_step="solver_init")
            return b

    def capacity_metrics(self, b: Built, rec: dict) -> dict:
        if isinstance(b.solver, newton.solvers.SolverImplicitMPM) and not b.extra.get("halted"):
            self.guard(b, b.solver.check_sparse_grid_rebuild_status)  # after graph replay (no check inside graphs)
        rep = super().capacity_metrics(b, rec)
        exc = b.extra.get("capacity_exception")
        if exc is not None:  # an exception is a loud, detected overflow; its row count is unknown (>= 1)
            rep["overflow_signals"]["capacity_exception"] = exc
            rep["contact_overflow"] = max(1, rep["contact_overflow"])
            rep.update(overflow_detected=True, silent_drop=False, halted_at_step=b.extra["halted_at_step"])
            rec["metrics"].update(rep)
            rec["metrics"].pop("severity", None)
        return rep

    def on_start(self, b: Built, rec: dict) -> None:
        """Memory point and fit after warm-up and graph capture (every lazy buffer exists by then). Done here, not in
        finish(), so a rung that later goes non-finite (the harness then skips finish()) still counts.
        mem_mb = Warp mempool growth since build() on CUDA, else the census of owning Warp arrays (alloc_mb)."""
        cfg, dev, mt = b.extra["config"], b.extra["device"], rec["metrics"]
        alloc = array_census_mb(b.model, b.solver, b.state_0, b.state_1, b.control, b.contacts, b.pipeline)
        pool, pool0 = _pool_mb(dev), b.extra.get("pool0_mb")
        on_pool = pool is not None and pool0 is not None
        mem = pool - pool0 if on_pool else alloc
        pts = _CACHE.setdefault(("S04mem", cfg.name, dev), {})
        if b.extra.get("halted_at_step") != "solver_init":  # that rung holds the default-capacity stand-in solver
            pts[b.nworld] = mem
        mt.update(scene=cfg.params["scene"], cap_mode=cfg.params["cap"], alloc_mb=alloc, mem_mb=mem,
                  mem_basis="gpu_mempool" if on_pool else "array_census", mem_points=dict(sorted(pts.items())))
        if len(pts) >= 3:
            n, y = np.array(sorted(pts), float), np.array([pts[k] for k in sorted(pts)])
            a, slope, r2 = linear_fit(n, y)
            mt.update(mb_intercept=a, mb_per_world=slope, r2=r2, linear_pass=r2 >= 0.99,
                      max_nworld_l4=int((L4_CAP_MB - a) / slope) if slope > 0 else None,
                      max_nworld_h100=int((H100_CAP_MB - a) / slope) if slope > 0 else None)
            rec["accuracy_error"] = r2

    def finish(self, b: Built, rec: dict) -> None:
        cfg, dev, mt = b.extra["config"], b.extra["device"], rec["metrics"]
        rep = self.capacity_metrics(b, rec)
        if cfg.params["cap"] == "default":
            mt["needed_per_world"] = self._store_need(b, rep, cfg, dev)
        # Law: any dropped row must carry an engine signal. The quarter config only tests that when rows were
        # actually dropped (overflow_exercised); a quarter run without a drop leaves the law untested (None).
        expect_overflow = cfg.params["cap"] == "quarter"
        exercised = rep["contact_overflow"] > 0
        overflow_law = (not rep["silent_drop"]) if (exercised or not expect_overflow) else None
        checks = [c for c in (overflow_law, mt.get("linear_pass")) if c is not None]
        mt.update(overflow_expected=expect_overflow, overflow_exercised=exercised, overflow_law_pass=overflow_law,
                  law_pass=all(checks) if checks else None)

    def _store_need(self, b: Built, rep: dict, cfg: Config, dev: str) -> dict:
        bufs, nw = rep["capacity_buffers"], b.nworld
        need = dict(NEED_FALLBACK)
        if "pipeline_rigid" in bufs:
            need["rigid"] = max(1, math.ceil(bufs["pipeline_rigid"]["peak"] / nw))
        if "mujoco_nacon" in bufs:
            need["nacon"] = max(1, math.ceil(bufs["mujoco_nacon"]["peak"] / nw))
        if "mujoco_nefc_per_world" in bufs:
            need["nefc"] = max(1, bufs["mujoco_nefc_per_world"]["peak"])
        if "vbd_vertex_tri" in bufs:
            need["vertex"] = max(1, bufs["vbd_vertex_tri"]["peak"])
        if "vbd_edge_edge" in bufs:
            need["edge"] = max(1, bufs["vbd_edge_edge"]["peak"])
        if "pipeline_soft" in bufs:
            need["soft"] = max(1, math.ceil(bufs["pipeline_soft"]["peak"] / nw))
        if b.model.particle_count and isinstance(b.solver, newton.solvers.SolverImplicitMPM):
            voxel = float(cfg.params["kw"].get("voxel_size", 0.02))
            q = b.state_0.particle_q.numpy()[b.model.particle_world.numpy() == 0]
            need["cells"] = int(len(np.unique(np.floor(q / voxel).astype(np.int64), axis=0)))  # occupied voxels,
            # a lower bound on the active cells (the solver also activates neighbours), so 0.25x always overflows
        _CACHE[("S04need", cfg.params["scene"], cfg.solver, dev)] = need
        return need


SCENARIOS = [Determinism(), Sleeping(), Throughput(), MemoryCapacity()]
