# SPDX-License-Identifier: Apache-2.0
"""Integration test helpers (WP12): assemble scenes, fault injectors, per-world digests, result rows.

Injectors are step observers (``Simulation.add_step_observer``) that fire once at a given step index (eager runs:
they write device arrays from the host):

- ``EscalateOnce``: adds energy to the coupling watchdog's interface-work ledger of one world, so the next
  ``watch_kernel`` marks it ESCALATED (windowed injection > ``escalate_gain_per_window``) and the period rolls back;
- ``KickOnce``: sets one body's linear speed above ``SPEED_FAIL_MPS`` in the post-step state, so health marks the
  world INVALID (SPEED_FAIL) and latches it before the next step.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy.core.defaults import SPEED_FAIL_MPS

__all__ = ["EscalateOnce", "KickOnce", "assemble", "cup_on_sand", "dc5_like", "digests", "git_commit", "record",
           "world_body"]

ROD_SELF_TEST_PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


def assemble(card: Any, lib: Any, device: str, worlds: int) -> Any:
    """``SceneAssembler(lib, Router).assemble`` with the rod self-test stubbed to a pass (tests only)."""
    from alloy.route.router import Router  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    router = Router(rod_self_test=lambda rod, h: dict(ROD_SELF_TEST_PASS))
    return SceneAssembler(lib, router).assemble(card, device, world_count=worlds)


def built_from_model(model: Any, entries: Sequence[Any], *, control_dt: float, precision_pairs: Sequence[Any] = (),
                     coupler: Any = None) -> Any:
    """A ``BuiltScene`` around a raw Model (WP3's peg-hole, WP4's coupling scenes) with a one-material table."""
    from alloy.core.ids import digest_of  # noqa: PLC0415
    from alloy.scene.types import BuiltScene, CouplerChoice, RouteReport  # noqa: PLC0415
    from alloy.testing.fakes import _one_material_table  # noqa: PLC0415

    import dataclasses  # noqa: PLC0415

    n, dev = int(model.world_count), model.device
    table = dataclasses.replace(_one_material_table(model), model=model)   # WP2's pair kernel reads table.model
    return BuiltScene(model, list(entries), coupler or CouplerChoice("none", ()), {}, table,
                      tuple(precision_pairs), float(control_dt), n, 0, wp.zeros(n, dtype=wp.vec3d, device=dev), None,
                      {}, (), RouteReport.empty(), digest_of({"raw_model": model.body_count, "worlds": n}))


def entry_spec(model: Any, capability: str, target_dt: float, **solver_kwargs: Any) -> Any:
    """One entry owning every body, particle, joint and shape of ``model``."""
    from alloy.scene.types import EntrySpec  # noqa: PLC0415

    return EntrySpec(capability, capability, tuple(range(model.body_count)), tuple(range(model.particle_count)),
                     tuple(range(model.joint_count)), tuple(range(model.shape_count)), 1, float(target_dt),
                     solver_kwargs=dict(solver_kwargs))


PAIR_K = 1.0e5      # N/m, the dc5_like precision pair (sdf mode)
PAIR_GROUP = 1      # collision group of the pair shapes


def _dc5_world(jack_particles: bool, tray_group: int | None, weld_n: float | None = None) -> Any:
    """One world: plug ball on a static jack plate (the precision pair, filtered, group 1), a static cold plate with
    a 4x4 cloth pad above it (VBD), and optionally a free tray box (another rigid entry) in ``tray_group``. With
    ``weld_n`` the plug is welded to the world where it starts, breaking at that force [N] (WP10's bond; inf holds)."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.contact.shadow import set_flat_solimp  # noqa: PLC0415 - WP3

    def cfg(particles: bool, group: int = -1) -> Any:
        return newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=5e-4, margin=0.0, collision_group=group,
                                               has_particle_collision=particles)

    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    jack = b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.01,
                           cfg=cfg(jack_particles, PAIR_GROUP), label="jack")
    plug_body = b.add_body(xform=wp.transform((0.0, 0.0, 0.0102), wp.quat_identity()), label="plug")
    plug = b.add_shape_sphere(plug_body, radius=0.01, cfg=cfg(False, PAIR_GROUP), label="plug")
    if weld_n is not None:
        from alloy.topology.bonds import add_weld_bond  # noqa: PLC0415 - WP10

        add_weld_bond(b, plug_body, -1, break_force=weld_n)
    b.add_shape_collision_filter_pair(jack, plug)
    for shape in (jack, plug):
        set_flat_solimp(b, shape, 0.99)
    b.add_shape_box(-1, xform=wp.transform((0.3, 0.0, -0.01), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.01,
                    cfg=cfg(True), label="cold_plate")
    b.add_cloth_grid(pos=wp.vec3(0.27, -0.03, 0.01), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=4, dim_y=4,
                     cell_x=0.015, cell_y=0.015, mass=1.0e-3, tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.0e2, edge_ke=0.01,
                     edge_kd=1.0e-4, particle_radius=0.004, label="pad")
    if tray_group is not None:
        b.add_shape_box(b.add_body(xform=wp.transform((-0.3, 0.0, 0.05), wp.quat_identity()), label="tray"),
                        hx=0.03, hy=0.03, hz=0.01, cfg=cfg(True, tray_group), label="tray")
    return b


def dc5_like(worlds: int, device: str, *, jack_particles: bool = False, tray_group: int | None = None,
             control_dt: float = 0.01, weld_n: float | None = None) -> Any:
    """SCR-DC5-3's scene shape: one MuJoCo entry (``mujoco``: the plug) with a precision pair (plug / jack), a VBD
    entry (``pad``) whose particles reach only the cold plate, and with ``tray_group`` an XPBD entry (``tray``).
    ``jack_particles=True`` lets the pad's particles contact the jack (the refused case); ``weld_n`` welds the plug
    to the world (``_dc5_world``)."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.contact.types import PrecisionPairSpec  # noqa: PLC0415 - WP3
    from alloy.scene.types import CouplerChoice, EntrySpec  # noqa: PLC0415

    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    top.replicate(_dc5_world(jack_particles, tray_group, weld_n), worlds)
    top.color()
    model = top.finalize(device=device)
    labels = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
    shape_labels = [str(x).rsplit("/", 1)[-1] for x in model.shape_label]
    body_joint = {int(c): j for j, c in enumerate(model.joint_child.numpy())}

    def rigid(name: str, capability: str, **kw: Any) -> Any:
        bodies = tuple(i for i, x in enumerate(labels) if x == name)
        shapes = tuple(i for i, x in enumerate(shape_labels) if x == name)
        return EntrySpec(name if capability != "mujoco" else "mujoco", capability, bodies, (),
                         tuple(body_joint[b] for b in bodies), shapes, 1, 1.0e-3, solver_kwargs=kw)

    entries = [rigid("plug", "mujoco", integrator="euler", cone="elliptic", nconmax=32, njmax=128),
               EntrySpec("pad", "vbd", (), tuple(range(model.particle_count)), (), (), 1, 1.0e-3,
                         solver_kwargs={"iterations": 10})]
    if tray_group is not None:
        entries.append(rigid("tray", "xpbd"))
    start = model.shape_world_start.numpy()
    jack, plug = (shape_labels.index(x) - int(start[0]) for x in ("jack", "plug"))
    spec = PrecisionPairSpec(jack, plug, mode="sdf", pair_stiffness=PAIR_K)
    return built_from_model(model, entries, control_dt=control_dt, precision_pairs=(spec,),
                            coupler=CouplerChoice("proxy", ()))


def _job2_world(static_jar: bool) -> Any:
    """One world: a jar (box, free MuJoCo body unless ``static_jar``) with a lid (box, free body) on it, the precision
    pair (filtered, group 1, both shapes with particle collision), and a 2x2x2 block of liquid particles in the jar."""
    from alloy._vendor import newton  # noqa: PLC0415

    def cfg() -> Any:
        return newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=5e-4, margin=0.0,
                                               collision_group=PAIR_GROUP, has_particle_collision=True)

    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    jar_body = -1 if static_jar else b.add_body(xform=wp.transform((0.0, 0.0, 0.03), wp.quat_identity()), label="jar")
    jar_x = wp.transform((0.0, 0.0, 0.03) if static_jar else (0.0, 0.0, 0.0), wp.quat_identity())
    jar = b.add_shape_box(jar_body, xform=jar_x, hx=0.03, hy=0.03, hz=0.03, cfg=cfg(), label="jar")
    lid = b.add_shape_box(b.add_body(xform=wp.transform((0.0, 0.0, 0.065), wp.quat_identity()), label="lid"),
                          hx=0.03, hy=0.03, hz=0.005, cfg=cfg(), label="lid")
    b.add_shape_collision_filter_pair(jar, lid)
    for i, j, k in np.ndindex(2, 2, 2):
        b.add_particle(pos=wp.vec3(0.01 * i, 0.01 * j, 0.02 + 0.01 * k), vel=wp.vec3(0.0), mass=1.0e-3, radius=0.002)
    return b


def job2_like(worlds: int, device: str, *, liquid: str = "sph", static_jar: bool = False) -> Any:
    """SCR-JOB2-1's scene shape: one MuJoCo entry (jar and lid, the precision pair jar / lid) and a ``liquid`` entry
    (capability ``sph``, or ``vbd`` for the refused contrast) whose particles sit in the jar, coupled by the proxy
    (MuJoCo source). ``static_jar=True`` makes the jar a static shape, so it is not a proxied body."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.contact.types import PrecisionPairSpec  # noqa: PLC0415 - WP3
    from alloy.scene.types import CouplerChoice, EntrySpec  # noqa: PLC0415

    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    top.replicate(_job2_world(static_jar), worlds)
    model = top.finalize(device=device)
    shape_labels = [str(x).rsplit("/", 1)[-1] for x in model.shape_label]
    mj = EntrySpec("mujoco", "mujoco", tuple(range(model.body_count)), (), tuple(range(model.joint_count)),
                   tuple(i for i, x in enumerate(shape_labels) if x in ("jar", "lid")), 1, 1.0e-3,
                   solver_kwargs={"integrator": "euler", "cone": "elliptic", "nconmax": 32, "njmax": 128})
    water = EntrySpec("water", liquid, (), tuple(range(model.particle_count)), (), (), 1, 1.0e-3)
    start = model.shape_world_start.numpy()
    jar, lid = (shape_labels.index(x) - int(start[0]) for x in ("jar", "lid"))
    spec = PrecisionPairSpec(jar, lid, mode="sdf", pair_stiffness=PAIR_K)
    return built_from_model(model, [mj, water], control_dt=0.01, precision_pairs=(spec,),
                            coupler=CouplerChoice("proxy", ()))


def launch_body(model: Any, label: str, world: int, speed: float) -> int:
    """Give ``world``'s free body ``label`` an initial upward speed [m/s] in the model (free joint and body velocity,
    so every solver starts from it); returns its body index."""
    labels = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
    body = next(b for b, (x, w) in enumerate(zip(labels, model.body_world.numpy())) if x == label and w == world)
    joint = int(np.flatnonzero(model.joint_child.numpy() == body)[0])
    qd = model.joint_qd.numpy()
    qd[int(model.joint_qd_start.numpy()[joint]) + 2] = speed           # free joint: linear (x, y, z) first
    model.joint_qd.assign(qd)
    bqd = model.body_qd.numpy()
    bqd[body, 2] = speed
    model.body_qd.assign(bqd)
    return body


def _wall_world(top_z: float) -> Any:
    """One world: a plywood sheet (top at ``top_z``, particle collision) on a bench body welded to the world by a FIXED
    joint, a free jar + lid (the parts the SPH entry gets as proxies), a free cup (particle collision, not proxied),
    and a 5x5x3 block of SPH liquid 1 cm above the plywood, away from the jar."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.solvers.sph.build import add_fluid_block  # noqa: PLC0415 - WP9
    from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes  # noqa: PLC0415

    def cfg(group: int = PAIR_GROUP) -> Any:
        return newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=5e-4, margin=0.0, collision_group=group,
                                               has_particle_collision=True)

    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    register_custom_attributes(b)
    base = wp.transform((0.0, 0.0, -0.3), wp.quat_identity())
    bench = b.add_link(xform=base, label="bench")
    b.add_articulation([b.add_joint_fixed(-1, bench, parent_xform=base, label="bench_fix")])
    b.add_shape_box(bench, xform=wp.transform((0.0, 0.0, 0.3 + top_z - 0.006), wp.quat_identity()), hx=0.2, hy=0.2,
                    hz=0.006, cfg=cfg(-1), label="plywood")
    for name, z, hz in (("jar", 0.03, 0.03), ("lid", 0.065, 0.005)):
        b.add_shape_box(b.add_body(xform=wp.transform((0.1, 0.0, z), wp.quat_identity()), label=name), hx=0.03,
                        hy=0.03, hz=hz, cfg=cfg(), label=name)
    b.add_shape_box(b.add_body(xform=wp.transform((-0.1, 0.1, 0.02), wp.quat_identity()), label="cup"), hx=0.02,
                    hy=0.02, hz=0.02, cfg=cfg(), label="cup")
    add_fluid_block(b, DFSPHConfig(particle_radius=2.0e-3), (-0.09, -0.01, top_z + 0.01), (5, 5, 3))
    return b


def static_wall_like(worlds: int, device: str, *, top_z: float = 0.0) -> Any:
    """SCR-JOB2-5's scene shape: a MuJoCo entry owning a welded bench (plywood), a jar, a lid and a cup, and an SPH
    entry whose proxies are the jar and lid only (JOB2's ``proxy_parts_only`` pattern), so the plywood and the cup are
    hidden from it unless static shapes are walls."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.core.newton_compat import SolverCoupledProxy  # noqa: PLC0415
    from alloy.coupling.types import PairPolicy  # noqa: PLC0415 - WP4
    from alloy.scene.types import CouplerChoice, EntrySpec  # noqa: PLC0415
    from alloy.solvers.sph.config import register_custom_attributes  # noqa: PLC0415

    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    register_custom_attributes(top)
    top.replicate(_wall_world(top_z), worlds)
    model = top.finalize(device=device)
    labels = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
    mj = EntrySpec("mujoco", "mujoco", tuple(range(model.body_count)), (), tuple(range(model.joint_count)), (), 1,
                   1.0e-3, solver_kwargs={"integrator": "euler", "cone": "elliptic", "nconmax": 32, "njmax": 128})
    water = EntrySpec("water", "sph", (), tuple(range(model.particle_count)), (), (), 1, 1.0e-3)
    parts = [b for b, x in enumerate(labels) if x in ("jar", "lid")]
    proxy = SolverCoupledProxy.Proxy(source="mujoco", destination="water", bodies=parts, particles=[], mode="lagged",
                                     proxy_relaxation_mode="fixed", collision_pipeline=None, collide_interval=None)
    pair = PairPolicy("mujoco", "water", mode="lagged", proxy=proxy)
    return built_from_model(model, [mj, water], control_dt=0.01, coupler=CouplerChoice("proxy", (pair,)))


def cup_on_sand(worlds: int = 1, *, cup_half: float = 0.015) -> tuple[Any, Any]:
    """(card, lib) of CPL-3's cup on MPM sand: a 60 x 60 x 24 mm sand bed (phi 33 deg, the grain_mpm route at 5 mm
    voxels) on a table inside four static walls, and a rigid ABS cup (a box, 2 * ``cup_half`` wide, 28 g at 15 mm)
    1 mm above it. The cup is far lighter than the bed, so D5 calls the mujoco -> mpm pair strong."""
    import math  # noqa: PLC0415

    from alloy.cards.types import ParticleSource  # noqa: PLC0415
    from alloy.testing import route as T  # noqa: PLC0415
    from alloy.testing import scene as S  # noqa: PLC0415

    bed, depth, top = 0.03, 0.024, 0.02             # bed half-width, depth; the table's top [m]
    sand = T.obj("sand", ParticleSource(((-bed, -bed, 0.0), (bed, bed, depth)), 2.5e-3, grain_diameter=3e-4),
                 T.material("sand", density=1600.0, E=1e7, nu=0.3, friction_angle=math.radians(33.0)))
    wood = T.material("wood", **T.WOOD)
    sizes = [(0.005, bed + 0.01, 0.03)] * 2 + [(bed, 0.005, 0.03)] * 2
    xy = [(bed + 0.005, 0.0), (-bed - 0.005, 0.0), (0.0, bed + 0.005), (0.0, -bed - 0.005)]
    walls = [T.box(f"wall{i}", wood, *ext, motion="static") for i, ext in enumerate(sizes)]
    places = [S.place("table"), S.place("sand", z=top + 0.0005), S.place("cup", z=top + depth + cup_half + 0.001)]
    places += [S.place(f"wall{i}", x=x, y=y, z=top + 0.03) for i, (x, y) in enumerate(xy)]
    return S.scene([S.table(), sand, S.box("cup", hx=cup_half), *walls], places, worlds=worlds)


def digests(sim: Any) -> np.ndarray:
    """[W, 2] uint64 per-world digests of State + every extra array the simulation snapshots (WP5 digest)."""
    from alloy.precision.digest import world_digests  # noqa: PLC0415

    return world_digests(sim.model, sim.state, sim.state_arrays(), extra_world=sim.state_world_maps())


def world_body(model: Any, world: int, local: int = 0) -> int:
    """Global id of the ``local``-th body of ``world`` (Newton replicate layout)."""
    starts = model.body_world_start.numpy()
    return int(starts[world]) + int(local)


class EscalateOnce:
    """Step observer: at step ``at`` add ``joules`` to the watchdog interface work of ``world`` (once)."""

    def __init__(self, sim: Any, world: int, at: int, joules: float = 1.0e3) -> None:
        self.sim, self.world, self.at, self.joules = sim, int(world), int(at), float(joules)
        self.fired = False
        self._watchdog = sim.stepper._watchdog      # WP4 Scheduler internals (test only)

    def __call__(self, s0, s1, control, contacts, dt) -> None:
        if self.fired or self.sim.steps != self.at:
            return
        work = self._watchdog.interface_work.numpy()
        work[self.world] += self.joules
        self._watchdog.interface_work.assign(work)
        self.fired = True


class KickOnce:
    """Step observer: at step ``at`` give ``body`` a linear speed of ``speed`` m/s in the post-step state (once)."""

    def __init__(self, sim: Any, body: int, at: int, speed: float = 1.5 * SPEED_FAIL_MPS) -> None:
        self.sim, self.body, self.at, self.speed = sim, int(body), int(at), float(speed)
        self.fired = False

    def __call__(self, s0, s1, control, contacts, dt) -> None:
        if self.fired or self.sim.steps != self.at:
            return
        qd = s1.body_qd.numpy()
        qd[self.body, 0:3] = (self.speed, 0.0, 0.0)
        s1.body_qd.assign(qd)
        self.fired = True


def git_commit() -> str:
    """Short HEAD sha; on a GPU VM (rsynced tree, no .git) the ``.wp12_commit`` file written before the sync."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
        if out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    marker = Path(".wp12_commit")
    return marker.read_text().strip() if marker.exists() else "unknown"


def record(metric: str, value: float | None, status: str, command: str, **extra: Any) -> Path:
    """Append one result row (JSON line) to ``$ALLOY_RESULTS_DIR/wp12/results.jsonl`` (default bench/results)."""
    root = Path(os.environ.get("ALLOY_RESULTS_DIR", "bench/results")) / "wp12"
    root.mkdir(parents=True, exist_ok=True)
    row = {"metric": metric, "value": value, "status": status, "command": command, "commit": git_commit(),
           "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": wp.get_preferred_device().alias,
           **extra}
    path = root / "results.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=_jsonable) + "\n")
    return path


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, Sequence):
        return list(x)
    return str(x)
