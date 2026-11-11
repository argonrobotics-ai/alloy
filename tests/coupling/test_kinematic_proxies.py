# SPDX-License-Identifier: Apache-2.0
"""SCR-SC8-2 (CPU): a source body welded to the world stays immovable in the destination (kinematic proxy).

Scene per world: a 9 kg plywood sheet on a FIXED joint to the world (MuJoCo merges it into the world), a free box on
its corner, and a VBD cloth or a film resting on it. Newton's stock proxy coupler integrates the plywood as a free
body in the destination, and the soft body resting on it is pushed every exchange. The Scheduler proxies it with mass
and inertia 0: the soft body rests at the level of SC8's interim (welded masses zeroed in the model), the source model
is untouched, the MuJoCo rollout is unchanged, and a kinematic proxy that moves anyway turns its world INVALID."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverMuJoCo, SolverVBD
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.kinematic import GUARD_BITS, world_welded_bodies
from alloy.coupling.scheduler import Scheduler
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags
from alloy.solvers.film import FilmConfig, SolverFilm
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.rigs import grid_mesh

pytestmark = pytest.mark.compile_heavy

DT = 0.02 / 32                      # 32 exchanges per 20 ms period
PLY_Z, PLY_HZ = 0.1, 0.005          # plywood centre and half thickness [m]
PLY_TOP = PLY_Z + PLY_HZ
SOFT = ("vbd", "film")


def _bench_world(dest: str) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    ply = b.add_link(xform=wp.transform((0.0, 0.0, PLY_Z), wp.quat_identity()), label="plywood")
    cfg = b.default_shape_cfg.copy()
    cfg.density = 9.0 / (0.3 * 0.3 * 2.0 * PLY_HZ)                       # 9 kg
    b.add_shape_box(ply, hx=0.15, hy=0.15, hz=PLY_HZ, cfg=cfg)
    b.add_articulation([b.add_joint_fixed(-1, ply, parent_xform=wp.transform((0.0, 0.0, PLY_Z), wp.quat_identity()))])
    box = b.add_body(xform=wp.transform((0.11, 0.11, PLY_TOP + 0.02), wp.quat_identity()), label="box")
    b.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02)
    if dest == "vbd":
        b.add_cloth_grid(pos=wp.vec3(-0.08, -0.08, PLY_TOP + 0.003), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                         dim_x=8, dim_y=8, cell_x=0.02, cell_y=0.02, mass=0.002, tri_ke=1.0e3, tri_ka=1.0e3,
                         tri_kd=1.0e-1, edge_ke=0.01, edge_kd=1.0e-4, particle_radius=0.003)
    else:
        v, t = grid_mesh(8, 8, 0.02, plane="xy")
        add_film_mesh(b, v, t, HDPE_15UM, pos=(-0.08, -0.08, PLY_TOP + HDPE_15UM.extras.contact_radius))
    return b


def _model(dest: str, worlds: int = 2, *, zero_welded: bool = False) -> newton.Model:
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    for _ in range(worlds):
        top.add_world(_bench_world(dest))
    top.color()
    model = top.finalize(device="cpu")
    model.soft_contact_ke, model.soft_contact_mu = 1.0e4, 0.5
    if zero_welded:                                     # SC8's interim: welded masses zeroed before the solvers
        welded = world_welded_bodies(model)
        for name in ("body_mass", "body_inv_mass", "body_inertia", "body_inv_inertia"):
            a = getattr(model, name).numpy()
            a[welded] = 0.0
            getattr(model, name).assign(a)
    return model


def _entries(model: newton.Model, dest: str) -> tuple[SolverCoupledProxy.Entry, SolverCoupledProxy.Entry, object]:
    soft = (lambda v: SolverVBD(v, iterations=10)) if dest == "vbd" else \
        (lambda v: SolverFilm(v, film=FilmConfig(), iterations=10))
    e_m = SolverCoupledProxy.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False,
                                                                             njmax=50),
                                   bodies=list(range(model.body_count)), joints=list(range(model.joint_count)))
    e_s = SolverCoupledProxy.Entry(name="soft", solver=soft, particles=list(range(model.particle_count)))
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="soft", bodies=list(range(model.body_count)),
                                     mode="lagged", proxy_relaxation_mode="fixed",
                                     collision_pipeline=lambda m: newton.CollisionPipeline(m), collide_interval=1)
    return e_m, e_s, proxy


def _scheduler(model: newton.Model, dest: str) -> Scheduler:
    e_m, e_s, proxy = _entries(model, dest)
    return Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_s, adaptive=False)],
                     [PairPolicy("mjc", "soft", mode="lagged", proxy=proxy)], SchedulerPolicy())


def _stock(model: newton.Model, dest: str) -> SolverCoupledProxy:
    e_m, e_s, proxy = _entries(model, dest)
    return SolverCoupledProxy(model=model, entries=[e_m, e_s],
                              coupling=SolverCoupledProxy.Config(proxies=[proxy], iterations=2))


def _rest(model: newton.Model, stepper, periods: int = 6, settle: int = 2) -> tuple[np.ndarray, list[np.ndarray]]:
    """Per-period particle displacement [m] over the periods after ``settle``, and body_q per exchange."""
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    marks, body_q = [], []
    for k in range(periods * 32):
        if k % 32 == 0:
            marks.append(s0.particle_q.numpy().copy())
        s0.clear_forces()
        pipe.collide(s0, contacts)
        stepper.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        body_q.append(s0.body_q.numpy().copy())
    marks.append(s0.particle_q.numpy().copy())
    disp = [np.linalg.norm(b - a, axis=1) for a, b in zip(marks[settle:], marks[settle + 1:])]
    return np.concatenate(disp), body_q


@pytest.mark.timeout(900)
@pytest.mark.parametrize("dest", SOFT)
def test_soft_body_rests_on_a_welded_plywood(dest):
    model = _model(dest)
    before = {n: getattr(model, n).numpy().copy() for n in ("body_mass", "body_inv_mass", "body_inertia",
                                                              "body_inv_inertia")}
    sched = _scheduler(model, dest)
    assert sched.kinematic_proxies() == {("mjc", "soft"): (0, 2)}           # the plywood of each world
    for n, a in before.items():                                             # source model untouched
        np.testing.assert_array_equal(getattr(model, n).numpy(), a, err_msg=n)
    fixed, _ = _rest(model, sched)
    zeroed_model = _model(dest, zero_welded=True)
    zeroed, _ = _rest(zeroed_model, _scheduler(zeroed_model, dest))
    p99 = {k: float(np.percentile(v, 99)) for k, v in (("fixed", fixed), ("zeroed", zeroed))}
    assert p99["fixed"] <= 1e-5, p99                                         # <= 0.01 mm per period
    assert p99["fixed"] <= max(2.0 * p99["zeroed"], 1e-7), p99
    assert np.abs(sched.energy_ledger()["interface_work"]).max() < 1e-6
    assert int(sched._kinematic_moved.numpy().max()) == 0


@pytest.mark.timeout(900)
@pytest.mark.parametrize("dest", SOFT)
def test_welded_proxy_is_kinematic_in_the_destination(dest):
    """Newton's stock coupler gives the welded plywood its 9 kg (inverse mass 1/9) in the destination view and the
    destination solver's copy; the Scheduler gives it inverse mass and inertia 0 in both, the free box keeps its."""
    def inv(stepper):
        e = stepper._entries["soft"]
        loc = [int(x) for m in stepper._proxy_mappings for x in m.proxy_ids_local.numpy()]
        return (e.view.body_inv_mass.numpy()[loc], e.solver.body_inv_mass_effective.numpy()[loc],
                np.abs(e.view.body_inv_inertia.numpy()[loc]).reshape(len(loc), -1).max(axis=1))

    stock = inv(_stock(_model(dest), dest))
    fixed = inv(_scheduler(_model(dest), dest))
    plywood, box = [0, 2], [1, 3]                       # proxy order: plywood, box per world
    assert (stock[0][plywood] > 0.1).all() and (stock[1][plywood] > 0.1).all()
    for arr in fixed:
        assert (arr[plywood] == 0.0).all() and (arr[box] > 0.0).all()


@pytest.mark.timeout(900)
def test_mujoco_rollout_unchanged_by_kinematic_proxies():
    """The free box (MuJoCo's) follows the same trajectory bitwise with the stock coupler and the Scheduler: the
    fix touches only the destination view."""
    model = _model("vbd", worlds=1)
    _, fixed = _rest(model, _scheduler(model, "vbd"), periods=2, settle=0)
    model2 = _model("vbd", worlds=1)
    _, stock = _rest(model2, _stock(model2, "vbd"), periods=2, settle=0)
    box = 1                                                                  # body 0 is the plywood
    np.testing.assert_array_equal(np.stack([q[box] for q in fixed]), np.stack([q[box] for q in stock]))


@pytest.mark.timeout(900)
def test_a_kinematic_proxy_that_moves_turns_its_world_invalid():
    model = _model("vbd")
    sched = _scheduler(model, "vbd")
    dst = sched._entries["soft"]
    local = wp.array(sched._kinematic_local_ids("soft")[:1], dtype=int, device=model.device)   # world 0's plywood
    dst.view.set_body_inertial_properties(local, wp.array([9.0], dtype=float, device=model.device),
                                          wp.array([wp.mat33(np.eye(3, dtype=np.float32) * 0.07)], dtype=wp.mat33,
                                                   device=model.device))
    dst.solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)   # a regression: dynamic again
    _rest(model, sched, periods=1, settle=0)
    moved = sched._kinematic_moved.numpy()
    assert moved.tolist() == [1, 0]
    flags = sched.capacity_flags()
    assert any(isinstance(f, tuple) and f[1] == GUARD_BITS for f in flags)
    s = model.state()
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), capacity_flags=flags)
    monitor.update(0, s, s, None)
    report = monitor.poll()
    assert report.invalid_worlds == (0,)
    assert int(report.flags[0]) & int(HealthFlags.JOINT_FAIL)


def test_world_welded_bodies():
    b = newton.ModelBuilder()
    a = b.add_link(label="a")
    c = b.add_link(label="c")
    d = b.add_link(label="d")
    j0 = b.add_joint_fixed(-1, a)
    j1 = b.add_joint_fixed(a, c)                 # welded through a
    j2 = b.add_joint_revolute(a, d)              # a hinge: one DOF to the world
    b.add_articulation([j0, j1, j2])
    b.add_body(label="free")                     # FREE joint
    model = b.finalize(device="cpu")
    assert world_welded_bodies(model).tolist() == [True, True, False, False]
