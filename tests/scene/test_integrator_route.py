# SPDX-License-Identifier: Apache-2.0
"""SCR-WP6-4: a free-body-only MuJoCo entry (no robot, articulated/damped/actuated dof, equality row or kinematic
body; no precision pair or noslip) runs rk4; everything else keeps implicitfast / discrete. The torque-free tumble
that motivated it conserves angular momentum under the routed rk4 and drifts under the first-order default."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import warp as wp

from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene import integrator_route as IR
from alloy.scene.assembler import SceneAssembler
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.scene.rigid_step import RigidStepRequest
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S

ABS = T.material("abs", density=1050.0, E=2.3e9, mu=0.4, e=0.3, k=2e6)
AUTO = RigidStepRequest("auto")    # the API's rigid step
W0 = np.array([0.2, 6.0, 0.3])          # rad/s, mostly about the intermediate axis: the unstable tumble
TUMBLE_S = 2.0
ROD_PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


def _box():
    return T.box("box", ABS, 0.04, 0.025, 0.01)


def _assemble(objs, placements, device: str = "cpu", step=None):
    card, _ = S.scene(objs, placements, worlds=1)
    asm = SceneAssembler(S.library(objs), Router(entries={Route.RIGID: "mujoco"}), rigid_step=step)
    return asm.assemble(card, device)


def _mujoco_kwargs(built) -> dict:
    (mj,) = [e for e in built.entries if e.capability == "mujoco"]
    return dict(mj.solver_kwargs)


def _rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _tumble_drift(sim: Simulation) -> float:
    """Largest |L(t) − L0| / |L0| of the box's world angular momentum over TUMBLE_S, launched at W0."""
    qd = sim.state.joint_qd.numpy()
    qd[3:6] = W0
    sim.state.joint_qd.assign(qd)
    sim.state.body_qd.assign(np.array([[0.0, 0.0, 0.0, *W0]], np.float32))
    inertia = sim.built.model.body_inertia.numpy()[0].astype(np.float64)

    def momentum() -> np.ndarray:
        q = sim.state.body_q.numpy()[0].astype(np.float64)
        r = _rot(q[3:])
        return r @ inertia @ r.T @ sim.state.body_qd.numpy()[0, 3:].astype(np.float64)

    l0, drift = momentum(), 0.0
    for _ in range(round(TUMBLE_S / sim.built.control_dt)):
        sim.step_period()
        drift = max(drift, float(np.linalg.norm(momentum() - l0) / np.linalg.norm(l0)))
    return drift


def test_r1_runs_on_mujoco_warp_only(monkeypatch):
    """RIGID-DEFAULT (coordinator ruling 2026-11-09 16:32Z): R1 on MuJoCo Warp (a GPU, or the suite's CPU warp route);
    clutter on the CPU route on MuJoCo C keeps rk4 (no friction-pull cap there); R1_ON False keeps rk4 everywhere."""
    assert IR.r1_off("cuda:0") == IR.r1_off(None) == IR.r1_off("cpu") == ""      # tests/conftest.py: warp
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    assert "MuJoCo C" in IR.r1_off("cpu") and IR.r1_off("cuda:0") == ""
    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = _assemble(*_clutter(IR.R1_MIN_FREE + 1), step=AUTO)
    assert _mujoco_kwargs(built)["integrator"] == "rk4" and IR.R1_EFFECT not in built.report.scene_effects
    assert any("rk4 (the CPU route on MuJoCo C" in r for r in built.report.decisions["box0"].reasons)
    monkeypatch.setattr(IR, "R1_ON", False)
    assert "off" in IR.r1_off("cuda:0")


def test_effect_integrator_precedence():
    assert IR.effect_integrator([]) == "implicitfast"
    assert IR.effect_integrator(["mujoco:integrator=discrete", "mujoco:integrator=implicitfast"]) == "discrete"
    assert IR.effect_integrator([IR.RK4_EFFECT]) == "rk4"


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_free_body_scene_runs_rk4_and_conserves_momentum(monkeypatch):
    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = _assemble([_box()], [S.place("box", z=1.0)])
        assert _mujoco_kwargs(built)["integrator"] == "rk4"
        assert IR.RK4_EFFECT in built.report.scene_effects
        sim = Simulation(built)
        assert int(sim.stepper.solvers()[0].mj_model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4)
        rk4 = _tumble_drift(sim)
        monkeypatch.setattr(IR, "choose_integrator", lambda *a, **k: IR.IntegratorChoice(False, "test"))
        first = _built_drift()
    assert rk4 < 1e-4 < 1e-3 < first and rk4 < first / 5.0   # CPU: rk4 + masked push ~1e-6 vs 4.3e-3


def _built_drift() -> float:
    built = _assemble([_box()], [S.place("box", z=1.0)])
    assert _mujoco_kwargs(built)["integrator"] == "implicitfast"
    return _tumble_drift(Simulation(built))


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_box_on_table_rests_under_rk4():
    """A static table adds no dof: the box on it is still a free-body-only entry, and comes to rest."""
    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = _assemble([S.table(), _box()], [S.place("table"), S.place("box", z=0.02 + 0.01 + 1e-3)])
        assert _mujoco_kwargs(built)["integrator"] == "rk4"
        sim = Simulation(built)
        for _ in range(round(1.0 / built.control_dt)):
            sim.step_period()
        z0 = sim.state.body_q.numpy()[0, :3].copy()
        for _ in range(round(0.25 / built.control_dt)):
            sim.step_period()
        assert np.isfinite(sim.state.body_q.numpy()).all()
        assert float(np.linalg.norm(sim.state.body_q.numpy()[0, :3] - z0)) < 1e-4


def _drop(e: float) -> tuple[float, float]:
    """(h1, §5.3 bound) of a 20 mm ball (K 2e6 per surface, both at e) dropped 0.1 m onto a static table, on rk4 (a
    lone ball, under the API's "auto" step too)."""
    from alloy.bench.analytic.restitution import H0, RADIUS, first_rebound

    mat = dict(density=1050.0, E=2.3e9, mu=0.01, e=e, k=2e6)
    ball = T.obj("ball", _sphere(RADIUS), T.material("abs", **mat))
    tab = T.box("table", T.material("wood", **(dict(T.WOOD) | {"mu": 0.01, "e": e, "k": 2e6})), 0.4, 0.4, 0.02,
                motion="static")
    built = _assemble([tab, ball], [S.place("table"), S.place("ball", z=0.02 + RADIUS + H0)], step=AUTO)
    assert _mujoco_kwargs(built)["integrator"] == "rk4"
    sim = Simulation(built)
    z = [0.02 + RADIUS + H0]
    sim.add_step_observer(lambda s_in, s_out, c, ct, dt: z.append(float(s_out.body_q.numpy()[0, 2])))
    for _ in range(round(0.6 / built.control_dt)):
        sim.step_period()
    h1 = max(first_rebound(np.arange(len(z)) * sim.clock.dt, np.asarray(z)) - 0.02 - RADIUS, 0.0)
    return h1, 0.02 * h1 + 5.0 * sim.clock.dt * np.sqrt(2.0 * 9.81 * h1)


def _sphere(r: float):
    from alloy.cards.types import Primitive

    return Primitive("sphere", (r,))


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
@pytest.mark.parametrize("e", [0.0, 0.8])
def test_rk4_contacts_keep_a6_restitution(e):
    """Regression: at the semi-implicit cap rk4 damped the contact (e 0.8 rebounded 51.7 mm vs 64 mm); the rk4 cap gives
    64.0 mm and a 0.03 mm hop at e = 0 (CPU)."""
    from alloy.bench.analytic.restitution import H0

    with isolated_registry(empty=True):
        register_builtin_solvers()
        h1, bound = _drop(e)
    assert abs(h1 - e * e * H0) <= max(bound, 1e-4), (h1, bound)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_rk4_rule_keeps_first_order_elsewhere():
    """A chain keeps discrete, a hinged MJCF object implicitfast, a kinematic body implicitfast; reasons logged."""
    from alloy.cards.library import CardLibrary
    from alloy.scene import coverage as C

    lib = CardLibrary()
    router = Router(rod_self_test=lambda rod, h: dict(ROD_PASS))     # the chain's routing, not its self-test
    for name, want in (("usb_cable_1m", "discrete"), ("hinged_box_plywood", "implicitfast")):
        card, _ = C.scene_for(name, lib, 1)
        with isolated_registry(empty=True):
            register_builtin_solvers()
            built = SceneAssembler(lib, router).assemble(card, "cpu", world_count=1)
        assert _mujoco_kwargs(built).get("integrator", "implicitfast") == want, name
        assert IR.RK4_EFFECT not in built.report.scene_effects
    kin = T.box("pusher", ABS, 0.01, 0.01, 0.01, motion="kinematic")
    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = _assemble([_box(), kin], [S.place("box", z=1.0), S.place("pusher", x=0.3, z=1.0)])
    assert _mujoco_kwargs(built)["integrator"] == "implicitfast"


def test_factory_drops_rk4_under_sleeping(monkeypatch, caplog):
    """rk4 drops to implicitfast under sleeping; rk4 and F6 R1 entries (GAINS_KEY, step 2c's too) build
    SyncedMuJoCo, except one that also asks for noslip, which keeps SolverMuJoCo and its noslip (WP2's review)."""
    from alloy.contact.light_body import GAINS_KEY
    from alloy.scene import builtin_plugins as BP
    from alloy.scene import mujoco_sync as MS
    from alloy.scene.noslip_route import NOSLIP_KEY

    seen, noslip = {}, []

    class _Fake:
        def __init__(self, view, **kw):
            seen.update(kw)

    class _FakeSynced(_Fake):
        pass

    monkeypatch.setattr(BP.newton.solvers, "SolverMuJoCo", _Fake)
    monkeypatch.setattr(MS, "SyncedMuJoCo", _FakeSynced)
    monkeypatch.setattr("alloy.scene.noslip_route.install_entry_noslip", lambda solver, spec: noslip.append(spec))
    assert not isinstance(BP._mujoco(None, {"integrator": "rk4", "enable_sleeping": True}), _FakeSynced)
    assert seen["integrator"] == "implicitfast" and "rk4 has no sleeping" in caplog.text
    assert isinstance(BP._mujoco(None, {"integrator": "rk4"}), _FakeSynced)    # rk4: masked push (mujoco_sync)
    assert seen["integrator"] == "rk4"
    assert isinstance(BP._mujoco(None, {"integrator": "implicitfast", GAINS_KEY: True}), _FakeSynced)
    solver = BP._mujoco(None, {"integrator": "implicitfast", GAINS_KEY: True, NOSLIP_KEY: {"iterations": 4}})
    assert not isinstance(solver, _FakeSynced) and noslip == [{"iterations": 4}]
    assert not isinstance(BP._mujoco(None, {"integrator": "implicitfast"}), _FakeSynced)


def test_factory_drops_rk4_on_contacts_made_once_per_step(monkeypatch):
    """PHYSA-1's mechanism: an entry whose contacts are made once per step (Newton's pipeline,
    ``use_mujoco_contacts=False``) must not run a multi-stage explicit integrator: rk4's stages would see the contacts
    frozen and bounces would gain energy. The factory runs it on implicitfast, on the plain solver; an rk4 entry on
    MuJoCo's own contacts (rebuilt at each stage) keeps rk4."""
    from alloy.scene import builtin_plugins as BP
    from alloy.scene import mujoco_sync as MS

    seen = {}

    class _Fake:
        def __init__(self, view, **kw):
            seen.clear()
            seen.update(kw)

    class _FakeSynced(_Fake):
        pass

    monkeypatch.setattr(BP.newton.solvers, "SolverMuJoCo", _Fake)
    monkeypatch.setattr(MS, "SyncedMuJoCo", _FakeSynced)
    assert not isinstance(BP._mujoco(None, {"integrator": "rk4", "use_mujoco_contacts": False}), _FakeSynced)
    assert seen["integrator"] == "implicitfast" and seen["use_mujoco_contacts"] is False
    assert isinstance(BP._mujoco(None, {"integrator": "rk4", "use_mujoco_contacts": True}), _FakeSynced)
    assert seen["integrator"] == "rk4"


@pytest.mark.gpu
@pytest.mark.timeout(900)
def test_rk4_period_graph_captures_and_replays():
    """A CUDA graph captures the rk4 control period (IntegratedSimulation, WP12) and replays it like eager steps."""
    from alloy.integration.simulation import IntegratedSimulation

    dev = str(wp.get_cuda_device())
    out = {}
    for graph in (False, True):
        with isolated_registry(empty=True):
            register_builtin_solvers()
            built = _assemble([S.table(), _box()], [S.place("table"), S.place("box", z=0.02 + 0.01 + 5e-3)], dev)
            assert _mujoco_kwargs(built)["integrator"] == "rk4"
            sim = IntegratedSimulation(built, use_graph=graph)
            for _ in range(20):
                sim.step_period()
            out[graph] = (sim.state.body_q.numpy().copy(), sim.graph_stats())
    q_eager, q_graph, stats = out[False][0], out[True][0], out[True][1]
    assert stats["mode"] == "graph" and stats["captures"] >= 1 and stats["host_syncs"] == 0, stats
    assert np.isfinite(q_graph).all()
    np.testing.assert_allclose(q_graph, q_eager, atol=1e-6)


# --- F6 R1 (SCR-WP13-SPEED-1 part 1, ruling 2026-11-08 14:18Z): clutter runs implicitfast on rk4's gains -----------

def _clutter(n: int, rod: bool = False):
    """``n`` boxes (``_box``'s, not floored) in a grid, and a light slender rod (floored) when ``rod``."""
    objs = [T.box(f"box{i}", ABS, 0.04, 0.025, 0.01) for i in range(n)]
    places = [S.place(f"box{i}", x=0.2 * (i % 4), y=0.2 * (i // 4), z=0.1) for i in range(n)]
    if rod:
        objs.append(T.box("rod", ABS, 0.05, 0.006, 0.006))      # l2/l1 0.12: a rigid volume, slender (ratio 35)
        places.append(S.place("rod", x=-0.3, z=0.1))
    return objs, places


def _forced_rk4(monkeypatch, objs, places, step=None):
    """The same scene with the rule forced to rk4 (today's gains), for a gains comparison."""
    with monkeypatch.context() as m:
        m.setattr(IR, "_clutter_or_rk4", lambda b, mj, dt, *_: IR.IntegratorChoice(True, "forced rk4 (test)"))
        return _assemble(objs, places, step=step)


def test_r1_clutter_runs_implicitfast_on_rk4_gains(monkeypatch):
    from alloy.contact.light_body import GAINS_KEY

    with isolated_registry(empty=True):
        register_builtin_solvers()
        objs, places = _clutter(IR.R1_MIN_FREE + 1)
        built = _assemble(objs, places)
        kw = _mujoco_kwargs(built)
        assert kw["integrator"] == "implicitfast" and kw[GAINS_KEY] is True
        assert IR.R1_EFFECT in built.report.scene_effects and IR.RK4_EFFECT not in built.report.scene_effects
        assert all(any(r.startswith("mujoco integrator implicitfast: ") and "more than 8" in r
                       for r in built.report.decisions[f"box{i}"].reasons) for i in range(len(objs)))
        rk4 = _forced_rk4(monkeypatch, objs, places)
        assert _mujoco_kwargs(rk4)["integrator"] == "rk4"
        for field in ("shape_material_ke", "shape_material_kd"):         # today's gains, bit for bit
            np.testing.assert_array_equal(getattr(built.model, field).numpy(), getattr(rk4.model, field).numpy())
        assert built.report.rigid_step.dt == rk4.report.rigid_step.dt


def test_r1_keeps_rk4_up_to_the_count_and_with_a_floored_body():
    with isolated_registry(empty=True):
        register_builtin_solvers()
        few = _assemble(*_clutter(IR.R1_MIN_FREE))
        assert _mujoco_kwargs(few)["integrator"] == "rk4"
        floored = _assemble(*_clutter(IR.R1_MIN_FREE + 1, rod=True))
        assert _mujoco_kwargs(floored)["integrator"] == "rk4"
        assert list(floored.report.inertia_floor) == ["rod"]
        assert any("the inertia floor raises rod" in r for r in floored.report.decisions["box0"].reasons)


def test_r1_precision_profile_keeps_rk4():
    from alloy.integration import wiring

    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = _assemble(*_clutter(IR.R1_MIN_FREE + 1), step=AUTO)
        parts = wiring.wire(built, "precision")
        (mj,) = [e for e in parts.built.entries if e.capability == "mujoco"]
        assert mj.solver_kwargs["integrator"] == "rk4"
        assert any("precision profile keeps rk4" in n for n in parts.notes)
        std = wiring.wire(built, "standard")
        (mj,) = [e for e in std.built.entries if e.capability == "mujoco"]
        assert mj.solver_kwargs["integrator"] == "implicitfast"


def _blocks(n: int, dims=(0.05, 0.05, 0.05)):
    """``n`` blocks of edges ``dims`` (a cube by default: inertia spread 0) in a row."""
    return ([T.box(f"block{i}", ABS, *dims) for i in range(n)],
            [S.place(f"block{i}", x=0.2 * i, z=0.1) for i in range(n)])


def test_few_isotropic_bodies_keep_rk4():
    """No R2 (withdrawn 2026-11-09): cubes and near-cubes keep rk4 at up to R1_MIN_FREE bodies, under the API's "auto"
    step too."""
    with isolated_registry(empty=True):
        register_builtin_solvers()
        for objs, places in (_blocks(3), _blocks(IR.R1_MIN_FREE, (0.05, 0.05, 0.0505)), _blocks(1)):
            built = _assemble(objs, places, step=AUTO)
            assert _mujoco_kwargs(built)["integrator"] == "rk4" and IR.R1_EFFECT not in built.report.scene_effects
            assert any("free flight stays accurate up to 8" in r for r in built.report.decisions["block0"].reasons)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_r1_scene_steps_on_implicitfast():
    """The factories pop the gains flag (SolverMuJoCo would refuse it) and MuJoCo runs implicitfast."""
    with isolated_registry(empty=True):
        register_builtin_solvers()
        sim = Simulation(_assemble(*_clutter(IR.R1_MIN_FREE + 1)))
        solver = sim.stepper.solvers()[0]
        assert int(solver.mj_model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
        for _ in range(5):
            sim.step_period()
        assert np.isfinite(sim.state.body_q.numpy()).all()


# --- step 2c: own gains (SCR-WP0-RIGIDDT-2; WP6 + WP3 2026-11-08 14:57Z, coordinator 15:16Z) -----------------------

def _stack(variant: str = "", n: int = 3, fixed: bool = False, profile: str = "standard", density: float = 10.0):
    """XB4-9's resting stack shape, built through the API (rigid step "auto"): n x n x n cubes of 1 m at 10 kg/m3,
    heavy, isotropic and material-bound. Variants change the first cube: "light" (0.5 m: cap-bound today), "aniso"
    (1 x 1 x 1.6 m), "pair" (friction 0.6: pair rows with every other cube), "mjcf" (plus a loose MJCF block)."""
    import alloy

    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground(friction=1.0)
    for c, (i, j, k) in enumerate((i, j, k) for i in range(n) for j in range(n) for k in range(n)):
        size = {"light": (0.5,) * 3, "aniso": (1.0, 1.0, 1.6)}.get(variant, (1.0,) * 3) if c == 0 else (1.0,) * 3
        scene.add_box(f"c{i}{j}{k}", size=size, pos=(2.0 * i, 2.0 * j, 0.5 * size[2] + 1.002 * k), density=density,
                      friction=0.6 if variant == "pair" and c == 0 else 1.0)
    if variant == "mjcf":
        scene.add_robot("block", mjcf="<mujoco><worldbody><body name='b' pos='9 0 0.25'><freejoint/><geom "
                                      "type='box' size='0.25 0.25 0.25' mass='1'/></body></worldbody></mujoco>")
    if fixed:
        scene.rigid_step = "fixed"
    return alloy.build(scene, device="cpu", profile=profile)


def _contacts(sim) -> dict[str, bytes]:
    """Every contact value the gains pass writes (shape gains, every mujoco:* attribute: pair rows, solmix), the
    inertias and the floor's armature, as exact bits."""
    m = sim.built.model
    keys = ("shape_material_ke", "shape_material_kd", "body_inertia", "joint_armature")
    out = {k: getattr(m, k).numpy().tobytes() for k in keys}
    ns = getattr(m, "mujoco", None)
    out.update({k: getattr(ns, k).numpy().tobytes() for k in dir(ns) if hasattr(getattr(ns, k), "numpy")})
    return out


def _says_2c(sim) -> bool:
    return any("step 2c" in r for d in sim.built.report.decisions.values() for r in d.reasons)


@pytest.mark.parametrize("variant", ["", "pair"])
def test_2c_stack_takes_its_own_gains_with_every_contact_as_todays(variant):
    """XB4-9's stack (and with 27 pair rows) steps at 5 ms instead of R1's 2.5 ms on implicitfast, and every shape
    gain, pair row and inertia equals today's (the fixed step under rk4's quarter cap) bit for bit: WP3's guard law,
    the premise of step 2c (only the step changes, never a contact)."""
    own, today = _stack(variant), _stack(variant, fixed=True)
    assert own.built.report.rigid_step.dt == pytest.approx(0.005, abs=0) and _says_2c(own)
    assert today.built.report.rigid_step.dt == pytest.approx(0.00125, abs=0) and not _says_2c(today)
    assert IR.R1_EFFECT in own.built.report.scene_effects
    assert int(own.engine.stepper.solver.mj_model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert _contacts(own) == _contacts(today)
    if variant == "pair":
        assert own.built.model.mujoco.pair_geom1.numpy().size > 0


@pytest.mark.parametrize("variant", ["light", "aniso", "mjcf"])
def test_2c_leaves_other_entries_exactly_as_r1_built_them(monkeypatch, variant):
    """A cap-bound (light) body, an anisotropic one, or a loose MJCF block (WP3's note: its stiffness is the light-body
    rule's; it is a robot, so the entry is not R1) keeps the entry off step 2c, and the build is bit for bit the one
    without step 2c (its trials run on copies of the template)."""
    with_2c = _stack(variant)
    monkeypatch.setattr(IR, "own_gains", lambda choice, *a, **k: choice)
    without = _stack(variant)
    assert not _says_2c(with_2c)
    assert with_2c.built.report.rigid_step.dt == without.built.report.rigid_step.dt
    assert _contacts(with_2c) == _contacts(without)


def test_2c_not_below_r1_count_or_under_precision():
    """Up to R1_MIN_FREE bodies the entry keeps rk4 (no R1, so no step 2c; F6 R2 withdrawn 2026-11-09); the precision
    profile keeps the fixed step and rk4 (WP6's objection: "own" needs the API's rigid step "auto")."""
    small = _stack(n=2)                                  # 8 cubes
    assert not _says_2c(small) and IR.R1_EFFECT not in small.built.report.scene_effects
    precise = _stack(profile="precision")
    assert not _says_2c(precise) and precise.built.report.rigid_step.dt == pytest.approx(0.00125, abs=0)


def test_2c_entry_refuses_the_precision_profile_on_the_engine_path():
    """WP6's review: an entry assembled under "auto" with own gains, then wired under precision, is refused (rk4 at
    the own step would run above rk4's quarter cap); alloy.build never asks "auto" under precision."""
    from alloy.core.errors import AssemblyError
    from alloy.integration import build_simulation

    built = _stack().built
    assert IR.OWN_EFFECT in built.report.scene_effects
    with pytest.raises(AssemblyError, match="rigid_step 'fixed'"):
        build_simulation(built, "precision")


def test_2c_guard_refuses_a_changed_contact(monkeypatch):
    """The guard decides: if the "own" trial wrote any contact value other than today's, the entry stays on R1."""
    real = IR._trial

    def tampered(b, mode, request, gains, report):
        values, dt = real(b, mode, request, gains, report)
        return ((values[0] + b"x", *values[1:]) if mode == "own" else values), dt
    monkeypatch.setattr(IR, "_trial", tampered)
    sim = _stack()
    assert not _says_2c(sim) and sim.built.report.rigid_step.dt == pytest.approx(0.0025, abs=0)


def test_2c_margin_keeps_a_denser_stack_at_5_ms(monkeypatch):
    """OWN_KE_FRACTION's margin sets the step: at 30 kg/m3 the cubes' ke is 0.79 of implicitfast's full cap at 10 ms,
    over 0.6, so 5 ms; at the full cap (the mutant) they would step at 10 ms, where WP6's piles failed."""
    from alloy.scene import native_gains as NG

    assert _stack(density=30.0).built.report.rigid_step.dt == pytest.approx(0.005, abs=0)
    monkeypatch.setitem(NG.KE_FRACTION, "own", 1.0)
    assert _stack(density=30.0).built.report.rigid_step.dt == pytest.approx(0.01, abs=0)


def test_2c_refuses_an_mjcf_labelled_body():
    """WP0's review: a body with Newton's MJCF label (light_body's scope: its rows are written at run time, outside the
    guard) keeps an entry off own gains, even if a future importer let one into an R1 entry."""
    from types import SimpleNamespace

    from alloy._vendor import newton

    b = newton.ModelBuilder()
    for label in ("cube", "cell.xml/worldbody/block"):
        body = b.add_body(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()), label=label)
        b.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
    objs = [SimpleNamespace(decision=SimpleNamespace(entry="mujoco"), ranges=SimpleNamespace(body=[0, 1]))]
    assert "MJCF body" in IR._not_eligible(b, objs)
    assert IR._not_eligible(b, [SimpleNamespace(decision=SimpleNamespace(entry="mujoco"),
                                                ranges=SimpleNamespace(body=[0]))]) == ""

