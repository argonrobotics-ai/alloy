# SPDX-License-Identifier: Apache-2.0
"""SCR-DC3-3 as amended by SCR-WP2-9: an unbreakable weld at a CHAIN's root is a tree joint (no equality row,
separation 0); every other weld/connect is an equality row sized at the design step (direct solref at ω·h 0.6), with a
flat solimp sized for 0.1 mm at the design force and capped at 0.999, and a route-report warning when the cap cannot
meet the target."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.scene import weld_impedance as WI
from alloy.scene.assembler import SceneAssembler

DT = 1.25e-3            # rigid_dt_max(0.02): the design step of a 20 ms control period
LOAD = 90.0             # N, SFF-8661 cable retention (DC3 QSFP28)
CUBE = "wp2_test_weld_cube"
BLOCK = "wp2_test_weld_block"
OMEGA2 = (WI.WELD_OMEGA_H / DT) ** 2


def _sep(m: float, d: float, f: float) -> float:
    return f * (1.0 - d) * d / (m * OMEGA2)              # (1 − d)/(d·k)·F/m with k = ω²/d²


def test_solimp_is_sized_for_the_target_and_capped():
    heavy = WI.weld_impedance(0.05, DT, 50.0)              # 50 g: the target needs less than the cap
    assert WI.WELD_IMP_MIN < heavy.imp < WI.WELD_IMP_CAP and heavy.meets_target
    assert heavy.separation == pytest.approx(WI.WELD_SEP_TOL, rel=1e-6)
    light = WI.weld_impedance(5.0e-4, DT, 50.0)            # 0.5 g: the cap cannot reach 0.1 mm
    assert light.imp == WI.WELD_IMP_CAP and not light.meets_target
    assert light.separation == pytest.approx(_sep(5.0e-4, WI.WELD_IMP_CAP, 50.0))
    assert "exceeds 0.1 mm" in WI.separation_warning("weld", "a", "b", light)
    assert WI.weld_impedance(5.0, DT, 1.0).imp == WI.WELD_IMP_MIN               # easy target: MuJoCo's default dmax
    assert WI.weld_impedance(5.0e-4, DT, None).imp == WI.WELD_IMP_CAP           # no design force: the cap
    w = WI.weld_impedance(5.0e-4, DT, LOAD)
    assert w.solref == pytest.approx((-OMEGA2, -2.0 * WI.WELD_OMEGA_H / DT)) and WI.WELD_OMEGA_H < WI.STABLE_OMEGA_H
    assert WI.reduced_mass(0.02, 5e-4) == pytest.approx(1 / (50 + 2000)) and WI.reduced_mass(0.0, 5e-4) == 5e-4


def test_card_overrides():
    soft = WI.weld_impedance(5.0e-4, DT, LOAD, timeconst=0.001)
    assert soft.solref == (0.001, 1.0) and soft.source == "card timeconst"
    clamped = WI.weld_impedance(5.0e-4, DT, LOAD, timeconst=2 * DT)
    assert soft.k_force == pytest.approx(clamped.k_force)                       # MuJoCo's refsafe τ ≥ 2·h
    damped = WI.weld_impedance(5.0e-4, DT, LOAD, dampratio=0.5)
    assert damped.solref == pytest.approx((-OMEGA2, -WI.WELD_OMEGA_H / DT)) and damped.source == "card dampratio"


def _library(tmp_path):
    try:
        from alloy.cards.library import CardLibrary, default_root
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    (tmp_path / "objects").mkdir(exist_ok=True)
    (tmp_path / "objects" / f"{CUBE}.yaml").write_text(        # 4 mm steel cube: 0.50 g, the DAC's first link
        f"schema: alloy.object/1\nname: {CUBE}\nshape:\n  primitive:\n    kind: box\n    size: [0.002, 0.002, 0.002]\n"
        "material: steel_zinc_plated\n")
    (tmp_path / "objects" / f"{BLOCK}.yaml").write_text(
        f"schema: alloy.object/1\nname: {BLOCK}\nshape:\n  primitive:\n    kind: box\n    size: [0.01, 0.01, 0.01]\n"
        "material: steel_zinc_plated\nmotion: static\n")
    return CardLibrary([tmp_path, default_root()])


def _assemble(lib, objects, attachments, worlds: int = 1, device: str = "cpu"):
    from alloy.cards.types import SceneCard
    from alloy.scene.builtin_plugins import register_builtin_solvers

    register_builtin_solvers()
    return SceneAssembler(lib).assemble(SceneCard(robots=(), objects=objects, attachments=attachments,
                                                  control_dt=0.02), device, world_count=worlds)


def _pull(built, body: int, force, periods: int):
    from alloy.sim import Simulation

    sim = Simulation(built)
    x0 = sim.state.body_q.numpy()[body, :3].astype(np.float64)
    for _ in range(periods):
        for st in (sim.state, sim._next):
            f = st.body_f.numpy()
            f[body] = (*force, 0.0, 0.0, 0.0)
            st.body_f.assign(f)
        sim.step_period()
    q = sim.state.body_q.numpy()
    assert np.isfinite(q).all()
    return float(np.linalg.norm(q[body, :3] - x0))


def test_equality_weld_at_the_cap_warns_and_holds_as_predicted(tmp_path):
    from alloy.cards.types import Attachment, Placement

    built = _assemble(_library(tmp_path), (Placement(BLOCK, (0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0)),
                                           Placement(CUBE, (0.02, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0))),
                      (Attachment(BLOCK, CUBE, "weld"),))
    line = next(x for x in built.report.entry_options if x.startswith("mujoco weld "))
    assert "SCR-DC3-3/WP2-9 rule" in line and "1.25 ms design step" in line
    assert any("exceeds 0.1 mm at the solimp cap" in w for w in built.report.warnings)     # 0.5 g at 50 N
    m = built.model
    assert m.mujoco.eq_solref.numpy()[0] == pytest.approx((-OMEGA2, -2.0 * WI.WELD_OMEGA_H / DT), rel=1e-6)
    assert m.mujoco.eq_solimp.numpy()[0][:2] == pytest.approx((WI.WELD_IMP_CAP, WI.WELD_IMP_CAP))
    body = built.handles[CUBE].template.body.start
    mass = float(m.body_mass.numpy()[body])
    sep = _pull(built, body, (LOAD, 0.0, 0.0), 10)        # 0.2 s at the load, settled after ~5/ω = 10 ms
    assert sep == pytest.approx(_sep(mass, WI.WELD_IMP_CAP, LOAD), rel=0.25)        # MuJoCo's default: 3.6 m


CABLE, PLUG = "usb_cable_1m", "usb_a_plug"
PLUG_POSE = (0.3, -0.31, 0.00415, 0.5, 0.5, 0.5, 0.5)                     # T09's layout (library/scenes/task_t09)
CABLE_POSE = (0.28, -0.31, 0.00225, 0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
ANCHOR = (0.0, -0.0019, -0.02, 0.0, 0.0, 0.0, 1.0)


def _cable_scene(*extra_atts, between: bool = False, break_force=None, worlds: int = 1, device: str = "cpu"):
    from alloy.cards.library import CardLibrary
    from alloy.cards.types import Attachment, Placement
    from alloy.cards.types import Q

    objs = [Placement(PLUG, PLUG_POSE), Placement(CABLE, CABLE_POSE)]
    if between:                                            # another object between plug and cable in scene order
        objs.insert(1, Placement("cube_beech_30", (0.6, 0.3, 0.0155, 0.0, 0.0, 0.0, 1.0)))
    weld = Attachment(PLUG, CABLE, "weld", anchor=ANCHOR,
                      break_force=None if break_force is None else Q(break_force, "model", note="test: a breakable weld"))
    return _assemble(CardLibrary(assets=None), tuple(objs), (weld, *extra_atts), worlds, device)


@pytest.mark.parametrize("between", [False, True])
def test_chain_root_weld_is_a_tree_joint(between):
    built = _cable_scene(between=between)
    line = next(x for x in built.report.entry_options if x.startswith("mujoco weld "))
    assert "tree joint" in line and "no equality row" in line
    m = built.model
    assert int(getattr(m.mujoco, "equality_constraint_count", 0)) == 0
    plug = built.handles[PLUG].template.body.start
    root = built.handles[CABLE].template.joint.start
    assert int(m.joint_parent.numpy()[root]) == plug
    arts = m.joint_articulation.numpy()
    assert arts[root] == arts[next(j for j, c in enumerate(m.joint_child.numpy()) if c == plug)]
    solver = next(x for x in built.report.entry_options if x.startswith("mujoco solver: "))
    assert "Newton for the chain attachment" in solver        # SCR-DC1-2: no contact in this scene, margin 0
    links = list(built.handles[CABLE].template.body)
    from alloy.sim import Simulation

    sim = Simulation(built)
    q0 = sim.state.body_q.numpy().astype(np.float64)
    for _ in range(2):
        sim.step_period()
    q = sim.state.body_q.numpy().astype(np.float64)
    assert np.isfinite(q).all()
    rel = lambda s: np.asarray(s[links[0], :3]) - np.asarray(s[plug, :3])   # noqa: E731
    assert np.linalg.norm(rel(q) - rel(q0)) < 1e-3                  # the plug and link 0 move as one body here


def test_breakable_weld_and_second_end_stay_equality_rows():
    from alloy.cards.types import Attachment

    built = _cable_scene(break_force=200.0)
    line = next(x for x in built.report.entry_options if x.startswith("mujoco weld "))
    assert "tree joint" not in line and "solref=" in line
    assert int(built.model.mujoco.equality_constraint_count) == 1
    loop = _cable_scene(Attachment(PLUG, f"{CABLE}:5", "weld"))            # a second weld on the same chain
    lines = [x for x in loop.report.entry_options if x.startswith("mujoco weld ")]
    assert "tree joint" in lines[0] and "closes the loop" in lines[1]
    assert int(loop.model.mujoco.equality_constraint_count) == 1


def test_tree_weld_link0_children_carry_no_armature():
    """SCR-DC3-14: joint 1 (link 0's child) of a tree-welded chain has no armature at emission, and again when the
    MuJoCo entry is built after a scene-level override; the other chain joints and the breakable (equality) path keep
    theirs. Without it the joint-1 armature acted against the plug's own rotation."""
    from alloy.integration.throughput import chain_armature
    from alloy.scene.tree_weld import TREE_LABEL, rigid_armature_off
    from alloy.sim import Simulation

    built = _cable_scene()
    m = built.model
    root = built.handles[CABLE].template.joint.start
    qd, arm = m.joint_qd_start.numpy(), m.joint_armature.numpy()
    j1 = range(int(qd[root + 1]), int(qd[root + 2]))
    later = range(int(qd[root + 2]), int(qd[root + 3]))
    assert str(m.joint_label[root]).endswith("/" + TREE_LABEL)
    assert not np.any(arm[list(j1)]) and np.all(arm[list(later)] > 0.0)
    chain_armature(built, 1.0e-5)                                       # DC3's labelled override, after assembly
    assert np.all(m.joint_armature.numpy()[list(j1)] == 1.0e-5)
    sim = Simulation(built)                                             # the MuJoCo entry zeroes it again
    mj = next(s for s in sim.stepper.solvers() if hasattr(s, "mj_model")).mj_model
    arm = m.joint_armature.numpy()
    assert not np.any(arm[list(j1)]) and np.all(arm[list(later)] == pytest.approx(1.0e-5))
    assert np.count_nonzero(mj.dof_armature == 0.0) >= 3
    assert rigid_armature_off(m) == 0                                   # idempotent
    eq = _cable_scene(break_force=200.0)                                # equality row: link 0 is free, nothing zeroed
    assert not any(str(x).endswith("/" + TREE_LABEL) for x in eq.model.joint_label)
    root = eq.handles[CABLE].template.joint.start
    qd = eq.model.joint_qd_start.numpy()
    assert np.all(eq.model.joint_armature.numpy()[int(qd[root + 1]):int(qd[root + 2])] > 0.0)


def _mujoco(built):
    from alloy.sim import Simulation

    sim = Simulation(built)
    return next(s for s in sim.stepper.solvers() if hasattr(s, "mj_model"))


def _invweight(solver, body: int) -> np.ndarray:
    """[worlds, 2] body_invweight0 of Newton body `body` (world 0's index) in every world."""
    m2n = solver.mjc_body_to_newton.numpy()[0]
    b = int(np.flatnonzero(m2n == body)[0])
    return solver.mjw_model.body_invweight0.numpy().reshape(-1, int(solver.mj_model.nbody), 2)[:, b].astype(np.float64)


def _heavier_plug(built, solver, world: int, k: float) -> None:
    """World `world`'s plug k times heavier (mass and inertia), notified as a body-inertia change (a reset's redraw)."""
    from alloy._vendor import newton

    m, per = built.model, built.model.body_count // built.model.world_count
    i = built.handles[PLUG].template.body.start + world * per
    for name in ("body_mass", "body_inertia"):
        a = getattr(m, name).numpy()
        a[i] *= k
        getattr(m, name).assign(a)
    inv = m.body_inv_mass.numpy()
    inv[i] /= k
    m.body_inv_mass.assign(inv)
    inv = m.body_inv_inertia.numpy()
    inv[i] /= k
    m.body_inv_inertia.assign(inv)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)


@pytest.mark.parametrize("route", ["warp", "mjc"])
def test_tree_weld_parent_keeps_the_equality_path_contact_weight(monkeypatch, route):
    """SCR-DC3-14 part 2: the plug's body_invweight0 on the tree path is the equality path's (the plug without the
    cable), in every world, after a per-world mass change too; MuJoCo's own tree value counts the cable. The cable's
    bodies keep MuJoCo's tree values, and a scene without a tree weld installs nothing. On both CPU routes: on the
    MuJoCo C route a drawn world's chainless weight comes from its own C model values (WP8's review, 2026-11-07;
    the suite's conftest runs the warp route otherwise)."""
    import copy

    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)

    import mujoco

    from alloy.scene.tree_invweight import tree_sets

    tree, eq = _cable_scene(worlds=2), _cable_scene(break_force=1.0e9, worlds=2)
    st, se = _mujoco(tree), _mujoco(eq)
    plug = tree.handles[PLUG].template.body.start
    assert eq.handles[PLUG].template.body.start == plug
    np.testing.assert_allclose(_invweight(st, plug), _invweight(se, plug), rtol=2e-5)
    raw = copy.deepcopy(st.mj_model)
    mujoco.mj_setConst(raw, mujoco.MjData(raw))
    sets = tree_sets(st)
    b = int(np.flatnonzero(st.mjc_body_to_newton.numpy()[0] == plug)[0])
    assert b in sets.target and np.all(raw.body_invweight0[b] < 0.95 * _invweight(se, plug)[0])
    w = st.mjw_model.body_invweight0.numpy().reshape(-1, int(st.mj_model.nbody), 2)[0]
    np.testing.assert_allclose(w[sets.chain], raw.body_invweight0[sets.chain], rtol=1e-3)   # float32 vs float64
    assert getattr(se, "_alloy_tree_sets", None) is None
    _heavier_plug(tree, st, 1, 2.0)
    _heavier_plug(eq, se, 1, 2.0)
    t, e = _invweight(st, plug), _invweight(se, plug)
    assert np.all(e[1] < 0.75 * e[0])                                   # world 1 moved
    np.testing.assert_allclose(t, e, rtol=2e-5)


def test_constraint_only_notify_skips_the_rule_and_its_host_read(monkeypatch):
    """Topology notifies CONSTRAINT_PROPERTIES every exchange in a scene with equality rows, inside a captured period
    on a GPU; the rule's body_invweight0.numpy() there aborted the capture (DC5: the showcase at 1,024 worlds). That
    flag alone recomputes no MuJoCo constant, so the rule is skipped and the weights stay; an inertia change still
    re-applies it (the per-world mass law above)."""
    from alloy._vendor import newton
    from alloy.scene import tree_invweight as TI

    flags = newton.ModelFlags
    assert not TI.recomputes_constants(flags.CONSTRAINT_PROPERTIES)
    assert all(TI.recomputes_constants(f) for f in flags if f not in (flags.CONSTRAINT_PROPERTIES,) and int(f))
    assert TI.recomputes_constants(flags.CONSTRAINT_PROPERTIES | flags.BODY_INERTIAL_PROPERTIES)
    tree = _cable_scene(worlds=2)
    st = _mujoco(tree)
    plug = tree.handles[PLUG].template.body.start
    before = _invweight(st, plug)
    calls = []
    real = TI.apply_tree_invweight
    monkeypatch.setattr(TI, "apply_tree_invweight", lambda s: calls.append(s) or real(s))
    st.notify_model_changed(flags.CONSTRAINT_PROPERTIES)
    assert calls == [] and np.array_equal(_invweight(st, plug), before)
    st.notify_model_changed(flags.BODY_INERTIAL_PROPERTIES)
    assert len(calls) == 1
    np.testing.assert_allclose(_invweight(st, plug), before, rtol=2e-5)     # the rule put the chainless weight back


@pytest.mark.gpu
@pytest.mark.timeout(1800)      # cold cache: the cable's rod self-test (rodcheck, on the CPU) runs before any CUDA work
def test_constraint_only_notify_is_capture_safe():
    """On a GPU the per-exchange notification of a tree-welded scene records into a CUDA graph (no host read)."""
    import warp as wp

    from alloy._vendor import newton

    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    tree = _cable_scene(worlds=2, device="cuda:0")
    st = _mujoco(tree)
    assert getattr(st, "_alloy_tree_sets", None) is not None
    with wp.ScopedCapture(device="cuda:0") as capture:
        st.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    wp.capture_launch(capture.graph)


@pytest.mark.parametrize("break_force", [None, 1.0e9], ids=["tree", "equality"])
def test_welded_plug_and_link0_do_not_collide(break_force):
    """SCR-DC3-14 part 3: the weld's two bodies get shape filter pairs on both paths (before the explicit MuJoCo
    pair rows, which skip filtered pairs)."""
    built = _cable_scene(break_force=break_force)
    m = built.model
    body = m.shape_body.numpy()
    plug = set(np.flatnonzero(body == built.handles[PLUG].template.body.start).tolist())
    link0 = set(np.flatnonzero(body == built.handles[CABLE].template.body.start).tolist())
    assert plug and link0
    pairs = {(int(a), int(b)) for a, b in m.shape_collision_filter_pairs}
    assert all((min(a, b), max(a, b)) in pairs for a in plug for b in link0)



def test_chain_root_welded_to_a_static_part_does_not_collide_with_it():
    """SCR-DC3-14 part 3, static parent (body -1): `attach` leaves link 0 fixed to the world and filters it against
    the static part's own shapes, as the equality path does (builder level: a new clamped chain would need a cold rod
    self-test)."""
    from types import SimpleNamespace

    from alloy._vendor import newton
    import warp as wp

    from alloy.scene.tree_weld import attach

    b = newton.ModelBuilder()
    part = b.add_shape_box(-1, hx=0.01, hy=0.01, hz=0.01)
    link0 = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()), mass=1e-3)
    s0 = b.add_shape_capsule(link0, radius=0.001, half_height=0.01)
    j = b.add_joint_fixed(-1, link0, collision_filter_parent=False)
    b.add_articulation([j])
    chain = SimpleNamespace(name="cable", ranges=SimpleNamespace(joint=range(j, j + 1)))
    assert attach(b, chain, -1, (part,)) == j
    assert b.joint_parent[j] == -1
    assert (min(part, s0), max(part, s0)) in b.shape_collision_filter_pairs
