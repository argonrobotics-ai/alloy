# SPDX-License-Identifier: Apache-2.0
"""AlloyContactLayer + MixedContactBinding on CPU: sidecar, stock-solver buffer checks, append vs newton, guards."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.pipeline import AlloyContactLayer, world_shape_pairs
from alloy.contact.shadow import add_shadow_shapes, check_shadow_filters, set_flat_solimp
from alloy.contact.types import ContactClass, ContactFlags, PrecisionPairSpec
from alloy.core.errors import AssemblyError, PrecisionSpecError

K = 1.0e5
G = 9.81


def _sphere_on_plate(device: str, worlds: int = 2, filt: bool = True, flat_imp: bool = True):
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -0.01), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.01,
                            cfg=cfg)
    body = t.add_body(xform=wp.transform((0, 0, 0.0102), wp.quat_identity()))
    ball = t.add_shape_sphere(body, radius=0.01, cfg=cfg)
    if filt:
        t.add_shape_collision_filter_pair(plate, ball)
    if flat_imp:
        for shape in (plate, ball):
            set_flat_solimp(t, shape, 0.99)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_ground_plane(height=-0.5)
    b.replicate(t, worlds)
    return b.finalize(device=device), PrecisionPairSpec(plate, ball, mode="sdf", pair_stiffness=K)


def _run(model, layer, solver, steps: int = 100, dt: float = 1e-3):
    s0, s1, c = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(steps):
        layer.collide(s0, layer.contacts(), dt=dt)
        solver.step(s0, s1, c, layer.contacts(), dt)
        s0, s1 = s1, s0
    return s0


def test_world_shape_pairs_layout(device):
    model, spec = _sphere_on_plate(device, worlds=3)
    wp_ = world_shape_pairs(model, [spec])
    starts = model.shape_world_start.numpy()
    assert wp_.shape == (1, 3, 2)
    assert [tuple(x) for x in wp_[0]] == [(starts[w], starts[w] + 1) for w in range(3)]


def test_unfiltered_pair_is_rejected(device):
    model, spec = _sphere_on_plate(device, filt=False)
    with pytest.raises(AssemblyError, match="filter"):
        AlloyContactLayer(model, [spec])


def test_hydroelastic_on_newton_is_refused(device):
    """SB-3: the layer refuses a (mode, native) combination in ``precision.UNSUPPORTED_NATIVE``."""
    model, spec = _sphere_on_plate(device)
    hydro = dataclasses.replace(spec, mode="hydroelastic")
    with pytest.raises(PrecisionSpecError, match="native='mujoco'"):
        AlloyContactLayer(model, [hydro], native="newton")


def test_sidecar_rows(device):
    model, spec = _sphere_on_plate(device)
    layer = AlloyContactLayer(model, [spec], native="newton")
    solver = newton.solvers.SolverXPBD(model)
    _run(model, layer, solver, steps=5)
    n = int(layer.contacts().rigid_contact_count.numpy()[0])
    side = layer.sidecar()
    cls, pair, world = side.cls.numpy()[:n], side.pair.numpy()[:n], side.world.numpy()[:n]
    prec = pair >= 0
    assert prec.sum() == 2 and set(world[prec]) == {0, 1}
    assert np.all(cls[prec] == int(ContactClass.PRECISION_SDF))
    assert np.all(cls[~prec] == int(ContactClass.RIGID))
    assert np.allclose(side.weight.numpy()[:n][prec], 1.0)
    side.assemble(layer.contacts())
    sc = side.stream_count.numpy()
    assert sc[int(ContactClass.PRECISION_SDF)] == 2 and sc.sum() == n


@pytest.mark.compile_heavy
def test_newton_contacts_pass_stock_solvers(device):
    """Acceptance: the layer's Contacts go through stock XPBD and MuJoCo (Newton-contacts mode) unchanged."""
    model, spec = _sphere_on_plate(device)
    layer = AlloyContactLayer(model, [spec], native="newton")
    for solver in (newton.solvers.SolverXPBD(model),
                   newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, nconmax=32, njmax=128)):
        s = _run(model, layer, solver, steps=3)
        assert np.all(np.isfinite(s.body_q.numpy()))


@pytest.mark.compile_heavy
@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_rest_height_equals_f_over_k(device, native):
    """A 4.19 g ball rests at F/K below contact on both native paths (exact sphere-box narrow phase). The flat
    imp 0.99 (D22) keeps K_real = K at dt 1 ms: m/((1-imp)·4dt²) = 1.05e5 >= 1e5 N/m."""
    model, spec = _sphere_on_plate(device)
    layer = AlloyContactLayer(model, [spec], native=native)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=(native == "mujoco"), integrator="euler",
                                         cone="elliptic", nconmax=32, njmax=128,
                                         use_speculative_contacts=(native == "newton"))
    if native == "mujoco":
        layer.bind_mujoco(solver).install()
    s = _run(model, layer, solver)
    mass = float(model.body_mass.numpy()[0])
    sink = 0.01 - s.body_q.numpy()[:, 2]
    np.testing.assert_allclose(sink, mass * G / K, rtol=0.02)
    if native == "mujoco":
        binding = layer._binding
        binding.precision_forces(layer.contacts())
        n = int(layer.contacts().rigid_contact_count.numpy()[0])
        fz = np.abs(layer.contacts().rigid_contact_force.numpy()[:n, 2])
        np.testing.assert_allclose(fz, mass * G, rtol=1e-3)


def test_split_pipelines_merge(device, monkeypatch):
    """Worlds split over 3 precision pipelines (cap forced to 1 world each) merge into one buffer: every world rests
    at F/K, matched rows keep their weight as PERSISTENT (match index read at the source slot), nothing dropped."""
    import alloy.contact.pipeline as pl  # noqa: PLC0415

    monkeypatch.setattr(pl, "TRIANGLE_PAIR_CAP", 1)
    model, spec = _sphere_on_plate(device, worlds=3)
    layer = AlloyContactLayer(model, [spec], native="mujoco")
    assert len(layer.precision_streams) == 3
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, integrator="euler", cone="elliptic",
                                         nconmax=32, njmax=128)
    layer.bind_mujoco(solver).install()
    s = _run(model, layer, solver)
    mass = float(model.body_mass.numpy()[0])
    np.testing.assert_allclose(0.01 - s.body_q.numpy()[:, 2], mass * G / K, rtol=0.02)
    n = int(layer.contacts().rigid_contact_count.numpy()[0])
    side = layer.sidecar()
    assert n == 3 and sorted(side.world.numpy()[:n]) == [0, 1, 2]
    assert np.all(side.flags.numpy()[:n] & int(ContactFlags.PERSISTENT))
    assert sum(layer.overflow_report().values()) == 0


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("integrator", "ccd", "ok"), [("euler", "speculative", True), ("implicitfast", "off", True),
                                                       ("discrete", "off", True), ("discrete", "speculative", False),
                                                       ("rk4", "off", False)])
def test_install_integrator_table(device, integrator, ccd, ok):
    model, spec = _sphere_on_plate(device)
    spec = PrecisionPairSpec(spec.shape_a, spec.shape_b, mode="sdf", pair_stiffness=K, ccd=ccd)
    layer = AlloyContactLayer(model, [spec])
    solver = newton.solvers.SolverMuJoCo(model, integrator=integrator, nconmax=32, njmax=128)
    binding = layer.bind_mujoco(solver)
    if ok:
        binding.install()
        assert binding.installed
    else:
        with pytest.raises(PrecisionSpecError):
            binding.install()


def test_bind_needs_native_mujoco(device):
    model, spec = _sphere_on_plate(device)
    layer = AlloyContactLayer(model, [spec], native="newton")
    with pytest.raises(PrecisionSpecError):
        layer.bind_mujoco(object())


def test_collide_rejects_foreign_buffer_and_bad_dt(device):
    model, spec = _sphere_on_plate(device)
    layer = AlloyContactLayer(model, [spec], native="newton")
    s = model.state()
    with pytest.raises(AssemblyError):
        layer.collide(s, newton.Contacts(8, 0, device=device), dt=1e-3)
    with pytest.raises(ValueError):
        layer.collide(s, layer.contacts(), dt=0.0)


def test_speculative_flag_marks_separated_rows(device):
    model, spec = _sphere_on_plate(device)
    layer = AlloyContactLayer(model, [spec], native="newton")
    s = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s)          # ball 0.2 mm above the plate (gap 0.5 mm)
    layer.collide(s, layer.contacts(), dt=1e-3)
    n = int(layer.contacts().rigid_contact_count.numpy()[0])
    side = layer.sidecar()
    prec = side.pair.numpy()[:n] >= 0
    assert prec.any()
    assert np.all(side.flags.numpy()[:n][prec] & int(ContactFlags.SPECULATIVE))


def test_shadow_shapes_and_filters(device):
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.3)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, hx=0.05, hy=0.05, hz=0.01, cfg=cfg)
    body = t.add_body(xform=wp.transform((0, 0, 0.03), wp.quat_identity()))
    ball = t.add_shape_sphere(body, radius=0.01, cfg=cfg)
    other = t.add_shape_box(-1, xform=wp.transform((0.2, 0, 0), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.01, cfg=cfg)
    spec = PrecisionPairSpec(plate, ball, mode="sdf", gap=5e-4, margin=1e-5)   # hydroelastic primitives need CUDA
    sa, sb = add_shadow_shapes(t, spec)
    assert t.shape_body[sa] == t.shape_body[plate] and t.shape_body[sb] == t.shape_body[ball]
    assert t.shape_gap[sa] == pytest.approx(5e-4) and t.shape_margin[sb] == pytest.approx(1e-5)
    assert t.custom_attributes["mujoco:geom_solimp"].values[sa][1] == pytest.approx(0.99)
    pairs = set(t.shape_collision_filter_pairs)
    assert (min(plate, ball), max(plate, ball)) in pairs
    assert (min(sa, other), max(sa, other)) in pairs and (min(sa, sb), max(sa, sb)) not in pairs
    model = t.finalize(device=device)
    check_shadow_filters(model, [(sa, sb)])
    late = newton.ModelBuilder()
    late.add_builder(t)
    late.add_shape_box(-1, xform=wp.transform((-0.2, 0, 0), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.01, cfg=cfg)
    with pytest.raises(AssemblyError, match="not filtered"):
        check_shadow_filters(late.finalize(device=device), [(sa, sb)])
