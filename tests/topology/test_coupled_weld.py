# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-21: a weld on a coupled MuJoCo entry breaks at its force, the same as on SolverMuJoCo alone.

Under a coupled stepper the MuJoCo entry runs on its entry ``ModelView``, whose equality arrays are its own copies.
The scene is DC5's composition, one-way SPH follower over Scheduler[(mujoco, vbd)]: the cloth scene of
``alloy.testing.coupling`` plus a SPH pool, and a body welded to the world beside them.
- Heavy body (weight over ``break_force``): coupled and alone break at the same step with the same weld force, the
  view's equality row goes off, and the body then falls the same way.
- Light body: neither breaks; both hold it the same way (MuJoCo's default soft weld lets it sag).
- A coupled view without its stepper logs a warning (the breaks could not reach MuJoCo).
- The view mirror maps parent rows through a non-identity projection.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverMuJoCo

from alloy.integration.stepper import entry_solvers
from alloy.testing import coupling as tc
from alloy.topology.bonds import add_weld_bond, bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.solver_hooks import EQ_FREQUENCY, EqViewMirror
from alloy.topology.types import TopologyBudget

SPEC = tc.ClothSceneSpec(cells=4)
BREAK_N = 23.4                     # DC5's UQD latch stand-in
HANG_Z, DT, STEPS = 1.0, 1.0e-3, 150


def _model(hang_kg: float) -> tuple[newton.Model, list[int]]:
    from alloy.solvers.sph.build import add_box_tank, add_fluid_block
    from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes

    cfg = DFSPHConfig(particle_radius=1.0e-3, coupling="one_way")
    b = tc._world_builder(SPEC)
    register_custom_attributes(b)
    s = cfg.spacing
    add_box_tank(b, (1.0, 0.0, 0.0), (1.0 + 4 * s, 4 * s, 7 * s), wall=cfg.h)
    first = b.particle_count
    add_fluid_block(b, cfg, (1.0, 0.0, 0.0), (4, 4, 3))
    sph = list(range(first, b.particle_count))
    hang = b.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, HANG_Z), wp.quat_identity()), label="hang")
    shape = b.default_shape_cfg.copy()
    shape.density = hang_kg / 0.1 ** 3
    b.add_shape_box(hang, hx=0.05, hy=0.05, hz=0.05, cfg=shape)
    add_weld_bond(b, hang, -1, break_force=BREAK_N)
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    register_custom_attributes(top)
    top.add_world(b)
    top.color()
    m = top.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_mu = SPEC.contact_ke, SPEC.contact_mu
    return m, sph


def _run(m: newton.Model, stepper, solvers: dict, coupled=None) -> dict:
    """Step with topology; returns the break step (-1: none), the weld force then, the hang body's z per step and
    the MuJoCo solver's own equality-enabled row at the end."""
    ctl = TopologyController(m, bonds_from_model(m), TopologyBudget(), entries=solvers, coupled=coupled)
    pipe = newton.CollisionPipeline(m)
    contacts = pipe.contacts()
    s0, s1, c = m.state(), m.state(), m.control()
    out = {"step": -1, "force": np.nan, "z": []}
    for k in range(STEPS):
        ctl.pre_step(s0, contacts, DT)
        pipe.collide(s0, contacts)
        stepper.step(s0, s1, c, contacts, DT)
        ctl.apply(s1)
        if out["step"] < 0 and not ctl.bonds.intact.numpy()[0]:
            out["step"], out["force"] = k, float(ctl.force.numpy()[0])
        out["z"].append(float(s1.body_q.numpy()[-1][2]))
        s0, s1 = s1, s0
    out["solver_eq"] = bool(ctl.hooks.mujoco[0].model.mujoco.equality_constraint_enabled.numpy()[0])
    return out


def _pair(hang_kg: float) -> tuple[dict, dict]:
    m, _ = _model(hang_kg)
    solo = SolverMuJoCo(m, use_mujoco_contacts=False, njmax=SPEC.njmax)
    alone = _run(m, solo, {"mujoco": solo})
    m2, sph = _model(hang_kg)
    follower = tc.cloth_sph_stepper(m2, sph, SPEC)
    return alone, _run(m2, follower, entry_solvers(follower), coupled=follower)


def test_weld_on_a_coupled_mujoco_entry_breaks_like_uncoupled():
    alone, coupled = _pair(20.0)
    print(f"break step alone {alone['step']} coupled {coupled['step']}; force {alone['force']:.2f} / "
          f"{coupled['force']:.2f} N; z end {alone['z'][-1]:.4f} / {coupled['z'][-1]:.4f} m")
    assert alone["step"] >= 0 and coupled["step"] == alone["step"]
    assert coupled["force"] == pytest.approx(alone["force"], rel=1e-4) and coupled["force"] >= BREAK_N
    assert not alone["solver_eq"] and not coupled["solver_eq"]
    np.testing.assert_allclose(coupled["z"], alone["z"], atol=1e-5)
    assert coupled["z"][-1] < HANG_Z - 0.08                       # released: about 0.11 m of free fall


def test_light_body_stays_welded_coupled_and_alone():
    alone, coupled = _pair(1.0)
    print(f"light: z end {alone['z'][-1]:.4f} / {coupled['z'][-1]:.4f} m, lowest {min(coupled['z']):.4f}")
    assert alone["step"] == coupled["step"] == -1
    assert alone["solver_eq"] and coupled["solver_eq"]
    np.testing.assert_allclose(coupled["z"], alone["z"], atol=1e-5)
    free_fall = HANG_Z - 0.5 * 9.81 * (STEPS * DT) ** 2
    assert coupled["z"][-1] > free_fall + 0.01                    # held, though MuJoCo's soft weld lets it sag


def test_coupled_view_without_its_stepper_warns(caplog):
    m, sph = _model(1.0)
    follower = tc.cloth_sph_stepper(m, sph, SPEC)
    with caplog.at_level(logging.WARNING, logger="alloy.topology"):
        TopologyController(m, bonds_from_model(m), TopologyBudget(), entries=entry_solvers(follower))
    assert any("coupled entry view" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_view_mirror_maps_parent_rows_through_the_projection():
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    for k in range(3):
        body = b.add_body(xform=wp.transform(wp.vec3(float(k), 0.0, 1.0), wp.quat_identity()), mass=1.0)
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        add_weld_bond(b, body, -1, break_force=1.0e9)
    m = b.finalize(device="cpu")
    m.mujoco.equality_constraint_enabled.assign([False, True, False])
    poses = np.zeros((3, 7), np.float32)
    poses[:, 0], poses[:, 6] = [1.0, 2.0, 3.0], 1.0
    m.mujoco.equality_constraint_relpose.assign(poses)
    view = SimpleNamespace(mujoco=SimpleNamespace(
        equality_constraint_enabled=wp.array([True, True], dtype=wp.bool, device="cpu"),
        equality_constraint_relpose=wp.zeros(2, dtype=wp.transform, device="cpu")))
    proj = SimpleNamespace(local_to_global=[2, 0], global_to_local=[1, -1, 0])      # row 1 is not in the view
    entry = SimpleNamespace(view=view, attribute_projections={EQ_FREQUENCY: proj})
    mirror = EqViewMirror.build(m, entry, np.array([0, 1, 2, -1]))
    assert mirror.src.numpy().tolist() == [0, 2] and mirror.dst.numpy().tolist() == [1, 0]
    mirror.launch()
    assert view.mujoco.equality_constraint_enabled.numpy().tolist() == [False, False]
    np.testing.assert_array_equal(view.mujoco.equality_constraint_relpose.numpy()[:, 0], [3.0, 1.0])
