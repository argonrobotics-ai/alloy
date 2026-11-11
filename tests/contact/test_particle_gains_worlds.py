# SPDX-License-Identifier: Apache-2.0
"""The MuJoCo-body contact gains land on the right records in a compacted multi-world view (WP4's review of
``alloy.contact.particle_gains``).

Two worlds of a cloth on the ground under a free box, beside a static box and a robot arm. The VBD entry's view is
compacted (Newton drops the arm's joints from it: 0 of 4), and compaction keeps shape ids global. Only records against
the free box's shapes, in both worlds, take the per-body share of the clock bound (that world's box)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.contact import particle_gains as PG

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.2"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20" kv="1"/></actuator></mujoco>"""


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0.0, 0.0, 0.003), resolution=8, mass=0.02, self_contact=False)
    scene.add_box("box", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.031), mass=0.2, material="wood_beech")
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0.0, 0.025), static=True)
    scene.add_robot("arm", mjcf=ARM, pos=(-0.3, 0.0, 0.0))
    s = alloy.build(scene, device="cpu")
    s.step(1)
    return s


@pytest.mark.compile_heavy
def test_records_on_the_free_box_in_both_worlds(sim):
    eng = sim.engine
    vbd = eng.parts.solvers["vbd"]
    model, view = eng.model, vbd.model
    assert view.joint_count < model.joint_count                    # the entry view is compacted (no arm joints) ...
    assert view.shape_count == model.shape_count                   # ... and its shape ids stay global
    bodies = np.asarray(sim.index.per_world(sim.index.body("box"), "body"))
    box_shapes = set(np.flatnonzero(np.isin(model.shape_body.numpy(), bodies)).tolist())
    assert len(box_shapes) == 2 and len({int(model.shape_world.numpy()[s]) for s in box_shapes}) == 2
    contacts = eng.stepper.get_proxy_contacts("mujoco", "vbd")
    n = int(contacts.soft_contact_count.numpy()[0])
    shape = contacts.soft_contact_shape.numpy()[:n]
    ke = vbd.body_particle_contact_penalty_k.numpy()[:n]
    kd = vbd.body_particle_contact_material_kd.numpy()[:n]
    on_box = np.isin(shape, list(box_shapes))
    assert {int(model.shape_world.numpy()[s]) for s in shape[on_box]} == {0, 1}
    body = view.shape_body.numpy()[shape[on_box]]                  # the box of each record's world, in the view
    m = 1.0 / vbd.body_inv_mass_effective.numpy()[body]
    records = vbd.body_particle_contact_counts.numpy()[body]
    dt, bound = float(eng._last_coupling.dt_x), 0.5                # ClockPolicy: safety 0.5 x omega_dt_max 1
    assert np.allclose(kd[on_box], bound * m / (dt * records), rtol=1e-5, atol=0.0)
    soft = vbd.model
    mix = 0.5 * (float(soft.soft_contact_ke) + soft.shape_material_ke.numpy()[shape])      # VBD's average
    assert np.allclose(ke[on_box], np.minimum(mix[on_box], bound**2 * m / (dt * dt * records)), rtol=1e-5, atol=0.0)
    assert np.allclose(ke[~on_box], mix[~on_box], rtol=1e-6, atol=0.0)     # ground and static block keep the mix


def test_a_view_with_other_shape_ids_keeps_vbds_mix(caplog):
    """A view whose shape count is not the Model's would misplace the gains: the view and the solver keep VBD's own
    values, with a warning (fails soft: the scene still builds; WP4's review)."""
    from types import SimpleNamespace

    gains = PG.ForceGains(np.array([1], np.int64), np.array([1e6], np.float32), model_shapes=4)
    view = SimpleNamespace(shape_count=3, name="vbd")
    solver = SimpleNamespace(model=view, _refresh_body_particle_contact_state=lambda c, r: None,
                             body_particle_contact_material_kd=None, body_particle_contact_counts=None,
                             body_inv_mass_effective=None)
    with caplog.at_level("WARNING", logger="alloy.contact"):
        PG.particle_view(gains)(view)
        assert PG.attach(solver, gains, 0.5) is False
    assert not hasattr(view, "shape_material_ke")
    assert sum("not global" in r.getMessage() for r in caplog.records) == 2
