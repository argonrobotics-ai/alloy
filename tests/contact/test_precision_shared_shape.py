# SPDX-License-Identifier: Apache-2.0
"""Two precision pairs that share a shape both keep their precision contact (2026-11-20).

The layer's per-shape table held one pair per shape, so a second pair on the same shape (one plate, two balls; one
hole plate, two pegs) overwrote the first: the first pair's rows came out as ordinary rigid rows, without the pair's
stiffness, and its ball sank onto MuJoCo's soft default contact instead of resting at F/K."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.shadow import set_flat_solimp
from alloy.contact.types import ContactClass, PrecisionPairSpec

K = (1.0e5, 5.0e4)      # one K per ball; both under the flat-imp cap m/((1-imp)·4dt²) = 1.05e5 N/m at dt 1 ms
G = 9.81
R = 0.01


def _two_balls_on_one_plate(device: str, worlds: int = 2):
    """A static plate (the shared shape) and two 4.19 g balls touching it, one precision pair per ball."""
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -0.01), wp.quat_identity()), hx=0.1, hy=0.05, hz=0.01,
                            cfg=cfg)
    balls = []
    for x in (-0.05, 0.05):
        body = t.add_body(xform=wp.transform((x, 0, R + 2e-4), wp.quat_identity()))
        balls.append(t.add_shape_sphere(body, radius=R, cfg=cfg))
    for shape in (plate, *balls):
        set_flat_solimp(t, shape, 0.99)
    for ball in balls:
        t.add_shape_collision_filter_pair(plate, ball)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_ground_plane(height=-0.5)
    b.replicate(t, worlds)
    specs = [PrecisionPairSpec(plate, ball, mode="sdf", pair_stiffness=k) for ball, k in zip(balls, K)]
    return b.finalize(device=device), specs


def _rows(layer: AlloyContactLayer) -> dict[str, np.ndarray]:
    c, side = layer.contacts(), layer.sidecar()
    n = int(c.rigid_contact_count.numpy()[0])
    return {"pair": side.pair.numpy()[:n], "cls": side.cls.numpy()[:n], "world": side.world.numpy()[:n],
            "k": c.rigid_contact_stiffness.numpy()[:n], "weight": side.weight.numpy()[:n]}


def test_both_pairs_on_a_shared_plate_are_precision_rows(device):
    """One collide: every world has one row of each pair, classed precision and carrying that pair's K (on main the
    first pair's rows were ordinary rigid rows with no pair)."""
    model, specs = _two_balls_on_one_plate(device)
    layer = AlloyContactLayer(model, specs, native="newton")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    layer.collide(state, layer.contacts(), dt=1e-3)
    r = _rows(layer)
    for p, k in enumerate(K):
        mine = r["pair"] == p
        assert sorted(r["world"][mine]) == [0, 1], (p, r["pair"].tolist(), r["cls"].tolist())
        assert np.all(r["cls"][mine] == int(ContactClass.PRECISION_SDF))
        np.testing.assert_allclose(r["k"][mine], k * r["weight"][mine], rtol=1e-6)
    assert np.all(r["pair"] >= 0), r["pair"].tolist()


@pytest.mark.compile_heavy
def test_two_balls_on_a_shared_plate_rest_at_their_own_f_over_k(device):
    """MuJoCo with the precision rows appended: each ball rests F/K below contact with its own pair's K (on main the
    first ball sank onto MuJoCo's soft default contact)."""
    model, specs = _two_balls_on_one_plate(device)
    layer = AlloyContactLayer(model, specs, native="mujoco")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True, integrator="euler", cone="elliptic",
                                         nconmax=32, njmax=128)
    layer.bind_mujoco(solver).install()
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(100):
        layer.collide(s0, layer.contacts(), dt=1e-3)
        solver.step(s0, s1, ctrl, layer.contacts(), 1e-3)
        s0, s1 = s1, s0
    mass = float(model.body_mass.numpy()[0])
    sink = (R - s0.body_q.numpy()[:, 2]).reshape(-1, 2)            # [world, ball]
    want = np.broadcast_to(mass * G / np.asarray(K), sink.shape)
    np.testing.assert_allclose(sink, want, rtol=0.02)
