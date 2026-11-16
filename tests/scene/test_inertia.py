# SPDX-License-Identifier: Apache-2.0
"""Regression (CARD-1, harness_lead): Newton's finalize raises any inertia whose smallest moment is below 1e-10 kg·m²
by 1e-6 on every diagonal; the assembler puts a valid authored inertia back (`alloy.scene.inertia`), but never one
under MuJoCo's limit (1e-14 kg·m² full, 1e-15 diagonal), and floors a kinematic body's before finalize (a 0.37 mm
thread segment at default density, 2026-11-15: MuJoCo raised a bare "inertia must have positive eigenvalues in
fullinertia" on both CPU routes, kinematic or free). The limit is MuJoCo's alone: a body another solver steps gets its
authored inertia back however small (a 0.1 m copper wire of radius 40 µm on the rod solver, WP5's review of
1993a8315: with Newton's 1e-6 it drooped 4.6 mm after 0.5 s instead of 11, and not at all beside a free box)."""

from __future__ import annotations

import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.api._thin import min_wall_extra
from alloy.scene import inertia
from alloy.scene.inertia import (floor_kinematic_inertia, mujoco_accepts, mujoco_bodies, restore_authored_inertia,
                                 valid_inertia)

LINK = np.diag([1.7e-10, 1.7e-10, 3.8e-11])      # a 5 mm harness-lead CHAIN link [kg·m²]
BAD = np.diag([1e-12, 1e-12, 1.0])               # breaks I1 + I2 >= I3: Newton's correction stays
TINY = np.diag([6.3e-16, 6.3e-16, 1.06e-15])     # valid, but under MuJoCo's 1e-15 for a diagonal tensor
LIMIT_WARNING = "under what MuJoCo accepts"


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    """The limit warning is once per body name per process: each law starts with none given."""
    monkeypatch.setattr(inertia, "_WARNED", set())


def _ours(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "alloy.api" and LIMIT_WARNING in r.getMessage()]


def _body(b: newton.ModelBuilder, mass: float, inertia: np.ndarray, kinematic: bool = False) -> int:
    k = b.add_body(mass=mass, inertia=inertia, is_kinematic=kinematic)
    b.add_shape_box(k, hx=0.0025, hy=0.0025, hz=0.0025, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    return k


def test_valid_inertia_rule():
    ok = valid_inertia(np.stack([LINK, BAD, -LINK, np.eye(3)]))
    assert ok.tolist() == [True, False, False, True]


def test_restore_authored_inertia_after_finalize():
    b = newton.ModelBuilder()
    link, bad = _body(b, 3e-4, LINK), _body(b, 1.0, BAD)
    with pytest.warns(Warning):
        model = b.finalize("cpu")
    assert model.body_inertia.numpy()[link][0, 0] > 1e-7            # Newton's floor: 1e-6, ~5,800x the link's
    assert restore_authored_inertia(b, model) == 1
    np.testing.assert_allclose(model.body_inertia.numpy()[link], LINK, rtol=1e-6)
    np.testing.assert_allclose(model.body_inv_inertia.numpy()[link], np.linalg.inv(LINK), rtol=1e-6)
    assert model.body_inertia.numpy()[bad][0, 0] > 1e-7             # invalid authored inertia keeps the correction
    assert restore_authored_inertia(b, model) == 0                  # idempotent: nothing left to restore


def test_mujoco_limit_rule():
    turned = np.array([[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    full = turned @ np.diag([2e-14, 3e-14, 4e-14]) @ turned.T                      # 2e-14 > 1e-14: accepted
    assert mujoco_accepts(np.stack([LINK, TINY, np.diag([2e-15] * 3), full, full * 0.4])).tolist() == \
        [True, False, True, True, False]


def test_restore_never_goes_under_mujocos_limit():
    b = newton.ModelBuilder()
    tiny = _body(b, 1e-8, TINY)
    with pytest.warns(Warning):
        model = b.finalize("cpu")
    assert restore_authored_inertia(b, model) == 0
    assert model.body_inertia.numpy()[tiny][0, 0] > 1e-7                # Newton's floored value is kept
    assert restore_authored_inertia(b, model, mujoco=np.array([False])) == 1     # another solver's body: restored
    np.testing.assert_allclose(model.body_inertia.numpy()[tiny], TINY, rtol=1e-6)


def test_mujoco_bodies_are_all_but_other_entries():
    def obj(entry: str, bodies: range) -> SimpleNamespace:
        return SimpleNamespace(decision=SimpleNamespace(entry=entry), ranges=SimpleNamespace(body=bodies))

    objs = [obj("mujoco", range(0, 1)), obj("vbd", range(1, 3)), obj("", range(3, 3))]
    assert mujoco_bodies(objs, 3, 2, 6).tolist() == [True, False, False, True, False, False]


def test_kinematic_inertia_floored_before_finalize_and_dynamic_left_alone():
    b = newton.ModelBuilder()
    kin, dyn = _body(b, 1e-8, TINY, kinematic=True), _body(b, 3e-4, LINK)
    assert floor_kinematic_inertia(b) == 1
    eig = np.linalg.eigvalsh(np.asarray(b.body_inertia[kin], np.float64).reshape(3, 3))
    assert eig.min() >= 1e-10 and eig.max() < 1.1e-10                  # Newton's absolute floor, no 1e-6 added
    np.testing.assert_allclose(np.asarray(b.body_inertia[dyn], np.float64).reshape(3, 3), LINK, rtol=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = b.finalize("cpu")
    floored = np.asarray(b.body_inertia[kin]).reshape(3, 3)
    np.testing.assert_allclose(model.body_inertia.numpy()[kin], floored, rtol=1e-6)
    assert floor_kinematic_inertia(b) == 0                              # idempotent


def _segment(length: float = 0.00037, r: float = 0.0001, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A closed 8-sided thread segment, 0.37 mm long (the sample's thread was 37 such meshes)."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
    v = np.vstack([np.c_[ring, np.full(n, -0.5 * length)], np.c_[ring, np.full(n, 0.5 * length)]])
    f = [[i, (i + 1) % n, n + (i + 1) % n] for i in range(n)] + [[i, n + (i + 1) % n, n + i] for i in range(n)]
    f += [[0, i + 1, i] for i in range(1, n - 1)] + [[n, n + i, n + i + 1] for i in range(1, n - 1)]
    return v, np.asarray(f)


@pytest.mark.parametrize("route", ["mjc", "warp"])
@pytest.mark.parametrize("kinematic", [True, False], ids=["kinematic", "free"])
def test_a_tiny_thread_segment_builds_and_steps(route, kinematic, monkeypatch, caplog):
    """Default density, the thin wall declared: a kinematic segment builds and steps with no inertia warning; a free
    one builds on Newton's floored inertia with one warning naming it."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    v, f = _segment()
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_mesh("seg", vertices=v, faces=f, pos=(0.0, 0.0, 0.05), rigid=True, card_extra=min_wall_extra(0.00018),
               **({"kinematic": True} if kinematic else {}))
    caplog.set_level(logging.WARNING, logger="alloy.api")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        sim = alloy.build(s, device="cpu")
        sim.step(5)
    assert sim.check()["ok"], sim.check()["problems"]
    solver = type(sim.engine.stepper.solver).__name__
    if route == "mjc" and not kinematic:                                # the C route refuses kinematic bodies
        assert solver == "MjcCoreSolver", solver
    newton_corrected = [w for w in seen if "Inertia validation corrected" in str(w.message)]
    ours = _ours(caplog)
    if kinematic:
        assert not newton_corrected and not ours, (newton_corrected, ours)
    else:
        assert newton_corrected and len(ours) == 1 and ours[0].startswith("seg: "), ours


# Tip drop of the wire below its clamp [mm] at (s), main 8eb799cd4 (before 1993a8315), CPU: bit for bit since
WIRE_DROP_MM = {"alone": {0.5: 11.03, 1.0: 19.62}, "box": {0.5: 8.57}}


@pytest.mark.cpu_route
@pytest.mark.parametrize("beside", ["alone", "box"])
def test_a_thin_copper_wire_on_the_rod_solver_keeps_its_inertia(beside, monkeypatch, caplog):
    """WP5's case: a 0.1 m copper wire of radius 40 µm clamped at its start (the rod solver; segments 4.5e-16 kg·m²),
    its thin wall declared, alone or beside a free box (MuJoCo C steps the box). It droops as before 1993a8315, within
    10 %, and no warning names its segments. With Newton's 1e-6 kept: 4.6 and 13.9 mm alone, 0 beside the box."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_rod("wire", length=0.1, radius=4e-5, pos=(0.0, 0.0, 0.3), clamp="start", material="copper_c110")
    s.items["wire"]["card_extra"] = {**s.items["wire"]["card_extra"], **min_wall_extra(8e-5)}
    if beside == "box":
        s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.3, 0.3, 0.025), mass=0.2)
    caplog.set_level(logging.WARNING, logger="alloy.api")
    sim = alloy.build(s, device="cpu")
    assert sim.routes()["wire"]["route"] == "rod_vbd"
    if beside == "box":
        assert "MjcCoreSolver" in [type(e).__name__ for e in sim.engine.solvers]
    got = {}
    for t in WIRE_DROP_MM[beside]:
        sim.step(int(round(t / sim.dt)) - int(round(sim.t / sim.dt)))
        p = sim.state(world=0)["wire.points"]
        got[t] = 1e3 * float(p[0, 2] - p[-1, 2])
    assert sim.check()["ok"], sim.check()["problems"]
    want = WIRE_DROP_MM[beside]
    assert all(abs(got[t] - want[t]) <= 0.1 * want[t] for t in want), (got, want)
    assert not _ours(caplog), _ours(caplog)                             # wire_edge_body_0..4 were named
