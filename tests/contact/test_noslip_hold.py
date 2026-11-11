# SPDX-License-Identifier: Apache-2.0
"""Opt-in static hold (`alloy.contact.noslip_hold`, coordinator ruling 2026-11-04 01:36Z). Friction-held contacts
below the cone creep <= 1 um/s at <= 0.5 mu N with `alloy:noslip_all` on the MuJoCo entry, on the A5 box rig and on
DC3's QSFP check scene. rk4 entries keep the creep (CPU)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.noslip_hold import HOLD_ITERATIONS, HOLD_KEY, HOLD_TOLERANCE, hold_request, install_hold
from alloy.scene.builtin_plugins import _mujoco
from alloy.testing import dcconn

MU, HALF, DT = 0.5, 0.025, 1.25e-3
CREEP_MAX = 1e-6          # m/s
ROUTE = {"solver": "newton", "cone": "elliptic", "impratio": 300.0, "integrator": "implicitfast"}


def test_hold_request():
    assert hold_request(True) == (HOLD_ITERATIONS, HOLD_TOLERANCE)
    assert hold_request(20) == (20, HOLD_TOLERANCE)
    assert hold_request((30, 1e-8)) == (30, 1e-8)
    assert hold_request(None) is None and hold_request(False) is None and hold_request(0) is None


def _box_rig(load: float) -> newton.Model:
    """The A5 box (50 mm, 1000 kg/m^3, mu 0.5 on box and plane) under gravity tilted to tan(th) = load x mu."""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=MU))
    body = b.add_body(xform=wp.transform((0.0, 0.0, HALF), wp.quat_identity()))
    b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(mu=MU, density=1000.0))
    m = b.finalize(device="cpu")
    th = math.atan(load * MU)
    m.gravity.assign(np.array([[9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)]] * m.gravity.shape[0], np.float32))
    return m


def _box_creep(kw: dict, t0: float = 0.5, t1: float = 1.5) -> float:
    """Along-load speed [m/s] of the box over [t0, t1] s at 0.5 mu N through the MuJoCo entry factory."""
    model = _box_rig(0.5)
    solver = _mujoco(model, kw)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    x = {}
    for k in range(int(round(t1 / DT))):
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, DT)
        s0, s1 = s1, s0
        for t in (t0, t1):
            if abs((k + 1) * DT - t) < DT / 2:
                x[t] = float(s0.body_q.numpy()[0, 0])
    return (x[t1] - x[t0]) / (t1 - t0)


def test_box_rig_creeps_on_the_route_and_holds_with_the_opt_in():
    assert _box_creep(dict(ROUTE)) > CREEP_MAX                       # 2.07 um/s at impratio 300
    assert abs(_box_creep({**ROUTE, HOLD_KEY: True})) <= CREEP_MAX


def test_mujoco_c_entries_get_mujocos_own_noslip():
    """The MuJoCo C entry keeps an mjw_model too: the hold must still take MuJoCo's own noslip there (it chose the
    scoped one and raised before)."""
    model = _box_rig(0.5)
    solver = newton.solvers.SolverMuJoCo(model, **{**ROUTE, "use_mujoco_cpu": True})
    assert install_hold(solver) == "mujoco"
    assert int(solver.mj_model.opt.noslip_iterations) == HOLD_ITERATIONS
    assert abs(_box_creep({**ROUTE, "use_mujoco_cpu": True, HOLD_KEY: True})) <= CREEP_MAX


def test_rk4_entries_keep_the_creep():
    model = _box_rig(0.5)
    solver = newton.solvers.SolverMuJoCo(model, **{**ROUTE, "integrator": "rk4"})
    assert install_hold(solver) == "off"
    assert int(solver.mj_model.opt.noslip_iterations) == 0


@pytest.mark.skipif(dcconn.assets_root() is None, reason="alloy-assets checkout not found")
def test_seated_qsfp_plug_holds_with_the_opt_in():
    """DC3's check scene: the latch pushes the seated plug out at about 0.25 mu N (109 um/s on the route at impratio
    300); with the hold it stays put."""
    from alloy.tasks import dc_conn_qsfp_cards as C
    from alloy.tasks import dc_conn_qsfp_sim as S

    sim = S.build(C.check_scene(cable=False), impratio=300.0, entry={HOLD_KEY: True})   # impratio 1 slides at once
    probe = S.Probe(sim)
    dcconn.run_periods(sim, 0.2)
    a = probe.read()
    dcconn.run_periods(sim, 1.0)
    b = probe.read()
    assert dcconn.invalid_worlds(sim) == ()
    assert bool(b["latched"][0])
    assert abs(b["x"][0] - a["x"][0]) * 1e-3 <= CREEP_MAX * 1.0      # probe x in mm, 1.0 s apart
    assert b["x"][0] > -0.2          # held where it stopped (about -0.10 mm), not on the catch (-0.31 mm)
