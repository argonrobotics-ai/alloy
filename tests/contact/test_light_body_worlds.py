# SPDX-License-Identifier: Apache-2.0
"""The light-body rule (`alloy.contact.light_body`) per world: each world's rows from its own masses (WP8's finding,
2026-11-07: every world took world 0's). Two worlds of a loose MJCF block on T12's 2.083 ms row, with the file's pair
to the floor. World 0's block is light (12.7 g: 1.66 ms at T12's step); world 1's is drawn heavy (5 kg: the file's
row is already stiffer, so it is kept), at build or by a mass change after the first step. On MuJoCo Warp (the
kernel reads each world's masses) and on the CPU's MuJoCo C route (each world model from its own masses and its own
pair rows)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.contact import light_body as LB
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.scene.builtin_plugins import builtin_capabilities
from alloy.scene.native_gains import stable_ke

DT = 0.0008333333333    # T12's step
TAU, DMAX, M_LIGHT, M_HEAVY = 0.0020833333333, 0.99, 0.0127, 5.0
TAU_PAIR_1 = 0.003      # world 1's own pair row (a per-world pair refit writes such rows)
SOLIMP = f"0.95 {DMAX} 0.001 0.5 2"
XML = f"""<mujoco><worldbody>
  <geom name="floor" type="plane" size="1 1 0.1" solref="{TAU} 1" solimp="{SOLIMP}"/>
  <body name="block" pos="0 0 0.0075"><freejoint/>
    <geom name="block_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" solref="{TAU} 1"
          solimp="{SOLIMP}"/>
  </body>
</worldbody>
<contact><pair geom1="floor" geom2="block_box" solref="{TAU} 1" solimp="{SOLIMP}"/></contact></mujoco>"""
ROUTES = ["warp", "mjc"]


def _tau_native(mass: float) -> float:
    ke = min(LB.K_PAIR * (1.0 - DMAX) / mass, stable_ke(1.0, DT, DMAX, "implicitfast"))
    return 1.0 / math.sqrt(ke)


def _scale_world(model: newton.Model, world: int, k: float) -> None:
    """World ``world``'s block mass and inertia x ``k`` (a density draw)."""
    body = int(np.flatnonzero(model.body_world.numpy() == world)[0])
    for name, power in (("body_mass", 1), ("body_inertia", 1), ("body_inv_mass", -1), ("body_inv_inertia", -1)):
        arr = getattr(model, name)
        v = arr.numpy()
        v[body] = v[body] * k ** power
        arr.assign(v)


def _model(heavy_at_build: bool) -> newton.Model:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(XML, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=2)
    model = b.finalize(device="cpu")
    if heavy_at_build:
        _scale_world(model, 1, M_HEAVY / M_LIGHT)
    rows = model.mujoco.pair_solref.numpy()
    rows[1] = (TAU_PAIR_1, 1.0)                     # world 1's pair row (row world * pairs + pair)
    model.mujoco.pair_solref.assign(rows)
    return model


def _step(model: newton.Model, solver) -> None:
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, model.control(), None, DT)


def _rows(solver, route: str) -> tuple[np.ndarray, np.ndarray]:
    """[world] the block's geom row and the pair row's tau, as the solver steps them."""
    m = solver.mj_model
    g = next(i for i in range(m.ngeom) if m.geom_bodyid[i] != 0)
    if route == "mjc":
        return (np.array([w.geom_solref[g, 0] for w in solver.worlds]),
                np.array([w.pair_solref[0, 0] for w in solver.worlds]))
    return solver.mjw_model.geom_solref.numpy()[:, g, 0], solver.mjw_model.pair_solref.numpy()[:, 0, 0]


def _solver(model: newton.Model, route: str):
    solver = next(c for c in builtin_capabilities() if c.name == "mujoco").factory(
        model, {CPU_ROUTE_KEY: route, "solver": "newton", "integrator": "implicitfast", "nconmax": 64, "njmax": 256})
    assert isinstance(solver, MjcCoreSolver) == (route == "mjc")
    return solver


@pytest.mark.parametrize("route", ROUTES)
def test_a_world_drawn_heavy_keeps_the_files_rows(route):
    """World 1 drawn at 5 kg: its block row keeps the file's 2.083 ms and its pair row its own 3 ms; world 0's light
    block gets 1.66 ms on both (on main world 1 took world 0's 1.66 ms rows)."""
    model = _model(heavy_at_build=True)
    solver = _solver(model, route)
    _step(model, solver)
    geom, pair = _rows(solver, route)
    assert geom[0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-5) and geom[1] == pytest.approx(TAU, rel=1e-6), geom
    assert pair[0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-5), pair
    assert pair[1] == pytest.approx(TAU_PAIR_1, rel=1e-6), pair


@pytest.mark.parametrize("route", ROUTES)
def test_a_mass_change_after_the_start_moves_the_rows(route):
    """World 1 drawn heavy at build (the file's 2.083 ms), then made light through notify_model_changed: from the next
    step its rows are 1.66 ms on both routes. Made heavy again, MuJoCo C's world goes back to the file's row (each
    world model is rebuilt from its own rows); MuJoCo Warp's keeps 1.66 ms, never softer than the file's (the kernel
    rewrites the rows in place and only stiffens; the module docstring)."""
    model = _model(heavy_at_build=True)
    solver = _solver(model, route)
    _step(model, solver)
    light = _tau_native(M_LIGHT)
    assert _rows(solver, route)[0][1] == pytest.approx(TAU, rel=1e-6)
    for k, want in ((M_LIGHT / M_HEAVY, light), (M_HEAVY / M_LIGHT, TAU if route == "mjc" else light)):
        _scale_world(model, 1, k)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _step(model, solver)
        geom, _ = _rows(solver, route)
        assert geom[0] == pytest.approx(light, rel=1e-5) and geom[1] == pytest.approx(want, rel=1e-5), (k, geom)
