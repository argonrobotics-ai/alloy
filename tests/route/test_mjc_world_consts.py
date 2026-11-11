# SPDX-License-Identifier: Apache-2.0
"""The CPU route's mass-derived constants per world (``alloy.route.mjc_core_consts``): a world drawn at another mass
carries the contact and limit weights, ``dof_M0`` and solver scale a one-world scene of that mass has, bit for bit,
and Newton's per-world Warp rows (what the pair refit reads) carry them too. Before, every world kept world 0's (the
closed known gap cpu-route-world-mass-softness: 12 mm after 0.5 s for a dropped box, and in CPU mode A a worker's
block took its first world's)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.route.mjc_core_consts import CONST_FIELDS
from alloy.scene.builtin_plugins import builtin_capabilities

pytestmark = pytest.mark.cpu_route

ARM = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="2 2 .1"/>
 <body pos="0 0 .3"><freejoint/><geom type="box" size=".05 .04 .03" mass=".5"/>
  <body pos=".1 0 0"><joint type="hinge" axis="0 1 0" range="-1 1" armature=".01"/>
   <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass=".3"/></body></body>
</worldbody></mujoco>"""


def _route(scale: list[float]) -> MjcCoreSolver:
    """A free box with a hinged link, ``len(scale)`` worlds, world w's masses and inertias x ``scale[w]``."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(ARM, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=len(scale))
    model = b.finalize(device="cpu")
    k = np.repeat(np.asarray(scale, np.float32), model.body_count // len(scale))
    for name, power in (("body_mass", 1), ("body_inertia", 1), ("body_inv_mass", -1), ("body_inv_inertia", -1)):
        arr = getattr(model, name)
        v = arr.numpy()
        arr.assign((v * (k if v.ndim == 1 else k[:, None, None]) ** power).astype(v.dtype))
    solver = next(c for c in builtin_capabilities() if c.name == "mujoco").factory(model, {CPU_ROUTE_KEY: "mjc"})
    assert isinstance(solver, MjcCoreSolver)
    DirectSim(model, solver, control_dt=0.016, substeps=8, collide=False, export_contacts=False)   # finalizes
    return solver


def test_a_drawn_world_has_the_constants_of_its_own_scene():
    """Worlds at 1x, 3x and 0.5x: each world model's mass-derived constants equal those of a one-world scene built at
    that mass, bit for bit; world 0 keeps the template's."""
    many = _route([1.0, 3.0, 0.5])
    for w, s in enumerate([1.0, 3.0, 0.5]):
        alone = _route([s]).worlds[0]
        mine = many.worlds[w]
        for f in CONST_FIELDS:
            assert np.array_equal(getattr(mine, f), getattr(alone, f)), (w, f)
        assert mine.stat.meanmass == alone.stat.meanmass, w
        # MuJoCo 3.17's mj_setConst leaves stat as compiled: Alloy derives meaninertia from M(qpos0), the compiler's
        # own definition, within the compiler's rounding (world 0 keeps the template's bit for bit)
        assert mine.stat.meaninertia == pytest.approx(alone.stat.meaninertia, rel=1e-9, abs=0), w
    assert not np.array_equal(many.worlds[1].body_invweight0, many.worlds[0].body_invweight0)
    assert np.array_equal(many.worlds[0].body_invweight0, many.mj_model.body_invweight0)


def test_newtons_per_world_rows_carry_them():
    """The Warp model's per-world weights (read by the pair refit) are each world's own, in float32."""
    many = _route([1.0, 3.0])
    rows = many.mjw_model.body_invweight0.numpy()
    for w in range(2):
        assert np.array_equal(rows[w], many.worlds[w].body_invweight0.astype(np.float32)), w


def _weld_scenes():
    """tests/scene/test_weld_impedance.py's cable scene (a plug on a welded cable chain)."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scene" / "test_weld_impedance.py"
    spec = importlib.util.spec_from_file_location("_weld_impedance", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)              # a cold MuJoCo Warp _linesearch compile passed 180 s on a Mac CPU (09:16Z run)
def test_the_tree_weld_weight_survives_a_model_change(monkeypatch):
    """C route, 2 worlds: each world's C model carries the tree-weld rule's chainless plug weight (SCR-DC3-14, equal
    to the equality path's) at build and after a body-inertia notify (world 1's plug 2x heavier). The per-world models
    are rebuilt before the next step, after the rule has run; before, they were rebuilt inside the notify, ahead of
    the rule, and both worlds took MuJoCo's tree weight (50.6 instead of 55.9 /kg). World 1's plug then takes world 0's
    chainless weight: the rule reads world 0's inputs on the CPU path (``tree_invweight.world_inputs``)."""
    from alloy.sim import Simulation

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    wi = _weld_scenes()
    tree, eq = wi._cable_scene(worlds=2), wi._cable_scene(break_force=1.0e9, worlds=2)
    sims = [Simulation(b) for b in (tree, eq)]
    st, se = (next(s for s in sim.stepper.solvers() if hasattr(s, "mj_model")) for sim in sims)
    assert isinstance(st, MjcCoreSolver) and isinstance(se, MjcCoreSolver)
    plug = tree.handles[wi.PLUG].template.body.start

    def c_weight(s):
        b = int(np.flatnonzero(s.mjc_body_to_newton.numpy()[0] == plug)[0])
        return np.array([m.body_invweight0[b] for m in s.worlds])

    for sim in sims:
        sim.step_period()
    np.testing.assert_allclose(c_weight(st), c_weight(se), rtol=2e-5)
    wi._heavier_plug(tree, st, 1, 2.0)
    wi._heavier_plug(eq, se, 1, 2.0)
    for sim in sims:
        sim.step_period()
    t, e = c_weight(st), c_weight(se)
    assert e[1, 0] < 0.75 * e[0, 0]                                    # the equality path's world 1 follows its mass
    np.testing.assert_allclose(t[0], e[0], rtol=2e-5)                   # world 0 keeps the chainless weight
