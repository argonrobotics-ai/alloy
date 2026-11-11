# SPDX-License-Identifier: Apache-2.0
"""Alloy's ordered SolverMuJoCo route (P3-1 round 5) on CPU: coverage, tripwire, scoping and equivalence.

GPU acceptance (R1/R2/R4, M7, graph capture on T12 at 1,024 worlds) is tests/precision/test_mjw_route_gpu.py.
"""

from __future__ import annotations

import types

import mujoco
import numpy as np
import pytest
import warp as wp

from alloy._vendor import mujoco_warp as mjw
from alloy.precision import mjw_launch, mjw_pin, mjw_route
from alloy.precision.determinism import _mjw_route_det, native_share
from alloy.core.protocols import DetCapability
from alloy.precision.types import Determinism

ARM = """
<mujoco>
  <option cone="{cone}" jacobian="{jac}" integrator="{integrator}" solver="{solver}" impratio="10"/>
  <worldbody>
    <geom type="plane" size="1 1 .1"/>
    <body name="l0" pos="0 0 .25" gravcomp="{gravcomp}">
      <joint name="j0" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 .2 0 0" size=".03"/>
      <body pos=".2 0 0" gravcomp="{gravcomp}"><joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 .2 0 0" size=".03"/>
        <body pos=".2 0 0" gravcomp="{gravcomp}"><joint name="j2" type="hinge" axis="0 1 0"/><geom type="box" size=".03 .03 .03"/>
          {sensor_site}</body>
      </body>
      <body pos="0 .1 0" gravcomp="{gravcomp}"><joint name="j3" type="hinge" axis="1 0 0"/><geom type="sphere" size=".03"/></body>
    </body>
    <body pos=".15 .12 .08"><freejoint/><geom type="box" size=".05 .05 .05"/></body>
    <body pos=".15 -.12 .08"><freejoint/><geom type="sphere" size=".05"/></body>
  </worldbody>
  <actuator><motor joint="j0" gear="5"/><motor joint="j1" gear="5"/><motor joint="j2" gear="2"/></actuator>
  {sensors}
</mujoco>
"""


def _xml(cone="elliptic", jac="sparse", solver="Newton", gravcomp=0, touch=False, integrator="implicitfast") -> str:
    site = '<site name="tip" size=".04"/>' if touch else ""
    sensors = '<sensor><touch site="tip"/></sensor>' if touch else ""
    return ARM.format(cone=cone, jac=jac, solver=solver, gravcomp=gravcomp, sensor_site=site, sensors=sensors,
                      integrator=integrator)


class SolverMuJoCo:  # the attributes enable_route reads from Newton's solver, over pure MuJoCo Warp
    def __init__(self, xml: str, nworld: int = 2) -> None:
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.mjw_model = mjw.put_model(self.mj_model)
        self.mjw_data = mjw.put_data(self.mj_model, self.mj_data, nworld=nworld, nconmax=32, njmax=128)
        self._deterministic = 0


@pytest.fixture(autouse=True)
def cpu_and_clean():
    with wp.ScopedDevice("cpu"):
        yield
    mjw_route.uninstall()


def _run(s: SolverMuJoCo, steps: int) -> np.ndarray:
    for _ in range(steps):
        mjw.step(s.mjw_model, s.mjw_data)
    return np.concatenate([s.mjw_data.qpos.numpy(), s.mjw_data.qvel.numpy()], axis=1)


def test_pin_sources() -> None:
    """Tripwire: the route was proven against these mujoco_warp sources; a pin bump reruns P3-1 round 5."""
    assert mjw_pin.changed() == [], "mujoco_warp changed under the ordered route: rerun test_mjw_route_gpu.py"


@pytest.mark.parametrize("solver", ["Newton", "CG"])  # CG: SCR-DC5-4 (ii), the CHAIN path, opt-in
def test_covered_model_is_enabled(solver: str) -> None:
    s = SolverMuJoCo(_xml(solver=solver))
    assert mjw_route.enable_route(s, allow_cg=solver == "CG") is None
    assert mjw_route.route_enabled(s)
    assert int(s.mjw_model.opt.deterministic) == mjw_route.SORT_BITS
    det = _mjw_route_det(s, DetCapability.WARP_LOWERING)
    assert det is not None and det.mode == Determinism.RUN_TO_RUN and det.reason is None
    assert native_share([s]) == 1.0


def test_gravcomp_arm_is_covered_by_the_ordered_sum() -> None:
    """Ordered gravity compensation (coordinator ruling 2026-11-06 06:23Z). An arm with gravcomp on four bodies takes
    the route, the route's per-dof sum replaces passive._gravity_force, and the result matches MuJoCo Warp's own atomic
    sum. On CPU the launches are serial, so the two differ only by the route's other ordered sums."""
    plain, routed = SolverMuJoCo(_xml(gravcomp=1)), SolverMuJoCo(_xml(gravcomp=1))
    assert mjw_route.enable_route(routed) is None and mjw_route.route_enabled(routed)
    before = mjw_launch.substituted_calls().get("passive._gravity_force", 0)
    mjw.step(plain.mjw_model, plain.mjw_data)
    mjw.step(routed.mjw_model, routed.mjw_data)
    assert mjw_launch.substituted_calls().get("passive._gravity_force", 0) == before + 1
    g = routed.mjw_data.qfrc_gravcomp.numpy()
    assert np.count_nonzero(g[:, :2]) == g.shape[0] * 2  # j0, j1 carry the shares of several bodies (j2, j3: none)
    np.testing.assert_allclose(g, plain.mjw_data.qfrc_gravcomp.numpy(), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kw, text", [
    ({"cone": "pyramidal"}, "PYRAMIDAL cone"),
    ({"jac": "dense"}, "dense Jacobian"),
    ({"integrator": "Euler"}, "integrator EULER"),
    ({"solver": "CG"}, "CG is opt-in"),
    ({"touch": True}, "sensor._sensor_touch"),
])
def test_uncovered_paths_keep_none(kw: dict, text: str) -> None:
    s = SolverMuJoCo(_xml(**kw))
    why = mjw_route.enable_route(s)
    assert why is not None and text in why, why
    assert not mjw_route.route_enabled(s)
    assert mjw_route.route_reason(s) == why
    det = _mjw_route_det(s, DetCapability.WARP_LOWERING)
    assert det is not None and det.mode == Determinism.NONE and text in det.reason
    assert native_share([s]) == 0.0


def test_tree_sums_match_mujoco_warp() -> None:
    """The one-launch range sums give MuJoCo Warp's tree sums up to rounding (same terms, body order)."""
    plain, routed = SolverMuJoCo(_xml()), SolverMuJoCo(_xml())
    mjw.set_const(plain.mjw_model, plain.mjw_data)  # MuJoCo Warp's constants on both sides (the route recomputes)
    _run(plain, 5)
    assert mjw_route.enable_route(routed) is None
    for s in (plain, routed):
        s.mjw_data.qpos.assign(plain.mjw_data.qpos.numpy())
        s.mjw_data.qvel.assign(plain.mjw_data.qvel.numpy())
        mjw.forward(s.mjw_model, s.mjw_data)
    a, b = plain.mjw_model.body_subtreemass.numpy(), routed.mjw_model.body_subtreemass.numpy()
    np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-7, err_msg="body_subtreemass")
    for f in ("subtree_com", "crb", "cfrc_int", "qfrc_bias", "qM" if hasattr(plain.mjw_data, "qM") else "M"):
        a, b = getattr(plain.mjw_data, f).numpy(), getattr(routed.mjw_data, f).numpy()
        np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6, err_msg=f)


def test_route_tracks_float_build() -> None:
    plain, routed = SolverMuJoCo(_xml()), SolverMuJoCo(_xml())
    assert mjw_route.enable_route(routed) is None
    a, b = _run(plain, 40), _run(routed, 40)
    assert int(routed.mjw_data.nacon.numpy()[0]) > 0
    assert np.abs(a - b).max() < 1e-4, np.abs(a - b).max()
    replaced = {f"{m.rsplit('.', 1)[1]}.{f}" for m, f in mjw_route.SUBSTITUTIONS}
    deep_tree_only = {"smooth._qLD_acc", "smooth._solve_LD_sparse_fused"}  # this arm's tree takes the dense blocks
    assert replaced - deep_tree_only <= set(mjw_launch.substituted_calls()) <= replaced, mjw_launch.substituted_calls()
    from alloy.precision.mjw_fixed import overruns

    assert overruns() == 0


def test_model_not_enabled_is_untouched() -> None:
    before = _run(SolverMuJoCo(_xml()), 20)
    routed = SolverMuJoCo(_xml())
    assert mjw_route.enable_route(routed) is None
    _run(routed, 2)
    after = _run(SolverMuJoCo(_xml()), 20)  # installed hooks, model not enabled: MuJoCo Warp's own kernels
    assert np.array_equal(before.view(np.uint32), after.view(np.uint32))


def test_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_MJW_ROUTE", "off")
    s = SolverMuJoCo(_xml())
    assert mjw_route.enable_route(s) == "ALLOY_MJW_ROUTE=off"


class _Model:  # weak-referenceable stand-in for a MuJoCo Warp Model
    pass


def test_substitution_is_scoped() -> None:
    """A substituted kernel's launch runs the handler only inside a scoped stage for an enabled model."""
    mod = types.ModuleType("alloy_fake_mjw")

    @wp.kernel
    def write_one(out: wp.array[float]):
        out[wp.tid()] = 1.0

    mod.factory = lambda flag: write_one
    mod.stage = lambda m, out: wp.launch(mod.factory(False), dim=out.shape[0], outputs=[out])
    import sys

    sys.modules["alloy_fake_mjw"] = mod
    seen = []
    try:
        mjw_launch.substitute("alloy_fake_mjw", "factory", lambda args, launch: seen.append((args, launch.dim)))
        mjw_launch.scope("alloy_fake_mjw", "stage")
        model_on, model_off = _Model(), _Model()
        mjw_launch.enable(model_on)
        out = wp.zeros(3, dtype=float)
        mod.stage(model_off, out)
        assert out.numpy().tolist() == [1.0, 1.0, 1.0] and not seen
        out.zero_()
        mod.stage(model_on, out)
        assert out.numpy().tolist() == [0.0, 0.0, 0.0] and seen == [((False,), 3)]
        wp.launch(mod.factory(False), dim=3, outputs=[out])  # outside any scope
        assert out.numpy().tolist() == [1.0, 1.0, 1.0]
    finally:
        mjw_launch.uninstall()
        del sys.modules["alloy_fake_mjw"]


@pytest.mark.parametrize("nworld", [1, 3])
def test_small_sorts_match_radix(monkeypatch, nworld: int) -> None:
    """The one-block rank sorts (few worlds / contact slots) give the radix path's order: identical trajectories."""
    from alloy.precision import mjw_order, mjw_rows

    runs = []
    for force_radix in (False, True):
        if force_radix:
            monkeypatch.setattr(mjw_rows, "SMALL_WORLDS", -1)
            monkeypatch.setattr(mjw_order, "SMALL_CONTACTS", -1)
        s = SolverMuJoCo(_xml(), nworld=nworld)
        assert mjw_route.enable_route(s) is None
        runs.append(_run(s, 30))
        mjw_route.uninstall()
    assert np.array_equal(runs[0].view(np.uint32), runs[1].view(np.uint32))


# --- coordinator ruling 2026-11-03 23:15Z: the route runs only on an explicit request ---------------------------
def _calls_to_enable_route(monkeypatch, cfg) -> list[bool]:
    """``enable_solver_determinism`` on a (fake-CUDA) SolverMuJoCo: the allow_cg of each enable_route call it makes."""
    from alloy.precision import determinism

    calls: list[bool] = []
    monkeypatch.setattr(determinism, "_device_of", lambda solvers: types.SimpleNamespace(is_cuda=True))
    monkeypatch.setattr(mjw_route, "enable_route", lambda s, allow_cg=False: calls.append(allow_cg))
    monkeypatch.setattr(mjw_route, "route_enabled", lambda s: False)
    determinism.enable_solver_determinism(cfg, [SolverMuJoCo(_xml())])
    return calls


def test_presets_do_not_turn_the_route_on(monkeypatch) -> None:
    from alloy.precision.config import preset
    from alloy.precision.types import GpuTier

    for name in ("standard", "precision", "reference", "fast"):
        assert _calls_to_enable_route(monkeypatch, preset(name, GpuTier.A)) == [], name


def test_the_route_runs_on_an_explicit_request(monkeypatch) -> None:
    import dataclasses

    from alloy.precision.config import preset
    from alloy.precision.types import GpuTier

    cfg = preset("standard", GpuTier.A)
    assert _calls_to_enable_route(monkeypatch, dataclasses.replace(cfg, mjw_route=True)) == [False]
    assert _calls_to_enable_route(monkeypatch, dataclasses.replace(cfg, mjw_route_cg=True)) == [True]
    fast = preset("fast", GpuTier.A)   # NONE requested: no route even when asked
    assert _calls_to_enable_route(monkeypatch, dataclasses.replace(fast, mjw_route=True)) == []


def test_a_solver_stepping_mujoco_c_is_refused() -> None:
    """SolverMuJoCo(use_mujoco_cpu=True) (WP8's MjcCoreSolver, SCR-WP8-6) keeps an unused MuJoCo Warp model: the route
    must not report itself enabled on it (WP8 2026-11-06)."""

    class _Solver:
        use_mujoco_cpu = True
        mjw_model = types.SimpleNamespace(opt=types.SimpleNamespace(deterministic=0))

    s = _Solver()
    why = mjw_route.enable_route(s)
    assert why is not None and "MuJoCo C" in why
    assert not mjw_route.route_enabled(s) and s.mjw_model.opt.deterministic == 0
