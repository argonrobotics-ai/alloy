# SPDX-License-Identifier: Apache-2.0
"""Lever #2: a released (unarmed) proxy adds no coupling residual (``residuals.ResidualInputs.gated``; speed-now S2).

WP4's P3 chain (MuJoCo -> VBD lagged, VBD -> SPH staggered, so the residual loop runs k_min 2 .. k_max 8 passes) with
a one-link arm 0.6 m from the tank in place of the block, swung by a servo: the gate releases the arm's pair. Its
lagged velocity residual is the link's lag behind its source, a dt_x (the servo's acceleration, not gravity, which the
proxy shares), the same in every pass and above tol_vel (0.01 m/s), so before the fix every exchange ran k_max passes
(Panda + towel + water, 64 worlds: 8 passes per 1.25 ms exchange). CPU, the gate on, released residuals counted
(``set_index`` emptied) against skipped:
- nothing touches (the sheet still above the water): every pass repeats the first, so skipping the released residual
  gives the same state bit for bit, in k_min passes instead of k_max;
- the sheet floating on the water (its staggered pair carries force): the loop stops on that pair's own residual, so
  the state stays within the loop's tolerances of the k_max run, in fewer passes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]

PERIODS = 3
# a soft servo with little damping keeps the link swinging (period 0.16 s); the body origin sits 0.1 m from the hinge
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0.1 0 0">
  <joint name="shoulder" type="hinge" pos="-0.1 0 0" axis="0 1 0" range="-2 2"/>
  <geom type="capsule" fromto="-0.1 0 0 0.1 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20" kv="0.2"/></actuator></mujoco>"""


def _chain(sheet_z: float, count_released: bool) -> dict[str, np.ndarray]:
    """Run the chain with the gate on; per period passes (max over worlds) and the final physics state."""
    import alloy
    import alloy.coupling.compose as compose
    from alloy.coupling.contact_gate import gate_for

    if not compose.proxy_dag_supported():
        pytest.skip("Newton without N-entry proxy coupling (pin < 11-07)")
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tank", inner=(0.05, 0.05, 0.05), wall=0.006, pos=(0, 0, 0))
    scene.add_liquid("water", size=(0.05, 0.05, 0.016), pos=(0, 0, 0.008))
    scene.add_cloth("sheet", size=(0.04, 0.04), pos=(0, 0, sheet_z), resolution=6, mass=0.003)
    scene.add_robot("arm", mjcf=ARM, pos=(0.6, 0, 0.3))
    sim = alloy.build(scene, device="cpu")
    sim.set_targets({"shoulder": 1.5})
    eng, st = sim.engine, sim.engine.stepper
    st._alloy_policy = dataclasses.replace(st._alloy_policy,
                                           clock=dataclasses.replace(st._alloy_policy.clock, contact_gate=True))
    st._gate = gate = gate_for(st)
    if count_released:
        gate.set_index = {}                               # before the fix: released endpoints counted
    passes, exch, armed = [], [], []
    for _ in range(PERIODS):
        eng.step_period()
        rep = eng._last_coupling
        passes.append(int(np.max(rep.iters_used)))
        exch.append(int(eng.clock.steps_per_period))
        armed.append(gate.armed_host.copy())
    s = eng.state
    return {"passes": np.array(passes), "exch": np.array(exch), "armed": np.array(armed),
            "k": np.array([st.iteration_loop.cfg.k_min, st.iteration_loop.cfg.k_max]), "is_body": gate.ends.is_body,
            "body_q": s.body_q.numpy().copy(), "body_qd": s.body_qd.numpy().copy(),
            "particle_q": s.particle_q.numpy().copy(), "particle_qd": s.particle_qd.numpy().copy()}


def _released_block(run: dict[str, np.ndarray]) -> None:
    assert run["k"].tolist() == [2, 8]
    assert not run["armed"][1:, run["is_body"] == 1].any()      # the link's proxy released from period 2
    assert (run["exch"][1:] < run["exch"][0]).all()      # the released clock


def test_released_residual_skipped_same_state_bitwise_when_nothing_touches() -> None:
    before, after = _chain(0.06, count_released=True), _chain(0.06, count_released=False)
    _released_block(after)
    assert (before["passes"][1:] == 8).all() and (after["passes"][1:] == 2).all()
    for key in ("body_q", "body_qd", "particle_q", "particle_qd"):
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)


def test_released_residual_skipped_floating_sheet_within_the_loop_tolerance() -> None:
    before, after = _chain(0.020, count_released=True), _chain(0.020, count_released=False)
    _released_block(after)
    assert after["passes"][1:].max() < before["passes"][1:].max()
    for key, tol in (("particle_q", 1e-3), ("body_q", 1e-3)):         # tol_pos of a non-precision pair [m]
        d = np.abs(after[key][..., :3] - before[key][..., :3]).max()
        assert d <= tol, (key, d)
