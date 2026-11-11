# SPDX-License-Identifier: Apache-2.0
"""Speed-now S2 lever: VBD iterations and Chebyshev rho scaled to the coupled step (``ClockPolicy.vbd_step_iterations``,
opt-in).

- the count: the route's at or above the step it was measured at (0.625 ms), n sqrt(h / 0.625 ms) below it, never under
  VBD's own 10 (15 for "super"'s 60) and never over the route's count;
- rho: the route's at or above 0.625 ms, e/(1 + e) below it with e = 9 (h / 0.625 ms)^2 for rho 0.9 (0.123 at 78 us);
- the solver's setter refuses a solver built without Chebyshev and AccelConfig's range;
- on WP4's P3 chain (MuJoCo block on a VBD sheet floating on SPH water, CPU): with the flag the sheet's entry runs the
  rule's count and rho for each attempt's dt_x and substeps, and both are in the graph key (another rho, another
  captured graph: a schedule's host weights never replay under another rho); without it, the route's 40 and 0.9.
"""

from __future__ import annotations

import dataclasses

import pytest

from alloy.coupling.vbd_iterations import FLOOR, H_MEASURED, step_iterations, step_rho


@pytest.mark.parametrize(("n", "h", "want"), [
    (40, H_MEASURED, 40), (40, 1.25e-3, 40), (40, 0.01 / 32, 29), (40, 0.01 / 128, 15), (40, 0.01 / 256, 10),
    (40, 1e-6, 10), (60, 0.01 / 128, 22), (10, 0.01 / 128, 10), (8, 0.01 / 128, 8), (40, 0.0, 40)])
def test_rule(n: int, h: float, want: int) -> None:
    assert step_iterations(n, h) == want
    assert FLOOR == 10


def test_super_keeps_more_iterations_than_the_default_at_every_step() -> None:
    for j in range(3, 12):
        h = 0.01 / 2 ** j
        assert step_iterations(60, h, floor=15) > step_iterations(40, h), j
    assert step_iterations(60, 0.01 / 1024, floor=15) == 15


@pytest.mark.parametrize(("h", "want"), [(H_MEASURED, 0.9), (1.25e-3, 0.9), (0.01 / 128, 0.140625 / 1.140625),
                                         (0.01 / 256, 0.03515625 / 1.03515625), (0.0, 0.9)])
def test_rho_tracks_the_sweep_radius(h: float, want: float) -> None:
    assert step_rho(0.9, h) == pytest.approx(want, rel=1e-12, abs=0)
    assert step_rho(0.9, h) <= 0.9


def test_the_setter_validates_and_needs_chebyshev_at_build() -> None:
    from alloy.solvers.vbd.accel import AccelConfig, AccelState

    on = AccelState(AccelConfig("adaptive", 0.9), 4, "cpu")
    on.set_rho(0.123)
    assert on.cfg.chebyshev_rho == 0.123 and on.cfg.initial_guess == "adaptive"
    with pytest.raises(ValueError):
        on.set_rho(1.0)
    with pytest.raises(ValueError, match="cannot be turned on later"):
        AccelState(AccelConfig("adaptive", None), 4, "cpu").set_rho(0.5)


def _chain_iterations(flag: bool) -> list[tuple[float, int, int, float]]:
    import alloy
    import alloy.coupling.compose as compose
    from alloy.integration.stepper import entry_solvers

    if not compose.proxy_dag_supported():
        pytest.skip("Newton without N-entry proxy coupling (pin < 11-07)")
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tank", inner=(0.05, 0.05, 0.05), wall=0.006, pos=(0, 0, 0))
    scene.add_liquid("water", size=(0.05, 0.05, 0.016), pos=(0, 0, 0.008))
    scene.add_cloth("sheet", size=(0.04, 0.04), pos=(0, 0, 0.020), resolution=6, mass=0.003)
    scene.add_box("block", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.037), mass=0.2)
    sim = alloy.build(scene, device="cpu")
    eng, st = sim.engine, sim.engine.stepper
    st._alloy_policy = dataclasses.replace(
        st._alloy_policy, clock=dataclasses.replace(st._alloy_policy.clock, vbd_step_iterations=flag))
    vbd = entry_solvers(st)["vbd"]
    rows = []
    for _ in range(2):
        eng.step_period()
        rep = eng._last_coupling
        rows.append((float(rep.dt_x), int(rep.substeps.get("vbd", 1)), int(vbd.iterations), vbd.chebyshev_rho))
    if flag:                                              # WP2's graph condition: another rho is another graph key
        key = st.graph_key()
        assert key[-1] == ("vbd", (("vbd", (rows[-1][2], rows[-1][3])),)), key
        st._vbd_scaler.apply(rows[-1][0] / 2, {"vbd": rows[-1][1]})
        assert st.graph_key() != key and vbd.chebyshev_rho == step_rho(0.9, rows[-1][0] / 2 / rows[-1][1])
    return rows


@pytest.mark.integration
@pytest.mark.timeout(1800)
def test_coupled_vbd_entry_runs_the_rule_with_the_flag_and_the_route_count_without() -> None:
    for dt_x, sub, n, rho in _chain_iterations(True):
        assert n == step_iterations(40, dt_x / sub) < 40, (dt_x, sub, n)
        assert rho == step_rho(0.9, dt_x / sub) < 0.9, (dt_x, sub, rho)
    assert {(n, rho) for _, _, n, rho in _chain_iterations(False)} == {(40, 0.9)}
