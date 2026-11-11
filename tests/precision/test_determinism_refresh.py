# SPDX-License-Identifier: Apache-2.0
"""After the ordered MuJoCo route is enabled, a coupler re-installs the proxy constants it copied at setup (CPU, fakes).

job_bottle_route (WP4 2026-11-06): the route recomputed SolverMuJoCo's invweight0 under ordered sums, but the proxy
coupler kept the effective masses it had installed from the racing build-time values, so in-process rebuilds
diverged on GPU from period 1.
"""

from __future__ import annotations

from types import SimpleNamespace

from alloy._vendor import newton
from alloy.precision import determinism, mjw_route


class SolverMuJoCo:            # the check goes by type name, as enable_solver_determinism does
    pass


class _Coupler:
    def __init__(self, *solvers):
        self._entries = {f"e{i}": SimpleNamespace(solver=s) for i, s in enumerate(solvers)}
        self.flags = []

    def notify_model_changed(self, flags):
        self.flags.append(int(flags))


def test_a_coupler_of_a_routed_solver_is_notified(monkeypatch) -> None:
    mj, other = SolverMuJoCo(), object()
    monkeypatch.setattr(mjw_route, "route_enabled", lambda s: s is mj)
    c = _Coupler(other, mj)
    assert determinism.refresh_coupled_constants(c)
    assert c.flags == [int(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)]


def test_nothing_routed_or_no_coupler_is_left_alone(monkeypatch) -> None:
    monkeypatch.setattr(mjw_route, "route_enabled", lambda s: False)
    c = _Coupler(SolverMuJoCo())
    assert not determinism.refresh_coupled_constants(c) and c.flags == []
    assert not determinism.refresh_coupled_constants(SolverMuJoCo())      # a bare solver: no _entries
    assert not determinism.refresh_coupled_constants(None)


def test_enable_refreshes_a_coupler_passed_in(monkeypatch) -> None:
    mj = SolverMuJoCo()
    c = _Coupler(mj)
    seen = []
    monkeypatch.setattr(determinism, "refresh_coupled_constants", lambda s: seen.append(s) or False)
    monkeypatch.setattr(determinism, "_device_of", lambda solvers: SimpleNamespace(is_cuda=True))
    monkeypatch.setattr(mjw_route, "route_enabled", lambda s: True)        # already on: enable_route not called
    cfg = SimpleNamespace(determinism=1, mjw_route=True)   # RUN_TO_RUN
    determinism.enable_solver_determinism(cfg, [c])
    assert seen == [c]
