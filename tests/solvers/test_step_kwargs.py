# SPDX-License-Identifier: Apache-2.0
"""Regression tests for forwarding extra step() keyword arguments in solver subclasses and mixins."""

from __future__ import annotations

from types import SimpleNamespace

from alloy.solvers.mpm_rheology.solver import SolverRheologyMPM
from alloy.solvers.vbd.reach import with_reach
from alloy.solvers.vbd.self_contact_rows import with_rows


def test_vbd_reach_and_rows_mixins_forward_step_kwargs():
    """ReachMixin and RowsMixin must forward extra step() keyword arguments such as observables."""
    seen: list[dict] = []

    class _Base:
        def __init__(self, model, *args, **kwargs):
            self.particle_enable_self_contact = False
            self.collision_pipeline = None

        def step(self, state_in, state_out, control, contacts, dt, **kwargs):
            seen.append(dict(kwargs))
            return "ok"

    model = SimpleNamespace(tri_count=0)
    marker = object()
    r = with_reach(_Base)(model, alloy_self_contact_speed=1.0)
    w = with_rows(_Base)(model)
    assert r.step(None, None, None, None, 0.01, observables=marker) == "ok"
    assert w.step(None, None, None, None, 0.01, observables=marker) == "ok"
    assert seen == [{"observables": marker}, {"observables": marker}]


def test_mpm_rheology_solver_forwards_step_kwargs(monkeypatch):
    """SolverRheologyMPM.step must forward extra keyword arguments such as observables to SolverImplicitMPM.step."""
    from alloy._vendor.newton.solvers import SolverImplicitMPM

    seen: list[dict] = []
    monkeypatch.setattr(SolverImplicitMPM, "step", lambda self, si, so, c, ct, dt, **kw: seen.append(dict(kw)))

    solver = object.__new__(SolverRheologyMPM)
    solver._hb = SimpleNamespace(shape=(0,))
    solver.rheology = None
    solver.model = SimpleNamespace(device="cpu")
    solver._grid = None
    marker = object()
    solver.step(None, None, None, None, 0.01, observables=marker)
    assert seen == [{"observables": marker}]
