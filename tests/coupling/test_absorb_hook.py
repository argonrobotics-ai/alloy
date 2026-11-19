# SPDX-License-Identifier: Apache-2.0
"""Where the proxy scheduler calls the absorb coupling (``alloy.coupling.absorb_hook``; GAP-LC row B, WP4's review
M1/M2/M6). A recording stand-in is installed as ``alloy.coupling.liquid_cloth_absorb``; the scene is the scheduler
tests' box on cloth (CPU, 2 worlds)."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.core.errors import CouplingError
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling import absorb_hook as AH
from alloy.coupling.mixin import SchedulerMixin
from alloy.coupling.scheduler import Scheduler
from alloy.coupling.world_clock_params import WorldParams
from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth

SPEC = ClothSceneSpec(box_mass=0.2, cells=8)
DT = 1.0 / 480.0


class Recorder:
    """The absorb coupling's two calls, logged with what ``state_out`` holds when the exchange hands it over."""

    def __init__(self, solver, log: list) -> None:
        self.solver, self.log = solver, log

    def finish_exchange(self, state_out) -> None:
        self.log.append(("finish", id(state_out), state_out.particle_q.numpy().copy()))

    def restore(self) -> None:
        self.log.append(("restore",))


@pytest.fixture(scope="module")
def model2():
    return coupling_box_on_cloth(2, SPEC, device="cpu")


@pytest.fixture
def log(monkeypatch) -> list:
    out: list = []
    mod = types.ModuleType(AH.MODULE)
    mod.exchange_coupling = lambda solver: Recorder(solver, out)
    monkeypatch.setitem(sys.modules, AH.MODULE, mod)
    return out


def _steps(model, solver, n: int) -> list:
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    outs = []
    for _ in range(n):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, DT)
        outs.append((id(s1), s1.particle_q.numpy().copy()))
        s0, s1 = s1, s0
    return outs


def _spy(obj, name: str, log: list, tag: str) -> None:
    real = getattr(obj, name)

    def spy(*a, **k):
        log.append((tag,))
        return real(*a, **k)

    setattr(obj, name, spy)


def test_the_hooks_sit_around_the_scheduler_mixin() -> None:
    """Restore first in the MRO (``reset_worlds``' chain ends in MatchResetMixin, which calls no super); the exchange
    hook between SchedulerMixin and Newton's coupler, so it runs inside SchedulerMixin.step before the watchdog."""
    mro = Scheduler.__mro__
    assert mro.index(AH.AbsorbRestoreMixin) == 1
    assert mro.index(SchedulerMixin) < mro.index(AH.AbsorbExchangeMixin) < mro.index(SolverCoupledProxy)


def test_without_the_absorb_module_the_hook_is_none(monkeypatch, model2) -> None:
    monkeypatch.setitem(sys.modules, AH.MODULE, None)
    assert AH.cloth_absorb(object()) is None
    assert cloth_scheduler(model2, SPEC, vbd_adaptive=False)._cloth_absorb is None


def test_finish_runs_on_the_exchanges_output_before_the_watchdog(log, model2) -> None:
    """Each exchange: finish_exchange gets that exchange's state_out once, holding the coupler's final result (nothing
    writes it after), and the energy watchdog reads it only after."""
    sched = cloth_scheduler(model2, SPEC, vbd_adaptive=False)
    assert isinstance(sched._cloth_absorb, Recorder) and sched._cloth_absorb.solver is sched
    _spy(sched._watchdog, "after_exchange", log, "watchdog")
    outs = _steps(model2, sched, 3)
    assert [e[0] for e in log] == ["finish", "watchdog"] * 3
    for (tag, sid, q), (oid, q_end) in zip(log[0::2], outs, strict=True):
        assert sid == oid and np.array_equal(q, q_end)


def test_restore_runs_after_every_state_restore_and_before_the_datum(log, model2) -> None:
    """Rollback and replay restore first; sim.reset(worlds) restores after the scheduler's own per-world reset; the
    stepper's reset restores before the watchdog re-bases its energy datum."""
    sched = cloth_scheduler(model2, SPEC, vbd_adaptive=False)
    _steps(model2, sched, 2)
    _spy(sched._watchdog, "reset_reference", log, "datum")
    log.clear()
    sched.on_rollback(None)
    sched.replay_period()
    sched.reset_worlds(None)
    assert log == [("restore",)] * 3
    log.clear()
    sched.reset(model2.state())
    assert log == [("restore",), ("datum",)]


def test_fine_pack_refuses_a_world_move_with_absorbing_cloth() -> None:
    absorbing = types.SimpleNamespace(stepper=types.SimpleNamespace(_cloth_absorb=object()))
    dry = types.SimpleNamespace(stepper=types.SimpleNamespace(_cloth_absorb=None))
    params = WorldParams.__new__(WorldParams)
    params.tables, params.rows = (), []
    for src, dst in ((absorbing, dry), (dry, absorbing)):
        params.src, params.dst = src, dst
        assert "takes up water" in params.refusal(0, 1)


def test_admm_refuses_absorbing_cloth(log) -> None:
    with pytest.raises(CouplingError, match="absorb="):
        AH.refuse_with_admm(object())
