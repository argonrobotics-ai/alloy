# SPDX-License-Identifier: Apache-2.0
"""Audit S1-006: Warp's bare ``RuntimeError: Failed to allocate ... bytes`` left the API from ``sim.check()`` (its
contact probe collides every world at once: 512 worlds of 16 decomposed bowls on an L4) and from ``sim.step()`` (the
first periods' MuJoCo Warp collision scratch: 2,048 worlds of the audit's scene). Both must raise ``SimulationError``
(an ``AlloyAPIError``) with a fix; any other error keeps its type (CPU, the failures planted with Warp's message)."""

from __future__ import annotations

import pytest

import alloy
from alloy.check import live

WARP_OOM = "Failed to allocate 360087552 bytes on device 'cuda:0'"


def _sim() -> alloy.Sim:
    s = alloy.Scene(dt=0.01, worlds=2)
    s.add_ground()
    s.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.2)
    sim = alloy.build(s, device="cpu")
    sim.step(1)
    return sim


def _failing_probe(exc: BaseException):
    class Probe:
        def __init__(self, built):
            raise exc

    return Probe


def test_the_checks_out_of_memory_is_a_simulation_error_with_a_fix(monkeypatch):
    sim = _sim()
    assert sim.check()["ok"]
    monkeypatch.setattr(live, "ContactProbe", _failing_probe(RuntimeError(WARP_OOM)))
    with pytest.raises(alloy.SimulationError) as err:
        sim.check()
    assert isinstance(err.value, alloy.AlloyAPIError)
    assert "sim.health()" in err.value.fix and "2 worlds" in err.value.message and WARP_OOM in err.value.message
    assert sim.health()["worlds"]                         # the fix's way still works
    sim.close()


def test_other_errors_of_the_check_pass_unchanged(monkeypatch):
    sim = _sim()
    monkeypatch.setattr(live, "ContactProbe", _failing_probe(RuntimeError("some other failure")))
    with pytest.raises(RuntimeError, match="some other failure") as err:
        sim.check()
    assert not isinstance(err.value, alloy.AlloyAPIError)
    sim.close()


def _failing_period(exc: BaseException):
    def step_period(*args, **kwargs):
        raise exc

    return step_period


def test_the_steps_out_of_memory_is_a_simulation_error_with_a_fix(monkeypatch):
    sim = _sim()
    monkeypatch.setattr(sim.engine, "step_period", _failing_period(RuntimeError(WARP_OOM)))
    with pytest.raises(alloy.SimulationError) as err:
        sim.step(3)
    assert "contact_capacity" in err.value.fix and "2 worlds" in err.value.message and WARP_OOM in err.value.message
    assert str(err.value.__cause__) == WARP_OOM                 # chained, as raise ... from
    with pytest.raises(alloy.SimulationError):
        sim.run(0.05)                                     # run() steps through step()
    sim.close()


@pytest.mark.parametrize("exc", [RuntimeError("CUDA error: an illegal memory access"),
                                 ValueError("maximum representable value of a signed 32-bit integer")])
def test_other_step_errors_keep_their_type(monkeypatch, exc):
    sim = _sim()
    monkeypatch.setattr(sim.engine, "step_period", _failing_period(exc))
    with pytest.raises(type(exc)) as err:
        sim.step(1)
    assert err.value is exc and not isinstance(err.value, alloy.AlloyAPIError)
    sim.close()
