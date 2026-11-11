# SPDX-License-Identifier: Apache-2.0
"""The CPU route fails closed on capacity (SCR-WP8-8): contacts it drops set MuJoCo Warp's contact overflow bit
(``NARROWPHASE``) in that world's ``mjw_data.overflow``, which WP5's HealthMonitor (CONTACT_OVERFLOW) and the bully's
``capacity_overflow`` row read; on both plug-in backends, on the per-step loop and the fast path."""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.bully.direct import DirectSim
from alloy.bully.monitors import BenchMonitors
from alloy.bully.types import BullyLimits
from alloy.precision.types import HealthFlags
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_KEY
from alloy.route.mjc_core_views import CONTACT_BIT, ROW_BIT, overflow_bits
from alloy.scene.builtin_plugins import builtin_capabilities

pytestmark = pytest.mark.cpu_route

XML = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>
 <body pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
 <body pos="0 0 .151"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
</worldbody></mujoco>"""                              # two stacked boxes on a floor: about 8 contacts per world
WORLDS, SUBSTEPS = 2, 8


def make_sim(backend: str, nconmax: int | None) -> DirectSim:
    """The stack on the CPU route, ``nconmax`` exported contacts per world (None: the route's default)."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(XML, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=WORLDS)
    model = b.finalize(device="cpu")
    kw = {CPU_ROUTE_KEY: "mjc"} if nconmax is None else {CPU_ROUTE_KEY: "mjc", "nconmax": nconmax}
    solver = next(c for c in builtin_capabilities() if c.name == "mujoco").factory(model, kw)
    assert isinstance(solver, MjcCoreSolver)
    if backend == "numpy":
        solver.route = solver.route.replace(backend="numpy")
        solver._make_outputs()
    elif solver.backend != "c":
        pytest.skip(f"WP2's C plug-ins unavailable: {solver.backend_reason}")
    return DirectSim(model, solver, control_dt=SUBSTEPS * 0.002, substeps=SUBSTEPS, collide=False,
                     export_contacts=True)


def run(sim: DirectSim, observe: bool) -> BenchMonitors:
    """Three periods with the bully's monitors on ``mjw_data.overflow`` (as ``EpisodeRunner``): per step when
    ``observe`` (the per-step loop), else once per period (the fast path)."""
    mon = BenchMonitors(sim.model, BullyLimits(), overflow=sim.solver.mjw_data.overflow)
    if observe:
        sim.add_step_observer(mon.observer)
    for _ in range(3):
        sim.step_period()
        if not observe:
            mon.update(sim.state, sim.state, sim.control, None, sim.control_dt)
    return mon


@pytest.mark.parametrize("backend", ["c", "numpy"])
@pytest.mark.parametrize("observe", [True, False], ids=["per_step", "fast_path"])
def test_dropped_contacts_set_the_overflow_bit(backend: str, observe: bool) -> None:
    sim = make_sim(backend, nconmax=1)
    mon = run(sim, observe)
    assert (sim.route_period_reason() is None) == (not observe)
    assert sim.solver.route_report()["counters"]["contact_overflow"] > 0
    assert np.all(sim.solver.mjw_data.overflow.numpy() & CONTACT_BIT)
    assert mon.report()["capacity_overflow"].tolist() == [1.0] * WORLDS


@pytest.mark.parametrize("backend", ["c", "numpy"])
def test_no_bit_when_everything_fits(backend: str) -> None:
    sim = make_sim(backend, nconmax=None)
    mon = run(sim, observe=True)
    assert sim.solver.route_report()["counters"]["contact_overflow"] == 0
    assert not sim.solver.mjw_data.overflow.numpy().any()
    assert mon.report()["capacity_overflow"].tolist() == [0.0] * WORLDS
    assert not np.any(sim.health.record.numpy()["flags"] & int(HealthFlags.CONTACT_OVERFLOW))


@pytest.mark.parametrize("backend", ["c", "numpy"])
def test_health_latches_contact_overflow(backend: str) -> None:
    from alloy.precision.health_adapters import solver_sources

    sim = make_sim(backend, nconmax=1)
    run(sim, observe=False)
    if not solver_sources(sim.solver):
        pytest.xfail("HealthMonitor does not read the route's mjw_data.overflow yet: health_adapters.solver_sources "
                     "keys on type(solver).__name__ == 'SolverMuJoCo' (WP5 ask: isinstance)")
    flags = sim.health.record.numpy()["flags"]
    assert np.all(flags & int(HealthFlags.CONTACT_OVERFLOW)), flags


def test_a_short_contacts_buffer_flags_the_worlds_it_drops() -> None:
    sim = make_sim("numpy", nconmax=None)
    sim.step_period()
    assert not sim.solver.mjw_data.overflow.numpy().any()
    short = newton.Contacts(1, 0, device="cpu")
    sim.solver.update_contacts(short)
    assert int(short.rigid_contact_count.numpy()[0]) == 1
    assert sim.solver.mjw_data.overflow.numpy().tolist() == [CONTACT_BIT] * WORLDS   # world 0 keeps 1 of ~8
    sim.step_period()
    assert not sim.solver.mjw_data.overflow.numpy().any()                           # zeroed per call, as Newton


def test_overflow_word() -> None:
    assert overflow_bits(0, 0, 0) == 0
    assert overflow_bits(3, 0, 0) == overflow_bits(0, 1, 0) == CONTACT_BIT
    assert overflow_bits(0, 0, 2) == ROW_BIT
    assert overflow_bits(1, 1, 1) == CONTACT_BIT | ROW_BIT
