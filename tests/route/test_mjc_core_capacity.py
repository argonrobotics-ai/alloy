# SPDX-License-Identifier: Apache-2.0
"""The CPU route fails closed on capacity (SCR-WP8-8): contacts it drops set MuJoCo Warp's contact overflow bit
(``NARROWPHASE``) in that world's ``mjw_data.overflow``, which WP5's HealthMonitor (CONTACT_OVERFLOW) and the bully's
``capacity_overflow`` row read; on both plug-in backends, on the per-step loop and the fast path. The export holds at
least what the entry holds on MuJoCo Warp (Newton's resolved nconmax), so only contacts beyond that are flagged."""

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
from alloy.route.mjc_core_export import start_cap
from alloy.route.mjc_core_views import CONTACT_BIT, ROW_BIT, overflow_bits
from alloy.scene.builtin_plugins import builtin_capabilities

pytestmark = pytest.mark.cpu_route

XML = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>
 <body pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
 <body pos="0 0 .151"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
</worldbody></mujoco>"""                              # two stacked boxes on a floor: about 8 contacts per world
WORLDS, SUBSTEPS = 2, 8


def make_sim(backend: str, nconmax: int | None, cap: int | None = None, mirror: int | None = None,
             xml: str = XML) -> DirectSim:
    """The stack on the CPU route: ``nconmax`` on the entry (None: the route's default); after the build, ``cap``
    forces each world's export buffer (it then grows on demand) and ``mirror`` the pooled mirror the readers see
    (the route sizes both from the start's contacts: ``alloy.route.mjc_core_export``)."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(xml, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=WORLDS)
    model = b.finalize(device="cpu")
    kw = {CPU_ROUTE_KEY: "mjc"} if nconmax is None else {CPU_ROUTE_KEY: "mjc", "nconmax": nconmax}
    solver = next(c for c in builtin_capabilities() if c.name == "mujoco").factory(model, kw)
    assert isinstance(solver, MjcCoreSolver)
    if backend == "numpy" or cap is not None:
        solver.route = solver.route.replace(backend="numpy" if backend == "numpy" else "auto",
                                            ncon_max=solver.route.ncon_max if cap is None else cap)
        solver._make_outputs()
    if mirror is not None:
        solver.mjw_data.naconmax = mirror          # a full mirror: the only place the route still drops contacts
    if backend != "numpy" and solver.backend != "c":
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
    """Contacts beyond the pooled mirror (1 per world here) are dropped for its readers: every world's bit is set,
    though each world's own export held them all."""
    sim = make_sim(backend, nconmax=None, mirror=WORLDS)
    mon = run(sim, observe)
    assert (sim.route_period_reason() is None) == (not observe)
    assert sim.solver.route_report()["counters"]["contact_overflow"] == 0
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

    sim = make_sim(backend, nconmax=None, mirror=WORLDS)
    run(sim, observe=False)
    if not solver_sources(sim.solver):
        pytest.xfail("HealthMonitor does not read the route's mjw_data.overflow yet: health_adapters.solver_sources "
                     "keys on type(solver).__name__ == 'SolverMuJoCo' (WP5 ask: isinstance)")
    flags = sim.health.record.numpy()["flags"]
    assert np.all(flags & int(HealthFlags.CONTACT_OVERFLOW)), flags


@pytest.mark.parametrize("backend", ["c", "numpy"])
def test_an_entry_nconmax_below_the_start_exports_every_contact(backend: str) -> None:
    """An entry sized below its start's contacts (the stack: 12 per world, nconmax 1; a dense rig: 1,081 against the
    router's 400) holds them all on MuJoCo Warp, where Newton raises nconmax, and MuJoCo C solves them all: the route
    exports them all too, so health does not freeze the world from step 0."""
    sim = make_sim(backend, nconmax=1)
    cap = sim.solver.route.ncon_max                  # 1.5 x the 12 the start makes, mirrored for every world
    assert cap >= start_cap(12, 1) == 18 and sim.solver.get_max_contact_count() == cap * WORLDS, cap
    mon = run(sim, observe=True)
    assert sim.solver.route_report()["counters"]["contact_overflow"] == 0
    assert [o.ncon for o in sim.solver.outs] == [d.ncon for d in sim.solver.datas] and sim.solver.outs[0].ncon >= 8
    assert not sim.solver.mjw_data.overflow.numpy().any()
    assert mon.report()["capacity_overflow"].tolist() == [0.0] * WORLDS
    assert not np.any(sim.health.record.numpy()["flags"] & int(HealthFlags.CONTACT_OVERFLOW))


def test_export_cap_only_raises_and_keeps_no_export() -> None:
    from alloy.route.mjc_core_config import RouteConfig, export_cap

    assert export_cap(RouteConfig(ncon_max=400), 1081).ncon_max == 1081
    assert export_cap(RouteConfig(ncon_max=256), 48).ncon_max == 256
    assert export_cap(RouteConfig(ncon_max=0), 1081).ncon_max == 0


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
