# SPDX-License-Identifier: Apache-2.0
"""The CPU route's contact copy follows MuJoCo C's own count (``alloy.route.mjc_core_export``; physics 2026-11-14:
the copy is for reporting only, so CPU users must not read CONTACT_OVERFLOW where MuJoCo C solved every contact).

- the pooled mirror the readers size their buffers from is sized in the solver's constructor, before any reader
  exists: 1.5 x the start's contacts when they outgrew the entry's nconmax, else the entry's own size (at least the
  route's cap), every contact array grown alike;
- a world whose contacts outgrow its export buffer mid-run grows it and copies that call's contacts again: the bully's
  exported ``Contacts`` (sized at build), its capacity monitor and health, all attached from the start, see every
  contact, and nothing is flagged;
- only contacts beyond the pooled mirror are still flagged (``test_mjc_core_capacity``).
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.precision.types import HealthFlags
from alloy.route.mjc_core_export import MARGIN, grow_mirror, start_cap

from test_mjc_core_capacity import WORLDS, make_sim, run

pytestmark = pytest.mark.cpu_route

DROP = """<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="1 1 .1"/>
 <body pos="0 0 .05"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
 <body pos="0 0 .151"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
 <body pos="0 0 .30"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
</worldbody></mujoco>"""                    # the stack, and a third box 4.9 cm above it that lands about 0.1 s in


def test_start_cap_and_grow_keep_every_array_alike():
    assert start_cap(1081, 400) == 1622 and MARGIN == 1.5                  # Newton raised 400 to the start's
    assert start_cap(8, 8) == 8 and start_cap(48, 0) == 48 and start_cap(0, 0) == 0   # the entry's own held
    sim = make_sim("numpy", nconmax=1)
    d = sim.solver.mjw_data
    pooled = int(d.naconmax)
    rows = {k: getattr(d.contact, k).shape[0] for k in ("dist", "pos", "frame", "geom", "worldid", "efc_address")}
    assert set(rows.values()) == {pooled}
    assert grow_mirror(d, pooled // WORLDS - 1, WORLDS) == pooled           # never shrinks
    assert grow_mirror(d, pooled // WORLDS + 4, WORLDS) == pooled + 4 * WORLDS
    assert {getattr(d.contact, k).shape[0] for k in rows} == {pooled + 4 * WORLDS}
    assert int(d.contact.efc_address.numpy()[-1].min()) == -1                # new rows take the initial fill


@pytest.mark.parametrize("backend", ["c", "numpy"])
def test_a_world_that_outgrows_its_export_mid_run_drops_nothing(backend: str) -> None:
    """Each world's export starts at what the stack makes before the third box lands (forced: the route would size
    it 1.5x above); the box lands in period 7 or so and adds contacts. The export grows then, once per world; the
    exported ``Contacts`` built before the first step hold every contact; no overflow bit, no monitor row, no health
    flag."""
    probe = make_sim(backend, nconmax=None, xml=DROP)
    probe.step_period()
    before = max(d.ncon for d in probe.solver.datas)
    sim = make_sim(backend, nconmax=None, cap=before, xml=DROP)
    contacts = sim.contacts
    assert contacts.rigid_contact_max == sim.solver.get_max_contact_count()   # sized from the mirror at build
    mon = run(sim, observe=True)                                             # 3 periods, monitors per step
    for _ in range(12):
        sim.step_period()
    s = sim.solver
    made = [d.ncon for d in s.datas]
    assert min(made) > before, (made, before)                                # the third box landed
    assert [o.ncon for o in s.outs] == made and min(o.ncon_max for o in s.outs) >= max(made)
    assert sim.contacts is contacts and int(contacts.rigid_contact_count.numpy()[0]) == sum(made)
    assert s.route_report()["counters"]["contact_overflow"] == 0
    assert not s.mjw_data.overflow.numpy().any()
    assert mon.report()["capacity_overflow"].tolist() == [0.0] * WORLDS
    assert not np.any(sim.health.record.numpy()["flags"] & int(HealthFlags.CONTACT_OVERFLOW))


def test_an_entry_without_export_keeps_its_mirror():
    """No export (ncon_max 0): the mirror is Newton's and the route copies nothing."""
    from alloy.route.mjc_core import MjcCoreSolver
    from alloy.route.mjc_core_config import RouteConfig
    from test_mjc_core_capacity import XML

    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(XML, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=WORLDS)
    model = b.finalize(device="cpu")
    plain = int(newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, separate_worlds=True).mjw_data.naconmax)
    s = MjcCoreSolver(model, route=RouteConfig(ncon_max=0))
    assert s.route.ncon_max == 0 and int(s.mjw_data.naconmax) == plain
