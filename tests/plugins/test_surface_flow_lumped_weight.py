# SPDX-License-Identifier: Apache-2.0
"""Surface-flow lumped bodies carry their water's weight (WP15's ruling on JOB3's sponge: a force at the centre of
mass, graph-safe, no live mass change; the water's inertia is the built one). CPU, harness."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow
from alloy.plugins.surface_flow.lumped_specs import Lumped

DT, N, R, W = 2.0e-3, 20, 0.01, 2.0e-6                    # W m³ of water in each body (world 0; twice in world 1)
BUILT = {"dry": 0.0, "wet": W, "wetter": 2.0 * W, "offset": 0.0}


def _four(b) -> None:
    b.add_shape_box(-1, xform=body_xform(0.0, 0.0, -1.0), hx=0.05, hy=0.05, hz=0.01, label="far")
    for name, x in (("dry", -0.1), ("wet", 0.0), ("wetter", 0.1)):
        b.add_shape_sphere(b.add_body(xform=body_xform(x, 0.0, 0.5), label=name), radius=R)
    k = b.add_body(xform=body_xform(0.2, 0.0, 0.5), label="offset")               # its COM 50 mm off its origin
    b.add_shape_sphere(k, xform=body_xform(0.05, 0.0, 0.0), radius=R)


def test_a_body_carries_its_waters_weight_beyond_its_built_mass(caplog):
    """Free 1 cm spheres hold 2 mL of water in world 0 and 4 mL in world 1 (WP15's S3: per world): one built dry
    (built_water 0) falls at g·(1 + ρ·water/m), one built with 2 mL at g·(1 + ρ·(water - 2 mL)/m), one with 4 mL at
    g·(1 + ρ·(water - 4 mL)/m) (held up by the water it lacks), each to 1e-5 of its drop (float32 positions near
    0.5 m); a dry-built one whose sphere sits 50 mm off its origin falls alike and does not spin (the force acts at its
    centre of mass). Their full water outweighs them, so the build warns once (WP15's S4); a negative built_water is
    refused."""
    lumps = tuple(Lumped(n, hold=1.0, area=0.01, built_water=bw) for n, bw in BUILT.items())
    flow = SurfaceFlow(charts=(PlaneChart("far", cells=(2, 2)),), material="water", lumped=lumps)
    water = np.array([W, 2.0 * W])
    with caplog.at_level(logging.WARNING, logger="alloy.plugins.surface_flow"):
        sim = build_simulation(TestScene(build=_four, init=lambda sim: flow.lumped.set_water(water), ground=False,
                                         gravity=(0.0, 0.0, -9.81), worlds=2, dt=DT, control_dt=DT), [flow], "cpu")
    assert sum("outweighs their built mass" in m for m in caplog.messages) == 1
    ids = flow.lumped.bodies                                                       # [W, B]
    q0 = sim.state.body_q.numpy()[ids, 2].astype(np.float64)
    for _ in range(N):
        sim.step_period()
    fall = q0 - sim.state.body_q.numpy()[ids, 2].astype(np.float64)
    m, rho = sim.model.body_mass.numpy()[ids].astype(np.float64), float(flow.mat.density)
    free = 9.81 * DT * DT * N * (N + 1) / 2.0
    built = np.array(list(BUILT.values()))
    np.testing.assert_allclose(flow.lumped.read()["lump_water"], np.broadcast_to(water[:, None], ids.shape),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(fall, free * (1.0 + rho * (water[:, None] - built[None, :]) / m), rtol=1e-5, atol=0.0)
    assert np.abs(sim.state.body_qd.numpy()[ids[:, 3], 3:]).max() < 1.0e-6         # no spin off the COM
    with pytest.raises(PluginError):
        Lumped("dry", built_water=-1.0e-6)


def test_lump_species_with_sph_drops_warn_that_they_are_lost(caplog):
    """WP15's round-5 S4: SPH particles carry no species, so with sph=True a lump's species that leave with its drops
    are lost (booked dropped): the build says so, once (no SPH liquid is needed for that; it binds at first use)."""
    flow = SurfaceFlow(charts=(PlaneChart("far", cells=(2, 2)),), material="water", sph=True, species=("soap",),
                       lumped=(Lumped("dry", hold=1.0, area=0.01, built_water=1.0e-3),))
    with caplog.at_level(logging.WARNING, logger="alloy.plugins.surface_flow"):
        build_simulation(TestScene(build=_four, ground=False, gravity=(0.0, 0.0, -9.81), worlds=1, dt=DT,
                                   control_dt=DT), [flow], "cpu")
    assert sum("leave with SPH drops are lost" in m for m in caplog.messages) == 1
