# SPDX-License-Identifier: Apache-2.0
"""Jets plunging into pools (supply_jetting.py; WP15's SCR with WP9: soil under a pool never eroded) on the stub film,
CPU: the wall shear on the flooded cells under a faucet's jet through a pool over the slab, by the free jet's decay
past its core and Beltaos & Rajaratnam's peak, in the box written this exchange; none where the jet meets no pool."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.surface_flow.supply import Nozzle, Pool
from alloy.plugins.surface_flow.supply_stub import CHART, FAUCET, RHO, StubFilm, faucet_network, slab_scene

AREA = 0.16                                             # the sink tank's area (faucet_network), the pool's footprint


def _run(depth: float | None, half_x: float = 0.2, second: bool = False, depth_1: float | None = None):
    """The faucet over the slab, under ``depth`` of pool (none: a bare slab); ``second``: a weaker jet 12 mm off;
    ``depth_1``: world 1's pool that deep instead (its tank set before the first exchange)."""
    net = faucet_network(tank_volume=(depth or 0.0) * AREA)
    nozzles = (FAUCET,)
    if second:
        net["edges"].append({"name": "spout2", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 3e-5,
                             "dp_lam": 1.0})
        nozzles = (FAUCET, Nozzle("faucet2", edge="spout2", pos=(0.012, 0.0, 0.3), exit_area=3e-5))
    if depth is not None:                               # over the slab (top at 0.02); the level reads the tank's area
        net["pools"] = [Pool("sink", floor=(0.0, 0.0, 0.02), half=(half_x, 0.2), flood=True)]
    p = StubFilm(charts=(CHART,), nozzles=nozzles, network=net)
    sim = build_simulation(TestScene(build=slab_scene(), ground=True, worlds=2), [p], "cpu")
    if depth_1 is not None:
        f, k = p.supply.fluid, p.supply.net.node_index("sink")
        kappa = p.supply.net.tank_kappa(f.gravity)["sink"]
        pr = f.pressure.numpy()
        pr[1, k] -= (depth - depth_1) * AREA / kappa
        f.pressure.assign(pr)
    for _ in range(3):
        sim.step_period()
    return p, sim


def _expected(p: StubFilm, w: int, r: int) -> tuple[np.ndarray, float]:
    """Ray r's wall shear on every cell (module law) from its own flow, speed, strike and depth, and its b."""
    s = p.supply
    noz = s.nozzle_spec.numpy()[s.ray_noz.numpy()[r]]
    q = -float(s.nozzle_q.numpy()[w, noz["edge"]])
    u0 = float(np.linalg.norm(s.hit["w"].numpy()[w, r]))
    d0, z = math.sqrt(4.0 * q / (math.pi * u0)), float(s.sub_z.numpy()[w, r])
    uc = u0 * min(1.0, 6.2 * d0 / z)
    b = 0.5 * d0 + 0.1 * z
    rr = np.hypot(*(p.tables.cpos.numpy()[:, :2] - s.sub_p.numpy()[w, r, :2]).T)
    return 0.16 / 6.2 ** 2 * RHO * uc * uc * np.exp(-(rr / b) ** 2), b


def _in_box(p: StubFilm, w: int) -> np.ndarray:
    i0, i1, j0, j1 = (int(v) for v in p.supply.written.numpy()[w, 0])
    nx, ny = p.tables.shapes[0]
    rows, cols = np.arange(nx)[:, None], np.arange(ny)[None, :]
    return ((rows >= i0) & (rows <= i1) & (cols >= j0) & (cols <= j1)).ravel()


@pytest.mark.parametrize("depth", [0.03, 0.10])
def test_a_jet_into_a_pool_shears_the_flooded_floor_by_its_decay(depth):
    """A faucet's jet (about 2.4 m/s, 7 mm across at the surface) into 30 mm of water (inside its 6.2·d0 core) and
    100 mm (past it, u_c = u0·6.2·d0/z): every cell's pool_jet_tau is τ0·exp(-(r/b)²) with τ0 = 0.16/6.2²·ρ·u_c² and
    b = d0/2 + 0.1·z from the jet's own flow, speed and depth (1e-5, float32), only on flooded cells inside the box
    written this exchange; a value left in that box is cleared at the next exchange (the clear rule)."""
    p, sim = _run(depth)
    s = p.supply
    for w in range(2):
        c0, z = int(s.sub_cell.numpy()[w, 0]), float(s.sub_z.numpy()[w, 0])
        assert c0 >= 0 and z == pytest.approx(depth, rel=0.05)              # straight down through the water
        q = -float(s.nozzle_q.numpy()[w, s.nozzle_spec.numpy()[0]["edge"]])
        u0 = float(np.linalg.norm(s.hit["w"].numpy()[w, 0]))
        d0 = math.sqrt(4.0 * q / (math.pi * u0))
        uc = u0 * min(1.0, 6.2 * d0 / z)
        assert (depth > 6.2 * d0) == (uc < u0)
        tau0, b = 0.16 / 6.2 ** 2 * RHO * uc * uc, 0.5 * d0 + 0.1 * z
        tau = p.pool_jet_tau.numpy()[w]
        on = tau > 0.0
        r = np.hypot(*(p.tables.cpos.numpy()[:, :2] - s.sub_p.numpy()[w, 0, :2]).T)
        assert on.sum() > 1 and np.allclose(tau[on], tau0 * np.exp(-(r[on] / b) ** 2), rtol=1e-5, atol=0.0)
        assert (p.flooded.numpy()[w, on] >= 0).all() and _in_box(p, w)[on].all()
        assert np.argmax(tau) == c0 or r[np.argmax(tau)] <= r[c0]
    corner = int(np.flatnonzero(_in_box(p, 0) & (p.pool_jet_tau.numpy()[0] > 0.0))[0])   # the box's first cell
    t = p.pool_jet_tau.numpy()
    t[0, corner] = 1.0e3
    p.pool_jet_tau.assign(t)
    sim.step_period()
    assert p.pool_jet_tau.numpy()[0, corner] < 1.0e3


def test_overlapping_jets_leave_the_larger_shear():
    """WP15's review: a weaker jet 12 mm beside the faucet under 100 mm of water, their footprints overlapping: every
    cell near both strikes holds the larger of the two (not the last one's)."""
    p, _ = _run(0.10, second=True)
    tau, cpos = p.pool_jet_tau.numpy(), p.tables.cpos.numpy()[:, :2]
    for w in range(2):
        (t0, b0), (t1, b1) = _expected(p, w, 0), _expected(p, w, 1)
        near = ((np.hypot(*(cpos - p.supply.sub_p.numpy()[w, 0, :2]).T) < 1.5 * b0)
                & (np.hypot(*(cpos - p.supply.sub_p.numpy()[w, 1, :2]).T) < 1.5 * b1))
        assert near.sum() >= 4 and (t1[near] < t0[near]).any()
        assert np.allclose(tau[w, near], np.maximum(t0, t1)[near], rtol=1e-5, atol=0.0)


def test_each_world_shears_by_its_own_depth():
    """WP15's review: world 0's pool 100 mm deep, world 1's 30 mm: each world's strike and every written cell go by
    its own depth."""
    p, _ = _run(0.10, depth_1=0.03)
    tau = p.pool_jet_tau.numpy()
    for w, depth in ((0, 0.10), (1, 0.03)):
        assert p.supply.sub_z.numpy()[w, 0] == pytest.approx(depth, rel=0.05)
        on = tau[w] > 0.0
        assert on.sum() > 1 and np.allclose(tau[w, on], _expected(p, w, 0)[0][on], rtol=1e-5, atol=0.0)


def test_only_flooded_cells_take_the_shear():
    """A pool 24 mm wide across the jet (100 mm deep): the jet's footprint (b = 14 mm) reaches unflooded cells beyond
    the pool's edges, and they take none."""
    p, _ = _run(0.10, half_x=0.012)
    tau, fl = p.pool_jet_tau.numpy(), p.flooded.numpy()
    r = np.hypot(*(p.tables.cpos.numpy()[:, :2] - p.supply.sub_p.numpy()[0, 0, :2]).T)
    assert ((fl[0] < 0) & (r < 0.02)).any() and (tau[fl < 0] == 0.0).all() and (tau[fl >= 0] > 0.0).any()


def test_a_jet_on_a_bare_slab_writes_no_pool_shear():
    """With no pool the jet strikes the slab as a jet (its jump): pool_jet_tau stays 0 everywhere."""
    p, _ = _run(None)
    assert (p.supply.hit["kind"].numpy()[:, 0] == 1).all() and (p.pool_jet_tau.numpy() == 0.0).all()
