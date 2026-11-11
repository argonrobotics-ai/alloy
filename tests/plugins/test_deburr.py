# SPDX-License-Identifier: Apache-2.0
"""Deburring laws R-4 on the removal plug-in's geometry feedback (engine item 10 part 2, alloy-sc9).

A small, stiff abrasive pin (r = 4 mm) rests on a plate under its weight and spins without friction (no torque, so it
stays centred and keeps its spin), with a large wear coefficient so it wears in fast:

- R-4a, the contact follows the removed depth: the pin's resting height drops by the removed depth under it (the
  pad-coverage-weighted mean of the depth grid), within 2 %.
- R-4b, constant-force sinking at the Archard rate: once the pin has worn in, wear is uniform over its face (the
  "uniform wear" state of disc clutches, p·r = const), so it sinks at dh/dt = (K/H)·(F/A)·(ωR/2), within 5 %. Before
  that, at uniform pressure, the mean rate is (K/H)·(F/A)·(2ωR/3): the run starts there and leaves it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene
from alloy.plugins.removal import Removal
from alloy.plugins.removal import laws as rl

pytestmark = pytest.mark.law

SPIN = 20.0                   # rad/s
RATE = 2.0e-6                 # K/H [1/Pa]: a large wear coefficient so the pin wears in within a second
PIN = {"friction": 0.0, "pad_radius": 0.004, "pad_center": (0.0, 0.0, -0.015), "pad_stiffness": 1.0e8,
       "wear_coefficient": RATE * 1.0e9, "hardness": 1.0e9, "size": (0.02, 0.02), "surface_offset": 0.005,
       "cells": (80, 80)}
PERIODS = 75                  # 1.5 s
STEP = 3.125e-4               # s: the stiff pin's tilt spring needs omega dt < 1 (check_dt)


@pytest.fixture(scope="module")
def spun():
    """The pin spinning in place for 1.5 s; returns (plugin read per period, tool z per period, the plug-in)."""
    plugin = Removal(**PIN)
    scene = TestScene(build=rl.plate_and_tool(plugin.config), init=rl.start_velocity(angular=(0.0, 0.0, SPIN)),
                      ground=False, periods=PERIODS, worlds=1, dt=STEP)
    run = run_scene(scene, [plugin], "cpu")
    assert run.error is None, run.error
    b = run.bodies(plugin.config.tool)[0]
    depth = np.asarray(run.trace.plugin(plugin.label, "depth"), np.float64)[:, 0]        # [K, nx, ny]
    z = run.trace.body_q[:, b, 2].astype(np.float64)
    xy = run.trace.body_q[:, b, :2].astype(np.float64)
    return run, plugin, depth, z, xy


def _coverage(p: Removal, centre: np.ndarray) -> np.ndarray:
    """Fraction of each cell under the pad (the kernel's boundary rule), pad centred at ``centre`` (x, y)."""
    nx, ny = p.depth.shape[1:]
    cx, cy = float(p.grid.cell[0]), float(p.grid.cell[1])
    x = (np.arange(nx) + 0.5) * cx - float(p.grid.half[0]) - centre[0]
    y = (np.arange(ny) + 0.5) * cy - float(p.grid.half[1]) - centre[1]
    r = np.hypot(x[:, None], y[None, :])
    return np.clip((p.config.pad_radius - r) / (0.5 * (cx + cy)) + 0.5, 0.0, 1.0)


def test_r4a_resting_height_follows_the_removed_depth(spun):
    run, p, depth, z, xy = spun
    cover = _coverage(p, xy[-1])
    h = float((depth[-1] * cover).sum() / cover.sum())
    drop = float(z[0] - z[-1])
    assert h > 1.0e-4, h                                       # the pin wore at least 0.1 mm
    assert abs(drop / h - 1.0) < 0.02, (drop, h)


def test_r4b_worn_in_pin_sinks_at_the_archard_rate(spun):
    run, p, depth, z, xy = spun
    c = p.config
    weight = float(run.model.body_mass.numpy()[run.bodies(c.tool)[0]]) * rl.G
    area = math.pi * c.pad_radius**2
    t = run.trace.t.astype(np.float64)
    late = slice(2 * PERIODS // 3, None)
    rate = -float(np.polyfit(t[late], z[late], 1)[0])
    uniform_wear = RATE * weight / area * 0.5 * SPIN * c.pad_radius
    assert abs(rate / uniform_wear - 1.0) < 0.05, (rate, uniform_wear)
    first = -float(np.polyfit(t[:4], z[:4], 1)[0])
    uniform_pressure = RATE * weight / area * 2.0 / 3.0 * SPIN * c.pad_radius
    assert first > 1.15 * uniform_wear and first < 1.05 * uniform_pressure, (first, uniform_wear, uniform_pressure)
