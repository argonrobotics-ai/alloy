# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P2 S2a, writes that do not go through ``alloy.Sim`` (WP4's review of S2a item 4, WP12's W1;
alloy/coupling/world_clock_writes.py): with ``clock_worlds`` a write straight to a resident's coarse rows would be
overwritten by the end-of-period mirror. One law per covered path:
- ``alloy.envs.placement.rest_on_floor`` evicts first: its write to a resident world is read back after the next
  period (and, planted: with the evict switched off the rig sees the write lost);
- ``alloy.envs.VectorEnv`` refuses a simulation built with ``clock_worlds`` before anything writes (its own builds
  never pass the option);
- ``alloy.live.worldops.WorldOps`` refuses one too.

Rig: test_world_clock_run.py's (4 worlds, a 3 cm box on a sheet on a table, the box's density drawn per world, so one
world resides from the first period). CPU, MuJoCo Warp + VBD."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import alloy
from alloy.envs import EnvError, Task, VectorEnv
from alloy.envs.placement import lowest_z, rest_on_floor
from alloy.live.worldops import WorldOps

W, SLOTS, SEED = 4, 1, 0
LIGHT, HEAVY = 500.0, 8000.0
LIFT = 0.006                       # [m] how far above where it rests rest_on_floor puts world 0's box (all worlds move)
DT = 0.01                          # [s] one period
FALL = 0.5 * 9.81 * DT ** 2        # [m] a free fall's drop over one period, beyond its start velocity's


def _scene() -> alloy.Scene:
    side = 0.03
    scene = alloy.Scene(worlds=W, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=(0, 0, 0.104 + 0.001 + side / 2), density=HEAVY)
    scene.randomize("box", "density", LIGHT, HEAVY)
    return scene


@pytest.fixture(scope="module")
def sim() -> Any:
    """The option's simulation after three periods (the boxes have landed): (sim, the resident world)."""
    s = alloy.build(_scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    s.step(3)
    k = np.flatnonzero(s.engine.world_clock.resident)
    assert k.size == 1, s.engine.world_clock.resident
    yield s, int(k[0])
    s.close()


def _z(sim: Any) -> tuple[np.ndarray, np.ndarray]:
    """Each world's box height and vertical velocity."""
    st = sim.state(["box"])
    return np.array(st["box.pos"][:, 2], copy=True), np.array(st["box.vel"][:, 2], copy=True)


def _lift_and_step(sim: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """rest_on_floor puts each world's box's lowest point at world 0's plus LIFT (each world moves by its own dz),
    then one period. Per world: the box's height change over the write and the period, what a write that lands gives
    (dz, then a free fall from the box's velocity: every box is mm clear of the sheet), and dz (W,)."""
    z0, vz = _z(sim)
    dz = np.asarray(rest_on_floor(sim, ["box"], floor=lowest_z(sim, "box") + LIFT)["box"])
    sim.step(1)
    return _z(sim)[0] - z0, dz + vz * DT - FALL, dz


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_rest_on_floor_reaches_a_resident_world(sim: Any) -> None:
    """Planted first (the driver's evict switched off): the resident's write is lost, the others' land. Then as
    shipped: rest_on_floor evicts the resident, and every world, the resident's included, reads its box lifted by dz
    and then in free fall for the next period."""
    s, k = sim
    clock = s.engine.world_clock
    rest = [w for w in range(W) if w != k]
    clock.evict = lambda *a, **kw: None                       # plant: the write lands on the parked slot
    try:
        moved, landed, dz = _lift_and_step(s)
    finally:
        del clock.evict
    assert dz[k] > 0.5 * LIFT and clock.resident[k], (dz, clock.resident)
    assert abs(moved[k]) < dz[k] / 4, (k, moved, dz)          # lost: the mirror wrote the fine slot over it
    np.testing.assert_allclose(moved[rest], landed[rest], atol=3e-4)
    evictions = s.report()["world_clock"]["evictions"]
    moved, landed, dz = _lift_and_step(s)
    assert s.report()["world_clock"]["evictions"] == evictions + 1
    np.testing.assert_allclose(moved, landed, atol=3e-4)      # every world, the resident's included
    assert s.check()["ok"]


@pytest.mark.law
@pytest.mark.timeout(900)
def test_vector_env_refuses_a_clock_worlds_simulation_before_any_write(sim: Any) -> None:
    """VectorEnv(sim) with clock_worlds on: EnvError naming clock_worlds, raised before the task's on_build runs; a
    VectorEnv that builds its own simulation has no way to pass the option."""
    s, _ = sim
    calls: list[Any] = []
    with pytest.raises(EnvError) as err:
        VectorEnv(s, Task(on_build=calls.append))
    assert err.value.field == "clock_worlds" and "clock_worlds" in err.value.message and not calls
    assert "clock_worlds" not in inspect.signature(VectorEnv.__init__).parameters


@pytest.mark.law
@pytest.mark.timeout(900)
def test_world_ops_refuse_a_clock_worlds_simulation(sim: Any) -> None:
    """WorldOps over an env whose simulation has clock_worlds on: EnvError naming clock_worlds."""
    s, _ = sim
    with pytest.raises(EnvError) as err:
        WorldOps(SimpleNamespace(sim=s, num_envs=W, device="cpu"))
    assert err.value.field == "clock_worlds"
