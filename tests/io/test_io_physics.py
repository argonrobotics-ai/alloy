# SPDX-License-Identifier: Apache-2.0
"""Physics round trip: a block sliding down a 20 degree ramp (mu 0.2) in Alloy, in MuJoCo from the exported MJCF,
and in Alloy again after importing that MJCF back into cards, against the analytic slide s = a t^2 / 2."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards.library import CardLibrary, default_root
from alloy.core.registry import isolated_registry
from alloy.io import export_scene, import_file
from alloy.testing.io import RAMP_ANGLE, RAMP_MU, ramp_library

mujoco = pytest.importorskip("mujoco")
PERIODS = 20                                   # 0.4 s at the scene's 0.02 s control period


def _alloy_path(lib: CardLibrary, scene: str, block: str) -> tuple[np.ndarray, float]:
    from alloy.scene.assembler import SceneAssembler
    from alloy.sim import Simulation

    built = SceneAssembler(lib).assemble(lib.scene(scene), "cpu", world_count=1)
    sim = Simulation(built)
    bi = built.handles[block].template.body[0]
    path = [sim.state.body_q.numpy()[bi, :3].copy()]
    for _ in range(PERIODS):
        sim.step_period()
        path.append(sim.state.body_q.numpy()[bi, :3].copy())
    return np.asarray(path), built.control_dt


def _mujoco_path(xml: str, control_dt: float) -> np.ndarray:
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "block")
    n = round(control_dt / m.opt.timestep)
    path = [d.xpos[b].copy()]
    for _ in range(PERIODS):
        mujoco.mj_step(m, d, nstep=n)
        path.append(d.xpos[b].copy())
    return np.asarray(path)


@pytest.mark.compile_heavy
def test_ramp_slide_alloy_mujoco_and_card_round_trip(tmp_path):
    with isolated_registry(empty=True):
        lib, name = ramp_library(tmp_path / "lib")
        alloy, dt = _alloy_path(lib, name, "block")
        res = export_scene(name, tmp_path / "out", formats=("mjcf",), library=lib)
        imp = import_file(res.files["mjcf"], tmp_path / "imp", name="rt")
        lib2 = CardLibrary([tmp_path / "imp", default_root()])
        again, _ = _alloy_path(lib2, "rt", "rt_block")
    mj = _mujoco_path(res.files["mjcf"], dt)
    t = np.arange(PERIODS + 1) * dt
    analytic = 0.5 * 9.81 * (math.sin(RAMP_ANGLE) - RAMP_MU * math.cos(RAMP_ANGLE)) * t ** 2
    slid = np.linalg.norm(alloy - alloy[0], axis=1)
    assert slid[-1] == pytest.approx(analytic[-1], rel=0.02)                 # 0.12 m in 0.4 s
    assert np.abs(np.linalg.norm(mj - alloy, axis=1)).max() < 1e-3          # MuJoCo on the export: same path
    assert np.abs(np.linalg.norm(again - alloy, axis=1)).max() < 1e-3       # cards -> MJCF -> cards -> Alloy
    assert "rt_block" in imp.objects
