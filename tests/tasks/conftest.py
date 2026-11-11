# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the task-scene tests: a small MuJoCo model that uses every part the converter writes."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from alloy.testing.tasks import WEDGE_LEVEL, synth_xml



@pytest.fixture
def synth_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(synth_xml())


@pytest.fixture
def synth_recording(synth_model: mujoco.MjModel, tmp_path: Path) -> Path:
    return record(synth_model, tmp_path)


@pytest.fixture
def synth_recording_level(tmp_path: Path) -> Path:
    return record(mujoco.MjModel.from_xml_string(synth_xml(WEDGE_LEVEL)), tmp_path)  # noslip 5, replayed as recorded


def record(m: mujoco.MjModel, tmp_path: Path) -> Path:
    """A recorded-trace directory in the [internal reference], made with CPU MuJoCo (offline reference)."""
    d = mujoco.MjData(m)
    steps, every = 300, 50
    rng = np.random.default_rng(7)
    ctrl = np.cumsum(rng.normal(0.0, 0.002, (steps + 1, m.nu)), axis=0)
    state0 = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m, d, state0, mujoco.mjtState.mjSTATE_INTEGRATION)
    ref = []
    for i in range(steps):
        d.ctrl[:] = ctrl[i]
        mujoco.mj_step(m, d)
        if (i + 1) % every == 0:
            ref.append(d.qpos.copy())
    root = tmp_path / "synth"
    root.mkdir()
    mujoco.mj_saveModel(m, str(root / "model.mjb"), None)
    np.save(root / "state0.npy", state0)
    np.save(root / "ref_qpos.npy", np.array(ref))
    empty = {f"{k}_cols": np.zeros(0, np.int64) for k in ("qfrc", "xfrc", "afr", "eq")}
    firsts = {"qfrc_first": np.zeros(m.nv), "xfrc_first": np.zeros(6 * m.nbody),
              "afr_first": m.jnt_actfrcrange.ravel().copy(), "eq_first": np.ones(m.neq)}  # fmt: skip
    vars_ = {f"{k}_var": np.zeros((steps + 1, 0)) for k in ("qfrc", "xfrc", "afr", "eq")}
    np.savez(root / "inputs.npz", ctrl_cols=np.arange(m.nu), ctrl_first=ctrl[0], ctrl_var=ctrl, **empty, **firsts,
             **vars_)  # fmt: skip
    return root
