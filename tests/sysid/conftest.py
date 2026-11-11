# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the sysid simulation tests: a two-joint arm with one servo of each kind."""

from __future__ import annotations

import numpy as np
import pytest

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.6">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2" frictionloss="0.1" armature="0.01"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  <body name="link2" pos="0.3 0 0"><joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.02" mass="0.5"/></body>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/>
  <general joint="elbow" gainprm="80" biastype="affine" biasprm="0 -80 -4" forcerange="-20 20"/></actuator></mujoco>"""
JOINTS = ["shoulder", "elbow"]
#: a "real robot" that differs from the file: softer shoulder, stiffer elbow, more drag, dry friction, delays
HIDDEN = {"kp": [120.0, 130.0], "kd": [6.0, 3.0], "damping": [1.2, 0.75], "friction": [0.5, 0.2],
          "delay": [0.012, 0.02]}


@pytest.fixture(scope="session")
def arm_file(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The arm as a robot file (``--robot`` takes a file or a shipped robot's name)."""
    path = tmp_path_factory.mktemp("sysid_arm") / "arm.xml"
    path.write_text(ARM)
    return str(path)


def hidden_file(joints: list[str] = JOINTS, hidden: dict[str, list[float]] = HIDDEN) -> dict:
    """``hidden`` as a hand-written sysid dict."""
    return {"schema": "alloy.sysid/1",
            "joints": {j: {f: float(v[i]) for f, v in hidden.items()} for i, j in enumerate(joints)}}


def replay(sim, name: str, commands: np.ndarray, joints: list[str] = JOINTS) -> np.ndarray:
    """Send ``commands`` (T, J) one control period each; the joint positions after every period, world 0, (T, J)."""
    keys = [f"{name}/{j}" for j in joints]
    out = []
    for row in commands:
        sim.set_targets(dict(zip(keys, row)))
        sim.step(1)
        state = sim.state(keys, world=0)
        out.append([float(state[f"{k}.q"]) for k in keys])
    return np.asarray(out)


@pytest.fixture(scope="session")
def arm_kit():
    """``(JOINTS, HIDDEN, hidden_file, replay)`` for tests (conftest names are not importable under importlib)."""
    return JOINTS, HIDDEN, hidden_file, replay
