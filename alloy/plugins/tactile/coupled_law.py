# SPDX-License-Identifier: Apache-2.0
"""The coupled-scene law of the contact-force sensors: a pad pressing a static block reads the same normal
force when liquid sits beside it, which makes the scene step on a coupled stepper, as in the rigid-only scene.

On a coupled stepper the plug-in host finds the MuJoCo entry among the coupled solvers and asks its step for the
contact export whatever contacts the coupler passes it (``alloy/plugins/host.py``). Before, such a scene raised a
BuildError, and finding the entry alone gave 0 N with ``sim.check()`` ok. The law builds its two scenes through the
public API (the harness's test scenes step one solver), on the harness's device.

Example::

    Law("coupled_force", coupled_check(lambda: Tactile(**PAD), lambda r: r["tactile"]["pad_force"][0, 0, 0]), ...)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from alloy.plugins.harness.scene import LawResult

PROBE = """<mujoco model="tactile_probe"><worldbody><body name="finger" pos="0 0 0.0275">
  <joint name="press_z" type="slide" axis="0 0 -1" range="0 0.05" damping="10"/>
  <geom name="pad" type="box" size="0.02 0.02 0.006" mass="0.4" solref="0.03 1"/></body></worldbody>
  <actuator><position joint="press_z" kp="500" kv="30"/></actuator></mujoco>"""
PAD = {"bodies": ("tactile_probe/worldbody/finger",), "origins": ((0.0, 0.0, -0.006),)}
PRESS_S, PRESS_M = 0.1, 0.005          # the pad starts 1.5 mm above the block; its target ramps 0 -> 5 mm in 0.1 s
BOUND = 0.02                          # coupled against rigid-only normal force, relative


def press_scene(liquid: bool) -> Any:
    """The probe's pad above a static block; with ``liquid``, water in a cup 0.3 m away (a coupled scene)."""
    import alloy  # noqa: PLC0415 - the harness imports this module without the API

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("probe", mjcf=PROBE)
    s.add_box("block", size=(0.05, 0.05, 0.02), pos=(0.0, 0.0, 0.01), static=True)
    if liquid:
        s.add_open_box("cup", inner=(0.04, 0.04, 0.03), wall=0.01, pos=(0.30, 0.0, 0.0))
        s.add_liquid("water", size=(0.02, 0.02, 0.016), pos=(0.30, 0.0, 0.018))   # on the 10 mm floor
    return s


def press(scene: Any, plugin: Any, normal: Callable[[dict], float], device: str) -> float:
    """The pad's normal force (N) after the press, read by ``normal`` from ``sim.plugins.read()``."""
    import alloy  # noqa: PLC0415

    sim = alloy.build(scene, device=device, plugins=[plugin])
    try:
        sim.run(PRESS_S, record=[], targets={"probe/press_z": (0.0, PRESS_M)})
        return float(normal(sim.plugins.read()))
    finally:
        sim.close()


def coupled_check(make: Callable[[], Any], normal: Callable[[dict], float]) -> Callable[[Any], LawResult]:
    """A law check: ``make()`` is a fresh plug-in on the probe's pad (``PAD``), ``normal`` reads its normal force."""

    def check(run: Any) -> LawResult:
        device = str(run.sim.model.device) if run.sim is not None else "cpu"
        alone = press(press_scene(False), make(), normal, device)
        beside = press(press_scene(True), make(), normal, device)
        rel = abs(beside - alone) / max(abs(alone), 1e-9)
        if abs(alone) < 1.0:                              # the press itself failed: not this law's question
            return LawResult(False, float(alone), 1.0, f"rigid-only pad normal {alone:.4g} N: the pad did not press",
                             "check the probe scene (coupled_law.PROBE)")
        return LawResult.within(rel, BOUND, f"pad normal {beside:.4g} N beside liquid (coupled) vs {alone:.4g} N alone",
                                "the plug-in host must export the MuJoCo entry's contact forces on coupled steppers "
                                "(alloy/plugins/host.py, contact_solver and the step's observables)")

    return check


def normal_pad_force(read: dict) -> float:
    return float(np.asarray(read["tactile"]["pad_force"])[0, 0, 0])


def normal_gel_force(read: dict) -> float:
    return float(np.asarray(read["gel_tactile"]["force"])[0, 0, 0])
