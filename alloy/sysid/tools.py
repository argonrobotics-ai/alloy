# SPDX-License-Identifier: Apache-2.0
"""The helpers behind the commands: ``excite`` (a trajectory for the real robot), ``demo`` (a synthetic recording
to try the fit without a robot) and ``report`` (text, and plots when the fit's curves are at hand)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from alloy.api.errors import FeatureUnavailableError
from alloy.sysid import excitation as E
from alloy.sysid import reporting as R
from alloy.sysid.recording import Recording
from alloy.sysid.replay import Bench
from alloy.sysid.result import SysidResult

DEMO_HIDDEN = {"kp": 0.7, "vel": 1.4, "friction_share": 0.02, "delay": 0.012}


def excite(robot: Any, joints: list[str] | None = None, *, rate: float = 100.0, seconds: float = 8.0,
           amplitude: float = 0.15, mode: str = "sequential", device: str | None = "cpu") -> E.Excitation:
    """A safe excitation trajectory for ``robot`` (a shipped robot's name or a robot file): small chirps and quick
    moves of each servoed joint (or of ``joints``) inside its limits. ``.save("excite.csv")`` writes it.

    Example: ``alloy.sysid.excite("panda", seconds=6).save("excite.csv")``."""
    bench = Bench(robot, joints, worlds=1, dt=1.0 / float(rate) if 50.0 <= rate <= 500.0 else 0.01, device=device)
    return E.trajectory(bench.joint_names, bench.lower, bench.upper, bench.hinge, rate=rate, seconds=seconds,
                        amplitude=amplitude, mode=mode)


def demo(robot: Any, joints: list[str] | None = None, *, seconds: float = 6.0, noise: float = 2e-4, seed: int = 0,
         device: str | None = "cpu") -> tuple[Recording, dict[str, np.ndarray]]:
    """``(recording, hidden)``: a synthetic "real robot" for trying the fit. The robot is simulated with joint
    parameters that differ from its file (servo stiffness x 0.7, kd and damping x 1.4, dry friction of 2 % of the
    joint's torque, a 12 ms action delay) and sensor noise; ``hidden`` holds those values, the answer a fit should
    find. Example: ``rec, hidden = alloy.sysid.demo("panda", ["joint1", "joint2"])``."""
    from alloy.sysid import synth  # noqa: PLC0415

    exc = excite(robot, joints, seconds=seconds, mode="together", device=device)
    bench = Bench(robot, exc.joints, worlds=1, dt=0.01, device=device)
    nom = bench.nominal
    torque = np.where(np.isfinite(nom["torque_limit"]), nom["torque_limit"], 0.2 * nom["kp"])
    hidden = {"kp": DEMO_HIDDEN["kp"] * nom["kp"], "kd": DEMO_HIDDEN["vel"] * nom["kd"],
              "damping": DEMO_HIDDEN["vel"] * nom["damping"],
              "friction": nom["friction"] + DEMO_HIDDEN["friction_share"] * torque,
              "delay": np.full(len(exc.joints), DEMO_HIDDEN["delay"])}
    return synth.record(robot, exc, hidden, noise=noise, seed=seed, device=device), hidden


def report(result: SysidResult | str | Path, folder: str | Path | None = None, robot_name: str = "robot") -> str:
    """The report text of a fit result or a saved file; with ``folder`` and a result of ``fit``, also one overlay
    plot per joint and ``report.txt`` there. Example: ``print(alloy.sysid.report("my_robot.sysid.json"))``."""
    data = result if isinstance(result, SysidResult) else R.from_file(result)
    out = R.text(data, robot_name)
    if folder is not None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.txt").write_text(out + "\n")
        if isinstance(result, SysidResult) and result.curves:
            try:
                R.plots(result, folder)
            except FeatureUnavailableError as exc:          # the text report stands without the plots
                out += f"\n\n(no overlay plots: {exc.message}; {exc.fix})"
    return out


__all__ = ["demo", "excite", "report"]
