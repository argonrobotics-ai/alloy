# SPDX-License-Identifier: Apache-2.0
"""System identification (optional): make a simulated robot's joints move like your real robot's.

Record your robot following joint targets, then::

    python -m alloy.sysid excite --robot panda --out excite.csv        # a safe trajectory to play on the robot
    python -m alloy.sysid fit --robot panda --data take.csv --out my_panda.sysid.json
    python -m alloy.sysid report my_panda.sysid.json

    scene.add_robot("panda", path, sysid="my_panda.sysid.json")        # the simulated robot now uses the fit

``fit`` replays the recorded commands in many parallel worlds, each with candidate joint parameters (servo gains,
joint damping, dry friction, action delay; optionally armature and a torque limit), and keeps the ones whose
simulated joint positions track the measured ones best. ``import alloy`` does not load this package, and a scene
without ``sysid=`` builds and steps exactly as before. Guide: docs/guide/sysid.md.
"""

from alloy.sysid.errors import SysidError
from alloy.sysid.excitation import Excitation, trajectory
from alloy.sysid.fitter import FitConfig, fit
from alloy.sysid.params import DEFAULT, PARAMETERS
from alloy.sysid.readers import load_recording
from alloy.sysid.recording import Episode, Recording, make_episode, save_csv
from alloy.sysid.result import SysidResult
from alloy.sysid.tools import demo, excite, report

__all__ = ["DEFAULT", "PARAMETERS", "Episode", "Excitation", "FitConfig", "Recording", "SysidError", "SysidResult",
           "demo", "excite", "fit", "load_recording", "make_episode", "report", "save_csv", "trajectory"]
