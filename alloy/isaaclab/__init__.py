# SPDX-License-Identifier: Apache-2.0
"""Alloy as a physics solver for Isaac Lab 3 (its Newton backend): switch with one setting, no edit to the task.

Command line (Isaac Lab's ``isaaclab`` command; works on tasks that offer ``physics=newton_mjwarp``)::

    isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=alloy

In Python::

    import alloy.isaaclab                                   # makes physics=alloy resolve, like the command line
    from alloy.isaaclab import AlloySolverCfg, alloy_physics
    env_cfg.sim.physics = alloy_physics(env_cfg.sim.physics)  # a NewtonCfg with MuJoCo Warp -> the same on Alloy
    NewtonCfg(solver_cfg=AlloySolverCfg(njmax=80, nconmax=25))  # or name it directly, like any Newton solver cfg

Needs Isaac Lab 3.0 installed in the same environment, with MuJoCo 3.17 (Alloy's): ``docs/guide/isaaclab.md``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AlloySolverCfg": "cfg",
    "NewtonAlloyManager": "manager",
    "AlloyNewtonSolver": "solver_api",
    "AlloyRigidSolver": "solver",
    "active_solver": "solver",
    "alloy_physics": "presets",
    "install": "presets",
    "bridge_model": "bridge",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'alloy.isaaclab' has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)  # raises the install line if absent


def _install() -> None:
    from .compat import install as keep_isaaclab_mujoco_warp_working

    keep_isaaclab_mujoco_warp_working()  # acts only on MuJoCo Warp 3.12 under MuJoCo 3.13+ (alloy.isaaclab.compat)
    from ._require import ISAACLAB_ERROR

    if ISAACLAB_ERROR is not None:
        return  # no Isaac Lab here: nothing to teach; the names raise the install line when used
    try:
        from .presets import install
        install()
    except ImportError:
        pass  # Isaac Lab without its task package: there is no preset resolver to teach


_install()
