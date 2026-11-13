# SPDX-License-Identifier: Apache-2.0
"""physics=alloy on unmodified Isaac Lab tasks: the config it resolves to. Needs Isaac Lab 3 (skipped without it)."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab_tasks", reason="Isaac Lab is not installed")
pytest.importorskip("isaaclab_newton", reason="Isaac Lab's Newton backend is not installed")

import isaaclab_tasks  # noqa: E402,F401  (registers the tasks)
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg  # noqa: E402
from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import alloy.isaaclab  # noqa: E402,F401  (installs physics=alloy, as Isaac Lab's entry point does)
from alloy.isaaclab import AlloySolverCfg, alloy_physics  # noqa: E402
from alloy.isaaclab._require import NEWTON_SOLVER_API  # noqa: E402
from alloy.isaaclab.presets import stock_mujoco_warp_left  # noqa: E402


def _physics(task: str, choice: str):
    env_cfg, _ = resolve_task_config(task, "", overrides=[f"physics={choice}"])
    return env_cfg, env_cfg.sim.physics


def test_alloy_takes_the_tasks_mujoco_warp_settings():
    _, stock = _physics("Isaac-Cartpole-Direct", "newton_mjwarp")
    env_cfg, phys = _physics("Isaac-Cartpole-Direct", "alloy")
    assert isinstance(phys, NewtonCfg) and isinstance(phys.solver_cfg, AlloySolverCfg)
    if NEWTON_SOLVER_API:   # Isaac Lab main: Isaac Lab's own manager, Alloy's NewtonSolver class
        assert phys.class_type == stock.class_type
        assert str(phys.solver_cfg.class_type).endswith("alloy.isaaclab.solver_api:AlloyNewtonSolver")
    else:                   # Isaac Lab 3.0: Alloy's NewtonManager subclass
        assert phys.class_type == "alloy.isaaclab.manager:NewtonAlloyManager"
    for field in ("njmax", "nconmax", "cone", "impratio", "integrator", "iterations", "use_mujoco_contacts"):
        assert getattr(phys.solver_cfg, field) == getattr(stock.solver_cfg, field), field
    assert phys.num_substeps == stock.num_substeps and phys.use_cuda_graph == stock.use_cuda_graph
    assert stock_mujoco_warp_left(env_cfg) == []


@pytest.mark.parametrize("task", ["Isaac-Velocity-Flat-AnymalD", "Isaac-Reach-Franka", "Isaac-Velocity-Flat-H1"])
def test_alloy_leaves_no_stock_mujoco_warp(task):
    env_cfg, phys = _physics(task, "alloy")
    assert isinstance(phys.solver_cfg, AlloySolverCfg)
    assert stock_mujoco_warp_left(env_cfg) == []


def test_default_stays_the_tasks_own():
    _, phys = _physics("Isaac-Cartpole-Direct", "newton_mjwarp")
    assert type(phys.solver_cfg) is MJWarpSolverCfg


def test_alloy_physics_in_python():
    cfg = alloy_physics(NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=40, cone="elliptic")))
    assert isinstance(cfg.solver_cfg, AlloySolverCfg) and cfg.solver_cfg.njmax == 40
    assert cfg.solver_cfg.cone == "elliptic" and cfg.solver_cfg.rigid_step == "task"   # the task's own step by default


def test_unknown_choice_still_errors():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_task_config("Isaac-Cartpole-Direct", "", overrides=["physics=no_such_engine"])
