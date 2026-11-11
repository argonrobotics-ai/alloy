# SPDX-License-Identifier: Apache-2.0
"""``task_t12_graph``: T12 on the product loop with CUDA graphs, WP8's "alloy_loop graph" row beside the eager
``task_t12`` (coordinator ask, scr/WP8.md 2026-11-06 09:30Z, item 3).

``task_t12`` (``alloy.bully.tasks``) steps the routed ``SolverMuJoCo`` through ``DirectSim``, eagerly, so the bully's
monitors see every step. ``task_t12_graph`` keeps everything else:

- the ``task_t12`` scene card, assembled at its own 1/1200 s period as ``task_t12`` does (the entry's step), then run
  at the bully's 20 ms period: the clock rule snaps that step to the same 0.625 ms (32 steps per period). The same
  per-world MuJoCo Warp capacities (``TaskTargetSpec``: nconmax 256, njmax 1024);
- the recorded start state and initial command, and the action space (14 position targets on
  ``control.mujoco.ctrl``, the same legal box and rate limit);
- WP5's ordered MuJoCo Warp route (profile ``mjw_route=True``), as ``task_t12``'s ``deterministic`` default.

``make_sim`` returns the ``IntegratedSimulation`` that ``build_simulation`` wires with ``graph="auto"``: a clock key's
first period runs eagerly, the next one captures, and later periods replay graphs. A command is written in place
(``sim.control.mujoco.ctrl.assign``) before ``step_period``, outside the graph, which reads it by pointer.

This target is for timing only. It does not wire the bully's per-step monitors: DirectSim's ``pre_substep`` hooks and
host observers do not run inside a replayed graph. SB-7 verdicts stay on ``task_t12``. Importing this module
registers the target (``alloy.bully.targets`` does not list it as a provider).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import numpy as np
import warp as wp

from alloy.integration.built_copy import replace_built

log = logging.getLogger("alloy.integration")

__all__ = ["NAME", "make_sim", "t12_graph_target"]

NAME = "task_t12_graph"


def _spec() -> Any:
    from alloy.bully.tasks import TaskTargetSpec  # noqa: PLC0415

    return TaskTargetSpec(scene="t12")


def _assemble(worlds: int, device: str, control_dt: float) -> Any:
    """The task_t12 card for ``worlds`` worlds with task_t12's per-world capacities, run at ``control_dt``."""
    from alloy.bully.tasks import SCENES  # noqa: PLC0415
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.parallel.types import PackSpec  # noqa: PLC0415
    from alloy.precision.mjw_ccd import T12_CCD_SLOTS, with_ccd_slots  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    spec = _spec()
    caps = PackSpec(device=device, world_ids=tuple(range(worlds)), n=worlds, stream_priority=0, kind="throughput",
                    rigid_contact_max=spec.nconmax * worlds, njmax=spec.njmax, nconmax=spec.nconmax,
                    shape_pairs_max=0, predicted_bytes=0)
    lib = CardLibrary()
    built = SceneAssembler(lib).assemble(lib.scene(SCENES[spec.scene][2]), device, world_count=worlds, capacities=caps)
    built = with_ccd_slots(built, T12_CCD_SLOTS)            # WP5 SCR-WP8-7: 32 CCD scratch slots per world, not nconmax
    return replace_built(built, control_dt=control_dt)      # the card's 1/1200 s stays the entry's target step


def make_sim(worlds: int, device: str, graph: str = "auto") -> Any:
    """T12 as an ``IntegratedSimulation`` (module docstring); ``graph="off"`` gives the same loop eagerly."""
    from alloy.bully.tasks import _scene_info, _set_initial  # noqa: PLC0415  (task_t12's start state)
    from alloy.integration.wiring import build_simulation  # noqa: PLC0415

    spec = _spec()
    info = _scene_info(spec)
    built = _assemble(worlds, device, spec.control_dt)
    _set_initial(built.model, info, worlds)       # before the solver exists: SolverMuJoCo copies it at construction
    with wp.ScopedDevice(device):
        sim = build_simulation(built, "standard", graph=graph, mjw_route=True)
    sim.control.mujoco.ctrl.assign(np.tile(info.ctrl0, worlds).astype(np.float32))
    log.info("%s: %d worlds on %s, graph=%s, %d steps of %.4g s per period", NAME, worlds, device, graph,
             sim.clock.steps_per_period, sim.clock.dt)
    return sim


def t12_graph_target() -> Any:
    """``task_t12``'s bully target with ``make_sim`` replaced by this module's (same actions and robot bodies)."""
    from alloy.bully.targets import get_target  # noqa: PLC0415

    return dataclasses.replace(get_target("task_t12"), name=NAME, make_sim=make_sim)


def _register() -> None:
    from alloy.bully.targets import register_target  # noqa: PLC0415

    register_target(NAME, t12_graph_target)


_register()
