# SPDX-License-Identifier: Apache-2.0
"""Unofficial ports of Gym-Anything for Robotics tasks (``alloy.tasks``) as Gymnasium vector environments, with
camera observations: painting (8), surface rinsing (7), concrete finishing (5), forklift (5), gift bags (2) and
reheat-porridge (1). The success checks are Alloy's re-implementation, none was run through Gym-Anything's verifier,
and most ports judge only part of a task's criteria: ``docs/guide/envs.md`` lists what each port judges.

Each environment runs its port's own episode (scene, script, the port's checks) one control period per ``step``:
the policy's action replaces the script's command, ``info["expert_action"]`` is the script's, and the port's verdict
decides success at the end (:mod:`alloy.envs.ports.env`). ``PORTS`` lists them by name; ``alloy.envs.make(name)``
builds one. Example::

    from alloy.envs import make
    env = make("PaintRollStrip-v0", size="small", camera=(64, 64))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(info["expert_action"])
"""

from __future__ import annotations

from typing import Any

from alloy.envs.ports.concrete import SPECS as _CONCRETE
from alloy.envs.ports.env import PortEnv, PortSpec
from alloy.envs.ports.forklift import SPECS as _FORKLIFT
from alloy.envs.ports.gifts_cooking import SPECS as _GIFTS_COOKING
from alloy.envs.ports.painting import SPECS as _PAINTING
from alloy.envs.ports.rinsing import SPECS as _RINSING

PORTS: dict[str, PortSpec] = {s.name: s for s in (*_PAINTING, *_RINSING, *_GIFTS_COOKING, *_FORKLIFT, *_CONCRETE)}


def make_port(name: str, num_envs: int = 1, device: str | None = None, seed: int = 0, **kwargs: Any) -> PortEnv:
    """The port environment ``name`` over ``num_envs`` worlds; keyword options go to the env (``camera=``, ``spp=``,
    ``autoreset=``) or to the port's run (``size="small"``, ``settle=``). Example:
    ``make_port("PaintClearCoat-v0", num_envs=8, size="small")``."""
    from alloy.envs.errors import EnvError  # noqa: PLC0415

    if name not in PORTS:
        raise EnvError(f"no port environment named {name!r}", f"use one of {sorted(PORTS)}", field="name")
    return PortEnv(PORTS[name], num_envs=num_envs, device=device, seed=seed, **kwargs)


__all__ = ["PORTS", "PortEnv", "PortSpec", "make_port"]
