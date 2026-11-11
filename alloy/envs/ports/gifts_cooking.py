# SPDX-License-Identifier: Apache-2.0
"""The gift-bag ports (unofficial ports of Gym-Anything gift-bags-tasks-v1: open-flaps, bag-candle) and the
reheat-porridge port (ingredient-mixing-tasks-v1) as environments, each on its port's own run and judged by the
port's own checks (``docs/guide/envs.md`` lists which of the task's criteria each judges).

| env | Gym-Anything task | what the policy moves |
|---|---|---|
| ``GiftOpenFlaps-v0`` | open-flaps | the gantry's x, y, z and wrist yaw servos (a hook tool); batched worlds |
| ``GiftBagCandle-v0`` | bag-candle | the gantry's x, y, z, wrist yaw and two finger servos; batched worlds |
| ``CookReheatPorridge-v0`` | reheat-porridge | the spoon (kinematic): pose and velocity; the hob's power |

The gift tasks are ``alloy.build`` scenes (cardboard and kraft paper on creased hinges); their verdicts need the
port's hands-off window at the end, so the episode includes it. The porridge port scores worlds together, so the
environment runs one simulation per world. The camera draws the paper (its triangles) and the porridge (one smooth
surface over its particles) through ``alloy.envs.ports.display``.
"""

from __future__ import annotations

from typing import Any

from alloy.envs.ports.display import PORRIDGE, particle_liquid, sim_deformables
from alloy.envs.ports.env import PortSpec
from alloy.envs.ports.movers import KinematicBody, Mover, PluginInput, ServoJoints

GANTRY = ("gantry_x", "gantry_y", "gantry_z", "wrist_yaw")
HOB_W = 3000.0               # W: the action box of the hob's power


def run_open_flaps(worlds: int, device: str, seed: int) -> dict:
    from alloy.tasks import gift_carton_flaps  # noqa: PLC0415

    return gift_carton_flaps.run(device=device, worlds=worlds, seed=seed)


def run_bag_candle(worlds: int, device: str, seed: int) -> dict:
    from alloy.tasks import gift_bag_candle  # noqa: PLC0415

    return gift_bag_candle.run(device=device, worlds=worlds, seed=seed)


def run_porridge(worlds: int, device: str, seed: int) -> dict:
    from alloy.tasks import cook_porridge  # noqa: PLC0415

    return cook_porridge.run(worlds=worlds, device=device, seed=seed)


def _world_success(r: Any, w: int) -> bool:
    return bool(r["worlds"][w]["success"])


SPECS = [
    PortSpec("GiftOpenFlaps-v0", "gift_bags/open-flaps", run_open_flaps, lambda: Mover(ServoJoints(GANTRY)),
             _world_success, batched=True, display=sim_deformables,
             summary="open the four flaps of a taped-shut carton with a hook on a gantry"),
    PortSpec("GiftBagCandle-v0", "gift_bags/bag-candle", run_bag_candle,
             lambda: Mover(ServoJoints((*GANTRY, "close_l", "close_r"))), _world_success, batched=True,
             display=sim_deformables, summary="put a candle into a standing kraft gift bag"),
    PortSpec("CookReheatPorridge-v0", "ingredient_mixing/reheat-porridge", run_porridge,
             lambda: Mover(KinematicBody("spoon"), PluginInput("heater_values", 0.0, HOB_W)),
             lambda r, w: int(r["success"]) >= 1, display=particle_liquid("porridge", PORRIDGE),
             summary="reheat four portions of porridge to 74 C, evenly, without burning it, stirring with a spoon"),
]

__all__ = ["SPECS", "run_bag_candle", "run_open_flaps", "run_porridge"]
