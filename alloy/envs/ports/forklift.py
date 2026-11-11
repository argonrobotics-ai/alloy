# SPDX-License-Identifier: Apache-2.0
"""The forklift ports (unofficial ports of Gym-Anything forklift-tasks-v1) as environments: five tasks a scripted
operator passes on the port's own checks, each on its port's own run (``alloy.tasks.forklift_ga_run.simulate``, one
world).

| env | Gym-Anything task |
|---|---|
| ``ForkliftStartAndPark-v0`` | start-and-park |
| ``ForkliftRampDownLoaded-v0`` | ramp-down-loaded |
| ``ForkliftPutAwayRack-v0`` | put-away-rack |
| ``ForkliftLoadTrailer-v0`` | load-trailer |
| ``ForkliftLoadBales-v0`` | load-bales |

The policy is the operator: its action is the truck's controls (``alloy.tasks.forklift_control.Controls``):
throttle, brake, steer [rad], park, direction (-1, 0, 1), lift, tilt, side-shift, key and engine, applied through the
port's own ``TruckIO.apply`` (the drive, the mast's valves and the signals the criteria read). Proprio is the
truck's pose and velocity, the mast's lift and tilt and the fork's side-shift as the port observes them. Success
and the reward are the port's verdict (its re-implementation of the task's criteria; partial score otherwise). The
two criteria the task checks on the operator (seated, head clear of the guard) are not judged: nobody is in the seat.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from alloy.envs.ports.env import PortSpec

FIELDS = ("throttle", "brake", "steer", "park", "direction", "lift", "tilt", "sideshift", "key_on", "engine_on")
LOW = np.array([-1.0, 0.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0], np.float32)
HIGH = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], np.float32)
_HOOK = threading.Lock()
_HOOKED: list[bool] = []


def _hook_truck_io() -> None:
    """Make each ``TruckIO`` remember the controls it last applied, and hang itself on its simulation, so the env can
    read the operator's controls and apply the policy's through the same path."""
    from alloy.tasks import forklift_ga_truck as ft  # noqa: PLC0415

    with _HOOK:
        if _HOOKED:
            return
        original = ft.TruckIO.apply

        def apply(self: Any, c: Any) -> None:
            original(self, c)
            self.last_controls = c
            self.sim._port_truck_io = self

        apply.__doc__, apply.__wrapped__ = original.__doc__, original  # type: ignore[attr-defined]
        ft.TruckIO.apply = apply
        _HOOKED.append(True)


def controls_of(c: Any) -> np.ndarray:
    return np.array([float(getattr(c, f)) for f in FIELDS], np.float32)


class TruckControls:
    """The operator's controls as the action (one world: the forklift ports run one)."""

    size, proprio_size, low, high = len(FIELDS), 0, LOW, HIGH

    def bind(self, sim: Any) -> None:
        self.io = getattr(sim, "_port_truck_io", None)
        if self.io is None:
            from alloy.envs.errors import EnvError  # noqa: PLC0415

            raise EnvError("the forklift port applied no controls before its first period",
                           "run a forklift port (alloy.tasks.forklift_ga_run.simulate)")
        obs = self.io.observe(0.0)
        self.proprio_size = len(_proprio(obs))

    def read(self, sim: Any) -> np.ndarray:  # noqa: ARG002
        return controls_of(self.io.last_controls)[None]

    def write(self, sim: Any, a: np.ndarray) -> None:  # noqa: ARG002
        a = np.asarray(a, np.float32).reshape(-1)
        if np.array_equal(a, controls_of(self.io.last_controls)):
            return
        from alloy.tasks.forklift_control import Controls  # noqa: PLC0415

        self.io.apply(Controls(throttle=float(a[0]), brake=float(a[1]), steer=float(a[2]), park=bool(a[3] > 0.5),
                               direction=int(np.clip(np.rint(a[4]), -1, 1)), lift=float(a[5]), tilt=float(a[6]),
                               sideshift=float(a[7]), key_on=bool(a[8] > 0.5), engine_on=bool(a[9] > 0.5)))

    def proprio(self, sim: Any) -> np.ndarray:  # noqa: ARG002
        return np.asarray(_proprio(self.io.observe(0.0)), np.float32)[None]


def _proprio(obs: Any) -> list[float]:
    """The numbers of the port's ``Observation``: every float field, and the items of vector fields."""
    out: list[float] = []
    for value in vars(obs).values():
        arr = np.asarray(value, dtype=object).reshape(-1)
        if arr.size and all(isinstance(v, (int, float, np.floating, np.integer, bool)) for v in arr):
            out.extend(float(v) for v in arr)
    return out


def _runner(module: str):  # noqa: ANN202
    def run(worlds: int, device: str, seed: int) -> Any:  # noqa: ARG001 - one world, one fixed start draw
        import importlib  # noqa: PLC0415

        from alloy.tasks.forklift_ga_run import simulate  # noqa: PLC0415

        _hook_truck_io()
        return simulate(importlib.import_module(f"alloy.tasks.{module}").task(), device=device)

    return run


def _mover() -> Any:
    from alloy.envs.ports.movers import Mover  # noqa: PLC0415

    return Mover(TruckControls())


SPECS = [
    PortSpec(name, f"forklift/{ga}", _runner(module), _mover, lambda r, w: bool(r[1].success),
             score=lambda r, w: 1.0 if r[1].success else float(r[1].score or 0.0), summary=summary)
    for name, ga, module, summary in (
        ("ForkliftStartAndPark-v0", "start-and-park", "forklift_ga_park", "start the truck and park it in a bay"),
        ("ForkliftRampDownLoaded-v0", "ramp-down-loaded", "forklift_ga_ramp", "drive a load down the yard ramp"),
        ("ForkliftPutAwayRack-v0", "put-away-rack", "forklift_ga_rack", "put a pallet away on a rack beam"),
        ("ForkliftLoadTrailer-v0", "load-trailer", "forklift_ga_dock", "load a pallet into a docked trailer"),
        ("ForkliftLoadBales-v0", "load-bales", "forklift_ga_bale_run", "load three bales and dunnage"))
]

__all__ = ["FIELDS", "SPECS", "TruckControls", "controls_of"]
