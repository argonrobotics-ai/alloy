# SPDX-License-Identifier: Apache-2.0
"""What a fit may change, the bounds of each parameter, and the unit box the search works in.

A user names parameters (``PARAMETERS``); the search works on variables, one column each, per joint:

- ``kp``, ``friction``, ``armature``, ``torque_limit``, ``delay`` are themselves;
- ``kd`` and ``damping`` act the same way on a position servo (both are a torque against joint velocity), so a
  recording determines only their sum. Asked for together (the default) they are one variable, ``vel``; the sum is
  then split in the robot file's ratio (all to ``damping`` when the file has neither).

Bounds are set from the robot file's values (``Box.around``): a factor each way for gains, zero up to a share of
the joint's torque for dry friction, zero to ``max_delay`` for the delay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.sysid.errors import SysidError

PARAMETERS = ("kp", "kd", "damping", "friction", "delay", "armature", "torque_limit")
DEFAULT = ("kp", "kd", "damping", "friction", "delay")
GROUPS = {"gains": ("kp", "kd"), "default": DEFAULT, "all": PARAMETERS}
UNITS = {"kp": "N·m/rad", "kd": "N·m·s/rad", "damping": "N·m·s/rad", "vel": "N·m·s/rad", "friction": "N·m",
         "delay": "s", "armature": "kg·m²", "torque_limit": "N·m"}
GAIN_FACTOR = 8.0           # kp, kd + damping: the file's value / 8 ... x 8
ARMATURE_FACTOR = 10.0
FRICTION_SHARE = 0.1        # dry friction: 0 ... this share of the joint's torque limit (or 5x the file's value)
MAX_DELAY = 0.05            # s


def variables(parameters: tuple[str, ...] | list[str] | None) -> list[str]:
    """Search variables for the parameter names asked for. Example: ``variables(["gains", "delay"])`` ->
    ``["kp", "kd", "delay"]``; ``variables(None)`` -> ``["kp", "vel", "friction", "delay"]``."""
    asked: list[str] = []
    for p in (parameters or DEFAULT):
        for q in GROUPS.get(p, (p,)):
            if q not in PARAMETERS:
                raise SysidError(f"{p!r} is not a parameter the fit knows",
                                 f"use any of {PARAMETERS} (or 'gains', 'default', 'all')", field="fit")
            if q not in asked:
                asked.append(q)
    if "kd" in asked and "damping" in asked:
        asked = ["vel" if q == "kd" else q for q in asked if q != "damping"]
    return [q for q in ("kp", "kd", "damping", "vel", "friction", "armature", "torque_limit", "delay") if q in asked]


@dataclass
class Box:
    """Bounds per variable and joint: ``lo``, ``hi`` (V, J) and ``log`` (V, J) bool (searched on a log scale)."""

    names: list[str]
    lo: np.ndarray
    hi: np.ndarray
    log: np.ndarray

    @classmethod
    def around(cls, names: list[str], nominal: dict[str, np.ndarray], max_delay: float = MAX_DELAY) -> Box:
        """Bounds around the robot file's values ``nominal`` (field -> (J,)). Example: ``Box.around(["kp"], nom)``."""
        count = len(nominal["kp"])
        lo, hi, log = (np.zeros((len(names), count)) for _ in range(3))
        torque = np.where(np.isfinite(nominal["torque_limit"]), nominal["torque_limit"], 0.2 * nominal["kp"])
        for v, name in enumerate(names):
            if name in ("kp", "kd", "damping", "vel", "armature"):
                base = nominal["kd"] + nominal["damping"] if name == "vel" else nominal[name]
                factor = ARMATURE_FACTOR if name == "armature" else GAIN_FACTOR
                roof = {"armature": 1e-3 * nominal["kp"]}.get(name, 0.2 * nominal["kp"])
                some = base > 0.0
                lo[v], hi[v], log[v] = np.where(some, base / factor, 0.0), np.where(some, base * factor, roof), some
            elif name == "friction":
                hi[v] = np.maximum(5.0 * nominal["friction"], FRICTION_SHARE * torque)
            elif name == "torque_limit":
                lo[v], hi[v], log[v] = 0.2 * torque, torque, True
            else:
                hi[v] = float(max_delay)
        return cls(list(names), lo, hi, log.astype(bool))

    def decode(self, u: np.ndarray) -> np.ndarray:
        """Unit-box points ``u`` (..., V, J) as physical values."""
        safe_lo = np.where(self.log, self.lo, 1.0)
        geo = safe_lo * (np.where(self.log, self.hi, 1.0) / safe_lo) ** u
        return np.where(self.log, geo, self.lo + u * (self.hi - self.lo))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Physical values ``x`` (..., V, J) as unit-box points, clipped into the box."""
        safe_lo = np.where(self.log, self.lo, 1.0)
        ratio = np.where(self.log, self.hi, 2.0) / safe_lo
        geo = np.log(np.maximum(np.where(self.log, x, 1.0), 1e-300) / safe_lo) / np.log(ratio)
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return np.clip(np.where(self.log, geo, (x - self.lo) / span), 0.0, 1.0)

    def start(self, nominal: dict[str, np.ndarray]) -> np.ndarray:
        """(V, J) physical start point: the robot file's values, and no delay."""
        rows = [nominal["kd"] + nominal["damping"] if n == "vel" else
                np.zeros_like(nominal["kp"]) if n == "delay" else
                np.where(np.isfinite(nominal[n]), nominal[n], self.hi[v]) for v, n in enumerate(self.names)]
        return np.stack(rows)


def to_fields(names: list[str], x: np.ndarray, nominal: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Variables ``x`` (..., V, J) as model fields (``alloy.sysid.model.FIELDS`` and ``"delay"``), each (..., J).
    ``vel`` is split into ``kd`` and ``damping`` in the file's ratio. Example: ``to_fields(["kp"], x, nom)["kp"]``."""
    out = {n: x[..., v, :] for v, n in enumerate(names) if n != "vel"}
    if "vel" in names:
        total = nominal["kd"] + nominal["damping"]
        share = np.where(total > 0.0, nominal["kd"] / np.where(total > 0.0, total, 1.0), 0.0)
        vel = x[..., names.index("vel"), :]
        out["kd"], out["damping"] = vel * share, vel * (1.0 - share)
    return out


__all__ = ["Box", "DEFAULT", "MAX_DELAY", "PARAMETERS", "UNITS", "to_fields", "variables"]
