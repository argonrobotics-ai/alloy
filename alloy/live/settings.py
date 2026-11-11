# SPDX-License-Identifier: Apache-2.0
"""Whitelisted run settings that steering may change: each has a getter, a setter and a range.

The trainer registers what it allows (a reward weight, the curriculum level, action noise, entropy, the learning
rate); nothing else can be set from the hub. Values are numbers; a value outside ``[lo, hi]`` is refused.

Example::

    settings = Settings()
    settings.add("lr", lambda: opt.param_groups[0]["lr"], set_lr, lo=1e-6, hi=1e-2)
    settings.set("lr", 1e-4)          # -> {"name": "lr", "old": 3e-4, "new": 1e-4}
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class SettingError(ValueError):
    """An unknown setting or a value out of its range."""


@dataclass
class Setting:
    name: str
    get: Callable[[], float]
    put: Callable[[float], None]
    lo: float
    hi: float
    label: str = ""


class Settings:
    """The whitelist."""

    def __init__(self) -> None:
        self._items: dict[str, Setting] = {}

    def add(self, name: str, get: Callable[[], float], put: Callable[[float], None], *, lo: float, hi: float,
            label: str = "") -> None:
        self._items[name] = Setting(name, get, put, float(lo), float(hi), label or name)

    def names(self) -> list[str]:
        return list(self._items)

    def describe(self) -> list[dict[str, Any]]:
        """Name, label, range and current value of every setting (sent to the hub in HELLO and on change)."""
        return [{"name": s.name, "label": s.label, "lo": s.lo, "hi": s.hi, "value": float(s.get())}
                for s in self._items.values()]

    def set(self, name: str, value: Any) -> dict[str, Any]:
        s = self._items.get(name)
        if s is None:
            raise SettingError(f"setting {name!r} is not whitelisted; settings are {self.names()}")
        try:
            v = float(value)
        except (TypeError, ValueError) as e:
            raise SettingError(f"{name}: value {value!r} is not a number") from e
        if not s.lo <= v <= s.hi:
            raise SettingError(f"{name}: {v} is outside [{s.lo}, {s.hi}]")
        old = float(s.get())
        s.put(v)
        return {"name": name, "old": old, "new": float(s.get())}
