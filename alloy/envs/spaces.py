# SPDX-License-Identifier: Apache-2.0
"""Observation and action spaces: Gymnasium's ``Box`` / ``Dict`` when Gymnasium is installed, else small stand-ins
with the same fields (``shape``, ``dtype``, ``low``, ``high``, ``sample()``, ``contains()``, ``seed()``).

Example::

    space = box(-1.0, 1.0, (3,))            # gymnasium.spaces.Box when available
    batched = batch(space, 64)              # shape (64, 3)
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from typing import Any

import numpy as np


def has_gymnasium() -> bool:
    """True when Gymnasium can be imported. Example: ``has_gymnasium()``."""
    return importlib.util.find_spec("gymnasium") is not None


class Box:
    """Stand-in for ``gymnasium.spaces.Box``: values in [low, high] of ``shape`` and ``dtype``."""

    def __init__(self, low: Any, high: Any, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
        self.shape, self.dtype = tuple(int(n) for n in shape), np.dtype(dtype)
        self.low = np.broadcast_to(np.asarray(low, self.dtype), self.shape).copy()
        self.high = np.broadcast_to(np.asarray(high, self.dtype), self.shape).copy()
        self._rng = np.random.default_rng()

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed ``sample()``. Example: ``space.seed(0)``."""
        self._rng = np.random.default_rng(seed)
        return [seed]

    def sample(self) -> np.ndarray:
        """A random value (uniform where bounded, normal where not). Example: ``space.sample()``."""
        lo, hi = self.low.astype(np.float64), self.high.astype(np.float64)
        bounded = np.isfinite(lo) & np.isfinite(hi)
        out = np.where(bounded, self._rng.uniform(np.where(bounded, lo, 0), np.where(bounded, hi, 1)),
                       self._rng.normal(size=self.shape))
        return out.astype(self.dtype)

    def contains(self, x: Any) -> bool:
        """True if ``x`` has the shape and lies within the bounds. Example: ``space.contains(a)``."""
        x = np.asarray(x)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self) -> str:
        return f"Box({self.shape}, {self.dtype})"


class Dict:
    """Stand-in for ``gymnasium.spaces.Dict``: a mapping of named spaces."""

    def __init__(self, spaces: Mapping[str, Any]) -> None:
        self.spaces = dict(spaces)

    def __getitem__(self, key: str) -> Any:
        return self.spaces[key]

    def seed(self, seed: int | None = None) -> list[int | None]:
        """Seed every sub-space. Example: ``space.seed(0)``."""
        for k, sp in enumerate(self.spaces.values()):
            sp.seed(None if seed is None else seed + k)
        return [seed]

    def sample(self) -> dict[str, Any]:
        """A random value of each sub-space. Example: ``space.sample()["state"]``."""
        return {k: sp.sample() for k, sp in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        """True if every key's value is in its space. Example: ``space.contains(obs)``."""
        return isinstance(x, Mapping) and all(k in x and sp.contains(x[k]) for k, sp in self.spaces.items())

    def __repr__(self) -> str:
        return f"Dict({self.spaces})"


def box(low: Any, high: Any, shape: tuple[int, ...], dtype: Any = np.float32) -> Any:
    """A Box space (Gymnasium's if installed). Example: ``box(-1, 1, (3,))``."""
    if has_gymnasium():
        from gymnasium import spaces  # noqa: PLC0415

        return spaces.Box(low=np.broadcast_to(np.asarray(low, dtype), shape).copy(),
                          high=np.broadcast_to(np.asarray(high, dtype), shape).copy(), shape=shape, dtype=dtype)
    return Box(low, high, shape, dtype)


def dict_space(spaces: Mapping[str, Any]) -> Any:
    """A Dict space (Gymnasium's if installed). Example: ``dict_space({"state": box(...)})``."""
    if has_gymnasium():
        from gymnasium import spaces as gs  # noqa: PLC0415

        return gs.Dict(dict(spaces))
    return Dict(spaces)


def batch(space: Any, n: int) -> Any:
    """The space of ``n`` stacked values (leading world axis). Example: ``batch(box(-1, 1, (3,)), 8).shape``."""
    if hasattr(space, "spaces"):
        return dict_space({k: batch(sp, n) for k, sp in space.spaces.items()})
    shape = (n, *space.shape)
    return box(np.broadcast_to(space.low, shape), np.broadcast_to(space.high, shape), shape, space.dtype)


__all__ = ["Box", "Dict", "batch", "box", "dict_space", "has_gymnasium"]
