# SPDX-License-Identifier: Apache-2.0
"""``alloy.Trajectory``: recorded frames of a run, as a dict of NumPy arrays with a leading frame axis.

Keys: ``"t"`` (frames,) seconds; per body ``"<name>.pos"`` (frames, worlds, 3), ``"<name>.quat_xyzw"`` and
``"<name>.quat_wxyz"`` (frames, worlds, 4), ``"<name>.vel"`` (frames, worlds, 3), and for a library object placed
by its anchor ``"<name>.body_pos"`` (its body origin; ``.pos`` is the anchor point); per joint ``"<name>.q"``
(frames, worlds), plus ``"<name>.rest"`` and ``"<name>.plastic_strain"`` for one that yields; per cloth / liquid
``"<name>.points"`` (frames, worlds, particles, 3), and for a metal sheet ``"<name>.plastic_strain"`` (frames,
worlds, vertices). Example::

    traj = sim.run(2.0, record=["cube"])
    z = traj["cube.pos"][:, 0, 2]             # world 0's height over time
    traj.world(0).save("out/traj.npz")         # drop the world axis, write with np.savez
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class Trajectory(dict):
    """A ``dict[str, np.ndarray]`` with helpers. ``meta`` holds run facts (dt, worlds, health at the end)."""

    def __init__(self, data: dict[str, np.ndarray] | None = None, meta: dict[str, Any] | None = None) -> None:
        super().__init__(data or {})
        self.meta: dict[str, Any] = dict(meta or {})

    @property
    def frames(self) -> int:
        """Number of recorded frames. Example: ``traj.frames``."""
        return int(len(self["t"])) if "t" in self else 0

    def world(self, w: int) -> Trajectory:
        """The same frames for world ``w`` only (world axis dropped). Example: ``traj.world(0)["cube.pos"].shape``."""
        out = {k: (v if k == "t" else v[:, w]) for k, v in self.items()}
        return Trajectory(out, {**self.meta, "world": w})

    def save(self, path: str | Path, **extra: Any) -> Path:
        """Write every array (plus ``extra`` arrays) with ``np.savez``; returns the path.

        Example: ``traj.world(0).save("out/traj.npz", faces=sim.faces("cloth"))``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{**self, **extra})
        return path

    def summary(self) -> dict[str, Any]:
        """Shapes and the run facts, JSON-ready. Example: ``{"frames": 201, "keys": {"cube.pos": [201, 1, 3]}}``."""
        return {"frames": self.frames, "keys": {k: list(np.shape(v)) for k, v in self.items()}, "meta": self.meta}
