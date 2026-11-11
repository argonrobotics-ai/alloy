# SPDX-License-Identifier: Apache-2.0
"""A recording of a robot's joints: time, commanded joint targets and measured joint positions, in episodes.

``Recording`` is what the fit reads, whatever the file was. Units inside are SI: seconds, radians (metres for a
slide), rad/s, N·m. The readers are in ``alloy.sysid.readers``; ``alloy.sysid.units`` checks the units.

CSV layout (also what ``save_csv`` writes): a header row, then one row per sample::

    t,cmd.joint1,pos.joint1,cmd.joint2,pos.joint2[,vel.joint1,tau.joint1,...][,episode]

``t`` in seconds, ``cmd.<joint>`` the commanded joint target, ``pos.<joint>`` the measured position, optional
``vel.<joint>`` and ``tau.<joint>``, and an optional integer ``episode`` column when the file holds several takes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.sysid.errors import SysidError

CSV_FIX = ("name the columns t, cmd.<joint>, pos.<joint> (optional vel.<joint>, tau.<joint>, episode); "
           "docs/guide/sysid.md shows a file")


@dataclass
class Episode:
    """One take: ``t`` (T,) s from 0, ``command`` and ``position`` (T, J); ``velocity`` / ``torque`` (T, J) or None."""

    t: np.ndarray
    command: np.ndarray
    position: np.ndarray
    velocity: np.ndarray | None = None
    torque: np.ndarray | None = None

    @property
    def duration(self) -> float:
        """Seconds from the first to the last sample."""
        return float(self.t[-1] - self.t[0])

    @property
    def dt(self) -> float:
        """Median sample period, s."""
        return float(np.median(np.diff(self.t)))


@dataclass
class Recording:
    """Episodes of the same ``joints`` (names as in the robot file, without the robot's name)."""

    joints: list[str]
    episodes: list[Episode]
    source: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def dt(self) -> float:
        """Median sample period over all episodes, s."""
        return float(np.median(np.concatenate([np.diff(e.t) for e in self.episodes])))

    @property
    def duration(self) -> float:
        """Total recorded seconds."""
        return float(sum(e.duration for e in self.episodes))

    def select(self, joints: list[str]) -> Recording:
        """The same recording with only ``joints``, in that order. Example: ``rec.select(["joint1"])``."""
        missing = [j for j in joints if j not in self.joints]
        if missing:
            raise SysidError(f"the recording has no joint {missing[0]!r}", f"its joints are {self.joints}",
                             field="joints")
        cols = [self.joints.index(j) for j in joints]

        def pick(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[:, cols]

        eps = [Episode(e.t, e.command[:, cols], e.position[:, cols], pick(e.velocity), pick(e.torque))
               for e in self.episodes]
        return Recording(list(joints), eps, self.source, list(self.notes))


def make_episode(t: Any, command: Any, position: Any, velocity: Any = None, torque: Any = None) -> Episode:
    """An ``Episode`` from array-likes, checked: finite, time rising, shapes agree; time is shifted to start at 0.

    Example: ``make_episode(t, cmd, pos)`` with ``t`` (T,), ``cmd`` and ``pos`` (T, J)."""
    t = np.asarray(t, np.float64).reshape(-1)
    arrays = {"command": command, "position": position, "velocity": velocity, "torque": torque}
    out: dict[str, np.ndarray | None] = {}
    columns = 1 if np.ndim(position) < 2 else np.shape(position)[1]
    for name, a in arrays.items():
        if a is None:
            out[name] = None
            continue
        a = np.asarray(a, np.float64)
        a = a.reshape(-1, 1) if a.ndim == 1 else a
        if a.ndim != 2 or a.shape[0] != len(t) or a.shape[1] != columns:
            raise SysidError(f"{name} has shape {a.shape}, which does not match {len(t)} samples of the positions",
                             "give one row per time sample and one column per joint", field=name)
        if not np.all(np.isfinite(a)):
            raise SysidError(f"{name} holds NaN or infinite values", "drop or repair those samples", field=name)
        out[name] = a
    if len(t) < 10:
        raise SysidError(f"an episode has only {len(t)} samples", "record at least a few seconds of motion",
                         field="t")
    if not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0.0):
        raise SysidError("time must be finite and strictly rising within an episode", "sort the samples by time and "
                         "drop repeated time stamps (add an `episode` column if the file holds several takes)",
                         field="t")
    return Episode(t - t[0], out["command"], out["position"], out["velocity"], out["torque"])


def load_csv(path: str | Path) -> Recording:
    """Read the CSV layout of this module's docstring. Example: ``load_csv("take.csv").joints``."""
    path = Path(path)
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        raise SysidError(f"{path.name} has no data rows", CSV_FIX, field="data")
    header = [h.strip() for h in rows[0]]
    try:
        table = np.asarray([[float(v) for v in r] for r in rows[1:] if r], np.float64)
    except ValueError as exc:
        raise SysidError(f"{path.name}: a cell is not a number ({exc})", CSV_FIX, field="data") from exc
    if table.ndim != 2 or table.shape[1] != len(header):
        raise SysidError(f"{path.name}: rows do not all have the header's {len(header)} columns", CSV_FIX,
                         field="data")
    joints = [h[4:] for h in header if h.startswith("pos.")]
    if "t" not in header or not joints:
        raise SysidError(f"{path.name}: the header {header[:6]}... has no `t` column or no `pos.<joint>` column",
                         CSV_FIX, field="data")
    missing = [j for j in joints if f"cmd.{j}" not in header]
    if missing:
        raise SysidError(f"{path.name}: joint {missing[0]!r} has a pos. column but no cmd. column", CSV_FIX,
                         field="data")

    def cols(prefix: str) -> np.ndarray | None:
        names = [f"{prefix}.{j}" for j in joints]
        return table[:, [header.index(n) for n in names]] if all(n in header for n in names) else None

    episode = table[:, header.index("episode")].astype(np.int64) if "episode" in header else np.zeros(len(table), int)
    t, parts = table[:, header.index("t")], {p: cols(p) for p in ("cmd", "pos", "vel", "tau")}
    eps = []
    for e in np.unique(episode):
        m = episode == e
        eps.append(make_episode(t[m], parts["cmd"][m], parts["pos"][m],
                                None if parts["vel"] is None else parts["vel"][m],
                                None if parts["tau"] is None else parts["tau"][m]))
    return Recording(joints, eps, source=str(path))


def save_csv(rec: Recording, path: str | Path) -> Path:
    """Write ``rec`` in the CSV layout. Example: ``save_csv(rec, "take.csv")``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kinds = [("cmd", "command"), ("pos", "position")]
    kinds += [(p, a) for p, a in (("vel", "velocity"), ("tau", "torque"))
              if all(getattr(e, a) is not None for e in rec.episodes)]
    header = ["t"] + [f"{p}.{j}" for j in rec.joints for p, _ in kinds] + ["episode"]
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(header)
        for k, e in enumerate(rec.episodes):
            for i in range(len(e.t)):
                row = [repr(float(e.t[i]))]
                row += [repr(float(getattr(e, a)[i, c])) for c in range(len(rec.joints)) for _, a in kinds]
                out.writerow([*row, k])
    return path


__all__ = ["Episode", "Recording", "load_csv", "make_episode", "save_csv"]
