# SPDX-License-Identifier: Apache-2.0
"""Batched, named state of every world at one moment: what observations and task checkers read.

Keys follow ``alloy.Sim.state`` and add velocities, episode time and the last action. Every value has a leading world
axis and is a backend array (a torch tensor on the GPU, or NumPy)::

    s = env.state
    s["cube.pos"]            # (W, 3) metres (origin)   s["cube.quat_wxyz"] / ["cube.quat_xyzw"]  (W, 4)
    s["cube.com"]            # (W, 3) centre of mass (a mesh's origin can sit at its base or a corner)
    s["cube.vel"]            # (W, 3) m/s               s["cube.angvel"]                          (W, 3) rad/s
    s["gripper/lift.q"]      # (W,) for a 1-dof joint   s["gripper/lift.qd"]                      (W,)
    s["water.points"]        # (W, n, 3) particles      s["water.center"]                         (W, 3)
    s["cable.points"]        # (W, N + 1, 3) a rod's centre line (Scene.add_rod, as sim.state() reads it)
    s["steps"], s["t"]       # (W,) steps and seconds since each world's episode started
    s["action"]              # (W, A) the last action
    s.start["cube.pos"]      # the same keys, at each world's last reset
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from alloy.envs.layout import Layout

#: engine State arrays an observation can read
ENGINE_ARRAYS = ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q")
BODY_FIELDS = ("pos", "body_pos", "com", "quat_xyzw", "quat_wxyz", "vel", "angvel")
#: model arrays a state can read (constant per body)
MODEL_ARRAYS = ("body_com",)
JOINT_FIELDS = ("q", "qd")
PARTICLE_FIELDS = ("points", "center", "active")
ROD_FIELDS = ("points",)


class Rows:
    """Backend index arrays for the layout's rows, made once and reused every step."""

    def __init__(self, layout: Layout, xp: Any, anchors: Mapping[str, Any] | None = None,
                 pours: Sequence[str] = ()) -> None:
        self.layout, self.xp = layout, xp
        self.pours = frozenset(pours)                          # add_pour streams: drops wait at the nozzle
        self._cache: dict[tuple[str, str], Any] = {}
        # objects placed by their anchor (Scene.add_object): "<name>.pos" is that body-frame point, as in sim.state()
        self.anchors = {n: xp.asarray(np.asarray(c, np.float32)) for n, c in (anchors or {}).items()
                        if n in layout.bodies}

    def get(self, kind: str, name: str) -> Any:
        """Rows of ``name``: kind "body", "q", "qd", "target_q" or "particle". Example: ``rows.get("body", "cube")``."""
        key = (kind, name)
        if key not in self._cache:
            lay = self.layout
            if kind == "body":
                raw = lay.body(name)
            elif kind == "particle":
                raw = lay.particle(name)
            else:
                raw = getattr(lay.joint(name), kind)
            self._cache[key] = self.xp.asarray(raw, "int64")
        return self._cache[key]

    def rod(self, name: str) -> tuple[Any, Any, Any]:
        """A rod's segment rows (N, W) and segment starts and ends in the body frame, each (N * W, 3), for
        ``"<rod>.points"``. Example: ``rows.rod("cable")``."""
        key = ("rod", name)
        if key not in self._cache:
            r = self.layout.rods[name]
            n, w = r.rows.shape
            tile = [self.xp.asarray(np.repeat(e[:, None, :], w, axis=1).reshape(n * w, 3)) for e in (r.start, r.end)]
            self._cache[key] = (self.xp.asarray(r.rows, "int64"), *tile)
        return self._cache[key]


class BatchState(Mapping[str, Any]):
    """Named state of every world; values are computed on first access and cached for this moment."""

    def __init__(self, arrays: Callable[[str], Any], rows: Rows, *, extra: Mapping[str, Any] | None = None,
                 start: BatchState | None = None) -> None:
        self._arrays, self.rows, self.xp = arrays, rows, rows.xp
        self._extra = dict(extra or {})
        self._cache: dict[str, Any] = {}
        self.start = start

    @property
    def worlds(self) -> int:
        """Number of worlds. Example: ``s.worlds``."""
        return self.rows.layout.worlds

    def __getitem__(self, key: str) -> Any:
        if key in self._extra:
            return self._extra[key]
        if key not in self._cache:
            self._cache[key] = self._compute(key)
        return self._cache[key]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self) -> list[str]:  # type: ignore[override]
        """Every key this state can give. Example: ``"cube.pos" in s.keys()``."""
        lay = self.rows.layout
        out = [f"{b}.{f}" for b in lay.bodies for f in BODY_FIELDS]
        out += [f"{j}.{f}" for j in lay.joints for f in JOINT_FIELDS]
        out += [f"{p}.{f}" for p in lay.particles for f in PARTICLE_FIELDS]
        out += [f"{r}.{f}" for r in lay.rods for f in ROD_FIELDS]
        return out + list(self._extra)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and (key in self._extra or key in self.keys())

    def _compute(self, key: str) -> Any:
        name, _, fld = key.rpartition(".")
        lay = self.rows.layout
        if not name:
            raise lay.unknown(key, "state key (\"<name>.<field>\")")
        if name in lay.bodies:
            return self._body(name, fld, key)
        if name in lay.joints:
            return self._joint(name, fld, key)
        if name in lay.particles:
            return self._particles(name, fld, key)
        raise lay.unknown(name)

    def _body(self, name: str, fld: str, key: str) -> Any:
        rows = self.rows.get("body", name)
        if fld in ("pos", "body_pos", "quat_xyzw", "quat_wxyz"):
            q = self._arrays("body_q")[rows]
            c = self.rows.anchors.get(name)
            if fld == "pos" and c is not None:                # the anchor point, turning with the object
                return q[:, 0:3] + rotate(self.xp, q[:, 3:7], c[None, :] + 0.0 * q[:, 0:3])
            if fld in ("pos", "body_pos"):                     # body_pos: the body frame's origin
                return q[:, 0:3]
            return q[:, 3:7] if fld == "quat_xyzw" else q[:, [6, 3, 4, 5]]
        if fld in ("vel", "angvel"):
            qd = self._arrays("body_qd")[rows]
            return qd[:, 0:3] if fld == "vel" else qd[:, 3:6]
        if fld == "com":                                    # centre of mass in the world (a mesh's origin may not be)
            q = self._arrays("body_q")[rows]
            return q[:, 0:3] + rotate(self.xp, q[:, 3:7], self._arrays("body_com")[rows])
        if name in self.rows.layout.rods:
            if fld in ROD_FIELDS:
                return self._rod_points(name)
            raise _bad_field(key, BODY_FIELDS + ROD_FIELDS)
        raise _bad_field(key, BODY_FIELDS)

    def _rod_points(self, name: str) -> Any:
        """(W, N + 1, 3): every segment's start, then the last segment's end (``alloy.api._rods.points``)."""
        rows, start, end = self.rows.rod(name)
        n, w = rows.shape
        q = self._arrays("body_q")[rows].reshape(n * w, 7)
        pos, quat = q[:, 0:3], q[:, 3:7]
        starts = (pos + rotate(self.xp, quat, start)).reshape(n, w, 3)
        last = pos[-w:] + rotate(self.xp, quat[-w:], end[-w:])
        return self.xp.stack([starts[k] for k in range(n)] + [last], 1)

    def _joint(self, name: str, fld: str, key: str) -> Any:
        if fld not in JOINT_FIELDS:
            raise _bad_field(key, JOINT_FIELDS)
        rows = self.rows.get(fld, name)
        vals = self._arrays("joint_q" if fld == "q" else "joint_qd")[rows]
        return vals[:, 0] if vals.shape[1] == 1 else vals

    def _particles(self, name: str, fld: str, key: str) -> Any:
        if fld not in PARTICLE_FIELDS:
            raise _bad_field(key, PARTICLE_FIELDS)
        rows = self.rows.get("particle", name)
        if fld == "active":
            return self._active(rows)
        pts = self._arrays("particle_q")[rows]
        if fld == "points" or name not in self.rows.pours:
            return pts if fld == "points" else pts.mean(1)
        xp = self.xp
        a = xp.float(self._active(rows))[..., None]           # a pour: the centre of what has been poured
        n = a.sum(1)
        return xp.where(n > 0, (pts * a).sum(1) / xp.clip(n, 1.0, float("inf")), pts.mean(1))

    def _active(self, rows: Any) -> Any:
        """(W, n) bool: the particles taking part (a pour's drops poured so far; none at a reset)."""
        try:
            return self._arrays("particle_active")[rows]
        except KeyError:                                        # the reset state: nothing poured yet
            return self.xp.zeros(tuple(rows.shape), "bool")


def rotate(xp: Any, q_xyzw: Any, v: Any) -> Any:
    """Rotate (W, 3) vectors by (W, 4) unit quaternions (x, y, z, w). Example: ``rotate(xp, q, v)``."""
    u, w = q_xyzw[:, 0:3], q_xyzw[:, 3:4]
    t = 2.0 * _cross(xp, u, v)
    return v + w * t + _cross(xp, u, t)


def _cross(xp: Any, a: Any, b: Any) -> Any:
    return xp.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], -1)


def _bad_field(key: str, fields: tuple[str, ...]) -> Exception:
    from alloy.envs.errors import EnvError  # noqa: PLC0415

    name = key.rpartition(".")[0]
    return EnvError(f"no state key {key!r}", f"use one of {', '.join(f'{name}.{f}' for f in fields)}", field=key)


__all__ = ["BODY_FIELDS", "BatchState", "ENGINE_ARRAYS", "JOINT_FIELDS", "MODEL_ARRAYS", "PARTICLE_FIELDS",
           "ROD_FIELDS", "Rows", "rotate"]
