# SPDX-License-Identifier: Apache-2.0
"""A stand-in for ``alloy.Sim`` so the command line and tool server can be tested without building (no kernels).

``fake_build`` has ``alloy.build``'s signature; its Sim has a free box "cube" falling at 0.1 m/s, an actuated
joint "arm/lift" (short name "lift") that jumps to its target, and a liquid "water" of 5 particles. World ``w``
goes INVALID once the cube passes ``invalid_below`` (to test failing health). Example::

    from alloy.cli.session import Session
    from alloy.cli.testing import fake_build
    session = Session(builder=fake_build)
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api.errors import SimulationError, UnknownNameError
from alloy.api.trajectory import Trajectory

PARTICLES = 5


class FakeSim:
    """See the module docstring. ``built`` records the build arguments; ``closed`` turns True on ``close()``."""

    invalid_below = -1.0

    def __init__(self, scene: Any, worlds: int | None = None, device: str | None = None, seed: int = 0, *,
                 profile: str = "standard", contacts: str = "auto") -> None:
        self.scene, self.worlds, self.dt = scene, int(worlds or scene.worlds), float(scene.dt)
        self.device, self.profile = device or "cpu", profile
        self.built = {"worlds": self.worlds, "device": self.device, "seed": seed, "profile": profile,
                      "contacts": contacts}
        self.closed = False
        self.reset()

    def reset(self) -> FakeSim:
        self.t, self.physics_steps = 0.0, 0
        self.z = np.full(self.worlds, 1.0, np.float32)
        self.q = np.zeros(self.worlds)
        self.target = np.zeros(self.worlds)
        return self

    def names(self) -> dict[str, list[str]]:
        return {"bodies": ["cube"], "joints": ["arm/lift", "lift"], "particles": ["water"], "static": ["ground"]}

    def routes(self) -> dict[str, dict[str, Any]]:
        return {"cube": {"route": "rigid", "solver": "mujoco", "reasons": []},
                "water": {"route": "liquid", "solver": "sph", "reasons": []}}

    def step(self, n: int = 1) -> FakeSim:
        self.t += n * self.dt
        self.physics_steps += 4 * n
        self.z = (self.z - np.float32(0.1 * n * self.dt)).astype(np.float32)
        self.q = self.target.copy()
        return self

    def state(self, names: Any = None, *, world: int | None = None) -> dict[str, Any]:
        names = ["cube", "arm/lift", "water"] if names is None else ([names] if isinstance(names, str) else names)
        out: dict[str, Any] = {"t": self.t}
        for n in names:
            out.update(self._read(n))
        if world is not None:
            if not 0 <= world < self.worlds:
                raise SimulationError(f"world {world} is outside 0..{self.worlds - 1}", "pass world=0 for the first")
            out = {k: (v if k == "t" else v[world]) for k, v in out.items()}
        return out

    def _read(self, name: str) -> dict[str, np.ndarray]:
        w = self.worlds
        if name == "cube":
            pos = np.stack([np.zeros(w, np.float32), np.zeros(w, np.float32), self.z], axis=1)
            return {"cube.pos": pos, "cube.quat_wxyz": np.tile(np.float32([1, 0, 0, 0]), (w, 1)),
                    "cube.vel": np.tile(np.float32([0, 0, -0.1]), (w, 1))}
        if name in ("arm/lift", "lift"):
            return {f"{name}.q": self.q.copy()}
        if name == "water":
            pts = np.zeros((w, PARTICLES, 3), np.float32) + np.arange(PARTICLES, dtype=np.float32)[None, :, None]
            return {"water.points": pts}
        raise UnknownNameError(f"no body, joint or particle object {name!r}", "use one of cube, arm/lift, water",
                               field=name)

    def set_targets(self, targets: dict[str, Any], kind: str = "position") -> FakeSim:
        for name, value in targets.items():
            self._read(name)
            if name in ("cube", "water"):
                raise UnknownNameError(f"{name!r} is not a joint", "use arm/lift", field=name)
            self.target = np.broadcast_to(np.asarray(value, float), (self.worlds,)).copy()
        return self

    def run(self, seconds: float, record: Any = None, every: int = 1, *, targets: Any = None) -> Trajectory:
        periods = round(seconds / self.dt)
        frames, times = [self.state(record)], [self.t]
        for k in range(1, periods + 1):
            for name, (a, b) in (targets or {}).items():
                self.set_targets({name: a + (b - a) * k / periods})
            self.step(1)
            if k % every == 0 or k == periods:
                frames.append(self.state(record))
                times.append(self.t)
        data = {"t": np.asarray(times)}
        for key in frames[0]:
            if key != "t":
                data[key] = np.stack([f[key] for f in frames])
        return Trajectory(data, {"wall_s": 0.01, "physics_steps": 4 * periods, "health": self.health()})

    def health(self) -> dict[str, Any]:
        bad = [w for w in range(self.worlds) if self.z[w] < self.invalid_below]
        rows = [{"severity": "INVALID", "flags": ["NONFINITE_Q"], "fixes": ["lower Scene(dt=...)"],
                 "problem": "a position became NaN/inf"} if w in bad else {"severity": "OK", "flags": [], "fixes": []}
                for w in range(self.worlds)]
        return {"ok": not bad, "monitored": True, "invalid_worlds": bad, "degraded_worlds": [], "worlds": rows}

    def check(self) -> dict[str, Any]:
        h = self.health()
        return {"ok": h["ok"], "source": "fake", "problems": [], "invalid_worlds": h["invalid_worlds"]}

    def render(self, camera: Any = None, *, world: int = 0, width: int = 64, height: int = 48,
               path: str | None = None, **kw: Any) -> np.ndarray:
        img = np.full((height, width, 3), 128, np.uint8)
        if path is not None:
            write_png(Path(path), img)
        return img

    def close(self) -> None:
        self.closed = True


def fake_build(scene: Any, **kwargs: Any) -> FakeSim:
    """``alloy.build`` stand-in. Example: ``Session(builder=fake_build)``."""
    return FakeSim(scene, **kwargs)


def write_png(path: Path, img: np.ndarray) -> Path:
    """A minimal RGB PNG (standard library only).

    Example: ``write_png(Path("a.png"), np.zeros((4, 4, 3), np.uint8))``."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    head = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw)) +
                     chunk(b"IEND", b""))
    return path
