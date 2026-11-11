# SPDX-License-Identifier: Apache-2.0
"""DC3 test helpers: the alloy-assets library, a force gauge for connector checks, and check-scene runs.

- ``assets_root()`` / ``conn_library()``: card access to ``alloy-assets`` (env ``ALLOY_ASSETS``, else the sibling
  checkout ``../alloy-assets``) through WP1's loader (SCR-DC-1, main 07315b12): ``CardLibrary()`` reads the assets
  repo recursively and resolves mesh/MJCF paths against its root.
- ``Gauge``: a force tester as a step observer. Per world it pulls one body along a world axis with a spring-damper
  to a moving target (a load cell on a crosshead, like the TE 108-2481 rate-controlled insertion tests); the
  recorded force is the spring force, positions are the body's. Other degrees of freedom stay free.
- ``check_sim``: assemble a scene card and build the standard IntegratedSimulation (health on).

Units SI. Everything here is host-side numpy over all worlds (eager runs only).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def assets_root() -> Path | None:
    """The alloy-assets checkout, or None (tests that need it skip)."""
    env = os.environ.get("ALLOY_ASSETS")
    root = Path(env) if env else REPO.parent / "alloy-assets"
    return root if (root / "objects").is_dir() else None


def conn_library(probe: str | None = None) -> Any:
    """``CardLibrary()`` over Alloy's library plus alloy-assets (raises FileNotFoundError without the checkout).

    The symlink-index stopgap is gone: with SCR-DC-1's recursive loader it indexed every assets card twice
    (CardError "defined twice") whenever its probe card was missing from the checkout; ``probe`` is ignored."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415

    root = assets_root()
    if root is None:
        raise FileNotFoundError("alloy-assets checkout not found (set ALLOY_ASSETS)")
    lib = CardLibrary()
    if lib.assets is None or lib.assets != root.resolve():
        raise FileNotFoundError(f"CardLibrary assets {lib.assets} is not the alloy-assets checkout {root}")
    return lib


@dataclass
class Gauge:
    """Spring-damper crosshead on ``body`` (template body index) in every world, along unit ``axis``.

    ``target(t, x_rel, f) -> (mode, value)`` is called once per step with the time since start, the body's
    displacement along the axis since start and the last force per world (arrays): mode ``"pos"`` sets the
    crosshead position (m, relative to the start), ``"force"`` applies a force directly (N), ``"free"`` applies 0.
    """

    body: int
    axis: tuple[float, float, float]
    target: Callable[[float, np.ndarray, np.ndarray], tuple[str, np.ndarray | float]]
    stiffness: float = 3000.0                  # N/m, the load-cell spring (an arm at the gripper: 300-3600 N/m)
    mass: float = 0.02                         # kg, sets critical damping
    every: int = 1                             # record every n-th step
    t: float = 0.0
    rows: list[tuple[float, np.ndarray, np.ndarray]] = field(default_factory=list)
    _x0: np.ndarray | None = None
    _f: np.ndarray | None = None
    _n: int = 0

    def bodies(self, model: Any) -> np.ndarray:
        per = int(model.body_count) // int(model.world_count)
        return self.body + per * np.arange(int(model.world_count))

    def __call__(self, s_in: Any, s_out: Any, control: Any, contacts: Any, dt: float) -> None:
        model = self._model
        ids = self.bodies(model)
        ax = np.asarray(self.axis, np.float64)
        q = s_out.body_q.numpy()[ids, :3].astype(np.float64) @ ax
        v = s_out.body_qd.numpy()[ids, :3].astype(np.float64) @ ax
        if self._x0 is None:
            self._x0, self._f = q.copy(), np.zeros_like(q)
        self.t += dt
        x = q - self._x0
        mode, val = self.target(self.t, x, self._f)
        if mode == "pos":
            f = self.stiffness * (np.asarray(val) - x) - 2.0 * np.sqrt(self.stiffness * self.mass) * v
        elif mode == "force":
            f = np.broadcast_to(np.asarray(val, np.float64), x.shape).copy()
        else:
            f = np.zeros_like(x)
        self._f = f
        bf = s_out.body_f.numpy()
        bf[ids, :3] = (f[:, None] * ax[None, :]).astype(bf.dtype)
        s_out.body_f.assign(bf)
        self._n += 1
        if self._n % self.every == 0:
            self.rows.append((self.t, x.copy(), f.copy()))

    def attach(self, sim: Any) -> "Gauge":
        self._model = sim.model
        sim.add_step_observer(self)
        return self

    def curve(self, world: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, x, f) arrays for one world."""
        t = np.array([r[0] for r in self.rows])
        return t, np.array([r[1][world] for r in self.rows]), np.array([r[2][world] for r in self.rows])


def ramp_to(speed: float, stop: float) -> Callable:
    """Crosshead at constant ``speed`` (m/s, signed) until it reaches ``stop`` (m), then held."""
    def target(t: float, x: np.ndarray, f: np.ndarray) -> tuple[str, float]:
        pos = speed * t
        return "pos", (min(pos, stop) if speed > 0 else max(pos, stop))
    return target


def force_ramp(rate: float, peak: float) -> Callable:
    """A direct force rising at ``rate`` (N/s, signed) to ``peak`` (N), then held (retention checks)."""
    def target(t: float, x: np.ndarray, f: np.ndarray) -> tuple[str, float]:
        val = rate * t
        return "force", (min(val, peak) if rate > 0 else max(val, peak))
    return target


def body_index(model: Any, suffix: str) -> int:
    """Template (world 0) body index whose label ends with ``suffix``."""
    per = int(model.body_count) // int(model.world_count)
    hits = [i for i, lab in enumerate(list(model.body_label)[:per]) if str(lab).endswith(suffix)]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} bodies end with {suffix!r}")
    return hits[0]


def joint_q(sim: Any, joint_suffix: str) -> np.ndarray:
    """Per-world coordinate of a 1-dof joint whose label ends with ``joint_suffix``."""
    m = sim.model
    per = int(m.joint_count) // int(m.world_count)
    labels = list(m.joint_label)[:per]
    hits = [i for i, lab in enumerate(labels) if str(lab).endswith(joint_suffix)]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} joints end with {joint_suffix!r}")
    starts = m.joint_q_start.numpy()
    q = sim.state.joint_q.numpy()
    return np.array([q[starts[hits[0] + w * per]] for w in range(int(m.world_count))])


def check_sim(scene: Any, lib: Any, device: str = "cpu", worlds: int = 1, **overrides: Any) -> Any:
    """Assemble ``scene`` (a SceneCard or a scene name) and build the standard IntegratedSimulation.

    CPU defaults: native MuJoCo contacts, no contact layer, no graph (the precision layer needs CUDA)."""
    from alloy.integration import build_simulation  # noqa: PLC0415
    from alloy.testing import integration as I  # noqa: PLC0415

    card = lib.scene(scene) if isinstance(scene, str) else scene
    built = I.assemble(card, lib, device, worlds)
    if device == "cpu":
        overrides = {"contact_layer": "never", "graph": "off", **overrides}
    return build_simulation(built, "standard", **overrides)


def run_periods(sim: Any, seconds: float) -> None:
    for _ in range(max(1, int(round(seconds / sim.built.control_dt)))):
        sim.step_period()


def invalid_worlds(sim: Any) -> Sequence[int]:
    return tuple(sim.report()["health"].invalid_worlds)
