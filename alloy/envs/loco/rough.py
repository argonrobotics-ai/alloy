# SPDX-License-Identifier: Apache-2.0
"""Go2 on rough ground with a difficulty curriculum: the flat velocity task on a field of terrain tiles.

Every world's robot starts on the platform of one tile. When a world's episode ends, its level goes up if the robot
walked more than half a tile away from its start, down if it covered less than half the distance its command asked
for, and it starts the next episode on a tile of the new level (a robot that clears the last level is sent to a
random one). ``env.levels`` is each world's level; ``env.state["terrain.level"]`` the same as a state key. Example::

    from alloy.envs.loco import go2_rough
    env = go2_rough(num_envs=1024)
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    float(env.levels.float().mean())          # mean terrain level (torch backend)

The policy gets no view of the ground ahead (no height scan): it walks blind, as on flat ground. An episode also
ends, as a termination, when the robot leaves the field. Heights (``base.height``, ``feet.height``) are measured
above the terrain under each point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.envs.checks import Check
from alloy.envs.errors import EnvError
from alloy.envs.loco import terms as T
from alloy.envs.loco.env import LocoConfig, LocoEnv
from alloy.envs.loco.go2 import ROBOT, Go2VelocityConfig, go2_velocity_task
from alloy.envs.loco.robot import GO2
from alloy.envs.loco.terrain import Terrain, TerrainSpec, terrain_mjcf

SPAWN_CLEARANCE = 0.02       # m the robot starts above its standing height on the platform


@dataclass
class Go2RoughConfig(Go2VelocityConfig):
    """Options of the rough-terrain Go2 environment: the flat task's, plus the terrain and the curriculum.

    Example: ``Go2RoughConfig(terrain=TerrainSpec(levels=3), start_levels=1)``."""

    terrain: TerrainSpec = field(default_factory=TerrainSpec)
    start_levels: int = 2            # worlds start on levels 0 .. start_levels - 1
    curriculum: bool = True          # False: every world keeps the level it was dealt


def go2_rough_scene(worlds: int = 1, seed: int = 0, dt: float = 0.02, terrain: Terrain | None = None) -> Any:
    """The Go2 standing in the middle of a terrain field (the library file with its floor replaced).

    Example: ``alloy.build(go2_rough_scene(16), worlds=16)``."""
    from alloy.api.scene import Scene  # noqa: PLC0415
    from alloy.cards.library import default_root  # noqa: PLC0415
    from alloy.envs.mjcf import without_bodies  # noqa: PLC0415

    path = default_root() / "robots" / "emb_e8_go2" / "scene.xml"
    if not path.is_file():
        raise EnvError(f"the Go2 robot file is missing ({path.name})", "install Alloy with its library (a source "
                       "checkout or the full wheel)", field="scene")
    scene = Scene(worlds=worlds, dt=dt)
    scene.add_robot(ROBOT, terrain_mjcf(without_bodies(path, ["obstacle"]), terrain or Terrain()))
    return scene


class RoughEnv(LocoEnv):
    """A ``LocoEnv`` on a ``Terrain``: per-world start tiles and the level curriculum. Example: ``go2_rough(64)``."""

    def __init__(self, sim: Any, task: Any, config: LocoConfig, terrain: Terrain, *, start_levels: int = 2,
                 curriculum: bool = True, **kwargs: Any) -> None:
        self.terrain, self.curriculum = terrain, curriculum
        self._start_levels = max(1, min(int(start_levels), terrain.levels))
        self._placed = False
        super().__init__(sim, task, config, **kwargs)
        xp, w = self.xp, self.num_envs
        self._base_q = xp.asarray(self.layout.free[f"{config.name}/{config.robot.base}"], "int64")
        self._xyz = self._base_q[:, None] + xp.asarray(np.arange(3), "int64")[None, :]
        self._stand = float(xp.to_numpy(self.state["base.height"]).mean())
        self.kinds = xp.asarray(np.arange(w) % terrain.kinds, "int64")
        self.levels = xp.asarray(np.arange(w) // terrain.kinds % self._start_levels, "int64")
        self.spawn = xp.zeros((w, 2))
        self.randomizer = _Placer(self.randomizer, self)
        self._placed = True

    def _ground_height(self) -> Any:
        return lambda xy: self.terrain.height_at(self.xp, xy)

    def _base_extra(self) -> dict[str, Any]:
        extra = super()._base_extra()
        if self._placed:
            extra["terrain.level"] = self.xp.float(self.levels)
            extra["terrain.spawn"] = self.spawn
        return extra

    def reset_worlds(self, mask: Any) -> None:
        """Reset the worlds in the bool (W,) ``mask``; each moves to a tile of its (updated) level."""
        mask = self._mask(mask)
        if self._placed and self.curriculum and self.xp.any(mask):
            self._update_levels(mask)
        super().reset_worlds(mask)

    def _update_levels(self, mask: Any) -> None:
        xp, t = self.xp, self.terrain
        s = self.state
        played = mask & (s["steps"] > 0)
        walked = xp.norm(s[f"{self.config.name}/{self.config.robot.base}.pos"][:, 0:2] - self.spawn, -1)
        asked = xp.norm(s["command"][:, 0:2], -1) * s["t"]
        up = played & (walked > 0.5 * t.spec.tile)
        down = played & ~up & (walked < 0.5 * asked)
        level = self.levels + xp.asarray(up, "int64") - xp.asarray(down, "int64")
        again = xp.asarray(xp.uniform((self.num_envs,), 0.0, float(t.levels)), "int64")
        level = xp.where(level >= t.levels, xp.clip(again, 0, t.levels - 1), level)
        self.levels = xp.clip(level, 0, t.levels - 1)

    def place(self, mask: Any) -> None:
        """Put the masked worlds' robots on the platform of their tile (called right after the engine reset)."""
        xp, t = self.xp, self.terrain
        x, y = t.tile_center(xp.float(self.levels), xp.float(self.kinds))
        xy = xp.stack([x, y], -1)
        z = t.height_at(xp, xy) + self._stand + SPAWN_CLEARANCE
        new = xp.cat([xy, z[:, None]], -1)
        self.spawn = xp.where(mask[:, None], xy, self.spawn)

        def write(q: Any) -> None:
            q[self._xyz] = xp.where(mask[:, None], new, q[self._xyz])
        xp.update(self.sim.engine.state.joint_q, write)


class _Placer:
    """The environment's randomizer, followed by the move to the start tile (one forward-kinematics pass)."""

    def __init__(self, inner: Any, env: RoughEnv) -> None:
        self.inner, self.env = inner, env
        self.jitters = [*inner.jitters, "terrain start"]

    def apply(self, mask: Any) -> None:
        if self.inner.jitters:
            self.inner.apply(mask)
        self.env.place(mask)
        self.inner._forward_kinematics(mask)


def off_field(terrain: Terrain, body: str) -> Check:
    """True when ``body`` has left the terrain field. Example: ``off_field(terrain, "go2/base")``."""
    return Check("off_field", lambda s: ~terrain.inside(s[f"{body}.pos"][:, 0:2]))


def go2_rough(num_envs: int = 1, device: str | None = None, seed: int = 0, config: Go2RoughConfig | None = None,
              **env_options: Any) -> RoughEnv:
    """The rough-terrain Go2 environment with ``num_envs`` worlds (``env_options``: ``backend=``, ``autoreset=``).

    Example: ``env = go2_rough(num_envs=1024, config=Go2RoughConfig(pushes=None))``."""
    cfg = config or Go2RoughConfig()
    terrain = Terrain(cfg.terrain)
    scene = go2_rough_scene(num_envs, seed, cfg.dt, terrain)
    for name, fld, low, high in cfg.scene_randomize:
        scene.randomize(name, fld, low, high)
    task = go2_velocity_task(cfg)
    task.name = "Go2Rough-v0"
    task.failure = T.fell() | off_field(terrain, f"{ROBOT}/{GO2.base}")
    loco = LocoConfig(robot=GO2, name=ROBOT, commands=cfg.commands, pushes=cfg.pushes,
                      action_scale=cfg.action_scale, contact=cfg.contact)
    env = RoughEnv(scene, task, loco, terrain, start_levels=cfg.start_levels, curriculum=cfg.curriculum,
                   worlds=num_envs, device=device, seed=seed, **env_options)
    env.reset(seed=seed)
    return env


__all__ = ["Go2RoughConfig", "RoughEnv", "go2_rough", "go2_rough_scene", "off_field"]
