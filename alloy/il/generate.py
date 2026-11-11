# SPDX-License-Identifier: Apache-2.0
"""Demonstrations at scale: run a scripted policy in every world until enough episodes are saved.

``generate`` drives a ``Recorder`` for you: all worlds step together, every finished episode goes through ``keep``
and to the writers, and the result says how many episodes were tried, how many succeeded and how fast it went.
Example::

    import alloy.envs
    from alloy.il import HDF5Writer, demonstrator, generate

    env = alloy.envs.make("GripperLift-v0", num_envs=256)
    result = generate(env, demonstrator(env, noise=0.1), [HDF5Writer("lift_1k.hdf5")], episodes=1000, seed=0)
    print(result.summary())          # saved, attempted, success rate with its interval, episodes per second

``demonstrator(env)`` is the scripted policy that ships for the env's task (``DEMONSTRATORS`` lists them); for
another task write a ``PhaseScript`` or any ``policy(obs) -> actions``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.il.episode import Episode, Writer
from alloy.il.evaluate import wilson_interval
from alloy.il.recorder import Recorder
from alloy.il.script import Phase, PhaseScript

log = logging.getLogger("alloy.il")


@dataclass
class GenerateResult:
    """What a ``generate`` run did. Example: ``generate(env, policy, writers, 100).summary()``."""

    saved: int             # episodes written
    wanted: int            # episodes asked for
    attempted: int         # episodes that ended during the run, kept or not
    successes: int         # of those, the ones that passed the task's success check and were never INVALID
    steps: int             # control steps taken (each steps every world)
    worlds: int
    wall_s: float          # wall time of the stepping loop, writers included
    seed: int | None = None
    invalid: int = 0       # attempted episodes whose world the engine marked INVALID (froze); never a success

    @property
    def complete(self) -> bool:
        """True when every wanted episode was saved. Example: ``assert result.complete``."""
        return self.saved >= self.wanted

    @property
    def success_rate(self) -> float:
        """Share of the attempted episodes that succeeded, 0 to 1. Example: ``result.success_rate``."""
        return self.successes / self.attempted if self.attempted else float("nan")

    @property
    def interval(self) -> tuple[float, float]:
        """95 % Wilson score interval of the success rate. Example: ``lo, hi = result.interval``."""
        return wilson_interval(self.successes, self.attempted)

    @property
    def episodes_per_s(self) -> float:
        """Saved episodes per wall second. Example: ``result.episodes_per_s``."""
        return self.saved / self.wall_s if self.wall_s > 0 else float("nan")

    @property
    def world_steps_per_s(self) -> float:
        """Control steps times worlds per wall second. Example: ``result.world_steps_per_s``."""
        return self.steps * self.worlds / self.wall_s if self.wall_s > 0 else float("nan")

    def summary(self) -> dict[str, Any]:
        """A JSON-ready summary. Example: ``json.dumps(result.summary())``."""
        lo, hi = self.interval
        return {"saved": self.saved, "wanted": self.wanted, "complete": self.complete, "attempted": self.attempted,
                "successes": self.successes, "invalid": self.invalid, "success_rate": self.success_rate,
                "success_interval_95": [lo, hi],
                "steps": self.steps, "worlds": self.worlds, "wall_s": self.wall_s,
                "episodes_per_s": self.episodes_per_s, "world_steps_per_s": self.world_steps_per_s,
                "seed": self.seed}


def generate(env: Any, policy: Callable[[Any], Any], writers: Sequence[Writer], episodes: int, *,
             seed: int | None = 0, keep: str | Callable[[Episode], bool] = "success", task: str | None = None,
             max_attempts: int | None = None, states: bool = True,
             progress: Callable[[GenerateResult], None] | None = None) -> GenerateResult:
    """Run ``policy`` in every world of ``env`` until ``episodes`` episodes are saved to ``writers``, then close
    the writers. ``keep``, ``task`` and ``states`` are the ``Recorder``'s. The run stops early, with
    ``result.complete`` False, once ``max_attempts`` episodes have ended (default: 20 per wanted episode, at least
    one round of every world), so a script that never succeeds cannot run for ever. ``progress(result)`` is called
    with the counts so far each time episodes end.

    Example: see the module docstring."""
    if episodes < 1:
        raise EnvError(f"episodes is {episodes}", "pass a positive number of episodes", field="episodes")
    worlds = int(env.num_envs)
    budget = max_attempts if max_attempts is not None else max(20 * episodes, worlds)
    to_np = env.xp.to_numpy
    result = GenerateResult(saved=0, wanted=int(episodes), attempted=0, successes=0, steps=0, worlds=worlds,
                            wall_s=0.0, seed=seed)
    with Recorder(env, writers, task=task, keep=keep, states=states, max_episodes=episodes) as rec:
        obs, _ = rec.reset(seed=seed)
        t0 = time.perf_counter()
        while not rec.full and result.attempted < budget:
            obs, _, _, _, info = rec.step(policy(obs))
            result.steps += 1
            if "_episode" not in info:
                continue
            done = np.asarray(to_np(info["_episode"]), bool)
            result.attempted += int(done.sum())
            result.invalid = rec.invalid
            result.successes += int((np.asarray(to_np(info["episode"]["success"]), bool) & done
                                     & ~rec.last_invalid).sum())
            result.saved, result.wall_s = rec.saved, time.perf_counter() - t0
            if progress is not None:
                progress(result)
        result.saved, result.wall_s = rec.saved, time.perf_counter() - t0
    if not result.complete:
        log.warning("generate: %d of %d episodes saved after %d attempts (%d succeeded); the run stopped at "
                    "max_attempts", result.saved, result.wanted, result.attempted, result.successes)
    if result.invalid:
        log.warning("generate: the engine marked %d of %d episodes INVALID (their worlds froze); none counts as a "
                    "success or is saved by keep='success'", result.invalid, result.attempted)
    return result


# --- shipped demonstrators -----------------------------------------------------------------------------------------


def gripper_lift(env: Any, *, noise: float = 0.0, seed: int | None = 0) -> PhaseScript:
    """GripperLift-v0 / GripperLiftCamera-v0: squeeze the cube for 25 steps, then lift and hold.

    Example: ``policy = gripper_lift(env, noise=0.1)``."""
    return PhaseScript(env, [Phase([0.0, 1.0, 1.0], steps=25, name="squeeze"),
                             Phase([1.0, 1.0, 1.0], name="lift")], noise=noise, seed=seed)


def pick_place(env: Any, *, noise: float = 0.0, seed: int | None = 0) -> PhaseScript:
    """PickPlace-v0: move over the object, go down, grip, lift, carry to the bin, drop. It follows the object's
    and the palm's positions (closed loop) and never checks that the grasp held.

    Example: ``policy = pick_place(env)``."""
    xp, bin_xy, n = env.xp, np.array([0.18, 0.0]), int(env.num_envs)

    def move(stage: str) -> Callable[[Any], np.ndarray]:
        def action(s: Any) -> np.ndarray:
            palm, obj = xp.to_numpy(s["gripper/palm.pos"]), xp.to_numpy(s["obj.com"])
            a = np.zeros((n, 4), np.float32)
            a[:, 3] = 1.0 if stage in ("grip", "up", "carry") else -1.0          # close / open the fingers
            if stage in ("over", "down"):
                a[:, :2] = np.clip((obj[:, :2] - palm[:, :2]) / 0.01, -1, 1)
            if stage == "down":
                a[:, 2] = np.clip((np.maximum(obj[:, 2] + 0.04, 0.075) - palm[:, 2]) / 0.01, -1, 1)
            if stage == "up":
                a[:, 2] = 1.0
            if stage == "carry":
                a[:, :2] = np.clip((bin_xy - palm[:, :2]) / 0.01, -1, 1)
            return a
        return action

    stages = [("over", 25), ("down", 25), ("grip", 15), ("up", 20), ("carry", 45), ("drop", 20)]
    phases = [Phase(move(name), steps=steps, name=name) for name, steps in stages]
    return PhaseScript(env, [*phases, Phase([0.0, 0.0, 0.0, 0.0], name="rest")], noise=noise, seed=seed)


#: task name -> the scripted policy that ships for it
DEMONSTRATORS: dict[str, Callable[..., PhaseScript]] = {
    "GripperLift-v0": gripper_lift, "GripperLiftCamera-v0": gripper_lift, "PickPlace-v0": pick_place}


def demonstrator(env: Any, *, noise: float = 0.0, seed: int | None = 0) -> PhaseScript:
    """The scripted policy shipped for ``env``'s task. Example: ``generate(env, demonstrator(env), writers, 100)``."""
    name = str(env.task.name)
    if name not in DEMONSTRATORS:
        raise EnvError(f"no scripted demonstrator ships for {name}", f"write one with PhaseScript (shipped: "
                       f"{sorted(DEMONSTRATORS)}), or pass any policy(obs) -> actions to generate", field="env")
    return DEMONSTRATORS[name](env, noise=noise, seed=seed)


__all__ = ["DEMONSTRATORS", "GenerateResult", "demonstrator", "generate", "gripper_lift", "pick_place"]
