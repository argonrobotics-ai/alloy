# SPDX-License-Identifier: Apache-2.0
"""Scripted demonstrators: a policy written as phases, each world at its own phase.

A ``PhaseScript`` is a ``policy(obs) -> actions`` for a ``VectorEnv``. Every world walks the phases in order from
the start of its episode; a phase gives the action (fixed columns, or a function of the env's state for closed-loop
moves) and ends after a number of steps or when a condition holds in that world. Worlds reset on their own, so at
any step different worlds are in different phases. Example::

    from alloy.il import Phase, PhaseScript

    script = PhaseScript(env, [
        Phase([0.0, 1.0, 1.0], steps=25, name="squeeze"),              # action columns, held for 25 steps
        Phase([1.0, 1.0, 1.0], until=lambda s: s["cube.pos"][:, 2] > 0.12, name="lift"),
        Phase([0.0, 1.0, 1.0], name="hold"),                           # the last phase lasts to the episode's end
    ], noise=0.1, seed=0)
    obs, _ = env.reset(seed=0)
    obs, *_ = env.step(script(obs))

``noise`` adds uniform noise to every action (what the env executes and the recorder stores), so the episodes differ
by more than their start states.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError

#: a phase's action: fixed columns (A,), or ``fn(state) -> (W, A)`` for the env's ``state``
Action = Sequence[float] | Callable[[Any], Any]


@dataclass
class Phase:
    """One stage of a script. Example: ``Phase([0, 1, 1], steps=25, name="squeeze")``."""

    action: Action
    steps: int | None = None                       # leave the phase after this many steps in it
    until: Callable[[Any], Any] | None = None      # or once ``until(env.state)`` (W,) bool holds for the world
    name: str = ""


class PhaseScript:
    """A phased demonstrator for ``env``: call it with the observation, get ``(num_envs, A)`` actions.

    Example: see the module docstring."""

    def __init__(self, env: Any, phases: Sequence[Phase], *, noise: float = 0.0, seed: int | None = 0) -> None:
        self.env, self.phases, self.noise = env, list(phases), float(noise)
        self.worlds, self.size = int(env.num_envs), int(env.writer.size)
        self.absolute = env.writer.spec.mode == "absolute"
        self._validate()
        self.rng = np.random.default_rng(seed)
        self.phase = np.zeros(self.worlds, np.int64)       # each world's phase
        self.since = np.zeros(self.worlds, np.int64)       # steps the world has spent in it
        self._steps = np.full(self.worlds, -1, np.int64)   # episode step each world was at on the last call

    def _validate(self) -> None:
        if not self.phases:
            raise EnvError("a script needs at least one phase", "pass a list of Phase(...)", field="phases")
        if self.noise < 0.0:
            raise EnvError(f"noise is {self.noise}", "pass a noise amplitude >= 0", field="noise")
        for k, p in enumerate(self.phases):
            if not callable(p.action) and np.shape(p.action) != (self.size,):
                raise EnvError(f"phase {k} ({p.name!r}) has an action of shape {np.shape(p.action)}",
                               f"give {self.size} values, one per action column {list(self.env.action_names())}, "
                               "or a function of the env's state", field="phases")
            if p.steps is not None and p.steps < 1:
                raise EnvError(f"phase {k} ({p.name!r}) has steps={p.steps}", "pass a positive number of steps, or "
                               "None with until=", field="phases")

    # --- the policy ------------------------------------------------------------------------------------------------

    def __call__(self, obs: Any = None) -> Any:
        """Actions for the coming step, as the env's array type. Example: ``env.step(script(obs))``."""
        state, to_np = self.env.state, self.env.xp.to_numpy
        steps = np.asarray(to_np(state["steps"]), np.int64)
        fresh = (steps < self._steps) | (steps == 0)        # the world's episode restarted since the last call
        self.phase[fresh], self.since[fresh] = 0, 0
        self._advance(state)
        actions = self._actions(state)
        if self.noise:
            actions = actions + self.rng.uniform(-self.noise, self.noise, actions.shape).astype(np.float32)
        if not self.absolute:
            actions = np.clip(actions, -1.0, 1.0)
        self.since += 1
        self._steps = steps
        return self.env.xp.asarray(actions.astype(np.float32, copy=False))

    def names(self) -> list[str]:
        """The phase each world is in now. Example: ``script.names()[0]`` -> ``"lift"``."""
        return [self.phases[k].name or str(k) for k in self.phase]

    def _advance(self, state: Any) -> None:
        """Move every world whose phase is over to the next one (a world can pass several in one step)."""
        last = len(self.phases) - 1
        for k, p in enumerate(self.phases[:-1]):
            here = self.phase == k
            if not here.any():
                continue
            over = np.zeros(self.worlds, bool)
            if p.steps is not None:
                over |= self.since >= p.steps
            if p.until is not None:
                over |= np.asarray(self.env.xp.to_numpy(p.until(state)), bool).reshape(self.worlds)
            move = here & over
            self.phase[move], self.since[move] = min(k + 1, last), 0

    def _actions(self, state: Any) -> np.ndarray:
        out = np.zeros((self.worlds, self.size), np.float32)
        for k, p in enumerate(self.phases):
            here = self.phase == k
            if not here.any():
                continue
            if callable(p.action):
                value = np.asarray(self.env.xp.to_numpy(p.action(state)), np.float32)
                if value.shape != out.shape:
                    raise EnvError(f"phase {k} ({p.name!r}) returned actions of shape {value.shape}",
                                   f"return ({self.worlds}, {self.size}): one row per world", field="phases")
                out[here] = value[here]
            else:
                out[here] = np.asarray(p.action, np.float32)
        return out


__all__ = ["Action", "Phase", "PhaseScript"]
