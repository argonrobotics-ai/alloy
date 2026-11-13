# SPDX-License-Identifier: Apache-2.0
"""A task: what the policy sees (observations), what it drives (actions), what it is rewarded for (task checkers),
and when an episode ends (success, failure, a step limit).

Example::

    from alloy.envs import Task, ActionSpec, checks as C
    task = Task(
        obs=["gripper", "cube"],                               # robot joints (q, qd); body pose and velocity
        actions=ActionSpec(["gripper/lift", "gripper/close_left", "gripper/close_right"]),
        reward={"reach": (-1.0, C.distance("gripper", "cube")), "lift": (5.0, C.rise("cube"))},
        success=C.lifted("cube", by=0.08), failure=C.height("cube") < -0.5, max_steps=200)

Observation entries: a body ("cube": pos, quat_wxyz, vel, angvel), a robot ("gripper": every joint's q and qd), a
joint ("gripper/lift": q, qd), a particle object ("water": its centre), a state key ("cube.pos"), "action" (the last
action), a ``Check`` (its value), or ``(label, fn)`` with ``fn(state) -> (W, k)``. ``obs=None``: every robot and
object. Cameras (``alloy.camera.Camera``) add image observations: ``Task(cameras={"front": cam})``. Random starts:
``Task(randomize=[jitter_body("cube", pos=(0.02, 0.02, 0)), jitter_joint("gripper/lift", by=0.01)])``. Goal points
redrawn in a box at every reset: ``Task(goals={"target": (lo_xyz, hi_xyz)})``, read as ``state["target.pos"]``, used
in checks like a body (``C.distance("arm/hand", "target")``) and observed by name (``obs=[..., "target"]``).

Observation groups (opt-in): ``Task(obs={"policy": ["gripper", "action"], "critic": ["gripper", "cube", "action"]})``
makes the environment return a dict with one (W, D) array per group, so a critic or teacher can see more than the
policy does. A "policy" group is required; a list (as above) keeps the one flat observation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alloy.envs.actions import ActionSpec
from alloy.envs.checks import Check
from alloy.envs.errors import EnvError
from alloy.envs.layout import Layout
from alloy.envs.randomize import Jitter
from alloy.envs.state import BatchState

ObsEntry = str | Check | tuple[str, Callable[[BatchState], Any]]
ObsGroups = Mapping[str, Sequence[ObsEntry] | None]
POLICY_GROUP = "policy"
RESERVED_GROUPS = ("state",)              # the key of the flat observation beside camera images
INVALID_MODES = ("truncate", "terminate")


@dataclass
class Task:
    """Observations, actions, reward terms and episode ends for an environment. Example: see the module docstring."""

    obs: Sequence[ObsEntry] | ObsGroups | None = None   # a list: one flat observation; a dict: one per group
    actions: Any = None                   # ActionSpec, a list of joint names, or a spec with make_writer (rl.md)
    reward: Mapping[str, tuple[float, Check]] | Sequence[tuple[float, Check]] = ()
    success: Check | None = None
    failure: Check | None = None
    max_steps: int = 500
    cameras: Mapping[str, Any] = field(default_factory=dict)
    end_on_success: bool = True
    on_invalid: str = "truncate"          # a world the engine froze (crashed): "truncate" or "terminate" its episode
    invalid_penalty: float = 0.0          # added to the reward of a world on the step it becomes invalid
    randomize: Sequence[Jitter] = ()      # random start changes at every reset (alloy.envs.randomize)
    goals: Mapping[str, tuple[Sequence[float], Sequence[float]]] = field(default_factory=dict)  # name -> (lo, hi)
    settle: float = 0.0                   # s run once at build; the settled state is then the reset start
    on_build: Callable[[Any], None] | None = None   # fn(sim) once after the build, before settle (e.g. rest_on_floor)
    layers: Sequence[Any] = ()            # opt-in stages on actions and observations (delay, noise, history: rl.md)
    name: str = "task"

    def validate(self) -> None:
        """Raise ``EnvError`` on a bad field. Example: ``Task(max_steps=0).validate()`` raises."""
        if int(self.max_steps) < 1:
            raise EnvError(f"max_steps is {self.max_steps}", "pass max_steps >= 1 (steps per episode)",
                           field="max_steps")
        if self.on_invalid not in INVALID_MODES:
            raise EnvError(f"on_invalid is {self.on_invalid!r}", f"use one of {INVALID_MODES}", field="on_invalid")
        if not isinstance(self.cameras, Mapping) or not all(isinstance(k, str) for k in self.cameras):
            raise EnvError(f"cameras is a {type(self.cameras).__name__}, not a dict of names to cameras",
                           "pass Task(cameras={\"front\": cam}): the name is the observation key", field="cameras")
        self._validate_groups()
        for layer in self.layers:
            if not hasattr(layer, "attach"):
                raise EnvError(f"layer {layer!r} has no attach(env) method", "pass layer objects such as "
                               "alloy.envs.realism.ObsNoise(...) in Task(layers=[...])", field="layers")
        for name, (_w, check) in self.reward_terms():
            if not callable(check):
                raise EnvError(f"reward term {name!r} is not a check", "pass (weight, alloy.envs.checks.<check>)",
                               field="reward")

    def obs_groups(self) -> dict[str, Sequence[ObsEntry] | None]:
        """``{group: entries}`` when ``obs`` is a dict of groups, else ``{}`` (one flat observation).

        Example: ``Task(obs={"policy": ["arm"], "critic": ["arm", "cube"]}).obs_groups()["critic"]``."""
        return dict(self.obs) if isinstance(self.obs, Mapping) else {}

    def _validate_groups(self) -> None:
        if isinstance(self.obs, str):
            raise EnvError(f"obs is the string {self.obs!r}", "pass a list of entries, e.g. obs=['cube']", field="obs")
        groups = self.obs_groups()
        if not isinstance(self.obs, Mapping):
            return
        if POLICY_GROUP not in groups:
            raise EnvError(f"observation groups {sorted(map(str, groups))} have no 'policy' group",
                           "name the group the policy acts on 'policy', e.g. obs={'policy': [...], 'critic': [...]}",
                           field="obs")
        taken = {*RESERVED_GROUPS, *self.cameras}
        for name, entries in groups.items():
            if not isinstance(name, str) or name in taken:
                raise EnvError(f"observation group name {name!r} is not a string, is 'state', or is a camera's name",
                               "rename the group (camera images keep their camera's name in the observation)",
                               field="obs")
            if isinstance(entries, str):
                raise EnvError(f"observation group {name!r} is the string {entries!r}",
                               f"pass a list of entries, e.g. {{'{name}': ['{entries}']}}", field="obs")

    def action_spec(self) -> ActionSpec:
        """The actions as an ``ActionSpec`` (a list of joint names means position targets in [-1, 1])."""
        if isinstance(self.actions, ActionSpec) or hasattr(self.actions, "make_writer"):
            return self.actions
        return ActionSpec(joints=None if self.actions is None else list(self.actions))

    def reward_terms(self) -> list[tuple[str, tuple[float, Check]]]:
        """``[(name, (weight, check)), ...]``. Example: ``task.reward_terms()[0][0] == "reach"``."""
        if isinstance(self.reward, Mapping):
            return [(str(k), (float(w), c)) for k, (w, c) in self.reward.items()]
        return [(getattr(c, "name", f"term{k}"), (float(w), c)) for k, (w, c) in enumerate(self.reward)]


class Observer:
    """Builds the (W, D) float32 state observation from a ``BatchState``, following ``Task.obs``."""

    def __init__(self, entries: Sequence[ObsEntry] | None, layout: Layout, goals: Sequence[str] = ()) -> None:
        self.layout, self.goals = layout, set(goals)
        self.terms: list[tuple[str, Callable[[BatchState], Any]]] = []
        for e in (default_obs(layout) if entries is None else entries):
            self.terms += self._expand(e)
        self.slices: list[tuple[str, int, int]] = []

    def _expand(self, e: ObsEntry) -> list[tuple[str, Callable[[BatchState], Any]]]:
        if isinstance(e, Check):
            return [(e.name, e)]
        if isinstance(e, tuple):
            label, fn = e
            return [(str(label), fn)]
        if not isinstance(e, str):
            raise EnvError(f"observation entry {e!r} is not a name, check or (label, fn)",
                           "pass names like 'cube' or 'gripper/lift', a Check, or ('label', fn)", field="obs")
        return [(k, _key(k)) for k in self._keys(e)]

    def _keys(self, e: str) -> list[str]:
        lay = self.layout
        if e == "action":
            return ["action"]
        if e in self.goals or (e.endswith(".pos") and e[:-4] in self.goals):
            return [e if e.endswith(".pos") else f"{e}.pos"]
        if e in lay.robots:
            return [f"{j}.{f}" for j in lay.robots[e] for f in ("q", "qd")]
        if e in lay.bodies:
            return [f"{e}.{f}" for f in ("pos", "quat_wxyz", "vel", "angvel")]
        if e in lay.joints:
            return [f"{e}.q", f"{e}.qd"]
        if e in lay.particles:
            return [f"{e}.center"]
        name = e.rpartition(".")[0]
        if name and (name in lay.bodies or name in lay.joints or name in lay.particles):
            return [e]
        raise lay.unknown(e, "observation (body, robot, joint, particle object or state key)")

    def __call__(self, s: BatchState) -> Any:
        """(W, D) float32 observation. Example: ``obs = observer(env.state)``."""
        xp, w = s.xp, s.worlds
        parts = [xp.float(fn(s)).reshape(w, -1) for _label, fn in self.terms]
        out = xp.cat(parts, -1) if parts else xp.zeros((w, 0))
        if not self.slices:
            self.slices = _slices(self.terms, parts)
        return out

    def names(self) -> list[str]:
        """One label per observation column (call after the first observation). Example: ``observer.names()[:3]``."""
        out = []
        for label, a, b in self.slices:
            out += [label] if b - a == 1 else [f"{label}[{k}]" for k in range(b - a)]
        return out


def default_obs(layout: Layout) -> list[str]:
    """Every robot (joints), every free object (pose, velocity) and every particle object (centre)."""
    return list(layout.robots) + list(layout.objects) + list(layout.particles)


def _key(key: str) -> Callable[[BatchState], Any]:
    return lambda s: s[key]


def _slices(terms: list[tuple[str, Any]], parts: list[Any]) -> list[tuple[str, int, int]]:
    out, k = [], 0
    for (label, _fn), p in zip(terms, parts):
        out.append((label, k, k + int(p.shape[1])))
        k += int(p.shape[1])
    return out


class Rewarder:
    """Sums ``weight * check(state)`` over the task's reward terms; also returns each term's value."""

    def __init__(self, task: Task) -> None:
        self.terms = task.reward_terms()

    def __call__(self, s: BatchState) -> tuple[Any, dict[str, Any]]:
        """``(reward (W,), {term: value (W,)})``. Example: ``r, terms = rewarder(env.state)``."""
        xp = s.xp
        total = xp.zeros(s.worlds)
        values = {}
        for name, (weight, check) in self.terms:
            v = xp.float(check(s)).reshape(s.worlds)
            values[name] = v
            total = total + weight * v
        return total, values


__all__ = ["INVALID_MODES", "POLICY_GROUP", "Observer", "Rewarder", "Task", "default_obs"]
