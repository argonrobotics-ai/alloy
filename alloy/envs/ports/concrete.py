# SPDX-License-Identifier: Apache-2.0
"""The concrete-finishing ports (unofficial ports of Gym-Anything concrete-finishing-tasks-v1) as environments: five
tasks on the surface-flow paste, each on its port's own run and judged by the port's own checks.

| env | Gym-Anything task | what the policy moves |
|---|---|---|
| ``ConcretePadEdge-v0`` | pad-edge | the edger on the tool carriage |
| ``ConcretePatchFlush-v0`` | patch-flush | the strike-off trowel and the float on the carriage |
| ``ConcreteToolJoints-v0`` | tool-joints | the jointer and the trowel on the carriage |
| ``ConcreteToolSealant-v0`` | tool-sealant | the sealant tools on the carriage; the rag's release |
| ``ConcretePatchDrywall-v0`` | patch-drywall | the two knives (kinematic): pose and velocity each |

The carriage (``alloy.tasks.concrete_tools.ToolCarriage``; no robot) is a privileged fixture, not Gym-Anything's
actor: their task interface is robot-only (a mobile FR3 holds the tools) and a carriage exists only in their probes,
marked ASSISTED (``probes/common.py:1-4``); the drywall knives are kinematic bodies a script poses. Each port's
verdict says what it judges of the task's criteria and how its scene differs (``judged``, ``not_judged``,
``scene_differs``, ``summary`` per world in ``info["verdict"]``; ``alloy.tasks.concrete_ports``): success is every
judged criterion and Alloy's own gates, not a pass of Gym-Anything's task. The carriage holds
each tool on a keyframed program. The action is, per tool, where it should be at the end of the coming period:
x, y, z, yaw, pitch, the normal force, the mode (0: press with the force, 1: hold the height) and the store's release
share; the expert's is the program's. A different action replaces that tool's program from then on with a ramp to it
over the period. Success is the port's verdict per world (``docs/guide/envs.md`` lists what it judges). Keyword
options of ``make`` set the port's task settings (``make("ConcretePadEdge-v0", window=(0.37, 0.37))``: a corner of
the pad, as its CPU proof; the default is the whole pad, hours on a CPU).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy.envs.errors import EnvError
from alloy.envs.ports.env import PortSpec
from alloy.envs.ports.movers import KinematicBody, Mover, PluginArray

FIELDS = 8                    # x, y, z, yaw, pitch, force, mode, release
LOW = np.array([-3.0, -3.0, -1.0, -np.pi, -np.pi / 2, 0.0, 0.0, 0.0], np.float32)
HIGH = np.array([3.0, 3.0, 2.0, np.pi, np.pi / 2, 500.0, 1.0, 1.0], np.float32)


def _at(table: np.ndarray, n: int, t: float) -> np.ndarray:
    """One program row's targets at time ``t`` (as the carriage's kernel: the segment's start key, then a linear
    ramp to the next): x, y, z, yaw, pitch, force, mode, release."""
    keys = table[:n]
    i = int(np.searchsorted(keys["t"], t, side="right")) - 1
    i = min(max(i, 0), n - 1)
    a = keys[i]
    pos, yaw, pitch, force = np.asarray(a["pos"], np.float64), float(a["yaw"]), float(a["pitch"]), float(a["force"])
    if i < n - 1:
        c = keys[i + 1]
        s = float(np.clip((t - float(a["t"])) / float(c["t"] - a["t"]), 0.0, 1.0))
        pos = pos + s * (np.asarray(c["pos"], np.float64) - pos)
        yaw, pitch = yaw + s * float(c["yaw"] - a["yaw"]), pitch + s * float(c["pitch"] - a["pitch"])
        force = force + s * float(c["force"] - a["force"])
    return np.array([*pos, yaw, pitch, force, float(a["mode"]), float(a["release"])], np.float32)


class CarriageCommand:
    """Every carriage tool's target at the coming period's end (see the module docstring); proprio is the tools'
    pose and velocity."""

    def bind(self, sim: Any) -> None:
        found = [p for p in sim.plugins.plugins.values() if type(p).__name__ == "ToolCarriage"]
        if len(found) != 1:
            raise EnvError(f"{len(found)} tool carriages in the port", "a concrete port has one")
        self.c = found[0]
        self.tools = self.c.tool.numpy()                                   # (W, T) body indices
        self.W, self.T = self.tools.shape
        self.size, self.proprio_size = FIELDS * self.T, 13 * self.T
        self.low, self.high = np.tile(LOW, self.T), np.tile(HIGH, self.T)
        self.dt = float(getattr(self.c.ctx, "control_dt", None) or sim.built.control_dt)
        self.overridden = False

    def _times(self) -> np.ndarray:
        return self.c.ctx.time.numpy().astype(np.float64).reshape(-1)[: self.W]

    def _targets(self, t: np.ndarray) -> np.ndarray:
        table, n = self.c.keys.numpy(), self.c.n.numpy()
        out = np.zeros((self.W, self.T, FIELDS), np.float32)
        for w in range(self.W):
            for k in range(self.T):
                r = w * self.T + k
                out[w, k] = _at(table[r], int(n[r]), float(t[w]))
        return out

    def read(self, sim: Any) -> np.ndarray:  # noqa: ARG002
        return self._targets(self._times() + self.dt).reshape(self.W, -1)

    def write(self, sim: Any, a: np.ndarray) -> None:
        a = np.asarray(a, np.float32).reshape(self.W, self.T, FIELDS)
        if not self.overridden and np.array_equal(a.reshape(self.W, -1), self.read(sim)):
            return                                                         # the program itself: untouched
        t = self._times()
        now = self._targets(t)
        table, n = self.c.keys.numpy().copy(), self.c.n.numpy().copy()
        for w in range(self.W):
            for k in range(self.T):
                r = w * self.T + k
                for j, (when, v) in enumerate(((t[w], now[w, k]), (t[w] + self.dt, a[w, k]))):
                    table[r, j] = (when, tuple(v[:3]), v[3], v[4], v[5], int(round(float(a[w, k, 6]))), a[w, k, 7])
                n[r] = 2
        self.c.keys.assign(table)                                          # in place: captured graphs keep it
        self.c.n.assign(n)
        self.overridden = True

    def proprio(self, sim: Any) -> np.ndarray:
        q, qd = sim.state.body_q.numpy()[self.tools], sim.state.body_qd.numpy()[self.tools]
        return np.concatenate([q, qd], axis=2).reshape(self.W, -1).astype(np.float32)


def _runner(module: str, settings: str):  # noqa: ANN202
    def run(worlds: int, device: str, seed: int, **task: Any) -> Any:  # noqa: ARG001 - worlds take seed + w
        """The port's run; keyword options set its task's settings (e.g. pad-edge's ``window=(0.37, 0.37)``)."""
        import importlib  # noqa: PLC0415

        mod = importlib.import_module(f"alloy.tasks.{module}")
        made = getattr(mod, settings)(**{k: tuple(v) if isinstance(v, list) else v for k, v in task.items()})
        return mod.run(made if task else None, worlds=worlds, device=device)

    return run


def _drywall_mover() -> Mover:
    from alloy.tasks import concrete_drywall as cd  # noqa: PLC0415

    return Mover(*(KinematicBody(knife.name) for knife in (cd.PUTTY, cd.JOINT)))


def _success(r: Any, w: int) -> bool:
    return bool(r.verdicts[w]["success"])


def _score(r: Any, w: int) -> float:
    v = r.verdicts[w]
    return 1.0 if v["success"] else float(v.get("score") or 0.0)


SPECS = [
    PortSpec("ConcretePadEdge-v0", "concrete_finishing/pad-edge", _runner("pad_edge", "PadEdge"),
             lambda: Mover(CarriageCommand()), _success, score=_score, batched=True,
             summary="edge a concrete pad along its forms with an edger"),
    PortSpec("ConcretePatchFlush-v0", "concrete_finishing/patch-flush", _runner("concrete_patch", "PatchFlush"),
             lambda: Mover(CarriageCommand()), _success, score=_score, batched=True,
             summary="fill a patch and strike it off flush, then float it"),
    PortSpec("ConcreteToolJoints-v0", "concrete_finishing/tool-joints", _runner("concrete_joints", "ToolJoints"),
             lambda: Mover(CarriageCommand()), _success, score=_score, batched=True,
             summary="tool the mortar joints of a block wall"),
    PortSpec("ConcreteToolSealant-v0", "concrete_finishing/tool-sealant", _runner("concrete_sealant", "ToolSealant"),
             lambda: Mover(CarriageCommand(), PluginArray("tool_release", 0.0, 1.0)), _success, score=_score,
             batched=True, summary="tool a sealant joint and wipe the excess"),
    PortSpec("ConcretePatchDrywall-v0", "concrete_finishing/patch-drywall", _runner("concrete_drywall", "PatchDrywall"),
             _drywall_mover, _success, score=_score, batched=True,
             summary="fill a drywall patch's anchor holes and knife it flush"),
]

__all__ = ["SPECS", "CarriageCommand"]
