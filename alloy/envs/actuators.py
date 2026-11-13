# SPDX-License-Identifier: Apache-2.0
"""Actuator models for RL actions: the policy picks a joint position target, a model between the policy and the joint
turns it into a torque (PD law, effort limit, optional DC-motor torque-speed curve, optional delay), and that torque
is what the joint gets.

How it runs. The torque is recomputed at EVERY physics step (about 1.25 ms at a 10 ms control period) from the joint's
live angle and speed, by a force plug-in (``ActuatorModel``) that writes ``control.joint_f`` before each solver step.
The target it follows changes once per control period (when the policy acts). So a stiff, damped PD is stable at the
physics step, not only at the control period. Delay is in whole control periods.

The robot must not have its own servo on those joints, or the file servo and the model both act (a shipped Go2's
``<general>`` servos with kp 50 moved a joint only 5 N m / kp under an extra effort). ``motorized(...)`` rewrites a
robot file's servo actuators into plain ``<motor>`` actuators (zero command, so they do nothing); the writer refuses a
joint that still has a servo.

Example::

    import alloy
    from alloy.envs import Task, VectorEnv
    from alloy.envs.actuators import ActuatorSpec, motorized

    spec = ActuatorSpec(["arm/shoulder"], model="dc_motor", kp=60.0, kd=2.0, effort_limit=12.0,
                        saturation_effort=14.0, velocity_limit=8.0, delay_steps=(0, 2), mode="normalized")
    scene = alloy.Scene(worlds=64, dt=0.01)
    scene.add_robot("arm", motorized("arm.xml"))                  # a path in, a servo-free path out
    sim = alloy.build(scene, plugins=[spec.plugin()])            # the model steps inside the simulation
    env = VectorEnv(sim, Task(actions=spec, max_steps=200))
    obs, _ = env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(env.action_space.sample() * 0)
    print(env.writer.torque.shape)                                # (64, 1): the torque the joint got

Not modelled: learned actuator networks, joint friction or backlash, motor heating, supply voltage, torque ripple.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy.envs.errors import EnvError
from alloy.envs.layout import UNBOUNDED, Layout
from alloy.envs.mjcf import cache_dir
from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.precision.health_work import register_work

LABEL = "actuators"                    # the plug-in's label in ``sim.plugins``
MODELS = ("pd", "dc_motor")
MODES = ("normalized", "absolute", "delta")
NO_LIMIT = 1.0e30
SERVO_TAGS = ("position", "velocity", "intvelocity", "general", "damper")


# --- device side: the torque law, once per physics step ------------------------------------------------------------

@wp.kernel
def torque_step(joint_q: wp.array(dtype=float), joint_qd: wp.array(dtype=float), q_index: wp.array2d(dtype=wp.int32),
                dof_index: wp.array2d(dtype=wp.int32), target: wp.array2d(dtype=float), kp: wp.array(dtype=float),
                kd: wp.array(dtype=float), effort: wp.array(dtype=float), sat: wp.array(dtype=float),
                vmax: wp.array(dtype=float), dc_motor: int, joint_f: wp.array(dtype=float),
                torque: wp.array2d(dtype=float), q_before: wp.array2d(dtype=float)):
    """tau = kp (target - q) - kd qd, kept inside the effort limit and (DC motor) the torque-speed envelope.

    Example: launched by ``ActuatorModel`` before every solver step."""
    w, i = wp.tid()
    q = joint_q[q_index[w, i]]
    v = joint_qd[dof_index[w, i]]
    tau = kp[i] * (target[w, i] - q) - kd[i] * v
    high = effort[i]
    low = -effort[i]
    if dc_motor != 0:
        high = wp.min(high, wp.max(sat[i] * (1.0 - v / vmax[i]), 0.0))
        low = wp.max(low, wp.min(-sat[i] * (1.0 + v / vmax[i]), 0.0))
    tau = wp.clamp(tau, low, high)
    joint_f[dof_index[w, i]] = tau
    torque[w, i] = tau
    q_before[w, i] = q


@wp.kernel
def book_work(joint_q: wp.array(dtype=float), q_index: wp.array2d(dtype=wp.int32), torque: wp.array2d(dtype=float),
              q_before: wp.array2d(dtype=float), work: wp.array(dtype=wp.float64)):
    """Per world, the work the torques did over the step (tau dq, joints in order): external work for health.

    Example: launched by ``ActuatorModel`` after every solver step."""
    w = wp.tid()
    total = wp.float64(0.0)
    for i in range(torque.shape[1]):
        total += wp.float64(torque[w, i] * (joint_q[q_index[w, i]] - q_before[w, i]))
    work[w] = work[w] + total


@dataclass
class ModelConfig:
    """Parameters of ``ActuatorModel`` (per joint lists). Example: ``ModelConfig(joints=["arm/j"], kp=[50.0])``."""

    joints: list[str] = field(default_factory=list)
    rows: list[list[int]] = field(default_factory=list)      # (worlds, joints): global Newton joint index
    model: str = "pd"
    kp: list[float] = field(default_factory=list)
    kd: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)
    saturation: list[float] = field(default_factory=list)
    velocity: list[float] = field(default_factory=list)


class ActuatorModel(ForcePlugin):
    """Writes each joint's model torque into ``control.joint_f`` before every solver step.

    Build it with ``ActuatorSpec.plugin()``. Example: ``alloy.build(scene, plugins=[spec.plugin()])``."""

    name = "actuator_model"
    summary = "joint torque from a position target: PD, effort limit, DC-motor torque-speed curve"
    Config = ModelConfig

    def setup(self, ctx: Any) -> None:
        c, m = self.config, ctx.model
        if not c.joints:
            raise PluginError("actuator_model: no joints given")
        rows = np.asarray(c.rows, np.int64)                                                       # (W, n)
        if rows.shape != (ctx.world_count, len(c.joints)):
            raise PluginError(f"actuator_model: rows {rows.shape} do not match (worlds, joints); use spec.plugin()")
        jtype = m.joint_type.numpy()
        if any(int(jtype[r]) not in (0, 1) for r in rows[0]):
            raise PluginError("actuator_model: only hinge and slide joints (one coordinate) take an actuator model")
        shape = (ctx.world_count, len(c.joints))
        self.q_index = ctx.array(m.joint_q_start.numpy()[rows].astype(np.int32), wp.int32)
        self.dof_index = ctx.array(m.joint_qd_start.numpy()[rows].astype(np.int32), wp.int32)
        for key in ("kp", "kd", "effort", "saturation", "velocity"):
            setattr(self, f"_{key}", ctx.array(np.asarray(getattr(c, key), np.float32)))
        self.torque = self.add_state("torque", ctx.zeros(shape))
        self.q_before = self.add_state("q_before", ctx.zeros(shape))
        self.target = self.add_input("target", ctx.zeros(shape))
        self.work = wp.zeros(ctx.world_count, dtype=wp.float64, device=ctx.device)      # J since health last booked
        register_work(m, self.work)             # the torque's work is external to the energy check

    def attach(self, stepper: Any) -> None:
        """Run the torque law in front of every solver step of ``stepper``. Example: the plug-in host calls it."""
        inner = stepper.step

        def step(state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, *args: Any,
                 **kwargs: Any) -> Any:
            self._write_torque(state_in, control)
            return inner(state_in, state_out, control, contacts, dt, *args, **kwargs)

        stepper.step = step

    def update(self, state_in: Any, state_out: Any, dt: float) -> None:
        """Book the work of the torques over the step as external work. Example: the plug-in host calls it."""
        wp.launch(book_work, dim=self.torque.shape[0], device=self.ctx.device, outputs=[self.work],
                  inputs=[state_out.joint_q, self.q_index, self.torque, self.q_before])

    def _write_torque(self, state: Any, control: Any) -> None:
        if getattr(control, "joint_f", None) is None:
            raise PluginError("actuator_model: the control has no joint_f array to write torques into")
        wp.launch(torque_step, dim=self.torque.shape, device=self.ctx.device,
                  outputs=[control.joint_f, self.torque, self.q_before],
                  inputs=[state.joint_q, state.joint_qd, self.q_index, self.dof_index, self.target, self._kp,
                          self._kd, self._effort, self._saturation, self._velocity,
                          int(self.config.model == "dc_motor")])


# --- the spec --------------------------------------------------------------------------------------------------------

@dataclass
class ActuatorSpec:
    """Position-target actions through an actuator model. Gains may be one number or one per joint.

    ``model="pd"``: tau = kp (target - q) - kd qd, clipped to +-``effort_limit``. ``model="dc_motor"``: also the
    available torque falls linearly from ``saturation_effort`` (stall) to zero at ``velocity_limit`` (no-load speed).
    ``delay_steps``: the target reaches the joint that many control periods late (an int, or ``(lo, hi)``: one
    whole-number delay per world, drawn again at each reset). ``mode``, ``low``, ``high``, ``delta`` as in
    ``ActionSpec`` (target limits default to the joint limits).

    Example: ``ActuatorSpec(["arm/elbow"], kp=40.0, kd=1.5, effort_limit=20.0, delay_steps=1)``."""

    joints: Sequence[str]
    model: str = "pd"
    kp: float | Sequence[float] = 50.0
    kd: float | Sequence[float] = 1.0
    effort_limit: float | Sequence[float] | None = None
    saturation_effort: float | Sequence[float] | None = None
    velocity_limit: float | Sequence[float] | None = None
    delay_steps: int | tuple[int, int] = 0
    mode: str = "normalized"
    low: Sequence[float] | None = None
    high: Sequence[float] | None = None
    delta: float | Sequence[float] = 0.02

    @property
    def delay_range(self) -> tuple[int, int]:
        """(lowest, highest) delay in control periods. Example: ``ActuatorSpec(["a/j"], delay_steps=2).delay_range``."""
        d = self.delay_steps
        return (int(d[0]), int(d[1])) if isinstance(d, tuple) else (int(d), int(d))

    def per_joint(self, name: str, value: Any, default: float | None = None) -> np.ndarray:
        """``value`` as one float32 per joint. Example: ``spec.per_joint("kp", spec.kp)``."""
        v = default if value is None else value
        arr = np.broadcast_to(np.asarray(v, np.float64), (len(self.joints),)) if v is not None else None
        if arr is None or not np.all(np.isfinite(arr)):
            raise EnvError(f"{name} is missing or not finite", f"give {name} as one number or {len(self.joints)} "
                           "numbers", field=name)
        return arr.astype(np.float32)

    def validate(self) -> None:
        """Raise ``EnvError`` on a bad model, mode, delay or gain. Example: ``ActuatorSpec(["a/j"], kp=-1).validate()``
        raises."""
        if not self.joints:
            raise EnvError("no joints named", "pass ActuatorSpec(['robot/joint', ...])", field="joints")
        if self.model not in MODELS:
            raise EnvError(f"unknown actuator model {self.model!r}", f"use one of {MODELS}", field="model")
        if self.mode not in MODES:
            raise EnvError(f"unknown action mode {self.mode!r}", f"use one of {MODES}", field="mode")
        lo, hi = self.delay_range
        if lo < 0 or hi < lo:
            raise EnvError(f"delay_steps {self.delay_steps!r}", "use a whole number >= 0, or (lo, hi) with lo <= hi",
                           field="delay_steps")
        self._arrays()

    def _arrays(self) -> dict[str, np.ndarray]:
        out = {"kp": self.per_joint("kp", self.kp), "kd": self.per_joint("kd", self.kd),
               "effort": self.per_joint("effort_limit", self.effort_limit, NO_LIMIT)}
        if min(out["kp"].min(), out["kd"].min()) < 0 or out["effort"].min() <= 0:
            raise EnvError("kp and kd must be >= 0 and effort_limit > 0", "give positive gains and limits", field="kp")
        dc = self.model == "dc_motor"
        if dc and self.velocity_limit is None:
            raise EnvError("model='dc_motor' needs velocity_limit (the no-load speed, rad/s or m/s)",
                           "pass velocity_limit=<speed>", field="velocity_limit")
        sat = self.saturation_effort if self.saturation_effort is not None else self.effort_limit
        if dc and sat is None:
            raise EnvError("model='dc_motor' needs saturation_effort (the stall torque)",
                           "pass saturation_effort=<N m> or effort_limit", field="saturation_effort")
        out["saturation"] = self.per_joint("saturation_effort", sat, NO_LIMIT)
        out["velocity"] = self.per_joint("velocity_limit", self.velocity_limit, 1.0)
        if dc and out["velocity"].min() <= 0:
            raise EnvError("velocity_limit must be > 0", "give the no-load speed", field="velocity_limit")
        return out

    def plugin(self) -> Any:
        """What ``alloy.build(scene, plugins=[...])`` takes: a function of the built scene that makes the force
        plug-in. Example: ``sim = alloy.build(scene, plugins=[spec.plugin()])``."""
        self.validate()
        a = {k: v.tolist() for k, v in self._arrays().items()}

        def make(built: Any) -> ActuatorModel:
            from alloy.api.names import NameIndex  # noqa: PLC0415 - the API imports the envs lazily too

            idx = NameIndex.from_built(built)
            rows = np.stack([idx.per_world(idx.joint(j), "joint") for j in self.joints], axis=1)
            cfg = ModelConfig(joints=list(self.joints), rows=rows.tolist(), model=self.model, **a)
            return ActuatorModel(cfg, label=LABEL)

        return make

    def make_writer(self, layout: Layout, engine: Any, xp: Any) -> ActuatorWriter:
        """The action writer the environment uses. Example: internal (``VectorEnv`` calls it)."""
        return ActuatorWriter(self, layout, engine, xp)


# --- the writer ------------------------------------------------------------------------------------------------------

class ActuatorWriter:
    """Turns (W, n) actions into targets (delayed per world) for the actuator model plug-in. Reads ``.torque``.

    Example: ``env.writer.torque`` (W, n) after ``env.step(...)``; ``VectorEnv`` builds the writer from the spec."""

    kind = "effort"

    def __init__(self, spec: ActuatorSpec, layout: Layout, engine: Any, xp: Any) -> None:
        spec.validate()
        self.spec, self.layout, self.engine, self.xp = spec, layout, engine, xp
        self.joints = list(spec.joints)
        refuse_file_servos(layout, engine, self.joints)
        self.plugin = _find_plugin(engine, spec)
        self._target = self.plugin.inputs()["target"]
        w = layout.worlds
        rows = np.concatenate([_one_coordinate(layout, j) for j in self.joints], axis=1)           # (W, n)
        lo, hi = self._limits()
        self.low, self.high, self.low_np, self.high_np = xp.asarray(lo), xp.asarray(hi), lo, hi
        self.delta = xp.asarray(np.broadcast_to(np.asarray(spec.delta, np.float32), (self.size,)).copy())
        self._rows = xp.asarray(rows, "int64")
        self._start = xp.copy(xp.view(engine.state.joint_q)[self._rows])                           # (W, n)
        self.command = xp.copy(self._start)
        self._lo_delay, self._hi_delay = spec.delay_range
        self._hist = xp.stack([self._start] * (self._hi_delay + 1), axis=0)                        # newest first
        self._worlds = xp.asarray(np.arange(w), "int64")
        self._delays = self._draw_delays()
        self._world: dict[str, Any] = {}
        self._write(self._start)

    @property
    def size(self) -> int:
        """Action columns per world. Example: ``writer.size``."""
        return len(self.joints)

    @property
    def torque(self) -> Any:
        """The torque last applied at each joint, (W, n) on the env backend. Example: ``env.writer.torque``."""
        return self.xp.view(self.plugin.torque)

    def column_names(self) -> list[str]:
        """One name per action column. Example: ``writer.column_names()``."""
        return list(self.joints)

    def _limits(self) -> tuple[np.ndarray, np.ndarray]:
        s, n = self.spec, self.size
        if s.low is not None and s.high is not None:
            lo, hi = np.asarray(s.low, np.float32).reshape(-1), np.asarray(s.high, np.float32).reshape(-1)
            if lo.shape != (n,) or hi.shape != (n,) or np.any(hi <= lo):
                raise EnvError(f"low/high need {n} values each with high > low", f"pass {n} numbers per bound "
                               f"(columns: {self.joints})", field="low")
            return lo, hi
        lo = np.concatenate([np.atleast_1d(self.layout.joint(j).lower) for j in self.joints]).astype(np.float64)
        hi = np.concatenate([np.atleast_1d(self.layout.joint(j).upper) for j in self.joints]).astype(np.float64)
        if np.any(~np.isfinite(lo) | ~np.isfinite(hi) | (np.abs(lo) >= UNBOUNDED) | (np.abs(hi) >= UNBOUNDED)):
            raise EnvError(f"joints without limits: {self.joints}", "give the joints a range in the robot file, or "
                           "pass ActuatorSpec(low=[...], high=[...])", field="joints")
        return lo.astype(np.float32), hi.astype(np.float32)

    def _draw_delays(self) -> Any:
        xp, w = self.xp, self.layout.worlds
        if self._lo_delay == self._hi_delay:
            return xp.asarray(np.full(w, self._lo_delay), "int64")
        u = xp.to_numpy(xp.uniform((w,), 0.0, 1.0))
        span = self._hi_delay - self._lo_delay + 1
        return xp.asarray(self._lo_delay + np.minimum((u * span).astype(np.int64), span - 1), "int64")

    # --- stepping --------------------------------------------------------------------------------------------------

    def apply(self, actions: Any) -> Any:
        """Set the position targets from (W, n) actions; returns the commanded targets (before the delay).

        Example: ``targets = writer.apply(np.zeros((env.num_envs, writer.size)))``."""
        xp, s = self.xp, self.spec
        a = self._check(actions)
        if s.mode == "absolute":
            cmd = xp.clip(a, self.low, self.high)
        else:
            a = xp.clip(a, -1.0, 1.0)
            if s.mode == "normalized":
                cmd = self.low + (a + 1.0) * 0.5 * (self.high - self.low)
            else:
                cmd = xp.clip(self.command + self.delta * a, self.low, self.high)
        self.command = cmd
        self._hist = xp.cat([cmd[None], self._hist[:-1]], axis=0)
        self._write(self._hist[self._delays, self._worlds])
        return cmd

    def _write(self, values: Any) -> None:
        self.xp.update(self._target, lambda t: t.__setitem__(slice(None), values))

    def _check(self, actions: Any) -> Any:
        shape = tuple(getattr(actions, "shape", np.shape(actions)))
        if shape != (self.layout.worlds, self.size):
            raise EnvError(f"actions have shape {shape}", f"pass an array of shape ({self.layout.worlds}, "
                           f"{self.size}): one row per world, columns {self.joints}", field="actions")
        return self.xp.asarray(actions, "float32")

    def reset(self, mask: Any) -> None:
        """Hold the start pose again in the worlds of the bool (W,) ``mask``. Example: ``writer.reset(done)``."""
        xp, m = self.xp, mask[:, None]
        self.command = xp.where(m, self._start, self.command)
        self._hist = xp.where(mask[None, :, None], self._start[None], self._hist)
        fresh = self._draw_delays()
        self._delays = xp.where(mask, fresh, self._delays)
        self._write(xp.where(m, self._start, self._hist[self._delays, self._worlds]))

    def hold(self, mask: Any) -> None:
        """After a reset's jitters: command each masked world's joints where they now are (a jittered start), so the
        model holds that pose instead of pulling back to the build pose. Example: internal (VectorEnv.reset_worlds)."""
        xp, m = self.xp, mask[:, None]
        q = xp.copy(xp.view(self.engine.state.joint_q)[self._rows])
        self.command = xp.where(m, q, self.command)
        self._hist = xp.where(mask[None, :, None], q[None], self._hist)
        self._write(self._hist[self._delays, self._worlds])


# --- helpers ---------------------------------------------------------------------------------------------------------

def _one_coordinate(layout: Layout, joint: str) -> np.ndarray:
    rows = layout.joint(joint).q
    if rows.shape[1] != 1:
        raise EnvError(f"joint {joint!r} has {rows.shape[1]} coordinates", "an actuator model drives hinge and slide "
                       "joints (one coordinate); name those", field="joints")
    return rows


def _find_plugin(engine: Any, spec: ActuatorSpec) -> ActuatorModel:
    host = getattr(engine, "plugins", None)
    plugin = getattr(host, "plugins", {}).get(LABEL)
    if plugin is None:
        raise EnvError("the simulation was built without the actuator model", "build it first: sim = alloy.build("
                       "scene, plugins=[spec.plugin()]), then VectorEnv(sim, Task(actions=spec))", field="actions")
    if list(plugin.config.joints) != list(spec.joints):
        raise EnvError(f"the plug-in drives {plugin.config.joints}, the spec names {list(spec.joints)}",
                       "make the plug-in from the same spec: spec.plugin()", field="joints")
    return plugin


def refuse_file_servos(layout: Layout, engine: Any, joints: Sequence[str]) -> None:
    """Raise ``EnvError`` if a robot file's servo still drives one of ``joints`` (its torque would add to the model's).

    Example: ``refuse_file_servos(layout, sim.engine, ["arm/j"])``."""
    bias = getattr(getattr(engine.model, "mujoco", None), "actuator_biasprm", None)
    bias = bias.numpy() if bias is not None else None
    held = []
    for j in joints:
        servo = layout.joint(j).actuated
        for a in layout.actuators.values():
            if a.joint == j and bias is not None and np.any(bias[int(a.rows[0]), 1:3] != 0.0):
                servo = True
        if servo:
            held.append(j)
    if held:
        raise EnvError(f"joints {held} have a servo in the robot file; its torque would add to the actuator model's",
                       "load the robot through motorized(...): scene.add_robot('arm', mjcf=motorized(xml_or_path)) "
                       "(it turns servos into zero-command motors)", field="joints")


def motorized(mjcf: str | Path) -> str | Path:
    """The MJCF with every joint servo (``<position>``, ``<velocity>``, ``<intvelocity>``, ``<general>``, ``<damper>``)
    rewritten as a plain ``<motor>`` on the same joint (same name, order and gear), so a policy or an actuator model
    owns the joint. A path gives a cached file path (included files too); XML text gives XML text.

    Example: ``scene.add_robot("arm", mjcf=motorized("<mujoco>...</mujoco>"))``, or ``add_robot("a", motorized(path))``.
    """
    if isinstance(mjcf, str) and mjcf.lstrip().startswith("<"):
        root = ET.fromstring(mjcf)
        _to_motors(root)
        return ET.tostring(root, encoding="unicode")
    src = Path(mjcf).resolve()
    if not src.is_file():
        raise EnvError(f"robot file {src.name!r} not found", "pass an MJCF path or the XML text", field="mjcf")
    return _motorized_file(src, None)


def _to_motors(root: ET.Element) -> int:
    count = 0
    for block in root.iter("actuator"):
        for k, el in enumerate(list(block)):
            if el.tag not in SERVO_TAGS or not el.get("joint"):
                continue
            motor = ET.Element("motor", {a: el.get(a) for a in ("name", "joint", "gear", "group") if el.get(a)})
            block.remove(el)
            block.insert(k, motor)
            count += 1
    return count


def _motorized_file(src: Path, meshdir: str | None) -> Path:
    text = src.read_text(encoding="utf-8")
    out = cache_dir() / f"{src.stem}-motor-{hashlib.sha256(text.encode()).hexdigest()[:16]}.xml"
    if out.is_file():
        return out
    root = ET.fromstring(text)
    compiler = root.find("compiler")
    dirs = dict(compiler.attrib) if compiler is not None else {}
    dirs.setdefault("meshdir", meshdir or "")
    for inc in root.iter("include"):
        inc.set("file", str(_motorized_file((src.parent / inc.get("file", "")).resolve(), dirs["meshdir"] or None)))
    _to_motors(root)
    _absolute(root, src.parent, dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    ET.ElementTree(root).write(tmp, encoding="unicode")
    tmp.replace(out)
    return out


def _absolute(root: ET.Element, base: Path, dirs: dict[str, str]) -> None:
    sub = {"mesh": dirs.get("meshdir"), "texture": dirs.get("texturedir"), "hfield": dirs.get("hfielddir")}
    for el in root.iter():
        f = el.get("file")
        if not f or el.tag == "include" or Path(f).is_absolute():
            continue
        folder = base / (sub.get(el.tag) or dirs.get("assetdir") or "")
        el.set("file", str((folder / f).resolve()))


__all__ = ["ActuatorModel", "ActuatorSpec", "ActuatorWriter", "ModelConfig", "motorized", "refuse_file_servos"]
