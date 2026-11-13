# SPDX-License-Identifier: Apache-2.0
"""``alloy.envs.VectorEnv``: every world of an Alloy simulation as one Gymnasium vector environment.

Observations, rewards and actions are batched over worlds (W = ``num_envs``) and stay on the simulation device as
torch tensors (zero-copy views of the engine's arrays) when PyTorch is installed, else NumPy. Finished worlds reset
by mask, without touching the others. Example::

    import alloy.api as alloy
    from alloy.envs import VectorEnv, Task, checks as C
    env = VectorEnv(scene, Task(success=C.lifted("cube", by=0.08), max_steps=200), worlds=256)
    obs, info = env.reset(seed=0)                        # (256, D) float32
    for _ in range(1000):
        obs, reward, terminated, truncated, info = env.step(policy(obs))   # (256, A) actions in [-1, 1]
    env.reset(options={"reset_mask": some_worlds})       # reset only these worlds

Autoreset (``autoreset=``): "same_step" (default; a finished world is reset inside ``step`` and its last observation
is ``info["final_obs"]``, mask ``info["_final_obs"]``), "next_step" (Gymnasium's default: reset on the next call,
which ignores that world's action) or "disabled" (reset worlds yourself with ``reset(options={"reset_mask": m})``).
``info["episode"]`` holds return ``r``, length ``l`` and ``success`` of the episodes that just ended (``_episode``).
Observation groups (``Task(obs={"policy": [...], "critic": [...]})``) make ``reset`` and ``step`` return a dict with
one (W, D) array per group (``env.obs_groups`` names them); a list keeps the one flat array.
Scenes whose engine state cannot be reset one world at a time (``env.together``; liquids could not before SPH's
snapshot kept each world apart) run every world on one episode clock: episodes end only at ``max_steps`` (success and
failure are still reported) and all reset together.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alloy.coupling.world_clock_writes import refuse_world_clock
from alloy.envs import spaces
from alloy.envs.actions import ActionWriter
from alloy.envs.backend import pick_backend
from alloy.envs.episode import EpisodeStats, Goals, LiveArrays, StartState
from alloy.envs.errors import EnvError
from alloy.envs.health import WorldHealth
from alloy.envs.layout import Layout
from alloy.envs.randomize import Randomizer
from alloy.envs.state import BatchState, Rows
from alloy.envs.task import POLICY_GROUP, Observer, Rewarder, Task

log = logging.getLogger("alloy.envs")

try:  # a Gymnasium VectorEnv when Gymnasium is installed; the same API without it
    from gymnasium.vector import AutoresetMode as _AutoresetMode
    from gymnasium.vector import VectorEnv as _Base
except ImportError:  # pragma: no cover - exercised where gymnasium is missing
    _Base, _AutoresetMode = object, None

AUTORESET = ("same_step", "next_step", "disabled")


class VectorEnv(_Base):  # type: ignore[misc, valid-type]
    """A batched Gymnasium environment over the worlds of one Alloy simulation. Example: see the module docstring."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "autoreset_mode": "same_step"}

    def __init__(self, sim: Any, task: Task | None = None, *, worlds: int | None = None, device: str | None = None,
                 seed: int = 0, backend: str = "auto", autoreset: str = "same_step",
                 render_mode: str | None = None, profile: str = "standard") -> None:
        self.task = task if task is not None else Task()
        self.task.validate()
        if autoreset not in AUTORESET:
            raise EnvError(f"autoreset is {autoreset!r}", f"use one of {AUTORESET}", field="autoreset")
        self._owns_sim = not hasattr(sim, "engine")
        self.sim = _build(sim, worlds, device, seed, profile) if self._owns_sim else sim
        refuse_world_clock(self.sim, "VectorEnv")     # its own writes bypass Sim (world_clock_writes)
        if self.task.on_build is not None:
            self.task.on_build(self.sim)
            self.sim.engine._initial.save(self.sim.engine.state)   # its edits are the start every reset returns to
            self.sim._rebase()
        if self.task.settle > 0:                       # objects start at rest, not a few mm above it
            self.sim.settle(self.task.settle)
        self.num_envs, self.device = int(self.sim.worlds), str(self.sim.device)
        self.autoreset, self.render_mode = autoreset, render_mode
        self.metadata = {**type(self).metadata, "autoreset_mode": _mode(autoreset)}
        self.xp = pick_backend(backend, self.device)
        self.layout = Layout.from_sim(self.sim)
        items = getattr(getattr(self.sim, "scene", None), "items", {})
        self._rows = Rows(self.layout, self.xp, anchors=getattr(self.sim.engine, "_alloy_anchors", {}),
                          pours=[n for n, it in items.items() if it.get("pour")])
        engine = self.sim.engine
        self.writer = _make_writer(self.task.action_spec(), self.layout, engine, self.xp)
        self.layers = list(self.task.layers)           # () by default: then nothing below changes a value
        self.health = WorldHealth(engine, self.xp)
        self.goals = Goals(self.task.goals, self.num_envs, self.xp)
        groups = self.task.obs_groups()                # {} unless Task(obs={...}): then one observation per group
        self.obs_groups: tuple[str, ...] = tuple(groups)
        self.observers = {g: Observer(entries, self.layout, goals=list(self.task.goals))
                          for g, entries in groups.items()}
        self.observer = self.observers[POLICY_GROUP] if groups else Observer(
            self.task.obs, self.layout, goals=list(self.task.goals))
        self.rewarder = Rewarder(self.task)
        self.cameras = dict(self.task.cameras)
        self.randomizer = Randomizer(self.task.randomize, self.layout, engine, self.xp)
        self.xp.seed(seed)
        self.start = StartState(engine, self.layout, self.xp)
        self.stats = EpisodeStats(self.num_envs, self.writer.size, self.xp)
        self._pending = self.xp.zeros(self.num_envs, "bool")
        self._ended = self.xp.zeros(self.num_envs, "bool")            # autoreset="disabled": ended, not yet reset
        self._ended_flags = (self._ended, self._ended)                # (terminated, truncated) when each ended
        self.invalid_episodes = self.xp.zeros(self.num_envs, "int32")   # per world, kept across resets
        self.together = not per_world_reset(engine)
        if self.together:
            log.info("VectorEnv %s: the engine cannot reset one world alone here; every world's episode ends at "
                     "max_steps and all reset together", self.task.name)
        for layer in self.layers:
            layer.attach(self)
        self._make_spaces(seed)
        self.closed = False
        log.info("VectorEnv %s: %d worlds on %s (%s), obs %s, actions %d", self.task.name, self.num_envs,
                 self.device, self.xp.name, self.single_observation_space, self.writer.size)

    # --- spaces ----------------------------------------------------------------------------------------------------

    def _make_spaces(self, seed: int) -> None:
        obs = self._observe(self.state)
        self.single_observation_space = _obs_space(obs, set(self.cameras))
        self.observation_space = spaces.batch(self.single_observation_space, self.num_envs)
        mode = self.writer.spec.mode
        lo, hi = (self.writer.low_np, self.writer.high_np) if mode == "absolute" else (-1.0, 1.0)
        self.single_action_space = spaces.box(lo, hi, (self.writer.size,))
        self.action_space = spaces.batch(self.single_action_space, self.num_envs)
        self.action_space.seed(seed)

    # --- state -----------------------------------------------------------------------------------------------------

    @property
    def state(self) -> BatchState:
        """Every world's named state now (``env.state["cube.pos"]``, ``.start``, ``["steps"]``, ``["action"]``)."""
        st = self.stats
        extra = {"steps": st.steps, "t": self.xp.float(st.steps) * float(self.sim.dt), "action": st.action,
                 **self.goals.extra()}
        start = BatchState(self.start.view, self._rows, extra=extra)
        return BatchState(LiveArrays(self.sim.engine, self.xp), self._rows, extra=extra, start=start)

    def obs_names(self, group: str | None = None) -> list[str]:
        """A label per column of the state observation (of one group: ``env.obs_names("critic")``; default: the
        policy's). Example: ``env.obs_names()[:4]``."""
        if group is None:
            return self.observer.names()
        if group not in self.observers:
            raise EnvError(f"no observation group named {group!r}", f"this environment has {list(self.obs_groups)} "
                           "(groups come from Task(obs={'policy': [...], ...}))", field="group")
        return self.observers[group].names()

    def action_names(self) -> list[str]:
        """A label per action column (joint names). Example: ``env.action_names()``."""
        return self.writer.column_names()

    def _observe(self, s: BatchState, only: Any = None) -> Any:
        """The observation now. With layers, ``only`` (a (W,) bool mask) marks the worlds that were just reset: a
        layer starts those afresh and returns its last output for the others (no second noise draw or history
        push within one step)."""
        obs = self._read_obs(s)
        for layer in self.layers:
            obs = layer.on_obs(obs, only)
        return obs

    def _read_obs(self, s: BatchState) -> Any:
        if self.observers:
            out = {g: observer(s) for g, observer in self.observers.items()}
            if not self.cameras:
                return out
        else:
            vec = self.observer(s)
            if not self.cameras:
                return vec
            out = {"state": vec}
        model, state = self._camera_source(), self.sim.engine.state
        from alloy.camera import Camera  # noqa: PLC0415
        for name, cam in self.cameras.items():
            src = (model, state) if isinstance(cam, Camera) else (self.sim, None)   # sensors of the physics (tactile)
            if self.xp.name == "torch":                                             # read the Sim, not the picture
                out[name] = cam.render_torch(*src).clone()
            else:
                out[name] = cam.render(*src).numpy().copy()    # on the CPU .numpy() is the camera's buffer
        return out

    # --- Gymnasium API ---------------------------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset every world, or only ``options={"reset_mask": (W,) bool}``. Returns ``(obs, info)``."""
        if seed is not None:
            self.action_space.seed(int(seed))
            self.xp.seed(int(seed))
        mask = (options or {}).get("reset_mask")
        mask = self.xp.zeros(self.num_envs, "bool") == 0 if mask is None else self._mask(mask)
        self.reset_worlds(mask)
        return self._observe(self.state, only=mask), {}

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """One control period for every world: ``(obs, reward, terminated, truncated, info)``."""
        self._require_open()
        sent = actions                                 # what the policy chose; a layer may delay what is applied
        for layer in self.layers:
            actions = layer.on_action(actions)
        clipped = self.writer.apply(actions)
        self.sim.step(1)
        self.stats.act(self._policy_action(sent))
        s = self.state
        reward, terms = self.rewarder(s)
        terminated, truncated, success, invalid = self._ends(s)
        frozen = self.xp.any(invalid)                  # a frozen world earns only its penalty and shows zeros
        if frozen:
            reward, terms = _void(reward, invalid), {k: _void(v, invalid) for k, v in terms.items()}
            self.invalid_episodes = self.invalid_episodes + (invalid & ~self._pending & ~self._ended)   # once
        reward = reward + self.task.invalid_penalty * self.xp.float(invalid)
        fresh = None
        if self.autoreset == "disabled":
            reward, terminated, truncated, fresh = self._hold_ended(reward, terminated, truncated)
        self.stats.add(reward)
        obs = self._observe(s)
        if frozen:
            obs = _void(obs, invalid)
        info: dict[str, Any] = {"success": success, "invalid": invalid, "terms": terms, "targets": clipped}
        done = terminated | truncated
        if self.autoreset == "next_step" and self.xp.any(self._pending):
            obs, reward, terminated, truncated = self._finish_pending(obs, reward, terminated, truncated, info)
            done = terminated | truncated
        ended = done if fresh is None else fresh                   # disabled: each episode reported once
        if self.xp.any(ended):
            info["episode"], info["_episode"] = self.stats.summary(success), ended
            info["episode"]["invalid"] = self.xp.copy(invalid)
            if self.autoreset == "same_step":
                info["final_obs"], info["_final_obs"] = obs, done
                self.reset_worlds(done)
                obs = self._observe(self.state, only=done)
        if self.autoreset == "next_step":
            self._pending = done
        return obs, reward, terminated, truncated, info

    def _camera_source(self) -> Any:
        """What the cameras render: the fast tracer's scene of the sim, the one ``Camera.render(sim)`` uses, so a
        policy image shows what a video of the sim shows: library looks, scanned objects' photo textures, cloth, soft
        bodies, grains, liquid and films from the state, and the scene items' looks (``alloy.vary`` textures and
        tints), which ``Camera.render(sim)`` draws on the full-material tracer and the policy camera, at a fraction of
        its cost, as colours and tiled colour maps (``alloy.camera.item_fast``). A cache lookup per render."""
        from alloy.camera import camera_scene  # noqa: PLC0415

        return camera_scene(self.sim, device=str(self.sim.engine.model.device), quality="fast")

    def _ends(self, s: BatchState) -> tuple[Any, Any, Any, Any]:
        xp, task = self.xp, self.task
        none = xp.zeros(self.num_envs, "bool")
        invalid = self.health.invalid()                # a frozen world never counts as a success
        success = _as_bool(xp, task.success(s)) & ~invalid if task.success is not None else none
        failure = _as_bool(xp, task.failure(s)) if task.failure is not None else none
        terminated = failure | (success & task.end_on_success)
        truncated = s["steps"] >= task.max_steps
        if self.together:                      # one episode clock for every world (see ``together``)
            return none, truncated, success, invalid
        if task.on_invalid == "terminate":
            terminated = terminated | invalid
        else:
            truncated = truncated | invalid
        return terminated, truncated & ~terminated, success, invalid

    def _finish_pending(self, obs: Any, reward: Any, terminated: Any, truncated: Any,
                        info: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        """next_step autoreset: worlds that ended last step restart now; their step's action, reward and info are void
        (``info["targets"]``: the targets in force after their reset)."""
        p, xp = self._pending, self.xp
        self.reset_worlds(p)
        info["success"], info["invalid"] = info["success"] & ~p, info["invalid"] & ~p
        info["terms"] = {k: xp.where(p, 0.0, v) for k, v in info["terms"].items()}
        w = getattr(self.writer, "inner", self.writer)
        if hasattr(w, "target") and hasattr(w, "rows"):
            info["targets"] = xp.where(p[:, None], xp.view(w.target)[w.rows], info["targets"])
        return self._observe(self.state, only=p), xp.where(p, 0.0, reward), terminated & ~p, truncated & ~p

    def _hold_ended(self, reward: Any, terminated: Any, truncated: Any) -> tuple[Any, Any, Any, Any]:
        """autoreset="disabled": a world that ended stays ended (its flags, reward 0, no new episode report) until
        ``reset(options={"reset_mask": ...})`` restarts it. Returns the worlds that ended at this step."""
        xp, e = self.xp, self._ended
        terminated = xp.where(e, self._ended_flags[0], terminated)
        truncated = xp.where(e, self._ended_flags[1], truncated)
        fresh = (terminated | truncated) & ~e
        self._ended_flags = (xp.where(fresh, terminated, self._ended_flags[0]),
                             xp.where(fresh, truncated, self._ended_flags[1]))
        self._ended = e | fresh
        return xp.where(e, 0.0, reward), terminated, truncated, fresh

    def reset_worlds(self, mask: Any) -> None:
        """Reset the worlds in the bool (W,) ``mask`` to their start state; the others keep running."""
        mask = self._mask(mask)
        if not self.xp.any(mask):
            return
        every = not self.xp.any(~mask)
        if self.together and not every:
            raise EnvError("this scene resets all worlds together (its liquid or solver state has no per-world "
                           "copy), but a reset of some worlds was asked", "call env.reset() without a reset_mask; "
                           "episodes here all end at max_steps", field="reset_mask")
        wmask = self.xp.to_warp_mask(mask)
        self.health.reset(wmask)
        import warp as wp  # noqa: PLC0415

        engine = self.sim.engine
        with wp.ScopedDevice(self.device):
            engine.reset(None if every else wmask)
        self.writer.reset(mask)
        if self.randomizer.jitters:
            self.randomizer.apply(mask)
            if hasattr(self.writer, "hold"):                 # an actuator model: hold the jittered start
                self.writer.hold(mask)
            if hasattr(engine.health, "reset_reference"):        # energy reference of the jittered start
                with wp.ScopedDevice(self.device):
                    engine.health.reset_reference(engine.state, wmask)
        self.start.capture(mask)
        self.goals.resample(mask)
        for cam in self.cameras.values():             # per-world sensor state (tactile stick-slip) restarts with
            if "worlds" in _params(getattr(cam, "reset", None)):   # its world only
                cam.reset(worlds=mask)
        for layer in self.layers:
            layer.on_reset(mask)
        self.stats.reset(mask)
        self._pending = self._pending & ~mask
        self._ended = self._ended & ~mask

    def render(self) -> Any:
        """``render_mode="rgb_array"``: an (H, W, 3) uint8 image of world 0. Example: ``env.render()``."""
        if self.render_mode != "rgb_array":
            return None
        return self.sim.render()

    def close(self, **kwargs: Any) -> None:
        """Free the simulation if this environment built it. Example: ``env.close()``."""
        if not self.closed and self._owns_sim:
            self.sim.close()
        self.closed = True

    # --- helpers ---------------------------------------------------------------------------------------------------

    def _mask(self, mask: Any) -> Any:
        m = self.xp.asarray(mask, "bool")
        if tuple(m.shape) != (self.num_envs,):
            raise EnvError(f"reset mask has shape {tuple(m.shape)}", f"pass a bool array of shape ({self.num_envs},)",
                           field="reset_mask")
        return m

    def _policy_action(self, actions: Any) -> Any:
        a = self.xp.asarray(actions, "float32")
        if self.writer.spec.mode != "absolute":
            return self.xp.clip(a, -1.0, 1.0)
        return self.xp.clip(a, self.writer.low, self.writer.high)      # as the writer applies it

    def _require_open(self) -> None:
        if self.closed:
            raise EnvError("the environment is closed", "make a new one")

    def __repr__(self) -> str:
        return f"VectorEnv({self.task.name!r}, num_envs={self.num_envs}, device={self.device!r}, {self.xp.name})"


def per_world_reset(engine: Any) -> bool:
    """False when the engine's start snapshot has arrays without a per-world axis: then only a
    reset of every world at once is possible. Example: ``per_world_reset(sim.engine)``."""
    snap = getattr(engine, "_initial", None)
    return not tuple(getattr(snap, "whole_arrays", ()) or ())


def _params(fn: Any) -> tuple[str, ...]:
    """Parameter names of ``fn`` (empty when it is not a function)."""
    import inspect  # noqa: PLC0415

    try:
        return tuple(inspect.signature(fn).parameters) if callable(fn) else ()
    except (TypeError, ValueError):
        return ()


def _obs_space(obs: Any, cameras: set[str]) -> Any:
    """The single-world observation space of ``obs``: a Box, or a Dict of state Boxes and uint8 image Boxes."""
    if not isinstance(obs, dict):
        return spaces.box(-np.inf, np.inf, tuple(int(d) for d in obs.shape[1:]))   # (D,), or (H, D) unflattened
    return spaces.dict_space({k: spaces.box(0, 255, tuple(v.shape[1:]), np.uint8) if k in cameras
                              else spaces.box(-np.inf, np.inf, tuple(int(d) for d in v.shape[1:]))
                              for k, v in obs.items()})


def _make_writer(spec: Any, layout: Layout, engine: Any, xp: Any) -> Any:
    """The action writer of ``spec``: ``ActionWriter`` for an ``ActionSpec``, else what the spec builds itself
    (``spec.make_writer(layout, engine, xp)``: task-space and actuator-model actions)."""
    build = getattr(spec, "make_writer", None)
    return build(layout, engine, xp) if build is not None else ActionWriter(spec, layout, engine, xp)


def _void(x: Any, mask: Any) -> Any:
    """``x`` with the rows of the masked worlds set to zero (a dict entry by entry); ``x``'s dtype is kept."""
    if isinstance(x, dict):
        return {k: _void(v, mask) for k, v in x.items()}
    return x * ~mask.reshape((-1,) + (1,) * (x.ndim - 1))


def _as_bool(xp: Any, x: Any) -> Any:
    return x if str(getattr(x, "dtype", "")) in ("bool", "torch.bool") else xp.asarray(x, "bool")


def _mode(name: str) -> Any:
    if _AutoresetMode is None:
        return name
    return {"same_step": _AutoresetMode.SAME_STEP, "next_step": _AutoresetMode.NEXT_STEP,
            "disabled": _AutoresetMode.DISABLED}[name]


def _build(scene: Any, worlds: int | None, device: str | None, seed: int, profile: str) -> Any:
    from alloy.api._build import build  # noqa: PLC0415

    return build(scene, worlds=worlds, device=device, seed=seed, profile=profile)


__all__ = ["AUTORESET", "VectorEnv"]
