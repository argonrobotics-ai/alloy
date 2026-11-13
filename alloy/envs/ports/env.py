# SPDX-License-Identifier: Apache-2.0
"""``PortEnv``: a task port (an unofficial port of a Gym-Anything task, ``alloy.tasks``) as a Gymnasium vector env.

The port's own run function is the episode: it builds the scene, its script writes the expert's command every control
period, and at the end it scores the task by the port's own checks. :class:`PortEnv` steps that run one
control period per ``step`` (:class:`alloy.envs.ports.run.PortRun`): the policy's action replaces the script's command
for that period (:mod:`alloy.envs.ports.movers`), and ``info["expert_action"]`` is what the script wrote. Stepping
with the expert's action reproduces the port's run exactly; any other action is the policy's own.

- ``obs``: ``{"state": (W, D) float32, "image": (W, H, W, 3) uint8}`` with a camera (``camera=(width, height)``),
  else the state alone. The state is the mover's proprio (joint angles and rates, or the tool's pose and velocity),
  then the simulated time. The image draws the port's films (paint, concrete paste, rinse water and soil), paper and
  porridge from the state each render.
- ``reward``: 0 every step and the task's score at the end (1.0 for a pass; the port's partial score where it has
  one); ``terminated`` when the run ends; ``info["success"]`` and ``info["verdict"]`` (the port's full verdict) then.
- Worlds: a port that batches worlds (``spec.batched``) runs once over all of them; others run one simulation per
  world, side by side. Episodes end with their run, and with ``autoreset="same_step"`` (default) a finished run
  starts again inside ``step`` (its last observation in ``info["final_obs"]``, mask ``info["_final_obs"]``).

Example::

    from alloy.envs import make
    env = make("PaintRollStrip-v0", size="small", camera=(64, 64))
    obs, info = env.reset(seed=0)
    while True:
        obs, reward, terminated, truncated, info = env.step(info["expert_action"])
        if terminated.all():
            break
    print(info["success"])
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.envs import spaces
from alloy.envs.errors import EnvError
from alloy.envs.ports.display import deform_scene
from alloy.envs.ports.run import PortRun

log = logging.getLogger("alloy.envs.ports")

try:  # a Gymnasium VectorEnv when Gymnasium is installed; the same API without it
    from gymnasium.vector import AutoresetMode as _AutoresetMode
    from gymnasium.vector import VectorEnv as _Base
except ImportError:  # pragma: no cover - exercised where gymnasium is missing
    _Base, _AutoresetMode = object, None


def _mode(name: str) -> Any:
    """Gymnasium's ``AutoresetMode`` for ``name`` (the plain name without Gymnasium)."""
    if _AutoresetMode is None:
        return name
    return {"same_step": _AutoresetMode.SAME_STEP, "disabled": _AutoresetMode.DISABLED}[name]


@dataclass(frozen=True)
class PortSpec:
    """One port as an environment. ``run(worlds, device, seed, **options)`` runs the port's episode and returns its
    result; ``success(result, w)`` / ``score(result, w)`` read world ``w``'s verdict (the port's) from it;
    ``mover()`` makes what the policy moves; ``look_at`` is the camera's (eye, target), or None to frame the scene
    (:func:`frame`); ``display(sim)`` gives the deformables the camera draws (``alloy.envs.ports.display``)."""

    name: str
    ga_task: str
    run: Callable[..., Any]
    mover: Callable[[], Any]
    success: Callable[[Any, int], bool]
    score: Callable[[Any, int], float] | None = None
    batched: bool = False
    look_at: tuple | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    summary: str = ""
    display: Callable[[Any], Any] | None = None      # the port's deformables for the camera (display.py)


class _Slot:
    """One run of the port and what binds to its simulation (the mover, the camera)."""

    def __init__(self, spec: PortSpec, worlds: int, device: str, seed: int, options: dict) -> None:
        self.spec, self.worlds, self.device, self.seed, self.options = spec, worlds, device, seed, options
        self.run: PortRun | None = None
        self.mover, self.camera, self.bound, self.scene = spec.mover(), None, None, None
        self.fresh = False                           # started and not stepped since

    def start(self) -> None:
        if self.run is not None:
            self.run.abort()
        spec, w, dev, seed, opt = self.spec, self.worlds, self.device, self.seed, self.options
        self.run = PortRun(lambda: spec.run(worlds=w, device=dev, seed=seed, **opt), name=spec.name)
        if not self.run.start():
            raise EnvError(f"{spec.name}: the port's run ended before its first control period",
                           "check the port's options (a run with no periods has nothing to control)")
        self._rebind()
        self.fresh = True

    def _rebind(self) -> None:
        sim = self.run.sim
        if sim is not self.bound:
            self.mover.bind(sim)
            self.bound, self.camera, self.scene = sim, None, None

    def step(self, action: np.ndarray) -> bool:
        """Write ``action`` (worlds, A) for this period and step it; False once the run has ended."""
        mover, self.fresh = self.mover, False
        alive = self.run.resume(write=lambda sim: mover.write(sim, action))
        if alive:
            self._rebind()
        return alive


class PortEnv(_Base):  # type: ignore[misc, valid-type]
    """A task port over ``num_envs`` worlds. Example: see the module docstring."""

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "autoreset_mode": _mode("same_step")}

    def __init__(self, spec: PortSpec, num_envs: int = 1, device: str | None = None, seed: int = 0,
                 camera: tuple[int, int] | bool | None = None, spp: int = 4, autoreset: str = "same_step",
                 backend: str = "numpy", **options: Any) -> None:
        if autoreset not in ("same_step", "disabled"):
            raise EnvError(f"autoreset is {autoreset!r}", "a task port restarts a finished run inside step: use "
                           "'same_step' (the default) or 'disabled'; 'next_step' is not supported", field="autoreset")
        if backend != "numpy":
            raise EnvError(f"backend is {backend!r}", "a task port returns NumPy arrays: leave backend out (or pass "
                           "'numpy') and convert, e.g. torch.as_tensor(obs)", field="backend")
        self.spec, self.num_envs, self.autoreset = spec, int(num_envs), autoreset
        self.metadata = {**type(self).metadata, "autoreset_mode": _mode(autoreset)}
        self.device = device or _default_device()
        self.options = {**spec.options, **options}
        check_options(spec, self.options)
        self.camera_size = (64, 64) if camera is True else (tuple(camera) if camera else None)
        self.spp, self.render_mode, self.closed = int(spp), "rgb_array", False
        per = self.num_envs if spec.batched else 1
        self._slots = [_Slot(spec, per, self.device, seed + k, self.options) for k in range(self.num_envs // per)]
        self._returns = np.zeros(self.num_envs, np.float32)
        self._lengths = np.zeros(self.num_envs, np.int64)
        self._start_all()
        self._make_spaces(seed)
        log.info("PortEnv %s (%s): %d worlds on %s, %d runs, actions %d", spec.name, spec.ga_task, self.num_envs,
                 self.device, len(self._slots), self.single_action_space.shape[0])

    # --- spaces -----------------------------------------------------------------------------------------------------

    def _make_spaces(self, seed: int) -> None:
        m = self._slots[0].mover
        state = spaces.box(-np.inf, np.inf, (m.proprio_size + 1,))
        single = state
        if self.camera_size:
            w, h = self.camera_size
            single = spaces.dict_space({"state": state, "image": spaces.box(0, 255, (h, w, 3), np.uint8)})
        self.single_observation_space = single
        self.observation_space = spaces.batch(single, self.num_envs)
        self.single_action_space = spaces.box(m.low, m.high, (m.size,))
        self.action_space = spaces.batch(self.single_action_space, self.num_envs)
        self.action_space.seed(seed)

    # --- episode ----------------------------------------------------------------------------------------------------

    def _start_all(self) -> None:
        for slot in self._slots:
            slot.start()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict]:
        """Start runs again: every world's, or only those in ``options={"reset_mask": (W,) bool}``. A port that runs
        all worlds in one simulation (``spec.batched``) restarts them only together and raises ``EnvError`` for a
        mask of some of them. ``seed`` reseeds the runs it restarts and the action space; a run that has not stepped
        since it started, with its seed unchanged, is kept."""
        if seed is not None:
            self.action_space.seed(int(seed))
        mask = self._reset_mask(options)
        picked = self._slots_in(mask)
        for k, slot in picked:
            wanted = slot.seed if seed is None else int(seed) + k
            if wanted != slot.seed or not slot.fresh:
                slot.seed = wanted
                slot.start()
        self._returns[mask], self._lengths[mask] = 0.0, 0
        return self._observe(), {"expert_action": self.expert_action()}

    def _reset_mask(self, options: dict | None) -> np.ndarray:
        m = (options or {}).get("reset_mask")
        if m is None:
            return np.ones(self.num_envs, bool)
        m = np.asarray(_numpy(m), bool).reshape(-1)
        if m.shape != (self.num_envs,):
            raise EnvError(f"reset_mask has {m.size} values", f"pass {self.num_envs}, one per world",
                           field="reset_mask")
        return m

    def _slots_in(self, mask: np.ndarray) -> list[tuple[int, _Slot]]:
        """(index, slot) of the runs whose worlds the mask holds; ``EnvError`` if it holds part of one run."""
        out, row = [], 0
        for k, slot in enumerate(self._slots):
            rows = mask[row:row + slot.worlds]
            row += slot.worlds
            if rows.any() and not rows.all():
                raise EnvError(f"{self.spec.name} runs its {slot.worlds} worlds in one simulation, so they restart "
                               "only together, but a reset of some of them was asked", "call reset() without a "
                               "reset_mask, or with all of them in it", field="reset_mask")
            if rows.any():
                out.append((k, slot))
        return out

    def expert_action(self) -> np.ndarray:
        """(W, A) the port's scripted command for the coming period (finished runs: zeros)."""
        rows = [s.mover.read(s.run.sim) if not s.run.finished else np.zeros((s.worlds, s.mover.size), np.float32)
                for s in self._slots]
        return np.concatenate(rows, axis=0)

    def step(self, actions: Any) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict]:
        a = np.asarray(_numpy(actions), dtype=np.float32).reshape(self.num_envs, -1)
        if a.shape[1] != self.single_action_space.shape[0]:
            raise EnvError(f"actions have {a.shape[1]} values per world", f"pass {self.single_action_space.shape[0]}",
                           field="actions")
        if any(slot.run.finished for slot in self._slots):
            raise EnvError("an episode has ended and autoreset is off", "call reset() before stepping again")
        reward = np.zeros(self.num_envs, np.float32)
        done = np.zeros(self.num_envs, bool)
        success = np.zeros(self.num_envs, bool)
        verdicts: list[Any] = [None] * self.num_envs
        row = 0
        for slot in self._slots:
            sl = slice(row, row + slot.worlds)
            if not slot.step(a[sl]):
                self._finish(slot, sl, reward, done, success, verdicts)
            row += slot.worlds
        self._returns += reward
        self._lengths += 1
        info: dict[str, Any] = {}
        if done.any():
            info = self._episode_info(done, success, verdicts)
        obs = self._observe(done if done.any() else None)
        if done.any() and self.autoreset == "same_step":
            info["final_obs"], info["_final_obs"] = obs, done.copy()
            self._restart(done)
            obs = self._observe()
        info["expert_action"] = self.expert_action()
        return obs, reward, done, np.zeros(self.num_envs, bool), info

    def _finish(self, slot: _Slot, sl: slice, reward: np.ndarray, done: np.ndarray, success: np.ndarray,
                verdicts: list) -> None:
        result = slot.run.result
        for k, w in enumerate(range(sl.start, sl.stop)):
            ok = bool(self.spec.success(result, k))
            score = float(self.spec.score(result, k)) if self.spec.score is not None else float(ok)
            reward[w], done[w], success[w], verdicts[w] = score, True, ok, result

    def _episode_info(self, done: np.ndarray, success: np.ndarray, verdicts: list) -> dict:
        ep = {"r": np.where(done, self._returns, 0.0), "l": np.where(done, self._lengths, 0), "success": success}
        info = {"success": success, "verdict": verdicts, "episode": ep, "_episode": done.copy()}
        self._returns[done], self._lengths[done] = 0.0, 0
        return info

    def _restart(self, done: np.ndarray) -> None:
        row = 0
        for slot in self._slots:
            if done[row:row + slot.worlds].any():
                slot.start()
            row += slot.worlds

    # --- observations -----------------------------------------------------------------------------------------------

    def _observe(self, done: np.ndarray | None = None) -> Any:  # noqa: ARG002
        states, images = [], []
        for slot in self._slots:
            sim = slot.run.sim
            t = float(getattr(getattr(sim, "clock", None), "t", 0.0))
            states.append(np.concatenate([slot.mover.proprio(sim), np.full((slot.worlds, 1), t, np.float32)], 1))
            if self.camera_size:
                images.append(self._image(slot))
        state = np.concatenate(states, axis=0).astype(np.float32)
        return {"state": state, "image": np.concatenate(images, axis=0)} if self.camera_size else state

    def _image(self, slot: _Slot) -> np.ndarray:
        from alloy.camera import Camera  # noqa: PLC0415

        sim = slot.run.sim
        _not_capturing(sim)
        if slot.camera is None:
            w, h = self.camera_size
            view = self.spec.look_at or frame(sim)
            slot.camera = Camera(self.spec.name, look_at=view, width=w, height=h, spp=self.spp)
            layout = self.spec.display(sim) if self.spec.display is not None else None
            # the port's deformables through its display hook; else a plug-in sim itself, whose SurfaceFlow films
            # (paint, paste, rinse water and soil) the camera draws (alloy.camera.films); else its model
            plain = sim if getattr(sim, "plugins", None) is not None else sim.model
            slot.scene = deform_scene(sim, layout) if layout is not None else plain
        img = slot.camera.render(slot.scene, sim.state).numpy()
        return img[: slot.worlds]

    def render(self) -> np.ndarray | None:
        """World 0's camera image (needs ``camera=``)."""
        if not self.camera_size:
            return None
        return self._image(self._slots[0])[0]

    def close(self, **kwargs: Any) -> None:  # noqa: ARG002
        if self.closed:
            return
        for slot in self._slots:
            if slot.run is not None:
                slot.run.abort()
        self.closed = True


def frame(sim: Any, world: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(eye, target) of a three-quarter view over ``world``'s shapes and particles now (planes, and bodies named
    "dummy", which the harness scenes park far away, left out), as ``alloy.camera.pose.auto_frame`` frames a box."""
    m = sim.model
    body, typ, scale = m.shape_body.numpy(), m.shape_type.numpy(), m.shape_scale.numpy()
    xf, sw = m.shape_transform.numpy(), m.shape_world.numpy()
    labels, q = [str(lbl) for lbl in m.body_label], sim.state.body_q.numpy()
    plane = int(_newton().GeoType.PLANE)
    keep = [k for k in range(len(body)) if typ[k] != plane and sw[k] in (-1, world)
            and not (body[k] >= 0 and labels[body[k]].rsplit("/", 1)[-1] == "dummy")]
    lo, hi = np.full(3, -0.5), np.full(3, 0.5)
    if keep:
        pos = np.array([_place(q[body[k]], xf[k][:3]) if body[k] >= 0 else xf[k][:3] for k in keep])
        r = np.minimum(np.abs(scale[keep]).max(axis=1), 1.0)[:, None]
        lo, hi = (pos - r).min(axis=0), (pos + r).max(axis=0)
    n = int(getattr(m, "particle_count", 0) or 0) // max(int(getattr(m, "world_count", 1) or 1), 1)
    if n:                                   # world 0's particles (cloth, paper, fluids) too
        pts = sim.state.particle_q.numpy()[world * n:(world + 1) * n]
        lo, hi = (np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))) if keep else (pts.min(0), pts.max(0))
    target = 0.5 * (lo + hi)
    span = max(float(np.linalg.norm(hi - lo)), 0.3)
    return target + span * np.array([0.9, -1.1, 0.8]), target


def _place(body_q: np.ndarray, local: np.ndarray) -> np.ndarray:
    """``local`` (body frame) in the world: body_q is (x, y, z, qx, qy, qz, qw)."""
    p, (x, y, z, w) = body_q[:3], body_q[3:7]
    v = np.asarray(local, np.float64)
    u = np.array([x, y, z])
    return p + v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _newton() -> Any:
    from alloy._vendor import newton  # noqa: PLC0415

    return newton


def _not_capturing(sim: Any) -> None:
    """Rendering launches device work: never inside a captured graph (the port pauses between periods)."""
    import warp as wp  # noqa: PLC0415

    dev = wp.get_device(str(sim.model.device))
    if dev.is_cuda and dev.is_capturing:
        raise EnvError("a render was asked for inside a graph capture", "render between control periods")


def check_options(spec: PortSpec, options: Mapping[str, Any]) -> None:
    """Raise ``EnvError`` for an option the port's run function does not take (it would otherwise fail inside the
    run with a ``TypeError``). Example: ``check_options(PORTS["PaintClearCoat-v0"], {"size": "small"})``."""
    try:
        params = inspect.signature(spec.run).parameters
    except (TypeError, ValueError):                  # a callable without a signature: nothing to check
        return
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    known = sorted(n for n in params if n not in ("worlds", "device", "seed"))
    unknown = sorted(set(options) - set(known))
    if unknown:
        raise EnvError(f"{spec.name} has no option {', '.join(map(repr, unknown))}",
                       f"its options are {known} (its family's module in alloy/envs/ports/ describes them)",
                       field=unknown[0])


def _default_device() -> str:
    import warp as wp  # noqa: PLC0415

    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _numpy(x: Any) -> Any:
    return x.detach().cpu().numpy() if hasattr(x, "detach") else x


__all__ = ["PortEnv", "PortSpec", "check_options", "frame"]
