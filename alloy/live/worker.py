# SPDX-License-Identifier: Apache-2.0
"""The in-process live client: the training loop calls it once per step.

    live = LiveWorker(env, LiveConfig(hub="<hub-host>:47100", worker="vm1", run="cable-lift"), settings)
    for ...:
        obs = live.boundary(obs)                 # applies steering commands; refreshes obs of changed worlds
        obs, rew, term, trunc, info = env.step(policy(obs))
        live.after_step(obs, rew, term | trunc, info)
    live.close()

``after_step`` folds the step into on-device counters and, on its own wall-clock cadence, starts the asynchronous
copies of: every world's numbers (1 Hz), the picked worlds' 3D state (12 Hz) and the picked worlds' camera images
(15 fps, encoded on the GPU in another thread). Nothing in it waits for the GPU or the network. ``boundary`` only
blocks while the user has paused the whole run.

Commands (``alloy.live.commands``) come from the hub, or from a command log when replaying (``replay=path``); every
applied command is appended to the worker's command log and reported back to the hub.
"""
from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.live.commands import CommandLog, CommandQueue, LogEntry, Replay
from alloy.live.gate import GATE
from alloy.live.frames import PickedFrames
from alloy.live.sampling import WorldNumbers
from alloy.live.sender import Link
from alloy.live.settings import Settings
from alloy.live.steer import Steering
from alloy.live.video import VideoTap
from alloy.live.wire import Kind

log = logging.getLogger(__name__)


@dataclass
class LiveConfig:
    """Where to publish and how often. ``hub`` is "host:port" of the hub's worker port (workers dial out)."""
    hub: str
    worker: str = field(default_factory=socket.gethostname)
    run: str = "run"
    camera: str | None = "front"           # the observation key of the policy camera, None for no video
    numbers_hz: float = 1.0
    state_hz: float = 12.0
    video_fps: int = 15
    video_size: int = 128                  # mosaic tile (px): the policy camera's own size
    max_picked: int = 8
    log_dir: str = "live_logs"
    replay: str | None = None              # a command log to replay instead of listening to the hub


def _robots(sim: Any) -> list[str]:
    """Names of the scene's robots (the viewer does not offer pushes on robot links)."""
    plan = getattr(sim.built, "_plan", None)
    return [r.name for r in plan.scene.robots] if plan is not None else []


class _Every:
    def __init__(self, hz: float) -> None:
        self.period, self.next = 1.0 / hz, 0.0

    def due(self, now: float) -> bool:
        if now < self.next:
            return False
        self.next = max(self.next + self.period, now)
        return True


class LiveWorker:
    """One worker's live publisher and steering endpoint."""

    def __init__(self, env: Any, cfg: LiveConfig, settings: Settings | None = None) -> None:
        self.env, self.cfg = env, cfg
        self.settings = settings or Settings()
        self.step_count = 0
        host, port = cfg.hub.rsplit(":", 1)
        self.steer = Steering(env, self.settings)
        self.numbers, self.frames3d = WorldNumbers(env), PickedFrames(env.sim)   # 3D: the engine's frame export
        self.picked: list[int] = []
        self.log = CommandLog(Path(cfg.log_dir) / cfg.run / f"commands_{cfg.worker}.jsonl")
        self.queue: CommandQueue = Replay(CommandLog.read(cfg.replay)) if cfg.replay else CommandQueue()
        hello = {"worker": cfg.worker, "run": cfg.run, "worlds": int(env.num_envs), "dt": float(env.sim.dt),
                 "settings": self.settings.describe(), "bodies": list(env.sim.index.bodies), "robots": _robots(env.sim),
                 "video": {"fps": cfg.video_fps, "tile": cfg.video_size}, "replay": bool(cfg.replay)}
        self.link = Link(host, int(port), hello, connect=None)
        self._conn_seen = 0                 # a new hub connection: the export resends every topology
        self.video = VideoTap(self._send_video, fps=cfg.video_fps, tile=cfg.video_size) if cfg.camera else None
        self._t_numbers, self._t_state = _Every(cfg.numbers_hz), _Every(cfg.state_hz)
        self._t_video = _Every(cfg.video_fps)
        self._rate_t, self._rate_n, self.steps_per_s = time.monotonic(), 0, 0.0
        self._enabled = True

    # --- the two calls of the training loop ------------------------------------------------------------------------

    def boundary(self, obs: Any) -> Any:
        """Apply due commands at this step boundary; returns obs (re-observed when a command changed worlds)."""
        self._poll_hub()
        changed = self._apply(self.queue.due(self.step_count))
        while self.steer.run_paused:                       # the user's pause: wait here, keep the view alive
            self._send_numbers(time.monotonic(), force=True)
            time.sleep(0.05)
            self._poll_hub()
            changed |= self._apply(self.queue.due(self.step_count))
        return self.env._observe(self.env.state) if changed else obs

    def env_step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        """``env.step`` with the GPU gate held: the live threads' CUDA calls wait while the engine may capture."""
        with GATE.training():
            return self.env.step(actions)

    def after_step(self, obs: Any, reward: Any, done: Any, info: dict[str, Any]) -> None:
        """Fold this step in and start whatever samples are due (never waits)."""
        self.step_count += 1
        self.steer.after_step()
        self.numbers.add(reward, done, info)
        now = time.monotonic()
        self._rate(now)
        self._send_numbers(now)
        if self.picked and self._t_state.due(now):
            if self.link.connections != self._conn_seen:
                self._conn_seen = self.link.connections
                self.frames3d.reset()
            self.frames3d.capture(self.picked)
            pack = self.frames3d.packer()
            if pack is not None:
                # wall time of the offer only: each frame carries its own sim time and period (WP13 review)
                self.link.offer(Kind.STATE, {"t": time.time(), **self.frames3d.stats}, pack)
        if self.video is not None and self.picked and self._t_video.due(now) and isinstance(obs, dict):
            img = obs.get(self.cfg.camera)
            if img is not None:
                self.video.offer(img, self.picked)

    def close(self) -> None:
        if self.video is not None:
            self.video.close()
        self.link.close()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, on: bool) -> None:
        """Off: the trainer stops calling the worker and the encoder idles (the live-off side of an A/B)."""
        self._enabled = bool(on)
        if self.video is not None:
            self.video.paused = not on

    @property
    def active(self) -> Any:
        """(W,) float mask, 0 for paused worlds (a trainer can weight its loss with it)."""
        m = np.ones(self.env.num_envs, np.float32)
        m[list(self.steer.ops.frozen)] = 0.0
        return self.env.xp.asarray(m)

    # --- internals -------------------------------------------------------------------------------------------------

    def _rate(self, now: float) -> None:
        self._rate_n += 1
        if now - self._rate_t >= 2.0:
            self.steps_per_s = self._rate_n / (now - self._rate_t)
            self._rate_t, self._rate_n = now, 0

    def _send_numbers(self, now: float, force: bool = False) -> None:
        if not (self._t_numbers.due(now) or (force and now >= self._t_numbers.next - self._t_numbers.period / 2)):
            return
        header = {"step": self.step_count, "t": time.time(), "steps_per_s": round(self.steps_per_s, 3),
                  "paused_run": self.steer.run_paused, "dropped": self.link.dropped,
                  "video_frames": self.video.frames if self.video else 0}
        self.link.offer(Kind.NUMBERS, header, self.numbers.sample(set(self.steer.ops.frozen)))

    def _send_video(self, tiles: list[int], t: float, key: bool, data: bytes) -> bool:
        v = self.video
        return self.link.offer(Kind.VIDEO, {"tiles": tiles, "tile": v.tile, "cols": v.cols, "t": t, "key": key}, data)

    def _poll_hub(self) -> None:
        for kind, header, _payload in self.link.poll():
            if kind == Kind.PICK:
                self.picked = [int(w) for w in header.get("worlds", [])][: self.cfg.max_picked]
                if self.video is not None:
                    self.video.keep(set(self.picked))
            elif kind == Kind.COMMAND and not self.cfg.replay:
                self.queue.push(header)

    def _apply(self, cmds: list[dict[str, Any]]) -> bool:
        changed = False
        for cmd in cmds:
            with GATE.training():                          # resets and forks run engine kernels too
                ok, result, did_change = self.steer.apply(cmd)
            changed |= did_change
            entry = LogEntry(cmd["id"], cmd["op"], cmd.get("args", {}), cmd.get("issuer", "user"), self.step_count,
                             time.time(), ok, result)
            self.log.append(entry)
            self.link.offer(Kind.RESULT, {"entry": entry.__dict__, "settings": self.settings.describe()})
            log.info("live: %s %s at step %d -> %s", cmd["op"], cmd.get("args"), self.step_count,
                     "ok" if ok else result)
        return changed
