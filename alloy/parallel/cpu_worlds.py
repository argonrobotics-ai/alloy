# SPDX-License-Identifier: Apache-2.0
"""CPU mode A: ``alloy.build(scene, worlds, device="cpu", processes=N)`` splits the worlds over N worker processes.

Warp runs CPU kernels on one thread, so one process uses one core whatever the world count (WP9's F16 estimate).
Mode A on the CPU gives each worker process one consecutive block of worlds (``multigpu.world_block``), as
multi-GPU mode A gives each GPU one. World ids, seeds and randomized values are global, so a world's trajectory does
not depend on which process steps it (the law in ``tests/parallel/test_cpu_worlds.py``) while every process keeps the
same clock. An adaptive time step (the coupler's escalation and rungs, a liquid's CFL step) follows one model's
worlds, so a block may take another clock than one process of all of them: the split sim warns once and reports
``worker_physics_steps`` (health rows' ``first_bad_step`` count each worker's own steps). The parent keeps no
physics: ``SplitSim`` forwards each call to the workers, which run it at the same time, and joins the results
(``cpu_merge``). A ``run`` steps every block independently to the end (no per-step exchange); an interactive loop of
``step`` and ``state`` pays one round trip per call. Blocks are static: when worlds cost unequal amounts (a floating
body in a liquid), the slowest block sets the pace. Workers run one thread each (``OMP_NUM_THREADS=1``).
"""

from __future__ import annotations

import logging
import os
import pickle
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import numpy as np

from alloy.api import _say
from alloy.api import errors as E
from alloy.api.trajectory import Trajectory
from alloy.parallel import cpu_merge as M
from alloy.parallel.cpu_worker import ENV_FD
from alloy.parallel.multigpu import WorldBlock, child_env

log = logging.getLogger("alloy.parallel")

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
ENV_SCENE_WORLDS = "ALLOY_SCENE_WORLDS"            # read by alloy.scene.builtin_plugins.scene_worlds
CLOSE_WAIT_S = 10.0
NOT_FORWARDED = "the engine objects live in the worker processes"


def cores() -> int:
    """CPU cores this process may use. Example: ``cores()`` -> 16."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def process_count(processes: int | str | None, worlds: int, device: str) -> int:
    """Worker processes for ``build(..., processes=)``: None -> 1; "auto" -> one per core on the CPU (1 on a GPU);
    an int as given; never more than ``worlds``. Example: ``process_count("auto", 64, "cpu")`` -> 16."""
    if processes is None:
        return 1
    if processes == "auto":
        return min(cores(), worlds) if device == "cpu" else 1
    if isinstance(processes, bool) or not isinstance(processes, (int, np.integer)) or processes < 1:
        raise E.BuildError(f"processes must be an int >= 1 or 'auto', got {processes!r}",
                           "pass processes='auto' (one per core) or e.g. processes=4", field="processes")
    n = min(int(processes), worlds)
    if n > 1 and device != "cpu":
        raise E.BuildError(f"processes={processes} splits worlds over CPU processes, but the device is {device!r}",
                           "on a GPU one process steps every world at once: drop processes= (several GPUs: "
                           "alloy.parallel.multigpu.launch_mode_a), or pass device='cpu'", field="processes")
    return n


class _Worker:
    """One worker process and its end of the socket pair."""

    def __init__(self, rank: int, block: WorldBlock, proc: subprocess.Popen, conn: Connection) -> None:
        self.rank, self.block, self.proc, self.conn = rank, block, proc, conn

    def send(self, msg: Any) -> None:
        try:
            self.conn.send(msg)
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            raise self.gone() from exc

    def recv(self) -> Any:
        try:
            status, value = self.conn.recv()
        except (EOFError, ConnectionResetError, OSError) as exc:
            raise self.gone() from exc
        if status == "err":
            raise self.error(value)
        return value

    def gone(self) -> E.SimulationError:
        code = self.proc.poll()
        return E.SimulationError(f"worker {self.rank} (worlds {self.span()}) stopped (exit code {code})",
                                 "see its output above; out of memory? use fewer processes or worlds")

    def error(self, d: dict[str, Any]) -> E.AlloyAPIError:
        if d["kind"] == "api":
            cls = getattr(E, d["error"], None)
            cls = cls if isinstance(cls, type) and issubclass(cls, E.AlloyAPIError) else E.SimulationError
            return cls(d["message"], d["fix"], field=d.get("field"))
        log.debug("worker %d traceback:\n%s", self.rank, d["traceback"])
        return E.SimulationError(f"worker {self.rank} (worlds {self.span()}): {d['type']}: {d['message']}",
                                 "the same call with processes=None shows the full error")

    def span(self) -> str:
        return f"{self.block.offset}..{self.block.offset + self.block.count - 1}"


def _spawn(rank: int, size: int, worlds: int, blocks: list[WorldBlock]) -> _Worker:
    import alloy  # noqa: PLC0415

    mine, theirs = socket.socketpair()
    env = child_env(rank, ["cpu"] * size, worlds)
    env.update({v: "1" for v in THREAD_VARS})
    env[ENV_FD] = str(theirs.fileno())
    env[ENV_SCENE_WORLDS] = str(worlds)                # builders that choose by world count see the whole scene
    root = str(Path(alloy.__file__).resolve().parent.parent)          # the parent's alloy, wherever it came from
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen([sys.executable, "-P", "-m", "alloy.parallel.cpu_worker"], env=env,
                            pass_fds=(theirs.fileno(),))
    theirs.close()
    return _Worker(rank, blocks[rank], proc, Connection(mine.detach()))


def _picklable(plugins: list[Any]) -> None:
    try:
        pickle.dumps(plugins)
    except Exception as exc:  # noqa: BLE001 - any pickling failure
        raise E.BuildError(f"plugins cannot go to the worker processes ({type(exc).__name__}: {exc})",
                           "define each plug-in (or the function that makes it) at module level, or drop "
                           "processes=", field="plugins") from exc


def build_processes(scene: Any, worlds: int, seed: int, processes: int, *, profile: str, contacts: str,
                    plugins: list[Any]) -> SplitSim:
    """Start ``processes`` workers, each building its block of ``scene``'s ``worlds`` worlds on the CPU (all at
    once), and return the ``SplitSim`` that drives them. Example: see ``alloy.build``."""
    _picklable(plugins)
    spans = M.blocks(worlds, processes)
    job = {"scene": scene.to_dict(), "worlds": worlds, "seed": seed, "profile": profile, "contacts": contacts,
           "plugins": plugins}
    t0 = time.perf_counter()
    workers: list[_Worker] = []
    try:
        for rank, b in enumerate(spans):
            workers.append(_spawn(rank, processes, worlds, spans))
            workers[-1].send({**job, "rank": rank, "block": (b.offset, b.count)})
        facts = _gather(workers, None)
        _hold_mpm_grids(workers, worlds)
    except BaseException:
        _stop(workers)
        raise
    log.info("built %d worlds in %d CPU processes in %.1f s (blocks of %s)", worlds, processes,
             time.perf_counter() - t0, sorted({b.count for b in spans}))
    return SplitSim(workers, scene=scene, worlds=worlds, profile=profile, facts=facts[0])


def _hold_mpm_grids(workers: list[_Worker], worlds: int) -> None:
    """MPM on the CPU keeps one dense grid box for all of a model's worlds, taken from their start: hold every
    worker's at the box one process of all the worlds takes (the union of the blocks' boxes), so a world's cells and
    weights round the same in any block (``mpm_worlds``)."""
    found = _gather(workers, [("mpm_boxes",)] * len(workers))
    if not any(b is not None for part in found for b in part):
        return
    boxes = []
    for per_solver in zip(*found):
        if per_solver[0] is None:
            boxes.append(None)
            continue
        lo = np.min(np.stack([b[0] for b in per_solver]), axis=0)
        hi = np.max(np.stack([b[1] for b in per_solver]), axis=0)
        # particles: the one-process model's count, as _dense_bounds budgets it (solver.model.particle_count: every
        # particle of every kind in the solver's model), summed over the blocks; not only the MPM particles
        boxes.append((lo, hi, sum(b[2] for b in per_solver)))
    _gather(workers, [("mpm_hold", boxes, worlds)] * len(workers))
    log.info("MPM dense grids held at the whole scene's start box in every worker: %s",
             [None if b is None else (b[0].tolist(), b[1].tolist()) for b in boxes])


def _gather(workers: list[_Worker], msgs: list[Any] | None) -> list[Any]:
    """Send each worker its message (None: only receive), then read every reply; the first error is raised after
    all replies are in, so every worker stays in step with the parent."""
    first: E.AlloyAPIError | None = None
    asked = [True] * len(workers)
    for i, (w, m) in enumerate(zip(workers, msgs or [])):
        try:
            w.send(m)
        except E.AlloyAPIError as exc:                      # a worker that is gone: the others still reply
            asked[i], first = False, first or exc
    out = []
    for w, ok in zip(workers, asked):
        try:
            out.append(w.recv() if ok else None)
        except E.AlloyAPIError as exc:
            out.append(None)
            first = first or exc
    if first is not None:
        raise first
    return out


def _stop(workers: list[_Worker]) -> None:
    for w in workers:
        try:
            w.conn.send(("close",))
        except OSError:
            pass
    deadline = time.monotonic() + CLOSE_WAIT_S
    for w in workers:
        try:
            w.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            log.warning("worker %d (pid %d) did not stop in %.0f s: killing it", w.rank, w.proc.pid, CLOSE_WAIT_S)
            w.proc.kill()
            w.proc.wait()
        w.conn.close()


class SplitSim:
    """An ``alloy.Sim`` whose worlds run in worker processes (``alloy.build(..., device="cpu", processes=N)``).

    Same calls as ``Sim``: per-world values (a leading axis of ``sim.worlds``) go to the worlds' processes, results
    come back joined in world order with global world ids. ``sim.engine``, ``sim.built`` and ``sim.plugins`` are not
    available (they live in the workers): build with ``processes=None`` to reach them. ``sim.close()`` (or a
    ``with`` block) stops the workers. Example::

        sim = alloy.build(scene, worlds=64, device="cpu", processes="auto")
        traj = sim.run(1.0, record=["cube"])            # (frames, 64, 3), each process stepping its block
        sim.close()"""

    device = "cpu"

    def __init__(self, workers: list[_Worker], *, scene: Any, worlds: int, profile: str,
                 facts: Mapping[str, Any]) -> None:
        self._workers, self.scene, self.profile = workers, scene, profile
        self._worlds, self._facts = int(worlds), dict(facts)
        self.blocks = [w.block for w in workers]
        self._warned_invalid: set[int] = set()
        self._warned_clock = False

    # --- facts ---------------------------------------------------------------------------------------------------

    @property
    def processes(self) -> int:
        """Worker processes. Example: ``sim.processes`` -> 16."""
        return len(self._workers)

    @property
    def worlds(self) -> int:
        return self._worlds

    @property
    def dt(self) -> float:
        return float(self._facts["dt"])

    @property
    def t(self) -> float:
        return float(self._one(0, ("get", "t")))

    @property
    def physics_steps(self) -> int:
        """The most physics steps any worker took (``worker_physics_steps``: each one's)."""
        return max(self.worker_physics_steps)

    @property
    def worker_physics_steps(self) -> list[int]:
        """Each worker's physics steps: equal while every process keeps the same clock."""
        return [int(s) for s in _gather(self._live(), [("get", "physics_steps")] * len(self._live()))]

    def names(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._facts["names"].items()}

    def routes(self) -> dict[str, dict[str, Any]]:
        return {k: dict(v) for k, v in self._facts["routes"].items()}

    # --- stepping ------------------------------------------------------------------------------------------------

    def step(self, n: int = 1) -> SplitSim:
        self._check_clocks(self._call_all("step", lambda b: ((n,), {})))
        return self

    def run(self, seconds: float, record: Iterable[str] | str | None = None, every: int = 1, *,
            targets: Mapping[str, Any] | None = None) -> Trajectory:
        rec = list(record) if record is not None and not isinstance(record, str) else record
        t0 = time.perf_counter()
        parts = self._call_all("run", lambda b: ((seconds, rec, every),
                                                 {"targets": M.cut_ramps(targets, b, self._worlds)}))
        meta = M.join_meta([p.meta for p in parts], self.blocks, time.perf_counter() - t0)
        self._warn_invalid(meta["health"])
        self._check_clocks(self.worker_physics_steps)
        return Trajectory(M.join_frames(parts), meta)

    def settle(self, seconds: float) -> SplitSim:
        self._check_clocks(self._call_all("settle", lambda b: ((seconds,), {})))
        return self

    def set_pose(self, name: str, pos: Any = None, quat_wxyz: Any = None, *, world: int | None = None) -> SplitSim:
        if world is not None:
            rank, local = M.owner(self.blocks, int(world))
            self._one(rank, ("call", "set_pose", (name, pos, quat_wxyz), {"world": local}))
            return self
        W = self._worlds
        self._call_all("set_pose", lambda b: ((name, M.cut(pos, b, W, "pos", row_ndim=1),
                                               M.cut(quat_wxyz, b, W, "quat_wxyz", row_ndim=1)), {}))
        return self

    def drive(self, name: str, *, velocity: Any = (0.0, 0.0, 0.0), angular_velocity: Any = (0.0, 0.0, 0.0),
              pivot: Any = None, duration: float = float("inf")) -> SplitSim:
        if pivot is None:                                    # a fixture turns about its centre in world 0
            pivot = self._one(0, ("pivot", name))
        kw = {"velocity": velocity, "angular_velocity": angular_velocity, "pivot": pivot, "duration": duration}
        self._call_all("drive", lambda b: ((name,), kw))
        return self

    def stop_drives(self) -> SplitSim:
        self._call_all("stop_drives", lambda b: ((), {}))
        return self

    def reset(self) -> SplitSim:
        self._call_all("reset", lambda b: ((), {}))
        return self

    # --- state and control ---------------------------------------------------------------------------------------

    def state(self, names: Iterable[str] | str | None = None, *, world: int | None = None) -> dict[str, Any]:
        names = list(names) if names is not None and not isinstance(names, str) else names
        if world is not None:
            rank, local = M.owner(self.blocks, int(world))
            return self._one(rank, ("call", "state", (names,), {"world": local}))
        return M.join_state(self._call_all("state", lambda b: ((names,), {})))

    def set_targets(self, targets: Mapping[str, Any], kind: str = "position") -> SplitSim:
        self._call_all("set_targets", lambda b: ((M.cut_map(targets, b, self._worlds), kind), {}))
        return self

    def set_controls(self, controls: Mapping[str, Any]) -> SplitSim:
        self._call_all("set_controls", lambda b: ((M.cut_map(controls, b, self._worlds),), {}))
        return self

    def faces(self, name: str) -> np.ndarray:
        return self._one(0, ("call", "faces", (name,), {}))

    def particle_volume(self, name: str) -> float:
        return self._one(0, ("call", "particle_volume", (name,), {}))

    def liquid_volume(self, name: str) -> float:
        return self._one(0, ("call", "liquid_volume", (name,), {}))

    # --- heat ------------------------------------------------------------------------------------------------------

    def temperature(self, name: str) -> np.ndarray:
        return np.concatenate(self._call_all("temperature", lambda b: ((name,), {})), axis=0)

    def heat(self, name: str) -> dict[str, np.ndarray]:
        return M.join_per_world(self._call_all("heat", lambda b: ((name,), {})))

    def set_heater(self, name: str, *, temperature: Any = None, power: Any = None) -> SplitSim:
        W = self._worlds
        self._call_all("set_heater", lambda b: ((name,), {"temperature": M.cut(temperature, b, W, "temperature"),
                                                          "power": M.cut(power, b, W, "power")}))
        return self

    # --- checks, render, report ----------------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return M.join_health(self._call_all("health", lambda b: ((), {})), self.blocks)

    def check(self) -> dict[str, Any]:
        """``Sim.check`` over every process; each process's own check report is in ``reports``."""
        return M.join_check(self._call_all("check", lambda b: ((), {})), self.blocks)

    def render(self, camera: Any = None, **kwargs: Any) -> Any:
        """One world's image (``world=``, default 0) from the process holding it; a ``Camera`` renders every world,
        joined in world order."""
        if camera is None:
            rank, local = M.owner(self.blocks, int(kwargs.pop("world", 0)))
            return self._one(rank, ("call", "render", (None,), {**kwargs, "world": local}))
        return np.concatenate(self._call_all("render", lambda b: ((camera,), kwargs)), axis=0)

    def report(self) -> dict[str, Any]:
        """``{"processes": N, "blocks": [(offset, count), ...], "workers": [each worker's engine report]}``."""
        return {"processes": self.processes, "blocks": [(b.offset, b.count) for b in self.blocks],
                "workers": self._call_all("report", lambda b: ((), {}))}

    @property
    def plugins(self) -> Any:
        raise E.FeatureUnavailableError(f"sim.plugins is not available with processes={self.processes} "
                                        f"({NOT_FORWARDED})", "read plug-in results with sim.state / sim.heat, or "
                                        "build with processes=None")

    def close(self) -> None:
        """Stop the worker processes (each frees its scene files). Example: ``sim.close()``."""
        workers, self._workers = self.__dict__.get("_workers", []), []
        if workers:
            _stop(workers)

    def __enter__(self) -> SplitSim:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name in ("_workers", "_worlds", "_facts", "blocks"):
            raise AttributeError(name)
        raise E.FeatureUnavailableError(f"sim.{name} is not available with processes ({NOT_FORWARDED})",
                                        "build with processes=None to reach it")

    def __repr__(self) -> str:
        return f"SplitSim(worlds={self._worlds}, processes={len(self._workers)}, dt={self.dt}, device='cpu')"

    # --- internals -----------------------------------------------------------------------------------------------

    def _live(self) -> list[_Worker]:
        if not self._workers:
            raise E.SimulationError("this sim was closed (its worker processes stopped)", "build it again")
        return self._workers

    def _one(self, rank: int, msg: tuple[Any, ...]) -> Any:
        w = self._live()[rank]
        w.send(msg)
        return w.recv()

    def _call_all(self, method: str, args_of: Callable[[WorldBlock], tuple[tuple[Any, ...], dict[str, Any]]]
                  ) -> list[Any]:
        """Call ``method`` in every worker at once, with ``args_of(block)``; replies in rank order."""
        workers = self._live()
        msgs = []
        for w in workers:
            args, kwargs = args_of(w.block)
            msgs.append(("call", method, args, kwargs))
        return _gather(workers, msgs)

    def _check_clocks(self, steps: list[int]) -> None:
        """Warn once when the workers' physics step counts part: an adaptive clock (the coupler's escalation or rung,
        a liquid's CFL step) chose for one process's worlds, so worlds no longer match one process bit for bit."""
        if self._warned_clock or len(set(steps)) <= 1:
            return
        self._warned_clock = True
        _say.warn("the %d worker processes took different numbers of physics steps (%s): an adaptive time step "
                  "followed each process's own worlds, so worlds no longer match a one-process run bit for bit (each "
                  "is still a valid run); pass processes=None for one shared clock", len(steps), steps)

    def _warn_invalid(self, health: dict[str, Any]) -> None:
        new = [w for w in health.get("invalid_worlds", []) if w not in self._warned_invalid]
        if new:
            self._warned_invalid.update(new)
            _say.warn("%d world(s) became INVALID and were frozen (e.g. world %d: %s); see sim.health()",
                      len(new), new[0], health["worlds"][new[0]].get("problem", ""))
