#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Row 7 driver: runs one environment (``env.py`` in the current directory) through the harness's phases.

Stand-alone (numpy + standard library), beside ``request.json`` (id, probe keys, defaults) and ``script.py`` (the
harness's action laws: ``PHASES`` and ``action(phase, t, probe, act_dim)``).

    python drive.py            # the scored run: writes out/rollout.npz and out/result.json
    python drive.py --quick    # self-test in about a minute: short phases, writes out_quick/, prints PROBLEMS

Stages run as separate processes (``--in-process`` keeps one): "main" (phases at seed 0), "seed1" and "seed0" (start
states only; "seed0" also times 200 steps), "large" (timing at 256 worlds, never fatal). A failure is recorded in
result.json with its traceback; everything recorded before it is still written.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
DRIVER_VERSION = 1
MAX_PHASE_BYTES = 200e6
START_TOL = 1e-3
QUICK_STEPS = 20
TIMING_STEPS = 200
WARMUP_STEPS = 10
RANDOM_SEED_OFFSET = 12345
STAGES = ("main", "seed1", "seed0", "large")
Law = Callable[[float, dict[str, np.ndarray]], Any]


class ContractError(RuntimeError):
    """The environment does not follow the contract (wrong shape, missing attribute or probe key)."""


@dataclass
class Config:
    request: dict[str, Any]
    num_envs: int
    seed: int
    quick: bool
    out: Path
    task_dir: Path


@dataclass
class PhaseSpec:
    name: str
    steps: int
    reset: bool
    kind: str  # "zero" | "script" | "random" | "policy"


def to_numpy(x: Any, dtype: Any = np.float32) -> np.ndarray:
    """A numpy copy of a numpy / torch / jax / warp array (or a nested list)."""
    x = x.detach() if hasattr(x, "detach") else x
    x = x.cpu() if hasattr(x, "cpu") else x
    if hasattr(x, "numpy") and not isinstance(x, np.ndarray):
        x = x.numpy()
    return np.array(x, dtype=dtype)


def action_converter(sample: Any) -> Callable[[np.ndarray], Any]:
    """Actions go to the env in the array type (and device) of the observation that ``reset`` returned."""
    root = type(sample).__module__.split(".")[0]
    if root == "torch":
        torch = importlib.import_module("torch")
        return lambda a: torch.as_tensor(a, device=sample.device)
    if root in ("jax", "jaxlib"):
        jnp = importlib.import_module("jax.numpy")
        return lambda a: jnp.asarray(a)
    return lambda a: a


class EnvIO:
    """The env behind numpy: converts arrays, checks shapes against the contract, notes violations."""

    def __init__(self, env: Any, probe_spec: dict[str, list], num_envs: int) -> None:
        missing = [a for a in ("num_envs", "obs_dim", "act_dim", "dt", "episode_steps", "reset", "step", "probe",
                               "success") if not hasattr(env, a)]
        if missing:
            raise ContractError(f"env lacks: {', '.join(missing)}")
        self.env, self.probe_spec = env, probe_spec
        self.n, self.obs_dim, self.act_dim = int(env.num_envs), int(env.obs_dim), int(env.act_dim)
        self.dt, self.episode_steps = float(env.dt), int(env.episode_steps)
        if self.n != num_envs:
            raise ContractError(f"make_env({num_envs}, ...) returned an env with num_envs = {self.n}")
        if not (self.dt > 0 and self.episode_steps > 0 and self.obs_dim > 0 and self.act_dim > 0):
            raise ContractError("dt, episode_steps, obs_dim and act_dim must be positive")
        self.violations: list[str] = []
        self.counts: dict[str, int] = {}
        self.raw_obs: Any = None
        self.last_obs: np.ndarray | None = None
        self._to_env: Callable[[np.ndarray], Any] | None = None

    def _fit(self, x: Any, shape: tuple[int, ...], what: str, dtype: Any = np.float32) -> np.ndarray:
        arr = to_numpy(x, dtype)
        if arr.shape == shape:
            return arr
        note = f"{what}: shape {arr.shape}, contract says {shape}"
        if note not in self.violations:
            self.violations.append(note)
        if arr.size != math.prod(shape):
            raise ContractError(note)
        return arr.reshape(shape)

    def _take_obs(self, raw: Any) -> np.ndarray:
        self.raw_obs = raw
        if self._to_env is None:
            self._to_env = action_converter(raw)
        self.last_obs = self._fit(raw, (self.n, self.obs_dim), "obs")
        return self.last_obs

    def reset(self) -> np.ndarray:
        return self._take_obs(self.env.reset())

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._to_env is None:
            raise ContractError("step before reset")
        out = self.env.step(self._to_env(np.array(action, dtype=np.float32)))
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise ContractError("env.step(action) must return (obs, reward, done)")
        obs = self._take_obs(out[0])
        return obs, self._fit(out[1], (self.n,), "reward"), self._fit(out[2], (self.n,), "done", bool)

    def _probe_shape(self, key: str, arr: np.ndarray) -> tuple[int, ...]:
        dims: list[int] = []
        for i, d in enumerate(self.probe_spec[key]):
            if isinstance(d, str):  # "V" / "P" / "K": any count, fixed for the run
                d = self.counts.setdefault(key, arr.shape[1 + i] if arr.ndim > 1 + i else 0)
            dims.append(int(d))
        return (self.n, *dims)

    def probe(self) -> dict[str, np.ndarray]:
        raw = self.env.probe()
        missing = [k for k in self.probe_spec if k not in raw]
        if missing:
            raise ContractError(f"env.probe() lacks keys: {', '.join(missing)}")
        out = {}
        for key in self.probe_spec:
            arr = to_numpy(raw[key])
            out[key] = self._fit(arr, self._probe_shape(key, arr), f"probe[{key!r}]")
        return out

    def success(self) -> np.ndarray:
        return self._fit(self.env.success(), (self.n,), "success", bool)

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001  a failing close must not hide the recorded run
                traceback.print_exc()


@dataclass
class Recorder:
    """Rows of one phase; index 0 is the state before the phase's first step."""

    rows: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def add(self, t: float, obs: np.ndarray, reward: np.ndarray, done: np.ndarray, success: np.ndarray,
            action: np.ndarray, probe: dict[str, np.ndarray]) -> None:
        row = {"t": np.float32(t), "obs": obs, "reward": reward, "done": done, "success": success, "action": action}
        row.update({f"probe/{k}": v for k, v in probe.items()})
        for key, value in row.items():
            self.rows.setdefault(key, []).append(np.array(value))

    def row_bytes(self) -> int:
        return sum(v[0].nbytes for v in self.rows.values())

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.rows.items()}


def shape_action(raw: Any, io: EnvIO) -> np.ndarray:
    a = np.broadcast_to(to_numpy(raw), (io.n, io.act_dim)).copy()
    if not np.all(np.isfinite(a)):
        raise ContractError("action law returned non-finite values")
    return np.clip(a, -1.0, 1.0)


def run_phase(io: EnvIO, spec: PhaseSpec, law: Law) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one phase; on an exception keep the rows recorded so far and report the error in the info."""
    rec = Recorder()
    info: dict[str, Any] = {"steps": spec.steps, "steps_done": 0, "reset": spec.reset, "kind": spec.kind,
                            "stride": 1, "error": None, "traceback": None}
    done_count = np.zeros(io.n, dtype=np.int64)
    t0 = time.perf_counter()
    try:
        obs = io.reset() if spec.reset or io.last_obs is None else io.last_obs
        probe = io.probe()
        zeros = np.zeros((io.n, io.act_dim), np.float32)
        rec.add(0.0, obs, np.full(io.n, np.nan, np.float32), np.zeros(io.n, bool), io.success(), zeros, probe)
        stride = info["stride"] = max(1, math.ceil(spec.steps * rec.row_bytes() / MAX_PHASE_BYTES))
        window = np.zeros(io.n, bool)
        for k in range(1, spec.steps + 1):
            action = shape_action(law((k - 1) * io.dt, probe), io)
            obs, reward, done = io.step(action)
            probe, success = io.probe(), io.success()
            window |= done
            done_count += done
            if k % stride == 0 or k == spec.steps:
                rec.add(k * io.dt, obs, reward, window.copy(), success, action, probe)
                window[:] = False
            info["steps_done"] = k
    except Exception as exc:  # noqa: BLE001  recorded; later phases still run
        info["error"], info["traceback"] = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    data = rec.arrays() if rec.rows else {}
    rows = {k: data[k][1:] if k == "reward" else data[k] for k in ("obs", "reward") if k in data}  # reward row 0 is nan
    nonfinite = {k: int(np.size(v) - np.count_nonzero(np.isfinite(v))) for k, v in rows.items()}
    info.update(wall_s=time.perf_counter() - t0, done_count=done_count.tolist(), nonfinite=nonfinite)
    return data, info


def load_script(task_dir: Path) -> Any:
    spec = importlib.util.spec_from_file_location("newenv_script", task_dir / "script.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_env_module() -> Any:
    if str(Path.cwd()) not in sys.path[:1]:
        sys.path.insert(0, str(Path.cwd()))
    return importlib.import_module("env")


def open_env(cfg: Config, num_envs: int, seed: int) -> EnvIO:
    module = import_env_module()
    if not hasattr(module, "make_env"):
        raise ContractError("env.py has no make_env(num_envs, seed)")
    return EnvIO(module.make_env(num_envs, seed), cfg.request.get("probe", {}), num_envs)


def plan_phases(cfg: Config, io: EnvIO, script: Any) -> list[PhaseSpec]:
    def steps(seconds: float) -> int:
        n = max(1, round(seconds / io.dt))
        return min(n, QUICK_STEPS) if cfg.quick else n

    plan = [PhaseSpec("rest", steps(float(cfg.request.get("rest_s", 2.0))), False, "zero")]
    for ph in getattr(script, "PHASES", []):
        plan.append(PhaseSpec(ph["name"], steps(float(ph["seconds"])), bool(ph.get("reset", False)), "script"))
    episode = min(io.episode_steps, QUICK_STEPS) if cfg.quick else io.episode_steps
    plan.append(PhaseSpec("random", 2 * episode, True, "random"))
    if cfg.request.get("policy"):
        plan.append(PhaseSpec("policy", episode, True, "policy"))
    return plan


def make_law(spec: PhaseSpec, io: EnvIO, script: Any, cfg: Config) -> Law:
    if spec.kind == "zero":
        return lambda t, probe: np.zeros((io.n, io.act_dim), np.float32)
    if spec.kind == "script":
        return lambda t, probe: script.action(spec.name, t, probe, io.act_dim)
    if spec.kind == "random":
        rng = np.random.default_rng(cfg.seed + RANDOM_SEED_OFFSET)
        return lambda t, probe: rng.uniform(-1.0, 1.0, (io.n, io.act_dim))
    policy = getattr(import_env_module(), "scripted_policy", None)
    if policy is None:
        raise ContractError("this request needs scripted_policy(env, obs) in env.py")
    return lambda t, probe: policy(io.env, io.raw_obs)


def capture_start(io: EnvIO) -> dict[str, np.ndarray]:
    return {"obs": io.reset(), **{f"probe/{k}": v for k, v in io.probe().items()}}


def run_phases(cfg: Config, io: EnvIO, res: dict[str, Any], rollout: dict[str, np.ndarray]) -> None:
    script = load_script(cfg.task_dir)
    for spec in plan_phases(cfg, io, script):
        try:
            law = make_law(spec, io, script, cfg)
        except Exception as exc:  # noqa: BLE001
            data, info = {}, {"steps": spec.steps, "steps_done": 0, "reset": spec.reset, "kind": spec.kind,
                              "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        else:
            data, info = run_phase(io, spec, law)
        rollout.update({f"{spec.name}/{k}": v for k, v in data.items()})
        res["phase_order"].append(spec.name)
        res["phases"][spec.name] = info
        if spec.kind == "policy" and data:  # success per world within its first episode (to its first done row)
            ok = np.any(data["success"] & ~(np.cumsum(data["done"], axis=0) - data["done"] > 0), axis=0)
            res["policy"] = {"success": ok.tolist(), "success_rate": float(np.mean(ok))}


def stage_main(cfg: Config) -> dict[str, Any]:
    res: dict[str, Any] = {"phase_order": [], "phases": {}, "violations": [], "error": None, "traceback": None}
    rollout: dict[str, np.ndarray] = {}
    io = None
    try:
        io = open_env(cfg, cfg.num_envs, cfg.seed)
        res.update(num_envs=io.n, obs_dim=io.obs_dim, act_dim=io.act_dim, dt=io.dt, episode_steps=io.episode_steps)
        np.savez(cfg.out / "_start_main.npz", **capture_start(io))
        run_phases(cfg, io, res, rollout)
    except Exception as exc:  # noqa: BLE001
        res["error"], res["traceback"] = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    finally:
        np.savez(cfg.out / "rollout.npz", **rollout)
        if io is not None:
            res["violations"], res["probe_counts"] = io.violations, io.counts
            io.close()
    return res


def time_steps(io: EnvIO, steps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    for _ in range(min(WARMUP_STEPS, steps)):
        io.step(rng.uniform(-1.0, 1.0, (io.n, io.act_dim)))
    t0 = time.perf_counter()
    for _ in range(steps):
        io.step(rng.uniform(-1.0, 1.0, (io.n, io.act_dim)))
    wall = time.perf_counter() - t0
    return {"steps": steps, "wall_s": wall, "steps_per_s": steps / wall, "world_steps_per_s": io.n * steps / wall,
            "realtime_factor": io.n * steps * io.dt / wall}


def stage_start(cfg: Config, name: str, num_envs: int, seed: int, save: bool, timing: bool) -> dict[str, Any]:
    """Make an env, reset it, optionally save its start state and time its steps."""
    res: dict[str, Any] = {"num_envs": num_envs, "seed": seed, "error": None, "traceback": None}
    io = None
    try:
        t0 = time.perf_counter()
        io = open_env(cfg, num_envs, seed)
        res["make_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        start = capture_start(io)
        res["reset_s"] = time.perf_counter() - t0
        if save:
            np.savez(cfg.out / f"_start_{name}.npz", **start)
        if timing:
            res.update(time_steps(io, QUICK_STEPS if cfg.quick else TIMING_STEPS, seed))
    except Exception as exc:  # noqa: BLE001
        res["error"], res["traceback"] = f"{type(exc).__name__}: {exc}", traceback.format_exc()
    finally:
        if io is not None:
            io.close()
    return res


def run_stage(cfg: Config, name: str) -> dict[str, Any]:
    if name == "main":
        return stage_main(cfg)
    seed, save, timing = {"seed1": (cfg.seed + 1, True, False), "seed0": (cfg.seed, True, True),
                          "large": (cfg.seed, False, True)}[name]
    num_envs = int(cfg.request.get("timing_num_envs", 256)) if name == "large" else cfg.num_envs
    return stage_start(cfg, name, num_envs, seed, save=save, timing=timing)


def run_stage_process(cfg: Config, name: str, argv: list[str]) -> dict[str, Any]:
    """One stage in a child process, so a hard crash (or a second engine start) cannot take the run down."""
    path = cfg.out / f"_stage_{name}.json"
    path.unlink(missing_ok=True)
    timeout = cfg.request.get("large_timeout_s", 600) if name == "large" else cfg.request.get("stage_timeout_s")
    try:
        code: Any = subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv, "--stage", name],
                                   timeout=timeout, check=False).returncode
    except subprocess.TimeoutExpired:
        code = f"timeout after {timeout} s"
    if not path.is_file():
        return {"error": f"stage {name!r} wrote no result (exit: {code})", "traceback": None}
    res = json.loads(path.read_text())
    path.unlink()
    return res


def max_start_diff(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    worst = 0.0
    for key in a:
        if key not in b or a[key].shape != b[key].shape:
            return float("inf")
        both_nan = np.isnan(a[key]) & np.isnan(b[key])
        diff = np.where(both_nan, 0.0, np.abs(a[key].astype(np.float64) - b[key]))
        worst = max(worst, float(np.max(np.nan_to_num(diff, nan=np.inf), initial=0.0)))
    return worst


def seed_check(out: Path) -> dict[str, Any]:
    """Seed 1 against seed 0 (should differ) and the second seed-0 make against the first (within START_TOL)."""
    starts = {}
    for name in ("main", "seed1", "seed0"):
        path = out / f"_start_{name}.npz"
        if path.is_file():
            with np.load(path) as z:
                starts[name] = {k: z[k] for k in z.files}
            path.unlink()
    res: dict[str, Any] = {"tolerance": START_TOL, "missing": [n for n in ("main", "seed1", "seed0") if n not in starts]}
    if "main" in starts:
        res["start_world_spread"] = max(float(np.nanmax(np.nanstd(v.reshape(len(v), -1), axis=0), initial=0.0))
                                        for v in starts["main"].values())
    if not res["missing"]:
        res["seed0_repeat_max_err"] = max_start_diff(starts["main"], starts["seed0"])
        res["seed0_repeat_ok"] = bool(res["seed0_repeat_max_err"] <= START_TOL)
        res["seed1_max_diff"] = max_start_diff(starts["main"], starts["seed1"])
        res["seed1_differs"] = bool(res["seed1_max_diff"] > START_TOL)
    return res


def merge(cfg: Config, stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    main = stages.get("main", {})
    res: dict[str, Any] = {"driver_version": DRIVER_VERSION, "id": cfg.request.get("id"), "supported": True,
                           "quick": cfg.quick, "seed": cfg.seed, "num_envs": cfg.num_envs}
    res.update({k: v for k, v in main.items() if k not in ("error", "traceback")})
    res["seeds"] = seed_check(cfg.out)
    res["timing"] = {"small": stages.get("seed0"), "large": stages.get("large")}
    errors = [{"stage": "main", "phase": p, "error": i["error"], "traceback": i.get("traceback")}
              for p, i in main.get("phases", {}).items() if i.get("error")]
    errors += [{"stage": s, "phase": None, "error": r["error"], "traceback": r.get("traceback")}
               for s, r in stages.items() if r.get("error")]
    res["errors"] = errors
    res["traceback"] = next((e["traceback"] for e in errors if e["traceback"]), None)
    fatal = [e for e in errors if e["stage"] != "large"]
    res["completed"] = bool(main.get("phases")) and not fatal and not res["seeds"]["missing"]
    return res


def problems(res: dict[str, Any], request: dict[str, Any]) -> list[str]:
    """What an agent should fix, in plain lines (the hidden checks look at more than this)."""
    out = [f"{e['stage']}{'/' + e['phase'] if e['phase'] else ''}: {e['error']}" for e in res["errors"]]
    out += [f"shape: {v}" for v in res.get("violations", [])]
    for name, info in res.get("phases", {}).items():
        bad = {k: n for k, n in info.get("nonfinite", {}).items() if n}
        if bad:
            out.append(f"{name}: non-finite values {bad}")
    seeds = res.get("seeds", {})
    if seeds.get("seed0_repeat_ok") is False:
        out.append(f"make_env(seed=0) twice: start states differ by {seeds['seed0_repeat_max_err']:.3g} (> 1e-3)")
    random = res.get("phases", {}).get("random")
    if random and not res["quick"] and not random.get("error") and min(random["done_count"]) == 0:
        out.append("random: done never fired in some world over two episodes")
    if request.get("policy") and "policy" not in res:
        out.append("policy: scripted_policy did not run")
    return out


def report(res: dict[str, Any], found: list[str], out: Path) -> None:
    print(f"\n== drive.py {'--quick ' if res['quick'] else ''}{res.get('id')}: wrote {out}/result.json, rollout.npz")
    for name in res.get("phase_order", []):
        info = res["phases"][name]
        print(f"  phase {name:<12} {info['steps_done']:>5}/{info['steps']} steps  {info.get('wall_s', 0):7.2f} s"
              f"  {'ERROR' if info['error'] else 'ok'}")
    for key, row in res["timing"].items():
        if row and not row.get("error") and "steps_per_s" in row:
            print(f"  timing {key}: {row['num_envs']} worlds, {row['steps_per_s']:.1f} steps/s")
    if "policy" in res:
        print(f"  policy success rate: {res['policy']['success_rate']:.2f}")
    print("PROBLEMS:" if found else "No problems found by the driver.")
    print("".join(f"  - {line}\n" for line in found), end="")
    if res["traceback"] and found:
        print(res["traceback"])


def dump_json(path: Path, obj: Any) -> None:
    def default(o: Any) -> Any:
        return o.tolist() if isinstance(o, np.ndarray) else o.item() if isinstance(o, np.generic) else str(o)
    path.write_text(json.dumps(obj, indent=1, default=default))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="short self-test: 20-step phases, no 256-world timing")
    ap.add_argument("--num-envs", type=int, default=None, help="worlds (default: the request's, 16)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: out, or out_quick)")
    ap.add_argument("--in-process", action="store_true", help="run every stage in this process")
    ap.add_argument("--stage", choices=STAGES, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    request = json.loads((HERE / "request.json").read_text())
    out = (args.out or Path("out_quick" if args.quick else "out")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(request, args.num_envs or int(request.get("num_envs", 16)), args.seed, args.quick, out, HERE)
    if args.stage:
        dump_json(out / f"_stage_{args.stage}.json", run_stage(cfg, args.stage))
        return 0
    names = [s for s in STAGES if not (cfg.quick and s == "large")]
    child_argv = [a for a in argv if a != "--in-process"]
    stages = {s: run_stage(cfg, s) if args.in_process else run_stage_process(cfg, s, child_argv) for s in names}
    res = merge(cfg, stages)
    dump_json(out / "result.json", res)
    found = problems(res, request)
    report(res, found, out)
    return 1 if found or not res["completed"] else 0


if __name__ == "__main__":
    sys.exit(main())
