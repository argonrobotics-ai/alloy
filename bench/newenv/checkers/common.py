# SPDX-License-Identifier: Apache-2.0
"""Shared by every row 7 checker: the rollout loader, Check / CheckResult, geometry helpers, generic checks.

A request checker (``checkers/<name>.py``) is::

    from bench.newenv.checkers import common as c

    def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
        checks = c.generic_checks(rollout, request)           # level 1 (+ reward and policy flags)
        rest = rollout.phase("rest")
        z = rest.probe["cube_pos"][-1, :, 2]
        checks.append(c.within("rest_height", z, 0.522, 0.528, level=2))
        return c.finish(checks, metrics={"rest_z": float(z.mean())})

Rows: index 0 of a phase is the state before its first step; row k holds the action of step k, the reward and done
that step returned, and the obs / probe / success read after it (``reward[0]`` is NaN, ``action[0]`` zero). For a
world whose ``done[k]`` is set, obs and probe at row k are already the next episode's start state.

Example:
    >>> import numpy as np
    >>> float(np.round(tilt_from_quat_wxyz(np.array([1.0, 0.0, 0.0, 0.0])), 6))
    0.0
    >>> finish([Check("a", True, 1.0, "> 0", 1), Check("b", False, 0.0, "> 0", 2)]).level
    1
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SEED_TOL = 1e-3
POLICY_RATE = 0.8
FIELDS = ("t", "obs", "reward", "done", "success", "action")


@dataclass(frozen=True)
class Check:
    """One named check and the verdict level (1, 2 or 3) it gates."""

    name: str
    ok: bool
    value: Any
    bound: Any
    level: int


@dataclass
class CheckResult:
    """``level`` is the verdict level reached (0-3); ``checks`` every check made; ``metrics`` free numbers."""

    level: int
    checks: list[Check]
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.ok]

    def to_dict(self) -> dict[str, Any]:
        return {"level": self.level, "failed": self.failed, "metrics": self.metrics,
                "checks": [{"name": c.name, "ok": bool(c.ok), "value": _plain(c.value), "bound": _plain(c.bound),
                            "level": c.level} for c in self.checks]}


@dataclass
class Phase:
    """One phase's rows: leading axes (T, N)."""

    name: str
    t: np.ndarray
    obs: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    success: np.ndarray
    action: np.ndarray
    probe: dict[str, np.ndarray]

    @property
    def steps(self) -> int:
        return len(self.t) - 1

    @property
    def num_envs(self) -> int:
        return self.obs.shape[1]


@dataclass
class Rollout:
    """All recorded phases plus ``meta`` (the run's result.json)."""

    phases: dict[str, Phase]
    meta: dict[str, Any]

    def phase(self, name: str) -> Phase:
        if name not in self.phases:
            raise KeyError(f"no phase {name!r} in the rollout; it has: {', '.join(self.phases) or 'none'}")
        return self.phases[name]

    def has(self, name: str) -> bool:
        return name in self.phases

    @classmethod
    def load(cls, out_dir: Path | str) -> "Rollout":
        """Read ``out_dir/rollout.npz`` and ``out_dir/result.json`` (either may be missing: empty then)."""
        out_dir = Path(out_dir)
        meta_file, npz = out_dir / "result.json", out_dir / "rollout.npz"
        meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
        if not npz.is_file():
            return cls({}, meta)
        with np.load(npz) as z:
            arrays = {k: z[k] for k in z.files}
        return cls.from_arrays(arrays, meta)

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> "Rollout":
        """From the flat npz layout: ``"<phase>/<field>"`` and ``"<phase>/probe/<key>"``."""
        grouped: dict[str, dict[str, np.ndarray]] = {}
        for key, value in arrays.items():
            name, _, rest = key.partition("/")
            grouped.setdefault(name, {})[rest] = value
        order = [n for n in meta.get("phase_order", []) if n in grouped] + \
                [n for n in grouped if n not in meta.get("phase_order", [])]
        phases = {}
        for name in order:
            g = grouped[name]
            probe = {k[len("probe/"):]: v for k, v in g.items() if k.startswith("probe/")}
            phases[name] = Phase(name, *(g[f] for f in FIELDS), probe)
        return cls(phases, meta)


def make_phase(name: str, t: np.ndarray, probe: dict[str, np.ndarray], *, obs: np.ndarray | None = None,
               reward: np.ndarray | None = None, done: np.ndarray | None = None, success: np.ndarray | None = None,
               action: np.ndarray | None = None) -> Phase:
    """A phase for tests of checkers: give t (T,) and probes (T, N, ...); the rest defaults to zeros / False."""
    t = np.asarray(t, np.float32)
    steps = len(t)
    n = next(iter(probe.values())).shape[1] if probe else np.asarray(obs).shape[1]

    def default(value: np.ndarray | None, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.zeros(shape, dtype) if value is None else np.asarray(value, dtype)

    return Phase(name, t, default(obs, (steps, n, 1), np.float32), default(reward, (steps, n), np.float32),
                 default(done, (steps, n), bool), default(success, (steps, n), bool),
                 default(action, (steps, n, 1), np.float32), {k: np.asarray(v, np.float32) for k, v in probe.items()})


def make_meta(rollout_phases: list[Phase], dt: float = 0.01, episode_steps: int = 500, **extra: Any) -> dict[str, Any]:
    """A result.json for a synthetic rollout that passes the generic checks' bookkeeping."""
    first = rollout_phases[0]
    meta = {"completed": True, "errors": [], "violations": [], "dt": dt, "episode_steps": episode_steps,
            "num_envs": first.num_envs, "obs_dim": first.obs.shape[2], "act_dim": first.action.shape[2],
            "phase_order": [p.name for p in rollout_phases],
            "phases": {p.name: {"steps": p.steps, "steps_done": p.steps, "stride": 1, "error": None,
                                "reset": p.name != "rest", "done_count": p.done.sum(axis=0).tolist()}
                       for p in rollout_phases},
            "seeds": {"missing": [], "seed0_repeat_max_err": 0.0, "seed0_repeat_ok": True, "seed1_max_diff": 1.0,
                      "seed1_differs": True, "start_world_spread": 1.0},
            "timing": {"small": None, "large": None}}
    meta.update(extra)
    return meta


def synthetic(phases: list[Phase], **meta: Any) -> Rollout:
    return Rollout({p.name: p for p in phases}, make_meta(phases, **meta))


# ---- small check builders -------------------------------------------------------------------------------------------

def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v.item() if isinstance(v, np.generic) else v


def within(name: str, value: Any, lo: float, hi: float, level: int, min_worlds: int | None = None) -> Check:
    """``lo <= value <= hi`` in every world (or in at least ``min_worlds``); value reported: the worst world."""
    v = np.atleast_1d(np.asarray(value, np.float64))
    good = np.isfinite(v) & (v >= lo) & (v <= hi)
    need = v.size if min_worlds is None else min_worlds
    mid = 0.5 * (lo + hi) if np.isfinite(lo) and np.isfinite(hi) else (lo if np.isfinite(lo) else hi)
    worst = v[int(np.argmax(np.where(np.isfinite(v), np.abs(v - mid), np.inf)))]
    return Check(name, bool(good.sum() >= need), float(worst), f"[{lo:g}, {hi:g}] in >= {need} of {v.size}", level)


def at_least(name: str, value: float, bound: float, level: int) -> Check:
    return Check(name, bool(np.isfinite(value) and value >= bound), float(value), f">= {bound:g}", level)


def at_most(name: str, value: float, bound: float, level: int) -> Check:
    return Check(name, bool(np.isfinite(value) and value <= bound), float(value), f"<= {bound:g}", level)


def finish(checks: list[Check], metrics: dict[str, Any] | None = None) -> CheckResult:
    """The level reached: the highest L such that every check gating a level <= L holds (3 needs a level-3 check)."""
    level = 0
    for lv in (1, 2, 3):
        mine = [c for c in checks if c.level == lv]
        if any(not c.ok for c in mine) or (lv == 3 and not mine):
            break
        level = lv
    return CheckResult(level, list(checks), dict(metrics or {}))


# ---- geometry helpers -----------------------------------------------------------------------------------------------

def tilt_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    """Angle (rad) between the body's z axis and world z, for quaternions (..., 4) in wxyz order."""
    q = np.asarray(quat, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    return np.arccos(np.clip(1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2), -1.0, 1.0))


def speed(pos: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Finite-difference speed (T, N, ...) of positions (T, N, ..., 3); row 0 repeats row 1."""
    dt = np.diff(np.asarray(t, np.float64)).reshape(-1, *([1] * (pos.ndim - 2)))
    v = np.linalg.norm(np.diff(np.asarray(pos, np.float64), axis=0), axis=-1) / dt
    return np.concatenate([v[:1], v], axis=0)


def settled(speed_rows: np.ndarray, bound: float, tail: int = 10) -> np.ndarray:
    """(N,) bool: every speed in the last ``tail`` rows is under ``bound`` (speeds (T, N) or (T, N, K))."""
    last = np.asarray(speed_rows)[-tail:]
    last = last.reshape(last.shape[0], last.shape[1], -1)
    return np.all(np.nan_to_num(last, nan=0.0) < bound, axis=(0, 2))


def _directions() -> np.ndarray:
    grid = np.array([[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) > (0, 0, 0)],
                    np.float64)
    return grid / np.linalg.norm(grid, axis=1, keepdims=True)


def extent(points: np.ndarray) -> np.ndarray:
    """Largest width (..., ) of point sets (..., V, 3) over 13 directions: the diameter to within a few percent."""
    proj = np.asarray(points, np.float64) @ _directions().T
    return np.max(np.nanmax(proj, axis=-2) - np.nanmin(proj, axis=-2), axis=-1)


def stretch(points: np.ndarray, rest_extent: float | None = None) -> np.ndarray:
    """Extent of a cloth / soft body (T, N, V, 3) over its rest extent (default: its own extent in row 0).

    A sheet cannot span more than its rest diagonal, so values above 1 + tolerance mean it stretched. For a rope use
    ``polyline_length``.
    """
    e = extent(points)
    return e / (e[0] if rest_extent is None else rest_extent)


def polyline_length(points: np.ndarray) -> np.ndarray:
    """Length (...,) of ordered points (..., K, 3)."""
    return np.linalg.norm(np.diff(np.asarray(points, np.float64), axis=-2), axis=-1).sum(axis=-1)


def inside_box(points: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Bool (...,): points (..., 3) inside the axis-aligned box [lo, hi] (each broadcastable to the points)."""
    p = np.asarray(points, np.float64)
    return np.all((p >= np.asarray(lo) - margin) & (p <= np.asarray(hi) + margin), axis=-1)


def inside_cylinder(points: np.ndarray, base: np.ndarray, radius: Any, height: Any, margin: float = 0.0) -> np.ndarray:
    """Bool (...,): points (..., 3) inside an upright cylinder with bottom centre ``base`` (broadcastable)."""
    d = np.asarray(points, np.float64) - np.asarray(base, np.float64)
    r_ok = np.linalg.norm(d[..., :2], axis=-1) <= np.asarray(radius) + margin
    return r_ok & (d[..., 2] >= -margin) & (d[..., 2] <= np.asarray(height) + margin)


def volume(points: np.ndarray, particle_volume: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Volume (T, N) of particles (T, N, P, 3): finite (emitted) points, optionally only where ``mask`` (T, N, P)."""
    live = np.all(np.isfinite(points), axis=-1)
    if mask is not None:
        live &= mask
    return live.sum(axis=-1) * np.asarray(particle_volume, np.float64)


# ---- generic checks (every request) ---------------------------------------------------------------------------------

def _check_completed(r: Rollout) -> Check:
    errors = [f"{e.get('stage')}/{e.get('phase')}: {e.get('error')}" for e in r.meta.get("errors", [])
              if e.get("stage") != "large"]
    needed = [n for n in ("rest", "random") if not r.has(n)]
    ok = bool(r.meta.get("completed")) and not errors and not needed
    return Check("completed", ok, errors + [f"missing phase {n}" for n in needed], "phases 1-4 and 6 ran", 1)


def _check_shapes(r: Rollout) -> Check:
    bad = list(r.meta.get("violations", []))
    n, od, ad = (r.meta.get(k) for k in ("num_envs", "obs_dim", "act_dim"))
    for p in r.phases.values():
        rows = len(p.t)
        want = {"obs": (rows, n, od), "reward": (rows, n), "done": (rows, n), "success": (rows, n),
                "action": (rows, n, ad)}
        bad += [f"{p.name}/{k}: {getattr(p, k).shape} != {s}" for k, s in want.items() if getattr(p, k).shape != s]
        bad += [f"{p.name}/probe/{k}: {v.shape}" for k, v in p.probe.items() if v.shape[:2] != (rows, n)]
    return Check("shapes", not bad, bad, "as the contract states", 1)


def _variable_keys(request: dict[str, Any]) -> set[str]:
    return {k for k, shape in request.get("probe", {}).items() if any(isinstance(d, str) for d in shape)}


def _check_finite(r: Rollout, request: dict[str, Any]) -> Check:
    loose = {k for k in _variable_keys(request) if "liquid" in k or "sand" in k}  # not-yet-emitted rows are NaN
    bad = []
    for p in r.phases.values():
        fields = {"obs": p.obs, "reward": p.reward[1:], "action": p.action}
        fields.update({f"probe/{k}": v for k, v in p.probe.items() if k not in loose})
        bad += [f"{p.name}/{k}" for k, v in fields.items() if not np.all(np.isfinite(v))]
        bad += [f"{p.name}/probe/{k} (all NaN)" for k in loose if k in p.probe and not np.any(np.isfinite(p.probe[k]))]
    return Check("finite", not bad, bad, "no NaN / inf", 1)


def _check_done_fires(r: Rollout) -> Check:
    info = r.meta.get("phases", {}).get("random", {})
    counts = info.get("done_count") or (r.phase("random").done.sum(axis=0).tolist() if r.has("random") else [0])
    return Check("done_fires", min(counts) >= 1, int(min(counts)), ">= 1 done per world in two episodes", 1)


def _start_rows(r: Rollout, key: str) -> np.ndarray:
    """Start states of a fixed-shape probe: row 0 of 'rest' and of every phase that began with a reset."""
    info = r.meta.get("phases", {})
    rows = [p.probe[key][0] for p in r.phases.values()
            if key in p.probe and (p.name == "rest" or info.get(p.name, {}).get("reset", p.name == "random"))]
    return np.concatenate(rows).reshape(sum(len(x) for x in rows), -1)


def _check_resets(r: Rollout, request: dict[str, Any]) -> Check:
    """After a done, a world's probes are back inside the range of start states (half a range of slack)."""
    bound = ">= 0.9 of post-done states inside the start ranges"
    if not r.has("random") or not r.phase("random").done.any():
        return Check("resets", False, None, bound, 1)
    p = r.phase("random")
    if r.meta.get("phases", {}).get("random", {}).get("stride", 1) > 1:
        return Check("resets", True, "not checked: rows are strided", bound, 1)
    inside = np.ones(int(p.done.sum()), bool)
    for key in set(p.probe) - _variable_keys(request):
        start = _start_rows(r, key)
        lo, hi = start.min(axis=0), start.max(axis=0)
        slack = 0.5 * (hi - lo) + 0.02 * np.abs(start).max(axis=0) + 1e-3
        after = p.probe[key][p.done].reshape(len(inside), -1)
        inside &= np.all((after >= lo - slack) & (after <= hi + slack), axis=1)
    return Check("resets", bool(inside.mean() >= 0.9), float(inside.mean()), bound, 1)


def _check_worlds_differ(r: Rollout) -> Check:
    if not r.has("random"):
        return Check("worlds_differ", False, None, "> 1e-6", 1)
    obs = np.where(np.isfinite(r.phase("random").obs), r.phase("random").obs, np.nan)
    spread = float(np.nanmax(np.nanstd(obs, axis=1)))
    return Check("worlds_differ", spread > 1e-6, spread, "obs spread across worlds > 1e-6 in 'random'", 1)


def _check_seeds(r: Rollout) -> list[Check]:
    s = r.meta.get("seeds", {})
    err = s.get("seed0_repeat_max_err", float("inf"))
    repeat = Check("seed_repeat", bool(err <= SEED_TOL), err, f"<= {SEED_TOL:g}", 1)
    random_start = s.get("start_world_spread", 0.0) > 1e-6  # a request with no random start has nothing to vary
    differs = Check("seed_differs", bool(s.get("seed1_differs", False)) or not random_start,
                    s.get("seed1_max_diff"), f"> {SEED_TOL:g} when starts vary per world", 1)
    return [repeat, differs]


def _check_reward(r: Rollout) -> Check:
    """Mean reward in success rows is not below the mean first-step reward (2 % of the reward range of slack)."""
    wins, starts, lo, hi = [], [], np.inf, -np.inf
    for p in r.phases.values():
        rew = p.reward[1:]
        if rew.size == 0:
            continue
        wins.append(rew[p.success[1:]])
        starts.append(rew[0])
        lo, hi = min(lo, float(np.nanmin(rew))), max(hi, float(np.nanmax(rew)))
    wins_all = np.concatenate(wins) if wins else np.zeros(0)
    bound = "reward in success rows >= reward at the start"
    if wins_all.size == 0:
        return Check("reward_success", True, "no success row: not checked", bound, 2)
    gap = float(np.mean(wins_all) - np.mean(np.concatenate(starts)))
    return Check("reward_success", bool(gap >= -0.02 * (hi - lo) - 1e-9), gap, bound, 2)


def _check_episode_length(r: Rollout, request: dict[str, Any]) -> Check:
    """An episode lasts at least the request's ``episode_s``: the harness's action phases are sized to fit in one.

    >>> r = Rollout({}, {"dt": 0.02, "episode_steps": 250})
    >>> _check_episode_length(r, {"episode_s": 5.0}).ok, _check_episode_length(r, {"episode_s": 8.0}).ok
    (True, False)
    """
    need = float(request.get("episode_s", 0.0))
    have = float(r.meta.get("dt", 0.0)) * int(r.meta.get("episode_steps", 0))
    return Check("episode_length", bool(have >= need - 1e-6), have, f">= episode_s = {need:g} s", 1)


def _check_policy_flag(r: Rollout) -> Check:
    rate = r.meta.get("policy", {}).get("success_rate", 0.0)
    return Check("policy_success_flag", bool(rate >= POLICY_RATE), float(rate), f">= {POLICY_RATE:g} of worlds", 3)


def generic_checks(rollout: Rollout, request: dict[str, Any]) -> list[Check]:
    """The checks every request shares; a check that cannot be evaluated fails instead of raising."""
    makers = [lambda: [_check_completed(rollout)], lambda: [_check_shapes(rollout)],
              lambda: [_check_finite(rollout, request)], lambda: [_check_done_fires(rollout)],
              lambda: [_check_resets(rollout, request)], lambda: [_check_worlds_differ(rollout)],
              lambda: _check_seeds(rollout), lambda: [_check_reward(rollout)],
              lambda: [_check_episode_length(rollout, request)]]
    if request.get("policy"):
        makers.append(lambda: [_check_policy_flag(rollout)])
    checks: list[Check] = []
    for i, make in enumerate(makers):
        try:
            checks += make()
        except Exception as exc:  # noqa: BLE001  a broken rollout must give a verdict, not a crash
            checks.append(Check(f"generic_{i}", False, f"{type(exc).__name__}: {exc}", "evaluable", 1))
    return checks


def run_checker(module: Any, out_dir: Path | str, request: dict[str, Any]) -> dict[str, Any]:
    """Load ``out_dir`` and run ``module.check``; a declined run or a checker crash still returns a dict."""
    rollout = Rollout.load(out_dir)
    if rollout.meta.get("supported") is False:
        return {"level": None, "declined": True, "reason": str(rollout.meta.get("reason", ""))[:300], "failed": [],
                "metrics": {}, "checks": []}
    try:
        return {"declined": False, **module.check(rollout, request).to_dict()}
    except Exception as exc:  # noqa: BLE001
        generic = finish(generic_checks(rollout, request))
        return {"declined": False, **generic.to_dict(), "level": min(generic.level, 1),
                "checker_error": f"{type(exc).__name__}: {exc}"}
