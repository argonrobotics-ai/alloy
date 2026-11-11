# SPDX-License-Identifier: Apache-2.0
"""Alloy rows of the hold-task bench (the physics-only protocol of skild.ai/blogs/steel): physics stepping only.

    PYTHONPATH=. python bench/scorecard/mixed/single_task/alloy_hold.py TASK --mode speed --worlds 4096 --out DIR
        [--tag TAG] [--step default|5ms] [--steps 100] [--blocks 3] [--warmup 200] [--device cuda:0] [--seed 0]

The Alloy twin of the Isaac Lab side's harness (not in this repository); both share ``pxs_score.py``. TASK is a name in ``tasks.py``
(a MuJoCo Menagerie stand-in for the task's robot) or ``usd:SPEC.json`` (the Isaac Lab task's own USD robot, gains,
default pose and props, from a task spec: ``usd_task.py``).
The scene is ``alloy.Scene(worlds=N, dt=0.005)`` with the task's robot file through ``add_robot``. One env step:
write every actuator's hold target + uniform noise (``pxs_score.NOISE``, 16 pre-drawn sets cycled) through
alloy.envs' zero-copy ``ActionWriter``, one ``sim.step(1)``, then a device sync. No policy, observations, rewards,
resets or cameras.

Modes (as on the Isaac Lab side):
- ``speed``: ``--warmup`` env steps (the first compiles kernels and captures graphs), then ``--blocks`` blocks of
  ``--steps`` steps; env-steps/s = worlds x steps / the median block's wall time. Alloy's health and the hold error
  at the end are recorded beside it.
- ``quality``: ``pxs_score.QUALITY_STEPS`` steps from the start state with ``pxs_score.Watch`` fed every step from
  ``sim.state`` (fastest body, lowest body origin, finiteness; tracked poses) -> ``<tag>.npz`` and its score (the
  state-based S1 rule shared with the Isaac Lab engines, and contact held), with Alloy's own ``sim.health`` beside
  it (INVALID worlds: a contact deeper than 1 cm, an impossible speed, a non-finite state, ...).

``--step default`` is Alloy as shipped (robots keep the fixed step: 4 x 1.25 ms per 5 ms env step). ``--step 5ms`` is
a PROBE, not a supported setting: it raises the internal constant ``alloy.scene.rigid_step.RIGID_TARGET`` to 5 ms
before the build, so the same route takes one 5 ms physics step per env step (the public API has no such option:
``scene.rigid_step = 0.005`` is ignored for robots).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pxs_score as S  # noqa: E402
import tasks as T  # noqa: E402

NOISE_BANK = 16           # pre-drawn noise sets, cycled, so drawing noise is not timed
FROZEN_MAX = 0.05         # a run with a larger share of INVALID (frozen) worlds is not valid


@dataclass
class Prepared:
    """A task ready to build: how to fill the scene, what to hold, what to watch and how to score it."""

    isaac_task: str
    name: str
    add: Callable[[Any], None]
    hold: dict[str, float]            # writer column ("r/<actuator or joint>") -> hold target
    tracked: list[str]                # Alloy body names, in the order of score["track"]
    score: dict[str, Any]             # the pxs_score task entry
    note: str
    source: dict[str, Any]


def prepare(name: str, workdir: Path) -> Prepared:
    if name.startswith("usd:"):
        import usd_task as U  # noqa: PLC0415 - needs pxr

        t = U.load(name[4:], workdir)
        score = U.score_spec(t, S.TASKS[t.isaac_task])
        return Prepared(t.isaac_task, "usd", lambda scene: U.add_to(scene, t), t.hold, [x[2] for x in score["track"]],
                        score, "Isaac Lab's own USD asset; " + "; ".join(t.notes), t.source)
    task = T.TASKS[name]
    path, hold, source = T.model(task, workdir)
    return Prepared(task.isaac_task, task.name, lambda scene: scene.add_robot("r", path), hold,
                    [f"r/{b}" for _, b in task.track], T.score_spec(task), task.note, source)


def build(prep: Prepared, a: argparse.Namespace) -> Any:
    import alloy  # noqa: PLC0415

    if a.step == "5ms":
        import alloy.scene.rigid_step as rs  # noqa: PLC0415

        rs.RIGID_TARGET = S.DT         # probe only (module docstring)
    scene = alloy.Scene(worlds=a.worlds, dt=S.DT)
    prep.add(scene)
    return alloy.build(scene, device=a.device, seed=a.seed)


def make_writer(sim: Any) -> tuple[Any, Any]:
    """(writer, layout); the writer is None for a scene without driven joints (an effort-driven task: it runs
    passive, as it does under zero-gain hold targets in Isaac Lab)."""
    from alloy.envs.actions import ActionSpec, ActionWriter  # noqa: PLC0415
    from alloy.envs.backend import pick_backend  # noqa: PLC0415
    from alloy.envs.errors import EnvError  # noqa: PLC0415
    from alloy.envs.layout import Layout  # noqa: PLC0415

    lay, xp = Layout.from_sim(sim), pick_backend("auto", str(sim.device))
    n = len(lay.actuated_joints()) or len(lay.actuators)
    if not n:
        return None, lay
    try:
        return ActionWriter(ActionSpec(mode="absolute"), lay, sim.engine, xp), lay
    except EnvError:                   # a driven joint without limits: unbounded targets
        return ActionWriter(ActionSpec(mode="absolute", low=[-1e6] * n, high=[1e6] * n), lay, sim.engine, xp), lay


def targets(writer: Any, hold: dict[str, float], worlds: int, seed: int) -> tuple[np.ndarray, list[Any]]:
    """(A,) hold targets in the writer's column order (a column the task names no target for keeps its build-time
    target), and NOISE_BANK (W, A) target sets = hold + uniform noise (rad; metres on columns with under 0.2 of
    travel, as on the Isaac Lab side)."""
    cols = writer.column_names()
    start = writer.xp.to_numpy(writer.xp.view(writer.target)[writer.rows])[0]
    hold_np = np.array([hold.get(c, start[i]) for i, c in enumerate(cols)], dtype=np.float32)
    span = np.abs(writer.high_np - writer.low_np)
    amp = np.where(span < 0.2, S.NOISE["prismatic"], S.NOISE["revolute"]).astype(np.float32)
    rng = np.random.default_rng(seed)
    bank = [writer.xp.asarray(hold_np + amp * rng.uniform(-1.0, 1.0, (worlds, len(cols))).astype(np.float32))
            for _ in range(NOISE_BANK)]
    return hold_np, bank


def make_body(sim: Any, writer: Any, bank: list[Any]) -> Any:
    import warp as wp  # noqa: PLC0415

    cuda = str(sim.device).startswith("cuda")

    def body(k: int) -> None:
        if writer is not None:
            writer.apply(bank[k % NOISE_BANK])
        sim.step(1)
        if cuda:
            wp.synchronize_device(sim.device)

    return body


def hold_error(sim: Any, writer: Any, lay: Any, hold_np: np.ndarray) -> dict[str, Any]:
    """Largest and median (over worlds, of the per-world largest) |joint position - hold target| on driven joints
    (rad or m), and the joint that is furthest off. A column "joint[k]" is coordinate k of a multi-dof joint."""
    cols = []
    for i, c in enumerate(writer.column_names()):
        m = re.fullmatch(r"(.*)\[(\d+)\]", c)
        joint, k = (m.group(1), int(m.group(2))) if m else (lay.actuators[c].joint if c in lay.actuators else c, 0)
        if joint:
            cols.append((i, joint, k))
    st = sim.state(sorted({j for _, j, _ in cols}))
    err = np.stack([np.abs(np.asarray(st[f"{j}.q"], dtype=np.float64).reshape(sim.worlds, -1)[:, k] - hold_np[i])
                    for i, j, k in cols], axis=1)                       # (W, columns)
    worst = np.nanmax(err, axis=1)
    return {"max": float(np.nanmax(worst)), "median": float(np.nanmedian(worst)),
            "worst_joint": cols[int(np.nanargmax(np.nanmax(err, axis=0)))][1]}


def health(sim: Any) -> dict[str, Any]:
    h = sim.health()
    flags = collections.Counter(f for w in h["worlds"] for f in w["flags"])
    return {"n": int(sim.worlds), "invalid": len(h["invalid_worlds"]), "degraded": len(h["degraded_worlds"]),
            "invalid_worlds": sorted(int(w) for w in h["invalid_worlds"])[:50], "flags": dict(flags),
            "max_penetration_m": max((w.get("max_penetration_m") or 0.0) for w in h["worlds"]),
            "check_ok": bool(sim.check()["ok"])}


def run_speed(a: argparse.Namespace, body: Any) -> dict[str, Any]:
    t0 = time.perf_counter()
    for k in range(a.warmup):
        body(k)
    warm, walls, k = time.perf_counter() - t0, [], a.warmup
    for _ in range(a.blocks):
        t0 = time.perf_counter()
        for _ in range(a.steps):
            body(k)
            k += 1
        walls.append(time.perf_counter() - t0)
    med = sorted(walls)[len(walls) // 2]
    return {"env_steps_per_s": a.worlds * a.steps / med, "block_walls_s": sorted(walls), "warmup_wall_s": warm}


def read_watch(sim: Any, bodies: list[str], tracked: list[str]) -> tuple[np.ndarray, ...]:
    """(speed, zmin, finite, pos, quat_xyzw) for ``Watch.step`` from the public state."""
    st = sim.state(bodies)
    pos = np.stack([np.asarray(st[f"{b}.pos"], dtype=np.float64) for b in bodies], axis=1)          # (W, B, 3)
    vel = np.stack([np.asarray(st[f"{b}.vel"], dtype=np.float64)[:, :3] for b in bodies], axis=1)
    finite = np.isfinite(pos).all(axis=(1, 2)) & np.isfinite(vel).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        speed, zmin = np.linalg.norm(vel, axis=-1).max(axis=1), pos[..., 2].min(axis=1)
    tp = np.stack([pos[:, bodies.index(b)] for b in tracked], axis=1)
    tq = np.stack([np.asarray(st[f"{b}.quat_xyzw"], dtype=np.float64) for b in tracked], axis=1)
    return speed, zmin, finite, tp, tq


def run_quality(a: argparse.Namespace, sim: Any, body: Any, prep: Prepared, meta: dict[str, Any]) -> dict[str, Any]:
    tracked = prep.tracked
    bodies = sorted(set(sim.names()["bodies"]) | set(tracked))
    watch = S.Watch(a.worlds, len(tracked))
    watch.step(0, *read_watch(sim, bodies, tracked))
    for k in range(S.QUALITY_STEPS):
        body(k)
        watch.step(k + 1, *read_watch(sim, bodies, tracked))
    path = Path(a.out) / f"{a.tag}.npz"
    watch.save(path, meta)
    res = S.score(path, prep.isaac_task, task=prep.score)
    res.pop("meta")
    return res


def run(a: argparse.Namespace) -> dict[str, Any]:
    import alloy  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    prep = prepare(a.task, Path(a.out) / "models")
    t0 = time.perf_counter()
    sim = build(prep, a)
    writer, lay = make_writer(sim)
    hold_np, bank = targets(writer, prep.hold, a.worlds, a.seed) if writer is not None else (np.zeros(0), [])
    body = make_body(sim, writer, bank)
    meta = {"engine": "alloy", "task": prep.isaac_task, "alloy_task": prep.name, "note": prep.note,
            "source": prep.source,
            "num_envs": a.worlds, "mode": a.mode, "dt": S.DT, "step_mode": a.step, "seed": a.seed, "noise": S.NOISE,
            "create_s": time.perf_counter() - t0, "actuators": writer.size if writer is not None else 0,
            "bodies": len(sim.names()["bodies"]),
            "targets": f"alloy.envs ActionWriter ({writer.kind}, backend {writer.xp.name})" if writer is not None
            else "none: no driven joint (passive)",
            "route": next(iter(sim.routes().values())), "device": str(sim.device),
            "gpu": wp.get_device(sim.device).name, "versions": {"alloy": alloy.__version__, "warp-lang": wp.__version__},
            "alloy_sha": os.environ.get("ALLOY_SHA")}        # the driver passes the tree it synced
    s0 = sim.physics_steps
    if a.mode == "speed":
        meta["speed"] = run_speed(a, body)
        n = a.warmup + a.blocks * a.steps
    else:
        meta["quality"] = run_quality(a, sim, body, prep, dict(meta))
        n = S.QUALITY_STEPS
    meta["solver_steps_per_env_step"] = (sim.physics_steps - s0) / n
    meta["physics_dt_s"] = S.DT / meta["solver_steps_per_env_step"]
    meta["health"] = health(sim)
    # Alloy freezes an INVALID world at its last good state: a run of mostly frozen worlds is not a physics speed, and
    # the outside quality rule (pxs_score) sees those worlds as calm. Readers must use these two fields.
    meta["frozen_share"] = meta["health"]["invalid"] / meta["health"]["n"]
    meta["valid"] = meta["frozen_share"] <= FROZEN_MAX
    try:
        meta["hold_error"] = hold_error(sim, writer, lay, hold_np) if writer is not None else None
    except Exception as e:  # noqa: BLE001  a diagnostic must not lose the row (a multi-axis joint has no ".q" key)
        meta["hold_error"] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", help=f"one of {sorted(T.TASKS)} or usd:SPEC.json")
    ap.add_argument("--mode", choices=("speed", "quality"), default="speed")
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--step", choices=("default", "5ms"), default="default")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.tag = a.tag or f"{Path(a.task[4:]).stem.split('_mjwarp')[0] + '_usd' if a.task.startswith('usd:') else a.task}_{a.step}_{a.mode}_{a.worlds}"
    meta = run(a)
    (Path(a.out) / f"{a.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    print("PXS", json.dumps({k: meta.get(k) for k in ("task", "num_envs", "step_mode", "solver_steps_per_env_step",
                                                      "speed", "quality", "valid", "frozen_share", "health",
                                                      "hold_error")}, default=str))


if __name__ == "__main__":
    main()
