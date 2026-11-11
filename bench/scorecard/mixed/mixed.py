# SPDX-License-Identifier: Apache-2.0
"""Mixed-batch rows of the launch scorecard (rows 1 and 2): K task types stepped together on one GPU.

    PYTHONPATH=. python bench/scorecard/mixed/mixed.py --engine alloy --tasks franka_lift,ur10_reach,go2_flat \
        --envs 3072 --mode speed --out DIR [--form multi|one_scene] [--pxs DIR]

Skild's mixed protocol (skild.ai/blogs/steel): 5 / 10 / 15 / 20 task types in one batch, one 5 ms env step, small
noisy hold-position targets, no policy, observations, rewards, resets or cameras, a device sync each env step.
``--envs`` is the total, split evenly over the task types. It reuses the single-task harness unchanged (``tasks.py``, ``pxs_score.py``, ``alloy_hold.py``, ``newton_hold.py``; ``--pxs`` or ``$PXS_DIR``).

Engines:
- ``alloy``: Alloy as shipped (robots step at 1.25 ms: 4 physics steps per env step).
- ``alloy_5ms``: a PROBE, not a supported setting (``alloy_hold.py``: one 5 ms step).
- ``newton_mjw``: Newton's ``SolverMuJoCo`` alone on the same files (Alloy's pinned Newton copy unless
  ``--newton-module newton``), ``--substeps`` steps per env step, buffers from ``--buffers``.

Forms (how K types share the GPU; neither Alloy nor Newton's MuJoCo solver steps different worlds in one model):
- ``multi``: one simulation per task type, all in this process, stepped in turn, one device sync per env step.
- ``one_scene`` (alloy only): one scene holding all K robots ``--spacing`` apart; a world counts as K envs.

Modes: ``speed`` (env-steps/s = total envs x steps / median block wall, the peak GPU memory, and each type's share
of the step time) and ``quality`` (``pxs_score``'s S1 rule and contact held per type, on ``--envs`` worlds).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_PXS = HERE / "single_task"
NOISE_BANK = 16


@dataclass
class Member:
    """One task type inside a group: its name, robot, the bodies quality tracks and how they are scored."""

    name: str                      # the task as given on the command line (a tasks.py name or usd:SPEC.json)
    isaac_task: str                # the Isaac Lab task it stands for
    robot: str                     # its robot's name in the scene
    tracked: list[str]             # body names in the order of score["track"]
    score: dict[str, Any]          # the pxs_score task entry
    source: dict[str, Any]


def label(name: str) -> str:
    """A short type label for tables and file names: the task name, or the spec's task for usd:SPEC.json."""
    return Path(name[4:]).stem.split("_mjwarp")[0] + "_usd" if name.startswith("usd:") else name


@dataclass
class Group:
    """One simulation: K members (one for ``multi``), its step, and what quality reads."""

    members: list[Member]
    worlds: int
    launch: Callable[[int], None]                  # one env step, no device sync
    read: Callable[[Member], tuple[np.ndarray, ...]]
    info: dict[str, Any] = field(default_factory=dict)
    after: Callable[[], dict[str, Any]] = dict     # end-of-run facts (health, hold error)


def load_pxs(pxs: Path) -> tuple[Any, ...]:
    """The single-task harness modules (pxs_score, tasks, alloy_hold, newton_hold)."""
    if not (pxs / "tasks.py").is_file():
        raise SystemExit(f"single-task harness not found in {pxs}: pass --pxs DIR (tasks.py, pxs_score.py, ...)")
    sys.path.insert(0, str(pxs))
    import alloy_hold  # noqa: PLC0415
    import newton_hold  # noqa: PLC0415
    import pxs_score  # noqa: PLC0415
    import tasks  # noqa: PLC0415

    sys.path.insert(0, str(HERE))
    import extra_tasks  # noqa: PLC0415

    extra_tasks.register(tasks)
    return pxs_score, tasks, alloy_hold, newton_hold


def gpu_used_mb() -> float | None:
    """Memory in use on GPU 0 as nvidia-smi reports it (MiB), or None without one."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def memory(device: str) -> dict[str, Any]:
    import warp as wp  # noqa: PLC0415

    rec: dict[str, Any] = {"nvidia_smi_used_mb": gpu_used_mb()}
    if str(device).startswith("cuda"):
        rec["warp_pool_high_mb"] = wp.get_mempool_used_mem_high(device) / 2**20
        rec["warp_pool_now_mb"] = wp.get_mempool_used_mem_current(device) / 2**20
    return rec


# ---------------------------------------------------------------------------------------------- Alloy
def grid_pos(i: int, count: int, spacing: float) -> tuple[float, float, float]:
    """Robot ``i`` of ``count`` on a square grid centred on the origin (health flags objects over 4 m from it)."""
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    return ((i % cols - (cols - 1) / 2) * spacing, (i // cols - (rows - 1) / 2) * spacing, 0.0)


def alloy_group(mods: tuple[Any, ...], names: list[str], worlds: int, a: argparse.Namespace) -> Group:
    """One Alloy simulation holding ``names`` (one robot each, on a grid ``--spacing`` apart)."""
    import alloy  # noqa: PLC0415

    S, T, AH, _ = mods
    if a.engine == "alloy_5ms":
        import alloy.scene.rigid_step as rs  # noqa: PLC0415

        rs.RIGID_TARGET = S.DT                 # probe only (alloy_hold.py)
    scene = alloy.Scene(worlds=worlds, dt=S.DT)
    members, hold = [], {}
    if len(names) == 1:                  # one type: the single-task harness fills the scene (Menagerie or usd:)
        prep = AH.prepare(names[0], Path(a.out) / "models")
        prep.add(scene)
        hold = dict(prep.hold)
        members.append(Member(names[0], prep.isaac_task, "r", list(prep.tracked), prep.score, prep.source))
    for i, name in enumerate(names if len(names) > 1 else ()):
        if name.startswith("usd:"):
            raise SystemExit("--form one_scene takes tasks.py tasks only; run usd: tasks with --form multi")
        task = T.TASKS[name]
        path, task_hold, source = T.model(task, Path(a.out) / "models")
        robot = f"r{i}"
        scene.add_robot(robot, path, pos=grid_pos(i, len(names), a.spacing))
        hold |= {robot + key[1:]: v for key, v in task_hold.items()}       # "r/<actuator>" -> "<robot>/<actuator>"
        spec = T.score_spec(task)
        spec["track"] = tuple((lb, robot, body) for lb, _, body in spec["track"])
        members.append(Member(name, task.isaac_task, robot, [f"{robot}/{b}" for _, b in task.track], spec, source))
    sim = alloy.build(scene, device=a.device, seed=a.seed)
    writer, lay = AH.make_writer(sim)
    hold_np, bank = AH.targets(writer, hold, worlds, a.seed)

    def launch(k: int) -> None:
        writer.apply(bank[k % NOISE_BANK])
        sim.step(1)

    def read(m: Member) -> tuple[np.ndarray, ...]:
        bodies = [b for b in sim.names()["bodies"] if b.startswith(m.robot + "/")]
        return AH.read_watch(sim, bodies, m.tracked)

    s0 = sim.physics_steps
    info = {"route": next(iter(sim.routes().values())), "actuators": writer.size, "device": str(sim.device),
            "bodies": len(sim.names()["bodies"])}

    def after() -> dict[str, Any]:
        h = AH.health(sim)
        try:
            hold = AH.hold_error(sim, writer, lay, hold_np)
        except Exception as e:  # noqa: BLE001 - a diagnostic must not lose the row (a multi-axis joint has no ".q")
            hold = {"error": f"{type(e).__name__}: {e}"[:300]}
        # Alloy freezes an INVALID world at its last good state: frozen worlds cost little and look calm to the
        # outside quality rule, so a group with more than AH.FROZEN_MAX of them is not a valid speed (build.py).
        share = h["invalid"] / h["n"]
        return {"physics_steps": sim.physics_steps - s0, "health": h, "frozen_share": share,
                "valid": share <= AH.FROZEN_MAX, "hold_error": hold}

    return Group(members, worlds, launch, read, info, after)


# --------------------------------------------------------------------------------------------- Newton
def newton_group(mods: tuple[Any, ...], name: str, worlds: int, a: argparse.Namespace) -> Group:
    """Stock Newton SolverMuJoCo on one task's file; solver steps captured as CUDA graphs (as newton_hold.py)."""
    import warp as wp  # noqa: PLC0415

    S, T, _, NH = mods
    if name.startswith("usd:"):
        raise SystemExit("newton_mjw runs tasks.py tasks (MJCF) only; usd: tasks run on the Alloy engines")
    task = T.TASKS[name]
    path, _, source = T.model(task, Path(a.out) / "models")
    sized = (a.buffers or {}).get(name) or {}          # measured sizes (mjw_buffers.py); none = the solver's own
    newton, model, s0, s1, ctrl, solver = NH.build(path, worlds, a.device, a.newton_module, **sized)
    bank = NH.ctrl_bank(path, ctrl, worlds, a.seed)
    sub, dt, cuda = a.substeps, S.DT / a.substeps, str(a.device).startswith("cuda")

    def steps(x: Any, y: Any) -> None:
        for _ in range(sub):
            x.clear_forces()
            solver.step(x, y, ctrl, None, dt)
            x, y = y, x

    pairs = ((s0, s1), (s1, s0))[: 1 + sub % 2]
    graphs = []
    for x, y in pairs:
        steps(x, y)                                # eager: compiles, allocates
    for x, y in pairs if cuda else ():
        with wp.ScopedCapture() as cap:
            steps(x, y)
        graphs.append(cap.graph)
    done = [len(pairs)]                            # env steps taken so far (parity picks the current state)

    def launch(k: int) -> None:
        wp.copy(ctrl.mujoco.ctrl, bank[k % NOISE_BANK])
        if cuda:
            wp.capture_launch(graphs[done[0] % len(graphs)])
        else:
            steps(*pairs[done[0] % len(pairs)])
        done[0] += 1

    names = NH.body_names(model)[: model.body_count // worlds]
    tracked = [next(i for i, n in enumerate(names) if n.split("/")[-1] == b) for _, b in task.track]

    def read(_: Member) -> tuple[np.ndarray, ...]:
        return NH.read_watch((s0, s1)[done[0] % 2 if sub % 2 else 0], worlds, tracked)

    info = {"solver": f"{a.newton_module}.solvers.SolverMuJoCo(model), buffers "
                      + ("measured (mjw_buffers.py)" if sized else "at the solver's own sizing"),
            "newton_module": a.newton_module, "newton_version": newton.__version__, "bodies": len(names)}
    member = Member(name, task.isaac_task, "r", [f"r/{b}" for _, b in task.track], T.score_spec(task), source)
    return Group([member], worlds, launch, read, info,
                 lambda: {"physics_steps": (done[0] - len(pairs)) * sub, "buffers": NH.buffers(solver, worlds)})


# ------------------------------------------------------------------------------------------------ runs
def build_groups(mods: tuple[Any, ...], names: list[str], a: argparse.Namespace) -> list[Group]:
    per = a.envs // len(names)
    if a.engine == "newton_mjw":
        return [newton_group(mods, n, per, a) for n in names]
    if a.form == "one_scene":
        return [alloy_group(mods, names, per, a)]
    return [alloy_group(mods, [n], per, a) for n in names]


def run_speed(groups: list[Group], sync: Callable[[], None], a: argparse.Namespace, envs: int) -> dict[str, Any]:
    def env_step(k: int) -> None:
        for g in groups:
            g.launch(k)
        sync()

    t0 = time.perf_counter()
    for k in range(a.warmup):
        env_step(k)
    warm, walls, k = time.perf_counter() - t0, [], a.warmup
    for _ in range(a.blocks):
        t0 = time.perf_counter()
        for _ in range(a.steps):
            env_step(k)
            k += 1
        walls.append(time.perf_counter() - t0)
    med = sorted(walls)[len(walls) // 2]
    mem = memory(a.device)                                   # before the share block
    share = [0.0] * len(groups)                              # one more block, synced after each group
    for _ in range(a.steps):
        for i, g in enumerate(groups):
            t0 = time.perf_counter()
            g.launch(k)
            sync()
            share[i] += time.perf_counter() - t0
        k += 1
    return {"env_steps_per_s": envs * a.steps / med, "block_walls_s": sorted(walls), "warmup_wall_s": warm,
            "env_steps": k, "memory": mem,
            "group_wall_s": [{"tasks": [label(m.name) for m in g.members], "wall_s": s} for g, s in zip(groups, share)]}


def run_quality(mods: tuple[Any, ...], groups: list[Group], sync: Callable[[], None], a: argparse.Namespace,
                meta: dict[str, Any]) -> list[dict[str, Any]]:
    S = mods[0]
    watched = [(g, m, S.Watch(g.worlds, len(m.tracked))) for g in groups for m in g.members]
    for g, m, w in watched:
        w.step(0, *g.read(m))
    for k in range(S.QUALITY_STEPS):
        for g in groups:
            g.launch(k)
        sync()
        for g, m, w in watched:
            w.step(k + 1, *g.read(m))
    out = []
    for g, m, w in watched:
        path = Path(a.out) / f"{a.tag}.{label(m.name)}.npz"
        w.save(path, meta | {"task": m.isaac_task, "alloy_task": m.name, "num_envs": g.worlds})
        res = S.score(path, m.isaac_task, task=m.score)
        res.pop("meta", None)
        out.append({"task": label(m.name), "isaac_task": m.isaac_task, "worlds": g.worlds, "score": res})
    return out


def run(a: argparse.Namespace) -> dict[str, Any]:
    import warp as wp  # noqa: PLC0415

    mods = load_pxs(Path(a.pxs))
    names = a.tasks.split(",")
    a.device = a.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    cuda = a.device.startswith("cuda")
    envs = (a.envs // len(names)) * len(names)
    meta: dict[str, Any] = {
        "protocol": "hold-mixed", "engine": a.engine, "form": a.form if a.engine != "newton_mjw" else "multi",
        "tasks": names, "task_types": len(names), "num_envs": envs, "envs_per_type": a.envs // len(names),
        "mode": a.mode, "dt": mods[0].DT, "seed": a.seed, "noise": mods[0].NOISE, "device": a.device,
        "gpu": wp.get_device(a.device).name, "memory_before": memory(a.device), "versions": {"warp-lang": wp.__version__}}
    t0 = time.perf_counter()
    with wp.ScopedDevice(a.device):
        groups = build_groups(mods, names, a)
        meta |= {"create_s": time.perf_counter() - t0, "memory_built": memory(a.device),
                 "groups": [g.info | {"tasks": [label(m.name) for m in g.members], "worlds": g.worlds,
                                      "sources": [m.source for m in g.members]} for g in groups]}

        def sync() -> None:
            if cuda:
                wp.synchronize_device(a.device)

        if a.mode == "speed":
            meta["speed"] = run_speed(groups, sync, a, envs)
            n = meta["speed"]["env_steps"]
        else:
            meta["quality"] = run_quality(mods, groups, sync, a, dict(meta))
            n = mods[0].QUALITY_STEPS
        meta["memory_end"] = memory(a.device)
        ends = [g.after() for g in groups]
        meta["group_end"] = ends
        meta["solver_steps_per_env_step"] = [e["physics_steps"] / n for e in ends]
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", choices=("alloy", "alloy_5ms", "newton_mjw"), required=True)
    ap.add_argument("--tasks", required=True, help="comma-separated task names of tasks.py")
    ap.add_argument("--envs", type=int, required=True, help="total envs, split evenly over the task types")
    ap.add_argument("--form", choices=("multi", "one_scene"), default="multi")
    ap.add_argument("--mode", choices=("speed", "quality"), default="speed")
    ap.add_argument("--substeps", type=int, default=1, help="newton_mjw: solver steps per env step")
    ap.add_argument("--spacing", type=float, default=2.0, help="one_scene: metres between robots")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--newton-module", default="alloy._vendor.newton",
                    help='newton_mjw: the Newton to run; "newton" measures a stock install (newton_hold.py)')
    ap.add_argument("--buffers", default=os.environ.get("MJW_BUFFERS"),
                    help="newton_mjw: JSON of mjw_buffers.py with njmax / nconmax per task (default: Newton's own)")
    ap.add_argument("--pxs", default=os.environ.get("PXS_DIR", str(DEFAULT_PXS)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.buffers = {k: v["use"] for k, v in json.loads(Path(a.buffers).read_text())["tasks"].items()} if a.buffers else None
    Path(a.out).mkdir(parents=True, exist_ok=True)
    a.tag = a.tag or f"mix{len(a.tasks.split(','))}_{a.engine}_{a.form}_sub{a.substeps}_{a.mode}_{a.envs}"
    try:
        meta = run(a)
    except Exception as e:  # noqa: BLE001 - an out-of-memory build is a result: record it, then fail
        (Path(a.out) / f"{a.tag}.json").write_text(json.dumps(
            {"tag": a.tag, "args": vars(a), "error": f"{type(e).__name__}: {e}"[:4000], "memory_end": {"nvidia_smi_used_mb": gpu_used_mb()}},
            indent=1, default=str))
        raise
    (Path(a.out) / f"{a.tag}.json").write_text(json.dumps(meta | {"tag": a.tag}, indent=1, default=str))
    print("MIX", json.dumps({k: meta.get(k) for k in ("engine", "form", "tasks", "num_envs", "solver_steps_per_env_step",
                                                      "speed", "quality", "memory_end")}, default=str))


if __name__ == "__main__":
    main()
