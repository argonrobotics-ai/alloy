# SPDX-License-Identifier: Apache-2.0
"""Bridge rows of the hold-task bench: Newton's SolverMuJoCo (MuJoCo Warp) on the Alloy rows' own scene files.

    python bench/scorecard/mixed/single_task/newton_hold.py TASK --mode speed --worlds 4096 --substeps 1 --out DIR
        [--newton-module alloy._vendor.newton] [--njmax N] [--nconmax N]

The Alloy rows (``alloy_hold.py``) run MuJoCo Menagerie files (``tasks.py``). This runs those same files through
Newton alone: ``ModelBuilder.add_mjcf`` + ``replicate`` + ``SolverMuJoCo`` (the file's own solver options, MuJoCo's
contacts), with none of Alloy's routing, contact settings or health checks, so the model difference and the engine
difference can be told apart. ``--newton-module`` names the Newton to run: by default the copy pinned inside Alloy
(``alloy._vendor.newton``: it carries Alloy's patches, and importing Alloy wraps ``SolverMuJoCo.__init__`` for its
coordinate carry), so rows from it are "Alloy's Newton copy", not stock Newton; pass ``newton`` to measure a stock
install. The module and its version are recorded in every row.

``--njmax`` / ``--nconmax`` size MuJoCo Warp's constraint and contact buffers per world (default: the solver's own
sizing from the start state). A robot that makes more contacts later overflows that sizing: constraints are dropped
and a warning is printed every step. Each row records the buffer sizes and whether the last step filled them
(``buffers.full_at_end``); size them with ``bench/scorecard/mixed/mjw_buffers.py``.

One env step = write the hold targets + noise into ``control.mujoco.ctrl`` (the same 16 noise sets as
``alloy_hold.py``), ``--substeps`` solver steps of 5 ms / substeps (1 = one 5 ms step; 4 = Alloy's 1.25 ms step), a
device sync. The steps run as captured CUDA graphs. Modes, timing and the quality watch are those of
``alloy_hold.py`` / ``pxs_score.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pxs_score as S  # noqa: E402
import tasks as T  # noqa: E402

NOISE_BANK = 16


DEFAULT_NEWTON = "alloy._vendor.newton"       # the copy pinned inside Alloy; "newton" = a stock install


def build(path: Path, worlds: int, device: str, module: str = DEFAULT_NEWTON, njmax: int | None = None,
          nconmax: int | None = None) -> tuple[Any, ...]:
    """(newton, model, state 0, state 1, control, solver) for ``worlds`` copies of the MJCF. ``njmax`` / ``nconmax``
    per world; None keeps the solver's own sizing."""
    import importlib  # noqa: PLC0415

    newton = importlib.import_module(module)
    sized = {k: v for k, v in (("njmax", njmax), ("nconmax", nconmax)) if v is not None}

    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(path))
    main = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(main)
    main.replicate(b, worlds)
    model = main.finalize(device=device)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    return newton, model, s0, s1, ctrl, newton.solvers.SolverMuJoCo(model, **sized)


def ctrl_bank(path: Path, ctrl: Any, worlds: int, seed: int) -> list[Any]:
    """NOISE_BANK device arrays shaped like ``control.mujoco.ctrl``: the baked keyframe's ctrl + uniform noise (rad;
    metres on actuators with under 0.2 of control range), drawn as ``alloy_hold.targets`` draws them."""
    import mujoco  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    m = mujoco.MjModel.from_xml_path(str(path))
    span = np.abs(m.actuator_ctrlrange[:, 1] - m.actuator_ctrlrange[:, 0])
    amp = np.where(span < 0.2, S.NOISE["prismatic"], S.NOISE["revolute"]).astype(np.float32)
    rng, hold, target = np.random.default_rng(seed), m.key_ctrl[0].astype(np.float32), ctrl.mujoco.ctrl
    bank = []
    for _ in range(NOISE_BANK):
        vals = np.clip(hold + amp * rng.uniform(-1.0, 1.0, (worlds, m.nu)).astype(np.float32),
                       m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]).astype(np.float32)
        bank.append(wp.array(vals.reshape(target.shape), dtype=target.dtype, device=target.device))
    return bank


def make_body(solver: Any, s0: Any, s1: Any, ctrl: Any, bank: list[Any], substeps: int, device: str) -> Any:
    """The env-step body. Solver steps are captured as graphs: one of ``substeps`` steps s0 -> ... (and, for an odd
    count, its mirror s1 -> ..., used on alternate env steps, so every graph keeps its buffers)."""
    import warp as wp  # noqa: PLC0415

    dt, cuda = S.DT / substeps, str(device).startswith("cuda")

    def steps(a: Any, b: Any) -> None:
        for _ in range(substeps):
            a.clear_forces()
            solver.step(a, b, ctrl, None, dt)
            a, b = b, a

    steps(s0, s1)                                        # eager first call: compiles, allocates
    if substeps % 2:
        steps(s1, s0)
    graphs: list[Any] = []
    if cuda:
        for a, b in ((s0, s1), (s1, s0))[:1 + substeps % 2]:
            with wp.ScopedCapture() as cap:
                steps(a, b)
            graphs.append(cap.graph)

    def body(k: int) -> None:
        wp.copy(ctrl.mujoco.ctrl, bank[k % NOISE_BANK])
        if cuda:
            wp.capture_launch(graphs[k % len(graphs)])
            wp.synchronize_device(device)
        else:
            steps(*((s0, s1) if k % 2 == 0 or substeps % 2 == 0 else (s1, s0)))

    return body


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


def buffers(solver: Any, worlds: int) -> dict[str, Any]:
    """MuJoCo Warp's buffer sizes per world, their use at the last step, and whether that step filled one (a full
    buffer means constraints or contacts were dropped)."""
    d = solver.mjw_data
    njmax, nconmax = int(d.njmax), int(d.naconmax) // worlds
    nefc, ncon = int(d.nefc.numpy().max()), int(d.nacon.numpy().max()) / worlds
    return {"njmax": njmax, "nconmax": nconmax, "nefc_max_at_end": nefc, "contacts_per_world_at_end": ncon,
            "full_at_end": nefc >= njmax or ncon >= nconmax}


def body_names(model: Any) -> list[str]:
    names = getattr(model, "body_label", None) or getattr(model, "body_key", None)
    return [str(n) for n in names]


def read_watch(state: Any, worlds: int, tracked: list[int]) -> tuple[np.ndarray, ...]:
    q = state.body_q.numpy().reshape(worlds, -1, 7).astype(np.float64)
    v = state.body_qd.numpy().reshape(worlds, -1, 6).astype(np.float64)[..., :3]
    finite = np.isfinite(q).all(axis=(1, 2)) & np.isfinite(v).all(axis=(1, 2))
    with np.errstate(invalid="ignore"):
        speed, zmin = np.linalg.norm(v, axis=-1).max(axis=1), q[..., 2].min(axis=1)
    return speed, zmin, finite, q[:, tracked, :3], q[:, tracked, 3:7]


def run_quality(a: argparse.Namespace, model: Any, states: tuple[Any, Any], body: Any, task: T.HoldTask,
                meta: dict[str, Any]) -> dict[str, Any]:
    names = body_names(model)[: model.body_count // a.worlds]
    tracked = [next(i for i, n in enumerate(names) if n.split("/")[-1] == b) for _, b in task.track]
    flips = a.substeps % 2                               # odd substeps: the current state alternates
    watch = S.Watch(a.worlds, len(tracked))
    watch.step(0, *read_watch(states[0], a.worlds, tracked))
    for k in range(S.QUALITY_STEPS):
        body(k)
        watch.step(k + 1, *read_watch(states[(k + 1) % 2 if flips else 0], a.worlds, tracked))
    path = Path(a.out) / f"{a.tag}.npz"
    watch.save(path, meta)
    res = S.score(path, task.isaac_task, task=T.score_spec(task))
    res.pop("meta")
    return res


def run(a: argparse.Namespace) -> dict[str, Any]:
    import warp as wp  # noqa: PLC0415

    task, out = T.TASKS[a.task], Path(a.out)
    path, _, source = T.model(task, out / "models")
    device = a.device or ("cuda:0" if wp.is_cuda_available() else "cpu")
    t0 = time.perf_counter()
    with wp.ScopedDevice(device):
        newton, model, s0, s1, ctrl, solver = build(path, a.worlds, device, a.newton_module, a.njmax, a.nconmax)
        bank = ctrl_bank(path, ctrl, a.worlds, a.seed)
        body = make_body(solver, s0, s1, ctrl, bank, a.substeps, device)
        meta = {"engine": "newton_mjw", "task": task.isaac_task, "alloy_task": task.name, "note": task.note,
                "source": source, "num_envs": a.worlds, "mode": a.mode, "dt": S.DT, "seed": a.seed, "noise": S.NOISE,
                "solver_steps_per_env_step": a.substeps, "physics_dt_s": S.DT / a.substeps,
                "create_s": time.perf_counter() - t0, "device": device, "gpu": wp.get_device(device).name,
                "solver": f"{a.newton_module}.solvers.SolverMuJoCo(model), buffers "
                          + ("as given" if a.njmax or a.nconmax else "at the solver's own sizing"),
                "newton_module": a.newton_module,
                "versions": {"newton": newton.__version__, "warp-lang": wp.__version__}}
        if a.mode == "speed":
            meta["speed"] = run_speed(a, body)
        else:
            meta["quality"] = run_quality(a, model, (s0, s1), body, task, dict(meta))
        meta["buffers"] = buffers(solver, a.worlds)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task", choices=sorted(T.TASKS))
    ap.add_argument("--mode", choices=("speed", "quality"), default="speed")
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--substeps", type=int, default=1)
    ap.add_argument("--newton-module", default=DEFAULT_NEWTON, help='"newton" measures a stock install')
    ap.add_argument("--njmax", type=int, default=None, help="constraint rows per world (default: the solver's sizing)")
    ap.add_argument("--nconmax", type=int, default=None, help="contacts per world (default: the solver's sizing)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    a.tag = a.tag or f"{a.task}_sub{a.substeps}_{a.mode}_{a.worlds}"
    meta = run(a)
    (Path(a.out) / f"{a.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    print("PXS", json.dumps({k: meta.get(k) for k in ("task", "num_envs", "solver_steps_per_env_step", "speed",
                                                      "quality", "buffers", "newton_module")}, default=str))


if __name__ == "__main__":
    main()
