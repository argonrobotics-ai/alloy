# SPDX-License-Identifier: Apache-2.0
"""P3-2 mesh-SDF contact laws A8.push and A9.thread_rundown: port packs, scorer and Alloy beside (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/sdf_pack.py pack [bench/xbench/ours/packs/laws] [--laws A8.push,...]
    PYTHONPATH=. python bench/xbench/ours/sdf_pack.py score RUN_DIR [--device cuda:0]
    PYTHONPATH=. python bench/xbench/ours/sdf_pack.py alloy RUN_DIR [--device cuda:0]       # Alloy beside

Pack ``packs/laws/<law>/<condition>/``, written from the model Alloy's runner builds (``sdf_scene``; each condition's
worlds in plan order, A8: 16 perturbed worlds, A9: one):

- ``meshes/*.obj``: the collision geometry in body frames, static parts in the world frame (``sdf_geom`` docstring);
- ``cond.json`` (schema ``alloy.xbench.ours.sdf/1``):
  - ``bodies``: explicit mass, COM, inertia;
  - ``joints``: per DOF axis, armature, damping, friction, limits, drive mode / gains / effort limit;
  - ``shapes`` and ``contact_pairs``: OBJ, body, per-shape mu as Alloy holds it, the pair mu Alloy's MuJoCo uses
    (max-mix); only the listed pair collides;
  - ``gravity``, ``dt``, ``steps``, ``record_every``, ``worlds``, ``samples`` and ``initial``: per world, Newton body
    q = x y z qx qy qz qw and COM qd, and joint q / qd, the exact float32 values Alloy starts from;
  - ``drive``: what is driven and its time profile;
  - ``alloy_collision``: how Alloy collides, as information only (``engine_collision``: engines use their own
    mesh-SDF contact at their documented recommended settings);
  - ``export``: the npz channels the scorer needs;
  - ``checks``: the static check, recomputed from the written files;
- ``drive.npz``: the twin's own drive per step k at t = k dt (A8 ``body_f``; A9 ``joint_f``, ``target_qd``).

Engine export ``RUN_DIR/<law>/<condition>.npz``: the README contract npz with a frame every physics step, frame 0
the initial state, and W = the condition's ``worlds``. A9 adds ``joint_q`` (z and theta of the nut, optional) and
``drive_tau`` (required: the torque the spin drive applied, i.e. the run-down torque the law judges), see
``sdf_scene.EXPORT``.

Score: the law's own twin, built as the runner builds it, replays the engine's frames, then ``scalars`` +
``judge_analytic`` with the LawSpec tolerances write ``RUN_DIR/<law>.scored.json``.
- A8: ``make_pipeline`` gives Alloy's layer and GapMonitor, then the twin's ``record`` on every frame. The monitor
  restarts at the ramp step, counted in time at the engine's dt. No narrow phase runs: contact rows only widen
  GapMonitor's p99 population, never pen_max_m. Prefer ``--device cuda:0``; ~19k samples per world per frame is slow
  on CPU.
- A9: ``record`` on the nut's joint coordinates, its ``tau`` channel taken from ``drive_tau``, then
  ``thread_metrics``. ``joint_q`` is checked against the poses. A mismatch is a ``port_error``, never a verdict.

Alloy beside: ``step_twin`` as ``run_law`` runs it, with ``record`` wrapped to keep every frame (A9 also the twin's
own z / theta / tau per step) -> the same npz plus ``RUN_DIR/<law>.alloy_direct.json``. A8 needs CUDA: Newton bakes
its texture SDFs on CUDA only, so on CPU it is ``not_run``, as Alloy's own law reports it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

import sdf_scene as S  # noqa: E402
from sdf_scene import LAWS, PUSH, THREAD  # noqa: E402

log = logging.getLogger("xbench.ours")
JOINT_TOL = {"z_m": 1.0e-6, "theta_rad": 1.0e-4}      # A9 port check: exported joint_q against the nut's poses
SUPPORT_LOST_M = 1.0e-3     # A8 guard: sink past 20x the 50 um pen bound = the peg left its support


def support_guard(monitors: dict) -> dict | None:
    """A8: pen_max_m only sees pairs GapMonitor still samples, so a peg that falls through its support scores 0.
    The guard fails a run whose peg sank more than ``SUPPORT_LOST_M`` (Alloy's own runs sink ~75 um)."""
    sink = max(monitors.get("sink_m") or [0.0])
    if sink <= SUPPORT_LOST_M:
        return None
    return {"name": "support_lost", "sink_m": sink, "bound_m": SUPPORT_LOST_M,
            "why": "the peg sank past its support, so GapMonitor sampled no pair and pen_max_m cannot judge it"}


def _status(rows: list[dict]) -> str:
    return "pass" if all(r["verdict"] == "pass" for r in rows) else "fail"


# ---------------------------------------------------------------- Alloy beside --------------------------------------

class _Capture:
    """Wraps ``twin.record``: body_q / body_qd after every step. For A9 the twin's own record also writes z, theta,
    tau into a stride-1 log every step (A9's record only writes its log; A8's updates GapMonitor, so it runs once)."""

    def __init__(self, twin, n_worlds: int, n_steps: int, dt: float, device: str, extras: bool) -> None:
        from alloy.bench.twin import DeviceTwinLog

        self.q: list[np.ndarray] = []
        self.qd: list[np.ndarray] = []
        self.log = DeviceTwinLog(n_worlds, n_steps, dt, device) if extras else None
        inner = twin.record

        def record(step, state, contacts, obs, tlog):
            self.q.append(state.body_q.numpy().copy())
            self.qd.append(state.body_qd.numpy().copy())
            if self.log is not None:
                inner(step, state, contacts, obs, self.log)
            inner(step, state, contacts, obs, tlog)

        twin.record = record

    def arrays(self) -> dict[str, np.ndarray]:
        """``body_q`` / ``body_qd`` [T, bodies, .]; A9 ``joint_q`` [T, W, 2] and ``drive_tau`` [T, W] (float64)."""
        out = {"body_q": np.stack(self.q), "body_qd": np.stack(self.qd)}
        if self.log is not None:
            z, th, tau = (self.log.series(k).T.astype(np.float64) for k in ("z", "theta", "tau"))
            out["joint_q"], out["drive_tau"] = np.stack([z, th], axis=-1), tau
        return out


def _write_npz(d: Path, worlds, model, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """One contract npz per condition (W axis = that condition's worlds in plan order)."""
    from score_laws import _world_bodies

    d.mkdir(parents=True, exist_ok=True)
    by_world = _world_bodies(model)
    groups: dict[str, list[int]] = {}
    for g, w in enumerate(worlds):
        groups.setdefault(w.condition.id, []).append(g)
    for cid, gs in groups.items():
        idx = np.array([[b for b, _ in by_world[g]] for g in gs])
        out = {"body_q": arrays["body_q"][:, idx].astype(np.float32),
               "body_qd": arrays["body_qd"][:, idx].astype(np.float32),
               **{k: arrays[k][:, gs] for k in ("joint_q", "drive_tau") if k in arrays}}
        np.savez_compressed(d / f"{cid}.npz", meta=np.array(json.dumps(dict(meta, condition=cid, worlds=len(gs)))),
                            body_names=np.array([n for _, n in by_world[gs[0]]]),
                            t=np.arange(out["body_q"].shape[0]) * meta["dt"], **out)


def alloy_law(law_id: str, out_dir: Path, device: str, duration_s: float | None = None) -> dict:
    """Alloy beside: the law's direct run and its frames as engine npz (``duration_s`` shortens it, tests only)."""
    from alloy.bench.runner import RunContext, judge_analytic, plan_worlds, step_twin
    from alloy.contact.law_device import CUDA_REASON, is_cuda
    from alloy_t12 import versions
    from score_laws import _verdict_rows

    if law_id == PUSH and not is_cuda(device):
        return {"law": law_id, "status": "not_run", "reason": CUDA_REASON}
    spec, twin = S.twin_of(law_id, duration_s)
    ctx = RunContext(spec, device, 0, "mujoco", False)
    worlds = plan_worlds(spec, twin.conditions(False), ctx, getattr(twin, "expand", None))
    n_steps = max(1, int(round(spec.duration_s / spec.dt)))
    cap = _Capture(twin, len(worlds), n_steps, spec.dt, device, extras=law_id == THREAD)
    t0 = time.perf_counter()
    run = step_twin(spec, twin, worlds, ctx)
    wall = time.perf_counter() - t0
    rows = _verdict_rows(judge_analytic(spec, twin, twin.scalars(run.log), {}, [w.condition for w in worlds]))
    meta = {"engine": "alloy", "engine_version": versions()["alloy"], "versions": versions(), "test": law_id,
            "dt": spec.dt, "substeps": spec.substeps, "record_every": 1, "row": "recommended", "wall_s": wall,
            "duration_s": spec.duration_s, "settings": twin.settings_record("mujoco"),
            "notes": "Alloy's direct run (alloy.bench.runner.step_twin), every condition one model"}
    _write_npz(out_dir / law_id, worlds, run.model, cap.arrays(), meta)
    return {"law": law_id, "status": _status(rows), "metrics": rows, "wall_s": wall, "duration_s": spec.duration_s,
            "monitors": json.loads(json.dumps(getattr(twin, "monitors", {}), default=str))}


# ---------------------------------------------------------------- scorer --------------------------------------------

def _replay_push(b: S.Built, q: np.ndarray, dt: float, tlog) -> dict:
    """A8: Alloy's layer + GapMonitor (``make_pipeline``, never collided) and the twin's record on every frame."""
    b.twin.make_pipeline(b.model, "mujoco")
    b.twin._ramp_step = int(round(b.twin.plan.settle_s / dt))     # the ramp starts at a time, not at a frame index
    state = b.model.state()
    for k in range(q.shape[0]):
        state.body_q.assign(q[k])
        b.twin.record(k, state, None, None, tlog)
    return {"replay": "GapMonitor on the engine's poses; no narrow phase, so p99 / samples count penetrating "
                      "samples only (pen_max_m is unaffected)"}


def _world_channel(runs: dict[str, dict], worlds, key: str) -> np.ndarray | None:
    """Extra channel ``key`` as [T, W_total, ...] in plan order; None if a condition lacks it."""
    seen: dict[str, int] = {}
    cols = []
    for w in worlds:
        cid = w.condition.id
        j = seen[cid] = seen.get(cid, -1) + 1
        if key not in runs[cid]:
            return None
        cols.append(np.asarray(runs[cid][key], np.float64)[:, j])
    return np.stack(cols, axis=1)


def _pose_joints(model, q: np.ndarray, qd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint q / qd of every frame from the poses (Newton ``eval_ik``), [T, coords] and [T, dofs]."""
    from alloy._vendor import newton

    st, jq, jqd = model.state(), [], []
    for k in range(q.shape[0]):
        st.body_q.assign(q[k])
        st.body_qd.assign(qd[k])
        newton.eval_ik(model, st, st.joint_q, st.joint_qd)
        jq.append(st.joint_q.numpy().astype(np.float64))
        jqd.append(st.joint_qd.numpy().copy())
    return np.stack(jq), np.stack(jqd)


def _thread_joints(b: S.Built, runs: dict[str, dict], q: np.ndarray, qd: np.ndarray) -> tuple:
    """A9: (joint q [T, coords], joint qd [T, dofs], port check): z / theta from ``joint_q`` when exported, checked
    against the poses (theta unwrapped over frames), else from the poses."""
    jq, jqd = _pose_joints(b.model, q, qd)
    sl = b.twin._drive_in.q_slide.numpy()
    jq[:, sl + 1] = np.unwrap(jq[:, sl + 1], axis=0)
    ex = _world_channel(runs, b.worlds, "joint_q")
    if ex is None:
        return jq, jqd, {"joint_q": "from body_q (eval_ik, theta unwrapped)", "ok": True}
    dz, dth = float(np.abs(ex[..., 0] - jq[:, sl]).max()), float(np.abs(ex[..., 1] - jq[:, sl + 1]).max())
    jq[:, sl], jq[:, sl + 1] = ex[..., 0], ex[..., 1]
    ok = dz <= JOINT_TOL["z_m"] and dth <= JOINT_TOL["theta_rad"]
    return jq, jqd, {"joint_q": "npz", "max_dz_m": dz, "max_dtheta_rad": dth, "tol": JOINT_TOL, "ok": ok}


def _replay_thread(b: S.Built, runs: dict[str, dict], q: np.ndarray, qd: np.ndarray, tlog) -> dict:
    """A9: the twin's record on the nut's joint coordinates every frame; its torque channel from ``drive_tau``."""
    tau = _world_channel(runs, b.worlds, "drive_tau")
    if tau is None:
        return {"error": "drive_tau missing: A9 needs the spin drive's applied torque (pack export)"}
    jq, jqd, check = _thread_joints(b, runs, q, qd)
    state = b.model.state()
    for k in range(q.shape[0]):
        state.body_q.assign(q[k])
        state.body_qd.assign(qd[k])
        state.joint_q.assign(jq[k].astype(np.float32))
        state.joint_qd.assign(jqd[k])
        b.twin.record(k, state, None, None, tlog)
    cols = [k for k in range(q.shape[0]) if tlog.column(k) >= 0]
    tlog.channel("tau").assign(np.ascontiguousarray(tau[cols].T.astype(np.float32)))
    return {"port_check": check}


def score_law(law_id: str, run_dir: Path, device: str = "cpu") -> dict:
    """Verdicts of one law from the engine's per-condition npz (``not_run`` if a condition is missing)."""
    from alloy.bench.runner import judge_analytic
    from alloy.bench.twin import DeviceTwinLog
    from score_laws import _frames, _load, _verdict_rows

    _, twin = S.twin_of(law_id)
    paths = {c.id: run_dir / law_id / f"{c.id}.npz" for c in twin.conditions(False)}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        return {"law": law_id, "status": "not_run", "missing": missing}
    runs = {cid: _load(p) for cid, p in paths.items()}
    b = S.build(law_id, device)
    q, qd = _frames(runs, b.worlds, b.model)
    first = next(iter(runs.values()))
    dt, n = float(first["meta"]["dt"]) * int(first["meta"].get("record_every", 1)), q.shape[0] - 1
    tlog = DeviceTwinLog(len(b.worlds), n, dt, device, stride=b.twin.record_stride(n))
    extra = _replay_push(b, q, dt, tlog) if law_id == PUSH else _replay_thread(b, runs, q, qd, tlog)
    if "error" in extra:
        return {"law": law_id, "status": "error", **extra}
    rows = _verdict_rows(judge_analytic(b.spec, b.twin, b.twin.scalars(tlog), {}, [w.condition for w in b.worlds]))
    ok = extra.get("port_check", {}).get("ok", True)
    finite = bool(np.isfinite(q).all() and np.isfinite(qd).all())
    monitors = json.loads(json.dumps(getattr(b.twin, "monitors", {}), default=str))
    guard = support_guard(monitors) if law_id == PUSH else None
    status = "port_error" if not ok else "fail" if guard else _status(rows)
    return {"law": law_id, "status": status, "metrics": rows, "dt": dt, "guard": guard,
            "law_dt": b.spec.dt, "frames": int(q.shape[0]), "finite": finite, "engine": first["meta"].get("engine"),
            "settings": first["meta"].get("settings"),
            "monitors": monitors, **extra}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pack", "score", "alloy"))
    p.add_argument("path", type=Path, nargs="?", default=S.OUT)
    p.add_argument("--laws", default=",".join(LAWS))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    a.path.mkdir(parents=True, exist_ok=True)
    for law_id in a.laws.split(","):
        if a.cmd == "pack":
            S.write_law(law_id, a.path)
            continue
        if a.cmd == "alloy":
            res = alloy_law(law_id, a.path, a.device)
            (a.path / f"{law_id}.alloy_direct.json").write_text(json.dumps(res, indent=1, default=str))
        else:
            try:
                res = score_law(law_id, a.path, a.device)
            except Exception as exc:  # noqa: BLE001 - a broken run is a row, not a crash of the whole scoring
                res = {"law": law_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            (a.path / f"{law_id}.scored.json").write_text(json.dumps(res, indent=1, default=str))
        log.info("%s %s: %s", a.cmd, law_id, res["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
