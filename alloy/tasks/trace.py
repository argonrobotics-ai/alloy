# SPDX-License-Identifier: Apache-2.0
"""Recorded traces as WP0 ``Trace`` objects, name-keyed so they replay on any model that contains the robot.

On disk (``bench/traces/<name>/``): ``trace.npz`` (inputs, initial state, reference poses) and ``trace.json``
(control_dt, columns, source, hashes, window). One input row is one recorded solve, so ``control_dt`` is the
recorded dt (8.333e-4 s or 1e-4 s) and the D4 grid holds with j = 0. Input keys:

- ``mujoco.ctrl`` [T, nu], columns = actuator names (MJCF order; Newton ``ctrl_direct=True``);
- ``joint_f`` [T, k], columns = hinge/slide joints with a non-zero recorded ``qfrc_applied``;
- ``mujoco.jnt_actfrcrange`` [T, k, 2], columns = joints whose force clip changes during the window.

``Trace.initial`` holds ``qpos``/``qvel``/``mujoco.qacc_warmstart`` in MuJoCo order and ``Trace.columns`` names the
joint of each element (SCR-WP13-1). :class:`TaskTrace` adds the recorded object poses the replay is scored against.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from alloy.core.ids import digest_of, sha256_hex
from alloy.parallel.types import Trace
from alloy.core.errors import TraceError
from alloy.tasks.mjb_convert import RecordedTrace
from alloy.tasks.mjmodel_info import (
    actuator_names,
    body_radius,
    element_columns,
    free_bodies,
    joint_names,
    joint_width,
    recorded_multiccd,
    split_state0,
)

log = logging.getLogger("alloy.tasks")

CTRL, JOINT_F, ACTFRC = "mujoco.ctrl", "joint_f", "mujoco.jnt_actfrcrange"
QPOS, QVEL, WARM = "qpos", "qvel", "mujoco.qacc_warmstart"
TRACE_NPZ, TRACE_JSON = "trace.npz", "trace.json"


@dataclass(frozen=True, eq=False)
class RefPoses:
    """Recorded object poses: ``pos`` [S, n, 3], ``quat`` [S, n, 4] xyzw, sampled at ``times`` [S] (s from start)."""

    times: np.ndarray
    names: tuple[str, ...]
    pos: np.ndarray
    quat: np.ndarray
    radius: np.ndarray                      # [n] m, body origin to the farthest collision bounding sphere


@dataclass(frozen=True, eq=False)
class TaskTrace:
    trace: Trace
    ref: RefPoses
    meta: dict = field(default_factory=dict)


def _forced_joints(rec: RecordedTrace, qfrc: np.ndarray) -> list[int]:
    """Joints with a non-zero recorded ``qfrc_applied``; only single-dof joints can carry one."""
    m, out = rec.model, []
    for k in range(m.njnt):
        da, nd = int(m.jnt_dofadr[k]), joint_width(m, k)[1]
        if not np.any(qfrc[:, da : da + nd] != 0.0):
            continue
        if nd > 1:
            raise TraceError(f"{rec.name}: applied forces on free/ball joints are not converted")
        out.append(k)
    return out


def _inputs(rec: RecordedTrace, rows: int) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]]]:
    m, names = rec.model, joint_names(rec.model)
    qfrc, afr = rec.inputs["qfrc"][:rows], rec.inputs["afr"][:rows].reshape(rows, m.njnt, 2)
    forced = _forced_joints(rec, qfrc)
    varying = [k for k in range(m.njnt) if np.any(afr[:, k] != afr[0, k])]
    inputs = {CTRL: rec.inputs["ctrl"][:rows].copy(), JOINT_F: qfrc[:, [int(m.jnt_dofadr[k]) for k in forced]],
              ACTFRC: afr[:, varying].copy()}  # fmt: skip
    columns = {CTRL: actuator_names(m), JOINT_F: tuple(names[k] for k in forced),
               ACTFRC: tuple(names[k] for k in varying)}  # fmt: skip
    return inputs, columns


def _initial(rec: RecordedTrace) -> dict[str, np.ndarray]:
    parts = split_state0(rec.model, rec.state0)
    return {QPOS: parts["qpos"].copy(), QVEL: parts["qvel"].copy(), WARM: parts["qacc_warmstart"].copy()}


def ref_poses(rec: RecordedTrace, samples: int) -> RefPoses:
    """Object poses from ``ref_qpos`` (free-joint bodies), quaternions converted to xyzw."""
    objs = free_bodies(rec.model)
    q = rec.ref_qpos[:samples]
    cols = np.array([qa for _, qa, _ in objs], dtype=np.int64)
    pos = q[:, cols[:, None] + np.arange(3)]
    quat = q[:, cols[:, None] + np.array([4, 5, 6, 3])]
    times = (np.arange(len(q)) + 1) * rec.ref_every * rec.dt
    radius = np.array([body_radius(rec.model, b) for b, _, _ in objs])
    return RefPoses(times, tuple(n for _, _, n in objs), pos, quat, radius)


def inputs_digest(inputs: dict[str, np.ndarray], columns: dict[str, tuple[str, ...]]) -> str:
    return digest_of({"inputs": {k: np.ascontiguousarray(v) for k, v in inputs.items()}, "columns": columns})


def build_trace(rec: RecordedTrace, scene_hash: str, seconds: float | None = None) -> TaskTrace:
    """Offline: turn a recorded window into a name-keyed :class:`TaskTrace` (``seconds`` crops it)."""
    rows = rec.steps if seconds is None else min(rec.steps, int(round(seconds / rec.dt)))
    rows -= rows % rec.ref_every
    inputs, columns = _inputs(rec, rows)
    columns |= {QPOS: element_columns(rec.model, 0), QVEL: element_columns(rec.model, 1),
                WARM: element_columns(rec.model, 1)}  # fmt: skip
    src = f"[internal reference]/{rec.name} (mjb -> MJCF -> SceneCard), rows 0..{rows - 1}"
    trace = Trace(scene_hash, rec.dt, inputs, src, inputs_digest(inputs, columns), columns, _initial(rec))
    meta = {"t0": float(rec.state0[0]), "recorded_rows": rec.steps, "ref_every": rec.ref_every,
            "mujoco_multiccd": recorded_multiccd(rec.model), "recording": rec.meta}  # fmt: skip
    return TaskTrace(trace, ref_poses(rec, rows // rec.ref_every), meta)


def save_trace(out_dir: Path, tt: TaskTrace) -> None:
    """Write ``trace.npz`` + ``trace.json`` (the json records the npz sha256)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {f"in/{k}": v for k, v in tt.trace.inputs.items()} | {f"init/{k}": v for k, v in tt.trace.initial.items()}
    r = tt.ref
    arrays |= {"ref/times": r.times, "ref/pos": r.pos, "ref/quat": r.quat, "ref/radius": r.radius}
    np.savez_compressed(out_dir / TRACE_NPZ, **arrays)
    meta = {"scene_hash": tt.trace.scene_hash, "control_dt": tt.trace.control_dt, "source": tt.trace.source,
            "sha256": tt.trace.sha256, "columns": {k: list(v) for k, v in tt.trace.columns.items()},
            "ref_names": list(r.names), "npz_sha256": sha256_hex((out_dir / TRACE_NPZ).read_bytes()),
            "meta": tt.meta}  # fmt: skip
    (out_dir / TRACE_JSON).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    log.info("saved trace %s: %d periods, %s", out_dir, tt.trace.periods, tt.trace.sha256[:12])


def load_trace(trace_dir: Path) -> TaskTrace:
    """Read a saved trace; raises :class:`TraceError` if the npz or the inputs digest changed."""
    meta = json.loads((trace_dir / TRACE_JSON).read_text())
    if sha256_hex((trace_dir / TRACE_NPZ).read_bytes()) != meta["npz_sha256"]:
        raise TraceError(f"{trace_dir}: trace.npz sha256 differs from trace.json")
    with np.load(trace_dir / TRACE_NPZ) as z:
        inputs = {k[3:]: z[k] for k in z.files if k.startswith("in/")}
        initial = {k[5:]: z[k] for k in z.files if k.startswith("init/")}
        ref = RefPoses(z["ref/times"], tuple(meta["ref_names"]), z["ref/pos"], z["ref/quat"], z["ref/radius"])
    columns = {k: tuple(v) for k, v in meta["columns"].items()}
    if inputs_digest(inputs, columns) != meta["sha256"]:
        raise TraceError(f"{trace_dir}: inputs digest differs from trace.json")
    trace = Trace(meta["scene_hash"], meta["control_dt"], inputs, meta["source"], meta["sha256"], columns, initial)
    return TaskTrace(trace, ref, meta["meta"])
