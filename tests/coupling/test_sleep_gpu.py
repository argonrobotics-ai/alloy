# SPDX-License-Identifier: Apache-2.0
"""CPL-8 (SPEC §5.2, reported): MuJoCo sleeping at 1,024 worlds through ``mujoco_sleep_kwargs``
(native_contacts="mujoco"): per-step speedup vs the same scene awake (CUDA-graph timing, findings/timestep.md §4.4
method), for all-resting worlds and for one object kept awake per world (awake1, the realistic batched case), plus
the end-state difference of the all-resting case."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverMuJoCo

from alloy.coupling.sleep import mujoco_sleep_kwargs
from alloy.coupling.types import SleepPolicy
from alloy.testing.coupling import record_metric, resting_boxes_model

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
WORLDS, PER_WORLD = 1024, 12
DT = 1.25e-3
SETTLE, WINDOW, REPEATS = 400, 200, 5
AWAKE_BODY = 7          # a floor box with nothing stacked on it: kept awake by mujoco:sleep_policy NEVER (awake1)
NCONMAX, NJMAX = 256, 1024   # per world; 12 resting boxes need ~152 contacts (an overflow invalidates the run)


def _solver(model, sleeping):
    nv = model.joint_dof_count // model.world_count
    kw = mujoco_sleep_kwargs(SleepPolicy(), nv=nv, native_contacts="mujoco", device=DEV) if sleeping else {}
    return SolverMuJoCo(model, use_mujoco_contacts=True, njmax=NJMAX, nconmax=NCONMAX, **kw), kw


def _run(sleeping, awake_one):
    model = resting_boxes_model(WORLDS, PER_WORLD, device=DEV, never_sleep=(AWAKE_BODY,) if awake_one else ())
    solver, kw = _solver(model, sleeping)
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    q_start = s0.body_q.numpy()[:, :3].copy()

    def step():
        solver.step(s0, s1, control, None, DT)
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)
        wp.copy(s0.joint_q, s1.joint_q)
        wp.copy(s0.joint_qd, s1.joint_qd)
    for _ in range(SETTLE):
        step()
    with wp.ScopedCapture(device=DEV) as cap:
        step()
    times = []
    for _ in range(REPEATS):
        wp.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(WINDOW):
            wp.capture_launch(cap.graph)
        wp.synchronize_device(DEV)
        times.append((time.perf_counter() - t0) / WINDOW)
    data = solver.mjw_data
    awake_w = data.nbody_awake.numpy().copy() if sleeping else None
    awake = float(np.mean(awake_w)) if sleeping else None
    overflow = int(data.overflow.numpy().max())
    assert overflow == 0, f"MuJoCo contact/row overflow ({overflow}): the timing would measure dropped contacts"
    return np.array(times), s0.body_q.numpy()[:, :3].copy(), kw, awake, q_start, awake_w


def _world_stats(dq: np.ndarray) -> dict[str, float]:
    """Per-world max body displacement norm [m] -> p50 / p99 / max and the share of worlds above 0.5 mm."""
    per_world = np.linalg.norm(dq, axis=1).reshape(WORLDS, PER_WORLD).max(axis=1)
    return {"p50": float(np.median(per_world)), "p99": float(np.percentile(per_world, 99)),
            "max": float(per_world.max()), "share_above_0.5mm": float(np.mean(per_world > 5.0e-4))}


def _nonfinite(q: np.ndarray, awake_w: np.ndarray | None) -> dict[str, Any] | None:
    """Where a row's end positions are non-finite (an intermittent on rc-1109, 11-09: 85-96 of 1,024 "on" worlds; WP5 +
    WP11 + WP4): the worlds, each one's first non-finite body and count, its awake bodies, and how the worlds fall in
    32- and 64-world blocks (a kernel race clusters in blocks; a per-world cause scatters). None when all finite."""
    bad = ~np.isfinite(q.reshape(WORLDS, PER_WORLD, 3)).all(axis=2)
    worlds = np.flatnonzero(bad.any(axis=1))
    if not worlds.size:
        return None
    return {"count": int(worlds.size), "worlds": worlds.tolist(), "first_body": bad[worlds].argmax(axis=1).tolist(),
            "bodies": bad[worlds].sum(axis=1).tolist(),
            "awake_bodies": None if awake_w is None else awake_w[worlds].tolist(),
            "mod32": np.bincount(worlds % 32, minlength=32).tolist(),
            "mod64": np.bincount(worlds % 64, minlength=64).tolist(),
            "per_block32": np.bincount(worlds // 32, minlength=WORLDS // 32).tolist()}


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_cpl8_sleeping_speedup_and_end_state():
    rows = {}
    for name, sleeping, one in (("off", False, False), ("on", True, False), ("off_awake1", False, True),
                                ("on_awake1", True, True)):
        rows[name] = _run(sleeping, one)
        t = rows[name][0]
        print(f"{name}: {np.median(t) * 1e3:.3f} ms/step (min {t.min() * 1e3:.3f}, max {t.max() * 1e3:.3f}); "
              f"kwargs {rows[name][2]}; mean awake bodies/world {rows[name][3]}", flush=True)
    speed_rest = float(np.median(rows["off"][0]) / np.median(rows["on"][0]))
    speed_one = float(np.median(rows["off_awake1"][0]) / np.median(rows["on_awake1"][0]))
    diff = float(np.max(np.abs(rows["off"][1] - rows["on"][1])))
    end_diff = _world_stats(rows["off"][1] - rows["on"][1])
    drift = {k: _world_stats(rows[k][1] - rows[k][4]) for k in ("off", "on")}
    bad = {k: r for k, r in ((k, _nonfinite(v[1], v[5])) for k, v in rows.items()) if r is not None}
    record_metric("CPL-8", worlds=WORLDS, bodies_per_world=PER_WORLD, speedup_all_resting=speed_rest,
                  speedup_awake1=speed_one, end_state_max_diff_m=diff, device=DEV, timing="CUDA graph, 5x200 steps",
                  ms={k: [float(x * 1e3) for x in v[0]] for k, v in rows.items()},
                  kwargs=rows["on"][2], awake_bodies_per_world={k: v[3] for k, v in rows.items()},
                  scene="12 boxes/world, 8 on a plane + 4 stacked (T12-like layout)",
                  end_state_per_world=end_diff, drift_from_start_per_world=drift, nonfinite=bad)
    print(f"CPL-8: speedup all resting {speed_rest:.2f}x, awake1 {speed_one:.2f}x; end state max diff "
          f"{diff * 1e3:.4f} mm; per world {end_diff}; drift from start {drift}")
    if bad:
        print(f"CPL-8 non-finite end positions: {bad}", flush=True)
    assert np.all(np.isfinite(rows["on"][1])), f"non-finite 'on' end positions: {bad.get('on')}"
