# SPDX-License-Identifier: Apache-2.0
"""The CPU route's float64 MuJoCo state is solver state: a restored snapshot and a resumed checkpoint go on bit for bit.

G1 of WP0's C-route triage (notes/wp0-mjc-reds-triage-1113.md): the route kept MuJoCo C's float64 state outside the
state registries, so a restore pushed the float32 ``State`` back (float32 -> float64) and the run left the unbroken one;
seven tests were red on ALLOY_TEST_CPU_ROUTE=mjc only (test_rollback_bitwise x2, test_period_ladder, the dc_show
checkpoint, test_mpm_history_rollback, test_rollback_health, test_pad_finish).

Laws, on the two-joint arm dragging a block (contacts, joint friction, servos), two worlds:
- a snapshot (``State`` and the registered state arrays, as ``WorldSnapshot`` copies them) restored into the same solver
  re-runs the next steps bit for bit; ``State`` alone does not (the cause: float32 pushed into float64);
- the same into a new solver (a checkpoint resumed in another simulation) runs on bit for bit;
- a world edited after the restore is pushed from float32 as before, and the other world still re-runs bit for bit;
- a reset forgets the saved rows (the next run is a fresh start's, not the snapshot's);
- repeat runs are bitwise.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core.registry import isolated_registry, state_arrays_for, state_world_maps_for
from alloy.route.mjc_core import MjcCoreSolver
from test_mjc_core import Run, arm_model, route

pytestmark = pytest.mark.cpu_route

KEYS = ("body_q", "body_qd", "joint_q", "joint_qd")
WARM, AFTER = 40, 60


def _copy(run: Run) -> tuple[dict, dict]:
    state = {k: getattr(run.s0, k).numpy().copy() for k in KEYS}
    return state, {k: a.numpy().copy() for k, a in state_arrays_for(run.s).items()}


def _put(run: Run, snap: tuple[dict, dict], arrays: bool = True) -> None:
    state, extra = snap
    for k, v in state.items():
        getattr(run.s0, k).assign(v)
    if arrays:
        for k, a in state_arrays_for(run.s).items():
            a.assign(extra[k])


def _poses(run: Run) -> dict[str, np.ndarray]:
    return {k: getattr(run.s0, k).numpy().copy() for k in KEYS}


def _same(a: dict, b: dict) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in KEYS)


def _reference() -> tuple[Run, tuple[dict, dict], dict]:
    run = Run(m := arm_model(2), route(m))
    run.step(WARM)
    snap = _copy(run)
    run.step(AFTER)
    return run, snap, _poses(run)


def test_the_route_registers_its_float64_state() -> None:
    from alloy.coupling.adapters import register_builtin_adapters

    with isolated_registry():
        register_builtin_adapters()                               # as a build does: the MuJoCo solver's own arrays
        run = Run(m := arm_model(2), route(m))
        run.step(1)
        arrays, maps = state_arrays_for(run.s), state_world_maps_for(run.s)
    assert isinstance(run.s, MjcCoreSolver)
    assert {"mjc.state", "mjc.valid", "mjc.q", "mjc.qd"} <= set(arrays) and {"mjc.q", "mjc.qd"} <= set(maps)
    assert arrays["mjc.state"].shape[0] == 2 and arrays["mjc.state"].dtype == wp.float64
    assert any(k.startswith("mjw.") for k in arrays)              # the MuJoCo solver's own arrays stay


def test_a_restored_snapshot_reruns_bit_for_bit_and_state_alone_does_not() -> None:
    run, snap, ref = _reference()
    _put(run, snap)
    run.step(AFTER)
    assert _same(_poses(run), ref)
    _put(run, snap, arrays=False)                                 # the old behaviour: float32 pushed into float64
    run.step(AFTER)
    assert not _same(_poses(run), ref)


def test_a_checkpoint_resumes_bit_for_bit_in_a_new_solver() -> None:
    _, snap, ref = _reference()
    fresh = Run(m := arm_model(2), route(m))
    _put(fresh, snap)
    fresh.step(AFTER)
    assert _same(_poses(fresh), ref)


def test_an_edit_after_a_restore_is_pushed_and_the_other_world_reruns() -> None:
    run, snap, ref = _reference()
    _put(run, snap)
    q = run.s0.joint_q.numpy()
    world = run.s.views.coord_world
    q[np.flatnonzero(world == 1)[0]] += 1e-3                      # world 1's j1, nudged
    run.s0.joint_q.assign(q)
    run.step(AFTER)
    now = _poses(run)
    w0 = run.s.views.coord_world == 0
    assert np.array_equal(now["joint_q"][w0], ref["joint_q"][w0])
    assert not np.array_equal(now["joint_q"][~w0], ref["joint_q"][~w0])


def test_a_reset_forgets_the_saved_rows() -> None:
    m = arm_model(2)
    run = Run(m, route(m))
    run.step(WARM)
    run.s.reset(run.s0)
    assert not run.s.live.valid.numpy().any()
    run.step(AFTER)
    fresh = Run(m2 := arm_model(2), route(m2))
    fresh.step(AFTER)
    assert _same(_poses(run), _poses(fresh))


def test_repeat_runs_are_bitwise() -> None:
    assert _same(_reference()[2], _reference()[2])
