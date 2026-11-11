# SPDX-License-Identifier: Apache-2.0
"""Explicit MuJoCo pair rows in MuJoCo's compiled order (alloy.contact.mjw_pairs; Newton pair misorder, WP0 ruling
2026-11-03): MuJoCo Warp's pair values equal MuJoCo's host model after the entry is built and after a later
notify(SHAPE_PROPERTIES); scenes with nothing to move stay bit-identical (CPU)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.mjw_pairs import newton_copy_map, pair_order, source_rows, use_mujoco_pair_order

FIELDS = ("pair_friction", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap")
# rows (shape a, shape b, mu, timeconst): MuJoCo groups the two (box0, box1) rows ahead of (box0, box2)
ROWS = ((0, 1, 0.1, 0.010), (0, 2, 0.2, 0.020), (1, 0, 0.3, 0.030))


def _boxes(rows=ROWS) -> newton.solvers.SolverMuJoCo:
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    shapes = []
    for x in (0.0, 0.1, 0.2):
        body = b.add_body(xform=wp.transform((x, 0.0, 0.5), wp.quat_identity()), mass=0.01)
        shapes.append(b.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01))
    for a, c, mu, tc in rows:     # values set by shape pair before the solver is built (as the socket-row writers do)
        b.add_custom_values(**{"mujoco:pair_world": b.current_world, "mujoco:pair_geom1": shapes[a],
                               "mujoco:pair_geom2": shapes[c], "mujoco:pair_friction": (mu, mu, 0.005, 1e-4, 1e-4),
                               "mujoco:pair_solref": (tc, 1.0)})
    main = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(main)
    main.replicate(b, 2)
    return newton.solvers.SolverMuJoCo(main.finalize(device="cpu"), use_mujoco_contacts=True)


def _device_equals_host(solver) -> bool:
    mj, wm = solver.mj_model, solver.mjw_model
    for f in FIELDS:
        host = np.asarray(getattr(mj, f), dtype=np.float64).reshape(mj.npair, -1)
        dev = getattr(wm, f).numpy().reshape(-1, mj.npair, host.shape[1]).astype(np.float64)
        if not np.allclose(dev, host[None], rtol=1e-6, atol=0.0):
            return False
    return True


def test_newton_still_copies_rows_by_index():
    """Tripwire: when this fails, Newton fills MuJoCo Warp's pair arrays in MuJoCo's order; drop mjw_pairs."""
    solver = _boxes()
    assert np.asarray(solver.mj_model.pair_friction[:, 0]).tolist() == pytest.approx([0.1, 0.3, 0.2])
    assert not _device_equals_host(solver), "Newton fixed the pair misorder: remove alloy.contact.mjw_pairs"


def test_device_equals_host_after_build_and_after_notify():
    solver = _boxes()
    assert pair_order(solver, 3).tolist() == [0, 2, 1]         # pair identity
    assert newton_copy_map(solver, 3).tolist() == [0, 1, 2]    # what Newton's update copies (by index)
    assert not _device_equals_host(solver)                     # the probe restored the as-built values
    assert use_mujoco_pair_order(solver) is None
    assert _device_equals_host(solver)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    assert _device_equals_host(solver)
    assert use_mujoco_pair_order(solver) == "Newton maps each pair to its own row"


def test_source_rows_follow_newtons_copy_map():
    """A Newton that copied rows through some other fixed map gets the rows that map needs (no index assumption)."""
    order, copy = np.array([0, 2, 1, 3]), np.array([3, 0, 1, 2])
    src = source_rows(copy, order)
    rows = np.arange(4)[src]                    # new row k holds old row src[k]
    assert [int(rows[copy[j]]) for j in range(4)] == order.tolist()
    assert source_rows(order, order).tolist() == [0, 1, 2, 3]


def test_rows_already_in_order_are_left_alone():
    solver = _boxes(rows=(ROWS[0], ROWS[2], ROWS[1]))     # grouped as MuJoCo groups them
    rows = {f: getattr(solver.model.mujoco, f).numpy().copy() for f in FIELDS}
    dev = {f: getattr(solver.mjw_model, f).numpy().copy() for f in FIELDS}
    assert use_mujoco_pair_order(solver) == "Newton maps each pair to its own row"
    for f in FIELDS:
        np.testing.assert_array_equal(getattr(solver.model.mujoco, f).numpy(), rows[f])
        np.testing.assert_array_equal(getattr(solver.mjw_model, f).numpy(), dev[f])
    assert _device_equals_host(solver)


def _mjw_arrays(wm) -> dict[str, np.ndarray]:
    out = {k: v.numpy().copy() for k, v in vars(wm).items() if isinstance(v, wp.array) and v.size}
    out.update({f"opt.{k}": v.numpy().copy() for k, v in vars(wm.opt).items() if isinstance(v, wp.array)})
    return out


@pytest.mark.compile_heavy
@pytest.mark.parametrize("scene", ["demo_peg_d12", "task_t12"])
def test_scenes_without_misplaced_values_stay_bit_identical(scene):
    from alloy.cards.library import CardLibrary
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary()
    built = SceneAssembler(lib).assemble(lib.scene(scene), "cpu", world_count=2)
    solver = newton.solvers.SolverMuJoCo(built.model, use_mujoco_contacts=True)
    assert _device_equals_host(solver)
    before = _mjw_arrays(solver.mjw_model)
    use_mujoco_pair_order(solver)
    after = _mjw_arrays(solver.mjw_model)
    changed = [k for k in before if not np.array_equal(before[k], after[k], equal_nan=True)]
    assert changed == []
