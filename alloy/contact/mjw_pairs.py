# SPDX-License-Identifier: Apache-2.0
"""Explicit MuJoCo pair rows in MuJoCo's compiled order, so MuJoCo Warp steps each pair with its own values.

Newton bug (newton@ac7f7830, mujoco 3.16): MuJoCo's compiler reorders explicit pairs (pairs between the same two
bodies end up together) and keeps each pair's values with it, so the host ``mj_model`` is right. Newton fills
MuJoCo Warp's pair arrays (friction, solref, solreffriction, solimp, margin, gap) with
``update_pair_properties_kernel``, at construction and on every ``notify_model_changed(SHAPE_PROPERTIES)``, by copying
Newton row ``world * pairs_per_world + j`` onto MuJoCo pair ``j``: it assumes the two orders are equal. Where they
differ, a pair steps with another pair's values. On main (CPU, 1 world; WP0 ruling 2026-11-03):
- task_dc_show_service 4,373 of 8,755 pairs (worst μ error 0.475);
- task_sc2_lid 1,160 of 4,573 (0.45);
- task_dc_conn_sfp 2,076 of 4,377 (0.34);
- task_sc2_conn 483, task_dc_conn_c13 500, task_dc_eth_swap 480, task_dc_conn_qsfp 316, task_dc_rack_dimm 130,
  task_dc_rack_server 64 (solref only), task_t11_cards 15 of 22, task_t12_cards 14 of 36;
- T09, T11, T12, T14, H1, T02 and the demos: none.

`use_mujoco_pair_order` does not assume the bug: it matches each MuJoCo pair's (geom1, geom2) back to its Newton row
through ``mjc_geom_to_newton_shape`` (pair identity), then probes which Newton row Newton's own update actually
copies onto each MuJoCo pair (a marker per row in ``pair_gap``, one republish, read back, rows restored). Only when
the two disagree does it permute the Newton pair rows (``model.mujoco.pair_*``, ``model.alloy.pair_*``) so that
Newton's copy delivers each pair its own row, and republish. On a Newton that maps pairs by identity (upstream fix,
`[internal note]`) it is a no-op. Writers that pick Newton rows by shape pair
(dc_rack_dimm_checks socket rows, dc_show_ziptie.stiffen_edges) stay correct; MuJoCo-side writers (dc_show_snip)
index MuJoCo pairs. `tests/contact/test_mjw_pairs.py::test_newton_still_copies_rows_by_index` fails when Newton stops
misordering: drop this module then.
"""

from __future__ import annotations

import numpy as np

from alloy.core.log import get_logger

log = get_logger("contact.mjw_pairs")


def _template_shapes(solver) -> np.ndarray | None:
    to_shape = getattr(solver, "mjc_geom_to_newton_shape", None)
    if to_shape is None:
        return None
    to_shape = to_shape.numpy()
    return to_shape[0] if to_shape.ndim == 2 else to_shape


def pair_order(solver, per_world: int) -> np.ndarray | None:
    """For each MuJoCo pair, the Newton row (within a world block) with the same two shapes; None when they do not
    match one-to-one. Rows naming the same shape pair twice match in order."""
    to_shape = _template_shapes(solver)
    if to_shape is None:
        return None
    attrs, mj = solver.model.mujoco, solver.mj_model
    g1, g2 = attrs.pair_geom1.numpy()[:per_world], attrs.pair_geom2.numpy()[:per_world]
    rows: dict[tuple[int, int], list[int]] = {}
    for i, (a, c) in enumerate(zip(g1.tolist(), g2.tolist())):
        rows.setdefault((min(a, c), max(a, c)), []).append(i)
    order = []
    for a, c in zip(to_shape[np.asarray(mj.pair_geom1)].tolist(), to_shape[np.asarray(mj.pair_geom2)].tolist()):
        hit = rows.get((min(a, c), max(a, c)))
        if not hit:
            return None
        order.append(hit.pop(0))
    return np.asarray(order, dtype=np.int64)


def newton_copy_map(solver, per_world: int) -> np.ndarray | None:
    """For each MuJoCo pair, the Newton row (within a world block) that Newton's own pair update copies onto it,
    probed with a marker per row in ``pair_gap``; the rows and MuJoCo Warp are restored. None when the update does
    not reach MuJoCo Warp's pair_gap."""
    from alloy._vendor import newton  # noqa: PLC0415

    gap = getattr(solver.model.mujoco, "pair_gap", None)
    if gap is None:
        return None
    rows, dev = gap.numpy().copy(), solver.mjw_model.pair_gap.numpy().copy()
    worlds = int(solver.model.world_count)
    gap.assign(np.tile(np.arange(per_world, dtype=rows.dtype) + 1.0, worlds))
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    seen = np.rint(solver.mjw_model.pair_gap.numpy().reshape(-1, per_world)[0]).astype(np.int64) - 1
    gap.assign(rows)
    solver.mjw_model.pair_gap.assign(dev)
    if seen.min() < 0 or seen.max() >= per_world:
        return None
    return seen


def source_rows(copy: np.ndarray, order: np.ndarray) -> np.ndarray:
    """New row k takes old row src[k] so that Newton's copy (row copy[j] onto pair j) delivers pair j its own row
    order[j] (pure)."""
    src = np.empty(len(order), dtype=np.int64)
    src[copy] = order
    return src


def align_pair_rows(model, per_world: int, src: np.ndarray) -> int:
    """Rearrange every ``model.<ns>.pair_*`` row array (ns: mujoco, alloy) within each world block so that new row
    k holds old row ``src[k]``; returns the arrays rearranged."""
    worlds, n = int(model.world_count), 0
    for ns in (getattr(model, "mujoco", None), getattr(model, "alloy", None)):
        for name, arr in (vars(ns).items() if ns is not None else ()):
            if not name.startswith("pair_") or not hasattr(arr, "numpy") or arr.shape[0] != worlds * per_world:
                continue
            a = arr.numpy()
            arr.assign(np.ascontiguousarray(a.reshape(worlds, per_world, *a.shape[1:])[:, src].reshape(a.shape)))
            n += 1
    return n


def use_mujoco_pair_order(solver) -> str | None:
    """Make Newton's pair update deliver each MuJoCo pair its own Newton row; returns why nothing changed (no pairs,
    MuJoCo C entry, rows that do not match, Newton already maps by identity), None when rows were rearranged."""
    from alloy._vendor import newton  # noqa: PLC0415

    model, mj = getattr(solver, "model", None), getattr(solver, "mj_model", None)
    attrs = getattr(model, "mujoco", None)
    if mj is None or int(mj.npair) == 0 or getattr(attrs, "pair_geom1", None) is None:
        return "no explicit pairs"
    if getattr(solver, "use_mujoco_cpu", False):
        return "MuJoCo C entry: the host model keeps each pair's values"
    per_world, npair = len(attrs.pair_geom1) // max(1, int(model.world_count)), int(mj.npair)
    order = pair_order(solver, per_world) if per_world == npair else None
    copy = newton_copy_map(solver, per_world) if order is not None else None
    if order is None or copy is None or len(np.unique(copy)) != npair:
        log.warning("mjw_pairs: %d Newton pair rows per world do not map one-to-one onto %d MuJoCo pairs; left as built",
                    per_world, npair)
        return "rows do not map one-to-one"
    if np.array_equal(copy, order):
        return "Newton maps each pair to its own row"
    align_pair_rows(model, per_world, source_rows(copy, order))
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    log.info("mjw_pairs: %d of %d pair rows rearranged so Newton's copy matches MuJoCo's pairs; republished",
             int(np.sum(copy != order)), npair)
    return None


__all__ = ["align_pair_rows", "newton_copy_map", "pair_order", "source_rows", "use_mujoco_pair_order"]
