# SPDX-License-Identifier: Apache-2.0
"""Caches keyed by ``id()`` or a device address die with their owner (WP4, 2026-11-03).

``test_mjw_order_gaps::test_imported_contacts_are_canonical`` failed only inside the full CPU suite: an earlier test's
MuJoCo Warp ``Data`` was freed, CPython gave its id to the new ``Data``, and the sort scratch cached under that id (other
sizes) was reused ("Invalid indexing in slice"). The route's device-address caches (qLD groups, tree sizes, range-sum
scratch) had the same hazard with a freed address reused by a new model.
- ``remember`` drops an entry when its owner is freed, and keeps a later entry stored under the same key.
- The contact and row sort scratch of a ``Data`` is gone once that ``Data`` is freed.
- The qLD groups and tree caches of a model are gone once that model is freed.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import mujoco
from alloy._vendor import mujoco_warp as mjw
from alloy._vendor import newton
import pytest
import warp as wp

from alloy.precision import mjw_order, mjw_qld, mjw_rows, mjw_sums
from alloy.precision.mjw_det import set_mjw_determinism
from alloy.precision.owned_cache import remember

sys.path.insert(0, str(Path(__file__).parent))
from test_mjw_order_gaps import SORTS, _box_pile, _step  # noqa: E402
from test_mjw_qld import _chain_xml  # noqa: E402


class _Owner:
    pass


def test_entry_dies_with_its_owner() -> None:
    cache: dict = {}
    a, b = _Owner(), _Owner()
    remember(cache, "k", a, 1)
    assert cache == {"k": 1}
    remember(cache, "k", b, 2)        # a later entry under the same key (another view of a live array)
    del a
    gc.collect()
    assert cache == {"k": 2}          # a's death leaves b's entry alone
    del b
    gc.collect()
    assert cache == {}


@pytest.fixture
def hooks():
    mjw_order.install_order_hooks()
    yield
    mjw_order.uninstall_order_hooks()


def test_sort_scratch_dies_with_its_data(hooks) -> None:
    model = _box_pile(2)
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, njmax=100, nconmax=32, solver="newton")
    set_mjw_determinism(solver, SORTS)
    pipeline = newton.CollisionPipeline(model)
    _step(solver, model, 2, pipeline.contacts(), pipeline)
    key = id(solver.mjw_data)
    assert any(k[0] == key for k in mjw_order._SCRATCH) and key in mjw_rows._SCRATCH
    del solver, pipeline
    gc.collect()
    assert not any(k[0] == key for k in mjw_order._SCRATCH)   # a new Data given this id starts clean
    assert key not in mjw_rows._SCRATCH


def test_route_caches_die_with_their_model() -> None:
    mjm = mujoco.MjModel.from_xml_string(_chain_xml())
    with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mjm)
        d = mjw.put_data(mjm, mujoco.MjData(mjm), nworld=1)
    mjw_qld.prepare_qld(m)
    mjw_sums.prepare_tree(m, (d.subtree_com,))
    levels = [lv.ptr for lv in m.qLD_updates if lv.size]
    parent, snap, fused = m.body_parentid.ptr, d.subtree_com.ptr, m.qLD_all_updates.ptr
    assert levels and all(p in mjw_qld._GROUPS for p in levels) and fused in mjw_qld._SOLVE
    assert parent in mjw_sums._SUBTREE and snap in mjw_sums._SNAP
    del m, d
    gc.collect()
    assert not any(p in mjw_qld._GROUPS for p in levels) and fused not in mjw_qld._SOLVE  # a reused address starts clean
    assert parent not in mjw_sums._SUBTREE and parent not in mjw_sums._DEEPEST and snap not in mjw_sums._SNAP
