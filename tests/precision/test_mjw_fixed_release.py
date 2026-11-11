# SPDX-License-Identifier: Apache-2.0
"""A dropped routed MuJoCo Warp solver is freed with its Fixed64 scratch (WP8 2026-11-05T00:14Z, status/WP8.md).

``mjw_fixed._REG`` held each registered Model and Data strongly, so every routed simulation stayed alive after it
was dropped, and with it every ``remember()``-keyed scratch tied to its Data. A long-lived scorer process grew
its memory with every build until it ran out. The registry now keeps weak references, and an entry goes when
its Data is freed.
"""

from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

from alloy._vendor import mujoco_warp as mjw
import pytest
import warp as wp

from alloy.core.errors import PrecisionError
from alloy.precision import mjw_fixed, mjw_route

sys.path.insert(0, str(Path(__file__).parent))
from test_mjw_route import SolverMuJoCo, _xml  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_and_clean():
    with wp.ScopedDevice("cpu"):
        yield
    mjw_route.uninstall()


def _routed_refs() -> tuple[weakref.ref, weakref.ref, int]:
    """Build and step a routed solver, drop it, and return weak refs to its Model and Data with its registry key."""
    s = SolverMuJoCo(_xml())
    assert mjw_route.enable_route(s) is None
    for _ in range(3):
        mjw.step(s.mjw_model, s.mjw_data)
    assert mjw_fixed.overruns() == 0
    key = s.mjw_data.efc.J.ptr
    assert key in mjw_fixed._REG
    return weakref.ref(s.mjw_model), weakref.ref(s.mjw_data), key


def test_dropped_routed_solver_is_freed_with_its_scratch():
    m_ref, d_ref, key = _routed_refs()
    gc.collect()
    assert d_ref() is None, "Data still alive after its solver was dropped"
    assert m_ref() is None, "Model still alive after its solver was dropped"
    assert key not in mjw_fixed._REG


def test_rebuilds_do_not_accumulate_entries():
    for _ in range(3):
        _routed_refs()
        gc.collect()
    assert len(mjw_fixed._REG) == 0


def test_live_solver_keeps_its_entry_and_unregistered_data_fails_closed():
    s = SolverMuJoCo(_xml())
    assert mjw_route.enable_route(s) is None
    gc.collect()
    assert mjw_fixed.entry(s.mjw_data.efc.J) is not None
    other = SolverMuJoCo(_xml())
    with pytest.raises(PrecisionError, match="not registered"):
        mjw_fixed.entry(other.mjw_data.efc.J)
