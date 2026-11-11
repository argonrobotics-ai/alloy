# SPDX-License-Identifier: Apache-2.0
"""Empty self-contact launches skipped on the CPU (alloy.solvers.vbd.base._size_self_contact_launches, S6): with no
self-contact pair detected, Newton's per-colour self-contact accumulation and planar truncation launch at size 0, and
the result is bitwise the same."""

from __future__ import annotations

import numpy as np
import pytest

import alloy


def _vbd(sim):
    return [s for s in sim.engine.solvers if hasattr(s, "_sc_launch_full")][0]


def _flat_or_stacked(stack: bool):
    s = alloy.Scene(dt=0.01, cloth_accuracy="default")
    s.add_ground()
    s.add_cloth("c", size=(0.2, 0.2), pos=(0.0, 0.0, 0.001), resolution=10, self_contact=True)
    if stack:     # a second sheet dropped onto the first: no pair while it falls (3 steps), pairs once it lands
        s.add_cloth("top", size=(0.14, 0.14), pos=(0.0, 0.0, 0.012), resolution=8, self_contact=True)
    return s


@pytest.mark.parametrize("stack", [False, True], ids=["flat", "stacked"])
def test_skipping_empty_self_contact_launches_is_bitwise(stack):
    """With no self-contact pair detected, Newton's per-colour self-contact accumulation and planar truncation add
    nothing; launching them at size 0 (base._size_self_contact_launches, S6) gives the same bits, on a flat sheet
    (never a pair) as on two stacked sheets (no pair while the top one falls, pairs once it lands)."""
    out, sizes, full = [], [], None
    for skip in (True, False):
        sim = alloy.build(_flat_or_stacked(stack), device="cpu")
        vbd = _vbd(sim)
        if skip:
            full = vbd._sc_launch_full
        else:
            vbd._sc_launch_full = None                 # the stock launches
        seen = set()
        for _ in range(10):                            # a step with pairs costs ~1 s on the CPU
            sim.step(1)
            seen.add(vbd.particle_self_contact_evaluation_kernel_launch_size)
        st = sim.state(["c", "top"] if stack else ["c"], world=0)
        out.append(np.concatenate([st[k] for k in sorted(st) if k.endswith(".points")]))
        sizes.append(seen)
        sim.close()
    assert np.array_equal(out[0], out[1])
    assert 0 in sizes[0] and 0 not in sizes[1]          # the skip fired; the stock run never shrank
    if stack:
        assert full in sizes[0]                         # pairs formed: the skip run went back to the full size
