# SPDX-License-Identifier: Apache-2.0
"""Law: a full-material render frees what it built when its sim closes. Build, sim.render(quality="full") and close,
three times in one process: no warp Mesh stays alive (the Bevel node's mesh of all static triangles was kept in a
module list, about 47 MiB per redraw frame on an L4), and the live warp arrays stop growing after the first cycle's
one-time tables. Only the objects the renders made count: the other tests in the worker leave meshes that can be
freed meanwhile (62 alive before the renders, 52 after, in a post-merge run). CPU.
"""
from __future__ import annotations

import gc
import weakref

import warp as wp

import alloy


def warp_objects() -> list:
    """Every warp Mesh and array alive after a full collection."""
    gc.collect()
    return [o for o in gc.get_objects() if isinstance(o, (wp.Mesh, wp.array))]


def live(before: weakref.WeakSet) -> tuple[int, int]:
    """(warp Meshes, warp arrays) alive that were not in ``before``: made since it was taken."""
    new = [o for o in warp_objects() if o not in before]
    return sum(isinstance(o, wp.Mesh) for o in new), sum(isinstance(o, wp.array) for o in new)


def build_render_close() -> None:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0.0, 0.0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.6), mass=0.2)
    sim = alloy.build(scene, device="cpu")
    sim.render(width=32, height=24, spp=1, quality="full")
    sim.close()


def test_three_full_renders_leave_no_mesh_alive():
    before = weakref.WeakSet(warp_objects())     # weak: what the other tests keep stays theirs to free
    counts = []
    for _ in range(3):
        build_render_close()
        counts.append(live(before))
    assert [m for m, _ in counts] == [0] * 3, counts
    assert counts[2][1] == counts[0][1], counts      # arrays: the first cycle's tables, nothing per cycle


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-q"])
