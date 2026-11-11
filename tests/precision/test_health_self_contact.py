# SPDX-License-Identifier: Apache-2.0
"""SELF_CONTACT_OVERFLOW (2026-11-08): health sees VBD self-contact pairs dropped
over a row's buffer (``docs/known_gaps/vbd-self-contact-buffer-overflow.md``), in the world that dropped them only.

Law: two cloth sheets lie stacked, two worlds. After build, world 1's vertex and edge rows are shrunk to one pair
each, so its detector finds more pairs than it can keep (the real overflow, at a size a test can reach: the default
rows hold 32 and 64). World 0 keeps the default rows as the control. Health must flag world 1 alone, DEGRADED with
check() ok, and its dropped count must equal a host recount of count - size over the rows after each step."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy._vendor import newton
from alloy.integration.stepper import entry_solvers
from alloy.precision import self_contact
from alloy.precision.types import HealthFlags


def _rows(v) -> list[tuple[str, str, np.ndarray]]:
    pw = v.model.particle_world.numpy()
    return [("vertex_colliding_triangles_count", "vertex_colliding_triangles_buffer_sizes", pw),
            ("edge_colliding_edges_count", "edge_colliding_edges_buffer_sizes", pw[v.model.edge_indices.numpy()[:, 2]])]


def _host_dropped(v, rows) -> list[int]:
    ci, out = v.trimesh_collision_info, [0, 0]
    for count, size, world in rows:
        over = getattr(ci, count).numpy() - getattr(ci, size).numpy()
        for w in (0, 1):
            out[w] = max(out[w], int(over[world == w].max()))
    return out


def test_only_the_world_that_dropped_self_contact_pairs_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOY_SELF_CONTACT_ROWS", "fixed")    # rows sized to the scene would grow the shrunk rows back
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_cloth("a", size=(0.2, 0.2), pos=(0, 0, 0.003), resolution=10, mass=0.02)
    scene.add_cloth("b", size=(0.2, 0.2), pos=(0, 0, 0.0065), resolution=10, mass=0.02)
    sim = alloy.build(scene, device="cpu")
    [v] = [s for s in entry_solvers(sim.engine.stepper).values() if isinstance(s, newton.solvers.SolverVBD)]
    assert self_contact.watch(v) is getattr(v, self_contact.WATCH_ATTR)        # health attached it at build
    rows = _rows(v)
    for _, size, world in rows:                                               # world 1: one pair per row
        sizes = getattr(v.trimesh_collision_info, size)
        sizes.assign(np.where(world == 1, np.minimum(sizes.numpy(), 1), sizes.numpy()).astype(np.int32))
    host = [0, 0]
    for _ in range(3):
        sim.step(1)
        host = [max(a, b) for a, b in zip(host, _host_dropped(v, rows))]
    assert host[0] == 0 and host[1] > 0                                       # the shrink really drops pairs
    assert self_contact.watch(v).dropped.numpy().tolist() == host
    h = sim.health()
    assert h["worlds"][0]["severity"] == "OK" and "SELF_CONTACT_OVERFLOW" not in h["worlds"][0]["flags"]
    assert h["worlds"][1]["severity"] == "DEGRADED" and h["worlds"][1]["flags"] == ["SELF_CONTACT_OVERFLOW"]
    assert "self_contact=False" in h["worlds"][1]["fixes"][0]
    assert h["degraded_worlds"] == [1] and sim.check()["ok"]


@pytest.mark.parametrize("row_world, expect", [((0, 1, 1), [3, 1]), ((-1, 0, 0), [3, 3]), ((0, 0, 0), [3, 0])])
def test_the_kernel_keeps_the_most_pairs_one_row_dropped_per_world(row_world, expect) -> None:
    """Rows (count, size): (5, 2) drops 3, (4, 4) none, (9, 8) drops 1; a global row (world -1) blames every world."""
    arr = lambda xs: wp.array(np.asarray(xs, dtype=np.int32), dtype=wp.int32, device="cpu")  # noqa: E731
    dropped = wp.zeros(2, dtype=wp.int32, device="cpu")
    wp.launch(self_contact._dropped_pairs, dim=3, inputs=[arr([5, 4, 9]), arr([2, 4, 8]), arr(row_world), 2],
              outputs=[dropped], device="cpu")
    assert dropped.numpy().tolist() == expect



def test_a_reset_clears_the_masked_worlds_only() -> None:
    """A reset starts a new run (WP5's review): ``reset(mask)`` zeroes the masked worlds, ``reset()`` all."""
    sc = object.__new__(self_contact.SelfContactWatch)
    sc.device, sc.dropped = "cpu", wp.array(np.array([3, 5, 7], dtype=np.int32), dtype=wp.int32, device="cpu")
    sc.reset(wp.array(np.array([0, 1, 0], dtype=np.int32), dtype=wp.int32, device="cpu"))
    assert sc.dropped.numpy().tolist() == [3, 0, 7]
    sc.reset()
    assert sc.dropped.numpy().tolist() == [0, 0, 0]

def test_the_flag_is_degraded_and_listed_everywhere_a_flag_is() -> None:
    from alloy.api.health import REMEDIES
    from alloy.precision.ladder import LADDER
    from alloy.precision.types import Severity

    assert HealthFlags.SELF_CONTACT_OVERFLOW == 1 << 22
    [row] = [r for r in LADDER if r.key == "SELF_CONTACT_OVERFLOW"]
    assert row.severity == Severity.DEGRADED and row.bully == "S2"
    assert "SELF_CONTACT_OVERFLOW" in REMEDIES
