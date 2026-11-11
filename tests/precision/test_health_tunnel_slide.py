# SPDX-License-Identifier: Apache-2.0
"""TUNNEL must not fire on a point that slides along a thin static box while shallowly inside it (GAP-E2, 2026-11-10).

DC2 eth_swap (WP2's bisect: 57/64 -> 38/64 at 8efe08e56): the plug's latch tab, a 0.9 mm plate whose centre sits
0.45 mm inside its own surface, slides under the port's 1.4 mm roof at about 1 mm contact depth. Its centre enters the
roof box by the front face and leaves by the back face while it is only a few tenths of a mm inside the bottom face;
the in-and-out rule read that as "left by the opposite side", made the world INVALID and froze it short of port 3.

Law: a box counts as crossed only through its thickness: the face nearest the point at its deepest inside (or, for a
one-step crossing, at the middle of the run inside) belongs to the entry / exit pair. Slides clean at every depth that
keeps the bottom face nearest and at every step; a crossing through the box's own thin dimension is still TUNNEL.
Hand-moved bodies on the CPU (``TunnelWatch.after_step`` on a built model), as in test_health_tunnel_kernels.py.

Second cause in the same task (WP2's L4 dump on main: the plug body x switch end wall, floor and port back walls in 24
of 33 flagged worlds): ``alloy.episode.layout`` moves each world's static switch after the build by writing
``model.shape_transform``, and the watch had copied the static poses at its construction, so it judged ghosts at the
template pose. Law: a static box moved after the watch is built is judged where it is now, not where it was.

AUDIT-STACK 17:51Z on the first fix: a one-step crossing judged only by the face pair nearest its mid-run point missed a
pass 2 mm inside a plate's edge and a ball top to bottom through a rail edge-on. Law: those are TUNNEL, in one step and
sampled inside (the mid-run point, or the deepest sample, is 2 mm or more inside: the side rule)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.precision import health_tunnel as HT
from alloy.precision import health_tunnel_setup as S

ROOF_HALF = (0.0007, 0.0035, 0.00246)          # m: DC2's port roof (1.4 x 7 x 4.92 mm)
ROOF_Z = 0.3
BOTTOM = ROOF_Z - ROOF_HALF[2]


@pytest.fixture(scope="module")
def model():
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground()
    scene.add_box("roof", size=tuple(2 * h for h in ROOF_HALF), pos=(0, 0, ROOF_Z), static=True)
    scene.add_box("block", size=(0.01, 0.01, 0.01), pos=(0.2, 0, ROOF_Z), static=True)          # a 1 cm cube
    scene.add_box("plate", size=(0.1, 0.1, 0.01), pos=(0.4, 0, ROOF_Z), static=True)            # 1 cm plate
    scene.add_box("rail", size=(0.1, 0.006, 0.015), pos=(0.6, 0, ROOF_Z), static=True)           # a 6 x 15 mm rail
    scene.add_box("tab", size=(0.0168, 0.0032, 0.0013), pos=(0.5, 0, 0.5), density=1000.0)   # the API's thinnest
    return alloy.build(scene, device="cpu").engine.model


def _watch(model) -> HT.TunnelWatch:
    return HT.TunnelWatch(model, S.judged_bodies(model), S.static_shapes(model))


def _sweep(model, path: np.ndarray, tw: HT.TunnelWatch | None = None) -> int:
    """Move the tab's centre along ``path`` (rows of xyz), one rigid step per row pair; the world's TUNNEL (a fresh
    watch unless ``tw`` is given)."""
    tw = tw or _watch(model)
    tab = [i for i, n in enumerate(model.body_label) if n.endswith("tab")][0]
    q = model.state().body_q.numpy().copy()
    for a, b in zip(path[:-1], path[1:]):
        s0, s1 = model.state(), model.state()
        q0, q1 = q.copy(), q.copy()
        q0[tab, :3], q1[tab, :3] = a, b
        s0.body_q.assign(q0)
        s1.body_q.assign(q1)
        tw.after_step(s0, s1, 0.00125)
    return int(tw.tunnel.numpy()[0])


def _line(x0: float, x1: float, step: float, z: float) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-12, step)
    return np.stack([xs, np.zeros_like(xs), np.full_like(xs, z)], axis=1)


@pytest.mark.parametrize("inside_mm", [0.1, 0.3, 0.6])
@pytest.mark.parametrize("step_mm", [0.2, 1.0, 2.0])
def test_a_point_sliding_under_a_thin_box_while_shallowly_inside_it_is_not_tunnel(model, inside_mm, step_mm) -> None:
    path = _line(-0.003, 0.003, step_mm * 1e-3, BOTTOM + inside_mm * 1e-3)   # in by the front face, out by the back
    assert _sweep(model, path) == 0


@pytest.mark.parametrize("step_mm", [0.2, 0.5, 2.0])
def test_a_point_crossing_the_thin_box_through_its_thickness_is_tunnel(model, step_mm) -> None:
    path = _line(-0.003, 0.003, step_mm * 1e-3, ROOF_Z)                       # through the middle of the 1.4 mm
    assert _sweep(model, path) == 1


def test_a_point_through_the_middle_of_a_cube_is_tunnel_where_every_face_pair_ties(model) -> None:
    zs = np.arange(ROOF_Z - 0.008, ROOF_Z + 0.008 + 1e-12, 0.001)               # passes the exact centre: a tie
    path = np.stack([np.full_like(zs, 0.2), np.zeros_like(zs), zs], axis=1)
    assert _sweep(model, path) == 1


def test_a_static_box_moved_after_the_build_is_judged_where_it_is_now(model) -> None:
    roof = [i for i, n in enumerate(model.shape_label) if str(n).endswith("roof")][0]
    tf = model.shape_transform.numpy()
    old = tf.copy()
    first, second = _watch(model), _watch(model)               # built before the move, as at alloy.build
    try:
        tf[roof, 0] += 0.05                                   # 5 cm along x, as a layout moves a world's switch
        model.shape_transform.assign(tf)
        assert _sweep(model, _line(-0.003, 0.003, 0.0002, ROOF_Z), first) == 0    # the old place: nothing there now
        assert _sweep(model, _line(0.047, 0.053, 0.0002, ROOF_Z), second) == 1    # through the box where it is
    finally:
        model.shape_transform.assign(old)


def _drop(x: float, y: float, z0: float, z1: float, step: float) -> np.ndarray:
    zs = np.arange(z0, z1 - 1e-12, -step)
    return np.stack([np.full_like(zs, x), np.full_like(zs, y), zs], axis=1)


@pytest.mark.parametrize("inside_mm", [2.0, 2.5, 4.0])
@pytest.mark.parametrize("step_mm", [0.5, 30.0])
def test_a_pass_near_a_plate_edge_is_tunnel_in_one_step_and_sampled(model, inside_mm, step_mm) -> None:
    y = 0.05 - inside_mm * 1e-3                    # this far inside the plate's +y edge (the tab's points lie along x)
    assert _sweep(model, _drop(0.4, y, ROOF_Z + 0.015, ROOF_Z - 0.015, step_mm * 1e-3)) == 1


@pytest.mark.parametrize("step_mm", [0.5, 30.0])
def test_a_pass_top_to_bottom_through_a_rail_edge_on_is_tunnel(model, step_mm) -> None:
    assert _sweep(model, _drop(0.6, 0.0, ROOF_Z + 0.02, ROOF_Z - 0.02, step_mm * 1e-3)) == 1


def test_a_one_step_slide_under_the_thin_box_is_still_not_tunnel(model) -> None:
    for inside_mm in (0.1, 0.3, 0.6):
        path = _line(-0.003, 0.003, 0.006, BOTTOM + inside_mm * 1e-3)        # one step across the 1.4 mm
        assert _sweep(model, path) == 0
