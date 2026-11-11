# SPDX-License-Identifier: Apache-2.0
"""Soil dig-and-pour task (alloy/tasks/soil_dig.py; docs/notes/soil-design.md): the scoop's geometry, its motion
plan and clearances, and the mass accounting, without a simulation (CPU, < 1 s)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.tasks.soil_dig import DigSpec, _legs, _local_boxes, _masses, _pivot0, _scoop_boxes, ramp_profile, scoop_path

VOXEL = 0.005                                       # the MPM grid cell of the soil cards


def test_the_plates_close_the_back_and_sides_of_the_floor() -> None:
    s = DigSpec()
    length, width, height, t = s.scoop
    boxes = _local_boxes(s)
    (fs, fc, _), (bs, bc, _) = boxes["scoop_floor"], boxes["scoop_back"]
    assert fc[0] - fs[0] / 2 == pytest.approx(t) and bc[0] - bs[0] / 2 == pytest.approx(0.0)   # back plate outside
    assert fc[2] - fs[2] / 2 == pytest.approx(0.0) and bc[2] - bs[2] / 2 == pytest.approx(0.0)  # pivot: bottom edge
    for side, sign in (("scoop_side_p", 1.0), ("scoop_side_n", -1.0)):
        size, c, roll = boxes[side]
        assert sign * (c[1] - sign * size[1] / 2) == pytest.approx(width / 2)   # inner face on the floor edge
        assert size[2] == pytest.approx(height) and c[0] == pytest.approx(fc[0]) and roll == 0.0


@pytest.mark.parametrize("flare", [10.0, 20.0])
def test_flared_sides_hinge_on_the_floor_edge_and_lean_out(flare: float) -> None:
    s = DigSpec(flare_deg=flare)
    length, width, height, t = s.scoop
    for side, sign in (("scoop_side_p", 1.0), ("scoop_side_n", -1.0)):
        _, c, roll = _local_boxes(s)[side]
        up = np.array([0.0, -math.sin(roll), math.cos(roll)])                    # the plate's local z after the roll
        bottom = np.asarray(c) - 0.5 * height * up
        assert bottom[1:] == pytest.approx([sign * (width / 2 + t / 2), 0.0])     # mid-plane on the floor's edge
        assert sign * up[1] == pytest.approx(math.sin(math.radians(flare)))       # the top leans outward


def test_the_scoop_starts_tilted_with_its_tip_just_above_the_bed() -> None:
    s = DigSpec()
    length, _, _, t = s.scoop
    a = math.radians(s.stab_deg)
    size, centre, rpy = _scoop_boxes(s)["scoop_floor"]
    assert rpy == (0.0, a, 0.0)
    along, normal = np.array([math.cos(a), 0.0, -math.sin(a)]), np.array([math.sin(a), 0.0, math.cos(a)])
    tip = np.asarray(centre) + 0.5 * size[0] * along - 0.5 * size[2] * normal    # front-bottom edge of the floor
    assert tip == pytest.approx([s.pivot_x + (length + t) * math.cos(a), 0.0, s.bed_top + s.clearance])
    assert _pivot0(s)[2] > s.bed_top                                             # the back stays out of the soil


@pytest.mark.parametrize("stab", [30.0, 45.0])
def test_the_legs_stab_to_depth_curl_back_and_pour_over_the_bucket(stab: float) -> None:
    s = DigSpec(stab_deg=stab, drag=0.0 if stab < 40 else DigSpec().drag)
    a, r = math.radians(stab), s.scoop[0] + s.scoop[3]
    legs = {name: (np.asarray(v), np.asarray(w), sec) for name, v, w, sec in _legs(s)}
    v, w, sec = legs["stab"]
    assert np.cross(v, [math.cos(a), 0.0, -math.sin(a)]) == pytest.approx(np.zeros(3), abs=1e-12)   # along the floor
    assert _pivot0(s)[2] + v[2] * sec - r * math.sin(a) == pytest.approx(s.bed_top - s.depth)       # tip at depth
    pivot = _pivot0(s) + sum(v * sec for v, _, sec in legs.values())
    assert pivot[0] == pytest.approx(s.bucket_x - s.pour_back)                   # pours beside the bucket centre
    turn = math.degrees(a + sum(w[1] * sec for _, w, sec in legs.values()))
    assert turn == pytest.approx(s.pour_deg)                                      # ends front down
    assert math.degrees(a + legs["curl"][1][1] * legs["curl"][2]) == pytest.approx(-s.tilt_deg)   # carried front up


@pytest.mark.parametrize("spec", [DigSpec(), DigSpec(stab_deg=30.0, drag=0.0)], ids=["stab45", "stab30"])
def test_the_scoop_keeps_clear_of_tray_and_bucket(spec: DigSpec) -> None:
    """In the soil: >= 2 voxels from the tray's walls and floor (closer squeezes grains out of the grid); below a
    wall's top: >= 1 voxel inside the tray or the bucket, never through a wall."""
    tx, ty, td = spec.tray_inner
    bx, by, bd = spec.bucket_inner
    for leg, p in scoop_path(spec):
        gap = np.stack([tx / 2 - np.abs(p[:, 0]), ty / 2 - np.abs(p[:, 1]), p[:, 2] - spec.tray_wall], axis=1)
        soil = p[:, 2] < spec.bed_top
        assert gap[soil].min(initial=1.0) >= 2 * VOXEL, leg
        low = p[:, 2] < spec.tray_wall + td
        assert gap[low, :2].min(initial=1.0) >= VOXEL, leg
        in_bucket_band = np.abs(p[:, 0] - spec.bucket_x) < bx / 2 + spec.bucket_wall + VOXEL
        assert not (in_bucket_band & (p[:, 2] < spec.bucket_wall + bd + VOXEL)).any(), leg


def test_masses_split_lifted_bucket_and_spill() -> None:
    s = DigSpec()
    top = 0.04
    pts = np.array([[[0.0, 0.0, 0.02],                       # in the bed
                     [s.tray_inner[0] / 2 + 0.5 * s.tray_wall, 0.0, 0.03],   # on the tray wall: not spilled
                     [s.bucket_x, 0.0, 0.01],                # in the bucket
                     [s.bucket_x, 0.0, 0.08],                # above the bucket (still in the scoop): lifted, spill
                     [0.30, 0.0, 0.005]]])                   # on the ground: spill
    sim = SimpleNamespace(state=lambda names: {"soil.points": pts},
                          engine=SimpleNamespace(model=SimpleNamespace(
                              particle_mass=SimpleNamespace(numpy=lambda: np.full(5, 0.002)))))
    m = _masses(sim, s, top)
    assert m == pytest.approx({"lifted_kg": 0.002, "bucket_kg": 0.002, "spill_kg": 0.004, "total_kg": 0.01})


@pytest.mark.parametrize("seconds", [0.1, 0.4, 1.45])
def test_the_ramp_keeps_the_leg_distance_and_starts_and_ends_slow(seconds: float) -> None:
    f = ramp_profile(seconds, 0.02, 0.1)
    assert f.sum() * 0.02 == pytest.approx(seconds)
    if len(f) >= 10:
        assert f[0] < 0.2 * f.max() and f[-1] < 0.2 * f.max() and np.allclose(f, f[::-1])
