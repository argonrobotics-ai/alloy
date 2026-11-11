# SPDX-License-Identifier: Apache-2.0
"""Vehicle plug-in, drivable surfaces (SCR-WP8-12 F1b): the wheel's path over a ramp foot, crest, seam and step;
a sprung floor's sag and a sliding floor's momentum (the reaction goes on the surface's body); a braked truck on a
ramp face. Closed forms in laws_surface."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alloy.plugins.vehicle import Vehicle
from alloy.plugins.vehicle import laws as vl
from alloy.plugins.vehicle import laws_grip as lg
from alloy.plugins.vehicle import laws_surface as ls

pytestmark = pytest.mark.law
G = 9.81
SLOW = pytest.mark.timeout(240)                      # 30-50 s alone on the CPU; a loaded gate host (-n 8) runs ~2x


def _cart(**over):
    cfg = {**ls.CART, **over}
    return cfg, Vehicle(**cfg).config


@SLOW
def test_r3_wheel_centre_follows_the_profile_and_crosses_crest_and_seam_without_a_jolt():
    cfg, c = _cart()
    r = lg.run(cfg, 18.0, {}, build=ls.profile_build(c), chassis=c.chassis, bodies=(c.chassis,),
               control=ls.cruise(0.5))
    rot = Rotation.from_quat(r["q"][:, 3:])
    centre = r["q"][:, :3] + rot.apply(np.asarray(c.positions[0]))           # front-left wheel centre
    x, z = centre[:, 0], centre[:, 2]
    fz = r["force"][:, 0, 2]
    want = ls.centre_path(x, 0.3)
    tol = 1.0e-3 + fz / 2.0e6
    moving = (x > 0.7) & (x < ls.SEGMENTS[-1][2][0] - 0.4)
    assert x.max() > ls.SEGMENTS[-1][1][0] + 0.5, x.max()                    # it got over the step
    err = np.abs(z - want)[moving]
    assert (err <= tol[moving]).all(), float((err - tol[moving]).max())
    crest = ls.SEGMENTS[0][2][0]
    near = np.abs(x - crest) < 0.3
    rear = r["q"][:, :3] + rot.apply(np.asarray(c.positions[2]))
    path = 0.5 * (want + ls.centre_path(rear[:, 0], 0.3))                     # the body midway between the axles
    jolt = np.gradient(r["qd"][:, 2], r["t"]) - np.gradient(np.gradient(path, r["t"]), r["t"])
    assert np.abs(jolt[near]).max() <= 0.05 * G, float(np.abs(jolt[near]).max())
    seam = ls.SEGMENTS[1][2][0]
    at = np.abs(x - seam) < 0.25
    level = float(np.median(fz[at]))
    assert np.abs(fz[at] - level).max() <= 0.02 * level, (float(np.abs(fz[at] - level).max()), level)


def test_r4_sprung_floor_sags_by_the_carried_weight():
    cfg, c = _cart(surfaces=("floor_top",), ground_plane=False)
    r = lg.run(cfg, 3.0, {"brake": 1.0}, build=ls.sprung_floor_build(c), chassis=c.chassis, bodies=(c.chassis,),
               extra=lambda: [ls.floor_vehicle()], track=("floor",))
    sag = ls.floor_rest() - float(r["floor.q"][-1, 2])
    want = (ls.FLOOR_MASS + vl.MASS) * G / (4.0 * ls.FLOOR_K)
    assert sag == pytest.approx(want, rel=0.03), (sag, want)


def test_r4_tyre_reaction_on_a_sliding_floor_keeps_momentum():
    cfg, c = _cart(surfaces=("floor_top",), ground_plane=False)
    r = lg.run(cfg, 3.0, {"throttle": 0.6}, build=ls.sliding_floor_build(c), chassis=c.chassis,
               bodies=(c.chassis,), schedule={1.5: {"throttle": 0.0, "brake": 1.0}}, track=("floor",))
    p = vl.MASS * r["qd"][:, 0] + ls.SLIDE_MASS * r["floor.qd"][:, 0]
    scale = vl.MASS * float(np.abs(r["qd"][:, 0]).max())
    assert scale > 500.0                                                     # it drove (> 0.6 m/s)
    assert float(np.abs(p).max()) <= 5.0e-6 * scale, (float(np.abs(p).max()), scale)   # float32: ~10 ulp in 2,400 steps


@SLOW
@pytest.mark.parametrize("front_uphill", [True, False])
def test_r2_braked_truck_holds_on_a_ramp_face(front_uphill):
    r = lg.run(ls.on_ramp(lg.TRUCK), 11.5, {"brake": 1.0}, build=ls.ramp_truck_build(front_uphill))
    start = r["t"] >= 1.5
    creep = float(np.linalg.norm(r["q"][-1, :3] - r["q"][np.argmax(start), :3]))
    assert creep <= 1.0e-3, creep
    assert np.abs(r["rate"][r["t"] >= 10.5]).max() < 1.0e-3


def test_seam_load_is_continuous_as_the_neighbour_face_moves_off_coplanar():
    """WP13: a wheel parked 1 cm before a seam; the neighbour face rises 0 -> 5 mm and tilts 0 -> 8° about the seam.
    The coplanar rule is blended (1-3 mm, 3-6°), so the wheel's weighted contact depth changes no faster than the
    motion (a switch would jump by the edge's whole depth, ~0.8 mm)."""
    import warp as wp

    from alloy.plugins.vehicle import surfaces as sf

    @wp.kernel
    def depth(s: sf.Surfaces, body_q: wp.array[wp.transform], centre: wp.vec3, out: wp.array[float]):
        p = sf.wheel_hits(0, centre, wp.vec3(0.0, 1.0, 0.0), 0.3, s, body_q)
        out[0] = wp.max(p.a.depth * p.a.weight, 0.0) + wp.max(p.b.depth * p.b.weight, 0.0)

    s = sf.Surfaces()
    s.shape = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
    s.shape_body = wp.array([-1, 0], dtype=wp.int32, device="cpu")
    s.shape_xf = wp.array([wp.transform((-1.0, 0.0, -0.02), wp.quat_identity()),
                           wp.transform((1.0, 0.0, -0.02), wp.quat_identity())], dtype=wp.transform, device="cpu")
    s.shape_scale = wp.array([(1.0, 1.0, 0.02), (1.0, 1.0, 0.02)], dtype=wp.vec3, device="cpu")
    s.ground, s.plane = 0.0, 0
    out = wp.zeros(1, dtype=float, device="cpu")
    centre = wp.vec3(-0.01, 0.0, 0.299)                                      # 1 mm into A, 1 cm before the seam
    got, steps = [], 1000
    for i in range(steps + 1):
        f = i / steps
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -math.radians(8.0) * f)   # B's far side up
        body_q = wp.array([wp.transform((0.0, 0.0, 0.005 * f), rot)], dtype=wp.transform, device="cpu")
        wp.launch(depth, dim=1, inputs=[s, body_q, centre], outputs=[out], device="cpu")
        got.append(float(out.numpy()[0]))
    jumps = np.abs(np.diff(got))
    assert got[0] == pytest.approx(1.0e-3, abs=1.0e-6)                      # A alone at first: B's edge is a seam
    assert got[-1] > 2.0e-3                                                  # then B's raised edge bears
    assert jumps.max() <= 3.0 * 0.005 / steps, jumps.max()                   # no faster than the motion


@SLOW
def test_a_wheel_parked_over_a_seam_keeps_its_load_smooth_as_the_floor_sinks():
    """WP13: the cart parks with its front wheel 5 mm past the seam, on the sprung floor; then the floor is pushed
    down 5 mm more, slowly, so the seam goes off coplanar under the wheel (A's edge rises against the tyre): the front
    wheel's load changes smoothly and the treads settle."""
    cfg, c = _cart(surfaces=("deck", "floor_top"), ground_plane=False)
    press = 4.0 * ls.FLOOR_K * 0.005
    r = lg.run(cfg, 13.0, {}, build=ls.seam_floor_build(c), chassis=c.chassis, bodies=(c.chassis,),
               extra=lambda: [ls.floor_vehicle(), ls.Press(force=press, rate=press / 2.5, start=8.0)], track=("floor",),
               control=ls.park_at(ls.SEAM_PARK + ls.PARK_PAST, 0.3, hold=7.5))
    fz = r["force"][:, 0, 2]
    pressing = (r["t"] >= 8.0) & (r["t"] <= 11.0)
    static = float(np.median(fz[pressing]))
    front = float(r["q"][np.argmax(r["t"] >= 8.0), 0]) + 0.6
    assert abs(front - ls.SEAM_PARK) < 0.02, front           # parked over the seam (both faces under the tyre)
    sunk = float(r["floor.q"][np.argmax(r["t"] >= 8.0), 2] - r["floor.q"][-1, 2])
    assert sunk > 4.0e-3, sunk                                               # the seam really went off coplanar
    assert np.abs(np.diff(fz[pressing])).max() <= 0.05 * static, (float(np.abs(np.diff(fz[pressing])).max()), static)
    assert np.abs(r["rate"][r["t"] >= 12.0]).max() < 1.0e-3                 # no tread ringing
