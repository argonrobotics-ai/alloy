# SPDX-License-Identifier: Apache-2.0
"""GapMonitor on CPU with analytic primitives: exact penetration, histogram, scripted thin-wall pass-through."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact.monitor import (REF_NORMAL, REF_WINDING, PairRef, SignAnchor, _crosses, _dist,
                                   _dist_anchored)
from alloy.contact.pipeline import AlloyContactLayer
from alloy.contact.types import PrecisionPairSpec
from alloy.precision.types import HealthFlags, StepHealth

PLATE_T = 2.5e-4          # 0.25 mm wall (CT-2 / WP3 acceptance)


def _scene(device: str, worlds: int = 2):
    """Static thin plate (top at z=0) + a free 1 mm-thick coin above it; precision pair (plate, coin)."""
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -PLATE_T / 2), wp.quat_identity()), hx=0.02, hy=0.02,
                            hz=PLATE_T / 2, cfg=cfg)
    body = t.add_body(xform=wp.transform((0, 0, 0.01), wp.quat_identity()))
    coin = t.add_shape_cylinder(body, radius=0.005, half_height=5e-4, cfg=cfg)
    t.add_shape_collision_filter_pair(plate, coin)
    b = newton.ModelBuilder()
    b.replicate(t, worlds)
    model = b.finalize(device=device)
    spec = PrecisionPairSpec(plate, coin, mode="sdf", pair_stiffness=1e5, clearance=2e-4, wall_thickness=PLATE_T)
    return model, spec


def _place(model, state, z: list[float]) -> None:
    q = state.body_q.numpy()
    for w, zz in enumerate(z):
        q[w] = (0.0, 0.0, zz, 0.0, 0.0, 0.0, 1.0)
    state.body_q.assign(q)


def _health(device: str, n: int) -> wp.array:
    h = StepHealth()
    h.min_pair_gap_m = np.inf
    return wp.array([h] * n, dtype=StepHealth, device=device)


def test_exact_penetration_and_histogram(device):
    model, spec = _scene(device)
    layer = AlloyContactLayer(model, [spec], native="newton", precision_contacts_per_world=32)
    mon = layer.monitor(sample_spacing=5e-4)
    assert tuple(mon.reference) == ("analytic",)
    s0, s1 = model.state(), model.state()
    _place(model, s0, [0.0105, 0.0105])
    _place(model, s1, [5e-4 - 30e-6, 5e-4 + 1e-4])        # world 0: 30 um into the plate; world 1: 0.1 mm gap
    health = _health(device, 2)
    layer.collide(s1, layer.contacts(), dt=1e-3)
    mon.update(s0, s1, health)
    rep = mon.report()
    assert rep.world_max_m[0] == pytest.approx(30e-6, abs=2e-7)
    assert rep.world_max_m[1] == 0.0
    assert rep.pair_max_m[0] == pytest.approx(30e-6, abs=2e-7)
    assert rep.pair_samples[0] > 0 and rep.histogram_um[0, 30] > 0
    assert rep.pair_p99_m[0] <= 31e-6
    h = health.numpy()
    assert h[0]["max_pair_pen_m"] == pytest.approx(30e-6, abs=2e-7)
    assert h[1]["min_pair_gap_m"] == pytest.approx(1e-4, abs=2e-7)
    # 30 um < S2 = 2*max_pen = 1e-4 m (max_pen = min(5e-5, 0.5*1e-4)): no flag
    assert int(h[0]["flags"]) & int(HealthFlags.PAIR_PEN_WARN | HealthFlags.PAIR_PEN_FAIL) == 0


def _two_coin_scene(device: str, worlds: int = 2):
    """The plate with two free coins above it (x = -/+ 10 mm): precision pairs 0 = (plate, coin0), 1 = (plate, coin1)."""
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -PLATE_T / 2), wp.quat_identity()), hx=0.02, hy=0.02,
                            hz=PLATE_T / 2, cfg=cfg)
    coins = []
    for x in (-0.01, 0.01):
        body = t.add_body(xform=wp.transform((x, 0, 0.01), wp.quat_identity()))
        coins.append(t.add_shape_cylinder(body, radius=0.005, half_height=5e-4, cfg=cfg))
        t.add_shape_collision_filter_pair(plate, coins[-1])
    b = newton.ModelBuilder()
    b.replicate(t, worlds)
    specs = [PrecisionPairSpec(plate, c, mode="sdf", pair_stiffness=1e5, clearance=2e-4, wall_thickness=PLATE_T)
             for c in coins]
    return b.finalize(device=device), specs


def _place_coins(state, z: list[tuple[float, float]]) -> None:
    q = state.body_q.numpy()
    for w, zz in enumerate(z):
        for i, (x, zi) in enumerate(zip((-0.01, 0.01), zz)):
            q[2 * w + i] = (x, 0.0, zi, 0.0, 0.0, 0.0, 1.0)
    state.body_q.assign(q)


def test_health_names_the_deepest_pair(device):
    """max_pair_id is the deepest pair's index (WP5's first-failure snapshot); -1 when none penetrates; a deeper pen
    already in the inbox keeps its own pair."""
    model, specs = _two_coin_scene(device, worlds=3)
    layer = AlloyContactLayer(model, specs, native="newton")
    mon = layer.monitor(sample_spacing=5e-4)
    s0, s1 = model.state(), model.state()
    _place_coins(s0, [(0.0105, 0.0105)] * 3)
    # world 0: coin1 40 um deep, coin0 20 um; world 1: both clear; world 2: coin0 30 um, inbox already holds 1 mm
    _place_coins(s1, [(5e-4 - 20e-6, 5e-4 - 40e-6), (5e-4 + 1e-4, 5e-4 + 1e-4), (5e-4 - 30e-6, 5e-4 + 1e-4)])
    health = _health(device, 3)
    h = health.numpy()
    h["max_pair_id"] = [-1, -1, 7]
    h["max_pair_pen_m"] = [0.0, 0.0, 1e-3]
    health.assign(h)
    mon.update(s0, s1, health)
    h = health.numpy()
    assert h[0]["max_pair_id"] == 1 and h[0]["max_pair_pen_m"] == pytest.approx(40e-6, abs=2e-7)
    assert h[1]["max_pair_id"] == -1 and h[1]["max_pair_pen_m"] == 0.0
    assert h[2]["max_pair_id"] == 7 and h[2]["max_pair_pen_m"] == pytest.approx(1e-3)


def test_flags_follow_pair_limits(device):
    model, spec = _scene(device, worlds=2)
    layer = AlloyContactLayer(model, [spec], native="newton")
    mon = layer.monitor()
    s0, s1 = model.state(), model.state()
    _place(model, s0, [5e-4, 5e-4])
    # 110 um: > S1 = min(5e-4, radial 1e-4). Deeper than half the 0.25 mm wall the exact SDF measures the far face.
    _place(model, s1, [5e-4 - 1.1e-4, 5e-4 - 1.1e-4])
    health = _health(device, 2)
    mon.update(s0, s1, health)
    flags = health.numpy()["flags"]
    assert all(int(f) & int(HealthFlags.PAIR_PEN_FAIL) for f in flags)


def test_scripted_full_pass_through_is_a_tunnel(device):
    """The coin jumps from 1 mm above the 0.25 mm wall to 1 mm below it in one step (both ends outside)."""
    model, spec = _scene(device, worlds=2)
    layer = AlloyContactLayer(model, [spec], native="newton")
    mon = layer.monitor()
    s0, s1 = model.state(), model.state()
    _place(model, s0, [1e-3 + 5e-4, 1e-3 + 5e-4])
    _place(model, s1, [-PLATE_T - 1e-3 - 5e-4, 1e-3])       # world 0 passes through; world 1 stays above
    health = _health(device, 2)
    mon.update(s0, s1, health)
    rep = mon.report()
    assert rep.tunnel_events[0, 0] > 0
    assert rep.tunnel_events[0, 1] == 0
    assert int(health.numpy()["flags"][0]) & int(HealthFlags.PAIR_PEN_FAIL)
    mon.reset()
    assert mon.report().tunnel_events.sum() == 0


@pytest.mark.parametrize("winding", [False, True])
def test_mesh_reference_penetration_and_tunnel(device, winding):
    """The plate as a closed mesh: pseudo-normal and winding-number signs give the same exact depth and tunnel."""
    from alloy.contact.monitor import GapMonitor  # noqa: PLC0415
    from alloy.testing.contact import box_tris, to_newton_mesh  # noqa: PLC0415

    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, gap=5e-4, margin=0.0)
    t = newton.ModelBuilder()
    plate = t.add_shape_mesh(-1, xform=wp.transform((0, 0, -PLATE_T / 2), wp.quat_identity()),
                             mesh=to_newton_mesh(*box_tris(0.02, 0.02, PLATE_T / 2), compute_inertia=False), cfg=cfg)
    body = t.add_body(xform=wp.transform((0, 0, 0.01), wp.quat_identity()))
    coin = t.add_shape_cylinder(body, radius=0.005, half_height=5e-4, cfg=cfg)
    t.add_shape_collision_filter_pair(plate, coin)
    b = newton.ModelBuilder()
    b.replicate(t, 3)
    model = b.finalize(device=device)
    spec = PrecisionPairSpec(plate, coin, mode="sdf", pair_stiffness=1e5, clearance=2e-4, wall_thickness=PLATE_T)
    mon = GapMonitor(AlloyContactLayer(model, [spec], native="newton"), winding=winding)
    assert mon.reference == ["winding_number" if winding else "pseudo_normal"]
    s0, s1 = model.state(), model.state()
    _place(model, s0, [1e-3 + 5e-4] * 3)
    _place(model, s1, [5e-4 - 30e-6, -PLATE_T - 1e-3 - 5e-4, 5e-4 + 1e-4])   # 30 um in; full pass; 0.1 mm gap
    mon.update(s0, s1, _health(device, 3))
    rep = mon.report()
    assert rep.world_max_m[0] == pytest.approx(30e-6, abs=2e-7)
    assert rep.world_max_m[2] == 0.0
    ev = rep.tunnel_events[0]
    assert ev[0] == 0 and ev[1] > 0 and ev[2] == 0


@wp.kernel
def _ref_dist(ref: PairRef, q: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = _dist(ref, 0, q[i])


def test_mesh_reference_sign_on_cavity_mid_plane():
    """The USB-A plug mesh (shell, insert, overmold: closed shells): Warp's pseudo-normal sign reads "inside" on the
    tongue cavity's mid-plane (equidistant from the insert top and the shell's inner top); the monitor confirms an
    inside reading with the winding number, so the seated socket tongue is not a 0.975 mm penetration. (The opposite
    error remains in ``_dist``, used for the previous pose and tunnel marching: inside the overmold or the insert the
    pseudo-normal sign can read "outside"; current poses use anchored winding-number signs, see below.)"""
    from alloy.cards.meshgen import _usb_a_plug

    m = _usb_a_plug()
    with wp.ScopedDevice("cpu"):
        mesh = wp.Mesh(wp.array(m.vertices.astype(np.float32), dtype=wp.vec3),
                       wp.array(m.faces.astype(np.int32).ravel(), dtype=wp.int32), support_winding_number=True)
        ref = PairRef()
        ref.kind = wp.array([REF_NORMAL], dtype=wp.int32)
        ref.geo, ref.scale = wp.zeros(1, dtype=wp.int32), wp.zeros(1, dtype=wp.vec3)
        ref.mesh = wp.array([mesh.id], dtype=wp.uint64)
        pts = np.array([[0.0, 0.00096, 0.006], [0.0026, 0.00096, 0.0036], [0.0, 0.0, -0.01]], np.float32)
        out = wp.zeros(3, dtype=float)
        wp.launch(_ref_dist, dim=3, inputs=[ref, wp.array(pts, dtype=wp.vec3), out])
        d = out.numpy()
    assert d[0] == pytest.approx(9.75e-4, abs=2e-6) and d[1] == pytest.approx(9.75e-4, abs=2e-6)  # cavity: outside
    assert d[2] == pytest.approx(-3.65e-3, abs=2e-6)  # overmold centre line: inside


@wp.kernel
def _walk(ref: PairRef, path: wp.array2d(dtype=wp.vec3), anchor: SignAnchor, exact: wp.array2d(dtype=float),
          cached: wp.array2d(dtype=float), normal: wp.array2d(dtype=float)):
    i = wp.tid()
    for k in range(path.shape[1]):
        exact[i, k] = _dist(ref, 1, path[i, k])           # REF_WINDING: the winding number for every query
        cached[i, k] = _dist_anchored(ref, 0, path[i, k], anchor, i)
        normal[i, k] = _dist(ref, 0, path[i, k])          # REF_NORMAL without anchors


def test_sign_anchor_is_exact():
    """Anchored signs equal the winding number's: points walking in 20 um steps through the USB-A plug mesh (across
    its outer surface, through the tongue cavity's mid-plane, along the overmold's centre line) and jumping back to
    their start. The pseudo-normal alone reads the overmold centre line "outside" (a missed 3.65 mm penetration)."""
    from alloy.cards.meshgen import _usb_a_plug

    m = _usb_a_plug()
    starts = np.array([[0.0, 0.00096, 0.006], [0.0026, 0.00096, 0.0036], [0.0, 0.0, -0.01], [0.0, 0.006, 0.0]])
    dirs = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    path = starts[:, None, :] + dirs[:, None, :] * (np.arange(400)[None, :, None] * 2e-5)
    path[:, -1] = starts                     # a jump back (a reset): the sign is positional, the anchor still holds
    with wp.ScopedDevice("cpu"):
        mesh = wp.Mesh(wp.array(m.vertices.astype(np.float32), dtype=wp.vec3),
                       wp.array(m.faces.astype(np.int32).ravel(), dtype=wp.int32), support_winding_number=True)
        ref = PairRef()
        ref.kind = wp.array([REF_NORMAL, REF_WINDING], dtype=wp.int32)
        ref.geo, ref.scale = wp.zeros(2, dtype=wp.int32), wp.zeros(2, dtype=wp.vec3)
        ref.mesh = wp.array([mesh.id, mesh.id], dtype=wp.uint64)
        anchor = SignAnchor()
        anchor.q, anchor.d = wp.zeros(len(path), dtype=wp.vec3), wp.zeros(len(path), dtype=wp.float32)
        exact, cached, normal = (wp.zeros(path.shape[:2], dtype=float) for _ in range(3))
        wp.launch(_walk, dim=len(path), inputs=[ref, wp.array(path.astype(np.float32), dtype=wp.vec3), anchor, exact,
                                                 cached, normal])
        a, b, c = exact.numpy(), cached.numpy(), normal.numpy()
    assert (a < 0).any() and (a > 0).any()
    assert (np.sign(a) == np.sign(b)).all() and np.allclose(a, b, rtol=0, atol=1e-7)
    assert (np.sign(c) != np.sign(a)).any()  # the pseudo-normal misreads some of these points


def _plates_scene(device: str, thick: tuple[float, ...]):
    """World w: a static plate thick[w] (top at z = 0) under a free 1 mm coin; one precision pair per world's
    template (CT-2's build: worlds differ in the pair's geometry, SCR-XB4-10-1)."""
    cfg = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, gap=5e-4, margin=0.0)
    b = newton.ModelBuilder()
    for t_w in thick:
        t = newton.ModelBuilder()
        plate = t.add_shape_box(-1, xform=wp.transform((0, 0, -t_w / 2), wp.quat_identity()), hx=0.02, hy=0.02,
                                hz=t_w / 2, cfg=cfg)
        coin = t.add_shape_cylinder(t.add_body(xform=wp.transform((0, 0, 0.01), wp.quat_identity())),
                                    radius=0.005, half_height=5e-4, cfg=cfg)
        t.add_shape_collision_filter_pair(plate, coin)
        b.add_world(t)
    spec = PrecisionPairSpec(plate, coin, mode="sdf", pair_stiffness=1e5, clearance=2e-4, wall_thickness=min(thick))
    return b.finalize(device=device), spec


def test_each_world_is_measured_against_its_own_plate(device):
    """World 0's coin is 0.1 mm above its 0.25 mm plate, world 1's 30 um into its 1 mm plate: only world 1
    penetrates. Measured against world 0's plate (the bug), world 1's coin sat 0.35 mm outside a 0.25 mm slab around
    its plate's centre and read 0."""
    model, spec = _plates_scene(device, (PLATE_T, 1e-3))
    layer = AlloyContactLayer(model, [spec], native="newton", precision_contacts_per_world=32)
    mon = layer.monitor(sample_spacing=5e-4)
    s0, s1 = model.state(), model.state()
    _place(model, s0, [0.0105, 0.0105])
    _place(model, s1, [5e-4 + 1e-4, 5e-4 - 30e-6])
    health = _health(device, 2)
    layer.collide(s1, layer.contacts(), dt=1e-3)
    mon.update(s0, s1, health)
    rep = mon.report()
    assert rep.world_max_m[0] == 0.0
    assert rep.world_max_m[1] == pytest.approx(30e-6, abs=2e-7)
    assert rep.tunnel_events.sum() == 0
    h = health.numpy()
    assert h[0]["min_pair_gap_m"] == pytest.approx(1e-4, abs=2e-7)
    assert h[1]["max_pair_pen_m"] == pytest.approx(30e-6, abs=2e-7)
    assert mon.slots == 2 and tuple(mon.reference) == ("analytic",)     # one geometry slot per plate


def test_replicated_worlds_share_one_slot(device):
    model, spec = _scene(device, worlds=3)
    mon = AlloyContactLayer(model, [spec], native="newton").monitor()
    assert mon.slots == 1 and mon.n_tasks == 3 * mon.n_samples


@wp.kernel
def _cross(ref: PairRef, a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = wp.int32(_crosses(ref, 0, a[i], b[i]))


def _box_ref(floor: float) -> PairRef:
    """The thin plate of ``_scene`` as an analytic reference (centred, half height PLATE_T / 2), its march at half the
    wall, and ``floor`` as the march's inside depth."""
    from alloy.contact.analytic_sdf import GEO_BOX
    from alloy.contact.monitor import REF_ANALYTIC

    ref = PairRef()
    ref.kind, ref.geo = wp.array([REF_ANALYTIC], dtype=wp.int32), wp.array([GEO_BOX], dtype=wp.int32)
    ref.scale = wp.array([(0.02, 0.02, PLATE_T / 2)], dtype=wp.vec3)
    ref.mesh, ref.margin = wp.zeros(1, dtype=wp.uint64), wp.zeros(1, dtype=wp.float32)
    ref.march, ref.cross_floor = wp.array([PLATE_T / 2], dtype=wp.float32), wp.array([floor], dtype=wp.float32)
    return ref


def _crossings(ref: PairRef, segments: list[tuple[tuple, tuple]]) -> list[int]:
    a = wp.array(np.array([s[0] for s in segments], np.float32), dtype=wp.vec3)
    b = wp.array(np.array([s[1] for s in segments], np.float32), dtype=wp.vec3)
    out = wp.zeros(len(segments), dtype=wp.int32)
    wp.launch(_cross, dim=len(segments), inputs=[ref, a, b, out])
    return out.numpy().tolist()


def test_a_sample_sliding_off_a_wall_it_touches_is_not_a_tunnel():
    """PHYSA-4 on an L4: a sample that slides along a face it touches (0.3 um out) and over the face's edge ends 0.3 um
    out beside it; the straight segment between the two poses dips 0.2 um into the corner. Within the pair's contact
    depth that is contact, not a pass-through: the march counts a point as inside only past ``cross_floor``. A real
    crossing of the 0.25 mm wall, square or oblique and at any phase of the march, still counts."""
    from alloy.contact.monitor import cross_floor

    top, edge = PLATE_T / 2, 0.02
    slide = [((edge - 6.0e-4, 0.0, top + 3.0e-7), (edge + 3.0e-7, 0.0, top - 3.0e-7))]
    through = [((0.0, 0.0, top + 5.0e-4 + dz), (0.0, 0.0, -top - 5.0e-4 + dz)) for dz in np.linspace(0.0, 1.2e-4, 7)]
    through.append(((-5.0e-4, 0.0, top + 5.0e-4), (5.0e-4, 0.0, -top - 5.0e-4)))
    floor = cross_floor(5.0e-5, PLATE_T / 2)              # the plate pair's max penetration (radial clearance 1e-4)
    assert floor == pytest.approx(0.25 * PLATE_T / 2) and cross_floor(1.0e-6, PLATE_T / 2) == 1.0e-6
    assert _crossings(_box_ref(0.0), slide) == [1]        # the march before 2026-11-10: a false tunnel
    assert _crossings(_box_ref(floor), slide) == [0]
    assert _crossings(_box_ref(floor), through) == [1] * len(through)
