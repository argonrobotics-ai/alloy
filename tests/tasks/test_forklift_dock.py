# SPDX-License-Identifier: Apache-2.0
"""GA's dock, leveler, trailer and yard ramp (SCR-WP8-12 F4) against closed forms in alloy.tasks.forklift_trailer and
alloy.tasks.forklift_dock: the trailer's rest height, sag and dock walk, the leveler floating with it, the truck at the
sill, and the force across the leveler's hinge (WP13). CPU, at the forklift task's step."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from alloy._vendor import newton
from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.vehicle import laws_grip as lg
from alloy.plugins.vehicle import laws_surface as ls
from alloy.tasks import forklift_dock as fd
from alloy.tasks import forklift_trailer as tr

pytestmark = pytest.mark.law
SLOW = pytest.mark.timeout(240)                      # 20-40 s alone on the CPU; a loaded gate host (-n 8) runs ~2x
HINGE = math.radians(4.5)


def _hinge_force(depth: float = 1.0e-3, steps: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """A disc (R 0.3 m) rolled at constant depth over a flat deck and a lip turned down 4.5° at x = 2 m: the x of its
    centre and the load vector sum(w·d·n) of its weighted contacts (unit stiffness), per step."""
    import warp as wp

    from alloy.plugins.vehicle import surfaces as sf

    @wp.kernel
    def load(s: sf.Surfaces, body_q: wp.array[wp.transform], centre: wp.vec3, out: wp.array[wp.vec3]):
        p = sf.wheel_hits(0, centre, wp.vec3(0.0, 1.0, 0.0), 0.3, s, body_q)
        out[0] = wp.max(p.a.depth * p.a.weight, 0.0) * p.a.normal + wp.max(p.b.depth * p.b.weight, 0.0) * p.b.normal

    lip = 0.4
    segments = (("deck", (0.0, 1.0), (2.0, 1.0)),                         # 1 m up: clear of centre_path's ground
                ("lip", (2.0, 1.0), (2.0 + lip * math.cos(HINGE), 1.0 - lip * math.sin(HINGE))))
    xfs, scales = [], []
    for _, p0, p1 in segments:
        theta = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        length = math.dist(p0, p1)
        normal = np.array([-math.sin(theta), 0.0, math.cos(theta)])
        mid = np.array([0.5 * (p0[0] + p1[0]), 0.0, 0.5 * (p0[1] + p1[1])]) - 0.02 * normal
        xfs.append(wp.transform(wp.vec3(*mid), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -theta)))
        scales.append((0.5 * length, 1.0, 0.02))
    s = sf.Surfaces()
    s.shape = wp.array([[0, 1]], dtype=wp.int32, device="cpu")
    s.shape_body = wp.array([-1, -1], dtype=wp.int32, device="cpu")
    s.shape_xf = wp.array(xfs, dtype=wp.transform, device="cpu")
    s.shape_scale = wp.array(scales, dtype=wp.vec3, device="cpu")
    s.ground, s.plane = -10.0, 0
    body_q = wp.zeros(1, dtype=wp.transform, device="cpu")
    out = wp.zeros(1, dtype=wp.vec3, device="cpu")
    x = np.linspace(1.8, 2.25, steps + 1)
    z = ls.centre_path(x, 0.3 - depth, segments)
    f = []
    for xi, zi in zip(x, z, strict=True):
        wp.launch(load, dim=1, inputs=[s, body_q, wp.vec3(float(xi), 0.0, float(zi))], outputs=[out], device="cpu")
        f.append(out.numpy()[0].copy())
    return x, np.asarray(f, np.float64)


def test_force_turns_smoothly_across_the_levelers_hinge():
    """WP13 (F4): over the deck-lip hinge the two contacts' load turns from the deck's normal to the lip's with no
    step. The disc's contact walks the hinge's arc in R·4.5° = 24 mm; the old split (the deeper contact bearing all its
    depth) stepped by c·d·|Δn|, 4 % of the load at the swap."""
    x, f = _hinge_force()
    mag = np.linalg.norm(f, axis=1)
    step = np.linalg.norm(np.diff(f, axis=0), axis=1) / mag[:-1]
    smooth = 1.0e-3 / (0.3 * HINGE) * 2.0 * math.sin(HINGE / 2)            # the arc's own turn per 0.11 mm step
    assert step.max() <= 3.0 * smooth, (step.max(), smooth)
    turn = lambda v: math.atan2(v[0], v[2])                                  # noqa: E731 - toward +x is the lip's lean
    assert turn(f[0]) == pytest.approx(0.0, abs=1.0e-6)
    assert turn(f[-1]) == pytest.approx(HINGE, abs=1.0e-3)
    assert mag.min() >= 0.97 * mag[0], (mag.min(), mag[0])                  # no load dip either


def _trailer_run(seconds: float, spot: tr.Spot, rest: tr.Loads, extra=None, companions=()):
    """Step the trailer on its supports with ``extra(b)`` built beside it; returns per period the trailer pose and
    the joint coordinates."""
    def build(b: newton.ModelBuilder) -> None:
        tr.add_trailer(b, spot)
        if extra is not None:
            extra(b)
    scene = TestScene(build=build, ground=False, worlds=1, dt=lg.STEP, control_dt=lg.PERIOD,
                      companions=(lambda: list(companions)) if companions else None, solver_kwargs=fd.CAPACITY)
    sim = build_simulation(scene, tr.supports(spot, rest), "cpu", worlds=1)
    body = list(sim.model.body_label).index(tr.LABEL)
    out = {"q": [sim.state.body_q.numpy()[body].copy()], "joint_q": [sim.state.joint_q.numpy().copy()]}
    for _ in range(int(round(seconds / lg.PERIOD))):
        sim.step_period()
        out["q"].append(sim.state.body_q.numpy()[body].copy())
        out["joint_q"].append(sim.state.joint_q.numpy().copy())
    labels, starts = list(sim.model.joint_label), sim.model.joint_q_start.numpy()
    return {k: np.asarray(v) for k, v in out.items()}, {name: int(starts[labels.index(name)]) for name in labels}


def _floor_point(q: np.ndarray, local) -> np.ndarray:
    return q[:3] + Rotation.from_quat(q[3:]).apply(np.asarray(local, np.float64))


def _dock(spot: tr.Spot):
    def extra(b: newton.ModelBuilder) -> None:
        fd.add_dock(b)
        fd.add_leveler(b, spot.bed)
    return extra


@pytest.mark.parametrize("bed_in", [44, 48, 52])
def test_trailer_at_a_dock_stands_at_its_spotted_height_with_the_leveler_on_it(bed_in):
    spot = fd.dock_spot(bed_in * tr.IN)
    r, joints = _trailer_run(2.0, spot, fd.dock_rest(spot), _dock(spot))
    for x in (0.0, tr.INSIDE_L):
        assert _floor_point(r["q"][-1], (x, 0.0, 0.0))[2] == pytest.approx(spot.bed, abs=1.0e-3)
    q_deck, q_lip = fd.deployed(spot.bed)
    assert r["joint_q"][-1, joints["leveler_lip"]] == pytest.approx(q_lip, abs=math.radians(0.1))   # on its stop
    assert r["joint_q"][-1, joints["leveler_deck"]] == pytest.approx(q_deck, abs=math.radians(0.2))


def test_yard_trailer_stands_at_its_height_with_the_ramp_plates_on_it():
    spot = tr.Spot(x=20.0, bed=48 * tr.IN)
    r, joints = _trailer_run(2.0, spot, fd.ramp_rest(spot.x), lambda b: fd.add_yard_ramp(b, spot.x, spot.bed))
    for x in (0.0, tr.INSIDE_L):
        assert _floor_point(r["q"][-1], (x, 0.0, 0.0))[2] == pytest.approx(spot.bed, abs=1.0e-3)
    plates = [r["joint_q"][-1, joints[f"ramp_lip_{k}"]] for k in range(fd.PLATES)]
    assert max(abs(p) for p in plates) < math.radians(1.0), plates                # lying on the floor


def _pressed(spot: tr.Spot, force: float, at: float, extra=None, seconds: float = 8.0):
    """A light box on the floor at ``at`` (trailer frame), pressed down by ``force`` ramped in over ~5 s."""
    def build(b: newton.ModelBuilder) -> None:
        box = b.add_body(xform=body_xform(spot.x + at, spot.y, spot.bed + 0.25), label="box")
        b.add_shape_box(box, hx=0.4, hy=0.4, hz=0.25,
                        cfg=newton.ModelBuilder.ShapeConfig(density=50.0 / 0.32, mu=0.8))
        if extra is not None:
            extra(b)
    press = ls.Press(body="box", force=force, rate=force / 5.0, start=0.5)
    return build, press


@SLOW
def test_sag_and_dock_walk_under_a_slow_press_match_the_supports_statics():
    """4 t pressed slowly onto the floor 1 m in from the sill of a held trailer: the sill sinks as the rigid body on its
    springs, and the tandem's trailing arms walk the trailer's ground-level points toward its nose (GA's dock walk)
    by the treads' balance (forklift_trailer.walk)."""
    spot, force, at = tr.Spot(x=0.2, bed=48 * tr.IN), 4000.0 * tr.G, 1.0
    build, press = _pressed(spot, force, at)
    r, _ = _trailer_run(8.0, spot, ((50.0 * tr.G, at),), build, (press,))
    sill = spot.bed - _floor_point(r["q"][-1], (0.0, 0.0, 0.0))[2]
    below_gear = (tr.GEAR_X, 0.0, -spot.bed)                                 # a body point at ground level
    ground = _floor_point(r["q"][-1], below_gear) - _floor_point(r["q"][0], below_gear)
    assert sill == pytest.approx(tr.floor_drop(force, at, 0.0), rel=0.03)
    assert float(ground[0]) == pytest.approx(tr.walk(force, at, ((50.0 * tr.G, at),)), rel=0.05)


def _resting_deck(q: np.ndarray) -> float:
    """The deck angle at which the lip (on its stop) rests its toe's lower corner on the trailer floor at pose ``q``."""
    floor_p = _floor_point(q, (0.0, 0.0, 0.0))
    floor_n = Rotation.from_quat(q[3:]).apply([0.0, 0.0, 1.0])

    def gap(theta: float) -> float:
        lip = theta + fd.LIP_STOP
        x = -fd.DECK_L + fd.DECK_L * math.cos(theta) + fd.LIP_L * math.cos(lip) - fd.LIP_T * math.sin(lip)
        z = fd.DOCK_H - fd.DECK_L * math.sin(theta) - fd.LIP_L * math.sin(lip) - fd.LIP_T * math.cos(lip)
        corner = np.array([x, 0.0, z])
        return float((corner - floor_p) @ floor_n)

    lo, hi = -fd.DECK_TRAVEL, fd.DECK_TRAVEL                                 # gap falls as the deck turns down
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) > 0.0 else (lo, mid)
    return 0.5 * (lo + hi)


@SLOW
def test_leveler_floats_down_with_the_trailer():
    """The same press at a dock: the lip stays on its stop with its toe on the floor, so the deck turns to the angle
    that rests the toe on the sunken, pitched floor."""
    spot, force, at = dataclasses.replace(fd.dock_spot(), x=fd.dock_spot().x + 0.02), 4000.0 * tr.G, 1.0
    build, press = _pressed(spot, force, at, _dock(spot))                     # 2 cm off the bumpers: no rubbing
    rest = (*fd.dock_rest(spot), (50.0 * tr.G, at))
    r, joints = _trailer_run(8.0, spot, rest, build, (press,))
    deck = r["joint_q"][:, joints["leveler_deck"]]
    assert deck[-1] - deck[0] > math.radians(1.0), np.degrees(deck[-1] - deck[0])   # it turned (the floor sank ~7 cm)
    for k in (int(0.5 / lg.PERIOD), -1):                                   # settled before the press; at its end
        assert deck[k] == pytest.approx(_resting_deck(r["q"][k]), abs=math.radians(0.05)), (k, deck[k])
    assert r["joint_q"][-1, joints["leveler_lip"]] == pytest.approx(fd.LIP_STOP, abs=math.radians(0.2))


def _truck_on_dock(spot: tr.Spot, front_x: float, load, seconds: float, inputs=None, **kw):
    """GA's truck (mast welded, forks 15 cm up: lowered they rest on the deck's rise) parked with its front axle at
    world ``front_x`` facing the trailer's nose, at a dock with the leveler and the trailer; the trailer's supports
    take the dock's static loads only."""
    from alloy.tasks import forklift_mast as fm  # noqa: PLC0415
    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    pose = ft.Pose(free=0.15)
    rest = fm.rest_at(pose, load, lg.TRUCK["radial_stiffness"][0])
    at = wp_transform_multiply((front_x, 0.0, spot.bed), rest)

    def build(b: newton.ModelBuilder) -> None:
        _dock(spot)(b)
        tr.add_trailer(b, spot)
        ft.build(pose, load, locked=True, at=at)(b)
    cfg = {**lg.TRUCK, "surfaces": (*fd.DRIVABLE, *tr.DRIVABLE), "resting": fd.RESTING}
    return lg.run(cfg, seconds, inputs or {"brake": 1.0}, build=build, extra=lambda: tr.supports(spot,
                  fd.dock_rest(spot)), track=(tr.LABEL, "leveler_deck", "leveler_lip"), solver_kwargs=fd.CAPACITY,
                  **kw)


def wp_transform_multiply(translation, rest):
    import warp as wp

    return wp.transform_multiply(wp.transform(wp.vec3(*translation), wp.quat_identity()), rest)


@SLOW
def test_laden_truck_at_the_sill_sinks_the_trailer_by_its_supports_statics():
    """R-4 on the dock rig: the truck with 1 t, front axle 1 m inside the trailer and rear axle on the leveler deck. The
    sill sinks by the trailer's statics under the front axle's load plus the leveler's extra tip load, the rear axle's
    load times its distance from the deck's hinge over the hinge-to-toe span (forklift_trailer.floor_drop, linear)."""
    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    spot = dataclasses.replace(fd.dock_spot(), x=fd.dock_spot().x + 0.02)
    front_x = spot.x + 1.0
    r = _truck_on_dock(spot, front_x, ft.Load(1000.0), 3.0)
    tail = r["t"] >= 2.0
    front, rear = (float(r["force"][tail, s, 2].sum(axis=1).mean()) for s in (slice(0, 2), slice(2, 4)))
    q = r[f"{tr.LABEL}.q"]
    sill = spot.bed - float(_floor_point(q[-1], (0.0, 0.0, 0.0))[2])        # run() records from its first period
    span = fd.DECK_L + fd.LIP_L
    rear_s = front_x - lg.L + fd.DECK_L                                      # rear axle from the deck's hinge
    want = (tr.floor_drop(front, front_x - spot.x, 0.0)
            + tr.floor_drop(rear * rear_s / span, fd.LIP_L - spot.x, 0.0))
    assert sill > 0.03, sill
    assert sill == pytest.approx(want, rel=0.03), (sill, want, front, rear)


@SLOW
def test_braked_truck_holds_on_the_sinking_trailer_floor():
    """R-2 on the free trailer: the braked laden truck parked inside the trailer as the floor sinks and pitches under
    it creeps no more than 1 mm along the floor (slip relative to the surface's body)."""
    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    spot = dataclasses.replace(fd.dock_spot(), x=fd.dock_spot().x + 0.02)
    front_x = spot.x + 4.0
    r = _truck_on_dock(spot, front_x, ft.Load(1000.0), 4.0)
    rel = [Rotation.from_quat(qt[3:]).inv().apply(qc[:3] - qt[:3]) for qc, qt in zip(r["q"], r[f"{tr.LABEL}.q"],
                                                                                      strict=True)]
    rel = np.asarray(rel)
    sag = spot.bed - float(_floor_point(r[f"{tr.LABEL}.q"][-1], (4.0, 0.0, 0.0))[2])
    assert sag > 0.03, sag
    settled = r["t"] >= 1.0
    creep = np.abs(rel[settled, 0] - rel[settled, 0][0]).max()
    assert creep <= 1.0e-3, creep


def _cruise_then_brake(speed: float, stop: float):
    def control(t: float, last: dict) -> dict[str, float]:
        if t >= stop:
            return {"throttle": 0.0, "brake": 1.0}
        u = 0.1 + 2.0 * (speed - float(last["qd"][0]))
        return {"throttle": float(np.clip(u, 0.0, 1.0)), "brake": float(np.clip(-u, 0.0, 1.0))}
    return control


@SLOW
def test_laden_truck_drives_over_the_leveler_into_the_trailer():
    """The leveler's deck and lip are ``resting`` faces (left out of the tyres' step check, which bounds a free 90 kg
    lip): the laden truck drives from the dock over them into the trailer at 1 m/s and brakes. Crossing, its wheel
    loads stay within 1.4x their level-floor values (1.22x: the lip's 16 mm toe onto the floor is a real step) and the
    chassis's vertical speed under 0.15 m/s; the step check's bound was a 7 MN blow-up at the pit's edge."""
    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    spot = dataclasses.replace(fd.dock_spot(), x=fd.dock_spot().x + 0.02)
    r = _truck_on_dock(spot, -3.0, ft.Load(1000.0), 6.5, inputs={"brake": 0.0}, control=_cruise_then_brake(1.0, 5.0))
    cruise = (r["t"] > 1.5) & (r["t"] < 4.9)                                 # at 1 m/s, over the deck and lip
    level = r["force"][(r["t"] > 1.0) & (r["t"] < 1.5), :, 2].mean(axis=0)  # on the dock floor
    assert r["q"][-1, 0] > spot.x + 1.5, r["q"][-1, 0]                       # both axles inside the trailer
    peak = r["force"][cruise, :, 2].max(axis=0) / level
    assert (peak <= 1.4).all(), peak
    assert np.abs(r["qd"][cruise, 2]).max() < 0.15, np.abs(r["qd"][cruise, 2]).max()


def test_a_resting_face_must_rest_on_a_checked_surface():
    """WP13: a resting face is left out of the step check, so it must name a support that is checked in its place: a
    drivable surface that is not itself resting, or the world."""
    from alloy.plugins.errors import PluginError  # noqa: PLC0415
    from alloy.plugins.vehicle import Vehicle  # noqa: PLC0415

    base = {**lg.TRUCK, "surfaces": (*fd.DRIVABLE, *tr.DRIVABLE)}
    Vehicle(**base, resting=fd.RESTING)
    Vehicle(**base, resting=(("leveler_deck", "world"),))
    for bad in ((("leveler_lip", "leveler_deck"), ("leveler_deck", "trailer_floor")),     # on a resting face
                (("leveler_lip", "pallet"),), (("pallet", "trailer_floor"),), (("leveler_lip",),)):
        with pytest.raises(PluginError, match="resting"):
            Vehicle(**base, resting=bad)


def _brake_at(speed: float, x_brake: float):
    """Cruise at ``speed`` until the chassis passes ``x_brake``, then brake hard and stay braked."""
    braked = []

    def control(t: float, last: dict) -> dict[str, float]:
        if braked or float(last["q"][0]) >= x_brake:
            braked.append(t)
            return {"throttle": 0.0, "brake": 1.0}
        u = 0.1 + 2.0 * (speed - float(last["qd"][0]))
        return {"throttle": float(np.clip(u, 0.0, 1.0)), "brake": float(np.clip(-u, 0.0, 1.0))}
    return control, braked


def _world(q_body: np.ndarray, local: tuple[float, float, float]) -> np.ndarray:
    """A point fixed in a body (``local``), in the world, per sample."""
    return Rotation.from_quat(q_body[:, 3:]).apply(np.asarray(local)) + q_body[:, :3]


def _in_trailer(q_trailer: np.ndarray, world: np.ndarray) -> np.ndarray:
    """World points (one per sample) in the trailer's frame."""
    return Rotation.from_quat(q_trailer[:, 3:]).inv().apply(world - q_trailer[:, :3])


@SLOW
def test_hard_braking_with_the_steer_wheels_on_the_lip():
    """WP13: the lip lies on the trailer floor held only by friction. The laden truck drives in at 1 m/s and brakes
    hard (both front brakes) as its steer wheels roll onto the lip, stopping with them on it.
    - The braking force surges the trailer about 2 mm toward its nose on its tyres. The lip, pinned to the dock
      through the deck and at its stop, cannot follow along x, so the floor slides under its toe and back: the toe
      stays put in the world within 0.5 mm, and its net slip on the floor once settled is under 0.5 mm (no walk).
    - Once stopped, the steer wheels' tread force holds steady, within 2 % of their load (a ringing presliding spring
      would swing by up to mu times it)."""
    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    spot = dataclasses.replace(fd.dock_spot(), x=fd.dock_spot().x + 0.02)
    rear = -1.649                                                             # steer axle, truck frame
    control, braked = _brake_at(1.0, 0.03 - rear)                             # steer axle 3 cm onto the lip
    r = _truck_on_dock(spot, -1.5, ft.Load(1000.0), 6.5, inputs={"brake": 0.0}, control=control)
    lip_x = fd.LIP_L * math.cos(fd.LIP_STOP)
    stop_axle = float(r["q"][-1, 0]) + rear
    assert 0.0 < stop_axle < lip_x, (stop_axle, lip_x)                        # stopped with the steer axle on it
    at_brake = r["t"] >= braked[0]
    toe = _world(r["leveler_lip.q"], (fd.LIP_L, 0.0, -fd.LIP_T))[at_brake]
    on_floor = _in_trailer(r[f"{tr.LABEL}.q"][at_brake], toe)
    surge = np.ptp(r[f"{tr.LABEL}.q"][at_brake, 0])
    assert surge > 1.0e-3, surge                                              # the floor did move under the toe
    assert np.ptp(toe[:, 0]) <= 0.5e-3, np.ptp(toe[:, 0])
    assert abs(on_floor[-1, 0] - on_floor[0, 0]) <= 0.5e-3, on_floor[[0, -1], 0]
    still = r["t"] >= braked[0] + 1.5
    tread, load = r["force"][still, 2:, 0], r["force"][still, 2:, 2]
    swing = (tread.max(axis=0) - tread.min(axis=0)) / load.mean(axis=0)
    assert (swing <= 0.02).all(), swing


@SLOW
def test_truck_braking_on_a_trailer_with_frictionless_supports_keeps_momentum():
    """R-4 on the trailer: with frictionless supports (both vehicles' μ 1e-6) nothing outside pushes along x, so the
    truck and trailer's x momentum stays what it was as the truck drives along the floor and brakes (float32 rounding
    over 3,200 steps). No leveler: its hinge is fixed to the world."""
    from alloy.tasks import forklift_mast as fm  # noqa: PLC0415
    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    spot = tr.Spot(x=0.0, bed=48 * tr.IN)
    pose, load = ft.Pose(free=0.15), ft.Load(1000.0)
    at = wp_transform_multiply((4.0, 0.0, spot.bed), fm.rest_at(pose, load, lg.TRUCK["radial_stiffness"][0]))
    sups = tr.supports(spot, mu_scale=1.0e-6)

    def build(b: newton.ModelBuilder) -> None:
        tr.add_trailer(b, spot)
        ft.build(pose, load, locked=True, at=at)(b)
    car = __import__("alloy.plugins.vehicle", fromlist=["Vehicle"]).Vehicle(**{**lg.TRUCK, "surfaces": tr.DRIVABLE,
                                                                             "ground_plane": False})
    scene = TestScene(build=build, ground=False, worlds=1, dt=lg.STEP, control_dt=lg.PERIOD,
                      companions=lambda: sups, solver_kwargs=fd.CAPACITY)
    sim = build_simulation(scene, [car], "cpu", worlds=1)
    mass = sim.model.body_mass.numpy()
    control = _cruise_then_brake(1.0, 2.5)
    momentum, peak = [], 0.0
    for p in range(int(4.0 / lg.PERIOD)):
        qd = sim.state.body_qd.numpy()
        for k, v in control(p * lg.PERIOD, {"qd": qd[list(sim.model.body_label).index("forklift")]}).items():
            car.set_input(k, v)
        sim.step_period()
        qd = sim.state.body_qd.numpy()
        momentum.append(float((mass * qd[:, 0]).sum()))                    # body_qd's linear part: the COM's
        peak = max(peak, float(mass[list(sim.model.body_label).index("forklift")] * abs(qd[list(sim.model.body_label)
                                                                                          .index("forklift"), 0])))
    assert peak > 2000.0, peak                                              # it drove (> 0.5 m/s)
    assert max(abs(m) for m in momentum) <= 5.0e-5 * peak, (max(abs(m) for m in momentum), peak)


@pytest.mark.gpu
def test_dock_rig_on_cuda():
    """L4 smoke: the laden truck drives over the leveler into the trailer on CUDA (MuJoCo Warp, float32), as on the
    CPU; prints the wall time per control period at the rig's contact count (the dock, leveler, trailer and truck)."""
    import time

    from alloy.tasks import forklift_truck as ft  # noqa: PLC0415

    spot = dataclasses.replace(fd.dock_spot(), x=fd.dock_spot().x + 0.02)
    _truck_on_dock(spot, -3.0, ft.Load(1000.0), 0.1, device="cuda:0")         # compile and capture
    start = time.perf_counter()
    r = _truck_on_dock(spot, -3.0, ft.Load(1000.0), 6.5, inputs={"brake": 0.0}, control=_cruise_then_brake(1.0, 5.0),
                       device="cuda:0")
    wall = time.perf_counter() - start
    print(f"[ok] dock rig on cuda: {1e3 * wall / len(r['t']):.1f} ms per 20 ms control period (16 steps), "
          f"build and plug-ins included")
    cruise = (r["t"] > 1.5) & (r["t"] < 4.9)
    level = r["force"][(r["t"] > 1.0) & (r["t"] < 1.5), :, 2].mean(axis=0)
    assert r["q"][-1, 0] > spot.x + 1.5, r["q"][-1, 0]
    assert (r["force"][cruise, :, 2].max(axis=0) <= 1.4 * level).all()
