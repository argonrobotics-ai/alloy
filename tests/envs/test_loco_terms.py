# SPDX-License-Identifier: Apache-2.0
"""Legged-robot parts that need no simulation (CPU, NumPy): quaternion maths, command draws, foot-contact buffers,
push timing and the reward terms, each checked on hand-made states against a number worked out by hand."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from alloy.envs import EnvError
from alloy.envs.backend import NumpyBackend
from alloy.envs.loco import CommandSampler, Pusher, Pushes, VelocityCommands
from alloy.envs.loco import terms as T
from alloy.envs.loco.sensors import GRAVITY, LocoSensors
from alloy.envs.loco.state import rotate_inverse
from alloy.envs.state import rotate

XP = NumpyBackend("cpu")
DT = 0.02


class FakeState(dict):
    """The few things a term reads from a state: items, ``xp`` and ``worlds``."""

    def __init__(self, worlds: int, **items: Any) -> None:
        super().__init__({k.replace("__", "."): np.asarray(v, np.float32) for k, v in items.items()})
        self.xp, self.worlds = XP, worlds


def _state(worlds: int = 1, **items: Any) -> FakeState:
    return FakeState(worlds, **items)


def _vec(*rows: Any) -> np.ndarray:
    return np.array(rows, np.float32)


# --- quaternion maths ---------------------------------------------------------------------------------------------

def test_rotate_inverse_undoes_rotate_for_random_rotations() -> None:
    """Rotating a vector and then taking it back into the frame returns the vector."""
    rng = np.random.default_rng(1)
    q = rng.normal(size=(64, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.normal(size=(64, 3)).astype(np.float32)
    assert np.allclose(rotate_inverse(XP, q, rotate(XP, q, v)), v, atol=1e-5)


def test_gravity_in_the_frame_of_a_base_pitched_30_degrees() -> None:
    """Pitch +30 deg about y: R^T applied to (0, 0, -1) is (+sin30, 0, -cos30) (the nose-down side sees +x)."""
    half = math.radians(30) / 2
    q = _vec([0.0, math.sin(half), 0.0, math.cos(half)])
    g = rotate_inverse(XP, q, _vec([0.0, 0.0, -1.0]))[0]
    assert np.allclose(g, [0.5, 0.0, -math.cos(math.radians(30))], atol=1e-6)


# --- commands -----------------------------------------------------------------------------------------------------

def _sampler(spec: VelocityCommands, worlds: int = 4096) -> CommandSampler:
    XP.seed(0)
    return CommandSampler(spec, worlds, DT, XP)


def test_commands_stay_in_range_and_centre_on_the_middle() -> None:
    """Uniform draws: every component inside its range, the mean near the range centre."""
    spec = VelocityCommands(lin_x=(-0.4, 1.0), lin_y=(0.0, 0.5), yaw=(0.2, 1.0), standing=0.0, deadband=0.0)
    cs = _sampler(spec)
    cs.draw(np.ones(4096, bool))
    lo, hi = np.array([-0.4, 0.0, 0.2]), np.array([1.0, 0.5, 1.0])
    assert (cs.value >= lo - 1e-6).all() and (cs.value <= hi + 1e-6).all()
    assert np.allclose(cs.value.mean(0), (lo + hi) / 2, atol=0.02)


def test_standing_share_matches_the_setting_within_three_sigma() -> None:
    """The share of all-zero commands is `standing` (binomial 3 sigma over 4096 draws)."""
    cs = _sampler(VelocityCommands(standing=0.1))
    cs.draw(np.ones(4096, bool))
    share = float((np.abs(cs.value).sum(1) == 0).mean())
    assert abs(share - 0.1) < 3 * math.sqrt(0.1 * 0.9 / 4096)


def test_planar_command_is_zero_or_above_the_deadband() -> None:
    """No drawn planar speed falls in (0, deadband)."""
    cs = _sampler(VelocityCommands(standing=0.0, deadband=0.3))
    cs.draw(np.ones(4096, bool))
    speed = np.linalg.norm(cs.value[:, 0:2], axis=1)
    assert ((speed == 0) | (speed >= 0.3)).all() and (speed == 0).any()


def test_tick_redraws_exactly_the_worlds_at_a_positive_multiple_of_the_period() -> None:
    """With a period of 10 steps, steps 10, 20 and 30 redraw; 0 and 15 do not."""
    cs = _sampler(VelocityCommands(resample_s=0.2, standing=0.0, deadband=0.0), worlds=5)
    assert cs.every == 10
    cs.tick(np.array([0, 10, 20, 15, 30]))
    assert (np.abs(cs.value).sum(1) > 0).tolist() == [False, True, True, False, True]


def test_set_holds_through_draw_and_tick_until_release() -> None:
    """A set command survives draws and ticks, and `release` lets draws change it again."""
    cs = _sampler(VelocityCommands(resample_s=0.2), worlds=8)
    cs.set((0.8, 0.0, 0.0))
    cs.draw(np.ones(8, bool))
    cs.tick(np.full(8, 10))
    assert np.allclose(cs.value, [0.8, 0.0, 0.0])
    cs.release()
    cs.draw(np.ones(8, bool))
    assert not np.allclose(cs.value, [0.8, 0.0, 0.0])


def test_bad_command_settings_raise_with_a_fix() -> None:
    """A reversed range, a wrong-shaped `set` and a bad push spec each raise `EnvError`."""
    with pytest.raises(EnvError):
        VelocityCommands(lin_x=(1.0, -1.0)).validate()
    with pytest.raises(EnvError):
        _sampler(VelocityCommands(), worlds=4).set((1.0, 2.0))
    for bad in (Pushes(every_s=0.0), Pushes(every_s=1.0, duration_s=1.0), Pushes(force=(-1.0, 0.0, 0.0))):
        with pytest.raises(EnvError):
            bad.validate()


# --- pushes (timing only, against a recording stand-in for the sim) -----------------------------------------------

class RecordingSim:
    worlds = 3

    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def apply_wrench(self, body: str, force: Any = None) -> None:
        self.calls.append(np.asarray(force, np.float32))


def test_pusher_pushes_for_the_set_length_each_period_and_not_at_tick_zero() -> None:
    """every 0.5 s / duration 0.1 s at dt 0.02: force held on ticks 25-29, 50-54, none before 25."""
    sim = RecordingSim()
    pusher = Pusher(Pushes(every_s=0.5, duration_s=0.1, force=(80.0, 80.0, 0.0)), sim, "go2/base", DT, XP)
    held = []
    for tick in range(56):
        pusher.before_step(tick)
        held.append(bool(np.abs(pusher.force).max() > 0))
    assert [t for t, h in enumerate(held) if h] == [*range(25, 30), *range(50, 55)]
    assert np.abs(pusher.force).max() == 0


def test_pusher_force_within_bounds_and_different_per_world() -> None:
    """Each component is within the limit, the z component is zero, and worlds get different forces."""
    sim = RecordingSim()
    pusher = Pusher(Pushes(every_s=0.5, duration_s=0.1, force=(80.0, 40.0, 0.0)), sim, "go2/base", DT, XP)
    pusher.before_step(25)
    f = pusher.force
    assert (np.abs(f[:, 0]) <= 80).all() and (np.abs(f[:, 1]) <= 40).all() and (f[:, 2] == 0).all()
    assert len({tuple(r) for r in f.round(3)}) == 3


def test_no_push_spec_never_applies_a_wrench() -> None:
    """With `pushes=None` the pusher stays silent and its force stays zero."""
    sim = RecordingSim()
    pusher = Pusher(None, sim, "go2/base", DT, XP)
    for tick in range(100):
        pusher.before_step(tick)
    assert sim.calls == [] and not pusher.force.any()


# --- foot contact buffers -----------------------------------------------------------------------------------------

def _sensor_state(height: float) -> FakeState:
    return _state(1, feet__height=[[height]], feet__pos=[[[0, 0, height]]], base__lin_vel_w=[[0, 0, 0]],
                  base__quat_xyzw=[[0, 0, 0, 1]],
                  joints__vel=[[0, 0]])


def test_touchdown_fires_once_with_the_flight_time() -> None:
    """Contact, 3 steps in the air, landing: one touchdown and a reported air time of 3 * dt."""
    sensors = LocoSensors(1, 1, 2, DT, XP)
    sensors.reset(_sensor_state(0.0), np.ones(1, bool))
    seen = []
    for h in (0.0, 0.05, 0.05, 0.05, 0.0, 0.0):
        sensors.advance(_sensor_state(h))
        ex = sensors.extra()
        seen.append(tuple(float(ex[k][0, 0]) for k in ("feet.contact", "feet.touchdown", "feet.air_time")))
    assert [s[0] for s in seen] == [1, 0, 0, 0, 1, 1]
    assert [s[1] for s in seen] == [0, 0, 0, 0, 1, 0]
    assert seen[4][2] == pytest.approx(3 * DT) and seen[5][2] == 0.0


def test_accelerometer_reads_plus_g_up_at_rest() -> None:
    """A base at rest and level reads (0, 0, +9.81)."""
    sensors = LocoSensors(1, 1, 2, DT, XP)
    sensors.reset(_sensor_state(0.0), np.ones(1, bool))
    sensors.advance(_sensor_state(0.0))
    assert np.allclose(sensors.extra()["base.accel_b"], [[0.0, 0.0, GRAVITY]], atol=1e-5)


# --- reward terms -------------------------------------------------------------------------------------------------

def test_track_lin_vel_is_one_when_exact_and_gaussian_in_the_error() -> None:
    """exp(-err^2 / 0.25): 1 for an exact match, exp(-1) when the base is 0.5 m/s short."""
    s = _state(2, command=[[0.5, 0, 0], [0.5, 0, 0]], base__lin_vel_b=[[0.5, 0, 0], [0.0, 0, 0]])
    assert np.allclose(T.track_lin_vel()(s), [1.0, math.exp(-1.0)], atol=1e-6)


def test_track_yaw_rate_is_one_when_exact_and_gaussian_in_the_error() -> None:
    """exp(-err^2 / 0.25): 1 at a match, exp(-1) when the yaw rate is 0.5 rad/s short."""
    s = _state(2, command=[[0, 0, 0.7], [0, 0, 0.7]], base__ang_vel_b=[[0, 0, 0.7], [0, 0, 0.2]])
    assert np.allclose(T.track_yaw_rate()(s), [1.0, math.exp(-1.0)], atol=1e-6)


def test_action_rate_is_the_sum_of_squared_action_changes() -> None:
    """(1 - 0)^2 + (0 - 2)^2 = 5."""
    s = _state(1, action=[[1.0, 0.0, 3.0]], action__prev=[[0.0, 2.0, 3.0]])
    assert float(T.action_rate()(s)[0]) == pytest.approx(5.0)


def _air_state(command: list[float]) -> FakeState:
    return _state(1, command=[command], feet__air_time=[[0.4, 0.9]], feet__touchdown=[[1.0, 0.0]])


def test_feet_air_time_pays_the_flight_beyond_the_threshold_only_when_moving() -> None:
    """A landing after 0.4 s earns 0.4 - 0.2 under a moving command, and nothing under a standing one."""
    assert float(T.feet_air_time()(_air_state([0.6, 0.0, 0.0]))[0]) == pytest.approx(0.2, abs=1e-6)
    assert float(T.feet_air_time()(_air_state([0.0, 0.0, 0.0]))[0]) == 0.0


def test_body_contacts_is_the_count_of_touching_parts() -> None:
    """The term passes the sensor's count through: 0 for a clean stance, 2 with two parts down."""
    assert T.body_contacts()(_state(2, body__contacts=[0.0, 2.0])).tolist() == [0.0, 2.0]


def test_pose_terms_measure_distance_from_the_stand() -> None:
    """pose_deviation sums every squared joint angle; hip_deviation only each leg's first joint."""
    q = np.zeros((1, 12), np.float32)
    q[0, 0], q[0, 3], q[0, 1] = 0.2, -0.1, 0.5
    s = _state(1, joints__pos=q)
    assert float(T.pose_deviation()(s)[0]) == pytest.approx(0.04 + 0.01 + 0.25, abs=1e-6)
    assert float(T.hip_deviation()(s)[0]) == pytest.approx(0.04 + 0.01, abs=1e-6)


def test_base_height_and_high_feet_terms() -> None:
    """base_height is the squared error to its target; feet_too_high counts only the part above the limit."""
    s = _state(1, base__height=[0.25], feet__height=[[0.02, 0.10, 0.15, 0.30]])
    assert float(T.base_height(0.28)(s)[0]) == pytest.approx(0.03 ** 2, abs=1e-7)
    assert float(T.feet_too_high(0.10)(s)[0]) == pytest.approx(0.05 + 0.20, abs=1e-6)


def test_feet_air_time_is_capped() -> None:
    """A 2 s flight pays as a 0.5 s one: 0.5 - 0.2."""
    s = _state(1, command=[[0.6, 0.0, 0.0]], feet__air_time=[[2.0]], feet__touchdown=[[1.0]])
    assert float(T.feet_air_time()(s)[0]) == pytest.approx(0.3, abs=1e-6)


def test_feet_slide_counts_planar_speed_of_touching_feet_only() -> None:
    """Two feet at 1 m/s and 2 m/s sideways: only the touching one counts, as its speed squared."""
    s = _state(1, feet__vel=[[[1.0, 0, 5.0], [0, 2.0, 0]]], feet__contact=[[1.0, 0.0]])
    assert float(T.feet_slide()(s)[0]) == pytest.approx(1.0, abs=1e-6)
    s = _state(1, feet__vel=[[[1.0, 0, 5.0], [0, 2.0, 0]]], feet__contact=[[1.0, 1.0]])
    assert float(T.feet_slide()(s)[0]) == pytest.approx(5.0, abs=1e-6)


def test_fell_below_height_or_beyond_sixty_degrees_but_not_upright() -> None:
    """Low (0.10 m) and tilted 70 deg are failures; upright at 0.28 m and tilted 50 deg are not."""
    tilt = lambda deg: [math.sin(math.radians(deg)), 0.0, -math.cos(math.radians(deg))]  # noqa: E731
    s = _state(4, base__height=[0.28, 0.10, 0.28, 0.28],
               base__gravity_b=[tilt(0), tilt(0), tilt(70), tilt(50)])
    assert T.fell()(s).tolist() == [False, True, True, False]


def test_flat_orientation_is_zero_when_level_and_sin_squared_when_tilted() -> None:
    """Zero for gravity straight down; sin^2(30 deg) = 0.25 for a 30 deg tilt."""
    s = _state(2, base__gravity_b=[[0, 0, -1], [0.5, 0, -math.cos(math.radians(30))]])
    assert np.allclose(T.flat_orientation()(s), [0.0, 0.25], atol=1e-6)


def test_commands_set_takes_lists_and_arrays() -> None:
    """set accepts a plain tuple, a NumPy array per world, and refuses a wrong shape."""
    sampler = CommandSampler(VelocityCommands(), 3, DT, XP)
    sampler.set((0.5, 0.0, -0.2))
    assert np.allclose(sampler.value, [[0.5, 0.0, -0.2]] * 3)
    per_world = np.arange(9, dtype=np.float64).reshape(3, 3)
    sampler.set(per_world)
    assert sampler.value.dtype == np.float32 and np.allclose(sampler.value, per_world)
    with pytest.raises(EnvError):
        sampler.set((1.0, 2.0))
