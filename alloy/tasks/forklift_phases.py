# SPDX-License-Identifier: Apache-2.0
"""Phase types for the forklift operator: a path-following drive and a stationary hold/wait.

Every phase has ``start(obs) -> Run`` and ``act(run, obs, spec) -> (Controls, finished)``. The ``Operator`` engine
only sequences them. A phase can servo the mast (lift, tilt, sideshift) while it does its main job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from alloy.tasks.forklift_control import (REST_SPEED, Controls, Observation, TruckSpec, clip, kmh, servo_command,
                                          servo_reached, speed_commands, speed_profile, SPEED_MARGIN)
from alloy.tasks.forklift_path import Path, Segment, pursue

ARRIVE_DISTANCE = 0.006      # m: a stopping segment brakes hard once this close to its end
PASS_DISTANCE = 0.05         # m: a non-stopping segment hands over this close to its end
DWELL = 0.25                 # s: conditions must hold this long before a phase finishes
INTEGRAL_RATE, INTEGRAL_LIMIT = 1.0, 1.0      # speed integrator: gain (1/s) and clamp (m/s²)
INTEGRAL_BAND = 0.3                           # m/s: integrate only when the speed error is this small
LATERAL_ACCEL, MIN_CURVE_SPEED = 0.6, 0.5   # m/s², m/s: curves are taken slower
ACCEL_LIMIT = 0.8                             # m/s²: how fast the speed reference may rise


@dataclass(frozen=True)
class Targets:
    """Mast targets a phase servos to (None = leave that axis alone)."""

    lift: float | None = None
    tilt: float | None = None
    sideshift: float | None = None


@dataclass
class Run:
    """Mutable per-phase state."""

    t0: float
    last_t: float
    path: Path | None = None
    index: int = 0
    integral: float = 0.0
    v_ref: float = 0.0
    ok_since: float | None = None
    level: float | None = None     # SetDown: the fork height last period
    opening: float = 0.0           # SetDown: the lowering valve's opening
    stalled: float = 0.0           # SetDown: seconds the forks have not come down while lowering


def mast_commands(targets: Targets, obs: Observation, spec: TruckSpec) -> tuple[float, float, float, bool]:
    """(lift, tilt, sideshift, all reached) servo commands for the targets that are set."""
    axes = ((targets.lift, obs.fork_height, spec.lift_servo), (targets.tilt, obs.tilt, spec.tilt_servo),
            (targets.sideshift, obs.sideshift, spec.sideshift_servo))
    cmds = [0.0 if tgt is None else servo_command(tgt, val, sv) for tgt, val, sv in axes]
    done = all(tgt is None or servo_reached(tgt, val, sv) for tgt, val, sv in axes)
    return cmds[0], cmds[1], cmds[2], done


def dwelled(run: Run, ok: bool, t: float, dwell: float = DWELL) -> bool:
    """True once ``ok`` has held for ``dwell`` seconds."""
    if not ok:
        run.ok_since = None
        return False
    if run.ok_since is None:
        run.ok_since = t
    return t - run.ok_since >= dwell


@dataclass(frozen=True)
class DriveSegment:
    """Follow one segment; ``stop`` brakes to rest at its last waypoint, else it hands over at speed."""

    name: str
    segment: Segment
    targets: Targets = Targets()
    stop: bool = True
    next_cap_kmh: float = 0.0          # for a non-stopping segment: the next segment's cap (sets the exit speed)

    @property
    def cap(self) -> float:
        """Speed cap in m/s."""
        return kmh(self.segment.cap_kmh)

    def curve_cap(self, run: Run) -> float:
        """The segment cap, lowered to keep lateral acceleration under ``LATERAL_ACCEL`` on the curves ahead."""
        kappa = run.path.curvature_ahead(run.index)
        return min(self.cap, max(math.sqrt(LATERAL_ACCEL / kappa), MIN_CURVE_SPEED)) if kappa > 1e-3 else self.cap

    def start(self, obs: Observation) -> Run:
        return Run(obs.t, obs.t, path=Path.from_segment(self.segment))

    def act(self, run: Run, obs: Observation, spec: TruckSpec) -> tuple[Controls, bool]:
        pursuit = pursue(run.path, obs.x, obs.y, obs.yaw, run.index, spec)
        run.index = pursuit.index
        lift, tilt, side, _ = mast_commands(self.targets, obs, spec)
        direction = self.segment.direction
        common = dict(direction=direction, lift=lift, tilt=tilt, sideshift=side, key_on=True, engine_on=True)
        if self.stop and pursuit.remaining <= ARRIVE_DISTANCE:
            held = Controls(throttle=0.0, brake=1.0, steer=0.0, **common)
            return held, dwelled(run, abs(obs.speed) < REST_SPEED, obs.t)
        exit_speed = 0.0 if self.stop else kmh(self.next_cap_kmh) * SPEED_MARGIN
        dt = clip(obs.t - run.last_t, 0.0, 0.1)
        run.last_t = obs.t
        before = run.v_ref
        ceiling = max(before, abs(obs.speed)) + ACCEL_LIMIT * dt            # the reference speeds up gently
        v_ref = run.v_ref = min(speed_profile(pursuit.remaining, self.curve_cap(run), exit_speed), ceiling)
        accel_ff = clip((v_ref - before) / dt, -2.0, ACCEL_LIMIT) if dt > 0 else 0.0
        error = direction * v_ref - obs.speed
        if abs(error) < INTEGRAL_BAND:
            run.integral = clip(run.integral + INTEGRAL_RATE * error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
        bias = run.integral + direction * accel_ff
        throttle, brake = speed_commands(v_ref, obs.speed, direction, obs.grade, bias, spec)
        ctrl = Controls(throttle=throttle, brake=brake, steer=pursuit.steer, **common)
        return ctrl, (not self.stop) and pursuit.remaining <= PASS_DISTANCE


@dataclass(frozen=True)
class TipsDown:
    """Lower the forks fully, then inch the mast forward until the blades slope down to their tips (pitch <= 0) with
    the tips within ``tip`` of the floor, as a driver tilts until the tips touch (the stop is degrees further: tilting
    on would pry the truck up on its tips). Brake held; ends at rest after ``dwell``."""

    name: str
    rate: float = 0.03             # tilt valve opening while inching (a few hundredths of a degree per period)
    tip: float = 0.003             # m
    dwell: float = 0.3

    def start(self, obs: Observation) -> Run:
        return Run(obs.t, obs.t)

    def act(self, run: Run, obs: Observation, spec: TruckSpec) -> tuple[Controls, bool]:
        lift, _, _, lowered = mast_commands(Targets(lift=0.0), obs, spec)
        down = obs.blade_pitch <= 0.0 and obs.tip_height <= self.tip
        stop = obs.tilt >= spec.tilt_limit
        tilt = self.rate if lowered and not (down or stop) else 0.0
        ctrl = Controls(throttle=0.0, brake=1.0, lift=lift, tilt=tilt, key_on=True, engine_on=True)
        return ctrl, dwelled(run, lowered and (down or stop) and abs(obs.speed) < REST_SPEED, obs.t, self.dwell)


@dataclass(frozen=True)
class SetDown:
    """Lower the forks to ``targets.lift`` at about ``speed`` (m/s), so a load lands gently on its support (GA's
    set-down limit is 0.10 m/s at onset): the lowering valve's opening is trimmed each period by the measured speed,
    as a driver feathers the lever. Lowered fully, the blades come to rest on a pallet's bottom boards before the
    mast's stop: the forks no longer coming down for ``stall`` s ends the lowering too. Tilt and side shift are held
    at their targets; brake held; ends at rest after ``dwell``."""

    name: str
    targets: Targets
    speed: float = 0.05            # half GA's limit: the landing reads the speed just before it
    first: float = 0.02            # opening of the first period
    stall: float = 0.5             # s
    dwell: float = 0.3

    def start(self, obs: Observation) -> Run:
        return Run(obs.t, obs.t)

    def act(self, run: Run, obs: Observation, spec: TruckSpec) -> tuple[Controls, bool]:
        _, tilt, side, held = mast_commands(Targets(tilt=self.targets.tilt, sideshift=self.targets.sideshift), obs,
                                            spec)
        target = spec.lift_servo.lo if self.targets.lift is None else self.targets.lift
        above = obs.fork_height > target + spec.lift_servo.deadband
        if run.level is not None and obs.t > run.last_t and run.opening > 0.0:
            v = (run.level - obs.fork_height) / (obs.t - run.last_t)
            stalled = v < 0.1 * self.speed
            run.stalled = run.stalled + (obs.t - run.last_t) if stalled else 0.0
            if not stalled:                                 # held while stalled: no full valve if the boards let go
                run.opening = clip(run.opening * clip(self.speed / max(v, 1.0e-3), 0.5, 1.5) ** 0.25, 0.002, 1.0)
        run.level, run.last_t = obs.fork_height, obs.t
        above = above and run.stalled < self.stall
        run.opening = (run.opening or self.first) if above else 0.0
        ctrl = Controls(throttle=0.0, brake=1.0, lift=-run.opening, tilt=tilt, sideshift=side, key_on=True,
                        engine_on=True)
        return ctrl, dwelled(run, not above and held and abs(obs.speed) < REST_SPEED, obs.t, self.dwell)


def always(obs: Observation) -> bool:
    return True


def restraint_not_red(obs: Observation) -> bool:
    """The trailer restraint is green, or there is no restraint system."""
    return obs.restraint_green is not False


@dataclass(frozen=True)
class Stationary:
    """Hold still (brake on, optional park), servo the mast, wait for ``until``, then dwell and finish."""

    name: str
    targets: Targets = Targets()
    key_on: bool = True
    engine_on: bool = True
    park: bool = False
    hold_brake: bool = True
    direction: int = 0
    dwell: float = 0.3
    until: Callable[[Observation], bool] = field(default=always)

    def start(self, obs: Observation) -> Run:
        return Run(obs.t, obs.t)

    def act(self, run: Run, obs: Observation, spec: TruckSpec) -> tuple[Controls, bool]:
        lift, tilt, side, reached = mast_commands(self.targets, obs, spec)
        at_rest = abs(obs.speed) < REST_SPEED
        ctrl = Controls(throttle=0.0, brake=1.0 if self.hold_brake else 0.0, park=self.park and at_rest,
                        direction=self.direction, lift=lift, tilt=tilt, sideshift=side,
                        key_on=self.key_on, engine_on=self.engine_on)
        return ctrl, dwelled(run, reached and at_rest and self.until(obs), obs.t, self.dwell)
