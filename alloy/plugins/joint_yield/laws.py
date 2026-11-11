# SPDX-License-Identifier: Apache-2.0
"""Closed-form laws of joint springs that yield (SCR-WP7-15 P2), run through the public API on one MuJoCo route.

The rig: one joint with a MuJoCo spring (stiffness k, springref 0) and a velocity servo (``<velocity kv>``), gravity
off. A segment drives the joint at a speed v for some seconds and is then released: the servo's target speed goes to
0, so it only damps, and the joint comes to rest where its spring torque is zero, at its spring reference s0 + q_p.

Closed form (``sets``). Isotropic hardening, linear: along one loading segment in direction σ = ±1, with q_e the
extreme step coordinate measured from s0 and q_r, ε̄p the set and plastic strain before it, the set grows by
    d = max(0, (k σ (q_e - q_r) - τ_y0 - k_h ε̄p) / (k + k_h)),   q_r += σ d,   ε̄p += d.
This is exact for the 1-D return in any number of steps along a monotone path (the return composes), so the joint
rests at s0 + q_r after each release. q_e is the extreme of the recorded step coordinates since the last release,
over the drive and the release (the joint overshoots after the drive stops); the control period is one MuJoCo step
so that every step coordinate is recorded.

- **J1**, a torsion spring cycled past yield (hinge, with hardening): forward overload, release (permanent set),
  reverse overload, release (reverse yield at -(τ_y0 + k_h ε̄p)), an elastic poke inside the yield band, release
  (no new set).
- **J2**, a clip overloaded and released (slide, perfectly plastic): the set is q_max - F_y / k.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

SETTLE_S = 1.2          # s of release per segment: the servo's damping settles the joint to its spring reference
STEP_S = 0.00125        # s: the control period is one MuJoCo step, so every step coordinate is recorded


@dataclass(frozen=True)
class Rig:
    """One yielding joint: ``kind`` "hinge" or "slide", spring k, yield ``limit``, ``hardening``, servo ``kv``,
    moving mass (kg; a 0.2 m arm for a hinge), its drive ``segments`` [(speed, seconds), ...] and the build spring
    reference ``springref`` s0 (a preloaded spring when it is not 0)."""
    kind: str
    k: float
    limit: float
    hardening: float
    kv: float
    mass: float
    segments: tuple[tuple[float, float], ...]
    springref: float = 0.0

    def mjcf(self) -> str:
        if self.kind == "hinge":
            joint = f'<joint name="j" type="hinge" axis="0 0 1" stiffness="{self.k}" springref="{self.springref}"/>'
            geom = f'<geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="{self.mass}"/>'
        else:
            joint = f'<joint name="j" type="slide" axis="1 0 0" stiffness="{self.k}" springref="{self.springref}"/>'
            geom = f'<geom type="box" size="0.02 0.02 0.02" mass="{self.mass}"/>'
        return (f'<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody><body name="b" pos="0 0 0.1">'
                f'{joint}{geom}</body></worldbody><actuator><velocity joint="j" kv="{self.kv}"/></actuator>'
                f'</mujoco>')


J1_RIG = Rig("hinge", k=2.0, limit=0.02, hardening=0.05, kv=0.25, mass=0.5,     # about critically damped
             segments=((0.5, 0.6), (-0.5, 1.4), (0.2, 0.1)))
J2_RIG = Rig("slide", k=500.0, limit=2.0, hardening=0.0, kv=50.0, mass=0.5, segments=((0.1, 0.25),),
             springref=-0.002)                     # a latch spring preloaded 1 N at q = 0 (yield 2 N)


@dataclass
class LawRows:
    """Measured and closed-form rest coordinates after each release, per world, and the run's health."""
    extremes: np.ndarray                 # [segments, W]
    rests: np.ndarray                    # [segments, W] measured
    closed: np.ndarray                   # [segments, W]
    plastic_strain: np.ndarray           # [W] ε̄p read back at the end
    closed_strain: np.ndarray            # [W]
    ok: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def err(self) -> float:
        return float(np.abs(self.rests - self.closed).max())


def sets(extremes: np.ndarray, signs: np.ndarray, k: float, limit: float, hardening: float) -> tuple[np.ndarray,
                                                                                                    np.ndarray]:
    """The closed-form set after each segment (module doc) and the final ε̄p; ``extremes`` [segments, W], measured
    from the build spring reference s0."""
    q_r = np.zeros(extremes.shape[1])
    ebar = np.zeros(extremes.shape[1])
    out = np.zeros_like(extremes, dtype=np.float64)
    for i, (q_e, s) in enumerate(zip(extremes, signs)):
        d = np.maximum(0.0, (k * s * (q_e - q_r) - limit - hardening * ebar) / (k + hardening))
        q_r, ebar = q_r + s * d, ebar + d
        out[i] = q_r
    return out, ebar


def run(rig: Rig, *, worlds: int = 2, device: str = "cpu", spread: float = 0.5, dt: float = STEP_S) -> LawRows:
    """The rig through ``alloy``: world w drives at (1 - spread w / (W - 1)) times each segment's speed. ``dt`` is the
    control period; at more than one MuJoCo step per period the frames miss step coordinates, so ``closed`` (from the
    recorded extremes) is exact only at ``STEP_S``, while the joint still yields every step."""
    import alloy  # noqa: PLC0415

    scene = alloy.Scene(worlds=worlds, dt=dt)
    scene.add_robot("rig", mjcf=rig.mjcf(), pos=(0, 0, 0.5))
    key = "torque" if rig.kind == "hinge" else "force"
    scene.set_joint_yield("rig/j", **{key: rig.limit}, hardening=rig.hardening)
    sim = alloy.build(scene, device=device)
    scale = 1.0 - spread * np.arange(worlds) / max(worlds - 1, 1)
    extremes, rests = [], []
    for speed, seconds in rig.segments:
        sim.set_targets({"rig/j": speed * scale}, kind="velocity")
        drive = sim.run(seconds, record=["rig/j"])["rig/j.q"]
        sim.set_targets({"rig/j": np.zeros(worlds)}, kind="velocity")
        release = sim.run(SETTLE_S, record=["rig/j"])["rig/j.q"]
        path = np.concatenate([drive, release])
        extremes.append(path.max(axis=0) if speed > 0 else path.min(axis=0))
        rests.append(release[-1])
    ext = np.asarray(extremes, np.float64)
    closed, ebar = sets(ext - rig.springref, np.sign([s for s, _ in rig.segments]), rig.k, rig.limit, rig.hardening)
    strain = sim.state(["rig/j"])["rig/j.plastic_strain"]
    return LawRows(ext, np.asarray(rests, np.float64), rig.springref + closed, strain, ebar, bool(sim.check()["ok"]),
                   {"route": type(sim.engine.solvers[0]).__name__})
