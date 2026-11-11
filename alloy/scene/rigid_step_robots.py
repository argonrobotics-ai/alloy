# SPDX-License-Identifier: Apache-2.0
"""Robots under the rigid step (``alloy.scene.rigid_step``): when a robot scene may leave the fixed step, and its caps.

Opt-in only. A scene with robots keeps the fixed step unless the user asked for another one: ``scene.rigid_step`` set
to a step in seconds, or set to "auto" (``RigidStepRequest.explicit``; the never-set default and a profile's request do
not count). The fast mode is not offered for robots yet. Every robot must be on the MuJoCo entry and the scene must
have no pour: a scene with a second dynamic entry (cloth, liquid, sand, soft bodies) steps on the exchange clock,
which never reads this step, so it stays refused and its report names the fixed step. The other refusals
(attachments, kinematic objects, precision pairs, objects off the MuJoCo entry) are ``rigid_step.why_not``'s.

Caps, each a function (facts, dt) -> None or why, on the facts ``rigid_step_robots_facts.collect`` reads from the
template builder:
- servo: omega dt <= max(SERVO_C, zeta) for every position servo, omega = sqrt(kp / I), zeta = (kv + joint damping)
  / (2 sqrt(kp I)), I the joint's own inertia at the template pose (the diagonal of the joint-space inertia: its
  subtree about the axis, plus armature); ``cap_servo`` has the numbers.
- damping: a robot entry integrates joint damping and a servo's kv implicitly, which dilutes a contact's push on the
  joint by 1 / (1 + dt b / I), b = kv + joint damping. (1 + dt b / I) may be at most DAMPING_DILUTION_MAX times its
  value at the fixed step: a link's contact sinks at most 25 % deeper than at the fixed step.
- solref: dt <= tau / 2 for the time constant of every robot contact shape, joint limit and equality row the file
  writes. MuJoCo raises a smaller tau to 2 dt (refsafe), so the row gets softer by (tau / 2 dt)^2: a grasp squeeze
  sinks deeper. Arm files written for 1.25 ms (tau 2.5 ms) therefore keep the fixed step under "auto".
- link_gap: v dt <= the robot's contact gap: a link travels at most its contact gap per step, so a contact starts
  at most that deep; v the speed of the
  farthest collision point below a joint at a declared joint speed (LINK_SPEED_RAD_S for a hinge, LINK_SPEED_M_S for
  a slide, or the file's joint velocity limit when that is lower).
- loose_body: a loose free body in a robot file (a box or a mug beside an arm) gets its contact time constant from
  the light-body rule (``alloy.contact.light_body``), whose stiffness is capped by the step; the time constant its
  contact acts at (that rule's, or 2 dt when larger) must not be above the one at the fixed step.
The free bodies beside the robots keep ``rigid_step.CAPS``. "auto" takes the largest grid step every cap allows; a
pinned step is taken as asked and the build warns once, naming each broken cap, the joint or shape, and the cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.contact.light_body import light_solref

SERVO_C = 1.0                   # servo cap: omega dt at most max(this, zeta) (``cap_servo``). Each joint alone, at the
                                # template pose: a folded arm has a smaller inertia and the coupled servo system a
                                # stiffer mode (up to 1.73x on the shipped robots); the cap sits 2x or more below the
                                # stability limit in omega dt, which covers both
DAMPING_DILUTION_MAX = 1.25     # damping cap: (1 + dt b / I) at most this times its value at the fixed step
POUR = "robots beside a pour keep the fixed step (the liquid's exchange clock sets the step)"
LINK_SPEED_RAD_S = 2.0          # link_gap cap: the declared hinge speed, one joint at a time (several joints moving
                                # together give a faster tip: a first contact starts deeper; stability is not touched)
LINK_SPEED_M_S = 0.5            # link_gap cap: the declared slide speed
ROUNDING = 1.0e-6               # relative slack of every cap: float32 file values
PLURAL = {"contact": "contacts", "joint limit": "joint limits", "equality": "equality rows"}
KEEPS = "robots keep the fixed step (scene.rigid_step = 'auto' or a step in seconds lets them step coarser)"
NO_FAST = "robots: the fast mode is not offered yet (scene.rigid_step = 'auto' or a step in seconds)"


@dataclass(frozen=True)
class Servo:
    """One position servo: ``kp`` [N m/rad, N/m] on a joint of inertia ``inertia`` [kg m2, kg] (armature included);
    ``damping`` [N m s/rad, N s/m]: its kv plus the joint's own damping."""

    name: str
    kp: float
    inertia: float
    damping: float = 0.0

    @property
    def omega(self) -> float:
        """Its natural frequency [rad/s]."""
        return math.sqrt(self.kp / self.inertia) if self.inertia > 0.0 else math.inf

    @property
    def zeta(self) -> float:
        """Its damping ratio (kv + joint damping) / (2 sqrt(kp I))."""
        return 0.5 * self.damping / math.sqrt(self.kp * self.inertia) if self.inertia > 0.0 else 0.0


@dataclass(frozen=True)
class Damped:
    """One robot joint whose links can touch something: ``damping`` [N m s/rad, N s/m] (kv + joint damping) on its
    inertia ``inertia`` [kg m2, kg]."""

    name: str
    damping: float
    inertia: float

    def dilution(self, dt: float) -> float:
        """1 + dt b / I: how much less a contact's push moves the joint at step ``dt`` than with no damping."""
        return 1.0 + dt * self.damping / self.inertia if self.inertia > 0.0 else 1.0


@dataclass(frozen=True)
class Row:
    """One constraint row's time constant ``tau`` [s]; ``kind``: "contact", "joint limit" or "equality"."""

    name: str
    kind: str
    tau: float


@dataclass(frozen=True)
class Loose:
    """One contact shape of a loose free body in a robot file: the file's ``tau`` [s] and damping ratio ``zeta``, the
    shape's impedance ``dmax`` and the body's ``mass`` [kg], as the light-body rule reads them."""

    name: str
    tau: float
    zeta: float
    dmax: float
    mass: float

    def acting(self, dt: float) -> float:
        """The time constant [s] the contact acts at when stepped at ``dt``: the light-body rule's, or MuJoCo's
        2 dt when that is larger."""
        row = light_solref(np.array([[self.tau, self.zeta]]), np.array([self.mass]), np.array([self.dmax]), dt, False)
        return max(float(row[0, 0]), 2.0 * dt)


@dataclass(frozen=True)
class Link:
    """The fastest collision point below a joint: ``speed`` [m/s] at the declared joint speed, against the robot's
    contact ``gap`` [m]; ``reach`` [m] from a hinge's axis (0 for a slide)."""

    name: str
    speed: float
    gap: float
    reach: float = 0.0


@dataclass(frozen=True)
class RobotFacts:
    """What the robot caps read; ``fixed``: the fixed step [s], the baseline of the predicted costs."""

    fixed: float
    servos: tuple[Servo, ...] = ()
    rows: tuple[Row, ...] = ()
    links: tuple[Link, ...] = ()
    loose: tuple[Loose, ...] = ()
    damped: tuple[Damped, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.servos or self.rows or self.links or self.loose or self.damped)


def over(x: float, limit: float) -> bool:
    """``x`` above ``limit`` beyond float32 rounding (the builder keeps the file's values in single precision: a
    2.5 ms time constant reads 2.49999994 ms), so a bound met exactly passes."""
    return x > limit * (1.0 + ROUNDING)


# --- eligibility ------------------------------------------------------------------------------------------------

def refusal(request: Any, scene: Any = None) -> str | None:
    """Why a scene with robots keeps the fixed step under ``request`` (a ``RigidStepRequest``), or None when the
    user asked for another step (a number, or an explicit "auto") and ``scene`` has no pour."""
    mode = getattr(request, "mode", None)
    asked = (isinstance(mode, (int, float)) and not isinstance(mode, bool)) or (
        mode == "auto" and getattr(request, "explicit", False))
    if asked:
        return POUR if getattr(scene, "emitters", None) else None
    if mode == "fast" and not getattr(request, "from_profile", False):
        return NO_FAST
    return KEEPS


def robot_why_not(t: Any, request: Any) -> str | None:
    """Why robot template ``t`` keeps the scene at the fixed step, or None."""
    why = refusal(request)
    if why is not None:
        return why
    entry = t.decision.entry
    return None if entry == "mujoco" else f"a robot on the {entry or 'static'} entry keeps the fixed step"


def stability_limit(zeta: float) -> float:
    """The omega dt below which a servo is stable on a robot entry: 2 zeta + 2 sqrt(zeta^2 + 1)."""
    return 2.0 * zeta + 2.0 * math.sqrt(zeta * zeta + 1.0)


# --- caps -------------------------------------------------------------------------------------------------------

def softening(tau: float, dt: float, fixed: float) -> float:
    """Stiffness of a row with time constant ``tau`` at step ``dt`` over its stiffness at the fixed step: MuJoCo uses
    max(tau, 2 dt), and stiffness goes as 1 / tau^2."""
    return (max(tau, 2.0 * fixed) / max(tau, 2.0 * dt)) ** 2


def cap_servo(f: RobotFacts, dt: float) -> str | None:
    """omega dt <= max(SERVO_C, zeta). A robot entry (implicitfast) takes kv and joint damping implicitly and kp
    explicitly: v+ = (v - dt omega^2 x) / (1 + 2 zeta omega dt), x+ = x + dt v+, stable while omega dt <
    2 zeta + 2 sqrt(zeta^2 + 1) (2 at zeta 0, 4.83 at 1, 20.2 at 5). The error is first order in the step: at
    omega dt = 1 a step response is off by 8.7 % of the step at zeta 1 (23 % at zeta 0.3). An overdamped servo
    follows its slow pole kp / kv, so it may go to omega dt = zeta (dt = (kv + damping) / (2 kp)): 11.7 % of the
    step there, at four times or more below the stability limit."""
    broken = [s for s in f.servos if over(s.omega * dt, max(SERVO_C, s.zeta))]
    if not broken:
        return None
    s = max(broken, key=lambda s: s.omega * dt / stability_limit(s.zeta))
    x, limit = s.omega * dt, stability_limit(s.zeta)
    cost = "unstable" if x >= limit else "its response is off by more than a tenth of a step"
    more = f" (and {len(broken) - 1} more)" if len(broken) > 1 else ""
    return (f"{s.name}{more}: servo omega dt {x:.3g} > {max(SERVO_C, s.zeta):.3g} (kp {s.kp:.4g}, inertia "
            f"{s.inertia:.3g}, zeta {s.zeta:.3g}; {cost}, stable below {limit:.3g})")


def cap_damping(f: RobotFacts, dt: float) -> str | None:
    """(1 + dt b / I) <= DAMPING_DILUTION_MAX x its value at the fixed step, per joint whose links can touch
    something (module docstring): implicit damping must not make a contact much softer than at the fixed step."""
    ratios = [(j, j.dilution(dt) / j.dilution(f.fixed)) for j in f.damped]
    broken = [(j, r) for j, r in ratios if over(r, DAMPING_DILUTION_MAX)]
    if not broken:
        return None
    j, r = max(broken, key=lambda row: row[1])
    more = f" (and {len(broken) - 1} more)" if len(broken) > 1 else ""
    return (f"{j.name}{more}: joint damping dilutes contacts (dt b / I {dt * j.damping / j.inertia:.3g}): contacts "
            f"on {j.name}'s links sink x{r:.3g} deeper than at {f.fixed * 1e3:g} ms")


def cap_solref(f: RobotFacts, dt: float) -> str | None:
    broken = [r for r in f.rows if over(dt, 0.5 * r.tau)]
    if not broken:
        return None
    r = min(broken, key=lambda r: r.tau)
    ratio = softening(r.tau, dt, f.fixed)
    more = f" (and {len(broken) - 1} more)" if len(broken) > 1 else ""
    return (f"{r.name}{more}: {r.kind} time constant {r.tau * 1e3:.3g} ms < 2 dt, so it acts as {2e3 * dt:.3g} ms: "
            f"{PLURAL[r.kind]} {ratio:.3g}x as stiff as at {f.fixed * 1e3:g} ms")


def cap_link_gap(f: RobotFacts, dt: float) -> str | None:
    broken = [k for k in f.links if over(k.speed * dt, k.gap)]
    if not broken:
        return None
    k = max(broken, key=lambda k: k.speed * dt - k.gap)
    step = k.speed * dt
    at = f"{k.reach * 1e3:.3g} mm from its axis" if k.reach > 0.0 else "a slide"
    more = f" (and {len(broken) - 1} more)" if len(broken) > 1 else ""
    return (f"{k.name}{more}: v dt {step * 1e3:.3g} mm > contact gap {k.gap * 1e3:.3g} mm (v {k.speed:.3g} m/s, "
            f"{at}): a contact can start {(step - k.gap) * 1e3:.3g} mm deep")


def cap_loose_body(f: RobotFacts, dt: float) -> str | None:
    at = [(x, x.acting(dt), x.acting(f.fixed)) for x in f.loose]
    broken = [(x, now, was) for x, now, was in at if over(now, was)]
    if not broken:
        return None
    x, now, was = max(broken, key=lambda row: row[1] / row[2])
    more = f" (and {len(broken) - 1} more)" if len(broken) > 1 else ""
    return (f"{x.name}{more}: a loose body's contact acts at {now * 1e3:.3g} ms against {was * 1e3:.3g} ms at the "
            f"fixed step: contacts {(was / now) ** 2:.3g}x as stiff as at {f.fixed * 1e3:g} ms")


ROBOT_CAPS: dict[str, Callable[[RobotFacts, float], str | None]] = {
    "servo": cap_servo, "damping": cap_damping, "solref": cap_solref, "link_gap": cap_link_gap,
    "loose_body": cap_loose_body}


def checks(b: Any, objs: Sequence[Any], scene: Any, fixed: float) -> dict[str, Callable[[float], str | None]]:
    """{cap: dt -> None or why} for the scene's robots; empty without robots (``rigid_step.scan`` merges them)."""
    from alloy.scene.rigid_step_robots_facts import collect  # noqa: PLC0415 - numpy work only for robot scenes

    if not any(t.obj is None for t in objs):
        return {}
    facts = collect(b, objs, scene, fixed)
    return {name: (lambda dt, cap=cap: cap(facts, dt)) for name, cap in ROBOT_CAPS.items()}


__all__ = ["DAMPING_DILUTION_MAX", "KEEPS", "LINK_SPEED_M_S", "LINK_SPEED_RAD_S", "NO_FAST", "POUR", "ROBOT_CAPS",
           "SERVO_C", "Damped", "Link", "Loose", "RobotFacts", "Row", "Servo", "cap_damping", "cap_link_gap",
           "cap_loose_body", "cap_servo", "cap_solref", "checks", "refusal", "robot_why_not", "softening",
           "stability_limit"]
