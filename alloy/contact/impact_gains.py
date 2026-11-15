# SPDX-License-Identifier: Apache-2.0
"""Impact-matched contact gains for MuJoCo's own contacts, on balls in free-body scenes.

``alloy.scene.native_gains`` writes them for balls (one-point contacts, no resting floor) in rk4 entries and in
implicitfast entries on rk4's or their own gains (free-body clutter and the fast mode) whose dynamic bodies are all
balls. Everything else keeps the static rule (``native_gains.static_gains``): boxes, capsules, cylinders and meshes,
whose multi-point contacts rock under the larger kd's friction pull, balls beside one of them, and every body beside
a robot, a kinematic object or a chain. The rigid-step planner keeps the static rule too, so its step choices do not
change. Two errors of the static rule, both on impacts:

- **Stiffness.** ``ke = K (1 - imp) / m`` gives a resting body the card's force-space stiffness K, because MuJoCo's
  soft contact holds a steady load with ``(1 - d) a0 = d^2 k r``. During an impact the same row is a spring of only
  ``m ke = K (1 - imp)``, 5 % of K at imp 0.95: a 6.8 kg ball at 1.71 m/s went 14.3-14.9 mm into a wall at every
  rigid step against the 4.5 mm of a K spring. Here ``ke = K / m``: the impact spring is K (MuJoCo's
  row oscillates at sqrt(ke) at d = dmax), and a resting load then sinks K / (1 - imp) less: at rest the contact is
  about 20x stiffer than the card, which only shrinks rest sag.
- **Restitution.** ``zeta = -ln e / sqrt(pi^2 + ln^2 e)`` is the damping ratio of a spring-damper that lets go at
  zero overlap. A MuJoCo contact only pushes, so it lets go when its push reaches zero, earlier: at zeta 1 (e 0)
  the body leaves at exp(-2) = 0.135 of its impact speed whatever the speed or stiffness (measured 0.11-0.15 at
  1-5 m/s on CPU), and every card reads high (0.3 -> 0.40, 0.5 -> 0.55). ``zeta_for_restitution`` inverts
  the zero-force release instead. e = 0 would need infinite damping: the target is max(e, E_FLOOR).

Both are capped for stability at the chosen step (``native_gains.stable_ke``, the same cap as before: more damping
lowers it). Where that cap at the target zeta lies under today's stiffness (light bodies), the row keeps today's
stiffness and takes the largest zeta that stiffness still allows (``zeta_at``): never softer than before, and e
between the target and today's.
"""

from __future__ import annotations

import functools
import math

E_FLOOR = 0.02          # smallest restitution aimed for (e = 0 needs infinite damping under the zero-force release)
_ZETA_MAX = 50.0        # bisection bracket; e(50) ~ 1e-4


def release_speed(zeta: float) -> float:
    """Outward speed / impact speed of a unilateral spring-damper x'' + 2 zeta w x' + w^2 x = 0 (x the overlap, from
    x = 0, x' = 1) at its first zero of push w^2 x + 2 zeta w x' after the start: the restitution it realises."""
    if zeta <= 0.0:
        return 1.0
    if zeta < 1.0:
        s = math.sqrt(1.0 - zeta * zeta)
        # push ~ e^(-zeta t) [((1 - 2 zeta^2) / s) sin(s t) + 2 zeta cos(s t)]: first zero at s t = theta
        theta = math.pi - math.atan2(2.0 * zeta * s, 1.0 - 2.0 * zeta * zeta)
        t = theta / s
        return math.exp(-zeta * t) * abs(math.cos(theta) - zeta / s * math.sin(theta))
    if zeta == 1.0:
        return math.exp(-2.0)                       # x = t e^-t; push zero at t = 2, speed (1 - t) e^-t
    r = math.sqrt(zeta * zeta - 1.0)
    a, b = -zeta + r, -zeta - r                     # x = (e^(a t) - e^(b t)) / (a - b)
    # push = x + 2 zeta x' = 0  <=>  e^((a - b) t) = (1 + 2 zeta b) / (1 + 2 zeta a), both factors negative
    t = math.log((1.0 + 2.0 * zeta * b) / (1.0 + 2.0 * zeta * a)) / (a - b)
    return abs(a * math.exp(a * t) - b * math.exp(b * t)) / (a - b)


@functools.lru_cache(maxsize=1024)
def zeta_for_restitution(e: float) -> float:
    """Damping ratio whose impact rebounds at ``max(e, E_FLOOR)`` on a contact that lets go at zero push
    (``release_speed``, decreasing in zeta). Example: ``zeta_for_restitution(0.5)`` is 0.255 (the zero-overlap rule
    gives 0.216, which rebounds at 0.55)."""
    if not 0.0 <= e <= 1.0:
        raise ValueError(f"restitution must be in [0, 1], got {e}")
    target = max(float(e), E_FLOOR)
    if target >= 1.0:
        return 0.0
    lo, hi = 0.0, _ZETA_MAX
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if release_speed(mid) > target else (lo, mid)
    return 0.5 * (lo + hi)


def zeta_at(ke: float, dt: float, imp: float, margin: float, fraction: float) -> float:
    """The largest zeta at which ``ke`` is still within the stability cap
    ``dt^2 ke / imp + 4 zeta dt sqrt(ke) <= 4 margin`` (times the entry's ``fraction``): inf when any zeta is."""
    u = dt * math.sqrt(ke / fraction)
    if u <= 0.0:
        return math.inf
    return max(0.0, (margin * imp - 0.25 * u * u) / (u * imp))
