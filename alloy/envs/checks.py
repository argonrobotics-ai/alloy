# SPDX-License-Identifier: Apache-2.0
"""Task checkers: per-world measurements and pass/fail tests over a ``BatchState``, for rewards and episode ends.

A ``Check`` maps the state of every world to a (W,) array: a number (a distance, a height) or a bool (lifted,
inside). Compare, combine and weight them::

    from alloy.envs import checks as C
    reach = C.distance("gripper", "cube")                  # (W,) metres
    held = C.between("cube", "gripper/finger_left", "gripper/finger_right")
    success = C.lifted("cube", by=0.08) & held             # bool (W,)
    fell = C.height("cube") < -0.5                         # comparisons make checks too
    spilled = 1.0 - C.fraction_inside("water", lo, hi)     # and arithmetic (a bool check counts 1 / 0)
    task = Task(reward={"reach": (-1.0, reach), "lift": (10.0, success)}, success=success, failure=fell)

Points are body names (their origin), "<body>.com" (centre of mass; a mesh's origin may sit at its base), goal points
("target" or "target.pos") or fixed (x, y, z) positions in metres. Every check has a readable ``name``.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Sequence
from typing import Any

from alloy.envs.state import BatchState

Point = str | Sequence[float]


class Check:
    """A named function of the batched state returning a (W,) array. Example: ``C.height("cube")(env.state)``."""

    def __init__(self, name: str, fn: Callable[[BatchState], Any]) -> None:
        self.name, self.fn = name, fn

    def __call__(self, s: BatchState) -> Any:
        return self.fn(s)

    def __repr__(self) -> str:
        return f"Check({self.name})"

    # comparisons and logic make new checks (bool per world)
    def _cmp(self, other: Any, op: Callable[[Any, Any], Any], sym: str) -> Check:
        if isinstance(other, Check):
            return Check(f"{self.name} {sym} {other.name}", lambda s: op(self(s), other(s)))
        return Check(f"{self.name} {sym} {other:g}", lambda s: op(self(s), other))

    def __lt__(self, other: Any) -> Check:
        return self._cmp(other, operator.lt, "<")

    def __le__(self, other: Any) -> Check:
        return self._cmp(other, operator.le, "<=")

    def __gt__(self, other: Any) -> Check:
        return self._cmp(other, operator.gt, ">")

    def __ge__(self, other: Any) -> Check:
        return self._cmp(other, operator.ge, ">=")

    def __and__(self, other: Check) -> Check:
        return Check(f"({self.name} & {other.name})", lambda s: _bool(s, self(s)) & _bool(s, other(s)))

    def __or__(self, other: Check) -> Check:
        return Check(f"({self.name} | {other.name})", lambda s: _bool(s, self(s)) | _bool(s, other(s)))

    def __invert__(self) -> Check:
        return Check(f"~{self.name}", lambda s: ~_bool(s, self(s)))

    # arithmetic makes numeric checks (a bool check counts 1 / 0): ``1.0 - in_cup``, ``reach + 0.5 * tilt``
    def _math(self, other: Any, op: Callable[[Any, Any], Any], sym: str, swap: bool = False) -> Check:
        if isinstance(other, Check):
            a, b = (other, self) if swap else (self, other)
            return Check(f"({a.name} {sym} {b.name})", lambda s: op(_num(s, a(s)), _num(s, b(s))))
        label = f"({other:g} {sym} {self.name})" if swap else f"({self.name} {sym} {other:g})"
        if swap:
            return Check(label, lambda s: op(other, _num(s, self(s))))
        return Check(label, lambda s: op(_num(s, self(s)), other))

    def __add__(self, other: Any) -> Check:
        return self._math(other, operator.add, "+")

    def __radd__(self, other: Any) -> Check:
        return self._math(other, operator.add, "+", swap=True)

    def __sub__(self, other: Any) -> Check:
        return self._math(other, operator.sub, "-")

    def __rsub__(self, other: Any) -> Check:
        return self._math(other, operator.sub, "-", swap=True)

    def __mul__(self, other: Any) -> Check:
        return self._math(other, operator.mul, "*")

    def __rmul__(self, other: Any) -> Check:
        return self._math(other, operator.mul, "*", swap=True)

    def __truediv__(self, other: Any) -> Check:
        return self._math(other, operator.truediv, "/")

    def __neg__(self) -> Check:
        return Check(f"-{self.name}", lambda s: -_num(s, self(s)))


def _bool(s: BatchState, x: Any) -> Any:
    return x if str(getattr(x, "dtype", "")) in ("bool", "torch.bool") else x != 0


def _num(s: BatchState, x: Any) -> Any:
    """``x`` as numbers: a bool check's worlds as 1.0 / 0.0."""
    return s.xp.float(x) if str(getattr(x, "dtype", "")) in ("bool", "torch.bool") else x


def _point(s: BatchState, p: Point) -> Any:
    """(W, 3) positions: a body name (its origin), "<body>.com" / "<goal>.pos" (a state key), or a fixed point."""
    if isinstance(p, str):
        return s[p] if p.rpartition(".")[2] in ("pos", "com", "center") else s[f"{p}.pos"]
    return s.xp.asarray(p, "float32").reshape(1, 3)


def _label(p: Point) -> str:
    return p if isinstance(p, str) else "(" + ", ".join(f"{float(v):g}" for v in p) + ")"


# --- measurements (numbers per world) --------------------------------------------------------------------------

def height(body: str) -> Check:
    """Height z of a body (m). Example: ``C.height("cube") > 0.1``."""
    return Check(f"height({body})", lambda s: _point(s, body)[:, 2])


def distance(a: Point, b: Point) -> Check:
    """Distance between two bodies or points (m). Example: ``C.distance("gripper", "cube")``."""
    return Check(f"distance({_label(a)}, {_label(b)})", lambda s: s.xp.norm(_point(s, a) - _point(s, b)))


def speed(body: str) -> Check:
    """Linear speed of a body (m/s). Example: ``C.speed("cube") < 0.01``."""
    return Check(f"speed({body})", lambda s: s.xp.norm(s[f"{body}.vel"]))


def joint(name: str) -> Check:
    """Value of a 1-dof joint (m or rad). Example: ``C.joint("gripper/lift") > 0.1``."""
    return Check(f"joint({name})", lambda s: s[f"{name}.q"])


def tilt(body: str) -> Check:
    """Angle (degrees) between a body's z axis and the world's up axis. Example: ``C.tilt("cup") < 10``."""
    def fn(s: BatchState) -> Any:
        q = s[f"{body}.quat_xyzw"]
        x, y = q[:, 0], q[:, 1]
        cos = s.xp.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)   # z-axis . up
        return _acos_deg(s, cos)
    return Check(f"tilt({body})", fn)


def rise(body: str) -> Check:
    """How far a body rose since the episode started (m, negative if it fell). Example: ``C.rise("cube") > 0.05``."""
    return Check(f"rise({body})", lambda s: _point(s, body)[:, 2] - _point(s.start, body)[:, 2])


def fraction_inside(particles: str, lo: Sequence[float], hi: Sequence[float]) -> Check:
    """Fraction of a liquid's / cloth's particles inside the box [lo, hi]; for a pour, of its whole volume (drops
    still at the nozzle count as outside).

    Example: ``C.fraction_inside("water", (-0.05, -0.05, 0), (0.05, 0.05, 0.1))``."""
    def fn(s: BatchState) -> Any:
        pts, xp = s[f"{particles}.points"], s.xp
        lo_a, hi_a = xp.asarray(lo).reshape(1, 1, 3), xp.asarray(hi).reshape(1, 1, 3)
        ins = (pts >= lo_a) & (pts <= hi_a)
        ins = ins[..., 0] & ins[..., 1] & ins[..., 2]
        if particles in s.rows.pours:
            ins = ins & s[f"{particles}.active"]
        return xp.float(ins).mean(-1)
    return Check(f"fraction_inside({particles})", fn)


def action_size() -> Check:
    """Squared size of the last action, summed over its entries (a smoothness penalty).

    Example: ``reward={"effort": (-0.01, C.action_size())}``."""
    return Check("action_size", lambda s: (s["action"] * s["action"]).sum(-1))


# --- tests (bool per world) -------------------------------------------------------------------------------------

def near(a: Point, b: Point, tol: float) -> Check:
    """Two bodies / points within ``tol`` metres. Example: ``C.near("cube", (0.3, 0, 0.03), 0.02)``."""
    c = distance(a, b) <= tol
    c.name = f"near({_label(a)}, {_label(b)}, {tol:g})"
    return c


def lifted(body: str, by: float) -> Check:
    """A body rose at least ``by`` metres since the episode started. Example: ``C.lifted("cube", by=0.08)``."""
    c = rise(body) >= by
    c.name = f"lifted({body}, {by:g})"
    return c


def upright(body: str, max_tilt_deg: float = 10.0) -> Check:
    """A body's z axis within ``max_tilt_deg`` of vertical. Example: ``C.upright("cup", 15)``."""
    c = tilt(body) <= max_tilt_deg
    c.name = f"upright({body}, {max_tilt_deg:g})"
    return c


def at_rest(body: str, max_speed: float = 0.01) -> Check:
    """A body moves slower than ``max_speed`` m/s. Example: ``C.at_rest("cube")``."""
    c = speed(body) <= max_speed
    c.name = f"at_rest({body}, {max_speed:g})"
    return c


def inside(body: str, lo: Sequence[float], hi: Sequence[float]) -> Check:
    """A body's centre inside the box [lo, hi] (m). Example: ``C.inside("cube", (-.1, -.1, 0), (.1, .1, .2))``."""
    def fn(s: BatchState) -> Any:
        p, xp = _point(s, body), s.xp
        ins = (p >= xp.asarray(lo).reshape(1, 3)) & (p <= xp.asarray(hi).reshape(1, 3))
        return ins[:, 0] & ins[:, 1] & ins[:, 2]
    return Check(f"inside({body})", fn)


def between(obj: str, a: str, b: str, max_offset: float = 0.03, max_gap: float = 0.15) -> Check:
    """``obj`` held between bodies ``a`` and ``b`` (two fingers): its centre projects inside the a-b segment, within
    ``max_offset`` m of it, and a-b is under ``max_gap`` m. Example: ``C.between("cube", "g/left", "g/right")``."""
    def fn(s: BatchState) -> Any:
        xp, p, pa, pb = s.xp, _point(s, obj), _point(s, a), _point(s, b)
        d = pb - pa
        length2 = (d * d).sum(-1)
        t = ((p - pa) * d).sum(-1) / xp.clip(length2, 1e-12, math.inf)
        off = xp.norm(p - (pa + t[:, None] * d))
        return (t > 0) & (t < 1) & (off <= max_offset) & (length2 < max_gap * max_gap)
    return Check(f"between({obj}, {a}, {b})", fn)


def all_of(*checks: Check) -> Check:
    """Every check holds. Example: ``C.all_of(C.lifted("cube", 0.05), C.upright("cube"))``."""
    out = checks[0]
    for c in checks[1:]:
        out = out & c
    return out


def any_of(*checks: Check) -> Check:
    """At least one check holds. Example: ``C.any_of(C.height("a") < 0, C.height("b") < 0)``."""
    out = checks[0]
    for c in checks[1:]:
        out = out | c
    return out


def custom(name: str, fn: Callable[[BatchState], Any]) -> Check:
    """Your own check: ``fn(state) -> (W,) array``. Example: ``C.custom("x", lambda s: s["cube.pos"][:, 0])``."""
    return Check(name, fn)


def _acos_deg(s: BatchState, cos: Any) -> Any:
    if s.xp.name == "torch":
        return s.xp.torch.rad2deg(s.xp.torch.arccos(cos))
    import numpy as np  # noqa: PLC0415

    return np.degrees(np.arccos(cos))


__all__ = ["Check", "action_size", "all_of", "any_of", "at_rest", "between", "custom", "distance", "fraction_inside",
           "height", "inside", "joint", "lifted", "near", "rise", "speed", "tilt", "upright"]
