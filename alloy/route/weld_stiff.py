# SPDX-License-Identifier: Apache-2.0
"""Stiffness of a MuJoCo weld / connect row, and the solver parameters of a stiff one (GAP-W).

A MuJoCo equality is a soft constraint (MuJoCo "Solver parameters"; engine_core_constraint.c, getsolparam /
getimpedance). A row with violation r and impedance d = d(r) has the reference acceleration -b v - k d r and the
regulariser R = (1 - d) / d · W, where W is the row's diagApprox (the two bodies' translational
``body_invweight0`` summed). At rest under a force F the row therefore sits at

    r = F · W · (1 - d) / (d² · k),      k = 1 / (dmax² τ² ζ²)   (solref = (τ, ζ) > 0)
                                         k = -solref[0] / dmax²  (direct form, solref < 0)

At the defaults (τ 0.02 s, ζ 1, solimp 0.9 / 0.95 / 1 mm) that is a spring of 1 / (2.0e-5 s² · W): 2.33 mm per kN on
a pair with W = 0.117 kg⁻¹. It is MuJoCo working as specified, not a fault of the step.

Two things make a row stiffer: a larger k, bounded by the step h (a positive solref is held at τ >= 2 h by MuJoCo's
``refsafe``; a direct one is stable for ω h < 0.83, ω² = -solref[0]), and an impedance nearer 1 (MuJoCo's own cap is
0.9999). Scene-card attachments are already sized that way (``alloy.scene.weld_impedance``: ω h = 0.6, impedance up
to 0.999). ``stiffen_default_rows`` gives the same rule to the weld / connect rows of a robot's MJCF that still hold
MuJoCo's defaults; a row whose file sets its own solref or solimp is left as written.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.scene.weld_impedance import WeldImpedance, reduced_mass, weld_impedance

log = logging.getLogger(__name__)

MJ_MAX_IMPEDANCE = 0.9999  # mjMAXIMP: MuJoCo clamps solimp's d0 and dmax to it
MJ_MIN_IMPEDANCE = 0.0001  # mjMINIMP
REFSAFE_FACTOR = 2.0  # MuJoCo's refsafe: time constant >= 2 x the step
DEFAULT_SOLREF = (0.02, 1.0)
DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
EQ_CONNECT, EQ_WELD = 0, 1  # mjtEq values in mujoco:equality_constraint_type
WELD_MODES = (None, "stiff", "mujoco")  # alloy.build(scene, welds=...); None is DEFAULT_WELDS
DEFAULT_WELDS = "stiff"  # ruling: MJCF welds at MuJoCo's defaults take the attachment rule

_requested: contextvars.ContextVar[str | None] = contextvars.ContextVar("alloy_welds", default=None)


@dataclass(frozen=True)
class RowParams:
    """Solver parameters of one equality row, as MuJoCo reads them."""

    solref: tuple[float, float] = DEFAULT_SOLREF
    solimp: tuple[float, float, float, float, float] = DEFAULT_SOLIMP


def _clamp_impedance(d: float) -> float:
    return min(max(float(d), MJ_MIN_IMPEDANCE), MJ_MAX_IMPEDANCE)


def impedance(solimp: tuple[float, ...], violation_m: float) -> float:
    """MuJoCo's d(r): d0 at r = 0 rising to dmax at r >= width along its power sigmoid (getimpedance)."""
    d0, dmax, width = _clamp_impedance(solimp[0]), _clamp_impedance(solimp[1]), float(solimp[2])
    mid = float(solimp[3]) if len(solimp) > 3 else 0.5
    power = float(solimp[4]) if len(solimp) > 4 else 2.0
    if d0 == dmax or width <= 0.0:
        return 0.5 * (d0 + dmax) if width <= 0.0 else d0
    x = min(abs(float(violation_m)) / width, 1.0)
    if x >= 1.0:
        return dmax
    if power <= 1.0:
        y = x
    elif x <= mid:
        y = (1.0 / mid ** (power - 1.0)) * x**power
    else:
        y = 1.0 - (1.0 / (1.0 - mid) ** (power - 1.0)) * (1.0 - x) ** power
    return d0 + y * (dmax - d0)


def row_stiffness_k(params: RowParams, step_s: float | None = None) -> float:
    """k in s⁻² (before the impedance factor). With ``step_s`` the time constant is held at 2 h, as refsafe does."""
    dmax = _clamp_impedance(params.solimp[1])
    ref0, ref1 = float(params.solref[0]), float(params.solref[1])
    if ref0 <= 0.0:
        return -ref0 / (dmax * dmax)
    tau = max(ref0, REFSAFE_FACTOR * step_s) if step_s else ref0
    return 1.0 / (dmax * dmax * tau * tau * ref1 * ref1)


def row_damping_b(params: RowParams, step_s: float | None = None) -> float:
    """b in s⁻¹: 2 / (dmax τ) for a time constant (held at 2 h with ``step_s``), -solref[1] / dmax direct."""
    dmax = _clamp_impedance(params.solimp[1])
    ref0, ref1 = float(params.solref[0]), float(params.solref[1])
    if ref0 <= 0.0:
        return -ref1 / dmax
    tau = max(ref0, REFSAFE_FACTOR * step_s) if step_s else ref0
    return 2.0 / (dmax * tau)


def spin_lag(params: RowParams, step_s: float, centripetal_accel: float, violation_m: float = math.inf) -> float:
    """Radial offset (m) of a weld whose second body is carried on a circle, from the semi-implicit step alone.

    Over one step of h the row's Jacobian is frozen, so a body carried at centripetal acceleration a keeps a row
    velocity of -a h / 2 (measured in plain MuJoCo C: 0.993-1.000 over 72 runs). The row's damping pays for it:
    K d r = B (h / 2) a, so r = B h a / (2 K d) whatever the impedance: about a h τ for a time constant τ, and
    (5/3) a h² under the attachment rule (ω h = 0.6). RK4 has no such term. The centripetal force itself adds the
    row's compliance times m a in the opposite sense (``row_compliance``)."""
    d = impedance(params.solimp, violation_m)
    return row_damping_b(params, step_s) * step_s * float(centripetal_accel) / (
        2.0 * row_stiffness_k(params, step_s) * d)


def row_compliance(params: RowParams, inv_weight: float, violation_m: float = math.inf,
                   step_s: float | None = None) -> float:
    """Rest stretch per unit force, m/N, at impedance d(``violation_m``): W (1 - d) / (d² k)."""
    d = impedance(params.solimp, violation_m)
    return float(inv_weight) * (1.0 - d) / (d * d * row_stiffness_k(params, step_s))


def rest_stretch(params: RowParams, inv_weight: float, force_n: float, step_s: float | None = None) -> float:
    """Rest stretch (m) under ``force_n``: the fixed point of r = F c(d(r)), found by bisection (c falls with r)."""
    force = abs(float(force_n))
    hi = force * row_compliance(params, inv_weight, 0.0, step_s)  # d0: the softest the row gets
    lo = force * row_compliance(params, inv_weight, math.inf, step_s)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if force * row_compliance(params, inv_weight, mid, step_s) > mid:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class StiffenedRows:
    """What ``stiffen_default_rows`` changed: the row indices, the step it sized for and one rule per row."""

    rows: tuple[int, ...]
    step_s: float
    rules: tuple[WeldImpedance, ...]

    @property
    def count(self) -> int:
        return len(self.rows)


def default_rows(mj: Any) -> np.ndarray:
    """Indices of the weld / connect rows that still hold MuJoCo's default solref and solimp."""
    kinds = mj.equality_constraint_type.numpy()
    solref, solimp = mj.eq_solref.numpy(), mj.eq_solimp.numpy()
    is_default = (np.all(np.isclose(solref, DEFAULT_SOLREF, rtol=1e-6, atol=0.0), axis=1)
                  & np.all(np.isclose(solimp, DEFAULT_SOLIMP, rtol=1e-6, atol=0.0), axis=1))
    return np.flatnonzero(is_default & np.isin(kinds, (EQ_CONNECT, EQ_WELD)))


def _body_mass(masses: np.ndarray, body: int) -> float:
    return float(masses[body]) if body >= 0 else 0.0  # the world: no mass, reduced_mass reads it as fixed


def stiffen_default_rows(model: Any, step_s: float) -> StiffenedRows:
    """Give every default weld / connect row of ``model`` the attachment rule at rigid step ``step_s``.

    Call it after assembly and before the solver and the health monitor are made: both read these arrays. Rows with
    an authored solref / solimp, and scenes without equality rows, are left alone.
    """
    mj = getattr(model, "mujoco", None)
    if mj is None or not int(getattr(mj, "equality_constraint_count", 0) or 0):
        return StiffenedRows((), float(step_s), ())
    rows = default_rows(mj)
    if rows.size == 0:
        return StiffenedRows((), float(step_s), ())
    masses = model.body_mass.numpy()
    b1, b2 = mj.equality_constraint_body1.numpy(), mj.equality_constraint_body2.numpy()
    solref, solimp = mj.eq_solref.numpy(), mj.eq_solimp.numpy()
    rules = []
    for e in rows:
        rule = weld_impedance(reduced_mass(_body_mass(masses, int(b1[e])), _body_mass(masses, int(b2[e]))), step_s)
        solref[e], solimp[e] = rule.solref, rule.solimp
        rules.append(rule)
    mj.eq_solref.assign(solref)
    mj.eq_solimp.assign(solimp)
    log.info("stiff welds: %d default weld / connect rows sized for the %.4g ms rigid step (solref %s, solimp %.4g)",
             rows.size, step_s * 1e3, rules[0].solref, rules[0].imp)
    return StiffenedRows(tuple(int(e) for e in rows), float(step_s), tuple(rules))


def check(welds: str | None, processes: Any, error: type[Exception]) -> None:
    """Refuse an unknown ``welds`` value, or one asked with worker processes; ``error(message, fix, field=)``."""
    if welds not in WELD_MODES:
        raise error(f"welds={welds!r} is not a weld setting", 'pass welds="stiff", welds="mujoco" or leave it out',
                    field="welds")
    if welds not in (None, DEFAULT_WELDS) and processes is not None:     # workers build with DEFAULT_WELDS
        raise error("welds is not passed to worker processes yet", "build in one process (processes=None)",
                    field="welds")


@contextlib.contextmanager
def requested(welds: str | None) -> Iterator[None]:
    """``alloy.build(scene, welds=...)`` for the builds made inside the block; ``apply`` reads it."""
    token = _requested.set(welds)
    try:
        yield
    finally:
        _requested.reset(token)


def apply(built: Any) -> StiffenedRows | None:
    """Stiffen the assembled scene's default weld / connect rows when the build asks for it (else None).
    Only a welds="stiff" the caller passed warns when it changes nothing: the default finds no row in most scenes."""
    asked = _requested.get()
    if (asked or DEFAULT_WELDS) != "stiff":
        return None
    done = stiffen_default_rows(built.model, built.rigid_dt)
    if not done.count and asked == "stiff":
        from alloy.api import _say  # noqa: PLC0415 - the API's warn-once channel

        _say.warn("welds='stiff' changed no row: this scene has no weld / connect equality at MuJoCo's default "
                  "solref / solimp (rows that set their own are kept as written)")
    return done
