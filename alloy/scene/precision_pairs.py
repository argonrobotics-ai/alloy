# SPDX-License-Identifier: Apache-2.0
"""PrecisionPairSpecs from cards and the per-pair K_real check (SPEC §3.4, §3.5 L2, D2, D22).

- max_penetration = min(5e-5 m, 0.5·radial clearance) (D2, PrecisionPairSpec.resolved_max_penetration).
- K sized from need: K = 2·design_force/max_penetration (D22) unless the spec gives pair_stiffness.
- dt: the largest grid value ≤ the rigid target with impact sink v_env·2dt/e ≤ max_penetration (e = Euler's number)
  and static sink design_force/K_real ≤ max_penetration/2, K_real from derive.k_real_mujoco at the flat imp 0.99;
  none down to control_dt/2^10 → PrecisionSpecError with the numbers.
- K_real < 0.8·K at that dt → PrecisionSpecError (D22); (declared, realised) goes to RouteReport.k_real.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from alloy.cards.types import PrecisionFeature, ResolvedObject
from alloy.contact.types import PrecisionPairSpec
from alloy.core.errors import PrecisionSpecError
from alloy.core.timebase import TICK_EXP, grid_dt
from alloy.scene import derive

log = logging.getLogger("alloy.scene")
K_REAL_MIN_RATIO = 0.8
VOXEL_DEFAULT = 5.0e-5


@dataclass(frozen=True)
class PrecisionPair:
    """A precision pair between two placed objects (template shape ids) with its numbers."""

    a: str
    b: str
    spec: PrecisionPairSpec
    k_declared: float
    k_real: float
    dt: float
    m_eff: float
    ladder: tuple[str, ...]


def _feature(obj: ResolvedObject, name: str | None) -> PrecisionFeature | None:
    feats = obj.card.precision_features
    if name is not None:
        return next((f for f in feats if f.name == name), None)
    return next((f for f in feats if f.clearance is not None), feats[0] if feats else None)


def partner_names(objs: Mapping[str, ResolvedObject], name: str, partner: str) -> list[str]:
    """The objects a partner entry of ``name``'s card means: the object of that name and every other object placed
    from the card of that name (PHYSA-4: a scene card's second copy is "peg_d12#1", and matching the object name alone
    made no pair for it, with no word; ``alloy.api`` writes its renamed cards' partners by object name itself)."""
    pname = partner.partition(":")[0]
    return sorted(n for n, o in objs.items() if n != name and (n == pname or o.card.name == pname))


def pair_features(objs: Mapping[str, ResolvedObject]) -> list[tuple[str, str, PrecisionFeature, PrecisionFeature]]:
    """(a, b, feature_a, feature_b) for every declared partner relation, each unordered pair once."""
    seen, out = set(), []
    for name in sorted(objs):
        obj = objs[name]
        for partner in obj.card.precision_partners:
            for pname in partner_names(objs, name, partner):
                if (key := (min(name, pname), max(name, pname))) in seen:
                    continue
                fa, fb = _feature(obj, None), _feature(objs[pname], partner.partition(":")[2] or None)
                if fa is None or fb is None:
                    raise PrecisionSpecError(f"{name} -> {partner}: precision partner without a PrecisionFeature")
                seen.add(key)
                out.append((name, pname, fa, fb))
    return out


def routed_features(report: Any, templates: Sequence[Any]) -> list[tuple[str, str, PrecisionFeature, PrecisionFeature]]:
    """``pair_features`` of the template's objects, said where a user reads it: each pair in both objects' route
    reasons (``Sim.routes()``), and a warning when two objects each name a precision partner that is not in the scene
    (a pair that was meant and cannot form: hand-written cards whose partner names do not match the scene's objects)."""
    objs = {t.name: t.obj for t in templates if t.obj is not None}
    rels = pair_features(objs)
    for a, b, fa, fb in rels:
        gap = _clearance(fa, fb)
        fit = f", clearance {1e3 * gap:g} mm" if gap is not None else ""
        for name, other, mine, theirs in ((a, b, fa, fb), (b, a, fb, fa)):
            d, line = report.decisions.get(name), f"precision pair with {other}: {mine.name} / {theirs.name}{fit}"
            if d is not None and line not in d.reasons:
                report.decisions[name] = replace(d, reasons=(*d.reasons, line))
    alone = {n: [p for p in o.card.precision_partners if not partner_names(objs, n, p)] for n, o in objs.items()}
    alone = {n: p for n, p in alone.items() if p and not any(n in (a, b) for a, b, _, _ in rels)}
    if len(alone) > 1:
        log.warning("no precision pair is made for %s: each names a precision partner that is not in this scene "
                    "(objects: %s), so they collide as plain shapes; a partner is the name of an object in the scene "
                    "or of its card", ", ".join(f"{n} -> {p[0]}" for n, p in sorted(alone.items())),
                    ", ".join(sorted(objs)))
    return rels


def _clearance(fa: PrecisionFeature, fb: PrecisionFeature) -> float | None:
    vals = [float(f.clearance.value) for f in (fa, fb) if f.clearance is not None]
    return min(vals) if vals else None


def l2_dt(spec: PrecisionPairSpec, k: float, m_eff: float, design_force: float, v_env: float, control_dt: float,
          rigid_dt: float, integrator: str) -> tuple[float, float]:
    """(dt, K_real) at the largest grid dt meeting both L2 conditions; PrecisionSpecError if none."""
    pen = spec.resolved_max_penetration()
    tau_req, zeta = derive.mujoco_solref_from_force(k, 0.0, m_eff, spec.mj_imp)
    for j in range(TICK_EXP + 1):
        dt = grid_dt(control_dt, j)
        if dt > rigid_dt:
            continue
        k_real = derive.k_real_mujoco(m_eff, spec.mj_imp, dt, tau_req, zeta, integrator=integrator)
        impact_ok = v_env * 2.0 * dt / math.e <= pen
        static_ok = design_force / k_real <= 0.5 * pen * (1.0 + 1e-9)   # K = 2F/pen makes this an equality
        if impact_ok and static_ok:
            return dt, k_real
    raise PrecisionSpecError(f"precision pair shapes {spec.shape_a}/{spec.shape_b}: L2 cannot be met down to "
                             f"control_dt/2^{TICK_EXP} = {grid_dt(control_dt, TICK_EXP):.3g} s (max_penetration "
                             f"{pen:.3g} m, K {k:.3g} N/m, m_eff {m_eff:.3g} kg, design force {design_force} N, "
                             f"v_env {v_env} m/s)")


def ladder_rungs(integrator: str, native: str = "mujoco") -> tuple[str, ...]:
    """Gap-ladder rungs per the §3.5 integrator table: L1 on euler/implicitfast for both native paths (the D10
    append spike carries speculative rows through callback.passive, SCR-WP3-2); never on `discrete`."""
    rungs = ["L0"]
    if integrator in ("euler", "implicitfast"):
        rungs.append("L1")
    return tuple(rungs + ["L2", "L3", "L4"])


@dataclass(frozen=True)
class PairInputs:
    shape_a: int
    shape_b: int
    mass_a: float          # kg, 0 for static
    mass_b: float
    design_force: float    # N
    v_env: float           # m/s (card contact_speed)


def build_pair(a: str, b: str, fa: PrecisionFeature, fb: PrecisionFeature, inp: PairInputs, control_dt: float,
               rigid_dt: float, integrator: str) -> PrecisionPair:
    clearance = _clearance(fa, fb)
    voxel = fa.voxel or fb.voxel or (min(VOXEL_DEFAULT, 0.5 * clearance) if clearance else VOXEL_DEFAULT)
    spec = PrecisionPairSpec(shape_a=inp.shape_a, shape_b=inp.shape_b, mode=fa.mode, voxel_size=voxel,
                             crop_a=fa.crop, crop_b=fb.crop, clearance=clearance,
                             analytic_id=fa.analytic_id or fb.analytic_id)
    pen = spec.resolved_max_penetration()
    k = spec.pair_stiffness or 2.0 * inp.design_force / pen
    spec = replace(spec, pair_stiffness=k)
    inv = sum(1.0 / m for m in (inp.mass_a, inp.mass_b) if m > 0.0)
    if inv <= 0.0:
        raise PrecisionSpecError(f"precision pair {a}/{b}: both sides static")
    m_eff = 1.0 / inv
    dt, k_real = l2_dt(spec, k, m_eff, inp.design_force, inp.v_env, control_dt, rigid_dt, integrator)
    if k_real < K_REAL_MIN_RATIO * k:
        raise PrecisionSpecError(f"precision pair {a}/{b}: K_real {k_real:.3g} N/m < 0.8 × declared {k:.3g} N/m at "
                                 f"dt {dt:.3g} s (m_eff {m_eff:.3g} kg, imp {spec.mj_imp})")
    log.info("precision pair %s/%s: max_pen %.3g m, K %.3g N/m, K_real %.3g N/m, dt %.3g s", a, b, pen, k, k_real, dt)
    return PrecisionPair(a, b, spec, k, k_real, dt, m_eff, ladder_rungs(integrator))


def precision_dt(pairs: Sequence[PrecisionPair]) -> float | None:
    return min((p.dt for p in pairs), default=None)
