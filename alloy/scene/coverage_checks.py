# SPDX-License-Identifier: Apache-2.0
"""The per-route checks of CARD-1 coverage (`alloy.scene.coverage`). Each check takes (name, library, run, device)
and returns (value, bound, passed, detail); `value` is the error the bound applies to.

- rest: largest body travel of the object (the probe for fixtures) over the rest window vs τ;
- rod_self_test: max(|sag_err|, |drop_err|) of the router's cantilever self-test vs 5 % (period ≤ 2 % also gated);
- sheet_navier: a square coupon of the card's material and thickness, its route forced to the object's route,
  boundary vertices pinned (simply supported), sags under self-weight; centre deflection (mean over the last
  SHEET_AVG_S) vs the Navier series. Side a such that w/h = SHEET_W_OVER_H (linear plate range). Isotropic D =
  E·h³/(12(1 − ν²)); a ClothAniso card uses the orthotropic series with D₁ = bend_warp, D₂ = bend_weft and Huber's
  H = √(D₁D₂) (Timoshenko & Woinowsky-Krieger 1959, §30 and §86);
- neohookean: A15 (`alloy.bench.analytic.neohookean`) at the card's Lamé constants and density on VBD;
- hydrostatic: A13 (`alloy.solvers.sph.laws`) at the card's density, viscosity and the router's SPH radius;
- runout: R8's aspect-2 column collapse (`alloy.scene.twins.granular`) on its synthetic sand dataset, runout only.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from alloy.cards import types as ct
from alloy.core.errors import AssemblyError
from alloy.scene import derive

log = logging.getLogger("alloy.scene")

SHEET_W_OVER_H = 0.035          # CARD-8's plate: w_max/h inside linear plate theory
SHEET_GRID = 17                 # vertices per side (CARD-8)
SHEET_S, SHEET_AVG_S = 1.0, 0.2
NAVIER_ALPHA = 0.00406          # w = α·q·a⁴/D, simply supported square plate (TWK table 8)
SHEET_BOUND, REST_TAU = 0.15, 1.0e-4
G = 9.81

CheckResult = tuple[float | None, float | None, bool, str]


# --- rest ------------------------------------------------------------------------------------------------------------

def check_rest(name: str, lib, run, device: str) -> CheckResult:
    tau = float(run.built.handles[run.tracked].decision.numbers.get("tolerance", REST_TAU))
    d = run.displacement
    if not math.isfinite(d):
        return None, tau, False, f"no bodies to watch on {run.tracked}"
    return d, tau, d <= tau, f"max travel of {run.tracked} over the window {d * 1e3:.4f} mm (tau {tau * 1e3:g} mm)"


# --- rods ------------------------------------------------------------------------------------------------------------

def check_rod(name: str, lib, run, device: str) -> CheckResult:
    from alloy.route import rodcheck

    st = run.handle.decision.self_test
    if not st:
        return None, rodcheck.SAG_BOUND, False, "chain decision carries no self-test"
    err = max(abs(st["sag_err"]), abs(st["drop_err"]))
    detail = (f"sag {100 * st['sag_err']:+.2f} %, drop {100 * st['drop_err']:+.2f} %, "
              f"period {100 * st['period_err']:+.2f} %")
    return err, rodcheck.SAG_BOUND, rodcheck.passes(dict(st)), detail


# --- sheets ----------------------------------------------------------------------------------------------------------

def plate_stiffness(mat: ct.MaterialCard, h: float) -> tuple[float, float, float]:
    """(D₁, D₂, H) [N·m] of a sheet of the card's material and thickness h."""
    el = mat.elastic
    if el is None:
        raise AssemblyError(f"{mat.name}: a sheet check needs elastic values")
    if el.cloth is not None:
        d1, d2 = float(el.cloth.bend_warp.value), float(el.cloth.bend_weft.value)
        return d1, d2, math.sqrt(d1 * d2)
    D = derive.sheet_bending(mat, float(el.youngs.value), float(el.poisson.value), h)   # card bending, else E·h³
    return D, D, D


def navier_centre(q: float, a: float, d1: float, d2: float, H: float, terms: int = 99) -> float:
    """Centre deflection [m] of a simply supported square orthotropic plate under uniform load q [Pa]."""
    s = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            s += (-1) ** ((m + n) // 2 - 1) / (m * n * (d1 * m**4 + 2.0 * H * m * m * n * n + d2 * n**4))
    return 16.0 * q * a**4 / math.pi**6 * s


def sheet_side(q: float, h: float, d_eff: float) -> float:
    return (SHEET_W_OVER_H * h * d_eff / (NAVIER_ALPHA * q)) ** 0.25


class _WithCoupon:
    """The card library plus one in-memory coupon card (`object()` serves it; everything else delegates)."""

    def __init__(self, lib, coupon: ct.ObjectCard) -> None:
        self._lib, self._coupon = lib, coupon

    def object(self, name: str, *a, **k) -> ct.ObjectCard:
        return self._coupon if name == self._coupon.name else self._lib.object(name, *a, **k)

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._lib, attr)


def _thickness(card: ct.ObjectCard) -> float:
    t = card.features.thickness
    if t is None:
        raise AssemblyError(f"{card.name}: a sheet check needs features.thickness")
    return float(t.value)


def _coupon(card: ct.ObjectCard, route: str, a: float, h: float) -> ct.ObjectCard:
    return dataclasses.replace(
        card, name=f"{card.name}__coupon", shape=ct.Primitive("box", (0.5 * a, 0.5 * a, 0.5 * h)), motion="dynamic",
        budget=ct.Budget(particles=SHEET_GRID * SHEET_GRID), topology=None, precision_partners=(),
        precision_features=(), route=ct.RouteOverride(route, f"CARD-1 sheet coupon of {card.name}"))


def _pin_edges(model, a: float, route: str) -> np.ndarray:
    """Fix the boundary vertices: zero mass for VBD (as CARD-8); Style3D pins by clearing ParticleFlags.ACTIVE
    (newton@ac7f7830 solvers/style3d/kernels.py:16, 194) and steps NaN on zero-mass vertices."""
    q0 = model.particle_q.numpy()
    edge = (np.abs(np.abs(q0[:, 0]) - 0.5 * a) < 1e-6 * max(a, 1.0)) | \
           (np.abs(np.abs(q0[:, 1]) - 0.5 * a) < 1e-6 * max(a, 1.0))
    if route == "cloth_style3d":
        from alloy._vendor import newton

        flags = model.particle_flags.numpy()
        flags[edge] &= ~int(newton.ParticleFlags.ACTIVE)
        model.particle_flags.assign(flags)
        return q0
    mass, inv = model.particle_mass.numpy(), model.particle_inv_mass.numpy()
    mass[edge], inv[edge] = 0.0, 0.0
    model.particle_mass.assign(mass)
    model.particle_inv_mass.assign(inv)
    return q0


def _sag(lib, coupon: ct.ObjectCard, a: float, device: str) -> tuple[float, int]:
    from alloy.core.registry import isolated_registry
    from alloy.route.router import Router
    from alloy.scene.assembler import SceneAssembler
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.sim import Simulation

    scene = ct.SceneCard(robots=(), objects=(ct.Placement(coupon.name, (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)),))
    with isolated_registry(empty=True):
        register_builtin_solvers()
        built = SceneAssembler(_WithCoupon(lib, coupon), Router()).assemble(scene, device, world_count=1)
        q0 = _pin_edges(built.model, a, str(coupon.route.route))
        sim = Simulation(built)
        periods, avg = round(SHEET_S / built.control_dt), round(SHEET_AVG_S / built.control_dt)
        centre = int(np.argmin(np.hypot(q0[:, 0], q0[:, 1])))
        w = []
        for k in range(periods):
            sim.step_period()
            if k >= periods - avg:
                w.append(float(q0[centre, 2] - sim.state.particle_q.numpy()[centre, 2]))
    return float(np.mean(w)), len(q0)


def check_sheet(name: str, lib, run, device: str) -> CheckResult:
    card = lib.object(name)
    mat = lib.material(card.material)
    h = _thickness(card)
    d1, d2, H = plate_stiffness(mat, h)
    q = float(mat.density.value) * h * G
    a = sheet_side(q, h, math.sqrt(d1 * d2))
    ref = navier_centre(q, a, d1, d2, H)
    w, n = _sag(lib, _coupon(card, str(run.handle.decision.route), a, h), a, device)
    err = w / ref - 1.0 if math.isfinite(w) else math.inf
    detail = f"coupon a = {a * 1e3:.1f} mm, h = {h * 1e3:g} mm, {n} vertices: sag {w:.4g} m vs Navier {ref:.4g} m"
    return err, SHEET_BOUND, abs(err) <= SHEET_BOUND, detail


# --- soft, liquid, granular (bench laws at the card's values) ---------------------------------------------------------

def _quick(run) -> bool:
    return bool(run.extra.get("quick", False))


def _law_check(res, metric: str) -> CheckResult:
    vs = [v for v in res.verdicts if v.name.split("@")[0] == metric]      # rig verdicts are metric@condition
    if not vs:
        return None, None, False, f"{res.law}: {res.verdict} ({res.monitors.get('error', 'no verdict')})"
    worst = max(vs, key=lambda v: abs(v.error) / v.bound if v.bound else math.inf)
    passed = all(v.verdict == "pass" for v in vs)
    return float(worst.error), float(worst.bound), passed, f"{res.law} {metric} over {len(vs)} condition(s)"


def check_soft(name: str, lib, run, device: str) -> CheckResult:
    from alloy.bench.runner import run_law
    from alloy.scene import coverage_twins as CT

    mat = lib.material(lib.object(name).material)
    twin = CT.card_neohookean(mat)
    return _law_check(run_law(CT.NEOHOOKEAN_LAW, twin, None, device=device, solver="vbd", quick=_quick(run)), "stress_rel_err")


def check_liquid(name: str, lib, run, device: str) -> CheckResult:
    from alloy.bench.runner import run_law
    from alloy.scene import coverage_twins as CT

    mat = lib.material(lib.object(name).material)
    r = float(run.handle.decision.numbers["particle_radius"])
    twin = CT.card_hydrostatic(mat, r)
    return _law_check(run_law(CT.HYDROSTATIC_LAW, twin, None, device=device, quick=_quick(run)), "wall_force")


def check_granular(name: str, lib, run, device: str) -> CheckResult:
    from alloy.bench.runner import run_law
    from alloy.scene.twins import granular as GR

    return _law_check(run_law(GR.LAW, GR.GranularTwin(), GR.dataset(), device=device, quick=_quick(run)), "runout")


def check_none(name: str, lib, run, device: str) -> CheckResult:
    return None, None, False, "no analytic law or self-test exists for this route (SPEC §5.3)"


CHECKS: Mapping[str, tuple[str, Callable[..., CheckResult]]] = {
    **{r: ("rest", check_rest) for r in ("rigid", "rigid_compliant", "rigid_pile", "panels", "static")},
    "chain": ("rod_self_test", check_rod),
    "cloth_vbd": ("sheet_navier", check_sheet),
    "cloth_style3d": ("sheet_navier", check_sheet),
    "soft_vbd": ("neohookean", check_soft),
    "liquid_sph": ("hydrostatic", check_liquid),
    "grain_mpm": ("runout", check_granular),
}


def check_for(route: str, checks: Mapping[str, Any]) -> tuple[str, Callable[..., CheckResult]]:
    return checks.get(route, ("none", check_none))


__all__ = ["CHECKS", "check_for", "navier_centre", "plate_stiffness", "sheet_side"]
