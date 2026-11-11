# SPDX-License-Identifier: Apache-2.0
"""Alloy's default route: the configurations Alloy uses when a scene overrides nothing (coordinator, 2026-11-01).

Headline rows SB-3, SB-4 and SB-12 take their verdict on this population and show "all configurations" beside it.
The route is read from the code defaults, so the scoreboard follows them if they change:
- precision pairs: the card default `PrecisionFeature.mode` (alloy/cards/types.py) under the default
  `PrecisionConfig.native_contacts` (alloy/precision/types.py);
- analytic laws: the router's capability for the law's object class (`alloy.route.router.ROUTE_ENTRY`): rigid laws
  (A1-A6, D1) -> `Route.RIGID`, soft solids (A15) -> `Route.SOFT_VBD`. A7 tests the hydroelastic contact model,
  which no card selects by default. Other laws (single-solver: A8-A14, A16) run on their own route;
- MuJoCo settings: the builtin `mujoco` capability's default_kwargs (newton solver, elliptic cone), the options the
  route fills in for a scene no MJCF file configures (`alloy.scene.entries.mujoco_overrides`; SCR-WP3-10: impratio
  300 once WP2 routes it, so the A5 kinetic metrics and A5.speed follow it) and the SCR-WP6-4 integrator rule:
  `rk4` when every joint is a plain free joint (no damping, armature, drive or equality row) and the scene has no
  precision pair and no noslip, else `implicitfast` (alloy/scene/integrator_route.py);
- noslip: a scene that asks for noslip gets Alloy's scoped noslip (`alloy.contact.noslip`), not MuJoCo Warp's
  (SCR-WP3-4: SB-5, SB-10; SB-6 for T12 is measured with noslip off and shows the scoped cost beside).
"""

from __future__ import annotations

import dataclasses
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from alloy.bench import solvers as SV
from alloy.bench.law import law_prefix

log = logging.getLogger("alloy.bench")

RIGID_LAWS = frozenset({"A1", "A2", "A3", "A4", "A5", "A6", "D1", "D2", "D3"})
SOFT_LAWS = frozenset({"A15", "A17"})
NOT_DEFAULT = {"A7": "hydroelastic precision mode; cards default to sdf"}
_FALLBACK = {"mode": "sdf", "native": "mujoco", "rigid": "mujoco", "soft": "vbd"}
_MUJOCO_FALLBACK = {"solver": "newton", "integrator": "implicitfast", "cone": "elliptic"}


@lru_cache(maxsize=1)
def defaults() -> dict[str, str]:
    """{"mode", "native", "rigid", "soft"} from the code defaults (fallback values if a module cannot import)."""
    try:
        from alloy.cards.types import PrecisionFeature
        from alloy.precision.types import PrecisionConfig
        from alloy.route.router import ROUTE_ENTRY
        from alloy.route.types import Route
    except ImportError as exc:  # a broken producer must not take the scoreboard down
        log.warning("default route: using fallback defaults (%s)", exc)
        return dict(_FALLBACK)
    field = {f.name: f.default for f in dataclasses.fields(PrecisionFeature)}
    native = {f.name: f.default for f in dataclasses.fields(PrecisionConfig)}
    return {"mode": str(field["mode"]), "native": str(native["native_contacts"]),
            "rigid": ROUTE_ENTRY[Route.RIGID], "soft": ROUTE_ENTRY[Route.SOFT_VBD]}


@lru_cache(maxsize=1)
def mujoco_default_kwargs() -> dict[str, object]:
    """The builtin `mujoco` capability's default_kwargs (alloy/scene/builtin_plugins.py); `routed_mujoco_settings`
    adds the route's fill-ins (`route_fill_ins`) and the integrator."""
    try:
        from alloy.scene.builtin_plugins import builtin_capabilities
    except ImportError as exc:
        log.warning("default route: builtin mujoco kwargs unavailable, using fallback (%s)", exc)
        return dict(_MUJOCO_FALLBACK)
    cap = next((c for c in builtin_capabilities() if c.name == "mujoco"), None)
    return dict(cap.default_kwargs) if cap is not None else dict(_MUJOCO_FALLBACK)


@lru_cache(maxsize=2)
def route_fill_ins(chain_newton: bool = False) -> dict[str, object]:
    """Options the MuJoCo route sets for a scene with no MJCF file, no precision pair and no route effect (the
    analytic laws' scenes): `alloy.scene.entries.mujoco_overrides` with nothing given. SCR-WP3-10: impratio 300
    (Coulomb sliding force independent of speed) since main 27da8424. Solimp stays Newton's. `chain_newton`: the
    scene steps a CHAIN on the Newton solver, where the route holds the impratio fill-in back (A10.capstan's rope
    went NaN with it, WP2's fix)."""
    try:
        from alloy.scene.entries import mujoco_overrides
    except ImportError as exc:
        log.warning("default route: MuJoCo route fill-ins unavailable, using none (%s)", exc)
        return {}
    return {k: v for k, (v, _) in mujoco_overrides(frozenset(), False, None, chain_newton=chain_newton).items()}


_NOT_FREE_FIELDS = ("joint_damping", "joint_armature", "joint_target_ke", "joint_target_kd", "joint_target_mode")


def _all_free(model: Any) -> bool:
    """Every joint a plain free joint of a dynamic body: no damping, armature, drive or equality row."""
    from alloy._vendor import newton

    if not model.joint_count or np.any(model.joint_type.numpy() != int(newton.JointType.FREE)):
        return False
    if np.any(model.body_mass.numpy() <= 0.0):
        return False
    for name in _NOT_FREE_FIELDS:
        arr = getattr(model, name, None)
        if arr is not None and np.any(arr.numpy()):
            return False
    eq = getattr(getattr(model, "mujoco", None), "equality_constraint_type", None)
    return eq is None or len(eq) == 0


def _r1(model: Any) -> bool:
    """F6 R1 on a twin's model: more than R1_MIN_FREE free bodies per world and none the inertia floor would raise
    (its slender test at today's 1.25 ms step, from the model's masses and inertias), on a route R1 applies to (MuJoCo
    Warp: ``integrator_route.r1_off``). No R2 (integrator_route)."""
    from alloy.contact.inertia_floor import SLENDER, floor_radius  # noqa: PLC0415
    from alloy.scene import integrator_route as IR  # noqa: PLC0415

    if IR.r1_off(str(model.device)) or model.body_count // max(int(model.world_count), 1) <= IR.R1_MIN_FREE:
        return False
    r0 = floor_radius(1.25e-3)
    mass = model.body_mass.numpy().astype(np.float64)
    principal = np.linalg.eigvalsh(model.body_inertia.numpy().astype(np.float64).reshape(-1, 3, 3))
    slender = (mass * r0 * r0 > principal[:, 0]) & (principal[:, -1] >= SLENDER * principal[:, 0])
    return not bool(slender.any())


def routed_integrator(model: Any = None, precision: bool = False, noslip: bool = False) -> str:
    """The integrator a routed MuJoCo entry takes for this model (SCR-WP6-4, F6 R1); the builtin default without a
    model."""
    default = str(mujoco_default_kwargs().get("integrator", "implicitfast"))
    if model is None or precision or noslip or not _all_free(model):
        return default
    return "implicitfast" if _r1(model) else "rk4"


def routed_gains(model: Any = None, precision: bool = False, noslip: bool = False) -> str:
    """The gains mode the assembler writes native contact gains for (``IntegratorChoice.gains``): rk4's quarter cap
    on every free-body-only twin, whether it runs rk4 or implicitfast under F6 R1; else its integrator."""
    integrator = routed_integrator(model, precision, noslip)
    if model is not None and not (precision or noslip) and _all_free(model):
        return "rk4"
    return integrator


def routed_mujoco_settings(model: Any = None, chain_newton: bool = False) -> SV.SolverSettings:
    """SolverMuJoCo as Alloy routes a non-precision scene with no MJCF file: builtin kwargs, the route's fill-ins
    (SCR-WP3-10 impratio, held back for a CHAIN on Newton: `chain_newton`) and the SCR-WP6-4 integrator."""
    kwargs = {**mujoco_default_kwargs(), **route_fill_ins(chain_newton), "integrator": routed_integrator(model)}
    return SV.SolverSettings("mujoco", kwargs)


def describe() -> str:
    d = defaults()
    fill = route_fill_ins().get("impratio")
    imp = f", MuJoCo impratio {fill:g} where no file sets one" if fill is not None else ""
    return (f"{d['mode']} precision pairs, native_contacts {d['native']}, rigid laws on {d['rigid']}{imp}, "
            f"soft solids on {d['soft']}, noslip as Alloy's scoped noslip")


def on_default_route(law_id: str, solver: str) -> bool:
    """True when Alloy would run this law's object class on `solver` by default."""
    prefix = law_prefix(law_id)
    if prefix in NOT_DEFAULT:
        return False
    d = defaults()
    if prefix in RIGID_LAWS:
        return solver == d["rigid"]
    if prefix in SOFT_LAWS:
        return solver == d["soft"]
    return True


def precision_label(prefix: str = "") -> str:
    """Regex for producer rows on the default precision route, e.g. "sdf / mujoco" (CT-1) or "routed / mujoco"."""
    d = defaults()
    return rf"{prefix or d['mode']} / {d['native']}\b"


__all__ = ["defaults", "describe", "mujoco_default_kwargs", "on_default_route", "precision_label", "routed_gains",
           "routed_integrator", "routed_mujoco_settings", "NOT_DEFAULT"]
