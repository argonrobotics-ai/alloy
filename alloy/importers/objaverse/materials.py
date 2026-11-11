# SPDX-License-Identifier: Apache-2.0
"""Material cards the Objaverse import adds: rigid stand-ins for food, which the engine library lacks. They are the
base materials of food categories; each object's ``_eff`` material copies their surface.

All values here are ESTIMATES (food is not in the engine's sourced library); the sourcing agent replaces them.

Example:
    >>> from alloy.importers.objaverse.materials import food_materials
    >>> sorted(m.name for m in food_materials())
    ['objv_bakery', 'objv_food_moist', 'objv_produce']
"""

from __future__ import annotations

from alloy.cards import types as ct
from alloy.core.quantity import Q


def _q(value: float, lo: float, hi: float, note: str) -> Q:
    return Q(value, "model", (lo, hi), f"ESTIMATE {note}")


def _food(name: str, rho: tuple[float, float, float], mu: tuple[float, float, float], e: tuple[float, float, float],
          what: str) -> ct.MaterialCard:
    mu_d = (mu[0] * 0.8, mu[1] * 0.8, mu[2] * 0.8)
    surface = ct.Surface(
        mu_static=_q(mu[1], mu[0], mu[2], f"{what} on dry tableware/gripper pads, typical range"),
        mu_dynamic=_q(mu_d[1], mu_d[0], mu_d[2], f"{what}: 0.8 x static (no single source)"),
        restitution=_q(e[1], e[0], e[2], f"{what}: soft, damped impacts"))
    return ct.MaterialCard(name=name, density=_q(rho[1], rho[0], rho[2], f"bulk density of {what}"), surface=surface)


def food_materials() -> list[ct.MaterialCard]:
    """Rigid stand-ins for food: produce (fruit and vegetables), bakery (bread, pastry), moist (meat, dairy, eggs)."""
    return [
        _food("objv_produce", (800.0, 950.0, 1100.0), (0.35, 0.5, 0.7), (0.15, 0.3, 0.5),
              "fresh fruit and vegetables (apple ~850, potato ~1080 kg/m^3)"),
        _food("objv_bakery", (120.0, 250.0, 450.0), (0.45, 0.6, 0.8), (0.02, 0.08, 0.2),
              "bread and pastry (crumb with crust)"),
        _food("objv_food_moist", (950.0, 1050.0, 1150.0), (0.3, 0.45, 0.65), (0.05, 0.15, 0.3),
              "meat, dairy and eggs"),
    ]
