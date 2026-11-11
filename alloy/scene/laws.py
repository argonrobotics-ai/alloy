# SPDX-License-Identifier: Apache-2.0
"""WP2's laws (SPEC §5.3: A10-A12; §6.4: rig twins R5, R6, R8, R9), registered with `register_law` on import.

`alloy.bench.law.discover()` imports this module for the A10 A11 A12 and R5 R6 R8 R9 catalogue entries. A family whose
module is missing (not merged yet) stays out of the registry and the catalogue reports it `not_run`.
"""

from __future__ import annotations

import importlib
import logging

from alloy.bench.law import register_law

log = logging.getLogger("alloy.scene")

# (module, LawSpec attribute, twin class attribute)
LAWS: tuple[tuple[str, str, str], ...] = (
    ("alloy.scene.analytic_capstan", "LAW", "CapstanTwin"),
    ("alloy.scene.analytic_rods", "LAW_HANGING", "HangingChainTwin"),
    ("alloy.scene.analytic_rods", "LAW_CATENARY", "CatenaryTwin"),
    ("alloy.scene.analytic_rods", "LAW_CANTILEVER", "CantileverTwin"),
    ("alloy.scene.analytic_torsion", "LAW_TORSION", "TorsionTwin"),
    ("alloy.scene.analytic_rods_vbd", "LAW_TORSION_VBD", "TorsionVBDTwin"),
    ("alloy.scene.analytic_rods_vbd", "LAW_CANTILEVER_VBD", "CantileverVBDTwin"),
    ("alloy.scene.analytic_rods_vbd", "LAW_HANGING_VBD", "HangingChainVBDTwin"),
    ("alloy.scene.analytic_rods_vbd", "LAW_CATENARY_VBD", "CatenaryVBDTwin"),
    ("alloy.scene.analytic_rods_vbd", "LAW_CAPSTAN_VBD", "CapstanVBDTwin"),
    ("alloy.scene.analytic_buckling", "LAW", "BucklingTwin"),
    ("alloy.scene.twins.squeeze", "LAW", "SqueezeTwin"),
    ("alloy.scene.twins.drape_route", "LAW", "DrapeRouteTwin"),   # measured fabrics, production route (SCR-WP2-14)
    ("alloy.scene.twins.granular", "LAW", "GranularTwin"),
    ("alloy.scene.twins.rope", "LAW", "RopeTwin"),
)


def _twin_class(mod, name: str | None):
    if name is not None:
        return getattr(mod, name)
    classes = [v for k, v in vars(mod).items() if k.endswith("Twin") and isinstance(v, type)
               and v.__module__ == mod.__name__]
    if len(classes) != 1:
        raise ImportError(f"{mod.__name__}: expected one *Twin class, found {[c.__name__ for c in classes]}")
    return classes[0]


def register_all() -> list[str]:
    """Register every law whose module imports; returns the registered law ids."""
    out = []
    for mod_name, spec_attr, twin_attr in LAWS:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name is None or not (mod_name == exc.name or mod_name.startswith(exc.name + ".")):
                raise                               # a dependency of an existing law module is missing: loud
            log.info("law module %s not available (%s)", mod_name, exc.name)
            continue
        spec = getattr(mod, spec_attr)
        register_law(spec, _twin_class(mod, twin_attr))
        out.append(spec.id)
    return out


REGISTERED = register_all()

__all__ = ["LAWS", "REGISTERED", "register_all"]
