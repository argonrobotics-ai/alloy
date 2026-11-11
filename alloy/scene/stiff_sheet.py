# SPDX-License-Identifier: Apache-2.0
"""Stiff CLOTH_VBD sheets start each step from Newton's inertial guess (launch item 2, coordinator 2026-11-08 17:45Z).

The cloth default (``cloth_accuracy``) starts every VBD step from the adaptive guess, which keeps only the share of
gravity a particle realised over its last step. On a fabric that is what lets a drape converge at 40 iterations (L5
drape: DC 0.728 at 0.36 % strain; from the inertial guess 0.36 at 4.1 %, alongside Chebyshev). On a stiff sheet, 40
iterations are far from converged and the result follows the guess, so dropping gravity on the up-swing and keeping it
on the down-swing pumps energy in. Sheets dropped onto two round supports (CPU, 3 s, check() ok throughout) never
rest: PET 0.5 mm climbs 28 mm above them, corrugated cardboard 3 mm still moves at 0.45 m/s and walks. From the
inertial guess (Chebyshev kept) the cardboard rests (0.0001 m/s) and nothing climbs. (SC10 / OSS-17 found it on films;
WP2 and WP11 measured it on CLOTH_VBD; docs/notes/wp2-results.md.)

Which sheets: the bending length c = (D / (rho h g))^(1/3), D the sheet's bending rigidity as the emitter uses it
(``derive.sheet_bending``). The library's fabrics are 7-20 mm (cotton, polyester, towels, Petrak's KES fabrics), soft
vinyl 33 mm, silicone 25 mm; PET, nylon, polypropylene, cardboard and metal sheets 110-990 mm. Above STIFF_BENDING_LENGTH
the router tags the decision with STIFF_EFFECT, and the VBD entry then runs the inertial start (``entry_start``). The
entry is shared, so a fabric in the same scene as a stiff sheet starts from the inertial guess too (and drapes less
converged). A stiff sheet at 40 iterations still sags far too deep (PET over a 10 cm span: 33 mm against 0.3 mm), but
it no longer gains energy (docs/known_gaps/cloth-stiff-sheet-sag.md).
"""

from __future__ import annotations

from typing import Any

from alloy.scene import derive

STIFF_EFFECT = "vbd:start=inertial"
STIFF_BENDING_LENGTH = 0.05          # m: between soft vinyl (33 mm) and the stiffest plastics' thinnest sheets (110 mm)
_G = 9.81


def bending_length(obj: Any) -> float:
    """c = (D / (rho h g))^(1/3) [m] of a sheet object, D from ``derive.sheet_bending``; 0 when it cannot be judged."""
    mat = obj.material
    el = getattr(mat, "elastic", None)
    t = obj.card.features.thickness
    h = float(t.value) if t is not None else float(obj.extents[2])
    if el is None or el.youngs is None or not h > 0.0:
        return 0.0
    nu = float(el.poisson.value) if el.poisson is not None else 0.3
    D = derive.sheet_bending(mat, float(el.youngs.value), nu, h)
    w = float(mat.density.value) * h * _G
    return (D / w) ** (1.0 / 3.0) if D > 0.0 and w > 0.0 else 0.0


def tag(obj: Any, effects: list[str], reasons: list[str]) -> None:
    """Tag a CLOTH_VBD decision whose sheet is stiff (bending length over STIFF_BENDING_LENGTH)."""
    c = bending_length(obj)
    if c > STIFF_BENDING_LENGTH:
        effects.append(STIFF_EFFECT)
        reasons.append(f"stiff sheet: bending length {c * 1e3:.0f} mm > {STIFF_BENDING_LENGTH * 1e3:.0f} mm, so the VBD "
                       "entry starts its steps from Newton's inertial guess, not the adaptive one (an unconverged "
                       "stiff sheet gains energy from it: alloy.scene.stiff_sheet)")


def entry_start(kw: dict[str, Any], effects: set[str]) -> dict[str, Any]:
    """The VBD entry's kwargs with the inertial start when a decision is a stiff sheet (the rest unchanged)."""
    if STIFF_EFFECT in effects and kw.get("alloy_initial_guess") == "adaptive":
        return {**kw, "alloy_initial_guess": "inertial"}
    return kw


__all__ = ["STIFF_BENDING_LENGTH", "STIFF_EFFECT", "bending_length", "entry_start", "tag"]
