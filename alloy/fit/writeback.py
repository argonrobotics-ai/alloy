# SPDX-License-Identifier: Apache-2.0
"""`write_back`: put fitted values into the card library with provenance, only with held-out evidence (SPEC §3.12).

Refusals (HeldOutGapError, nothing is written): no judged validation verdict; a held-out gap
RE(valid) - RE(calib) > 0.5 x bound for any metric@condition (BEN-2); a judged verdict without a finite gap (a NaN
simulated value, an error verdict, or no calib counterpart). Every updated card is validated before the first write
(CardError, nothing written), e.g. a fitted mu_dynamic above mu_static. Otherwise each touched card gets one
`CardLibrary.put` with the fitted `Q(source="fit", unc, note)` leaves; a pair leaf whose pair card does not exist
creates the card from the two materials (harmonic mu, D13, source "model") with the fitted leaves set.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alloy.cards.library import CardLibrary
from alloy.cards.validate import raise_on_error, validate_material, validate_pair
from alloy.core.quantity import Q
from alloy.fit.errors import FitError, HeldOutGapError
from alloy.fit.params import LeafRef, new_pair_card, parse_leaf, set_leaf
from alloy.fit.types import FitResult

log = logging.getLogger("alloy.fit")
GAP_FRACTION = 0.5


def gap_violations(result: FitResult) -> dict[str, tuple[float, float]]:
    """metric@condition -> (gap, bound) for every verdict except not_run whose gap is missing, non-finite (incl. a
    NaN simulated value or an error verdict) or > 0.5 bound."""
    bad: dict[str, tuple[float, float]] = {}
    for v in result.verdicts:
        if v.verdict == "not_run":
            continue
        gap = result.heldout_gap.get(v.name, math.nan) if v.verdict != "error" else math.nan
        if not math.isfinite(gap) or gap > GAP_FRACTION * v.bound:
            bad[v.name] = (gap, float(v.bound))
    return bad


def check_heldout(result: FitResult) -> None:
    """Raise HeldOutGapError unless every judged metric@condition has a held-out gap <= 0.5 x bound."""
    if not any(v.verdict != "not_run" for v in result.verdicts):
        raise HeldOutGapError(f"{result.fit_id}: no judged validation verdict; refusing to write without held-out "
                              "evidence")
    bad = gap_violations(result)
    if bad:
        text = ", ".join(f"{k}: gap {g:.4g} > {GAP_FRACTION} x {b:.4g}" for k, (g, b) in sorted(bad.items()))
        raise HeldOutGapError(f"{result.fit_id}: held-out gap too large ({text})", bad)


def _card_for(lib: CardLibrary, ref: LeafRef) -> Any:
    if ref.kind == "material":
        return lib.material(ref.name)
    a, b = ref.materials
    card = lib.pair(a, b)
    return card if card is not None else new_pair_card(lib.material(a), lib.material(b))


def _updated_cards(result: FitResult, lib: CardLibrary) -> dict[tuple[str, str], Any]:
    cards: dict[tuple[str, str], Any] = {}
    for leaf, q in result.params.items():
        ref = parse_leaf(leaf)
        key = (ref.kind, ref.name)
        if key not in cards:
            cards[key] = _card_for(lib, ref)
        if q.source != "fit":
            raise FitError(f"{leaf}: write_back takes fitted values only (source {q.source!r})")
        cards[key] = set_leaf(cards[key], ref.path, q)
    return cards


def write_back(result: FitResult, lib: CardLibrary, *, reason: str) -> list[Path]:
    """Write every fitted leaf (one put per card); refuses (HeldOutGapError) without passing held-out evidence."""
    if not reason.strip():
        raise FitError(f"{result.fit_id}: write_back needs a reason")
    check_heldout(result)
    cards = _updated_cards(result, lib)
    for (kind, name), card in sorted(cards.items()):   # all or nothing: validate every card before the first put
        issues = validate_material(card) if kind == "material" else validate_pair(card, lib)
        raise_on_error(issues, f"write_back {result.fit_id} {kind}/{name}")
    paths = []
    for (kind, name), card in sorted(cards.items()):
        leaves = [k for k in result.params if parse_leaf(k).name == name and parse_leaf(k).kind == kind]
        paths.append(lib.put(card, reason=f"{reason} [{result.fit_id}: {', '.join(sorted(leaves))}]"))
        log.info("write_back %s %s/%s: %s", result.fit_id, kind, name, leaves)
    return paths


def values_of(result: FitResult) -> Mapping[str, float]:
    return {leaf: q.value for leaf, q in result.params.items()}


__all__ = ["write_back", "check_heldout", "gap_violations", "values_of", "GAP_FRACTION"]
