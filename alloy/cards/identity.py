# SPDX-License-Identifier: Apache-2.0
"""Resolution and identity: a placed object card becomes a ``ResolvedObject`` with extents, worst-case values of
its randomized fields, the list of default leaves and a content hash.

``card_hash`` is ``digest_of`` the resolved content: the object card, its materials by value with defaults filled
(so editing a material changes every object that uses it) and the placement's randomization ranges (the router
evaluates at the worst case, so two placements with different ranges are different routing inputs). Pose and world
count are not part of it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from alloy.cards import types as ct
from alloy.cards.codec import normalize
from alloy.cards.defaults import default_leaf_paths, fill_view
from alloy.cards.paths import CardView, get_value
from alloy.cards.validate import raise_on_error
from alloy.cards.validate_object import shape_extents
from alloy.core.errors import CardError
from alloy.core.ids import digest_of

if TYPE_CHECKING:
    from alloy.cards.library import CardLibrary

log = logging.getLogger("alloy.cards")

# Which end of a randomized range is worse for routing: toward softer, heavier, faster, tighter (router §3.3:
# δ_loc = F(1−ν²)/(2aE), sag γ = ρAgL³/EI, impact sink ∝ v, tolerance τ). Anything else: the end farther from the
# nominal value (ties: hi).
WORST_LO: tuple[str, ...] = ("*elastic.youngs", "*elastic.poisson", "*elastic.bending", "loads.tolerance",
                             "features.thickness", "features.passage_width", "features.min_wall", "features.bend_stiffness",
                             "features.twist_stiffness", "*.clearance")
WORST_HI: tuple[str, ...] = ("*density", "loads.grip_force", "loads.bend_force", "loads.design_force",
                             "loads.contact_speed", "*fluid.viscosity")


def worst_end(path: str, lo: float, hi: float, nominal: float) -> float:
    """The routing-worst end of [lo, hi] for the leaf at ``path``."""
    if any(fnmatchcase(path, p) for p in WORST_LO):
        return lo
    if any(fnmatchcase(path, p) for p in WORST_HI):
        return hi
    return lo if abs(lo - nominal) > abs(hi - nominal) else hi


def worst_case(view: CardView, randomize: Sequence[ct.Randomization]) -> dict[str, float]:
    return {r.field: worst_end(r.field, r.lo, r.hi, get_value(view, r.field)) for r in randomize}


def card_hash(view: CardView, randomize: Sequence[ct.Randomization] = ()) -> str:
    """Content hash of a resolved object (``view`` with defaults filled) and its randomization ranges."""
    payload = {"object": normalize(view.card), "material": normalize(view.material),
               "part_materials": {k: normalize(view.parts[k]) for k in sorted(view.parts)},
               "randomize": tuple(ct.Randomization(r.field, float(r.lo), float(r.hi)) for r in randomize)}
    return digest_of(payload)


def _parts(card: ct.ObjectCard, lib: CardLibrary) -> Mapping[str, ct.MaterialCard]:
    return {k: lib.material(v) for k, v in sorted(card.part_materials.items())}


def resolve_object(card: ct.ObjectCard, lib: CardLibrary, placement: ct.Placement | None = None,
                   rigid_dt: float = 1.25e-3) -> ct.ResolvedObject:
    """Validate and resolve one object (optionally as placed); any error-level issue raises ``CardError``."""
    from alloy.cards.validate_object import validate_object  # noqa: PLC0415 - import cycle via validate_scene
    from alloy.cards.validate_scene import placement_issues  # noqa: PLC0415

    if placement is not None and placement.object != card.name:
        raise CardError(card.name, "object", f"placement is for {placement.object!r}, not {card.name!r}")
    try:
        own = lib.has_object(card.name) and lib.peek_object(card.name) == card
    except CardError:
        own = False
    issues = list(lib.object_issues(card.name, rigid_dt)) if own else validate_object(card, lib, rigid_dt)
    if placement is not None:
        issues += placement_issues(lib, card, placement, rigid_dt)
    raise_on_error(issues, "resolve")
    view = fill_view(CardView(card, lib.material(card.material), _parts(card, lib),
                              tuple(placement.pose) if placement else None))
    randomize = tuple(placement.randomize) if placement else ()
    try:
        extents = shape_extents(card, lib.mesh_for(card))
    except KeyError as exc:
        raise CardError(card.name, "shape.body", str(exc)) from exc
    if extents is None:
        raise CardError(card.name, "shape", f"extents of a {type(card.shape).__name__} are not computed in v1; "
                        "use a primitive, mesh, curve, particle or MJCF source")
    resolved = ct.ResolvedObject(
        card=card, material=view.material, part_materials=dict(view.parts), card_hash=card_hash(view, randomize),
        extents=extents, worst_case=worst_case(view, randomize), default_leaves=default_leaf_paths(view))
    log.debug("resolved %s hash=%s extents=%s defaults=%d", card.name, resolved.card_hash[:12], extents,
              len(resolved.default_leaves))
    return resolved
