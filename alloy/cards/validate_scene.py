# SPDX-License-Identifier: Apache-2.0
"""Scene validation: placements, poses, randomization ranges (each end re-validated), inline pairs, attachments,
emitters and robots must all resolve (SPEC §3.2 cross-reference rules)."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from alloy.cards import types as ct
from alloy.cards.paths import CardView, PathError, get_value, with_value
from alloy.cards.validate import (Issues, check_nonneg, check_plain_floats, check_positive, errors, fmt, material_issues,
                                  pair_issues)
from alloy.core.errors import CardError

if TYPE_CHECKING:
    from alloy.cards.library import CardLibrary

log = logging.getLogger("alloy.cards")

QUAT_TOL = 1e-6


def _check_pose(iss: Issues, path: str, pose: tuple[float, ...]) -> None:
    if len(pose) != 7:
        iss.error(path, f"pose is (x, y, z, qx, qy, qz, qw), got {len(pose)} values")
        return
    if not all(math.isfinite(v) for v in pose):
        iss.error(path, "non-finite pose")
        return
    norm = math.sqrt(sum(v * v for v in pose[3:]))
    if abs(norm - 1.0) > QUAT_TOL:
        iss.error(path, f"quaternion norm {norm:.9g} is not 1 (tolerance {QUAT_TOL})")


def _view(lib: CardLibrary, card: ct.ObjectCard, placement: ct.Placement) -> CardView:
    parts = {k: lib.material(v) for k, v in card.part_materials.items()}
    return CardView(card, lib.material(card.material), parts, tuple(placement.pose))


def _end_errors(lib: CardLibrary, view: CardView, path: str, value: float, rigid_dt: float) -> list[ct.CardIssue]:
    """Errors the card would have with ``path`` set to ``value`` (materials and object re-validated)."""
    from alloy.cards.validate_object import object_issues  # noqa: PLC0415 - avoids an import cycle

    v2 = with_value(view, path, value)
    iss = Issues(view.card.name)
    if path.startswith("material."):
        material_issues(v2.material, iss.sub("material"))
    elif path.startswith("part_materials."):
        part = path.split(".")[1]
        material_issues(v2.parts[part], iss.sub(f"part_materials.{part}"))
    elif not path.startswith("pose."):
        object_issues(v2.card, lib, iss, rigid_dt)
    return errors(iss.items)


def _check_randomize(iss: Issues, lib: CardLibrary, view: CardView, prefix: str, pl: ct.Placement,
                     rigid_dt: float) -> None:
    seen: set[str] = set()
    for j, r in enumerate(pl.randomize):
        p = f"{prefix}.randomize.{j}"
        if r.field in seen:
            iss.error(f"{p}.field", f"{r.field!r} randomized twice")
        seen.add(r.field)
        try:
            get_value(view, r.field)
        except PathError as exc:
            iss.error(f"{p}.field", f"not a numeric leaf of {pl.object!r}: {exc.message}")
            continue
        if not (math.isfinite(r.lo) and math.isfinite(r.hi)) or r.lo > r.hi:
            iss.error(f"{p}.lo", f"need finite lo <= hi, got [{fmt(r.lo)}, {fmt(r.hi)}]")
            continue
        for end, value in (("lo", r.lo), ("hi", r.hi)):
            bad = _end_errors(lib, view, r.field, value, rigid_dt)
            if bad:
                iss.error(f"{p}.{end}", f"{r.field} = {fmt(value)} makes the card invalid: {bad[0].field}: {bad[0].message}")


def _check_objects(iss: Issues, scene: ct.SceneCard, lib: CardLibrary, rigid_dt: float) -> list[ct.CardIssue]:
    obj_issues: list[ct.CardIssue] = []
    checked: set[str] = set()
    for i, pl in enumerate(scene.objects):
        _check_pose(iss, f"objects.{i}.pose", pl.pose)
        if not lib.has_object(pl.object):
            iss.error(f"objects.{i}.object", f"unknown object {pl.object!r}")
            continue
        try:
            view = _view(lib, lib.peek_object(pl.object), pl)
        except CardError as exc:
            iss.error(f"objects.{i}.object", f"object {pl.object!r} does not resolve: {exc}")
            continue
        if pl.object not in checked:
            checked.add(pl.object)
            obj_issues.extend(lib.object_issues(pl.object, rigid_dt))
        _check_randomize(iss, lib, view, f"objects.{i}", pl, rigid_dt)
    return obj_issues


def placement_issues(lib: CardLibrary, card: ct.ObjectCard, placement: ct.Placement,
                     rigid_dt: float = 1.25e-3) -> list[ct.CardIssue]:
    """Pose and randomization rules for one placement of ``card`` (paths start with ``placement.``)."""
    iss = Issues(card.name)
    _check_pose(iss, "placement.pose", placement.pose)
    try:
        view = _view(lib, card, placement)
    except CardError as exc:
        iss.error("placement.object", f"materials of {card.name!r} do not resolve: {exc}")
        return iss.items
    _check_randomize(iss, lib, view, "placement", placement, rigid_dt)
    return iss.items


def _placed_names(scene: ct.SceneCard) -> set[str]:
    """Placement names as the assembler gives them (WP2 ``SceneAssembler._resolve``): the first placement of an
    object is its name, the k-th repeat ``name#k`` (k = 1, 2, ...); robots by name."""
    names: set[str] = set()
    counts: dict[str, int] = {}
    for pl in scene.objects:
        k = counts.get(pl.object, 0)
        counts[pl.object] = k + 1
        names.add(pl.object if k == 0 else f"{pl.object}#{k}")
    return names | {r.name for r in scene.robots}


def _check_impedance(iss: Issues, p: str, at: ct.Attachment) -> None:
    """``timeconst`` / ``dampratio`` (SCR-DC3-3): > 0, and only on weld/joint (they set an equality's solref)."""
    for field in ("timeconst", "dampratio"):
        q = getattr(at, field)
        if q is None:
            continue
        if at.kind == "spring":
            iss.error(f"{p}.{field}", f"{field} sets a weld/joint equality's solref; a spring has none")
        check_positive(iss, f"{p}.{field}", q)


def _check_attachments(iss: Issues, scene: ct.SceneCard) -> None:
    names = _placed_names(scene)
    counts: dict[str, int] = {}
    for pl in scene.objects:
        counts[pl.object] = counts.get(pl.object, 0) + 1
    for i, at in enumerate(scene.attachments):
        p = f"attachments.{i}"
        for side in ("a", "b"):
            ref = getattr(at, side).partition(":")[0]
            if ref not in names:
                iss.error(f"{p}.{side}", f"{getattr(at, side)!r} names no placed object or robot "
                                         "(repeat placements are name#1, name#2, ...)")
            elif counts.get(ref, 0) > 1:
                iss.warn(f"{p}.{side}", f"{ref!r} is placed {counts[ref]} times; this end is the first placement "
                                        f"({ref}#1 ... name the others)")
        if at.a == at.b:
            iss.error(f"{p}.b", "an attachment joins two different things")
        check_nonneg(iss, f"{p}.break_force", at.break_force)
        check_nonneg(iss, f"{p}.break_strain", at.break_strain)
        _check_impedance(iss, p, at)
        if at.anchor is not None:
            _check_pose(iss, f"{p}.anchor", at.anchor)


def _check_emitters(iss: Issues, scene: ct.SceneCard, lib: CardLibrary) -> None:
    for i, em in enumerate(scene.emitters):
        p = f"emitters.{i}"
        if not lib.has_material(em.material):
            iss.error(f"{p}.material", f"unknown material {em.material!r}")
        elif em.kind == "emitter" and lib.material(em.material).fluid is None:
            iss.error(f"{p}.material", f"material {em.material!r} has no fluid block")
        _check_pose(iss, f"{p}.pose", em.pose)
        for k, h in enumerate(em.half_extents):
            check_positive(iss, f"{p}.half_extents.{k}", h, " m")
        check_nonneg(iss, f"{p}.rate", em.rate, " m^3/s")
        if em.kind == "emitter" and em.rate == 0.0:
            iss.warn(f"{p}.rate", "an emitter with rate 0 emits nothing")
        check_nonneg(iss, f"{p}.start_s", em.start_s, " s")
        if em.stop_s is not None and em.stop_s <= em.start_s:
            iss.error(f"{p}.stop_s", f"stop {fmt(em.stop_s)} s must be after start {fmt(em.start_s)} s")


def _check_robots(iss: Issues, scene: ct.SceneCard, lib: CardLibrary) -> None:
    seen: set[str] = set()
    for i, rb in enumerate(scene.robots):
        p = f"robots.{i}"
        if rb.name in seen:
            iss.error(f"{p}.name", f"duplicate robot name {rb.name!r}")
        seen.add(rb.name)
        _check_pose(iss, f"{p}.base_pose", rb.base_pose)
        if rb.contact_gap is not None and not 0.0 <= rb.contact_gap < float("inf"):
            iss.error(f"{p}.contact_gap", f"contact gap {fmt(rb.contact_gap)} m must be finite and >= 0")
        try:
            path = lib.asset_path(rb.path)
        except CardError as exc:                   # the path exists under both library/ and alloy-assets
            iss.error(f"{p}.path", exc.message)
            continue
        if path is None or not path.is_file():
            iss.error(f"{p}.path", f"robot file not found: {rb.path!r}")


def _check_partners_placed(iss: Issues, scene: ct.SceneCard, lib: CardLibrary) -> None:
    placed = {pl.object for pl in scene.objects}
    for i, pl in enumerate(scene.objects):
        try:
            partners = lib.peek_object(pl.object).precision_partners if lib.has_object(pl.object) else ()
        except CardError:
            continue                                   # already reported by _check_objects
        for entry in partners:
            other = entry.partition(":")[0]
            if other not in placed:
                iss.warn(f"objects.{i}.object", f"precision partner {other!r} of {pl.object!r} is not in the scene")


def validate_scene(scene: ct.SceneCard, lib: CardLibrary, rigid_dt: float = 1.25e-3,
                   name: str = "scene") -> list[ct.CardIssue]:
    """Scene rules plus the rules of every placed object (object issues carry the object's name)."""
    iss = Issues(name)
    check_plain_floats(iss, scene)
    if scene.worlds < 1:
        iss.error("worlds", f"worlds must be >= 1, got {scene.worlds}")
    check_positive(iss, "control_dt", scene.control_dt, " s")
    _check_robots(iss, scene, lib)
    obj_issues = _check_objects(iss, scene, lib, rigid_dt)
    for i, pair in enumerate(scene.pairs):
        pair_issues(pair, iss.sub(f"pairs.{i}"), lib)
    _check_attachments(iss, scene)
    _check_emitters(iss, scene, lib)
    _check_partners_placed(iss, scene, lib)
    return iss.items + obj_issues
