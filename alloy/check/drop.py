# SPDX-License-Identifier: Apache-2.0
"""``check_object(obj)``: the physics check of one object card on its own, resting on a generated floor (what an
importer calls after each object it writes).

The object is placed in its authored orientation with its lowest point 0.5 mm above a static floor (plywood, sized to
three times the object's footprint and at least 1 m), then checked like a scene: card rules, overlap at load, rest
drift, penetration, health and energy over a short run. Tipping from the authored orientation is a warning here, not
an error (an imported object need not be authored in a resting pose); the settled pose comes with the finding.

Example::

    >>> report = check_object("cube_beech_30", worlds=2, seconds=0.25, device="cpu")
    >>> report.ok, report.scene
    (True, 'drop test: cube_beech_30')
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from alloy.cards import types as ct
from alloy.check.report import CheckReport, Finding, verdict

log = logging.getLogger("alloy.check")

FLOOR = "alloy_check_floor"
FLOOR_TEMPLATE = "table_top_plywood"     # library card whose material and motion the floor copies
FLOOR_HALF_Z = 0.01
GAP_M = 5e-4
MIN_FLOOR_M = 1.0


def check_object(obj: Any, worlds: int | None = None, seconds: float | None = None, *, library: Any = None,
                 device: str | None = None, profile: str = "standard", seed: int = 0) -> CheckReport:
    """Drop-test check of ``obj`` (a library object name, an object YAML path or an ``ObjectCard``).

    Example::

        >>> check_object("library/objects/mug_stoneware.yaml").checks["rest_drift"]       # doctest: +SKIP
        'pass'
    """
    from alloy.check.api import check, resolve_settings  # noqa: PLC0415
    from alloy.check.source import _library  # noqa: PLC0415

    dev, worlds, seconds = resolve_settings(device, worlds, seconds)
    base = _library(library, None)
    card = _object_card(obj, base)
    missing = _missing_files(card, base)
    if missing:
        return missing
    with tempfile.TemporaryDirectory(prefix="alloy-check-") as tmp:
        lib = _drop_library(Path(tmp), card, base)
        scene = _drop_scene(card, lib, dev)
        if isinstance(scene, CheckReport):
            return scene
        report = check(scene, worlds, seconds, device=dev, profile=profile, seed=seed, library=lib)
    report.scene = f"drop test: {card.name}"
    report.findings = [_soften_tip(f) for f in report.findings if FLOOR not in f.objects or len(f.objects) > 1]
    for name, state in report.checks.items():
        if not state.startswith("skipped"):
            report.checks[name] = verdict([f for f in report.findings if f.check == name])
    return report


def _missing_files(card: ct.ObjectCard, lib: Any) -> CheckReport | None:
    """A failed report when ``card`` is an imported object whose files cannot be fetched (``alloy.check.fetch``)."""
    from alloy.check.fetch import fetch_imported  # noqa: PLC0415

    found = fetch_imported(lib, [card.name])
    if not found:
        return None
    report = CheckReport(scene=f"drop test: {card.name}")
    report.add(found, "cards")
    for name in ("overlap", "rest_drift", "penetration", "health", "energy"):
        report.skip(name, "the object's files are missing")
    return report


def _object_card(obj: Any, lib: Any) -> ct.ObjectCard:
    if isinstance(obj, ct.ObjectCard):
        return obj
    path = Path(str(obj)).expanduser()
    if path.is_file():
        from alloy.cards.loader import load_object  # noqa: PLC0415

        return load_object(path)
    if isinstance(obj, str) and lib.has_object(obj):
        return lib.peek_object(obj)
    raise ValueError(f"check_object({obj!r}): pass a library object name, an object YAML path or an ObjectCard")


def _drop_library(root: Path, card: ct.ObjectCard, base: Any) -> Any:
    """A library with ``card`` (written under ``root`` unless ``base`` already holds it) in front of ``base``."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.cards.loader import dump_card  # noqa: PLC0415

    (root / "objects").mkdir(parents=True)
    if not base.has_object(card.name) or base.peek_object(card.name) != card:
        (root / "objects" / f"{card.name}.yaml").write_text(dump_card(plain_card(card)), encoding="utf-8")
    return CardLibrary([root, *base.roots], assets=base.assets)


def _drop_scene(card: ct.ObjectCard, lib: Any, device: str) -> ct.SceneCard | CheckReport:
    """The object alone at the origin is assembled once to measure it; the floor is resized to fit and the object
    lifted so its lowest point clears the floor top (z = 0) by ``GAP_M``."""
    from alloy.check.bounds import object_aabb  # noqa: PLC0415
    from alloy.check.objects import ObjectIndex  # noqa: PLC0415
    from alloy.check.overlap import initial_state  # noqa: PLC0415
    from alloy.core.errors import AlloyError  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    alone = ct.SceneCard(robots=(), objects=(ct.Placement(card.name, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)),))
    try:
        built = SceneAssembler(lib).assemble(alone, "cpu", world_count=1)
    except AlloyError as exc:
        report = CheckReport(scene=f"drop test: {card.name}")
        report.add([Finding("cards", "error", f"{card.name} does not assemble: {type(exc).__name__}: {exc}",
                            "change what the message names in the object card", objects=(card.name,))], "cards")
        return report
    index = ObjectIndex.build(built, alone, lib)
    box = object_aabb(built.model, initial_state(built.model), index, card.name)
    lo, hi = (box if box is not None else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    size = float(max(MIN_FLOOR_M, 3.0 * max(hi[0] - lo[0], hi[1] - lo[1])))   # numpy scalars break the YAML dump
    _write_floor(lib, size)
    pose = (-0.5 * (lo[0] + hi[0]), -0.5 * (lo[1] + hi[1]), GAP_M - lo[2], 0.0, 0.0, 0.0, 1.0)
    log.info("check_object %s: extent %s m, floor %.3g m, start z %.4g m", card.name,
             [round(float(h - l), 4) for l, h in zip(lo, hi)], size, pose[2])
    floor = ct.Placement(FLOOR, (0.0, 0.0, -FLOOR_HALF_Z, 0.0, 0.0, 0.0, 1.0))
    return ct.SceneCard(robots=(), objects=(floor, ct.Placement(card.name, tuple(float(v) for v in pose))))


def _write_floor(lib: Any, size: float) -> None:
    """A static plywood box ``size`` x ``size`` x 20 mm (the template card's material) in the drop library."""
    from alloy.cards.loader import dump_card  # noqa: PLC0415

    floor = dataclasses.replace(lib.peek_object(FLOOR_TEMPLATE), name=FLOOR,
                                shape=ct.Primitive("box", (float(size) / 2, float(size) / 2, FLOOR_HALF_Z)))
    (lib.roots[0] / "objects" / f"{FLOOR}.yaml").write_text(dump_card(floor), encoding="utf-8")
    lib.rescan()


def plain_card(card: Any) -> Any:
    """``card`` with every numpy scalar (``np.float64(0.25)``) replaced by its Python value, so ``dump_card`` can
    write it; cards built from numpy arrays (meshes, measurements) carry them.

    Example::

        >>> plain_card(ct.Primitive("box", (np.float64(0.25), 0.25, 0.01))).size
        (0.25, 0.25, 0.01)
    """
    if isinstance(card, np.generic):
        return card.item()
    if isinstance(card, tuple):
        return tuple(plain_card(v) for v in card)
    if isinstance(card, list):
        return [plain_card(v) for v in card]
    if isinstance(card, Mapping):
        return {k: plain_card(v) for k, v in card.items()}
    if dataclasses.is_dataclass(card) and not isinstance(card, type):
        return dataclasses.replace(card, **{f.name: plain_card(getattr(card, f.name))
                                            for f in dataclasses.fields(card) if f.init})
    return card


def _soften_tip(f: Finding) -> Finding:
    if f.check == "rest_drift" and f.level == "error" and "tipped" in f.message:
        return dataclasses.replace(f, level="warn", fix=f"{f.fix}; for an imported object, author it in a resting "
                                   f"orientation (the suggested pose's rotation) if it should stand as placed")
    return f


__all__ = ["check_object", "plain_card"]
