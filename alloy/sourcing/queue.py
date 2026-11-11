# SPDX-License-Identifier: Apache-2.0
"""The work queue: every value on an imported object card that is missing or ESTIMATE (optionally the weak ones).

Scans ``<assets>/imports/<library>/`` for object cards, resolves each card's material (import folder first, then the
engine ``library/materials``), and reads the tag of each material leaf (``alloy.sourcing.tags``). A per-object record
in ``<assets>/sources/imports/<library>/<object>.yaml`` supersedes the card's tag for that field.

Importer hand-off files add hints (category, product query, extra fields such as ``mass`` for an effective density):
the pipeline's ``queue.jsonl`` and this package's ``sourcing_queue.jsonl`` (``alloy.sourcing.handoff``).

Example:
    >>> from alloy.sourcing.queue import QueueItem
    >>> QueueItem(library="gso", object="mug", card="objects/mug.yaml", material="mug_mat", field="mass",
    ...           card_path="material.density", tag="ESTIMATE").needs_work()
    True
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from alloy.cards import types as ct
from alloy.cards.library import default_root
from alloy.cards.loader import load_any
from alloy.cards.paths import iter_q_leaves
from alloy.core.errors import CardError
from alloy.sourcing.fields import PATH_TO_FIELD
from alloy.sourcing.handoff import read_hints
from alloy.sourcing.paths import imports_root, sources_root
from alloy.sourcing.store import load_records
from alloy.sourcing.tags import has_link, tag_of

log = logging.getLogger("alloy.sourcing")

OPEN_TAGS = ("ESTIMATE", "MISSING")
EFF_SUFFIX = "_eff"  # the importer pipeline's per-object effective material
WEAK_TAGS = ("SNIPPET", "UNVERIFIED")


@dataclass(frozen=True)
class QueueItem:
    """One value to source for one object. ``tag`` is the effective tag (record if any, else card)."""

    library: str
    object: str
    card: str
    material: str
    field: str
    card_path: str
    tag: str
    value: float | None = None
    lo: float | None = None
    hi: float | None = None
    note: str = ""
    category: str = ""
    product: str = ""
    url: str = ""
    recorded: bool = False
    documented: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    card_tag: str = ""
    """The card's own tag, before any record (``tag`` is the record's when there is one)."""

    def needs_work(self, weak: bool = False) -> bool:
        """True if the value is ESTIMATE or MISSING (or SNIPPET/UNVERIFIED with ``weak``)."""
        return self.tag in OPEN_TAGS or (weak and self.tag in WEAK_TAGS)

    def to_json(self) -> dict[str, Any]:
        """Plain dict for the CLI."""
        return asdict(self)


@dataclass
class Scan:
    """Every item found (open or not) plus files that could not be read, with the reason."""

    items: list[QueueItem] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def open(self, weak: bool = False) -> list[QueueItem]:
        """Items that still need sourcing."""
        return [i for i in self.items if i.needs_work(weak)]


def libraries(imports: Path) -> list[str]:
    """Importer library folders under ``<assets>/imports`` (hidden folders skipped)."""
    if not imports.is_dir():
        return []
    return sorted(p.name for p in imports.iterdir() if p.is_dir() and not p.name.startswith("."))


def _cards(folder: Path, scan: Scan) -> Iterator[tuple[Path, Any]]:
    for path in sorted(folder.rglob("*.yaml")):
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        try:
            yield path, load_any(path)
        except (CardError, OSError, ValueError, AttributeError) as e:
            scan.skipped.append({"file": str(path), "reason": str(e)[:300]})


def _materials(folder: Path, scan: Scan) -> dict[str, ct.MaterialCard]:
    mats: dict[str, ct.MaterialCard] = {}
    for root in (default_root() / "materials", folder):
        for _, card in _cards(root, scan) if root.is_dir() else ():
            if isinstance(card, ct.MaterialCard):
                mats[card.name] = card
    return mats


def _documented(rec: Any, q: Any) -> bool:
    if rec is not None:
        return any(ev.tag in ("DOC", "SNIPPET") for ev in rec.evidence)
    return q is not None and has_link(q.note)


def _leaf_items(base: dict[str, Any], mat: ct.MaterialCard | None, recorded: dict[str, Any]) -> Iterator[QueueItem]:
    leaves = dict(iter_q_leaves(mat)) if mat is not None else {}
    for rel, fname in PATH_TO_FIELD.items():
        if rel not in leaves and fname in ("youngs", "poisson"):
            continue  # rigid-only material: no elastic values to source
        q = leaves.get(rel)
        rec = recorded.get(fname)
        tag = rec.tag if rec is not None else tag_of(q)
        yield QueueItem(**base, field=fname, card_path=f"material.{rel}", tag=tag, card_tag=tag_of(q),
                        value=None if q is None else q.value, lo=q.unc[0] if q and q.unc else None,
                        hi=q.unc[1] if q and q.unc else None, note="" if q is None else q.note,
                        recorded=rec is not None, documented=_documented(rec, q))


def _as_mass(density: QueueItem, recorded: dict[str, Any]) -> QueueItem:
    """The density item of an effective (per-object) material, restated as the object's mass to source."""
    rec = recorded.get("mass")
    if rec is None:
        return dataclasses.replace(density, field="mass")
    return dataclasses.replace(density, field="mass", tag=rec.tag, recorded=True,
                               documented=_documented(rec, None))


def _effective(card: ct.ObjectCard, hint: dict[str, Any]) -> bool:
    """True if the card's density is a per-object effective density (so the value to source is the mass): the
    material is ``<card>_eff``, or the importer's queue names the card's density as the estimate."""
    return card.material == f"{card.name}{EFF_SUFFIX}" or bool(hint.get("density_is_mass"))


def _mass_item(base: dict[str, Any], hint: dict[str, Any], recorded: dict[str, Any], density: QueueItem) -> QueueItem:
    """The object's mass when its material is shared (density x mesh volume): its own item, tagged by the record,
    else ESTIMATE when the importer says it estimated it, else the density's tag."""
    rec = recorded.get("mass")
    note = str(hint.get("note") or "")
    card_tag = "ESTIMATE" if note.startswith("ESTIMATE") else (density.card_tag or density.tag)
    tag = rec.tag if rec is not None else card_tag
    own = base["extra"].get("mass_kg")
    return QueueItem(**base, field="mass", card_path=density.card_path, tag=tag, card_tag=card_tag,
                     value=float(own) if isinstance(own, (int, float)) else None, note=note,
                     recorded=rec is not None, documented=_documented(rec, None) if rec is not None else False)


def _object_items(lib: str, rel: str, card: ct.ObjectCard, mats: dict[str, ct.MaterialCard],
                  hint: dict[str, Any], recorded: dict[str, Any]) -> list[QueueItem]:
    base = {"library": lib, "object": card.name, "card": rel, "material": card.material,
            "category": str(hint.get("category", "")), "product": str(hint.get("product", "")),
            "url": str(hint.get("url", "")), "extra": dict(hint.get("extra") or {})}
    items = list(_leaf_items(base, mats.get(card.material), recorded))
    if _effective(card, hint):
        return [_as_mass(i, recorded) if i.field == "density" else i for i in items]
    if "mass" in (hint.get("fields") or ()) or "mass" in recorded:
        density = next(i for i in items if i.field == "density")
        items.append(_mass_item(base, hint, recorded, density))
    return items


def scan_library(lib: str, assets: Path | None = None, sources: Path | None = None) -> Scan:
    """All value items for one importer library (``<assets>/imports/<lib>``).

    Example:
        >>> import tempfile
        >>> scan_library("none", Path(tempfile.mkdtemp())).items
        []
    """
    folder = imports_root(assets) / lib
    scan = Scan()
    if not folder.is_dir():
        return scan
    mats = _materials(folder, scan)
    hints = read_hints(folder)
    records = load_records(lib, sources or sources_root(assets))
    for path, card in _cards(folder, scan):
        if not isinstance(card, ct.ObjectCard):
            continue
        if card.material not in mats:
            scan.skipped.append({"file": str(path), "reason": f"material {card.material!r} not found"})
        rec = records.get(card.name)
        recorded = rec.values if rec is not None else {}
        scan.items.extend(_object_items(lib, str(path.relative_to(folder)), card, mats,
                                        hints.get(card.name, {}), recorded))
    log.info("queue: %s: %d items, %d skipped files", lib, len(scan.items), len(scan.skipped))
    return scan


def scan_all(assets: Path | None = None, only: list[str] | None = None,
             sources: Path | None = None) -> dict[str, Scan]:
    """``scan_library`` for every importer folder (or the named ones); ``sources`` overrides where records are read."""
    libs = only or libraries(imports_root(assets))
    return {lib: scan_library(lib, assets, sources) for lib in libs}
