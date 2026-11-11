# SPDX-License-Identifier: Apache-2.0
"""Fill open queue items from the category cache, writing per-object records.

A category value applied to one object is a prior for that object, so it is written as ``ESTIMATE`` with the reason
naming the category; the category's evidence (product pages, datasheets) stays attached, so a reader can see the
range is documented. A value already sourced on the card (``DOC``/``SNIPPET``/``UNVERIFIED``) is never replaced.

Example:
    >>> from alloy.sourcing.cache import CacheEntry
    >>> from alloy.sourcing.record import SourcedValue, is_category_prior
    >>> e = CacheEntry(key="apple", kind="category", title="Apple, medium", values={"mass": SourcedValue(
    ...     field="mass", value=0.18, lo=0.13, hi=0.23, tag="ESTIMATE", reason="USDA medium apple 182 g")})
    >>> as_prior(e.values["mass"], e).reason.startswith("category apple")
    True
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from alloy.sourcing.cache import Cache, CacheEntry, normalize_key
from alloy.sourcing.queue import QueueItem, Scan
from alloy.sourcing.record import SourcedValue, is_category_prior
from alloy.sourcing.store import ObjectRecord, load_records, remove_record, write_record

log = logging.getLogger("alloy.sourcing")


def as_prior(v: SourcedValue, entry: CacheEntry, via: str = "") -> SourcedValue:
    """``v`` re-tagged ESTIMATE as a prior from ``entry`` (evidence kept, reason names the category)."""
    basis = v.reason or v.derivation or f"{v.tag} value for the category"
    prefix = f"category {entry.key} ({entry.title})" + (f", material {via}" if via else "")
    return dataclasses.replace(v, tag="ESTIMATE", q_source=None, reason=f"{prefix}: {basis}"[:600])


def _ranged(v: SourcedValue) -> bool:
    return v.lo < v.hi


def entry_values(entry: CacheEntry, cache: Cache) -> dict[str, SourcedValue]:
    """The category's values, then its material's values for fields the category lacks, all as priors. A point
    value (``lo == hi``) is not used: a prior needs a range, and none is invented here."""
    out = {k: as_prior(v, entry) for k, v in entry.values.items() if _ranged(v)}
    mat = cache.get(entry.material, "material") if entry.material else None
    for k, v in (mat.values.items() if mat else ()):
        if _ranged(v):
            out.setdefault(k, as_prior(v, entry, via=mat.key))
    return out


MATERIAL_TARGET: Final[str] = "material_"
"""A name-map target ``material:<key>`` (normalized ``material_<key>``): the object is a single-material object of
that cached material, so it gets the material's values (friction, restitution, elasticity) and no mass prior."""


def material_entry(target: str, cache: Cache) -> CacheEntry | None:
    """A category-like entry for a ``material:<key>`` map target, or None if ``target`` is not one.

    Example:
        >>> import tempfile
        >>> c = Cache(Path(tempfile.mkdtemp()))
        >>> _ = c.put(CacheEntry(key="abs", kind="material", title="ABS"))
        >>> material_entry("material:abs", c).material, material_entry("mug", c)
        ('abs', None)
    """
    key = normalize_key(target)
    mat = cache.get(key.removeprefix(MATERIAL_TARGET), "material") if key.startswith(MATERIAL_TARGET) else None
    if mat is None:
        return None
    return CacheEntry(key=key, kind="category", title=f"single-material object of {mat.title}", material=mat.key)


def find_entry(items: list[QueueItem], cache: Cache, fuzzy: bool = False) -> CacheEntry | None:
    """The cache entry for one object: a dataset name map (curated per object; a category, or ``material:<key>``
    for a single-material object), else a hinted category that is a cache key or exact alias. With ``fuzzy``, also
    alias matches inside the category, product query or object name; off by default because names mislead (a YCB
    "apple" is a plastic replica, not fruit)."""
    first = items[0]
    mapped = cache.mapped(first.object) or ""
    for entry in (cache.lookup(mapped) if mapped else None, material_entry(mapped, cache),
                  cache.lookup(first.category) if first.category else None):
        if entry is not None:
            return entry
    for name in (first.category, first.product, first.object) if fuzzy else ():
        entry = cache.find(name) if name else None
        if entry is not None:
            return entry
    return None


SPECIFIC_NOTE: Final[re.Pattern[str]] = re.compile(
    r"product description|same make|packing fraction|published|catalogue weight", re.IGNORECASE)
"""Card estimates built from this object's own data (label net contents, a same-make sibling's measured mass): a
category prior does not replace them. Category rules (e.g. "YCB toys as analogues") are replaceable."""


def _replaceable(it: QueueItem, prior: SourcedValue) -> bool:
    if it.tag == "MISSING" or it.card_tag == "MISSING":  # judged on the card's own tag, so a refill is idempotent
        return True
    if prior.reason.split(":", 1)[0] in it.note:
        return True  # the card already carries this prior (an importer applied the record): keep the record behind it
    if SPECIFIC_NOTE.search(it.note):
        return False
    return any(ev.tag in ("DOC", "SNIPPET") for ev in prior.evidence)


@dataclass
class FillReport:
    """What ``fill_from_cache`` did: objects written, objects with no matching category, per-field fills."""

    written: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    filled: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    flags: list[dict[str, Any]] = field(default_factory=list)
    friction_dropped: int = 0

    def to_json(self) -> dict[str, object]:
        """Plain dict for the CLI (``flags``: objects whose own mass estimate disagrees with the documented range;
        ``friction_dropped``: category friction priors left out because they contradict the card's other value)."""
        return {"written": len(self.written), "unmatched": self.unmatched, "filled": dict(self.filled),
                "flagged": len(self.flags), "friction_dropped": self.friction_dropped}


def _mass_ranges(priors: dict[str, SourcedValue], it: QueueItem) -> list[tuple[str, SourcedValue]]:
    """Documented mass ranges for one object: the category's effective density x this scan's volume (size-aware)
    first, then the category's mass range."""
    out: list[tuple[str, SourcedValue]] = []
    rho, vol = priors.get("effective_density"), it.extra.get("volume_m3")
    if rho is not None and isinstance(vol, (int, float)) and vol > 0:
        how = f"effective density {rho.lo:.4g}-{rho.hi:.4g} kg/m^3 x scan volume {vol:.4g} m^3"
        out.append((how, dataclasses.replace(
            rho, field="mass", value=rho.value * vol, lo=rho.lo * vol, hi=rho.hi * vol,
            derivation=(f"{how}; " + rho.derivation)[:600])))
    if "mass" in priors:
        out.append(("category mass range", priors["mass"]))
    return out


FIXED_NOTE: Final[re.Pattern[str]] = re.compile(r"scan volume not used", re.IGNORECASE)
"""A card mass that ignores the object's size (a typical mass): a documented range replaces it outright."""
UNDOCUMENTED_NOTE: Final[re.Pattern[str]] = re.compile(r"engineering estimate", re.IGNORECASE)
"""A card mass with no source behind it: a documented range bounds it (clamped, with the reason)."""


def _mass_value(it: QueueItem, priors: dict[str, SourcedValue], flags: list[dict[str, Any]]) -> SourcedValue | None:
    """The object's mass as a documented category ESTIMATE (coordinator ruling 2026-11-06: generic goods get
    category-level ranges). The object's own scan-based estimate is kept when a documented range contains it; a
    typical mass that ignores size is replaced by the category value; any other estimate outside the range is
    clamped into it, the reason says so, and the object is listed for the importer to review."""
    ranges = _mass_ranges(priors, it)
    own = it.extra.get("mass_kg")
    if not ranges:
        return None
    if not isinstance(own, (int, float)) or not own > 0 or FIXED_NOTE.search(it.note):
        return ranges[0][1]
    for how, r in ranges:
        if r.lo <= own <= r.hi:
            note = f"; value = this object's scan-based estimate {own:.4g} kg, inside the documented range ({how})"
            return dataclasses.replace(r, value=float(own), reason=(r.reason + note)[:700])
    how, r = ranges[0]
    value = min(max(float(own), r.lo), r.hi)
    kind = "undocumented" if UNDOCUMENTED_NOTE.search(it.note) else "analogue-based"
    note = f"; the importer's {kind} estimate {own:.4g} kg lies outside the documented range ({how}), clamped"
    flags.append({"object": it.object, "own_kg": round(float(own), 5), "recorded_kg": round(value, 5),
                  "card_note": it.note[:160], "documented": {h: [round(x.lo, 5), round(x.hi, 5)] for h, x in ranges}})
    return dataclasses.replace(r, value=value, reason=(r.reason + note)[:700])


MIXED_CONTENTS: Final[float] = 1.5
"""An object may be up to this much denser than its category material (contents, inserts, metal parts)."""


def _plausible(it: QueueItem, v: SourcedValue, rho_max: float, flags: list[dict[str, Any]]) -> bool:
    """False (and a flag) if a mass implies a density outside 10 kg/m^3 .. ``rho_max`` for this scan's volume (a scan
    at the wrong scale, or a category that does not fit the object). ``rho_max`` is 1.5 x the category material's
    highest documented density, else 25,000 kg/m^3."""
    vol = it.extra.get("volume_m3")
    if it.field != "mass" or not isinstance(vol, (int, float)) or not vol > 0:
        return True
    rho = v.value / vol
    if 10.0 <= rho <= rho_max:
        return True
    flags.append({"object": it.object, "implied_density": round(rho, 1), "max_density": round(rho_max, 1),
                  "volume_m3": vol, "recorded_kg": None,
                  "reason": "category mass implies an impossible density for this scan; not recorded"})
    return False


THIN_WALL: Final[float] = 2.5
"""How much less wall a scan of an open, thin-walled object may enclose than the real object: scanners thin the walls
(a 9 cm plant pot's mesh holds 24 cm^3 against about 60 cm^3 of real wall), so mass over scan volume can exceed the
material's density by this factor. Applied to categories with no effective density (open objects: pots, bowls)."""


def _density_cap(entry: CacheEntry, cache: Cache) -> float:
    """Highest plausible mass over scan volume for the category: 1.5 x its material's documented density (contents,
    inserts), times ``THIN_WALL`` for open categories (no effective density); 25,000 kg/m^3 at most or without one."""
    mat = cache.get(entry.material, "material") if entry.material else None
    rho = mat.values.get("density") if mat else None
    if rho is None:
        return 25_000.0
    thin = 1.0 if "effective_density" in entry.values else THIN_WALL
    return min(MIXED_CONTENTS * thin * rho.hi, 25_000.0)


FRICTION: Final[tuple[str, str]] = ("mu_static", "mu_dynamic")


def pair_friction(values: dict[str, SourcedValue], card: dict[str, float]) -> list[str]:
    """Drop the category friction priors in ``values`` if, applied over the card (``card``: the card's own
    mu_static / mu_dynamic), they leave mu_dynamic > mu_static, which the material validator rejects. A prior never
    pairs with a card value it contradicts; the card keeps its own pair. Returns the fields dropped.

    Example:
        >>> e = CacheEntry(key="pack", kind="category", title="Pack")
        >>> s = as_prior(SourcedValue(field="mu_static", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE", reason="pet"), e)
        >>> pair_friction({"mu_static": s}, {"mu_static": 0.5, "mu_dynamic": 0.4})
        ['mu_static']
    """
    mu_s, mu_d = (values[k].value if k in values else card.get(k) for k in FRICTION)
    if mu_s is None or mu_d is None or mu_d <= mu_s:
        return []
    dropped = [k for k in FRICTION if k in values and is_category_prior(values[k])]
    for k in dropped:
        del values[k]
    return dropped


def _record_for(items: list[QueueItem], entry: CacheEntry, cache: Cache, old: ObjectRecord | None,
                flags: list[dict[str, Any]]) -> ObjectRecord | None:
    values = dict(old.values) if old else {}
    priors = entry_values(entry, cache)
    rho_max = _density_cap(entry, cache)
    changed = False
    for it in items:
        if it.field in values and not is_category_prior(values[it.field]):
            continue  # a sourced value, or one found for this object (ingest): category rules never touch it
        prior = _mass_value(it, priors, flags) if it.field == "mass" else priors.get(it.field)
        if prior is None or not _replaceable(it, prior) or not _plausible(it, prior, rho_max, flags):
            changed |= values.pop(it.field, None) is not None  # drop a stale estimate this rule no longer backs
            continue
        values[it.field] = prior
        changed = True
    if "mass" in values and "density" in values and is_category_prior(values["density"]):
        del values["density"]  # the mass sets the density through the scan volume; a bulk prior would override it
        changed = True
    if not changed:
        return None
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return ObjectRecord(library=items[0].library, object=items[0].object, values=values,
                        category=entry.key, material=items[0].material, updated=today)


def fill_from_cache(scan: Scan, cache: Cache, sources: Path | None = None, dry_run: bool = False,
                    fuzzy: bool = False) -> FillReport:
    """Write a record for every object whose open items a cache category covers. A prior replaces a card ESTIMATE
    only if it has DOC/SNIPPET evidence and the card's estimate is not built from the object's own data.

    Example:
        >>> import tempfile
        >>> fill_from_cache(Scan(), Cache(Path(tempfile.mkdtemp()))).to_json()["written"]
        0
    """
    report = FillReport()
    by_obj: dict[tuple[str, str], list[QueueItem]] = defaultdict(list)
    card_friction: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for it in scan.items:
        if it.field in FRICTION and it.value is not None:
            card_friction[(it.library, it.object)][it.field] = it.value
    for it in scan.open():
        by_obj[(it.library, it.object)].append(it)
    olds: dict[str, dict[str, ObjectRecord]] = {}
    for (lib, obj), items in sorted(by_obj.items()):
        entry = find_entry(items, cache, fuzzy)
        if entry is None:
            report.unmatched.append(f"{lib}/{obj}")
            continue
        if lib not in olds:
            olds[lib] = load_records(lib, sources)  # once per library (setdefault would re-read every time)
        old = olds[lib].get(obj)
        flags: list[dict[str, Any]] = []
        rec = _record_for(items, entry, cache, old, flags)
        report.flags.extend({"library": lib, "category": entry.key, **f} for f in flags)
        if rec is None:
            continue
        report.friction_dropped += len(pair_friction(rec.values, card_friction.get((lib, obj), {})))
        for k in rec.values:
            report.filled[k] += 1
        if dry_run:
            pass
        elif rec.values:
            write_record(rec, sources)
        else:
            remove_record(lib, obj, sources)
        report.written.append(f"{lib}/{obj}")
    log.info("fill: %d records, %d unmatched objects", len(report.written), len(report.unmatched))
    return report
