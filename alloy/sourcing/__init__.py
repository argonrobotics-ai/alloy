# SPDX-License-Identifier: Apache-2.0
"""Sourcing: fill missing or ESTIMATE values on imported object cards with cited numbers, and report coverage.

Every value is a :class:`SourcedValue`: SI number, range ``[lo, hi]``, tag and evidence (URL + exact quote).
Tags: ``DOC`` (page read), ``SNIPPET`` (search summary only), ``UNVERIFIED`` (cited, not re-read), ``ESTIMATE`` (a prior
with a range and the reason). Never a bare guess.

- ``queue``: what still needs a value (``scan_all``), from the importer cards in ``<alloy-assets>/imports/``.
- ``cache``: per-category and per-material sourced values (``Cache.find("coffee mug")``), so repeats are cheap.
- ``store``: per-object records in ``<alloy-assets>/sources/imports/<library>/<object>.yaml``; ``card_overrides``
  turns one into card quantities (a sourced mass becomes density through the mesh volume).
- ``match``: fill records from the cache (category values become ESTIMATE priors for the object).
- ``coverage``: shares sourced / weak / estimated / missing.
- CLI: ``python -m alloy.sourcing queue|fill|coverage|lookup|validate`` (JSON out).

Example:
    >>> from alloy.sourcing import SourcedValue
    >>> SourcedValue(field="mass", value=0.35, lo=0.28, hi=0.45, tag="ESTIMATE",
    ...              reason="11-15 oz ceramic mugs weigh 280-450 g (4 listings)").to_q().source
    'model'
"""

from __future__ import annotations

from alloy.sourcing.cache import Cache, CacheEntry, normalize_key
from alloy.sourcing.fields import FIELDS, SourcingError, parse_value
from alloy.sourcing.record import TAGS, Evidence, SourcedValue
from alloy.sourcing.store import ObjectRecord, apply_to_material, card_overrides, load_records, write_record

__all__ = [
    "FIELDS", "TAGS", "Cache", "CacheEntry", "Evidence", "ObjectRecord", "SourcedValue", "SourcingError",
    "apply_to_material", "card_overrides", "load_records", "normalize_key", "parse_value", "write_record",
]
