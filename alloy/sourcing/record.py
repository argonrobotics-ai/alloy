# SPDX-License-Identifier: Apache-2.0
"""One sourced value: number, range, tag and the evidence behind it (the alloy-assets sourcing convention).

Tags, strongest first:

- ``DOC``: the page or PDF was fetched and the quote was read in it.
- ``SNIPPET``: the quote was seen only in a search-result summary; re-check before publishing a gate on it.
- ``UNVERIFIED``: a cited reference (handbook, table, older note) that was not re-read in this pass.
- ``ESTIMATE``: a derived value or a prior (category range, proxy material, back-calculation). It always carries a
  range (``lo < hi``) and the reason. Never a bare guess: the reason says what the range rests on.

A value's tag can be no stronger than its strongest evidence, and ``DOC``/``SNIPPET`` evidence needs a URL and an
exact quote. Records serialise to plain dicts (YAML/JSON) with :func:`value_to_dict` / :func:`value_from_dict`.

Example:
    >>> from alloy.sourcing.record import Evidence, SourcedValue
    >>> ev = Evidence(url="https://example.org/mug", quote="Weight: 12 oz", tag="DOC")
    >>> v = SourcedValue(field="mass", value=0.340, lo=0.330, hi=0.350, tag="DOC", evidence=(ev,),
    ...                  derivation="12 oz x 0.02835 kg/oz = 0.340 kg")
    >>> v.to_q().source
    'catalog'
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final, Literal

from alloy.core.quantity import Q
from alloy.sourcing.fields import SourcingError, canonical, check_bounds, parse_value

Tag = Literal["DOC", "SNIPPET", "UNVERIFIED", "ESTIMATE"]
TAGS: Final[tuple[str, ...]] = ("DOC", "SNIPPET", "UNVERIFIED", "ESTIMATE")
"""Strongest first; ``rank`` compares them."""
QSource = Literal["measured", "catalog", "fit", "model"]
_Q_SOURCE: Final[dict[str, str]] = {"DOC": "catalog", "SNIPPET": "catalog", "UNVERIFIED": "catalog", "ESTIMATE": "model"}
_NOTE_MAX: Final[int] = 400


def rank(tag: str) -> int:
    """0 for ``DOC`` up to 3 for ``ESTIMATE``; raises ``SourcingError`` on an unknown tag.

    Example:
        >>> rank("DOC") < rank("ESTIMATE")
        True
    """
    try:
        return TAGS.index(tag)
    except ValueError:
        raise SourcingError(f"unknown tag {tag!r}; use one of {list(TAGS)}") from None


@dataclass(frozen=True)
class Evidence:
    """Where a number came from: URL (or full citation for ``UNVERIFIED``), exact quote, tag, location in the source.

    Example:
        >>> Evidence(url="https://example.org", quote="Net wt 16 oz", tag="DOC", where="product page").tag
        'DOC'
    """

    url: str
    quote: str
    tag: Tag
    where: str = ""
    accessed: str = ""

    def check(self, label: str) -> None:
        """Raise ``SourcingError`` if the evidence cannot back a value."""
        rank(self.tag)
        if self.tag == "ESTIMATE":
            raise SourcingError(f"{label}: evidence cannot be tagged ESTIMATE; put the prior in the value's reason")
        if not self.url.strip():
            raise SourcingError(f"{label}: evidence needs a url (or a full citation for UNVERIFIED)")
        if self.tag in ("DOC", "SNIPPET") and not self.url.startswith(("http://", "https://", "doi:")):
            raise SourcingError(f"{label}: {self.tag} evidence needs an http(s) or doi: url, got {self.url!r}")
        if self.tag in ("DOC", "SNIPPET") and not self.quote.strip():
            raise SourcingError(f"{label}: {self.tag} evidence needs the exact quote from the page")


@dataclass(frozen=True)
class SourcedValue:
    """A value in SI with its range ``[lo, hi]``, tag, evidence, and (for ESTIMATE) the reason.

    ``derivation`` shows the arithmetic from the quoted numbers to SI (unit conversions, mass / volume). ``conditions``
    states what the number applies to (counterface, size, temperature). ``q_source`` overrides the card source
    (``measured`` for a published measurement, ``fit`` for a value fitted from a measured whole-object quantity).

    Example:
        >>> SourcedValue(field="mu_static", value=0.5, lo=0.3, hi=0.7, tag="ESTIMATE",
        ...              reason="glazed ceramic on wood, proxy from ceramic-on-steel tables").to_q().unc
        (0.3, 0.7)
    """

    field: str
    value: float
    lo: float
    hi: float
    tag: Tag
    evidence: tuple[Evidence, ...] = ()
    reason: str = ""
    derivation: str = ""
    conditions: str = ""
    q_source: QSource | None = None

    def __post_init__(self) -> None:
        self.check(self.field)

    def check(self, label: str) -> None:
        """Raise ``SourcingError`` naming what to change if the value breaks a sourcing rule."""
        rank(self.tag)
        for name in ("value", "lo", "hi"):
            if not math.isfinite(getattr(self, name)):
                raise SourcingError(f"{label}: {name} must be finite")
        if not (self.lo <= self.value <= self.hi):
            raise SourcingError(f"{label}: need lo <= value <= hi, got {self.lo:g} <= {self.value:g} <= {self.hi:g}")
        check_bounds(self.field, self.value, label)
        for i, ev in enumerate(self.evidence):
            ev.check(f"{label}.evidence[{i}]")
        if self.tag == "ESTIMATE":
            _check_estimate(self, label)
            return
        if not self.evidence:
            raise SourcingError(f"{label}: a {self.tag} value needs evidence (url + quote); else tag it ESTIMATE")
        best = min(rank(ev.tag) for ev in self.evidence)
        if best > rank(self.tag):
            raise SourcingError(
                f"{label}: tag {self.tag} is stronger than its best evidence ({TAGS[best]}); lower the tag")

    def to_q(self) -> Q:
        """The card quantity: value, source (``model`` for ESTIMATE, else ``catalog`` or ``q_source``), unc, note."""
        source = self.q_source or _Q_SOURCE[self.tag]
        return Q(self.value, source, (self.lo, self.hi), note_for(self))


def _check_estimate(v: SourcedValue, label: str) -> None:
    if not v.reason.strip():
        raise SourcingError(f"{label}: an ESTIMATE needs the reason (what the prior or range rests on)")
    if not v.lo < v.hi:
        raise SourcingError(f"{label}: an ESTIMATE needs a real range (lo < hi), got [{v.lo:g}, {v.hi:g}]")
    if v.q_source in ("measured", "catalog"):
        raise SourcingError(f"{label}: an ESTIMATE cannot have q_source {v.q_source!r}; use model or fit")


def is_category_prior(v: SourcedValue) -> bool:
    """True if ``v`` is a category prior written by ``fill`` (ESTIMATE, reason "category ..."), not a value found for
    this object; ``fill`` may rewrite or drop only these.

    Example:
        >>> is_category_prior(SourcedValue(field="mass", value=0.3, lo=0.2, hi=0.4, tag="ESTIMATE",
        ...                                reason="category mug (Mug): 4 listings"))
        True
    """
    return v.tag == "ESTIMATE" and v.reason.startswith("category ")


def note_for(v: SourcedValue) -> str:
    """A one-line card note: tag, first quote and URL, derivation or reason (at most 400 characters).

    Example:
        >>> note_for(SourcedValue(field="restitution", value=0.5, lo=0.3, hi=0.7, tag="ESTIMATE", reason="ceramic"))
        'ESTIMATE: ceramic'
    """
    parts = [v.tag + ":"]
    if v.evidence:
        ev = min(v.evidence, key=lambda e: rank(e.tag))
        parts.append(f'"{ev.quote.strip()}" ({ev.url.strip()})' if ev.quote.strip() else ev.url.strip())
    for extra in (v.derivation, v.conditions, v.reason):
        if extra.strip():
            parts.append(extra.strip())
    note = " ".join(parts)
    return note if len(note) <= _NOTE_MAX else note[: _NOTE_MAX - 3] + "..."


def evidence_to_dict(ev: Evidence) -> dict[str, Any]:
    """Plain dict for YAML/JSON (empty optional keys dropped)."""
    out: dict[str, Any] = {"url": ev.url, "quote": ev.quote, "tag": ev.tag}
    out.update({k: getattr(ev, k) for k in ("where", "accessed") if getattr(ev, k)})
    return out


def value_to_dict(v: SourcedValue) -> dict[str, Any]:
    """Plain dict for YAML/JSON; ``value_from_dict(value_to_dict(v)) == v``.

    Example:
        >>> d = value_to_dict(SourcedValue(field="mass", value=0.1, lo=0.05, hi=0.2, tag="ESTIMATE", reason="r"))
        >>> sorted(d)
        ['hi', 'lo', 'reason', 'tag', 'value']
    """
    out: dict[str, Any] = {"value": v.value, "lo": v.lo, "hi": v.hi, "tag": v.tag}
    if v.evidence:
        out["evidence"] = [evidence_to_dict(e) for e in v.evidence]
    out.update({k: getattr(v, k) for k in ("reason", "derivation", "conditions", "q_source") if getattr(v, k)})
    return out


_VALUE_KEYS: Final[frozenset[str]] = frozenset(
    {"value", "lo", "hi", "tag", "evidence", "reason", "derivation", "conditions", "q_source"})
_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset({"url", "quote", "tag", "where", "accessed"})


def _unknown(raw: dict[str, Any], allowed: frozenset[str], label: str) -> None:
    extra = sorted(set(raw) - allowed)
    if extra:
        raise SourcingError(f"{label}: unknown keys {extra}; allowed {sorted(allowed)}")


def evidence_from_dict(raw: Any, label: str) -> Evidence:
    """Parse one evidence mapping; raises ``SourcingError`` naming the bad key."""
    if not isinstance(raw, dict):
        raise SourcingError(f"{label}: evidence must be a mapping with url, quote, tag")
    _unknown(raw, _EVIDENCE_KEYS, label)
    try:
        return Evidence(**{k: str(raw[k]) for k in raw})
    except TypeError as e:
        raise SourcingError(f"{label}: evidence needs url, quote and tag ({e})") from None


def value_from_dict(raw: Any, field_name: str, label: str = "") -> SourcedValue:
    """Parse a value mapping; ``value``/``lo``/``hi`` may be unit strings (``"350 g"``).

    Example:
        >>> round(value_from_dict({"value": "350 g", "lo": "300 g", "hi": "400 g", "tag": "ESTIMATE", "reason": "r"},
        ...                       "mass").value, 6)
        0.35
    """
    field_name = canonical(field_name)
    label = label or field_name
    if not isinstance(raw, dict):
        raise SourcingError(f"{label}: a value must be a mapping with value, lo, hi, tag")
    _unknown(raw, _VALUE_KEYS, label)
    missing = [k for k in ("value", "lo", "hi", "tag") if k not in raw]
    if missing:
        raise SourcingError(f"{label}: missing keys {missing}")
    evidence = tuple(evidence_from_dict(e, f"{label}.evidence[{i}]") for i, e in enumerate(raw.get("evidence") or ()))
    return SourcedValue(
        field=field_name, value=parse_value(raw["value"], field_name), lo=parse_value(raw["lo"], field_name),
        hi=parse_value(raw["hi"], field_name), tag=str(raw["tag"]), evidence=evidence,  # type: ignore[arg-type]
        reason=str(raw.get("reason", "")), derivation=str(raw.get("derivation", "")),
        conditions=str(raw.get("conditions", "")), q_source=raw.get("q_source"))


def values_from_dict(raw: Any, label: str) -> dict[str, SourcedValue]:
    """Parse a ``{field: value-mapping}`` block (fields checked against ``alloy.sourcing.fields.FIELDS``)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SourcingError(f"{label}.values must be a mapping of field -> value")
    return {str(k): value_from_dict(v, str(k), f"{label}.values.{k}") for k, v in raw.items()}

