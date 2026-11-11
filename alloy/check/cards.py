# SPDX-License-Identifier: Apache-2.0
"""Card validation as check findings: every ``CardIssue`` of the scene and its placed objects (WP1's
``validate_scene``), each with the edit that fixes it.

Example::

    >>> from alloy.check.source import load_source
    >>> found = card_findings(load_source("demo_peg_d12"))
    >>> all(f.check == "cards" for f in found)
    True
"""

from __future__ import annotations

import difflib
import logging
import re

from alloy.cards import types as ct
from alloy.check.source import Source
from alloy.check.report import Finding
from alloy.core.errors import CardError

log = logging.getLogger("alloy.check")

_RANGE = re.compile(r"^(?P<v>\S+?)(?P<u>[^\d\s.\-+]*) outside (?P<rng>[\[(].*[\])])")
_DEFAULT_LEAF = "still at source 'default'"
_UNKNOWN_OBJ = re.compile(r"unknown object '(?P<name>[^']+)'")


def card_findings(src: Source, rigid_dt: float = 1.25e-3) -> list[Finding]:
    """Findings for every card issue of ``src`` (errors first); a validator crash becomes one error finding.

    Example::

        >>> found = card_findings(load_source("demo_peg_d12"))
        >>> all(f.check == "cards" for f in found)
        True
    """
    from alloy.cards.validate import validate_scene  # noqa: PLC0415

    try:
        issues = validate_scene(src.card, src.lib, rigid_dt, name=src.label)
    except CardError as exc:
        return [card_error_finding(exc)]
    defaults = [i for i in issues if _DEFAULT_LEAF in i.message]
    out = [issue_finding(i, src) for i in issues if _DEFAULT_LEAF not in i.message] + _default_leaves(defaults)
    log.info("check cards %s: %d errors, %d warnings", src.label, sum(f.level == "error" for f in out),
             sum(f.level == "warn" for f in out))
    return sorted(out, key=lambda f: f.level != "error")


def card_error_finding(exc: CardError) -> Finding:
    """A ``CardError`` raised while reading a card (bad YAML, unknown key, wrong unit) as an error finding.

    Example::

        >>> from alloy.core.errors import CardError
        >>> card_error_finding(CardError("mug", "mass", "must be > 0")).field
        'mug.mass'
    """
    card, fld, msg = str(exc.card), str(exc.field), str(exc.message)
    where = f"{card}.{fld}" if fld and fld != "<file>" else card
    return Finding("cards", "error", f"{where}: {msg}", f"edit {where} in the card file: {_fix_text(msg, where)}",
                   field=f"{card}.{fld}" if fld else card)


def issue_finding(issue: ct.CardIssue, src: Source) -> Finding:
    """One ``CardIssue`` as a finding; ``field`` is ``<card>.<dotted path>``.

    Example::

        >>> issue_finding(issue, src).check  # doctest: +SKIP
        'cards'
    """
    where = f"{issue.card}.{issue.field}" if issue.field else issue.card
    m = _UNKNOWN_OBJ.search(issue.message)
    if m is not None:
        return _unknown_object(issue, src, m.group("name"), where)
    obj = (issue.card,) if issue.card != src.label and issue.card != "scene" else ()
    return Finding("cards", issue.level, f"{where}: {issue.message}", _fix_text(issue.message, where),
                   objects=obj, field=where)


def _default_leaves(issues: list[ct.CardIssue]) -> list[Finding]:
    """One info finding per card for leaves still at their default that change contact or routing (common in
    library cards; worth sourcing, not a defect of the scene)."""
    by_card: dict[str, list[str]] = {}
    for i in issues:
        by_card.setdefault(i.card, []).append(i.field)
    return [Finding("cards", "info", f"{card}: {len(fields)} field(s) still at the engine default that change contact "
                    f"or routing: {', '.join(fields)}", f"give them sourced values (value, source, note) in the "
                    f"{card} card when the results depend on them", objects=(card,), field=f"{card}.{fields[0]}")
            for card, fields in by_card.items()]


def _unknown_object(issue: ct.CardIssue, src: Source, name: str, where: str) -> Finding:
    close = difflib.get_close_matches(name, src.lib.names("objects"), n=3, cutoff=0.6)
    fix = (f"use one of {close} in {where}" if close else
           f"add an object card objects/{name}.yaml to a library root, or name an existing object in {where}")
    return Finding("cards", "error", f"{where}: unknown object {name!r}", fix, objects=(name,), field=where,
                   suggestion={"field": where, "replace_with": close[0]} if close else None)


def _fix_text(message: str, where: str) -> str:
    """The edit for a validator message: a range to land in, a note to add, a default to replace."""
    m = _RANGE.match(message)
    if m is not None:
        return f"set {where} inside {m.group('rng')}{m.group('u')} (it is {m.group('v')}{m.group('u')})"
    if _DEFAULT_LEAF in message:
        return (f"give {where} a sourced value (value, source, note: datasheet or measurement); it is still the "
                f"default and it changes contact or routing")
    if "needs a note" in message:
        return f"add a note to {where} citing where the value comes from"
    if "unknown key" in message or "unknown field" in message:
        return f"remove or rename {where} (check the spelling against docs/SPEC.md §3.2)"
    if "non-finite" in message:
        return f"set {where} to a finite number"
    if "mu_dynamic" in message and "> mu_static" in message:
        return f"make mu_dynamic <= mu_static at {where.rsplit('.', 1)[0]}"
    if "watertight" in message:
        return f"repair the mesh named at {where} (close holes, fix normals) or use a primitive or convex shape"
    return f"edit {where}: {message}"


__all__ = ["card_error_finding", "card_findings", "issue_finding"]
