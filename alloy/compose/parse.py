# SPDX-License-Identifier: Apache-2.0
"""A sentence -> ``SceneSpec``: which objects, how many, on or in what, how cluttered.

Rule-based and deterministic (no model call), so the same sentence always gives the same spec. ``notes`` on the
spec say what was understood and what was ignored; an agent that wants exact control writes the spec as JSON
instead (``SceneSpec.from_dict``).

Example::

    >>> spec = parse("a cluttered table with a mug, two spoons and three nuts in a tray")
    >>> [(i.query, i.count, i.relation, i.target) for i in spec.items]
    [('mug', 1, None, None), ('spoon', 2, None, None), ('tray', 1, None, None), ('nut', 3, 'in', 'tray')]
    >>> spec.support, spec.clutter
    ('table', 'cluttered')
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from alloy.compose import words as W
from alloy.compose.soft import parse as S
from alloy.compose.soft.vocab import FORM_WORDS
from alloy.compose.spec import ComposeError, ItemSpec, RobotSpec, SceneSpec

_SPLIT = re.compile(r"(,|;|\band\b|\bwith\b|\bplus\b|\bas well as\b|\balong with\b)")
_TOKEN = re.compile(r"[a-z]+|\d+(?:\.\d+)?")
_WORLDS = re.compile(r"\b(?:in\s+)?(\d+)\s+(?:parallel\s+)?(?:worlds?|copies|envs?|environments?)\b")
_SUPPORT_WORDS = "|".join(sorted(W.SUPPORTS, key=len, reverse=True))
_SUPPORT_TAIL = re.compile(rf"\b(?:on|onto|across|over|atop)\s+(?:a|an|the)?\s*((?:[a-z]+\s+){{0,2}}?)"
                           rf"({_SUPPORT_WORDS})\b")
_SUPPORT_HEAD = re.compile(rf"^\s*(?:a|an|the)?\s*((?:[a-z]+\s+){{0,2}}?)({_SUPPORT_WORDS})\b(?=\s+(?:with|and|,))")
_JITTER = re.compile(r"\b(?:randomi[sz]ed|jittered|jitter|shuffled)\b")
_ROW = re.compile(r"\b(?:in\s+a\s+(row|line|grid)|lined\s+up|in\s+rows)\b")
_PLURAL_DEFAULT = 3                      # "cubes" with no number
_PRONOUNS = frozenset({"it", "them", "this", "that"})


@dataclass
class _Draft:
    support: str = "table"
    clutter: str = "normal"
    arrangement: str = "scatter"
    worlds: int = 1
    jitter: bool = False
    notes: list[str] = field(default_factory=list)
    items: list[ItemSpec] = field(default_factory=list)
    robots: list[str] = field(default_factory=list)       # robot files ("robot arm" -> the library arm)
    pending_fill: float | None = None                      # "a cup filled" waiting for "with water"
    edge: bool = False                                     # "over the edge of the table"


def parse(sentence: str) -> SceneSpec:
    """Read ``sentence`` into a ``SceneSpec`` (see the module docstring); ``ComposeError`` if it names no object.

    Example: ``parse("three wooden cubes on a table").items[0].words == ("wooden",)``."""
    if not isinstance(sentence, str) or not sentence.strip():
        raise ComposeError("the sentence is empty", "describe the scene, e.g. 'a mug and two cubes on a table'",
                           field="sentence")
    d = _Draft()
    text = _globals(sentence.lower(), d)
    text = S.take_robots(text, d)
    text = S.prepare(text, d)
    text = _support(text, d)
    parts = _SPLIT.split(text)
    for k in range(0, len(parts), 2):
        _chunk(parts[k], d, after_with=k > 0 and parts[k - 1].strip() == "with")
    if not d.items:
        raise ComposeError(f"no object found in {sentence!r}", "name objects with nouns, e.g. 'a mug, three cubes "
                           "and a tray on a table' (alloy.compose.search('mug') lists matches)", field="sentence")
    robots = tuple(RobotSpec(path, name="arm" if k == 0 else f"arm{k + 1}") for k, path in enumerate(d.robots))
    return SceneSpec(items=tuple(_order(S.normalise(d.items))), support=d.support, clutter=d.clutter,
                     arrangement=d.arrangement, worlds=d.worlds, jitter=d.jitter, sentence=sentence.strip(),
                     notes=tuple(d.notes), robots=robots)


def _globals(text: str, d: _Draft) -> str:
    """Jitter, world count, arrangement and clutter: recorded on ``d`` and removed from the text."""
    m = _JITTER.search(text)
    if m:
        d.jitter = True
        d.notes.append(f"jitter: every world moves the free objects a little (from {m.group(0)!r})")
        text = text[:m.start()] + " " + text[m.end():]
    m = _WORLDS.search(text)
    if m:
        d.worlds = max(1, int(m.group(1)))
        d.notes.append(f"worlds: {d.worlds} (from {m.group(0).strip()!r})")
        text = text[:m.start()] + " " + text[m.end():]
    m = _ROW.search(text)
    if m:
        d.arrangement = "grid" if (m.group(1) == "grid" or "rows" in m.group(0)) else "row"
        d.notes.append(f"arrangement: {d.arrangement} (from {m.group(0)!r})")
        text = text[:m.start()] + " " + text[m.end():]
    for w in W.TOKEN.findall(text):
        if w in W.CLUTTER:
            d.clutter = W.CLUTTER[w]
            d.notes.append(f"clutter: {d.clutter} (from {w!r})")
            text = re.sub(rf"\b{w}\b", " ", text)
            break
    return text


def _support(text: str, d: _Draft) -> str:
    """The surface everything stands on ("on a table", "a desk with ..."); default the table."""
    for pattern in (_SUPPORT_TAIL, _SUPPORT_HEAD):
        m = pattern.search(text)
        if m:
            d.support = W.SUPPORTS[m.group(2)]
            adjectives = [w for w in m.group(1).split() if w not in W.STOP and w not in W.NUMBERS]
            note = f"support: {d.support} (from {' '.join(m.group(0).split())!r})"
            d.notes.append(note + (f"; {', '.join(adjectives)} not used (one {d.support} kind)" if adjectives else ""))
            return text[:m.start()] + " , " + text[m.end():]
    d.notes.append("support: table (default; say 'on the floor' for the floor)")
    return text


def _chunk(chunk: str, d: _Draft, after_with: bool = False) -> None:
    """One item phrase: "three wooden cubes", "a nut in the tray" (soft phrases: ``alloy.compose.soft.parse``)."""
    raw = _TOKEN.findall(chunk)
    if not raw or S.soft_chunk(raw, d, after_with):
        return
    rel_at = next((i for i, w in enumerate(raw) if w in W.RELATIONS and i > 0), None)
    head, tail = (raw, []) if rel_at is None else (raw[:rel_at], raw[rel_at + 1:])
    count, plural, head = _count(head)
    noun, adjectives = _noun(head, d)
    if noun is None:
        return
    if plural and count is None:
        count = _PLURAL_DEFAULT
        d.notes.append(f"{noun}: plural with no number, placing {count}")
    relation = target = None
    if rel_at is not None:
        _, _, tail = _count(tail)
        target, target_adj = _noun(tail, d)
        if target is None and d.items and set(tail) & _PRONOUNS:     # "a tray with two nuts on it"
            target, target_adj = d.items[-1].query, list(d.items[-1].words)
        if target is not None:
            relation = W.RELATIONS[raw[rel_at]]
            if not any(i.query == target for i in d.items):
                d.items.append(ItemSpec(target, 1, tuple(target_adj)))
    count = min(count or 1, W.MAX_COUNT)
    d.items.append(ItemSpec(noun, count, tuple(adjectives), relation, target))


def _count(tokens: list[str]) -> tuple[int | None, bool, list[str]]:
    """(count or None, whether the noun looks plural, tokens without the count words)."""
    count = None
    rest = []
    for i, w in enumerate(tokens):
        if count is None and w.isdigit():
            count = int(w)
        elif count is None and w in W.NUMBERS and not (w == "a" and i + 1 < len(tokens) and tokens[i + 1] in
                                                        ("few", "couple", "pair", "dozen")):
            count = W.NUMBERS[w]
        elif w not in ("of",):
            rest.append(w)
    nouns = [w for w in rest if w not in W.STOP]
    plural = bool(nouns) and W.singular(nouns[-1]) != nouns[-1]
    return count, plural, rest


def _noun(tokens: list[str], d: _Draft) -> tuple[str | None, list[str]]:
    """The head noun (last content word, singular) and the words before it."""
    content = [w for w in tokens if w not in W.STOP and w not in W.CLUTTER and w not in W.SUPPORTS and not w.isdigit()
               and w not in W.NUMBERS and w not in FORM_WORDS]
    if not content:
        return None, []
    return W.singular(content[-1]), [W.singular(w) if w not in W.MATERIAL_WORDS else w for w in content[:-1]]


def _order(items: list[ItemSpec]) -> list[ItemSpec]:
    """Targets before the items placed on or in them (stable otherwise)."""
    out: list[ItemSpec] = []
    pending = list(items)
    while pending:
        ready = [i for i in pending if i.target is None or any(o.query == i.target for o in out)]
        if not ready:                      # a cycle ("a on b, b on a"): keep the rest as given
            out.extend(pending)
            break
        out.extend(ready)
        pending = [i for i in pending if all(i is not r for r in ready)]
    return out


__all__ = ["parse"]
