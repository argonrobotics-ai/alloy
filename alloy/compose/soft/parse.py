# SPDX-License-Identifier: Apache-2.0
"""The soft half of the sentence parser: liquids, grains, cloth, cables and pours in a phrase -> ``ItemSpec``.

Called by ``alloy.compose.parse`` for each phrase before the rigid rules. It reads amounts ("200 ml", "0.3 kg",
"half full"), lengths ("a 2 m cable", "a 50 x 30 cm towel", "a 3 mm cable" = the gauge), containers ("a bowl of
rice", "a cup with water"), forms ("draped over", "folded in half", "hanging over the edge", "a pile of"), cable
ends ("plugged into a laptop", "from the laptop to the monitor") and pours ("pour water from a bottle into a cup").

Example::

    >>> from alloy.compose.parse import parse
    >>> items = parse("a cup with 200 ml of water").items
    >>> [(i.query, i.relation, i.target, i.soft and i.soft.to_dict()) for i in items]
    [('cup', None, None, None), ('water', 'in', 'cup', {'kind': 'liquid', 'form': 'fill', 'volume_ml': 200.0})]
"""

from __future__ import annotations

import re
from typing import Any

from alloy.compose import words as W
from alloy.compose.errors import ComposeError
from alloy.compose.soft import vocab as V
from alloy.compose.soft.spec import SoftSpec
from alloy.compose.spec import ItemSpec

_NUM = re.compile(r"^\d+(?:\.\d+)?$")
_ROBOT = re.compile(r"\b(?:(a|an|the|one|two)\s+)?(?:(?:robot(?:ic)?|panda|franka)\s+)?(?:arms?|manipulators?)\b"
                    r"|\b(?:(a|an|the|one|two)\s+)?(?:robots?|panda|franka)\b")
_HALF = re.compile(r"\bin\s+(?:half|two|halves)\b")
_EDGE = re.compile(r"\b(?:over|off|on|from)\s+(?:the\s+)?(?:edge|side)\s+of\s+(?:the|a)\s+")
_ROLES = ("from", "to", "into", "in", "onto", "on", "over", "inside")
_LINK = frozenset({"to", "into", "in", "onto", "on", "a", "an", "the", "its", "of", "with", "at"})


def take_robots(text: str, d: Any) -> str:
    """"a robot arm", "two arms", "a Panda": recorded as library robots on ``d`` and removed from the text."""
    found = 0
    for m in _ROBOT.finditer(text):
        n = W.NUMBERS.get(m.group(1) or m.group(2) or "a", 1)
        found += 2 if (n == 1 and m.group(0).rstrip().endswith("s") and not (m.group(1) or m.group(2))) else n
    if not found:
        return text
    found = min(found, 2)
    d.robots.extend([V.DEFAULT_ROBOT] * found)
    d.notes.append(f"robots: {found} x the library Franka Panda arm ({V.DEFAULT_ROBOT}); pass robots=[...] for "
                   "your own")
    text = _ROBOT.sub(" ", text)
    return text


def prepare(text: str, d: Any) -> str:
    """Sentence-level rewrites before the phrase split: "in half" -> folded, "over the edge of the" -> hanging."""
    text = _HALF.sub(" folded ", text)
    if _EDGE.search(text):
        d.edge = True
        text = _EDGE.sub(" hanging over the ", text)
    return text


def soft_chunk(raw: list[str], d: Any, after_with: bool) -> bool:
    """Add the soft item(s) of one phrase to ``d.items``; False when the phrase names no soft noun (rigid path)."""
    tokens = list(raw)
    amounts, tokens = take_amounts(tokens)
    fill, tokens = take_fill(tokens)
    fill = fill if fill is not None else d.pending_fill
    tilt = any(t in V.TILT_WORDS for t in tokens) and ("into" in tokens or "from" in tokens)
    if tilt or any(t in V.POUR_WORDS for t in tokens):
        return _pour(tokens, amounts, d)
    noun_at = _soft_noun(tokens)
    if noun_at is None:
        unknown = unknown_material(raw)
        if unknown is not None:
            raise ComposeError(f"{unknown!r} is not a liquid or grain the scene builder has", _MATERIALS_FIX,
                               field="sentence")
        if fill is not None:
            d.pending_fill = fill                   # "a cup filled" + "with water"
        return False
    kind = V.kind_of(W.singular(tokens[noun_at]))
    if kind in ("liquid", "grain"):
        _fill_or_pile(tokens, noun_at, kind, {**amounts, "fill": fill}, d, after_with)
    elif kind == "cloth":
        _cloth(tokens, noun_at, amounts, d)
    else:
        _cable(tokens, noun_at, amounts, d)
    d.pending_fill = None
    return True


#: what to ask for instead of a liquid or grain the builder does not have (the words it reads as one)
_MATERIALS_FIX = (f"ask for one of the liquids ({', '.join(sorted(V.LIQUIDS))}; milk, juice, coffee, tea, wine and "
                  f"beer are made as water) or grains ({', '.join(sorted(V.GRAINS))})")


def unknown_material(tokens: list[str]) -> str | None:
    """The noun of "<number> <volume or mass unit> of <noun>" ("200 ml of detergent") when no other soft noun is in
    the phrase: an amount of a material the builder does not have, not a count of objects. None otherwise.

    Example: ``unknown_material("a cup with 200 ml of detergent".split()) == "detergent"``."""
    for i in range(len(tokens)):
        dims, j = _dims(tokens, i)
        if dims and j < len(tokens) and tokens[j] == "of" and V.unit_kind(tokens[j - 1]) in ("volume", "mass"):
            noun, _adj, _n = _phrase([t for t in tokens[j + 1:] if t not in _ROLES])
            return noun
    return None


def _pour_noun(tokens: list[str]) -> str | None:
    """The word a pour names where its liquid should be ("pour detergent from a bottle": "detergent"), or None."""
    at = next((i for i, t in enumerate(tokens) if t in V.POUR_WORDS), None)
    if at is None:
        return None
    tail = tokens[at + 1:]
    stop = next((i for i, t in enumerate(tail) if t in ("from", "out", "into", "in", "inside", "onto", "over")),
                len(tail))
    noun, _adj, _n = _phrase(tail[:stop])
    return noun


# --- amounts -----------------------------------------------------------------------------------------------------


def take_amounts(tokens: list[str]) -> tuple[dict[str, float], list[str]]:
    """Volume (ml), mass (g) and lengths (m) written as "<number> <unit>" (also "half a litre", "50 x 30 cm"),
    removed from the tokens with a following "of"."""
    out: dict[str, float] = {}
    rest: list[str] = []
    i = 0
    while i < len(tokens):
        dims, j = _dims(tokens, i)
        if dims:
            _record(out, dims, tokens[j - 1])
            i = j + (1 if j < len(tokens) and tokens[j] == "of" else 0)
            continue
        rest.append(tokens[i])
        i += 1
    return out, rest


def _number(tokens: list[str], i: int) -> tuple[float | None, int]:
    """A number at ``i`` ("200", "0.5", "two", "a", "half a") and the index after it."""
    t = tokens[i]
    if _NUM.match(t):
        return float(t), i + 1
    if t == "half":
        return 0.5, i + (2 if i + 1 < len(tokens) and tokens[i + 1] in ("a", "an") else 1)
    if t in W.NUMBERS and t not in ("some", "many", "lots", "few", "several"):
        return float(W.NUMBERS[t]), i + 1
    return None, i


def _dims(tokens: list[str], i: int) -> tuple[list[float], int]:
    """(values, index after the unit) for "<n> unit" or "<n> x <n> unit"; ([], i) when there is none."""
    a, j = _number(tokens, i)
    if a is None:
        return [], i
    vals = [a]
    while j + 1 < len(tokens) and tokens[j] in ("x", "by") and _number(tokens, j + 1)[0] is not None:
        b, j = _number(tokens, j + 1)
        vals.append(b)
    if j < len(tokens) and V.unit_kind(tokens[j]) is not None and not (len(vals) == 1 and tokens[i] in ("a", "an")
                                                                       and V.unit_kind(tokens[j]) == "length"):
        return vals, j + 1
    return [], i


def _record(out: dict[str, float], vals: list[float], unit: str) -> None:
    kind = V.unit_kind(unit)
    if kind == "volume":
        out["volume_ml"] = V.to_ml(vals[0], unit)
    elif kind == "mass":
        out["mass_g"] = V.to_g(vals[0], unit)
    else:
        m = sorted((V.to_m(v, unit) for v in vals), reverse=True)
        out["length_m"] = m[0]
        if len(m) > 1:
            out["width_m"] = m[1]


def take_tilt(tokens: list[str]) -> tuple[float | None, float | None, list[str]]:
    """(rate deg/s, final angle deg, the rest): "40 deg/s", "40 degrees per second", "to 120 degrees"."""
    rate = top = None
    rest: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if _NUM.match(t) and i + 1 < len(tokens) and tokens[i + 1] in V.DEGREES:
            j = i + 2
            per = j < len(tokens) and tokens[j] in ("s", "sec", "second", "per", "a")
            while j < len(tokens) and tokens[j] in ("s", "sec", "second", "per", "a"):
                j += 1
            if per:
                rate = float(t)
            else:
                top = float(t)
                if rest and rest[-1] in ("to", "until", "up"):
                    rest.pop()
            i = j
            continue
        rest.append(t)
        i += 1
    return rate, top, rest


def take_fill(tokens: list[str]) -> tuple[float | None, list[str]]:
    """A fill word ("half full", "full", "a little"): its share of the container depth, removed from the tokens."""
    text = " " + " ".join(tokens) + " "
    for phrase in sorted(V.FILL_WORDS, key=len, reverse=True):
        if f" {phrase} " in text:
            return V.FILL_WORDS[phrase], text.replace(f" {phrase} ", " ", 1).split()
    return None, tokens


# --- one soft phrase ----------------------------------------------------------------------------------------------


def _soft_noun(tokens: list[str]) -> int | None:
    """Index of the soft noun: the first liquid / grain, else the last cloth / cable noun before a role word; one
    used as an adjective of an object noun ("water bottle", "paper bag", "wire basket") does not count."""
    for i, t in enumerate(tokens):
        if V.kind_of(W.singular(t)) in ("liquid", "grain") and not _part_of_name(tokens, i):
            return i
    head = tokens[:next((i for i, t in enumerate(tokens) if t in _ROLES or t in V.ATTACH_WORDS), len(tokens))]
    for i in range(len(head) - 1, -1, -1):
        if V.kind_of(W.singular(head[i])) in ("cloth", "cable"):
            return None if _part_of_name(head, i) and head[i + 1] not in W.SIZE_WORDS else i
    return None


def _part_of_name(tokens: list[str], i: int) -> bool:
    """"water bottle", "oil can", "sand paper": a liquid / grain word used as an adjective of an object noun."""
    nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
    return bool(nxt) and nxt not in _LINK and nxt not in _ROLES and nxt not in W.STOP and V.kind_of(
        W.singular(nxt)) is None and nxt not in V.FORM_WORDS and nxt not in W.RELATIONS


def _phrase(tokens: list[str]) -> tuple[str | None, list[str], int]:
    """(noun, adjectives, count) of an object phrase like "two ceramic cups"; noun None when there is none."""
    count = 1
    words: list[str] = []
    for t in tokens:
        if t in W.NUMBERS and not words:
            count = W.NUMBERS[t]
        elif _NUM.match(t) and not words:
            count = int(float(t))
        elif t not in W.STOP and t not in W.NUMBERS and t not in V.FORM_WORDS and t not in _LINK:
            words.append(t)
    if not words:
        return None, [], count
    return W.singular(words[-1]), [w if w in W.MATERIAL_WORDS else W.singular(w) for w in words[:-1]], count


def _ensure(d: Any, noun: str, adjectives: list[str], count: int = 1) -> str:
    """The query of an item named ``noun``, added (before the soft item) when it is new."""
    if not any(i.query == noun for i in d.items):
        d.items.append(ItemSpec(noun, min(max(count, 1), W.MAX_COUNT), tuple(adjectives)))
    return noun


def _after(tokens: list[str], words: frozenset[str] | tuple[str, ...]) -> tuple[int | None, list[str]]:
    """Index of the first token in ``words`` and the object phrase after it (up to the next role word)."""
    at = next((i for i, t in enumerate(tokens) if t in words), None)
    if at is None:
        return None, []
    tail = tokens[at + 1:]
    stop = next((i for i, t in enumerate(tail) if t in _ROLES and i > 0), len(tail))
    return at, tail[:stop]


def _fill_or_pile(tokens: list[str], at: int, kind: str, amounts: dict[str, Any], d: Any, after_with: bool) -> None:
    noun = W.singular(tokens[at])
    adjectives = [t for t in tokens[:at] if t not in W.STOP and t not in W.NUMBERS and t not in V.FORM_WORDS
                  and t not in _LINK and not _NUM.match(t)]
    pile = any(t in V.PILE_WORDS for t in tokens)
    before = [t for t in tokens[:at] if t not in V.PILE_WORDS]
    container, cadj, ccount = _phrase(before[:-1] if before and before[-1] == "of" else [])
    if container is not None and container not in V.PILE_WORDS and "of" in tokens[:at]:
        target, relation = _ensure(d, container, cadj, ccount), "in"         # "a bowl of rice"
        adjectives = []
    else:
        relation, target = _role_target(tokens[at + 1:], d)
        if target is None and after_with and d.items and d.items[-1].soft is None:
            relation, target = "in", d.items[-1].query                          # "a cup with water"
    form = "pile" if kind == "grain" and (pile or relation != "in") else "fill"
    if kind == "liquid" and relation != "in":
        raise ComposeError(f"the {noun} has no container", f"say what holds it, e.g. 'a cup with 200 ml of {noun}' or "
                           f"'pour {noun} into a cup'", field="sentence")
    soft = SoftSpec(kind, form, volume_ml=amounts.get("volume_ml"), mass_g=amounts.get("mass_g"),
                    fill=amounts.get("fill") if form == "fill" and not (amounts.get("volume_ml") or
                                                                      amounts.get("mass_g")) else None)
    d.items.append(ItemSpec(noun, 1, tuple(adjectives), relation if target else None, target, soft=soft))
    _note_amount(d, noun, soft)


def _role_target(tail: list[str], d: Any) -> tuple[str | None, str | None]:
    """("in"/"on", target query) of "in a bowl" / "on a plate" after a soft noun (the target is added if new)."""
    at = next((i for i, t in enumerate(tail) if t in W.RELATIONS), None)
    if at is None:
        return None, None
    noun, adjectives, count = _phrase(tail[at + 1:])
    if noun is None:
        return None, None
    return W.RELATIONS[tail[at]] if tail[at] != "over" else "on", _ensure(d, noun, adjectives, count)


def _pour(tokens: list[str], amounts: dict[str, float], d: Any) -> bool:
    at = next((i for i, t in enumerate(tokens) if V.kind_of(W.singular(t)) == "liquid"), None)
    if at is None:
        noun = _pour_noun(tokens)
        if noun is not None:
            raise ComposeError(f"{noun!r} is not a liquid the scene builder has", f"pour one of {', '.join(sorted(
                V.LIQUIDS))}, e.g. 'pour water from a bottle into a cup' (milk, juice, coffee, tea, wine and beer are "
                "made as water)", field="sentence")
        raise ComposeError("a pour names no liquid", "say what pours, e.g. 'pour water from a bottle into a cup'",
                           field="sentence")
    noun = W.singular(tokens[at])
    _, src_words = _after(tokens, ("from", "out"))
    _, dst_words = _after(tokens, ("into", "in", "inside"))
    rate, top, tokens = take_tilt(tokens)
    if rate is None and top is None and any(t in V.TILT_WORDS for t in tokens):
        rate = V.DEFAULT_TILT_DEG_S
    _, src_words = _after(tokens, ("from", "out"))
    _, dst_words = _after(tokens, ("into", "in", "inside"))
    source, sadj, _ = _phrase([w for w in src_words if w not in V.TILT_WORDS])
    target, tadj, _ = _phrase(dst_words)
    if target is None:
        raise ComposeError(f"the {noun} pours into nothing", f"name the container it pours into, e.g. 'pour {noun} "
                           "into a cup'", field="sentence")
    src = _ensure(d, source, sadj) if source else None
    dst = _ensure(d, target, tadj)
    if (rate or top) and src is None:
        raise ComposeError(f"the {noun} tilts out of nothing", f"name the vessel it pours from, e.g. 'pour {noun} "
                           "from a jar tilted at 40 deg/s into a mug'", field="sentence")
    soft = SoftSpec("liquid", "pour", volume_ml=amounts.get("volume_ml"), mass_g=amounts.get("mass_g"), source=src,
                    tilt_deg_s=rate, tilt_to_deg=top)
    d.items.append(ItemSpec(noun, 1, (), "in", dst, soft=soft))
    _note_amount(d, noun, soft)
    return True


def _cloth(tokens: list[str], at: int, amounts: dict[str, float], d: Any) -> None:
    noun = W.singular(tokens[at])
    adjectives = [t for t in tokens[:at] if t not in W.STOP and t not in W.NUMBERS and t not in V.FORM_WORDS
                  and not _NUM.match(t) and t not in _LINK]
    form = next((V.CLOTH_FORMS[t] for t in tokens if t in V.CLOTH_FORMS), None)
    relation = target = None
    over_at, over = _after(tokens[at + 1:], ("over", "on", "onto", "across", "atop", "covering"))
    if over_at is not None:
        tnoun, tadj, tcount = _phrase(over)
        if tnoun is not None:
            target = _ensure(d, tnoun, tadj, tcount)
            draped = form in ("draped", "hanging", "folded") or tokens[at + 1 + over_at] in ("over", "across",
                                                                                              "covering")
            relation, form = ("over", "draped") if draped else ("on", form or "flat")
    if form is None:
        form = "hanging" if d.edge else "flat"
    count = _count_of(tokens[:at], tokens[at], d)
    soft = SoftSpec("cloth", form, length_m=amounts.get("length_m"), width_m=amounts.get("width_m"))
    d.items.append(ItemSpec(noun, count, tuple(adjectives), relation, target, soft=soft))


def _cable(tokens: list[str], at: int, amounts: dict[str, float], d: Any) -> None:
    noun = W.singular(tokens[at])
    adjectives = [t for t in tokens[:at] if t not in W.STOP and t not in W.NUMBERS and t not in V.FORM_WORDS
                  and not _NUM.match(t) and t not in _LINK]
    form = next((V.CABLE_FORMS[t] for t in tokens if t in V.CABLE_FORMS), "loose")
    length, gauge = amounts.get("length_m"), None
    if length is not None and length < 0.02:                                    # "a 3 mm cable": the gauge
        length, gauge = None, length * 1e3
    ends = _cable_ends(tokens[at + 1:], d)
    count = _count_of(tokens[:at], tokens[at], d)
    soft = SoftSpec("cable", form, length_m=length, diameter_mm=gauge, attach=tuple(ends))
    d.items.append(ItemSpec(noun, count, tuple(adjectives), soft=soft))


def _cable_ends(tail: list[str], d: Any) -> list[str]:
    """Items a cable's ends are fixed to: "plugged into a laptop", "from the laptop to the monitor"."""
    ends: list[str] = []
    if "from" in tail:
        at, phrase = _after(tail, ("from",))
        noun, adj, _ = _phrase(phrase)
        if noun is not None:
            ends.append(_ensure(d, noun, adj))
        tail = tail[at + 1 + len(phrase):]
    attach_at = next((i for i, t in enumerate(tail) if t in V.ATTACH_WORDS or (t == "to" and ends)), None)
    if attach_at is not None:
        rest = [t for t in tail[attach_at + 1:] if t not in ("into", "to", "onto", "in", "on", "at")]
        noun, adj, _ = _phrase(rest)
        if noun is not None:
            ends.append(_ensure(d, noun, adj))
    return ends[:2]


def _count_of(tokens: list[str], noun: str, d: Any) -> int:
    """How many of a soft item: a number word or numeral before its ``noun`` ("two towels", "a few cables"), else
    ``W.PLURAL_COUNT`` for a bare plural ("cables", as "cubes" for objects; noted in ``d.notes``), else 1."""
    said = [W.NUMBERS[t] for i, t in enumerate(tokens) if t in W.NUMBERS
            and not (t in ("a", "an") and tokens[i + 1:i + 2] and tokens[i + 1] in W.NUMBERS)]      # "a few": few
    n = said[0] if said else next((int(float(t)) for t in tokens if _NUM.match(t)), None)
    if n is None and W.singular(noun) != noun:
        n = W.PLURAL_COUNT
        d.notes.append(f"{W.singular(noun)}: plural with no number, placing {n}")
    return min(max(int(n or 1), 1), W.MAX_COUNT)


def normalise(items: list[ItemSpec]) -> list[ItemSpec]:
    """Items named by a soft noun but written without ``soft`` (a target such as the towel in "a mug on a towel", or
    a hand-written spec) get the default soft form: a flat cloth, a loose cable, a grain pile. A liquid needs its
    container, so one without ``soft`` is an error that says how to write it.

    Example: ``normalise([ItemSpec("towel")])[0].soft.form == "flat"``."""
    out = []
    for item in items:
        kind = V.kind_of(W.singular(item.query)) if item.soft is None and item.object is None else None
        if kind == "liquid":
            raise ComposeError(f"item {item.query!r} is a liquid with no container", f"give it relation 'in' a "
                               f"container and soft={{'kind': 'liquid'}}, e.g. 'a cup with 200 ml of {item.query}'",
                               field="items.soft")
        if kind is not None and item.relation in (None, "on"):
            item = ItemSpec(item.query, item.count, item.words, item.relation, item.target, item.object,
                            SoftSpec(kind, V.DEFAULT_FORMS[kind]))
        out.append(item)
    return out


def _note_amount(d: Any, noun: str, soft: SoftSpec) -> None:
    if soft.volume_ml is None and soft.mass_g is None and soft.fill is None:
        what = (f"fills {int(V.DEFAULT_FILL * 100)} % of its container" if soft.form == "fill" else
                f"{V.DEFAULT_PILE_ML:.0f} ml" if soft.form == "pile" else "100 ml")
        d.notes.append(f"{noun}: no amount given, using {what} (say '200 ml of {noun}' or '300 g of {noun}')")


__all__ = ["normalise", "prepare", "soft_chunk", "take_amounts", "take_fill", "take_robots"]
