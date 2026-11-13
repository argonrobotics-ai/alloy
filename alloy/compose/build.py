# SPDX-License-Identifier: Apache-2.0
"""``compose(sentence)``: a sentence (or a spec) in, a physics-checked scene and N variations out.

Steps per scene: choose a card per item (catalog search), size the work area, put the objects down one at a time
(largest first, containers before their contents) with ``alloy.check``'s load checks after each, then run the full
check and repair what it flags. Variations use new seeds: new poses and, with ``swap=True``, other equally good
objects; only variations that pass are kept (up to ``2 * variations + 1`` tries).

Example::

    >>> r = compose("a mug, a spoon and three wooden cubes on a table", variations=2, device="cpu")  # doctest: +SKIP
    >>> print(r.summary())                                                                          # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alloy.compose.body import Body
from alloy.compose.catalog import SHELL_ROUTES, Catalog
from alloy.compose.choose import choose, choose_one, soft_queries
from alloy.compose.critic import BROKEN, MODES, OK, RETRY, Critic
from alloy.compose.fixtures import Support, support
from alloy.compose.layout import CLEARANCE, Layout, Placed
from alloy.compose.parse import parse
from alloy.compose.result import ComposedScene, ComposeResult
from alloy.compose.robots import place_robots, reach_area
from alloy.compose.soft.parse import normalise
from alloy.compose.soft.plan import SoftPlanner
from alloy.compose.spec import ComposeError, RobotSpec, SceneSpec

log = logging.getLogger("alloy.compose")

ENGINE_TRIES = 3            # positions tried per object when the engine's load check rejects one
JITTER_SHARE = 0.45         # per-world x/y spread as a share of the clutter gap (< 1/2: no new overlaps)
MAX_VARIATIONS = 100
#: words that ask for a soft body ("a soft apple"); without one a soft solid (fruit, foam) is placed rigid
SOFT_WORDS = frozenset({"soft", "squishy", "squashy", "deformable", "squeezable", "spongy", "compressible"})
SOFT_COST = ("a soft body steps far slower than a rigid one (measured on a CPU: about 90 s per 20 ms control step "
             "with three soft apples, about 0.1 s with them rigid)")


@dataclass(frozen=True)
class _Instance:
    item: int
    query: str
    card: str
    body: Body
    target: str | None             # query of the item it goes on or in
    mode: str = "rigid"            # "rigid", or a soft item: "layout" (sheet, pile), "drape", "hang"


def compose(request: str | SceneSpec | dict, *, variations: int = 0, seed: int = 0, check: str = "full",
            device: str | None = None, worlds: int | None = None, seconds: float | None = None,
            roots: Sequence[str | Path] = (), catalog: Catalog | None = None, swap: bool = True,
            robots: Sequence[RobotSpec | dict | str] = (), area: Sequence[float] | None = None,
            jitter: bool | None = None, scene_worlds: int | None = None) -> ComposeResult:
    """Build a checked scene from ``request`` (a sentence, a ``SceneSpec`` or its dict) plus ``variations`` more.

    ``check``: "full" (load checks per placement, full check with repairs at the end), "load" or "none";
    ``device``/``worlds``/``seconds``: the full check's settings (``alloy.check`` defaults); ``roots``: your own card
    folders, searched first; ``swap``: variations may use other equally good objects; ``robots``: robot files to
    stand at the table (``RobotSpec``, its dict, or a path); ``area``: (x0, y0, x1, y1) where objects may go;
    ``jitter`` / ``scene_worlds``: override the spec's per-world jitter and the saved card's world count.
    Example: ``compose("three nuts in a tray on a table", variations=4, robots=["arm.xml"]).save("out")``."""
    t0 = time.perf_counter()
    spec = _with(_spec(request), robots, area, jitter, scene_worlds)
    if check not in MODES:
        raise ComposeError(f"check={check!r} is unknown", f"use one of {MODES}", field="check")
    if not isinstance(variations, int) or not 0 <= variations <= MAX_VARIATIONS:
        raise ComposeError(f"variations={variations!r}", f"pass an int from 0 to {MAX_VARIATIONS}",
                           field="variations")
    cat = catalog if catalog is not None else Catalog(roots)
    sup = support(spec.support, cat)
    work = Path(tempfile.mkdtemp(prefix="alloy-compose-"))
    files: dict[str, bytes] = {}
    try:
        spec = _robot_copies(spec, cat, files, work)              # raises on a robot file that is missing or broken
        critic = Critic(_library(cat, sup, work), device, check, worlds, seconds, spec.worlds,
                        jitter=JITTER_SHARE * CLEARANCE[spec.clutter] if spec.jitter else 0.0, work=work)
        scenes = _scenes(spec, cat, sup, critic, seed, variations, swap)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    generated = dict(sup.cards)
    for s in scenes:
        generated.update(s.cards)
        files.update(s.files)
    sources = sorted({cat.entries[o["card"]].source for s in scenes for o in s.objects
                      if o["card"] in cat.entries} - {"library", "assets"})
    result = ComposeResult(spec, scenes, generated, sources, list(spec.notes), files=files)
    result.timings_s = {"total": round(time.perf_counter() - t0, 2), "checks": critic.checks_run}
    log.info("compose: %s in %.1f s", "OK" if result.ok else "NOT OK", result.timings_s["total"])
    return result


def _spec(request: str | SceneSpec | dict) -> SceneSpec:
    if isinstance(request, SceneSpec):
        return dataclasses.replace(request, items=tuple(normalise(list(request.items))))
    if isinstance(request, dict):
        return _spec(SceneSpec.from_dict(request))
    if isinstance(request, str):
        return parse(request)
    raise ComposeError(f"request of type {type(request).__name__}", "pass a sentence, a SceneSpec or its dict",
                       field="request")


def _with(spec: SceneSpec, robots: Sequence[RobotSpec | dict | str], area: Sequence[float] | None,
          jitter: bool | None = None, worlds: int | None = None) -> SceneSpec:
    """The spec plus robots, a work area, jitter and a world count given as arguments."""
    more = tuple(r if isinstance(r, RobotSpec) else RobotSpec.from_dict(r) for r in robots)
    return dataclasses.replace(spec, robots=spec.robots + more,
                               area=tuple(float(v) for v in area) if area is not None else spec.area,
                               jitter=spec.jitter if jitter is None else bool(jitter),
                               worlds=spec.worlds if worlds is None else worlds)


def _robot_copies(spec: SceneSpec, cat: Catalog, files: dict[str, bytes], work: Path) -> SceneSpec:
    """Robots whose MJCF also holds a floor, a table or objects, swapped for robot-only copies written into
    ``work`` (and kept in ``files`` for the saved folder). Each robot's file is checked first (``robot_file``): a
    missing or unreadable one raises before any object is placed, in every check mode."""
    from alloy.compose.robot_files import robot_copy, robot_file  # noqa: PLC0415

    robots, notes = [], list(spec.notes)
    for r in [robot_file(r, cat.library) for r in spec.robots]:
        r2, note = robot_copy(r, cat.library, files)
        robots.append(r2)
        notes += [note] if note else []
    for rel, data in files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_bytes(data)
    return dataclasses.replace(spec, robots=tuple(robots), notes=tuple(notes))


def _library(cat: Catalog, sup: Support, work: Path):
    """The catalog's library with the work folder (generated cards) in front."""
    from alloy.cards.library import CardLibrary  # noqa: PLC0415

    if sup.cards:
        (work / "objects").mkdir(parents=True, exist_ok=True)
        for stem, text in sup.cards.items():
            (work / "objects" / f"{stem}.yaml").write_text(text, encoding="utf-8")
    return CardLibrary(roots=[work, *cat.library.roots], assets=cat.library.assets)


def _scenes(spec: SceneSpec, cat: Catalog, sup: Support, critic: Critic, seed: int, variations: int,
            swap: bool) -> list[ComposedScene]:
    scenes = [_one(spec, cat, sup, critic, seed, "scene", swap=False)]
    k = 0
    while len(scenes) < variations + 1 and k < 2 * variations + 1:
        k += 1
        s = _one(spec, cat, sup, critic, seed + k, f"scene_v{len(scenes):02d}", swap=swap)
        if s.ok:
            scenes.append(s)
        else:
            log.info("compose: variation seed %d not kept (%s)", seed + k, "; ".join(s.missing) or "check failed")
    return scenes


def _one(spec: SceneSpec, cat: Catalog, sup: Support, critic: Critic, seed: int, name: str, *,
         swap: bool) -> ComposedScene:
    rng = np.random.default_rng(seed)
    choice = choose(spec, cat, rng, swap=swap)
    planner = SoftPlanner(spec, cat, critic.library, critic.work, rng, name)
    soft_items = planner.layout_items()
    queue = _instances(spec, choice.cards, cat) + [_Instance(li.item, li.query, li.card, li.body, li.target, li.mode)
                                                   for li in soft_items]
    queue = _ordered(queue, spec)
    critic.soft = planner.soft
    layout = _layout(spec, sup, critic, rng)
    layout.plan_area([i.body for i in queue if i.target is None and i.mode in ("rigid", "layout")],
                     spec.arrangement, _within(spec, layout, critic))
    critic.log = list(choice.notes)
    _solids(queue, spec, cat, critic)
    skip = soft_queries(spec)
    unmatched = [f"{i.count} x {i.query}" for i in spec.items if i.query not in skip and not choice.cards.get(i.query)]
    missing = list(unmatched)
    first: dict[str, str] = {}
    counts: dict[str, int] = {}
    k = 0
    while k < len(queue):
        inst = queue[k]
        k += 1
        target = first.get(inst.target) if inst.target else None
        if inst.target and target is None:
            critic.log.append(f"{inst.card}: not placed ({inst.target} was not placed)")
            continue
        n = counts[inst.card] = counts.get(inst.card, 0) + 1
        key = inst.card if n == 1 else f"{inst.card}#{n - 1}"
        placed = Placed(key, inst.card, inst.body, 0.0, np.zeros(3), target, inst.item)
        status = _place(layout, critic, placed, _soft_placer(planner, inst, soft_items))
        if status == OK:
            first.setdefault(inst.query, key)
        elif status == BROKEN and inst.mode == "rigid" and _swap_card(queue, k - 1, spec, cat, critic,
                                                                      rng if swap else None):
            k -= 1                                                   # try the same item with the new card
        else:
            critic.log.append(f"{key}: not placed (no free spot passed the checks)")
    planner.finish(layout)
    critic.after_soft(layout)
    report = critic.final(layout, lambda p: _place(layout, critic, p, _soft_placer(planner, _by_card(queue, p.card),
                                                                                  soft_items)) == OK,
                          lambda p, why: _swap_failed(queue, layout, p, why, spec, cat, critic, rng if swap else None))
    missing += [f"{i.query} ({i.card})" for i in _dropped(queue, layout)]
    missing += planner.soft.missing + planner.soft.dropped(layout.placed)
    critic.log.extend(_shell_notes(layout, cat))
    scene = _composed(name, seed, layout, critic, spec, report, _dedupe(missing))
    scene.unmatched = unmatched
    return scene


def _soft_placer(planner: SoftPlanner, inst: _Instance | None, soft_items: list) -> Any:
    """For a drape or a hanging sheet: ``place(layout, placed) -> Placed | None``; None for everything else."""
    if inst is None or inst.mode not in ("drape", "hang"):
        return None
    li = next(s for s in soft_items if s.card == inst.card)
    return lambda layout, p: planner.place(layout, li, p.key, p.target)


def _by_card(queue: list[_Instance], card: str) -> _Instance | None:
    return next((i for i in queue if i.card == card), None)


def _ordered(queue: list[_Instance], spec: SceneSpec) -> list[_Instance]:
    """Containers and covered objects first (an object a cloth is draped over before any other), then by footprint,
    largest first (soft sheets and piles included). A drape goes right after the object it covers, so the other
    items keep out of the cloth's spread."""
    depth = {i.query: 0 for i in queue}
    for _ in range(len(spec.items)):
        for i in queue:
            if i.target is not None:
                depth[i.query] = max(depth[i.query], depth.get(i.target, 0) + 1)
    covered = {i.target for i in queue if i.mode == "drape"}
    order = sorted((i for i in queue if i.mode != "drape"),
                   key=lambda i: (depth[i.query], i.query not in covered, -i.body.area))
    for d in (i for i in queue if i.mode == "drape"):
        at = max((k for k, i in enumerate(order) if i.query == d.target), default=len(order) - 1)
        order.insert(at + 1, d)
    return order


def _free_body(cat: Catalog, card: str) -> bool:
    """A dynamic rigid object (the engine randomizes poses through a free root joint)."""
    info = cat.info(card)
    return info.motion == "dynamic" and info.route in ("rigid", "rigid_compliant")


def _solids(queue: list[_Instance], spec: SceneSpec, cat: Catalog, critic: Critic) -> None:
    """Decide how the queue's soft solids are placed, and say so in the log. A library object whose material makes
    it a soft body (fruit, foam, a sponge) goes in as a rigid copy (``<card>_rigid``: same shape, mass and
    friction), since a soft body steps far slower; an item that asks for one ("a soft apple") keeps it, with the
    cost in the log. Also sets which cards get per-world jitter (free rigid bodies, the copies included)."""
    rigid, notes = set(), {}
    for i in queue:
        if i.mode != "rigid" or cat.info(i.card).route != "soft_vbd":
            continue
        if SOFT_WORDS & set(spec.items[i.item].words):
            notes[f"{i.card}: a soft body, as asked; {SOFT_COST}"] = logging.WARNING
        else:
            rigid.add(i.card)
            notes[f"{i.card}: placed as a rigid body ({i.card}_rigid, same shape, mass and friction); its material "
                  f"{cat.info(i.card).material} would make it a soft body, and {SOFT_COST}. Ask for 'a soft "
                  f"{i.query}' to get the soft body"] = logging.INFO
    for note, level in notes.items():
        if note not in critic.log:
            critic.log.append(note)
            log.log(level, "compose: %s", note)
    critic.rigid_cards = frozenset(rigid)
    critic.jitter_cards = frozenset(i.card for i in queue if i.mode == "rigid" and cat.info(i.card).motion == "dynamic"
                                    and (_free_body(cat, i.card) or i.card in rigid))


def _shell_notes(layout: Layout, cat: Catalog) -> list[str]:
    """A log line per film object (an open bag) that free rigid objects rest on or in: that contact is slow."""
    out = []
    for p in layout.placed.values():
        if p.card not in cat.entries or cat.info(p.card).route not in SHELL_ROUTES:     # generated soft cards too
            continue
        riders = [q.key for q in layout.placed.values()
                  if q.target == p.key and q.card in cat.entries and _free_body(cat, q.card)]
        if riders:
            out.append(f"{p.key}: {', '.join(riders)} rest on this film object; film contact with free bodies steps "
                       "at a fine time step set by the lightest of them (0.16 ms for a 420 g box), so the scene runs "
                       "several times slower than the film object alone")
    return out


def _layout(spec: SceneSpec, sup: Support, critic: Critic, rng: np.random.Generator) -> Layout:
    """The layout on the support, with the robots stood at the table edge (kept free around their bases)."""
    layout = Layout(sup.placed, spec.clutter, rng)
    if spec.robots:
        critic.robots = place_robots(spec.robots, *layout.table_box())
        critic.seat_robots(layout)
        layout.obstacles = [r.disc() for r in critic.robots]
    return layout


def _within(spec: SceneSpec, layout: Layout, critic: Critic) -> tuple[np.ndarray, np.ndarray] | None:
    """The given work area, else the robots' reach, else None (the whole table)."""
    if spec.area is not None:
        return np.array(spec.area[:2]), np.array(spec.area[2:])
    return reach_area(critic.robots, *layout.table_box()) if critic.robots else None


def _place(layout: Layout, critic: Critic, p: Placed, placer: Any = None) -> str:
    """Put ``p`` down (fresh pose; ``placer(layout, p)`` for a drape or a hanging sheet) and engine-check it;
    another pose after a "retry", up to ``ENGINE_TRIES``. Returns "ok", "retry" (no pose passed) or "broken"."""
    for _ in range(ENGINE_TRIES):
        done = placer(layout, p) if placer is not None else layout.place(p.body, p.card, p.key, p.target, p.item)
        if done is None:
            return RETRY
        status = critic.after_place(layout, p.key)
        if status == OK:
            return OK
        layout.remove(p.key)
        if status == BROKEN:
            return BROKEN
    return RETRY


def _swap_card(queue: list[_Instance], at: int, spec: SceneSpec, cat: Catalog, critic: Critic,
               rng: np.random.Generator | None) -> bool:
    """After ``queue[at]``'s card failed to build: mark it, choose another card for that item and use it for the
    item's remaining instances. False when no other card is left."""
    old = queue[at]
    cat.mark_broken(old.card, critic.log[-1] if critic.log else "failed the load check")
    new = choose_one(spec.items[old.item], cat, rng, critic.log)
    if new is None or new == old.card:
        return False
    critic.log.append(f"{old.query}: {old.card} does not build, using {new}")
    body = cat.body(new)
    queue[at:] = [dataclasses.replace(q, card=new, body=body) if q.card == old.card else q for q in queue[at:]]
    _solids(queue, spec, cat, critic)
    return True


def _swap_failed(queue: list[_Instance], layout: Layout, old: Placed, why: str, spec: SceneSpec, cat: Catalog,
                 critic: Critic, rng: np.random.Generator | None) -> bool:
    """After ``old`` (taken off the layout) failed the full check in two places: put another object of the same
    item down instead, and do not offer its card again (an object that will not rest anywhere, such as a scanned
    mug that slides on one device, is the object's fault, not the place's). False when the item names its card,
    is a soft part, or no other object matches or finds a place."""
    at = next((k for k, q in enumerate(queue) if q.card == old.card and q.mode == "rigid"), None)
    if at is None or spec.items[queue[at].item].object is not None:
        return False
    cat.mark_broken(old.card, why[:300], "does not rest")
    new = choose_one(spec.items[queue[at].item], cat, rng, critic.log)
    if new is None or new == old.card:
        return False
    queue[at] = dataclasses.replace(queue[at], card=new, body=cat.body(new))
    _solids(queue, spec, cat, critic)
    n = sum(p.card == new for p in layout.placed.values())
    key = next(k for k in (new if j == 0 else f"{new}#{j}" for j in range(n + 1)) if k not in layout.placed)
    if _place(layout, critic, Placed(key, new, queue[at].body, 0.0, np.zeros(3), old.target, old.item)) != OK:
        critic.log.append(f"{queue[at].query}: {old.card} failed the full check twice ({why}); {new} found no place")
        return False
    critic.log.append(f"{queue[at].query}: {old.card} failed the full check twice ({why}); using {new} instead")
    return True


def _instances(spec: SceneSpec, cards: dict[str, str | None], cat: Catalog) -> list[_Instance]:
    """One entry per rigid object to place (``_ordered`` sorts them with the soft ones)."""
    out = []
    for idx, item in enumerate(spec.items):
        card = cards.get(item.query)
        if card:
            out += [_Instance(idx, item.query, card, cat.body(card), item.target)] * item.count
    return out


def _dropped(instances: list[_Instance], layout: Layout) -> list[_Instance]:
    """Instances placed once but removed by the final repairs."""
    have: dict[str, int] = {}
    for p in layout.placed.values():
        have[p.card] = have.get(p.card, 0) + 1
    out = []
    for i in instances:
        if have.get(i.card, 0) > 0:
            have[i.card] -= 1
        else:
            out.append(i)
    return out


def _dedupe(items: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for m in items:
        counts[m] = counts.get(m, 0) + 1
    return [m if n == 1 else f"{n} x {m}" for m, n in counts.items()]


def _composed(name: str, seed: int, layout: Layout, critic: Critic, spec: SceneSpec, report: dict | None,
              missing: list[str]) -> ComposedScene:
    card, names = critic.card(layout)
    keys = {v: k for k, v in names.items()}
    rows: list[dict[str, Any]] = []
    for p in layout.placed.values():
        rows.append({"name": keys[p.key], "card": p.card, "item": spec.items[p.item].query, "pose": list(p.pose),
                     "on": keys.get(p.target) if p.target else keys[layout.support.key]})
    soft = critic.soft
    out = ComposedScene(name, seed, card, rows, missing, report, list(critic.log))
    if soft is not None and (soft.cards or soft.emitters):
        made = {r["card"]: r for r in soft.rows}
        layout_cards = {p.card for p in layout.placed.values()}
        for pl in card.objects:
            if pl.object in made and pl.object not in layout_cards:
                rows.append({"name": pl.object, "card": pl.object, "item": made[pl.object]["item"],
                             "pose": list(pl.pose), "on": made[pl.object].get("in")})
        out.soft = {**soft.to_dict(), "items": soft.rows_for(layout.placed)}
        used = {pl.object for pl in card.objects}
        out.cards = {n: c.text for n, c in soft.cards.items() if n in used}
        out.files = {rel: data for n, c in soft.cards.items() if n in used for rel, data in c.files.items()}
    out.cards.update({n: text for n, text in critic.copies.items() if any(pl.object == n for pl in card.objects)})
    return out


__all__ = ["compose"]
