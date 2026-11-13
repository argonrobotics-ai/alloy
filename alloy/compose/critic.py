# SPDX-License-Identifier: Apache-2.0
"""The physics critic in the loop: ``alloy.check`` after every placement, and a full check with repairs at the end.

After each object is put down, the load checks (cards, overlap in the engine's own collision shapes) run on the
scene so far. Findings on the new object are fixed with ``alloy.check.apply_fixes`` (a lift out of an overlap) and
checked again; an object that still fails is taken off and tried elsewhere. At the end the full check runs (rest
drift, penetration, health, energy); each object an error names is moved once, then dropped if it fails again.

Example::

    >>> critic = Critic(library, device="cpu")                      # doctest: +SKIP
    >>> ok = critic.after_place(layout, "cube_beech_30#1")           # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.cards import types as ct
from alloy.compose.errors import ComposeError
from alloy.compose.layout import Layout, Placed

log = logging.getLogger("alloy.compose")

FIX_ROUNDS = 2              # load-check -> apply_fixes rounds per placement
REPAIR_ROUNDS = 2           # full-check -> move / drop rounds
MAX_LIFT = 0.01             # m: a suggested lift larger than this is not taken (the pose is wrong, not the gap)
MODES = ("full", "load", "none")
OK, RETRY, BROKEN = "ok", "retry", "broken"          # after_place outcomes
MIX_FIX = ("ask for fewer soft kinds in one scene: the engine couples the rigid bodies with one deformable kind "
           "(cloth, or grains and pastes) and adds a liquid only beside some of them; drop the liquid, the cloth or "
           "the grains")


@dataclass
class Critic:
    """Runs the checks with one card library and device. ``mode``: "full" (load checks per placement and a full
    check at the end), "load" (load checks only) or "none" (geometry only)."""

    library: object
    device: str | None = None
    mode: str = "full"
    worlds: int | None = None
    seconds: float | None = None
    scene_worlds: int = 1
    robots: list = field(default_factory=list)          # PlacedRobot: added to every scene card
    jitter: float = 0.0                                  # m, per-world x/y spread of the ``jitter_cards``
    jitter_cards: frozenset[str] = frozenset()
    log: list[str] = field(default_factory=list)
    checks_run: int = 0
    work: Path | None = None                             # the card root generated cards are written to
    soft: Any = None                                     # alloy.compose.soft.scene.SoftScene of the scene in hand
    rigid_cards: frozenset[str] = frozenset()            # soft solids to place as rigid copies (build._solids)
    copies: dict[str, str] = field(default_factory=dict)  # generated rigid copies: card name -> YAML

    # --- cards -----------------------------------------------------------------------------------------------------

    def card(self, layout: Layout) -> tuple[ct.SceneCard, dict[str, str]]:
        """The scene card of the layout (support first) and engine name -> layout key."""
        objs = [layout.support, *layout.placed.values()]
        seen: dict[str, int] = {}
        names: dict[str, str] = {}
        placements = []
        bases = {q.target for q in layout.placed.values()}
        for p in objs:
            still = self.soft is not None and self.soft.hold_still and p is not layout.support
            card = self.soft.still(p.card, self.library, self.work) if still else \
                self._rigid(p.card) if p.card in self.rigid_cards else p.card
            k = seen.get(card, 0)
            seen[card] = k + 1
            names[card if k == 0 else f"{card}#{k}"] = p.key
            free = self.jitter > 0 and p.card in self.jitter_cards and p.target is None and p.key not in bases
            placements.append(ct.Placement(card, p.pose, _spread(p.pose, self.jitter) if free and not still else ()))
        robots = tuple(ct.RobotRef(r.spec.path, r.spec.kind, r.base_pose(), r.spec.name) for r in self.robots)
        if self.soft is None:
            return ct.SceneCard(robots=robots, objects=tuple(placements), worlds=self.scene_worlds), names
        extra = self.soft.placements(layout.placed)
        names.update({pl.object: pl.object for pl in extra})
        return ct.SceneCard(robots=robots, objects=tuple(placements + extra),
                            attachments=self.soft.attachments_for(layout.placed, names),
                            emitters=self.soft.emitters_for(layout.placed), worlds=self.scene_worlds), names

    def _rigid(self, card: str) -> str:
        """The name of a rigid copy of the soft solid ``card`` (made once, written into the work card root)."""
        from alloy.compose.soft.cards import rigid_copy  # noqa: PLC0415

        name = f"{card}_rigid"
        if name not in self.copies:
            self.copies[name] = rigid_copy(name, self.library.object(card)).text
            (self.work / "objects").mkdir(parents=True, exist_ok=True)
            (self.work / "objects" / f"{name}.yaml").write_text(self.copies[name], encoding="utf-8")
            self.library.rescan()                                 # the checks read the new card from the work root
        return name

    # --- robots ----------------------------------------------------------------------------------------------------

    def seat_robots(self, layout: Layout) -> None:
        """Lift each robot whose base geometry reaches into the support (a mount flange below the base origin) by
        the overlap the load check measures, so it stands on the table top. Lifts are kept per robot file. Robots
        that fail that check otherwise (the scene does not assemble, a mesh they name is missing) raise
        ``ComposeError`` (field "robots") naming their files, before any object is placed."""
        if self.mode == "none" or not self.robots:
            return
        from alloy.check import check_load  # noqa: PLC0415

        cache = self.__dict__.setdefault("_robot_lifts", {})
        base, lift = list(self.robots), dict(cache)
        sup = ct.Placement(layout.support.card, layout.support.pose)
        for _ in range(3):
            self.robots = [dataclasses.replace(r, pose=(*r.pose[:2], r.pose[2] + lift.get(r.spec.path, 0.0),
                                                        r.pose[3])) for r in base]
            robots = tuple(ct.RobotRef(r.spec.path, r.spec.kind, r.base_pose(), r.spec.name) for r in self.robots)
            try:
                report = check_load(ct.SceneCard(robots=robots, objects=(sup,)), worlds=1, device=self.device,
                                    library=self.library)
            except Exception as exc:         # the loader could not read a robot (a mesh it names is missing)
                raise _robots_fail(self.robots, _why(exc)) from exc
            self.checks_run += 1
            bad = [f for f in report.errors() if f.check != "overlap"]      # overlaps: lifted below
            if bad:
                raise _robots_fail(self.robots, bad[0].message, bad[0].fix)
            deep = {o: float(f.value or 0.0) for f in report.findings if f.check == "overlap" and f.level == "error"
                    for o in f.objects if o != layout.support.card}
            if not deep:
                cache.update(lift)
                return
            for r in base:
                if r.spec.name in deep:
                    lift[r.spec.path] = lift.get(r.spec.path, 0.0) + deep[r.spec.name] + 5e-4
                    self.log.append(f"{r.spec.name}: lifted {deep[r.spec.name] * 1e3:.1f} mm so its base stands on "
                                    "the table (its geometry reaches below its origin)")

    # --- after the soft pass -----------------------------------------------------------------------------------------

    def drop_refused_pairs(self, layout: Layout) -> None:
        """Take out one side of each solver pair the engine refuses in one scene (``alloy.compose.soft.pairs``: a
        towel beside a bin of rice), found from the plan before any build, in every mode; what is taken out is
        reported missing with why and what to ask for instead."""
        if self.soft is None or not (self.soft.emitters or any(_solid(c) for c in self.soft.cards.values())):
            return
        from alloy.compose.soft import pairs  # noqa: PLC0415

        rods = frozenset(n for n, c in self.soft.cards.items() if not _solid(c))
        for _ in range(3):
            card, names = self.card(layout)
            hit = pairs.refused(card, self.library, rods)
            if hit is None:
                return
            exc, routes = hit
            extras = {e.card.name for e in self.soft.extras}
            gone = pairs.pick(exc.pair, routes, extras)
            label = {r["name"]: r["item"] for r in self.soft.rows}
            why = pairs.reason(exc.pair, routes, gone, exc, label)[:900]
            keys = [names[n] for n in gone if n not in extras and names.get(n) in layout.placed]
            self.soft.extras = [e for e in self.soft.extras if e.card.name not in gone]
            for key in keys:
                layout.remove(key)
            for n in gone:
                self.soft.missing.append(f"{label.get(n, n)}: not placed: {why}")
                self.log.append(f"{n}: not placed ({why})")
            if not keys and not extras & set(gone):
                return                                       # nothing this check can take out: the build says why

    def after_soft(self, layout: Layout) -> None:
        """Load-check the scene with its fills, pours and cables; a soft part an error names is taken out (and
        reported missing), twice at most."""
        self.drop_refused_pairs(layout)
        if self.mode == "none" or self.soft is None or not (self.soft.extras or self.soft.emitters):
            return
        from alloy.check import check_load  # noqa: PLC0415

        for _ in range(2):
            card, _names = self.card(layout)
            try:
                report = check_load(card, worlds=1, device=self.device, library=self.library)
            except Exception as exc:  # noqa: BLE001 - a soft part that breaks the build is reported, with why
                self._drop_soft(f"the scene does not build with its soft items ({_why(exc)})", None)
                return
            self.checks_run += 1
            bad = {o: f.message for f in report.findings if f.level == "error" for o in f.objects
                   if any(e.card.name == o for e in self.soft.extras)}
            if not bad and report.ok:
                return
            if not bad:
                errors = [f.message for f in report.findings if f.level == "error"] or ["load check failed"]
                self._drop_soft(errors[0] + (f"; {MIX_FIX}" if _too_many_kinds(errors[0]) else ""), None)
                return
            for name, why in bad.items():
                self._drop_soft(why, name)

    def _drop_soft(self, why: str, name: str | None) -> None:
        gone = [e for e in self.soft.extras if name is None or e.card.name == name]
        self.soft.extras = [e for e in self.soft.extras if e not in gone]
        if name is None:
            self.soft.emitters = []
        for e in gone:
            self.soft.missing.append(f"{e.card.name}: {why[:500]}")
            self.log.append(f"{e.card.name}: not placed ({why[:500]})")

    def run_settings(self) -> tuple[float | None, bool]:
        """(seconds, expect_rest) of the full check: long enough to see a soft scene settle or a pour end."""
        soft = self.soft
        if soft is None or soft.settle_s <= 0:
            return self.seconds, True
        from alloy.check.api import resolve_settings  # noqa: PLC0415

        _dev, _w, secs = resolve_settings(self.device, self.worlds, self.seconds)
        return max(secs, soft.settle_s + max(1.0, soft.settle_s)), soft.expect_rest

    # --- after each placement --------------------------------------------------------------------------------------

    def after_place(self, layout: Layout, key: str) -> str:
        """Load-check the scene with ``key`` just placed: "ok" (after fixes), "retry" (this pose fails; another may
        not) or "broken" (the object does not build or its card fails: no pose will help)."""
        if self.mode == "none":
            return OK
        from alloy.check import apply_fixes, check_load  # noqa: PLC0415

        for _ in range(FIX_ROUNDS + 1):
            card, names = self.card(layout)
            try:
                report = check_load(card, worlds=1, device=self.device, library=self.library)
                if _pair_refused(report):        # not this pose's doing: drop_refused_pairs takes a side out later
                    card, names = self._without_refused_pairs(card, names, key)
                    report = check_load(card, worlds=1, device=self.device, library=self.library)
            except Exception as exc:  # noqa: BLE001 - an object that breaks the build is not placed, with why
                self.log.append(f"{key}: the scene does not build with it ({_why(exc)})")
                return BROKEN
            self.checks_run += 1
            # the scene passed before this object, so every error is its doing (an assembly error names no object)
            mine = [f for f in report.findings if f.level == "error" or _overlap_warning(f, names, key)]
            if not mine:
                return OK
            if any(f.check == "cards" for f in mine):
                self.log.append(f"{key}: card problem: {mine[0].message[:300]}")
                return BROKEN
            sub = dataclasses.replace(report, findings=mine)
            fixed, applied = apply_fixes(card, sub)
            if not applied or not self._take_lifts(layout, card, fixed, names):
                self.log.append(f"{key}: {mine[0].message}")
                return RETRY
            self.log.extend(f"{key}: fixed: {a}" for a in applied)
        return RETRY

    def _without_refused_pairs(self, card: ct.SceneCard, names: dict[str, str],
                               key: str) -> tuple[ct.SceneCard, dict[str, str]]:
        """``card`` and ``names`` for ``key``'s load check without the partners of each solver pair the engine refuses
        in one scene (a towel and a pile of sand): the side without ``key`` (the second side when ``key`` is on
        neither) is left out of the check only; ``drop_refused_pairs`` later takes a side out of the scene, with why.
        """
        from alloy.compose.soft import pairs  # noqa: PLC0415

        rods = frozenset(n for n, c in self.soft.cards.items() if not _solid(c)) if self.soft else frozenset()
        for _ in range(3):
            hit = pairs.refused(card, self.library, rods)
            if hit is None:
                break
            exc, routes = hit
            sides = [{n for n, r in routes.items() if r == str(p)} for p in exc.pair]
            out = sides[0] if any(names.get(n) == key for n in sides[1]) else sides[1]
            keep = [n not in out for n in names]
            card = dataclasses.replace(
                card, objects=tuple(o for o, k in zip(card.objects, keep) if k),
                attachments=tuple(a for a in card.attachments if not {a.a.split(":")[0], a.b.split(":")[0]} & out))
            names = {n: v for n, v in names.items() if n not in out}
        return card, names

    def _take_lifts(self, layout: Layout, before: ct.SceneCard, after: ct.SceneCard, names: dict[str, str]) -> bool:
        """Copy small position changes back into the layout; False if a fix wants a rotation or a big move."""
        keys = list(names.values())
        moved = False
        for i, (a, b) in enumerate(zip(before.objects, after.objects)):
            if a.pose == b.pose:
                continue
            d = np.subtract(b.pose[:3], a.pose[:3])
            if i == 0 or not np.allclose(a.pose[3:], b.pose[3:], atol=1e-6) or np.linalg.norm(d) > MAX_LIFT:
                return False
            layout.placed[keys[i]].pos = layout.placed[keys[i]].pos + d
            moved = True
        return moved

    # --- at the end -------------------------------------------------------------------------------------------------

    def final(self, layout: Layout, replace, swap=None) -> dict | None:
        """Full check with repairs; ``replace(placed) -> bool`` puts a removed object down again, and
        ``swap(placed, why) -> bool`` puts another object of the same item in the place of one that failed twice.
        The last report as a dict (None in "load"/"none" mode)."""
        if self.mode != "full":
            return None
        from alloy.check import check  # noqa: PLC0415

        moved: set[str] = set()
        report = None
        for _ in range(REPAIR_ROUNDS + 1):
            if not layout.placed:
                self.log.append("full check skipped: nothing is placed")
                return None
            card, names = self.card(layout)
            seconds, rest = self.run_settings()
            try:
                report = check(card, self.worlds, seconds, device=self.device, library=self.library,
                               expect_rest=rest, extra=self._extra(layout, card, names))
            except Exception as exc:  # noqa: BLE001 - a check that raises fails the scene, with the reason
                self.log.append(f"full check raised {type(exc).__name__}: {exc}")
                return {"schema": "alloy.check/1", "ok": False, "errors": 1, "checks": {},
                        "findings": [], "raised": f"{type(exc).__name__}: {exc}"}
            self.checks_run += 1
            culprits = _culprits(report, names, layout.support.key)
            if not culprits:
                break
            for key, why in culprits.items():
                self._repair(layout, key, why, key in moved, replace, swap)
                moved.add(key)
        return report.to_dict() if report is not None else None

    def _extra(self, layout: Layout, card: ct.SceneCard, names: dict[str, str]) -> list:
        """The soft checks (``alloy.compose.soft.probe``) as an extra run check of the full check, or none."""
        if self.soft is None or not self.soft.watches:
            return []
        from alloy.compose.soft.probe import soft_check  # noqa: PLC0415

        present = {pl.object for pl in card.objects}
        watches = [w for w in self.soft.watches if (w.name in present if w.name else w.key in layout.placed)]
        poses = {pl.object: pl.pose for pl in card.objects}
        return [soft_check(watches, names, layout.support.outline, poses=poses)] if watches else []

    def _repair(self, layout: Layout, key: str, why: str, again: bool, replace, swap=None) -> None:
        """Move an object the full check names; one that fails again in its new place gives its place to another
        object of the same item (``swap``), and is dropped when there is none."""
        if key not in layout.placed:
            return
        removed = layout.remove(key)
        if again and not (swap is not None and swap(removed[0], why)):
            self.log.extend(f"{p.key}: dropped ({'failed the full check twice: ' + why if p.key == key else 'its '
                            'support was dropped'})" for p in removed)
            return
        if not again:
            self.log.append(f"{key}: moved (full check: {why})")
        for p in removed[1:] if again else removed:
            if not replace(p):
                self.log.append(f"{p.key}: dropped (no other place found)")


def _spread(pose: tuple[float, ...], j: float) -> tuple[ct.Randomization, ...]:
    """Per-world x and y within +-j of the pose (the engine draws one value per world)."""
    return (ct.Randomization("pose.0", pose[0] - j, pose[0] + j), ct.Randomization("pose.1", pose[1] - j, pose[1] + j))


def _pair_refused(report: Any) -> bool:
    """A load check that failed at assembly on a solver pair the engine refuses in one scene
    (``alloy.coupling.unsupported``), which ``drop_refused_pairs`` resolves, not the placement."""
    from alloy.coupling.unsupported import UnsupportedPairError  # noqa: PLC0415

    tag = f"{UnsupportedPairError.__name__}:"
    return any(f.check == "cards" and f.level == "error" and tag in f.message for f in report.findings)


def _too_many_kinds(message: str) -> bool:
    """The engine's refusal of a liquid beside more deformable kinds than it couples (``UnsupportedSceneError``;
    older refusals of it carry their decision tag)."""
    from alloy.scene.coupler_select import UnsupportedSceneError  # noqa: PLC0415

    return f"{UnsupportedSceneError.__name__}:" in message or "SCR-DC5-7" in message


def _solid(card: Any) -> bool:
    """A generated card that is not a cable (whose curve route runs a slow self-test and is in no refused pair).

    Example: ``_solid(K.grain_region(...))`` is True, ``_solid(K.cable_curve(...))`` False."""
    return not isinstance(getattr(card.obj, "shape", None), ct.CurveSource)


def _why(exc: Exception, most: int = 200) -> str:
    """``"Type: message"`` cut to ``most`` characters, plus the error's ``fix`` (an engine error's nearest supported
    alternative) when the cut dropped it.

    Example: ``_why(ValueError("x" * 300))`` is ``"ValueError: " + "x" * 200 + "..."``."""
    text, fix = str(exc), getattr(exc, "fix", None)
    out = f"{type(exc).__name__}: {text[:most]}" + ("..." if len(text) > most else "")
    if isinstance(fix, str) and fix and fix not in text[:most]:
        out += f"; to fix: {fix[:most]}"
    return out


def _robots_fail(robots: list, why: str, fix: str | None = None) -> ComposeError:
    """The error for robots that fail the load check on the bare table (``seat_robots``): a robot the engine cannot
    read or assemble would otherwise fail every later placement and be blamed on each object in turn."""
    files = ", ".join(repr(r.spec.path) for r in robots)
    what = f"robot {files} does" if len(robots) == 1 else f"robots {files} do"
    return ComposeError(f"the {what} not load on the table: {why}", fix or "fix the robot file or give another robot",
                        field="robots")


def _overlap_warning(f, names: dict[str, str], key: str) -> bool:
    return f.level == "warn" and f.check == "overlap" and any(names.get(o) == key for o in f.objects)


def _culprits(report, names: dict[str, str], support: str) -> dict[str, str]:
    """Layout key -> the first error message that names it (the support is never moved)."""
    keys: dict[str, str] = {}
    for f in report.errors():
        for o in f.objects:
            k = names.get(o)
            if k is not None and k != support and k not in keys:
                keys[k] = f.message
    return keys


def placed_summary(p: Placed) -> dict:
    """JSON row for one placed object."""
    return {"key": p.key, "card": p.card, "pose": list(p.pose), "on": p.target}


__all__ = ["BROKEN", "Critic", "MODES", "OK", "RETRY", "placed_summary"]
