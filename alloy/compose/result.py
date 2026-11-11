# SPDX-License-Identifier: Apache-2.0
"""What ``alloy.compose.compose`` returns: the checked scene and its variations, as scene cards plus JSON.

``save(folder)`` writes a card root that loads on its own: ``scenes/<name>.yaml`` per scene, any generated card
(the floor) under ``objects/``, and ``compose.json`` (the request, every placement and every check report).
``scene(i)`` gives an editable ``alloy.Scene`` (``alloy.api.build(scene)`` runs it).

Example::

    >>> result = alloy.compose.compose("a mug and two cubes on a table", variations=2)    # doctest: +SKIP
    >>> result.ok, len(result.scenes)                                                     # doctest: +SKIP
    (True, 3)
    >>> paths = result.save("my_scenes")                                                  # doctest: +SKIP
    >>> sim = alloy.api.build(result.scene(0), worlds=16)                                     # doctest: +SKIP
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy.cards import types as ct
from alloy.compose.spec import ComposeError, SceneSpec

RESULT_SCHEMA = "alloy.compose/1"


@dataclass
class ComposedScene:
    """One built scene: its card, whether it passed, what was placed and what was not.

    ``ok``: the full check found no error (or checks were off) and every requested object was placed;
    ``report``: the last ``alloy.check`` report as a dict (None when the full check was off); ``objects``: one row
    per placed object (engine name, card, pose, what it rests on); ``missing``: requested objects not placed;
    ``log``: what the builder changed on the way (fixes, moves, drops); ``soft``: the liquids, grains, cloth and
    cables it made (one row each: amount, container, route, ...), ``estimated`` (what it assumed) and ``settle_s``
    (how long the scene takes to come to rest, or to finish a pour)."""

    name: str
    seed: int
    card: ct.SceneCard
    objects: list[dict[str, Any]]
    missing: list[str] = field(default_factory=list)
    report: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)
    soft: dict[str, Any] = field(default_factory=dict)         # soft items: rows, estimates, settle time
    cards: dict[str, str] = field(default_factory=dict)        # generated cards this scene uses (name -> YAML)
    files: dict[str, bytes] = field(default_factory=dict)       # generated card files (meshes), saved

    @property
    def checked_ok(self) -> bool:
        return self.report is None or bool(self.report.get("ok"))

    @property
    def ok(self) -> bool:
        return self.checked_ok and not self.missing

    def metrics(self) -> dict[str, Any]:
        """Scene-quality rates from the check report: ``stable`` (share of objects with no rest-drift finding),
        ``overlap_free`` (share with no overlap finding at load), ``placed`` and ``missing`` counts."""
        out: dict[str, Any] = {"placed": len(self.objects), "missing": len(self.missing)}
        names = {o["name"] for o in self.objects}
        if self.report is None or not names:
            return out
        for key, check in (("stable", "rest_drift"), ("overlap_free", "overlap")):
            hit = {o for f in self.report.get("findings", []) if f.get("check") == check and f.get("level") in
                   ("error", "warn") for o in f.get("objects", []) if o in names}
            out[key] = round(1.0 - len(hit) / len(names), 3)
        return out

    def yaml(self, header: str = "") -> str:
        """The scene card as YAML text."""
        from alloy.cards.loader import dump_card  # noqa: PLC0415

        return dump_card(self.card, header=header)

    def to_dict(self) -> dict[str, Any]:
        out = {"name": self.name, "seed": self.seed, "ok": self.ok, "metrics": self.metrics(),
               "objects": self.objects, "missing": self.missing, "log": self.log, "check": self.report}
        if self.soft:
            out["soft"] = self.soft
        return out


@dataclass
class ComposeResult:
    """The base scene (``scenes[0]``) and its variations; ``generated`` holds generated cards (stem -> YAML) and
    ``sources`` the import libraries the scenes use (their roots must be visible to load them)."""

    spec: SceneSpec
    scenes: list[ComposedScene]
    generated: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)       # other generated files (relative path -> bytes)

    @property
    def ok(self) -> bool:
        """The base scene passed (see ``ComposedScene.ok``)."""
        return bool(self.scenes) and self.scenes[0].ok

    @property
    def variations(self) -> list[ComposedScene]:
        return self.scenes[1:]

    def summary(self) -> str:
        """A few lines for people: one per scene with its verdict and object count, and under a scene whose check
        failed, its first errors with the change that fixes each."""
        lines = [f"{self.spec.sentence or 'spec'}: {'OK' if self.ok else 'NOT OK'}"]
        for s in self.scenes:
            miss = f", missing {', '.join(s.missing)}" if s.missing else ""
            check = "not run" if s.report is None else ("pass" if s.checked_ok else "FAIL")
            lines.append(f"  {s.name}: {len(s.objects)} objects, check {check}{miss}")
            errors = [f for f in (s.report or {}).get("findings", []) if f.get("level") == "error"]
            lines += [f"    {f.get('check')}: {f.get('message')} (fix: {f.get('fix')})" for f in errors[:3]]
            if s.report is not None and not s.checked_ok and s.report.get("raised"):
                lines.append(f"    the check raised {s.report['raised']}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESULT_SCHEMA, "ok": self.ok, "spec": self.spec.to_dict(), "sources": self.sources,
                "notes": self.notes, "generated_cards": sorted(self.generated), "timings_s": self.timings_s,
                "scenes": [s.to_dict() for s in self.scenes]}

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    def save(self, folder: str | Path) -> list[Path]:
        """Write ``scenes/<name>.yaml`` per scene, generated cards to ``objects/`` and ``compose.json``.

        Example: ``result.save("out")`` -> ``[out/scenes/scene.yaml, ..., out/compose.json]``."""
        root = Path(folder).expanduser()
        (root / "scenes").mkdir(parents=True, exist_ok=True)
        paths = []
        for stem, text in self.generated.items():
            (root / "objects").mkdir(exist_ok=True)
            p = root / "objects" / f"{stem}.yaml"
            p.write_text(text, encoding="utf-8")
            paths.append(p)
        for rel, data in self.files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            paths.append(p)
        for s in self.scenes:
            p = root / "scenes" / f"{s.name}.yaml"
            p.write_text(s.yaml(self._header(s)), encoding="utf-8")
            paths.append(p)
        p = root / "compose.json"
        p.write_text(self.to_json() + "\n", encoding="utf-8")
        return [*paths, p]

    def _header(self, s: ComposedScene) -> str:
        lines = [f"generated by alloy.compose from: {self.spec.sentence or '(spec)'} (seed {s.seed})",
                 f"check: {'not run' if s.report is None else ('pass' if s.checked_ok else 'FAIL')}"
                 + (f"; missing: {', '.join(s.missing)}" if s.missing else "")]
        if self.sources:
            lines.append(f"uses imported objects from: {', '.join(self.sources)} "
                         "(load with alloy.compose.load_scene, which adds their card roots)")
        return "\n".join(lines)

    def scene(self, index: int = 0):
        """An editable ``alloy.Scene`` of scene ``index`` (0: the base scene).

        Example: ``alloy.api.build(result.scene())``."""
        if not 0 <= index < len(self.scenes):
            raise ComposeError(f"scene index {index} is out of range", f"use 0..{len(self.scenes) - 1}",
                               field="index")
        import tempfile  # noqa: PLC0415

        from alloy.compose.load import load_scene  # noqa: PLC0415

        folder = Path(tempfile.mkdtemp(prefix="alloy-compose-"))
        self.save(folder)
        return load_scene(folder / "scenes" / f"{self.scenes[index].name}.yaml")


__all__ = ["ComposeResult", "ComposedScene", "RESULT_SCHEMA"]
