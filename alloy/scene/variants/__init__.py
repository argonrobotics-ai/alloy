# SPDX-License-Identifier: Apache-2.0
"""Scene variants: n edited copies of one ``alloy.Scene`` for visual and placement diversity, deterministic from a
seed. Not the engine's per-world randomization (``Scene.randomize``): each variant is its own scene.

- Placement: every free object's start x/y (and an optional small yaw) moves within +-``pos`` / +-``yaw_deg``;
  robots, fixtures, static and kinematic objects, liquids and objects held by a robot stay. Objects that touch at the
  start move together. A draw that overlaps anything or takes an object off what it rests on is redrawn
  (``alloy.scene.variants.place``). ``settle=`` seconds also builds each variant and runs ``alloy.check`` (start
  overlap, a short settle, health) before keeping it.
- Looks: each object's look takes another texture of its family (wood -> wood, fabric -> fabric), read from the
  look files' metadata, plus a small colour tint (``alloy.scene.variants.looks``). A variant item carries its look
  as ``item["look"]`` ({materials, assign, default}); ``scene_look(variant)`` turns a variant into a camera look.
  Ground and floor items take floor textures only. Large surfaces keep contrast: a support and what it stands on
  differ in family or in mean albedo by ``contrast`` (0.15; on a floor, in albedo), and ``max_albedo`` caps them
  (``alloy.scene.variants.contrast``); looks are redrawn until they do.
- Files: with ``out=``, ``variant_000.json`` ... (``Scene.to_json``), ``variants.json`` (seed, settings, each
  variant's moves and textures) and, with ``contact_sheet=True``, ``contact_sheet.png``.

Example::

    from alloy.scene.variants import vary
    scenes = vary(scene, 8, pos=0.01, yaw_deg=10, seed=3, out="variants", contact_sheet=True)
    sim = alloy.build(scenes[0])
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alloy.scene.variants import contrast as C
from alloy.scene.variants import looks as L
from alloy.scene.variants import sheet as SH
from alloy.scene.variants.base import Base, assemble, free_objects
from alloy.scene.variants.place import Placer, edit_poses
from alloy.scene.variants.xform import Move

log = logging.getLogger("alloy.scene")

SCHEMA = "alloy.variants/1"
scene_look = L.scene_look


@dataclass(frozen=True)
class VaryOptions:
    """``vary``'s settings (see its docstring)."""

    pos: float = 0.005
    yaw_deg: float = 0.0
    textures: str = "same-family"
    seed: int = 0
    tint: float = 0.05
    max_tries: int = 200
    settle: float | None = None
    contrast: float = C.ALBEDO_GAP
    max_albedo: float | None = None

    def check(self, n: int) -> None:
        from alloy.api.errors import SceneError  # noqa: PLC0415

        bad = [(f, v) for f, v in (("pos", self.pos), ("yaw_deg", self.yaw_deg), ("tint", self.tint),
                                   ("contrast", self.contrast))
               if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0.0)]
        if not isinstance(n, int) or n < 1:
            raise SceneError(f"n must be an int >= 1, got {n!r}", "pass n=4 (or more)", field="n")
        if bad:
            raise SceneError(f"{bad[0][0]} must be a number >= 0, got {bad[0][1]!r}", "pass e.g. pos=0.005, "
                             "yaw_deg=10, tint=0.05", field=bad[0][0])
        if self.textures not in L.MODES:
            raise SceneError(f"textures must be one of {list(L.MODES)}, got {self.textures!r}",
                             "pass textures='same-family' (swap any look with a family), 'textured' (swap textured "
                             "looks only) or 'keep' (tint only)", field="textures")
        if self.max_albedo is not None and not 0.0 < self.max_albedo <= 1.0:
            raise SceneError(f"max_albedo must be in (0, 1] or None, got {self.max_albedo!r}", "pass max_albedo=0.6",
                             field="max_albedo")
        if self.settle is not None and not self.settle > 0.0:
            raise SceneError(f"settle must be seconds > 0 or None, got {self.settle!r}", "pass settle=0.2",
                             field="settle")


@dataclass
class Variant:
    """One variant: the edited scene, its moves, its textures {item: {look material: texture}}, the placement and
    look draws taken, and the look rule its kept looks still break (None when they pass)."""

    scene: Any
    moves: list[Move]
    textures: dict[str, dict[str, str]]
    draws: int
    look_draws: int = 1
    look_rule: str | None = None


@dataclass(frozen=True)
class LookPlan:
    """What every variant's looks are drawn from: the texture pool, each item's current look, the large surfaces."""

    pool: dict
    current: dict
    surfaces: C.Surfaces


LOOK_TRIES = 500    # look draws per variant before keeping the last one with a warning (each is cheap)


def vary(scene: Any, n: int, pos: float = 0.005, yaw_deg: float = 0.0, textures: str = "same-family", seed: int = 0,
         *, tint: float = 0.05, out: str | Path | None = None, contact_sheet: bool = False,
         sheet: dict | None = None, max_tries: int = 200, settle: float | None = None,
         contrast: float = C.ALBEDO_GAP, max_albedo: float | None = None) -> list[Any]:
    """``n`` edited copies of ``scene`` (module docstring); returns the ``alloy.Scene`` variants.

    ``pos`` [m] and ``yaw_deg`` bound the start jitter; ``textures``: "same-family", "textured" or "keep"
    (``alloy.scene.variants.looks``); ``tint``: relative colour jitter (0: none); ``seed``: same seed, same variants.
    ``out``: folder for the JSON files (and the contact sheet); ``sheet``: contact-sheet settings ``{"width",
    "height", "spp", "quality": "full" | "fast"}``;
    ``max_tries``: draws per variant before giving up (SceneError); ``settle``: seconds of an ``alloy.check`` run per
    variant (built on the CPU), redrawn while it reports an error; ``contrast``: albedo difference a support and
    what it stands on need when they share a texture family (0: no rule); ``max_albedo``: cap on the mean albedo of
    large surfaces (supports, ground), None for none."""
    opts = VaryOptions(pos, yaw_deg, textures, int(seed), tint, int(max_tries), settle, contrast, max_albedo)
    opts.check(n)
    base = assemble(scene)
    free = free_objects(scene, base)
    placer = Placer.make(base, free, opts.pos, opts.yaw_deg)
    plan = LookPlan(L.texture_pool(base.lib) if opts.textures != "keep" else {},
                    {name: L.item_look(name, item, base.lib) for name, item in scene.items.items()},
                    C.surfaces(scene, base, placer.supports, free))
    variants = [_variant(scene, k, opts, base, placer, plan) for k in range(n)]
    log.info("variants: %d of %s (%d free groups, %d texture families, contrast pairs %s)", n, list(scene.items)[:4],
             len(placer.groups), len(plan.pool), list(plan.surfaces.pairs))
    if out is not None:
        write(Path(out), variants, opts, plan, base, placer, sheet if contact_sheet else None)
    return [v.scene for v in variants]


def _variant(scene: Any, k: int, opts: VaryOptions, base: Base, placer: Placer, plan: LookPlan) -> Variant:
    from alloy.api.scene import Scene  # noqa: PLC0415

    rng_place = np.random.default_rng([opts.seed, k, 0])
    rng_look = np.random.default_rng([opts.seed, k, 1])
    draws = 0
    while True:
        moves, tries = placer.place(rng_place, opts.max_tries - draws) if placer.groups else ([], 0)
        draws += tries
        v = Scene.from_dict(scene.to_dict())
        edit_poses(v, moves, base)
        why = _settle_problem(v, opts.settle) if opts.settle else None
        if why is None:
            break
        log.info("variants: variant %d draw %d refused by alloy.check (%s)", k, draws, why)
        if draws >= opts.max_tries or not placer.groups:
            from alloy.api.errors import SceneError  # noqa: PLC0415

            raise SceneError(f"variant {k}: no draw passed alloy.check in {draws} draws (last: {why})",
                             "lower pos / yaw_deg, or fix the base scene (run alloy.check on it)", field="settle")
    looks, chosen, look_draws, rule = _looks(k, opts, plan, rng_look)
    for name, look in looks.items():
        v.items[name]["look"] = look
    return Variant(v, moves, chosen, draws, look_draws, rule)


def _looks(k: int, opts: VaryOptions, plan: LookPlan, rng: np.random.Generator) -> tuple[dict, dict, int, str | None]:
    """(looks, textures, draws, rule still broken or None): looks drawn until the large surfaces pass the contrast
    rules (``alloy.scene.variants.contrast``), at most ``LOOK_TRIES`` times; then the last draw, with a warning."""
    why = None
    for tries in range(1, LOOK_TRIES + 1):
        looks, chosen = {}, {}
        for name, look in plan.current.items():
            if look is not None:
                family = C.FLOOR_FAMILY if name in plan.surfaces.floors else None
                looks[name], chosen[name] = L.vary_look(look, plan.pool, rng, opts.textures, opts.tint, family)
        why = C.problem(looks, plan.surfaces, plan.pool, opts.contrast, opts.max_albedo)
        if why is None:
            return looks, chosen, tries, None
    log.warning("variants: variant %d keeps looks that break a surface rule after %d draws: %s", k, LOOK_TRIES, why)
    return looks, chosen, LOOK_TRIES, why


def _settle_problem(scene: Any, seconds: float) -> str | None:
    """The first ``alloy.check`` error of ``scene`` built on the CPU and run ``seconds`` (None: it passes)."""
    import tempfile  # noqa: PLC0415

    from alloy.api import catalog  # noqa: PLC0415
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.cards.roots import assets_root  # noqa: PLC0415
    from alloy.check.api import check  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="alloy-variant-check-") as tmp:
        path = scene.save(tmp)
        lib = CardLibrary(roots=catalog.card_roots([tmp, *scene.roots]), assets=assets_root())
        report = check(str(path), worlds=1, seconds=seconds, device="cpu", library=lib, budget_s=None)
    errors = [f for f in report.findings if f.level == "error"]
    return None if report.ok else (errors[0].message if errors else "check not ok")


def write(out: Path, variants: list[Variant], opts: VaryOptions, plan: LookPlan, base: Base, placer: Placer,
          sheet: dict | None) -> Path:
    """Write the variant JSONs, the manifest and (``sheet`` not None) the contact sheet; returns the manifest."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, v in enumerate(variants):
        name = f"variant_{k:03d}.json"
        (out / name).write_text(v.scene.to_json())
        rows.append({"file": name, "draws": v.draws, "textures": v.textures, "look_draws": v.look_draws,
                     "look_rule": v.look_rule,
                     "moves": {n: {"dx": mv.offset[0], "dy": mv.offset[1], "yaw_deg": math.degrees(mv.yaw)}
                               for mv in v.moves for n in mv.names}})
    manifest = {"schema": SCHEMA, "n": len(variants), "seed": opts.seed, "pos": opts.pos, "yaw_deg": opts.yaw_deg,
                "textures": opts.textures, "tint": opts.tint, "settle": opts.settle, "contrast": opts.contrast,
                "max_albedo": opts.max_albedo, "groups": placer.groups,
                "surfaces": {"large": list(plan.surfaces.large), "pairs": [list(p) for p in plan.surfaces.pairs],
                             "floors": list(plan.surfaces.floors)},
                "families": {f: [t.ref for t in ts] for f, ts in plan.pool.items()}, "variants": rows}
    if sheet is not None:
        manifest["contact_sheet"] = _sheet(out, variants, base, placer, sheet)
    path = out / "variants.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path


def _sheet(out: Path, variants: list[Variant], base: Base, placer: Placer, cfg: dict) -> dict:
    size = (int(cfg.get("width", 240)), int(cfg.get("height", 180)))
    spp, quality = int(cfg.get("spp", 4)), str(cfg.get("quality", "full"))
    look_at = SH.view(base, placer.state_for([]), [n for g in placer.groups for n in g])
    images, paths, dropped = [], [], set()
    for v in variants:
        look = None
        if quality == "full":
            look, left_out = SH.drawable(L.scene_look(v.scene, [SH.SUN]))
            dropped.update(left_out)
        img, used = SH.render_tile(base, placer.state_for(v.moves), look, look_at, size, spp)
        images.append(img)
        paths.append(used)
    SH.write_sheet(out / "contact_sheet.png", images)
    return {"file": "contact_sheet.png", "tiles": paths, "size": list(size), "spp": spp,
            "left_out": sorted(dropped)}


__all__ = ["LOOK_TRIES", "VaryOptions", "Variant", "scene_look", "vary", "write"]
