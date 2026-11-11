# SPDX-License-Identifier: Apache-2.0
"""Handler of the ``vary`` operation: ``n`` variants of one scene (free objects moved, textures swapped within their
family), written as Scene JSON files that every other operation takes as ``scene``, plus an optional contact sheet.

It wraps ``alloy.scene.variants.vary`` (WP2; imported on first use). Example::

    from alloy.cli.session import Session
    from alloy.cli.ops_vary import op_vary
    op_vary(Session(), {"scene": "demo_peg_d12", "n": 4, "pos": 0.01, "yaw": 10.0, "textures": "same-family",
                        "tint": 0.05, "seed": 0, "settle": None, "max_tries": 200, "out": "variants",
                        "sheet": True, "sheet_options": None, "library": None})
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from alloy.cli.errors import RequestError
from alloy.cli.scenes import to_scene
from alloy.cli.session import Session

log = logging.getLogger("alloy.cli")

TEXTURE_MODES = ("same-family", "textured", "keep")      # alloy.scene.variants.looks.MODES
SHEET_DEFAULTS = {"width": 240, "height": 180, "spp": 4, "quality": "full"}
SHEET_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"width": {"type": "integer", "minimum": 16, "description": "tile width, pixels"},
                   "height": {"type": "integer", "minimum": 16, "description": "tile height, pixels"},
                   "spp": {"type": "integer", "minimum": 1, "description": "samples per pixel"},
                   "quality": {"enum": ["full", "fast"], "description": "full: materials and textures; fast: "
                                                                        "shape colours only"}}}


def op_vary(session: Session, a: dict[str, Any]) -> dict[str, Any]:
    """Write ``n`` variants of ``a["scene"]`` to ``a["out"]`` (a new temporary folder when unset) and return their
    files, moves and textures; ``sheet`` adds ``contact_sheet.png`` (one tile per variant, ``alloy.camera``)."""
    from alloy.scene.variants import vary  # noqa: PLC0415

    scene = to_scene(a["scene"], library=a["library"] or ())
    out = Path(a["out"]).expanduser() if a["out"] else Path(tempfile.mkdtemp(prefix="alloy-vary-"))
    sheet = sheet_settings(a["sheet_options"]) if a["sheet"] else None
    vary(scene, a["n"], pos=a["pos"], yaw_deg=a["yaw"], textures=a["textures"], seed=a["seed"], tint=a["tint"],
         out=out, contact_sheet=sheet is not None, sheet=sheet, max_tries=a["max_tries"],
         settle=a["settle"] or None)                      # settle 0: no check run
    log.info("vary: %d variants in %s", a["n"], out)
    return summary(out)


def sheet_settings(options: dict[str, Any] | None) -> dict[str, Any]:
    """Contact-sheet settings: the defaults updated by ``options`` (keys of ``SHEET_DEFAULTS`` only).

    Example: ``sheet_settings({"quality": "fast"})["width"] == 240``."""
    options = options or {}
    unknown = sorted(set(options) - set(SHEET_DEFAULTS))
    if unknown:
        raise RequestError(f"vary: unknown sheet_options {unknown}", f"use only {sorted(SHEET_DEFAULTS)}, e.g. "
                           '{"width": 240, "height": 180, "spp": 4, "quality": "fast"}', field="sheet_options")
    merged = {**SHEET_DEFAULTS, **options}
    if merged["quality"] not in ("full", "fast"):
        raise RequestError(f"vary: sheet_options.quality must be 'full' or 'fast', got {merged['quality']!r}",
                           'pass "quality": "fast" (shape colours) or "full" (textures)', field="sheet_options")
    for key in ("width", "height", "spp"):
        value, low = merged[key], 1 if key == "spp" else 16
        if not isinstance(value, int) or isinstance(value, bool) or value < low:
            raise RequestError(f"vary: sheet_options.{key} must be an integer >= {low}, got {value!r}",
                               f'pass "{key}": {SHEET_DEFAULTS[key]}', field="sheet_options")
    return merged


def summary(out: Path) -> dict[str, Any]:
    """The reply of ``vary`` from the folder it wrote: absolute file paths, groups, texture families (counts) and per
    variant its draws, moves and textures. Example: ``summary(Path("variants"))["files"][0]``."""
    manifest = json.loads((out / "variants.json").read_text())
    root = out.resolve()
    rows = [{**row, "file": str(root / row["file"])} for row in manifest["variants"]]
    sheet = manifest.get("contact_sheet")
    return {"n": manifest["n"], "seed": manifest["seed"], "out": str(root), "manifest": str(root / "variants.json"),
            "files": [row["file"] for row in rows], "groups": manifest["groups"],
            "families": {fam: len(refs) for fam, refs in manifest["families"].items()},
            "variants": rows, "sheet": str(root / sheet["file"]) if sheet else None,
            "sheet_left_out": sheet["left_out"] if sheet else []}
