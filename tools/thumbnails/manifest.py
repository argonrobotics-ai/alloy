# SPDX-License-Identifier: Apache-2.0
"""Thumbnail manifest of a library (``thumbs.json``, schema ``alloy.thumbs/1``) beside its ``manifest.json``.

Per object: the thumbnail's path (library-relative), sha256 and bytes (what the library's own manifest and asset store
need), how it was rendered (textured, glass shown opaque, simplifications) and ``showcase_ok``. Showcase images in docs,
READMEs, examples and the launch page use unbranded objects only: a scan or model whose name or label shows a
third-party brand stays in the library but gets ``showcase_ok: false`` with the reason (``showcase.json``: exact names
and name prefixes).

Example::

    m = build(Path("imports/ycb_gso"), "ycb_gso", rows, load_showcase(SHOWCASE))
    m["objects"]["ycb_003_cracker_box"]["showcase_ok"]           # False ("Cheez-It packaging")
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "alloy.thumbs/1"
SHOWCASE = Path(__file__).resolve().parent / "showcase.json"


def load_showcase(path: Path = SHOWCASE) -> dict[str, dict[str, str]]:
    """``{"names": {name: reason}, "prefixes": {prefix: reason}}`` of objects not to show publicly."""
    d = json.loads(Path(path).read_text())
    return {"names": dict(d.get("names", {})), "prefixes": dict(d.get("prefixes", {}))}


def showcase(name: str, rules: dict[str, dict[str, str]]) -> tuple[bool, str]:
    """(showcase_ok, reason): False with the reason when the name is listed or starts with a listed prefix."""
    if name in rules["names"]:
        return False, rules["names"][name]
    hit = max((p for p in rules["prefixes"] if name.startswith(p)), key=len, default=None)
    return (False, rules["prefixes"][hit]) if hit else (True, "")


def _entry(root: Path, row: dict[str, Any], rules: dict[str, dict[str, str]]) -> dict[str, Any]:
    thumb = Path(row["thumb"])
    ok, why = showcase(row["name"], rules)
    e = {"thumb": str(thumb.relative_to(root)) if thumb.is_relative_to(root) else thumb.name,
         "sha256": row["sha256"], "bytes": thumb.stat().st_size if thumb.is_file() else None,
         "textured": bool(row.get("textured")), "glass_shown_opaque": bool(row.get("glass")),
         "simplified": row.get("notes") or [], "showcase_ok": ok}
    if why:
        e["showcase_note"] = why
    return e


def build(root: Path, lib: str, rows: dict[str, dict], rules: dict[str, dict[str, str]],
          settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """The manifest of every ``ok`` row (``batch.read_state``); failed objects are listed with their reasons."""
    root = Path(root).resolve()
    ok = {k: _entry(root, r, rules) for k, r in sorted(rows.items()) if r.get("status") == "ok"}
    return {"schema": SCHEMA, "lib": lib, "renderer": settings or {}, "count": len(ok),
            "showcase_ok": sum(e["showcase_ok"] for e in ok.values()), "objects": ok,
            "failed": {k: r.get("reason", "") for k, r in sorted(rows.items()) if r.get("status") != "ok"}}


def update_library_manifest(root: Path, thumbs: dict[str, Any]) -> int:
    """Set each thumbnail's sha256 and bytes in the library's own ``manifest.json`` (the thumbnail entries only; every
    other field is left as it is, and key order is kept). Returns how many entries changed."""
    path = Path(root) / "manifest.json"
    text = path.read_text()
    man = json.loads(text)
    shared = "objects" in man
    table = man["objects"] if shared else man["assets"]
    changed = 0
    for name, e in thumbs["objects"].items():
        files = table.get(name, {}).get("files")
        new = {"bytes": e["bytes"], "sha256": e["sha256"]}
        key = "thumb.png" if shared else e["thumb"]
        if files is None or files.get(key) == new:
            continue
        files.setdefault(key, {}).update(new)          # keeps the entry's key order
        changed += 1
    indent = len(text.split("\n", 2)[1]) - len(text.split("\n", 2)[1].lstrip(" ")) if "\n" in text else 1
    path.write_text(json.dumps(man, indent=indent or 1) + ("\n" if text.endswith("\n") else ""), encoding="utf-8")
    return changed


def write(root: Path, manifest: dict[str, Any]) -> Path:
    """Write ``<root>/thumbs.json`` (sorted keys, one object per line block)."""
    path = Path(root) / "thumbs.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return path
