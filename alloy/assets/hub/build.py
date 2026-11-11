# SPDX-License-Identifier: Apache-2.0
"""Stage an imported object library as dataset files (layout: ``alloy.assets.hub.layout``), ready for upload.

Maintainer tool. Cards, material cards, looks and manifests come from the import root (a checkout); binaries come from
a local copy of the asset store (``<store>/<library>/<path in the root>``). Every file is checked against the
manifest's sha256, then hard-linked (or copied) into ``<out>/objects/<library>/<name>/``. One index row per object is
appended to ``--rows`` (JSON lines), and the library's shared files go to ``<out>/libraries/<library>/``.

Example::

    python -m alloy.assets.hub.build stage --library ycb_gso --root imports/ycb_gso --store store --out ds \\
        --rows rows/ycb_gso.jsonl              # prints {"library": "ycb_gso", "objects": 1111, "bytes": ...}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alloy.assets.hub.layout import (LIBRARY_FILES, dataset_file, library_dir, object_dir, sha256_file,
                                     tree_digest)
from alloy.assets.hub.provenance import provenance

log = logging.getLogger("alloy.assets.hub")

# Root files the dataset carries for a library; build records (checks, queues, reports) stay out.
LIBRARY_KEEP = ("manifest.json", "thumbs.json", "ATTRIBUTION.md", "README.md")
LIBRARY_DIRS = ("materials", "looks")


class BuildError(RuntimeError):
    """A file is missing or differs from its manifest; the message names it and what to do."""


@dataclass
class ObjectSpec:
    """One object of an import root: its files (path in the root -> sha256, None when unlisted) and index fields."""
    name: str
    card: str
    home: str
    files: dict[str, str | None]
    material: str | None = None
    thumb: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    rename: dict[str, str] = field(default_factory=dict)   # path in the root -> name in the object folder


def _pack_lines(root: Path) -> dict[str, str]:
    """RoboCasa ATTRIBUTION.md: ``- **<pack>** (N assets): <licence>. <credit>`` -> {pack: credit}. The licence id
    has dots of its own (``CC-BY-4.0.``), so it is read as one word."""
    path = root / "ATTRIBUTION.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return {m[1]: m[2].strip() for m in re.finditer(r"^- \*\*([^*]+)\*\* \([^)]*\): \S+\. (.+)$", text, re.M)}


def _attribution(e: dict[str, Any], packs: dict[str, str]) -> str:
    if e.get("attribution"):
        return e["attribution"]
    if e.get("author"):
        return (f"\"{e.get('title', '')}\" by {e['author']} ({e.get('source', '')}), licensed {e.get('licence')}; "
                "changed for Alloy (rescaled, re-oriented, repaired, decimated)")
    return packs.get(e.get("pack", ""), "")


def _source(e: dict[str, Any], credit: str) -> str:
    """The upstream URL: the manifest entry's, else the ``Source: <url>`` its pack's credit line ends with."""
    m = re.search(r"Source: (https?://\S+)", credit)
    return e.get("source") or (m.group(1).rstrip(".") if m else "")


def _flags(card: Path) -> str:
    """The ``# Flags: ...`` line an importer writes into a card it imported with a plausibility or check flag."""
    text = card.read_text(encoding="utf-8") if card.is_file() else ""
    m = re.search(r"^# Flags: (.+)$", text, re.M)
    return m.group(1).strip() if m else "flagged by the importer (see the card header)"


def import_specs(root: Path, library: str) -> dict[str, ObjectSpec]:
    """Every object an import root's ``manifest.json`` lists (both importer layouts), with its thumbnail and
    showcase verdict from ``thumbs.json``. Example: ``len(import_specs(Path("imports/ycb_gso"), "ycb_gso"))``."""
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    thumbs_path = root / "thumbs.json"
    thumbs = json.loads(thumbs_path.read_text(encoding="utf-8"))["objects"] if thumbs_path.is_file() else {}
    packs = _pack_lines(root)
    specs: dict[str, ObjectSpec] = {}
    shared = isinstance(man.get("objects"), dict)
    for name, e in (man["objects"] if shared else man["assets"]).items():
        home = f"meshes/{name}" if shared else str(Path(e["model"]).parent)
        files = {(f"{home}/{f}" if shared else f): m["sha256"] for f, m in e["files"].items()}
        files.setdefault(e["card"], None)
        if e.get("material"):
            files.setdefault(e["material"], None)
        t = thumbs.get(name, {})
        thumb = t.get("thumb")                                   # relative to the root in both layouts
        folder = Path(e["card"]).parent.name
        credit = _attribution(e, packs)
        meta = {"category": e.get("category") if folder == library else folder, "title": e.get("title", name),
                "licence": e.get("licence"), "attribution": credit, "source": _source(e, credit),
                "showcase_ok": t.get("showcase_ok", True), "showcase_note": t.get("showcase_note", ""),
                "physics": e.get("physics") or {},
                "flags": _flags(root / e["card"]) if e.get("status", "ok") != "ok" else ""}
        specs[name] = ObjectSpec(name, e["card"], home, files, e.get("material"), thumb, meta)
    return specs


def _place(src: Path, dst: Path, rewrite: list[list[str]] | None = None) -> None:
    """Hard-link (or copy) ``src`` to ``dst``; with ``rewrite`` ([old, new] pairs), write the edited text instead."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if rewrite:
        text = src.read_text(encoding="utf-8")
        for old, new in rewrite:
            text = text.replace(old, new)
        dst.write_text(text, encoding="utf-8")
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _edits(rel: str, src: Path, rewrites: dict[str, list[list[str]]] | None) -> list[list[str]] | None:
    """The [old, new] pairs for ``rel``: its own (each must occur) plus the ``"*"`` ones that occur in it."""
    if not rewrites or src.suffix not in (".yaml", ".xml", ".json", ".md", ".txt"):
        return None
    text = src.read_text(encoding="utf-8")
    own = rewrites.get(rel, [])
    gone = [old for old, _ in own if old not in text]
    if gone:
        raise BuildError(f"{rel}: rewrite text not found (the source changed?): {gone[0][:80]!r}")
    edits = own + [e for e in rewrites.get("*", []) if e[0] in text]
    return edits or None


def _source_file(rel: str, digest: str | None, root: Path, store: Path) -> Path:
    for cand in (root / rel, store / rel):
        if cand.is_file() and (digest is None or sha256_file(cand) == digest):
            return cand
    raise BuildError(f"{rel}: not in the import root or the store copy with sha256 {digest}; refresh the store copy "
                     f"({store}) or rerun the library's importer")


def _write(rel: str, src: Path, dst: Path, rewrites: dict | None) -> dict[str, Any]:
    edits = _edits(rel, src, rewrites)
    _place(src, dst, edits)
    return {"path": rel, "sha256": sha256_file(dst), "bytes": dst.stat().st_size}


def stage_object(spec: ObjectSpec, library: str, root: Path, store: Path, out: Path,
                 rewrites: dict[str, list[list[str]]] | None = None) -> dict[str, Any]:
    """Copy one object's files into its dataset folder (``rewrites``: text edits per path, for an export that must
    differ from the source); return its index row. Raises ``BuildError`` on a bad file."""
    files = []
    for rel, digest in sorted(spec.files.items()):
        src = _source_file(rel, digest, root, store)
        name = spec.rename.get(rel) or dataset_file(rel, spec.home, spec.card, spec.material, spec.thumb)
        files.append({"file": name, **_write(rel, src, out / object_dir(library, spec.name) / name, rewrites)})
    names = [f["file"] for f in files]
    if len(set(names)) != len(names):
        raise BuildError(f"{spec.name}: two files map to one name in its folder: {sorted(names)}")
    m = spec.meta
    row = {"name": spec.name, "library": library, "category": m["category"], "title": m["title"],
           "licence": m["licence"], "attribution": m["attribution"], "source": m["source"],
           "showcase_ok": bool(m["showcase_ok"]), **({"showcase_note": m["showcase_note"]} if m["showcase_note"] else {}),
           **({"flags": m["flags"]} if m.get("flags") else {}),
           **provenance(root, spec.card, spec.material, m["physics"]),
           "size": sum(f["bytes"] for f in files), "sha256": tree_digest(files), "files": files}
    return row


def import_library_files(root: Path, owned: set[str]) -> list[str]:
    """An import root's shared files: manifest, thumbs.json, attribution, README, material cards and looks no object
    owns."""
    rels = [r for r in LIBRARY_KEEP if (root / r).is_file()]
    for d in LIBRARY_DIRS:
        rels += sorted(str(p.relative_to(root)) for p in (root / d).rglob("*") if p.is_file()) if (root / d).is_dir() else []
    return [r for r in rels if r not in owned and not Path(r).name.startswith(".")]


def stage_library_files(root: Path, library: str, rels: list[str], out: Path,
                        rewrites: dict[str, list[list[str]]] | None = None) -> dict[str, Any]:
    """Shared files to ``libraries/<library>/``, listed with sha256 in its ``files.json``.
    Example: ``stage_library_files(root, "robocasa", import_library_files(root, owned), out)``."""
    files = [{"file": rel, **_write(rel, root / rel, out / library_dir(library) / rel, rewrites)} for rel in rels]
    doc = {"library": library, "files": files, "sha256": tree_digest(files)}
    (out / library_dir(library)).mkdir(parents=True, exist_ok=True)
    (out / library_dir(library) / LIBRARY_FILES).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def stage(library: str, root: Path, store: Path, out: Path, rows_path: Path, names: set[str] | None = None,
          authored: tuple[dict[str, ObjectSpec], list[str]] | None = None,
          rewrites: dict[str, list[list[str]]] | None = None) -> dict:
    """Stage a library (or the ``names`` of it): objects, then shared files. ``authored``: (specs, shared files) of
    hand-authored objects instead of an import manifest. Appends rows; returns a summary."""
    specs, shared = authored if authored else (import_specs(root, library), None)
    if shared is None:
        shared = import_library_files(root, {rel for s in specs.values() for rel in s.files})
    todo = [s for n, s in sorted(specs.items()) if names is None or n in names]
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    total, failed = 0, {}
    with open(rows_path, "a", encoding="utf-8") as fh:
        for i, spec in enumerate(todo):
            try:
                row = stage_object(spec, library, root, store / library, out, rewrites)
            except (BuildError, OSError) as exc:
                failed[spec.name] = str(exc)[:300]
                continue
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            total += row["size"]
            if i % 250 == 0:
                log.info("%s %d/%d %s", library, i + 1, len(todo), spec.name)
    lib = stage_library_files(root, library, shared, out, rewrites)
    return {"library": library, "objects": len(todo) - len(failed), "bytes": total, "failed": failed,
            "library_files": len(lib["files"])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.assets.hub.build", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["stage", "stage-authored"])
    ap.add_argument("--library", required=True)
    ap.add_argument("--root", required=True, type=Path, help="the library's import root, or the card root (authored)")
    ap.add_argument("--store", type=Path, default=Path("."), help="local copy of the asset store (holds <library>/)")
    ap.add_argument("--out", required=True, type=Path, help="dataset folder to fill")
    ap.add_argument("--rows", required=True, type=Path, help="JSON-lines file the index rows are appended to")
    ap.add_argument("--names", type=Path, help="file with one object name per line (a shard); default all")
    ap.add_argument("--cards", type=Path, help="stage-authored: file with one card path (under --root) per line")
    ap.add_argument("--meta", type=Path, help="stage-authored: JSON {name: {category, title, licence, ...}}")
    ap.add_argument("--rewrite", type=Path, help="JSON {path: [[old, new], ...], '*': [...]}: text edits on export")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    names = set(a.names.read_text().split()) if a.names else None
    rewrites = json.loads(a.rewrite.read_text()) if a.rewrite else None
    authored = None
    if a.cmd == "stage-authored":
        from alloy.assets.hub.authored import authored_specs  # noqa: PLC0415

        meta = json.loads(a.meta.read_text()) if a.meta else {}
        authored = authored_specs(a.root, a.cards.read_text().split(), meta)
    res = stage(a.library, a.root, a.store, a.out, a.rows, names, authored, rewrites)
    json.dump(res, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if not res["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
