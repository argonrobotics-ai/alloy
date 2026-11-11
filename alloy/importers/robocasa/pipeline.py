# SPDX-License-Identifier: Apache-2.0
"""The per-asset pipeline: extract, import, card, validate, physics check, thumbnail, record.

``import_asset`` runs every stage for one asset and returns a JSON-able record (also written to
``<out>/records/<name>.json``); it never raises for a bad asset: the record's ``status`` is ``ok``, ``flagged``
(imported, but a plausibility flag or check failed) or ``failed`` (with ``error``). Status rules:

- ``failed``: the import raised, or the card has a validation error.
- ``flagged``: a mass/size plausibility flag, a failed physics check, or no thumbnail.
- ``pending``: imported with ``validate=False`` or ``thumbs=False`` (a batch machine without the engine or GL);
  ``complete_record`` (CLI ``post``) validates the card, renders the thumbnail and sets the final status.

Example::

    >>> from alloy.importers.robocasa.pipeline import import_asset
    >>> rec = import_asset("objects_lightwheel", "glass_cup/GlassCup003", Path("imports/robocasa"))  # doctest: +SKIP
    >>> rec["status"], rec["mass_kg"]                                                                # doctest: +SKIP
    ('ok', 0.2759)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import shutil
import tempfile
import traceback
from pathlib import Path

from alloy.cards import CardLibrary, default_root
from alloy.importers.robocasa import cards_out, naming
from alloy.importers.robocasa.checks import CheckResult, drop_test, fixture_test
from alloy.importers.robocasa.fixtures import import_fixture
from alloy.importers.robocasa.materials import default_table
from alloy.importers.robocasa.objects import import_object
from alloy.importers.robocasa.packs import require_kept
from alloy.importers.robocasa.source import PackReader

log = logging.getLogger("alloy.importers.robocasa")

RECORDS = "records"
THUMBS = "thumbs"


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_table(root: Path, rels: list[str]) -> dict[str, dict]:
    """rel path -> {sha256, bytes} for files that exist under ``root``."""
    out = {}
    for rel in sorted(rels):
        p = root / rel
        if p.is_file():
            out[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def _import(kind: str, folder: Path, asset: str, out: Path, header: str, hulls_from: str = "coacd"):
    if kind == "object":
        return import_object(folder, asset, out, header=header, hulls_from=hulls_from)
    return import_fixture(folder, asset, out, header=header)


def _header_lines(result) -> list[str]:
    lines = [f"Mass: {result.mass_kg:.4g} kg. {result.mass_note}."]
    if result.flags:
        lines.append("Flags: " + "; ".join(result.flags) + ".")
    return lines


def _card_issues(out: Path, name: str, lib: CardLibrary | None = None) -> list[dict]:
    lib = lib or CardLibrary([default_root(), out])
    return [dataclasses.asdict(i) for i in lib.object_issues(name)]


def _check(kind: str, model: Path) -> CheckResult:
    return drop_test(model) if kind == "object" else fixture_test(model)


def _thumbnail(model: Path, out: Path, rel: str) -> str | None:
    from alloy.importers.robocasa.thumbs import render_thumbnail  # noqa: PLC0415 - GL setup only when used

    try:
        render_thumbnail(model, out / rel)
        return rel
    except Exception as exc:  # noqa: BLE001 - no GL backend must not fail the import
        log.warning("thumbnail of %s failed: %s", model, exc)
        return None


def _status(record: dict) -> str:
    if record.get("error") or any(i["level"] == "error" for i in record.get("card_issues") or []):
        return "failed"
    if record.get("card_issues") is None or record.get("thumbnail") == "pending":
        return "pending"
    if record["flags"] or not record["check"]["ok"] or not record.get("thumbnail"):
        return "flagged"
    return "ok"


def _files(out: Path, rec: dict) -> dict[str, dict]:
    model_dir = out / Path(rec["model"]).parent
    rels = [str(p.relative_to(out)) for p in model_dir.rglob("*") if p.is_file()]
    thumb = rec.get("thumbnail")
    return file_table(out, rels + [rec["card"]] + ([thumb] if thumb and thumb != "pending" else []))


def _thumb_rel(rec: dict) -> str:
    return f"{THUMBS}/{naming.category(rec['asset'])}/{rec['name']}.png"


def _record(pack, asset: str, kind: str, result, out: Path, validate: bool = True, thumbs: bool = True) -> dict:
    """Card, checks and thumbnail for an imported asset, gathered into the record."""
    pack_obj = require_kept(pack)
    motion = "dynamic" if kind == "object" else "static"
    card = cards_out.object_card(asset, result.surface, motion)
    header = cards_out.provenance_header(asset, pack_obj, _header_lines(result))
    cards_out.write_object_card(out, asset, card, header)
    cards_out.write_material_cards(out, default_table(), {result.surface})
    rec = {k: v for k, v in dataclasses.asdict(result).items() if k != "inertial"}
    rec.update(pack=pack, kind=kind, licence=pack_obj.licence, source_url=pack_obj.source_url,
               card=naming.card_rel(asset), card_issues=_card_issues(out, result.name) if validate else None)
    rec["check"] = dataclasses.asdict(_check(kind, out / result.model))
    rec["thumbnail"] = _thumbnail(out / result.model, out, _thumb_rel(rec)) if thumbs else "pending"
    rec["files"] = _files(out, rec)
    return rec


def complete_record(out: Path, rec: dict, lib: CardLibrary | None = None) -> dict:
    """Validate the card and render the thumbnail of a ``pending`` record (or one missing a thumbnail); rewrites
    the record file and returns it. Records that failed to import are returned unchanged. Pass ``lib`` (a library
    over ``out``) when completing many records, so the card index is built once."""
    if rec.get("error") or not rec.get("model"):
        return rec
    if rec.get("card_issues") is None:
        rec["card_issues"] = _card_issues(out, rec["name"], lib)
    if rec.get("thumbnail") in (None, "pending"):
        rec["thumbnail"] = _thumbnail(out / rec["model"], out, _thumb_rel(rec))
    rec["files"] = _files(out, rec)
    rec["status"] = _status(rec)
    (out / RECORDS / f"{rec['name']}.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    return rec


def import_asset(pack: str, asset: str, out: Path, location: str | Path | None = None,
                 work: Path | None = None, validate: bool = True, thumbs: bool = True, hulls_from: str = "coacd") -> dict:
    """Run the whole pipeline for one asset (see module doc); writes and returns its record."""
    kind = require_kept(pack).kind
    name = naming.card_name(asset)
    tmp = Path(tempfile.mkdtemp(prefix="robocasa_", dir=work))
    try:
        reader = PackReader.open(pack, location)
        folder = reader.extract(asset, tmp)
        result = _import(kind, folder, asset, out, f"{name}: generated by alloy.importers.robocasa from RoboCasa "
                                                   f"{pack}/{asset} ({reader.pack.licence})", hulls_from)
        rec = _record(pack, asset, kind, result, out, validate, thumbs)
    except Exception as exc:  # noqa: BLE001 - one bad asset must not stop a batch; the record says why
        log.error("import of %s/%s failed: %s", pack, asset, exc)
        rec = {"name": name, "asset": asset, "pack": pack, "kind": kind, "flags": [], "check": {"ok": False},
               "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rec["status"] = _status(rec)
    path = out / RECORDS / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
    return rec


def material_names(records: list[dict]) -> set[str]:
    """Surface materials the records use (for writing the new material cards)."""
    return {r["surface"] for r in records if r.get("surface")}


def write_materials(out: Path, records: list[dict]) -> list[Path]:
    """The new ``robocasa_*`` material cards the imported cards point at."""
    return cards_out.write_material_cards(out, default_table(), material_names(records))


__all__ = ["RECORDS", "THUMBS", "complete_record", "file_table", "import_asset", "material_names", "sha256", "write_materials"]
