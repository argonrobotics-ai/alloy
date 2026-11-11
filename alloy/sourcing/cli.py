# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.sourcing``: JSON in, JSON out.

Commands (flags, or one JSON request with ``--request '{"cmd": "queue", "library": "ycb_gso"}'``):

- ``queue [--library L] [--weak] [--limit N]``: open items (ESTIMATE / MISSING; ``--weak`` adds SNIPPET, UNVERIFIED).
- ``fill [--library L] [--dry-run] [--fuzzy]``: write per-object records from the category cache (name maps and
  exact category keys; ``--fuzzy`` also matches aliases inside object names).
- ``coverage [--write]``: sourced / weak / estimated / missing shares; ``--write`` saves coverage.json + COVERAGE.md.
- ``lookup NAME [--kind category|material]``: the cache entry that matches a name.
- ``validate PATH...``: check cache entries and object records; exit 1 if any is invalid.
- ``ingest FILE --library L [--dry-run]``: merge per-object findings (JSON lines) into records; a finding lands only
  if its tag is stronger than the record's.
- ``verify [PATH...]``: re-fetch every DOC page and check its quote (default: the whole cache); exit 1 if a quote
  is missing from its page. PDFs and blocked pages are listed as unchecked.

Every command takes ``--assets PATH`` (default ``$ALLOY_ASSETS`` or ``../alloy-assets``) for the importer cards and
``--sources PATH`` for the cache and records (default ``<assets>/sources/imports``).

Example:
    >>> import json, tempfile
    >>> out = run({"cmd": "queue", "assets": tempfile.mkdtemp()})
    >>> out["ok"], out["open"]
    (True, 0)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from alloy.sourcing.cache import Cache, entry_from_dict, entry_to_dict
from alloy.sourcing.coverage import summarize, write_report
from alloy.sourcing.fields import SourcingError, yaml_load
from alloy.sourcing.ingest import ingest_file
from alloy.sourcing.match import fill_from_cache
from alloy.sourcing.paths import sources_root
from alloy.sourcing.queue import scan_all
from alloy.sourcing.store import record_from_dict
from alloy.sourcing.verify import to_json as verify_json
from alloy.sourcing.verify import verify_files


def _assets(req: dict[str, Any]) -> Path | None:
    return Path(req["assets"]) if req.get("assets") else None


def _sources(req: dict[str, Any]) -> Path:
    return Path(req["sources"]) if req.get("sources") else sources_root(_assets(req))


def _libs(req: dict[str, Any]) -> list[str] | None:
    return [req["library"]] if req.get("library") else None


def cmd_queue(req: dict[str, Any]) -> dict[str, Any]:
    """Open items across libraries."""
    scans = scan_all(_assets(req), _libs(req), _sources(req))
    items = [i for s in scans.values() for i in s.open(bool(req.get("weak")))]
    limit = int(req.get("limit") or 0)
    by_field = Counter(i.field for i in items)
    return {"ok": True, "open": len(items), "by_field": dict(sorted(by_field.items())),
            "items": [i.to_json() for i in (items[:limit] if limit else items)],
            "skipped": [s for sc in scans.values() for s in sc.skipped]}


def cmd_fill(req: dict[str, Any]) -> dict[str, Any]:
    """Records from the cache for every open item a category covers."""
    sources = _sources(req)
    report = {}
    cache = Cache(sources)
    for lib, scan in scan_all(_assets(req), _libs(req), sources).items():
        rep = fill_from_cache(scan, cache, sources, bool(req.get("dry_run")), bool(req.get("fuzzy")))
        report[lib] = rep.to_json()
        if not req.get("dry_run"):
            path = sources / "review" / f"{lib}_mass_disagreements.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rep.flags, indent=1) + "\n", encoding="utf-8")
            report[lib]["flags_file"] = str(path)
    return {"ok": True, "libraries": report}


def cmd_coverage(req: dict[str, Any]) -> dict[str, Any]:
    """Coverage summary; optionally written next to the records."""
    cov = summarize(scan_all(_assets(req), _libs(req), _sources(req)))
    if req.get("write"):
        js, md = write_report(cov, _sources(req))
        cov["written"] = [str(js), str(md)]
    return {"ok": True, **cov}


def cmd_lookup(req: dict[str, Any]) -> dict[str, Any]:
    """The matching cache entry, or ok=false with what to do."""
    cache = Cache(_sources(req))
    kind = req.get("kind") or "category"
    entry = cache.get(req["name"], kind) or cache.find(req["name"], kind)
    if entry is None:
        return {"ok": False, "error": f"no {kind} entry matches {req['name']!r}; source one and add it to the cache"}
    return {"ok": True, "entry": entry_to_dict(entry)}


def validate_file(path: Path) -> None:
    """Parse one cache entry or object record by its ``schema``; raises ``SourcingError`` with what to change."""
    raw = yaml_load(path.read_text(encoding="utf-8"))
    schema = raw.get("schema") if isinstance(raw, dict) else None
    if schema == "alloy.sourcing.category/1":
        entry = entry_from_dict(raw, str(path))
        if path.stem != entry.key:
            raise SourcingError(f"{path}: file name must be {entry.key}.yaml")
    elif schema == "alloy.sourcing.object/1":
        rec = record_from_dict(raw, str(path))
        if path.stem != rec.object:
            raise SourcingError(f"{path}: file name must be {rec.object}.yaml")
    else:
        raise SourcingError(f"{path}: unknown schema {schema!r}; use alloy.sourcing.category/1 or object/1")


def cmd_validate(req: dict[str, Any]) -> dict[str, Any]:
    """Validate files; ok=false lists each error."""
    errors = []
    paths = [Path(p) for p in req.get("paths") or ()]
    for p in paths:
        try:
            validate_file(p)
        except (SourcingError, OSError, yaml.YAMLError) as e:
            errors.append({"file": str(p), "error": str(e)})
    return {"ok": not errors, "checked": len(paths), "errors": errors}


def _yaml_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend(sorted(p.rglob("*.yaml")) if p.is_dir() else [p])
    return out


def cmd_verify(req: dict[str, Any]) -> dict[str, Any]:
    """Check DOC quotes against their pages; ok=false if any quote is missing from its page."""
    paths = [Path(p) for p in req.get("paths") or ()] or [_sources(req) / "cache"]
    result = verify_json(verify_files(_yaml_files(paths)))
    return {"ok": not result["missing"], **result}


def cmd_ingest(req: dict[str, Any]) -> dict[str, Any]:
    """Merge findings files into one library's records; ok=false if any line was rejected."""
    if not req.get("library"):
        raise SourcingError("ingest needs --library (the importer folder name, e.g. ycb_gso)")
    reports = {p: ingest_file(Path(p), req["library"], _sources(req), bool(req.get("dry_run"))).to_json()
               for p in req.get("paths") or ()}
    return {"ok": not any(r["rejected"] for r in reports.values()), "files": reports}


COMMANDS = {"queue": cmd_queue, "ingest": cmd_ingest, "fill": cmd_fill, "coverage": cmd_coverage, "lookup": cmd_lookup,
            "validate": cmd_validate, "verify": cmd_verify}


def run(req: dict[str, Any]) -> dict[str, Any]:
    """Run one request dict (``cmd`` plus that command's keys); errors come back as ok=false, never raised."""
    cmd = req.get("cmd")
    if cmd not in COMMANDS:
        return {"ok": False, "error": f"unknown cmd {cmd!r}; use one of {sorted(COMMANDS)}"}
    try:
        return COMMANDS[cmd](req)
    except SourcingError as e:
        return {"ok": False, "error": str(e)}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alloy.sourcing", description=__doc__.split("\n")[0])
    p.add_argument("cmd", nargs="?", choices=sorted(COMMANDS))
    p.add_argument("args", nargs="*", help="lookup: NAME; validate / verify / ingest: PATH...")
    p.add_argument("--request", help="one JSON request instead of flags")
    p.add_argument("--assets"), p.add_argument("--sources"), p.add_argument("--library"), p.add_argument("--kind")
    p.add_argument("--limit", type=int), p.add_argument("--weak", action="store_true")
    p.add_argument("--dry-run", action="store_true"), p.add_argument("--write", action="store_true")
    p.add_argument("--fuzzy", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse flags or ``--request`` JSON, print the JSON result, exit 0 if ok."""
    ns = _parser().parse_args(argv)
    if ns.request:
        req = json.loads(ns.request)
    else:
        req = {k: v for k, v in vars(ns).items() if k not in ("args", "request") and v not in (None, False)}
        if ns.cmd == "lookup" and ns.args:
            req["name"] = " ".join(ns.args)
        if ns.cmd in ("validate", "verify", "ingest"):
            req["paths"] = ns.args
    out = run(req)
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if out.get("ok") else 1
