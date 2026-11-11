# SPDX-License-Identifier: Apache-2.0
"""Batch import (one asset per process, in parallel) and the finishing step that writes the manifest and reports.

Each asset runs in its own ``python -m alloy.importers.robocasa import`` process with a time limit, so a mesh that
blows up memory or hangs CoACD costs only itself. ``finalize`` merges ``records/*.json`` into ``manifest.json``
(every file with its SHA-256), ``report.json`` (counts by status and category), ``sourcing_queue.jsonl`` (values that
are estimates or flagged, for the sourcing agent) and ``ATTRIBUTION.md``.

Example::

    >>> from alloy.importers.robocasa.batch import run_batch, finalize
    >>> run_batch("objects_lightwheel", ["glass_cup/GlassCup003"], Path("out"), jobs=4)   # doctest: +SKIP
    {'ok': 1, 'flagged': 0, 'failed': 0}
    >>> finalize(Path("out"))["assets"]                                                  # doctest: +SKIP
    1
"""

from __future__ import annotations

import collections
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alloy.importers.robocasa import naming
from alloy.importers.robocasa.packs import PACKS, kept_packs
from alloy.importers.robocasa.pipeline import RECORDS, write_materials

log = logging.getLogger("alloy.importers.robocasa")

TIMEOUT_S = 1800


def _one(pack: str, asset: str, out: Path, location: str | None, timeout: float, extra: tuple[str, ...]) -> str:
    """Run one asset in a child process; returns its status (``failed`` on timeout or crash)."""
    cmd = [sys.executable, "-m", "alloy.importers.robocasa", "import", "--pack", pack, "--asset", asset,
           "--out", str(out), *extra]
    if location:
        cmd += ["--zip", str(location)]
    record = out / RECORDS / f"{naming.card_name(asset)}.json"
    for attempt in (cmd, cmd + ["--hulls", "given"]):     # a CoACD crash kills the process: retry on given hulls
        try:
            subprocess.run(attempt, check=False, timeout=timeout, capture_output=True, env=os.environ.copy())
        except subprocess.TimeoutExpired:
            _write_failure(record, pack, asset, f"timed out after {timeout:g} s")
        if record.is_file():
            break
    if not record.is_file():
        _write_failure(record, pack, asset, "the import process died without a record (also with given hulls)")
    return json.loads(record.read_text()).get("status", "failed")


def _write_failure(record: Path, pack: str, asset: str, error: str) -> None:
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"name": record.stem, "asset": asset, "pack": pack, "status": "failed",
                                  "error": error, "flags": []}, indent=1))


def _done(out: Path, asset: str) -> bool:
    """True if the asset already has a record that is not ``failed`` (failed ones are retried)."""
    path = out / RECORDS / f"{naming.card_name(asset)}.json"
    return path.is_file() and json.loads(path.read_text()).get("status") != "failed"


def run_batch(pack: str, assets: list[str], out: Path, location: str | None = None, jobs: int = 4,
              timeout: float = TIMEOUT_S, skip_done: bool = True, validate: bool = True,
              thumbs: bool = True) -> dict[str, int]:
    """Import ``assets`` with ``jobs`` processes; with ``skip_done`` an asset whose record exists is not redone.
    ``validate``/``thumbs`` False leave records ``pending`` for ``post_pass`` (machines without the engine/GL)."""
    extra = (() if validate else ("--no-validate",)) + (() if thumbs else ("--no-thumbs",))
    todo = [a for a in assets if not (skip_done and _done(out, a))]
    log.info("batch %s: %d assets (%d already done), %d jobs", pack, len(todo), len(assets) - len(todo), jobs)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        statuses = list(pool.map(lambda a: _one(pack, a, out, location, timeout, extra), todo))
    return dict(collections.Counter(statuses))


def post_pass(out: Path) -> dict[str, int]:
    """Complete every pending record (card validation, thumbnail) in this process; returns status counts."""
    from alloy.cards import CardLibrary, default_root  # noqa: PLC0415
    from alloy.importers.robocasa.pipeline import complete_record, write_materials  # noqa: PLC0415

    records = load_records(out)
    write_materials(out, records)                      # cards must resolve their robocasa_* materials
    lib = CardLibrary([default_root(), out])
    done = [complete_record(out, r, lib) if r.get("status") in ("pending", "flagged") else r for r in records]
    return dict(collections.Counter(r.get("status", "failed") for r in done))


def load_records(out: Path) -> list[dict]:
    """Every record under ``out/records``, sorted by card name."""
    return [json.loads(p.read_text()) for p in sorted((out / RECORDS).glob("*.json"))]


def _queue_rows(rec: dict) -> list[dict]:
    """Sourcing-queue rows for one record: its estimated mass, plus any flag."""
    if rec.get("status") == "failed":
        return []
    rows = [{"card": rec["name"], "field": "mass", "value": rec.get("mass_kg"), "unit": "kg",
             "category": rec.get("category") or rec.get("kind"), "why": "ESTIMATE", "note": rec.get("mass_note")}]
    rows += [{"card": rec["name"], "field": "flag", "category": rec.get("category"), "why": f} for f in rec["flags"]]
    return rows


def _attribution(records: list[dict]) -> str:
    used = sorted({r["pack"] for r in records if r.get("status") != "failed"})
    lines = ["# Attribution", "", "Assets in this folder are derived from RoboCasa asset packs and carry their licence.", ""]
    for key in used:
        p = PACKS[key]
        n = sum(1 for r in records if r["pack"] == key and r.get("status") != "failed")
        lines += [f"- **{key}** ({n} assets): {p.licence}. {p.attribution}. Source: {p.source_url}"]
    lines += ["", "Not imported: " + "; ".join(f"{p.key} ({p.decision}: {p.reason})" for p in PACKS.values()
                                               if p not in kept_packs()) + "."]
    return "\n".join(lines) + "\n"


def finalize(out: Path) -> dict:
    """Write manifest.json, report.json, sourcing_queue.jsonl, ATTRIBUTION.md and the material cards."""
    records = load_records(out)
    ok = [r for r in records if r.get("status") != "failed"]
    manifest = {"schema": "alloy.import_manifest/1", "source": "robocasa",
                "assets": {r["name"]: {k: r.get(k) for k in ("asset", "pack", "kind", "status", "card", "model",
                                                          "licence", "files")} for r in ok}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    by_status = collections.Counter(r.get("status", "failed") for r in records)
    by_cat = collections.defaultdict(collections.Counter)
    for r in records:
        by_cat[naming.category(r["asset"])][r.get("status", "failed")] += 1
    report = {"assets": len(records), "status": dict(by_status), "by_category": {k: dict(v) for k, v in by_cat.items()},
              "failed": {r["name"]: r.get("error") for r in records if r.get("status") == "failed"}}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    with open(out / "sourcing_queue.jsonl", "w") as fh:
        for r in ok:
            for row in _queue_rows(r):
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    (out / "ATTRIBUTION.md").write_text(_attribution(records))
    write_materials(out, ok)
    return report


__all__ = ["finalize", "load_records", "post_pass", "run_batch"]
