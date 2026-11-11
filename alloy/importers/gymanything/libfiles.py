# SPDX-License-Identifier: Apache-2.0
"""The library root's shared files (``alloy.importers.common`` formats): ``manifest.json`` (binaries by sha256),
``queue.jsonl`` (values still estimated, for the sourcing agent) and ``ATTRIBUTION.md`` (Gym-Anything's MIT notice).

Example::

    >>> write_library_files(Path("gar"), specs, records)           # doctest: +SKIP
    {'objects': 393, 'queue_rows': 512}
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from alloy.importers.common import manifest as mf
from alloy.importers.common import queue as qu
from alloy.importers.gymanything import convert as cv
from alloy.importers.gymanything.provenance import (UPSTREAM_COMMIT, UPSTREAM_LICENCE, UPSTREAM_URL, cite_ref)
from alloy.importers.gymanything.spec import Cited, GarSpec

log = logging.getLogger("alloy.importers.gymanything")

LIB = "gymanything"
ESTIMATE_BASES = ("estimate", "authored")
MIT_TEXT = """MIT License

Copyright (c) 2026 cmu-l3

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _source_url(spec: GarSpec) -> str:
    ref = cite_ref(spec.sim, spec.source_ref) if spec.source_ref else ""
    path = ref.split(":", 1)[1] if ":" in ref else ""
    path, _, line = path.rpartition(":") if path.rpartition(":")[2].split("-")[0].isdigit() else (path, "", "")
    anchor = f"#L{line.replace('-', '-L')}" if line else ""
    return f"{UPSTREAM_URL}/blob/{UPSTREAM_COMMIT}/{path}{anchor}" if path else UPSTREAM_URL


def manifest_entry(root: Path, spec: GarSpec, rec: cv.ImportRecord) -> dict[str, Any]:
    """One object's manifest entry: provenance, licence, card paths, binaries and physics summary."""
    d = root / "meshes" / spec.card_name
    files = {f.name: mf.file_entry(f) for f in sorted(d.glob("*")) if f.is_file()} if d.is_dir() else {}
    return {"library": LIB, "sim": spec.sim, "title": spec.name, "category": spec.category, "kind": spec.kind,
            "source": _source_url(spec), "licence": "MIT", "attribution": f"Gym-Anything for Robotics ({UPSTREAM_LICENCE})",
            "card": cv.object_rel(spec), "material": cv.material_rel(spec), "files": files,
            "physics": {"volume": rec.volume_m3, "mass_kg": rec.mass_kg, "mass_basis": rec.mass_basis,
                        "friction_basis": rec.friction_basis, "collision": rec.collision, "n_hulls": rec.n_hulls,
                        "hull_volume_ratio": rec.hull_volume_ratio, "convex_ratio": rec.convex_ratio,
                        "defaults": rec.defaults}}


def _row(spec: GarSpec, fld: str, value: float, c: Cited | None, query: str) -> qu.QueueRow:
    label = f"their label '{c.basis}' ({cite_ref(spec.sim, c.cite)}): {c.quote}" if c else "no value in their code"
    unc = c.unc if c and c.unc and fld != "material.density" else None
    return qu.QueueRow(spec.card_name, fld, float(value), unc, f"ESTIMATE: {label}"[:500], spec.category, query,
                       {"sim": spec.sim, "name": spec.name, "material": spec.material.name})


def queue_rows(spec: GarSpec, rec: cv.ImportRecord, density: float) -> list[qu.QueueRow]:
    """Rows for every value that is an estimate or unsourced: object mass (as the effective density), friction,
    restitution."""
    rows, done = [], set(rec.sourced)
    if spec.kind in ("rigid", "soft") and (spec.mass is None or spec.mass.basis in ESTIMATE_BASES) \
            and not done & {"mass", "density"}:
        rows.append(_row(spec, "material.density", density, spec.mass, f"{spec.name} weight kg"))
    if (spec.friction is None or spec.friction.basis in ESTIMATE_BASES) and "mu_static" not in done:
        partner = f" on {spec.friction_partner}" if spec.friction_partner else ""
        rows.append(_row(spec, "material.surface.mu_static", spec.friction.value if spec.friction else cv.DEFAULT_MU,
                         spec.friction, f"{spec.material.name}{partner} coefficient of friction"))
    if spec.restitution is None and "restitution" not in done:
        rows.append(_row(spec, "material.surface.restitution", cv.DEFAULT_RESTITUTION, None,
                         f"{spec.material.name} coefficient of restitution"))
    return rows


def attribution_text(sims: list[str]) -> str:
    return (f"# Attribution\n\nEvery card, material and mesh in this folder is derived from **Gym-Anything for "
            f"Robotics** ({UPSTREAM_URL}, commit `{UPSTREAM_COMMIT}`), {UPSTREAM_LICENCE}: object geometry exported by "
            f"running their builder code, and their cited values (each card note names the file and line).\n\n"
            f"Simulations used: {', '.join(sorted(sims))}.\n\nTheir licence:\n\n```\n{MIT_TEXT}```\n")


def write_library_files(root: Path, specs: list[GarSpec], recs: list[cv.ImportRecord],
                        densities: dict[str, float], sims: list[str]) -> dict[str, int]:
    """Replace ``sims``' objects in ``manifest.json`` and ``queue.jsonl`` with this run's; rewrite ``ATTRIBUTION.md``."""
    by_card = {s.card_name: s for s in specs}
    man = mf.load(root / "manifest.json", LIB)
    man["objects"] = {k: v for k, v in man["objects"].items() if v.get("sim") not in sims}
    rows: list[qu.QueueRow] = []
    for rec in recs:
        spec = by_card[rec.card]
        man["objects"][rec.card] = manifest_entry(root, spec, rec)
        rows += queue_rows(spec, rec, densities.get(rec.card, 0.0))
    mf.save(root / "manifest.json", man)
    stale = [k for k in qu.read(root / "queue.jsonl") if any(k[0].startswith(f"gar_{s}_") for s in sims)]
    qu.drop(root / "queue.jsonl", stale)
    n = qu.add(root / "queue.jsonl", rows)
    sims = sorted({o["sim"] for o in man["objects"].values()})
    (root / "ATTRIBUTION.md").write_text(attribution_text(sims), encoding="utf-8")
    log.info("library files: %d objects in the manifest, %d queue rows", len(man["objects"]), n)
    return {"objects": len(man["objects"]), "queue_rows": n}
