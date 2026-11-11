# SPDX-License-Identifier: Apache-2.0
"""Batch job lists (``alloy.importers.common.batch`` format) for every YCB scan and every GSO model.

Example::

    python -m alloy.importers.ycb_gso jobs --lib gso --chunks 4 --out jobs/   # writes jobs/gso_0.json ... gso_3.json
"""

from __future__ import annotations

import json
from pathlib import Path

from alloy.importers.ycb_gso import gso, ycb


def ycb_jobs() -> list[dict]:
    """One job per YCB ``google_16k`` scan.

    Example::

        len(ycb_jobs())   # 79
    """
    return [{"name": o.card_name, "url": o.url, "mesh": o.mesh_member, "scale": 1.0, "up_axis": "z"}
            for o in ycb.OBJECTS]


def gso_job(model: str) -> dict:
    """The job of one GSO model (texture copied next to the mesh, metadata kept).

    Example::

        gso_job("Vtech_Cruise_Learn_Car_25_Years")["mesh"]   # "meshes/model.obj"
    """
    return {"name": gso.card_name(model), "url": gso.archive_url(model), "mesh": gso.MESH_MEMBER,
            "extract": ["meshes", "materials"], "copy": [["materials/textures/texture.png", "meshes/texture.png"]],
            "keep": [gso.METADATA], "scale": 1.0, "up_axis": "z", "source_name": model}


def gso_jobs(listing: list[dict]) -> list[dict]:
    """Jobs for every CC BY model in a Fuel listing (others are skipped and never imported).

    Example::

        gso_jobs(gso.fetch_listing())
    """
    return [gso_job(m["name"]) for m in sorted(listing, key=lambda m: m["name"])
            if m.get("license_name") == gso.LICENCE_NAME]


def write_chunks(jobs: list[dict], out_dir: Path, prefix: str, chunks: int) -> list[Path]:
    """Split ``jobs`` round-robin into ``chunks`` job files ``<prefix>_<k>.json``; returns their paths.

    Example::

        write_chunks(ycb_jobs(), Path("jobs"), "ycb", 1)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k in range(chunks):
        p = out_dir / f"{prefix}_{k}.json"
        p.write_text(json.dumps({"jobs": jobs[k::chunks]}, indent=1), encoding="utf-8")
        paths.append(p)
    return paths
