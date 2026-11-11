# SPDX-License-Identifier: Apache-2.0
"""The soft-asset pack's film and cloth objects (plastic film sheets, a grocery bag, paper sheets, towels, cloth
sheets): their expected routes and a drop check of each: the object 0.5 mm above the library table top, through
``alloy.check`` (cards, overlap, rest drift, health, penetration, energy). ``alloy.check.check_object`` runs the same
checks on a floor it sizes to the object; the floor here stays the table top card, the surface these objects are used
on.

Run all checks and print one JSON row per object::

    python -m alloy.testing.soft_film --device cpu
    python -m alloy.testing.soft_film --device cuda:0 --worlds 64 --out rows.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

#: library object -> the route the router gives it
OBJECTS: dict[str, str] = {
    "film_hdpe_sheet": "cloth_film",
    "film_ldpe_sheet": "cloth_film",
    "film_lldpe_sheet": "cloth_film",
    "bag_hdpe_tshirt": "cloth_film",
    "paper_kraft_sheet": "cloth_film",
    "paper_wrap_sheet": "cloth_film",
    "paper_board_sheet": "cloth_film",
    "paper_kraft_scored_sheet": "cloth_film",
    "towel_cotton": "cloth_vbd",
    "towel_cotton_hand": "cloth_vbd",
    "cloth_cotton_sheet": "cloth_vbd",
    "cloth_polyester_sheet": "cloth_vbd",
}


FLOOR = "table_top_plywood"
GAP_M = 5.0e-4
FILM_RADIUS_M = 5.0e-4                 # contact radius of film particles (alloy.solvers.film.materials.FilmExtras)


def drop_card(name: str, library: Any = None) -> Any:
    """A SceneCard: the table top (top face at z = 0) and ``name`` with its lowest contact surface GAP_M above it."""
    from alloy.cards import types as ct  # noqa: PLC0415
    from alloy.cards.library import CardLibrary  # noqa: PLC0415

    lib = library or CardLibrary()
    card = lib.peek_object(name)
    h = float(card.features.thickness.value) if card.features.thickness is not None else 0.0
    radius = FILM_RADIUS_M if OBJECTS.get(name) == "cloth_film" else 0.5 * h
    if isinstance(card.shape, ct.MeshSource):
        below = -float(min(float(line.split()[3]) for line in lib.asset_path(card.shape.path).read_text().splitlines()
                           if line.startswith("v ")))
    else:
        below = 0.0                    # a sheet primitive: the cloth mesh is its mid-plane
    floor_hz = float(lib.peek_object(FLOOR).shape.size[2])
    return ct.SceneCard(robots=(), objects=(ct.Placement(FLOOR, (0.0, 0.0, -floor_hz, 0.0, 0.0, 0.0, 1.0)),
                                            ct.Placement(name, (0.0, 0.0, below + radius + GAP_M, 0.0, 0.0, 0.0, 1.0))))


def check_row(name: str, *, device: str | None = None, worlds: int | None = None, seconds: float | None = None,
              budget_s: float | None = 25.0) -> dict[str, Any]:
    """Drop check of one object: ok, the per-check verdicts, errors / warnings (with their messages) and wall time."""
    import alloy.check  # noqa: PLC0415

    t0 = time.perf_counter()
    report = alloy.check(drop_card(name), worlds, seconds, device=device, budget_s=budget_s)
    data = report.to_dict()
    notes = [f.message for f in report.findings if f.level == "info"]
    return {"object": name, "ok": report.ok, "checks": data.get("checks", {}), "device": device,
            "worlds": worlds, "seconds": seconds, "wall_s": round(time.perf_counter() - t0, 1),
            "errors": [f.message for f in report.errors()], "warnings": [f.message for f in report.warnings()],
            "info": notes[:6]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--worlds", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--budget", type=float, default=25.0, help="wall budget per check in s (0 = none)")
    ap.add_argument("--only", nargs="*", default=None, help="object names (default: all)")
    ap.add_argument("--out", type=Path, default=None, help="also write the rows as one JSON list")
    a = ap.parse_args(argv)
    rows = []
    for name in a.only or list(OBJECTS):
        row = check_row(name, device=a.device, worlds=a.worlds, seconds=a.seconds, budget_s=a.budget or None)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if a.out is not None:
        a.out.write_text(json.dumps(rows, indent=1) + "\n")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
