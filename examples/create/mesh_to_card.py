# SPDX-License-Identifier: Apache-2.0
"""Turn a mesh file into a checked object card.

A 3D-printed cup saved in millimetres (made here with trimesh, so the example needs no download) becomes
``my_objects/objects/printed_cup.yaml``: units read as mm, the watertight physics mesh and convex hulls built, the
material and mass chosen with their sources, and a drop test run. The output lists the estimates and what to verify.

Run (needs the ``import`` extra: ``pip install -e '.[import]'``; CPU is fine):

    python examples/create/mesh_to_card.py                  # a short summary
    python examples/create/mesh_to_card.py --json           # the full JSON result
    python examples/create/mesh_to_card.py --mass 0.041     # you weighed it: the mass becomes USER, not ESTIMATE

The same from the command line: ``python -m alloy.create --mesh cup.stl --material plastic --out my_objects``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import alloy.create


def write_cup(path: Path) -> Path:
    """An 80 mm wide, 95 mm tall cup with 3 mm walls and a 4 mm floor, in millimetres (like most CAD exports)."""
    import trimesh

    outer = trimesh.creation.cylinder(radius=40.0, height=95.0, sections=64)
    inner = trimesh.creation.cylinder(radius=37.0, height=95.0, sections=64)
    inner.apply_translation((0.0, 0.0, 4.0))
    cup = outer.difference(inner)
    cup.apply_translation((0.0, 0.0, 47.5))
    cup.export(path)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="my_objects", help="the folder the cards go to")
    p.add_argument("--mass", type=float, default=None, help="the real mass in kg, if you weighed it")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    mesh = write_cup(Path(a.out) / "printed_cup.stl")
    result = alloy.create.create(mesh=str(mesh), material="plastic", mass=a.mass,
                                 mass_source="kitchen scale" if a.mass else None, out=a.out, device="cpu")
    print(result.to_json() if a.json else result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
