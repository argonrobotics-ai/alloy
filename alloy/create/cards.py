# SPDX-License-Identifier: Apache-2.0
"""Write a created object's two cards into the output card root, in the same form as the imported libraries:
``objects/<name>.yaml`` (``meshes/<name>/physics.obj``, ``collision: decompose`` over the CoACD hulls) and
``materials/<name>_eff.yaml`` (effective density from the mass over the mesh volume; friction, restitution and
elasticity copied from the base library material, every note naming its source).

A mass you gave (with or without a source) becomes a ``fit`` density; an estimated mass becomes a ``model`` density
whose note starts with ``ESTIMATE`` and carries the range.

Example::

    >>> files = write_cards(out, "red_mug", req, proc, base, mat_row, mass_row, size_row)     # doctest: +SKIP
    >>> sorted(files)                                                                          # doctest: +SKIP
    ['material', 'object', 'physics.hulls.obj', 'physics.obj', 'thumb.png', 'visual.glb']
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alloy.create.physics import Value

MASS_REL_UNC = 0.02                 # a mass you weighed: +-2 % unless you give a range


def write_cards(out: Path, name: str, req, proc: dict[str, Any], base, mat_row: Value, mass: Value,
                size: Value) -> dict[str, str]:
    """Write both cards; returns label -> path of every file the object needs."""
    from alloy.importers.common.cards import EFF_SUFFIX, CardSpec, MassSpec, card_texts  # noqa: PLC0415

    volume = float(proc["volume"])
    if mass.tag == "ESTIMATE":
        lo, hi = (mass.lo or mass.value) / volume, (mass.hi or mass.value) / volume
        spec_mass = MassSpec("density", mass.value / volume, (lo, hi), f"{mass.source} -> "
                             f"{mass.value * 1e3:.4g} g ({(mass.lo or mass.value) * 1e3:.4g}-"
                             f"{(mass.hi or mass.value) * 1e3:.4g} g) over the {volume * 1e6:.4g} cm^3 mesh")
    else:
        spec_mass = MassSpec("mass", mass.value, (mass.value * (1 - MASS_REL_UNC), mass.value * (1 + MASS_REL_UNC)),
                             mass.source)
    spec = CardSpec(name=name, base_material=base.name, material_note=mat_row.source, mass=spec_mass, volume=volume,
                    mesh_path=f"meshes/{name}/physics.obj", mesh_sha256=proc["files"]["physics.obj"],
                    max_parts=int(proc["n_hulls"]), header=_header(name, req, size), min_wall=_min_wall(out, name))
    obj_text, mat_text = card_texts(spec, base)
    paths = {"object": out / "objects" / f"{name}.yaml", "material": out / "materials" / f"{name}{EFF_SUFFIX}.yaml"}
    for p, text in ((paths["object"], obj_text), (paths["material"], mat_text)):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    files = {k: str(v) for k, v in paths.items()}
    files.update({f: str(out / "meshes" / name / f) for f in proc["files"]})
    return files


def _header(name: str, req, size: Value) -> str:
    what = f"text {req.text!r}" if req.kind == "text" else f"{req.kind} {Path(req.mesh or req.photo).name}"
    return "\n".join([
        f"created by alloy.create from {what}; rerun alloy.create to change it",
        f"size: longest side {size.value:.4g} m [{size.tag}] {size.source}",
        f"files: meshes/{name}/physics.obj (watertight), physics.hulls.obj (CoACD hulls), visual.glb, thumb.png",
    ])


def _min_wall(out: Path, name: str) -> float | None:
    """The thinnest wall when it is under the validator's threshold (the card must then declare it)."""
    from alloy.cards import geometry as geo  # noqa: PLC0415
    from alloy.importers.common.checks import THIN  # noqa: PLC0415

    wall = float(geo.wall_thickness(geo.read_mesh(out / "meshes" / name / "physics.obj")))
    return wall if wall < THIN else None


__all__ = ["write_cards"]
