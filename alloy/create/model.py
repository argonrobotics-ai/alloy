# SPDX-License-Identifier: Apache-2.0
"""``create`` for MJCF, URDF and USD files: ``alloy.io.import_file`` writes the cards (joints and articulation as
the file declares them), then every object gets the ``alloy.check`` drop test. Values come from the file (tag
``MODEL``); what the cards could not hold, and the importer's issues, become things to verify.

Example::

    >>> r = create_from_model(CreateRequest(mesh="hinged_box.xml"), Path("my_objects"))      # doctest: +SKIP
    >>> r.objects                                                                             # doctest: +SKIP
    ['hinged_box']
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from alloy.create.request import CreateError, CreateRequest
from alloy.create.result import CreateResult


#: request fields a model file answers itself (its own masses, materials, units and axes), with their defaults
FILE_FIELDS = {"material": None, "mass": None, "mass_source": None, "size": None, "units": None, "up": "z"}


def create_from_model(req: CreateRequest, out: Path, *, check: bool = True, device: str | None = None
                      ) -> CreateResult:
    """Import the model file into ``out`` as objects (joints kept, passive) and drop-test them. A file with one
    object names its card ``req.name`` (default: the file's stem); a file with driven joints is a robot and is
    refused."""
    from alloy.io import import_file  # noqa: PLC0415

    path = Path(req.mesh).expanduser()
    if not path.is_file():
        raise CreateError(f"model file {str(path)!r} not found", "pass the path of an existing .xml/.urdf/.usd file",
                          field="mesh")
    _refuse_overrides(req, path)
    t0 = time.perf_counter()
    res = import_file(path, out, name=req.name, as_object=True)
    if not res.objects:
        raise CreateError(f"{path.name} holds no objects, only robots with driven joints "
                          f"({[r.get('name') for r in res.robots]})",
                          "a file with actuators is a robot: use Scene.add_robot(name, path) or "
                          "alloy.io.import_file; create makes objects (joints without actuators are kept)",
                          field="mesh")
    rows = [_model_row(name, res.masses.get(name, 0.0)) for name in res.objects]
    rows += [{"field": "issue", "value": 0.0, "unit": "", "tag": "MODEL", "source": i.get("card", ""),
              "verify": f"{i.get('field')}: {i.get('message')}"} for i in res.issues if i.get("level") != "info"]
    rows += [{"field": "unmapped", "value": 0.0, "unit": "", "tag": "MODEL", "source": str(u),
              "verify": f"not carried into the cards: {u}"} for u in res.unmapped]
    first = res.objects[0]
    result = CreateResult(name=first, card=str(out / "objects" / f"{first}.yaml"), root=str(out),
                          files={f"file{i}": str(f) for i, f in enumerate(res.files)}, values=rows,
                          objects=list(res.objects), notes=_notes(res, path))
    if check:
        result.check = _drop_tests(res.objects, out, device)
    result.timings_s["total"] = round(time.perf_counter() - t0, 2)
    return result


def _refuse_overrides(req: CreateRequest, path: Path) -> None:
    """A model file carries its own masses, materials, units and axes; an override that would be dropped raises."""
    given = [k for k, default in FILE_FIELDS.items() if getattr(req, k) != default]
    if given:
        raise CreateError(f"{', '.join(k + '=' for k in given)} cannot be applied to the model file {path.name}: "
                          "its bodies keep the file's own masses, materials, units (metres) and axes",
                          f"leave {', '.join(given)} out and set the values in the file (MJCF: geom mass= or "
                          "density=, friction=; URDF: <inertial><mass>), or pass a mesh file (OBJ, STL) to have "
                          "create set them", field=given[0])


def _model_row(name: str, mass: float) -> dict[str, Any]:
    return {"field": f"{name}.mass", "value": round(float(mass), 6), "unit": "kg", "tag": "MODEL",
            "source": "the sum of the model file's body masses", "verify": f"{name}: check the file's masses"}


def _notes(res, path: Path) -> list[str]:
    notes = [f"imported {len(res.objects)} object(s) from {res.name}"]
    if path.suffix.lower() == ".urdf":
        notes.append("URDF joints are kept as passive joints (a URDF has no actuators); when it has a moving joint "
                     "its base link is fixed in place, else the object is free. For a driven robot use "
                     "Scene.add_robot")
    if res.robots:
        notes.append(f"robots in the file (driven joints; not made into objects): "
                     f"{[r.get('name') for r in res.robots]}")
    if res.scene:
        notes.append(f"scene with the file's layout: {res.scene}")
    return notes


def _drop_tests(names: list[str], out: Path, device: str | None) -> dict[str, Any]:
    """One drop test per object; the first failing report (or the last passing one) with every verdict listed."""
    from alloy.check import check_object  # noqa: PLC0415
    from alloy.compose.load import library  # noqa: PLC0415

    lib = library([out])
    reports = {n: check_object(n, device=device, library=lib).to_dict() for n in names}
    bad = [r for r in reports.values() if not r.get("ok")]
    out_report = dict(bad[0] if bad else list(reports.values())[-1])
    out_report["per_object"] = {n: bool(r.get("ok")) for n, r in reports.items()}
    return out_report


__all__ = ["create_from_model"]
