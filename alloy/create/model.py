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


def create_from_model(req: CreateRequest, out: Path, *, check: bool = True, device: str | None = None
                      ) -> CreateResult:
    """Import the model file into ``out`` and drop-test its objects (robots are imported, not dropped)."""
    from alloy.io import import_file  # noqa: PLC0415

    path = Path(req.mesh).expanduser()
    if not path.is_file():
        raise CreateError(f"model file {str(path)!r} not found", "pass the path of an existing .xml/.urdf/.usd file",
                          field="mesh")
    t0 = time.perf_counter()
    res = import_file(path, out, name=req.name)
    if not res.objects:
        raise CreateError(f"{path.name} holds no objects (robots: {[r.get('name') for r in res.robots]})",
                          "import robots with alloy.io.import_file / Scene.add_robot; create makes objects",
                          field="mesh")
    rows = [_model_row(name) for name in res.objects]
    rows += [{"field": "issue", "value": 0.0, "unit": "", "tag": "MODEL", "source": i.get("card", ""),
              "verify": f"{i.get('field')}: {i.get('message')}"} for i in res.issues if i.get("level") != "info"]
    rows += [{"field": "unmapped", "value": 0.0, "unit": "", "tag": "MODEL", "source": str(u),
              "verify": f"not carried into the cards: {u}"} for u in res.unmapped]
    first = res.objects[0]
    result = CreateResult(name=first, card=str(out / "objects" / f"{first}.yaml"), root=str(out),
                          files={f"file{i}": str(f) for i, f in enumerate(res.files)}, values=rows,
                          objects=list(res.objects), notes=_notes(res))
    if check:
        result.check = _drop_tests(res.objects, out, device)
    result.timings_s["total"] = round(time.perf_counter() - t0, 2)
    return result


def _model_row(name: str) -> dict[str, Any]:
    return {"field": f"{name}.mass", "value": 0.0, "unit": "kg", "tag": "MODEL",
            "source": "mass and inertia from the model file", "verify": f"{name}: check the file's masses"}


def _notes(res) -> list[str]:
    notes = [f"imported {len(res.objects)} object(s) from {res.name}"]
    if res.robots:
        notes.append(f"robots in the file (not drop-tested): {[r.get('name') for r in res.robots]}")
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
