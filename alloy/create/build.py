# SPDX-License-Identifier: Apache-2.0
"""``create(...)``: a mesh, a photo or a text description in; a validated, physics-sourced object card out.

Steps: (photo/text: generate a mesh, ``alloy.create.generate``) -> find the category -> real size (``scale``) ->
repair, watertight physics mesh, CoACD hulls, visual GLB and thumbnail (``alloy.importers.common.process``) -> base
material and mass with their sources (``physics``) -> object and material cards in ``out`` -> ``alloy.check`` drop
test. MJCF, URDF and USD files go through ``alloy.io.import_file`` (articulation kept) and are drop-tested.

Example::

    >>> r = create(mesh="mug.obj", size=0.11, category="mug", out="my_objects")      # doctest: +SKIP
    >>> r.ok, r.estimates                                                             # doctest: +SKIP
    (True, ['material', 'mass'])
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from alloy.create.physics import Category, Value, choose_material, find_category, mass_value
from alloy.create.request import MESH_SUFFIXES, MODEL_SUFFIXES, CreateError, CreateRequest
from alloy.create.result import CreateResult
from alloy.create.scale import resolve_scale

log = logging.getLogger("alloy.create")

MESH_DIR = "meshes"


def create(request: CreateRequest | dict[str, Any] | None = None, *, check: bool = True, device: str | None = None,
           **fields: Any) -> CreateResult:
    """Make one object card (see the module docstring). Pass a ``CreateRequest``, its dict, or its fields as keyword
    arguments (``mesh=``, ``photo=``, ``text=``, ``size=``, ``mass=``, ``material=``, ``category=``, ``out=``...).
    ``check=False`` skips the drop test. Example: ``create(text="a ceramic mug", size=0.11)``."""
    t0 = time.perf_counter()
    req = _request(request, fields)
    out = Path(req.out).expanduser().resolve()
    if req.mesh and Path(req.mesh).suffix.lower() in MODEL_SUFFIXES:
        from alloy.create.model import create_from_model  # noqa: PLC0415

        return create_from_model(req, out, check=check, device=device)
    src, generator = _source(req, out)
    result = _make(req, out, Path(src), generator)
    if check:
        result.check = _drop_test(result, out, device)
    result.timings_s["total"] = round(time.perf_counter() - t0, 2)
    log.info("create %s: %s in %.1f s", result.name, "OK" if result.ok else "check failed", result.timings_s["total"])
    return result


def _request(request: CreateRequest | dict[str, Any] | None, fields: dict[str, Any]) -> CreateRequest:
    if isinstance(request, CreateRequest):
        return request if not fields else CreateRequest.from_dict({**request.to_dict(), **fields})
    if isinstance(request, dict):
        return CreateRequest.from_dict({**request, **fields})
    if request is None:
        return CreateRequest.from_dict(fields)
    raise CreateError(f"request of type {type(request).__name__}", "pass a CreateRequest, a dict, or keywords")


def _source(req: CreateRequest, out: Path) -> tuple[str, dict[str, Any] | None]:
    """The mesh file to process: yours, or one generated from the photo or text."""
    if req.mesh:
        path = Path(req.mesh).expanduser()
        if not path.is_file():
            raise CreateError(f"mesh file {str(path)!r} not found", "pass the path of an existing mesh file",
                              field="mesh")
        if path.suffix.lower() not in MESH_SUFFIXES:
            raise CreateError(f"{path.suffix} is not a mesh format create reads",
                              f"use one of {MESH_SUFFIXES + MODEL_SUFFIXES}", field="mesh")
        return str(path), None
    from alloy.create.generate import generate  # noqa: PLC0415

    gen = generate(photo=req.photo, text=req.text, work=out / ".generated" / req.card_name)
    return _level_generated(Path(gen["mesh"]), gen), gen


def _level_generated(mesh_path: Path, gen: dict[str, Any]) -> str:
    """The generated mesh stood upright (``alloy.create.level``), written next to it; notes the angle in ``gen``."""
    import trimesh  # noqa: PLC0415

    from alloy.create.level import level  # noqa: PLC0415

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    gen["closed"] = bool(mesh.is_volume)
    done = level(mesh)
    gen["leveled_deg"] = round(done.degrees, 1)
    if not done.applied:
        return str(mesh_path)
    out = mesh_path.with_name("mesh_level.obj")
    mesh.export(out)
    return str(out)


def _make(req: CreateRequest, out: Path, src: Path, generator: dict[str, Any] | None) -> CreateResult:
    from alloy.importers.common.process import ProcessOptions, process  # noqa: PLC0415

    name = req.card_name
    lib = _library(out, name)
    cat = find_category(req.text, req.category, name)
    up = req.up if generator is None else "z"          # a generated mesh is already stood on its base
    scale = resolve_scale(_longest(src, up), req.size, req.units, generator is not None, cat)
    t = time.perf_counter()
    proc = process(src, out / MESH_DIR / name, name, ProcessOptions(scale=scale.factor, up_axis=up))
    base_name, mat_val = choose_material(lib, req.material, cat, req.text)
    base = lib.material(base_name)
    mass = mass_value(req, cat, base.density.value, proc["volume"], base_name)
    files = _write_cards(out, name, req, proc, base, mat_val, mass, scale.value)
    values = [scale.value, mat_val, mass, *_surface_rows(base), *_shape_rows(req, generator)]
    notes = [f"category: {cat.name}"] if cat else ["category: unknown (pass category= for better estimates)"]
    if generator and generator.get("leveled_deg"):
        notes.append(f"generated mesh turned {generator['leveled_deg']} degrees onto the face it rests on")
    if generator and req.up != "z":
        notes.append(f"up={req.up!r} ignored: a generated mesh is stood on its base automatically")
    return CreateResult(name=name, card=str(out / "objects" / f"{name}.yaml"), root=str(out), files=files,
                        values=[v.to_dict() for v in values], notes=notes, generator=_gen_meta(generator),
                        timings_s={"process": round(time.perf_counter() - t, 2)})


def _library(out: Path, name: str):
    """Alloy's library with ``out`` in front; refuses a name another library already uses."""
    from alloy.compose.load import library  # noqa: PLC0415

    lib = library([out])
    if lib.has_object(name) and not Path(lib.path_of("objects", name)).resolve().is_relative_to(out):
        raise CreateError(f"an object named {name!r} already exists in the library", "pass another name=",
                          field="name")
    return lib


def _longest(src: Path, up: str) -> float:
    from alloy.importers.common.meshops import load_source  # noqa: PLC0415

    mesh = load_source(src, 1.0, up)
    return float(max(mesh.extents))


def _write_cards(out: Path, name: str, req: CreateRequest, proc: dict[str, Any], base, mat_val: Value,
                 mass: Value, size: Value) -> dict[str, str]:
    from alloy.create.cards import write_cards  # noqa: PLC0415

    return write_cards(out, name, req, proc, base, mat_val, mass, size)


def _surface_rows(base) -> list[Value]:
    s = base.surface
    return [Value(k, float(getattr(s, k).value), "", "LIBRARY", f"library material {base.name}")
            for k in ("mu_static", "mu_dynamic", "restitution")]


def _shape_rows(req: CreateRequest, generator: dict[str, Any] | None) -> list[Value]:
    """A generated shape is a guess from one image: say so, and what to look at."""
    if generator is None:
        return []
    from alloy.create.level import BIG_TILT_DEG  # noqa: PLC0415

    model = ((generator.get("models") or {}).get("image_to_3d") or {}).get("name", "an image-to-3D model")
    verify = "look at the thumbnail: the hidden back is guessed and openings may come out closed"
    if not generator.get("closed", True):
        verify = ("the generated mesh is not a closed solid, often because the object blends into the background: "
                  "look at the thumbnail; retry with a photo on a plain background of another colour")
    elif generator.get("leveled_deg", 0.0) > BIG_TILT_DEG:
        verify = (f"it was turned {generator['leveled_deg']} degrees to rest stably: check in the thumbnail that it "
                  "stands the right way up; " + verify)
    return [Value("shape", 0.0, "", "ESTIMATE", f"generated from the {req.kind} by {model} (one view)",
                  verify=verify)]


def _gen_meta(generator: dict[str, Any] | None) -> dict[str, Any] | None:
    if generator is None:
        return None
    return {k: v for k, v in generator.items() if k != "mesh"}


def _drop_test(result: CreateResult, out: Path, device: str | None) -> dict[str, Any]:
    from alloy.check import check_object  # noqa: PLC0415
    from alloy.compose.load import library  # noqa: PLC0415

    t = time.perf_counter()
    try:
        report = check_object(result.card, device=device, library=library([out])).to_dict()
    except Exception as exc:  # noqa: BLE001 - the card is written; a drop test that cannot run is a finding
        log.exception("create: the drop test of %s raised", result.name)
        report = {"ok": False, "errors": 1, "warnings": 0, "findings": [{
            "check": "drop_test", "level": "error", "message": f"the drop test could not run: {type(exc).__name__}: "
            f"{str(exc)[:300]}", "fix": "pass a rigid material= (e.g. 'plastic', 'wood'); soft bodies from meshes "
            "may not be supported by this engine version; check=False skips the test"}]}
    result.timings_s["check"] = round(time.perf_counter() - t, 2)
    return report


__all__ = ["create"]
