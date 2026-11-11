# SPDX-License-Identifier: Apache-2.0
"""Helpers for the per-simulation exporters. Runs in Gym-Anything's Python env (numpy, trimesh, manifold3d), never
imports Alloy. Each ``ex_<sim>.py`` defines ``export(x: Exporter)`` and calls ``x.add(...)`` once per object.

Example (inside an exporter)::

    c = geometry.get("glass_pure")                                     # their own builder
    x.add("glass_pure", "Bordeaux wine glass", "container", revolve(c.solid),
          material={"name": "Tritan crystal"}, mass=cited(c.mass, "datasheet", where(geometry.pure_bordeaux)),
          friction=cited(P.MU["glass_quartz"], "literature", line_of(P, "MU =")), friction_partner="quartz bar top")
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import math
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

log = logging.getLogger("gar.export")
HERE = Path(__file__).resolve().parent
SPEC_PY = HERE.parents[1] / "alloy" / "importers" / "gymanything" / "spec.py"   # stdlib only
ENV_DIR = "benchmarks/roboworld/environments"
_SPEC_MOD = None


def spec_module():
    """The importer's ``spec.py`` loaded by path (stdlib only, so Alloy itself is never imported here)."""
    global _SPEC_MOD
    if _SPEC_MOD is None:
        s = importlib.util.spec_from_file_location("gar_spec", SPEC_PY)
        _SPEC_MOD = importlib.util.module_from_spec(s)
        sys.modules["gar_spec"] = _SPEC_MOD          # dataclasses look their module up while the class is built
        s.loader.exec_module(_SPEC_MOD)  # type: ignore[union-attr]
    return _SPEC_MOD


# ------------------------------------------------------------------------------------------------ citations
def cited(value: float, basis: str, cite: str, quote: str = "", unc: tuple[float, float] | None = None) -> dict:
    """A sourced number. ``basis`` is Gym-Anything's ledger label (datasheet, standard, literature-read, literature,
    search, derived, estimate, authored, measured, calibrated); ``cite`` is ``<sim>/<file>.py:<line>``."""
    out: dict[str, Any] = {"value": float(value), "basis": basis, "cite": cite}
    if quote:
        out["quote"] = quote
    if unc is not None:
        out["unc"] = [float(unc[0]), float(unc[1])]
    return out


def dim(name: str, value_m: float, measure: str, cite: str) -> dict:
    """A cited dimension; ``measure`` in x, y, z, max, mid, min, diameter_xy (checked against the mesh)."""
    return {"name": name, "value_m": float(value_m), "measure": measure, "cite": cite}


def _rel(path: str | Path) -> str:
    p = Path(path).resolve().as_posix()
    i = p.find(ENV_DIR + "/")
    if i >= 0:
        return p[i + len(ENV_DIR) + 1:]
    j = p.find("/src/")
    return p[j + 1:] if j >= 0 else Path(p).name


def where(obj: Any) -> str:
    """``<sim>/<file>.py:<line>`` of a function or class in their code (via ``inspect``)."""
    return f"{_rel(inspect.getsourcefile(obj))}:{inspect.getsourcelines(obj)[1]}"


def line_of(module: Any, pattern: str) -> str:
    """``<sim>/<file>.py:<line>`` of the first line of ``module``'s source containing ``pattern`` (plain text)."""
    path = Path(inspect.getsourcefile(module))
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if pattern in line:
            return f"{_rel(path)}:{n}"
    raise KeyError(f"{pattern!r} not found in {path}")


def comment_of(module: Any, pattern: str) -> str:
    """The trailing ``# ...`` comment of the line ``line_of`` finds (their provenance text), stripped."""
    path = Path(inspect.getsourcefile(module))
    for line in path.read_text().splitlines():
        if pattern in line and "#" in line:
            return re.sub(r"\s+", " ", line.split("#", 1)[1]).strip()
    return ""


# ------------------------------------------------------------------------------------------------ meshes
def mesh(vertices: Any, faces: Any) -> trimesh.Trimesh:
    """A processed trimesh (merged vertices, fixed normals)."""
    m = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces, np.int64), process=True)
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    return m


def revolve(poly_rz: Any, sectors: int = 96) -> trimesh.Trimesh:
    """Watertight solid of revolution about z of a closed (r, z) polygon (r >= 0; points on the axis allowed)."""
    poly = np.asarray(poly_rz, float)
    if np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    ring = np.vstack([poly, poly[:1]])
    m = trimesh.creation.revolve(ring, sections=sectors) if hasattr(trimesh.creation, "revolve") else None
    if m is None or not m.is_watertight:
        from shapely.geometry import Polygon  # noqa: PLC0415
        m = _revolve_polygon(Polygon(poly), sectors)
    return m


def _revolve_polygon(poly: Any, sectors: int) -> trimesh.Trimesh:
    """Fallback: revolve by sweeping the polygon outline with manifold3d (closed by construction)."""
    import manifold3d as mf  # noqa: PLC0415
    cs = mf.CrossSection([np.asarray(poly.exterior.coords)[:-1]])
    out = mf.Manifold.revolve(cs, sectors).to_mesh()
    return mesh(out.vert_properties[:, :3], out.tri_verts)


def union(parts: list[trimesh.Trimesh], snap: float = 0.0) -> trimesh.Trimesh:
    """Boolean union (manifold3d engine): overlapping parts become one closed surface with no inner faces. With
    ``snap`` > 0 the vertices are rounded to that many metres and merged (helps a union that reloads open from
    OBJ; it can also break a fine mesh, so check ``is_watertight`` after)."""
    m = parts[0] if len(parts) == 1 else trimesh.boolean.union(parts, engine="manifold")
    if snap > 0:
        m = trimesh.Trimesh(np.round(m.vertices / snap) * snap, m.faces, process=True)
        m.merge_vertices()
        trimesh.repair.fix_normals(m)
    return m


def transformed(m: trimesh.Trimesh, pos: Any = (0, 0, 0), quat_xyzw: Any = (0, 0, 0, 1)) -> trimesh.Trimesh:
    """A copy moved by a rotation (x, y, z, w) and a translation."""
    x, y, z, w = quat_xyzw
    T = trimesh.transformations.quaternion_matrix([w, x, y, z])
    T[:3, 3] = pos
    return m.copy().apply_transform(T)


def box(half: Any, pos: Any = (0, 0, 0), quat_xyzw: Any = (0, 0, 0, 1)) -> trimesh.Trimesh:
    return transformed(trimesh.creation.box(extents=2 * np.asarray(half, float)), pos, quat_xyzw)


def cylinder(radius: float, half_height: float, pos: Any = (0, 0, 0), quat_xyzw: Any = (0, 0, 0, 1),
             sections: int = 48) -> trimesh.Trimesh:
    """Axis = local z."""
    return transformed(trimesh.creation.cylinder(radius, 2 * half_height, sections=sections), pos, quat_xyzw)


def capsule(radius: float, half_height: float, pos: Any = (0, 0, 0), quat_xyzw: Any = (0, 0, 0, 1),
            arc: int = 12) -> trimesh.Trimesh:
    """Axis = local z; ``half_height`` excludes the caps. Built as a revolved profile, so it is watertight."""
    top = [(radius * math.cos(a), half_height + radius * math.sin(a)) for a in np.linspace(0, math.pi / 2, arc)]
    bot = [(r, -z) for r, z in reversed(top)]
    prof = [(0.0, -half_height - radius), *bot[1:], *top[:-1], (0.0, half_height + radius)]
    return transformed(_revolve_polygon(_Poly(prof), 48), pos, quat_xyzw)


class _Poly:
    """Minimal stand-in for a shapely polygon (``exterior.coords``) so ``_revolve_polygon`` needs no shapely."""

    def __init__(self, pts: list) -> None:
        self.exterior = type("Ring", (), {"coords": [*pts, pts[0]]})()


def sphere(radius: float, pos: Any = (0, 0, 0)) -> trimesh.Trimesh:
    return transformed(trimesh.creation.icosphere(4, radius), pos)


# ------------------------------------------------------------------------------------------------ visual parts
def vis(geom: trimesh.Trimesh, base_color: Any, cite: str, roughness: float | None = None,
        metallic: float | None = None, image: Any = None, uv: Any = None, name: str = "") -> dict:
    """One render part for ``Exporter.add(..., visual=[...])``: body-frame geometry, sRGB base colour 0..1 (their
    render code's value, converted from 0-255 if needed), roughness/metallic if they set them, ``cite`` (file:line of
    the colour). ``image`` (PIL image or HxWx3/4 uint8 array) + ``uv`` (per-vertex (V, 2), OpenGL/Blender convention:
    origin bottom-left) give a base-colour texture (their generated artwork, labels)."""
    col = [float(c) / (255.0 if max(base_color) > 1.0 else 1.0) for c in base_color][:3]
    return {"_geom": geom, "_image": image, "_uv": uv, "base_color": col, "cite": cite, "roughness": roughness,
            "metallic": metallic, "name": name}


def _write_obj_uv(path: Path, v: np.ndarray, f: np.ndarray, uv: np.ndarray | None) -> None:
    lines = [f"v {a:.7g} {b:.7g} {c:.7g}" for a, b, c in v]
    if uv is not None:
        lines += [f"vt {a:.6g} {b:.6g}" for a, b in uv]
        lines += [f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}" for a, b, c in f]
    else:
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n")


def _write_visual(d: Path, key: str, parts: list[dict]) -> list[dict]:
    """Write each part's OBJ (and PNG) next to the spec; return the spec entries (paths relative to the spec)."""
    out = []
    for i, p in enumerate(parts):
        g, img, uv = p["_geom"], p["_image"], p["_uv"]
        stem = f"{key}__vis{i}"
        if uv is None and img is not None and getattr(g.visual, "uv", None) is not None:
            uv = np.asarray(g.visual.uv)
        _write_obj_uv(d / f"{stem}.obj", np.asarray(g.vertices), np.asarray(g.faces), None if img is None else uv)
        entry = {k: v for k, v in p.items() if not k.startswith("_") and v not in (None, "")}
        entry["mesh"] = f"{stem}.obj"
        if img is not None:
            from PIL import Image  # noqa: PLC0415
            (img if hasattr(img, "save") else Image.fromarray(np.asarray(img))).save(d / f"{stem}.png")
            entry["texture"] = f"{stem}.png"
        out.append(entry)
    return out


# ------------------------------------------------------------------------------------------------ exporter
class Exporter:
    """Collects one simulation's objects: ``add`` writes ``<out>/<sim>/<key>.json`` + ``<key>.obj``; failures are
    recorded (with the reason) instead of stopping the other objects."""

    def __init__(self, sim: str, out: Path) -> None:
        self.sim, self.dir = sim, Path(out) / sim
        self.dir.mkdir(parents=True, exist_ok=True)
        self.written: list[str] = []
        self.failed: dict[str, str] = {}
        self.warnings: dict[str, list[str]] = {}

    def add(self, key: str, name: str, category: str, geom: trimesh.Trimesh | None, **fields: Any) -> None:
        """Write one object. ``fields`` are the remaining ``gar.spec/1`` keys (material, mass, friction, ...)."""
        try:
            self._add(key, name, category, geom, fields)
        except Exception as exc:  # noqa: BLE001 - one bad object must not stop the sim
            self.fail(key, f"{type(exc).__name__}: {exc}")
            log.debug("%s/%s failed:\n%s", self.sim, key, traceback.format_exc())

    def _add(self, key: str, name: str, category: str, geom: trimesh.Trimesh | None, fields: dict) -> None:
        parts = fields.pop("visual", None)
        raw: dict[str, Any] = {"schema": "gar.spec/1", "key": key, "sim": self.sim, "name": name,
                               "category": category, **fields}
        if parts:
            raw["visual"] = _write_visual(self.dir, key, parts)
        warns = []
        if geom is not None:
            if not geom.is_watertight:
                warns.append("mesh not watertight (the importer will use its convex hull)")
            if not all(math.isfinite(v) for v in geom.bounds.ravel()):
                raise ValueError("mesh has non-finite vertices")
            geom.export(self.dir / f"{key}.obj")
            raw["mesh"] = f"{key}.obj"
        spec = spec_module()
        spec.parse_spec(raw, f"{self.sim}/{key}", self.dir)
        (self.dir / f"{key}.json").write_text(json.dumps(raw, indent=1))
        self.written.append(key)
        if warns:
            self.warnings[key] = warns

    def fail(self, key: str, reason: str) -> None:
        """Record an object that cannot be exported (it goes to the import index as skipped, with the reason)."""
        self.failed[key] = reason
        log.warning("%s/%s not exported: %s", self.sim, key, reason)

    def summary(self) -> dict:
        return {"sim": self.sim, "written": self.written, "failed": self.failed, "warnings": self.warnings}
