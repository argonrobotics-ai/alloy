# SPDX-License-Identifier: Apache-2.0
"""Look files: loading, validation against a bundle, frame selection and cost estimate."""
from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

LOOK_VERSION = 1
WARN = "warning: "
# Calibration from the L4 benchmark: seconds per frame at 1080x1920 and 512 samples, and the spot price.
REF_SECONDS = 45.0
REF_PIXELS = 1080 * 1920
REF_SAMPLES = 512
SPOT_USD_PER_HOUR = 0.28
DEFAULT_RESOLUTION = (1080, 1920)
DEFAULT_SAMPLES = 512
_PRESETS_FILE = Path(__file__).resolve().parent / "blender" / "presets.py"
_PRESET_NAMES = ("PRESETS", "MATERIAL_PRESETS", "MATERIALS")


class LookError(ValueError):
    """Raised for an unreadable or unsupported look file."""


@dataclass(frozen=True)
class CostEstimate:
    """Estimated render cost for a frame list."""

    frames: int
    seconds_per_frame: float
    gpu_hours: float
    usd: float


def load_look(path: str | Path) -> dict[str, Any]:
    """Read a look file and check its version."""
    p = Path(path)
    try:
        look = json.loads(p.read_text())
    except OSError as exc:
        raise LookError(f"cannot read look file {p}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LookError(f"look file {p} is not valid JSON: {exc}") from exc
    if not isinstance(look, dict):
        raise LookError(f"look file {p}: top level must be an object, got {type(look).__name__}")
    if look.get("look") != LOOK_VERSION:
        raise LookError(f'look file {p}: field "look" is {look.get("look")!r}, this CLI supports {LOOK_VERSION}')
    return look


def preset_names(presets_file: Path = _PRESETS_FILE) -> set[str] | None:
    """Material preset names parsed (not imported) from presets.py, or None if unavailable."""
    try:
        tree = ast.parse(presets_file.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        names = _assigned_names(node)
        value = getattr(node, "value", None)
        if not isinstance(value, ast.Dict) or not names.intersection(_PRESET_NAMES):
            continue
        keys = {k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if keys:
            return keys
    return None


def _assigned_names(node: ast.stmt) -> set[str]:
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _regex_problem(pattern: Any, where: str) -> str | None:
    if not isinstance(pattern, str):
        return f"{where}: regex must be a string, got {pattern!r}"
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"{where}: regex {pattern!r} does not compile: {exc}"
    return None


def _check_render(look: dict[str, Any]) -> list[str]:
    render = look.get("render")
    if not isinstance(render, dict):
        return ['render: missing or not an object']
    out = []
    res = render.get("resolution")
    if not (isinstance(res, list) and len(res) == 2 and all(isinstance(v, int) and v > 0 for v in res)):
        out.append(f"render.resolution: expected [width, height] positive ints, got {res!r}")
    samples = render.get("samples")
    if not (isinstance(samples, int) and samples > 0):
        out.append(f"render.samples: expected a positive int, got {samples!r}")
    wb = render.get("white_balance", "auto")
    if not (wb in ("auto", "off") or (isinstance(wb, dict) and isinstance(wb.get("kelvin"), (int, float)))):
        out.append(f'render.white_balance: expected "auto", "off" or {{"kelvin": K, "tint": t}}, got {wb!r}')
    return out


def _check_props(look: dict[str, Any], bundle: Any) -> list[str]:
    out = []
    root = Path(getattr(bundle, "root", "."))
    for i, prop in enumerate(look.get("props") or []):
        where = f"props[{i}]"
        model, pose = (prop.get("model"), prop.get("pose")) if isinstance(prop, dict) else (None, None)
        if not isinstance(model, str) or not model.startswith(("polyhaven:", "file:")):
            out.append(f'{where}.model: expected "polyhaven:<id>" or "file:<bundle-relative path>", got {model!r}')
        elif model.startswith("file:") and not (root / model[5:]).is_file():
            out.append(f"{where}.model: {model[5:]!r} not found under the bundle root {root}")
        if not (isinstance(pose, list) and len(pose) == 7):
            out.append(f"{where}.pose: expected [x, y, z, qx, qy, qz, qw], got {pose!r}")
    ground = look.get("ground")
    if ground is not None and not (isinstance(ground, dict) and isinstance(ground.get("z"), (int, float))
                                   and isinstance(ground.get("material"), str)):
        out.append(f"ground: expected {{\"z\": height, \"material\": name, \"size_m\": 4000}}, got {ground!r}")
    return out


def _check_focus(focus: Any, where: str) -> list[str]:
    if focus is None:
        return []
    if not isinstance(focus, dict) or len(focus) != 1:
        return [f'{where}.focus: expected one of "label", "point", "distance", got {focus!r}']
    kind, value = next(iter(focus.items()))
    if kind == "label":
        err = _regex_problem(value, f"{where}.focus.label")
        return [err] if err else []
    if kind not in ("point", "distance"):
        return [f'{where}.focus: unknown key {kind!r} (use "label", "point" or "distance")']
    return []


def _check_cameras(look: dict[str, Any]) -> list[str]:
    cams = look.get("cameras")
    if not isinstance(cams, dict) or not cams:
        return ["cameras: need at least one named camera"]
    out = []
    for name, cam in cams.items():
        where = f"cameras.{name}"
        keys = cam.get("keys") if isinstance(cam, dict) else None
        if not keys or not isinstance(keys, list):
            out.append(f"{where}.keys: need at least one {{t, eye, target}} key")
            continue
        for i, key in enumerate(keys):
            for field in ("t", "eye", "target"):
                if not isinstance(key, dict) or field not in key:
                    out.append(f"{where}.keys[{i}]: missing {field!r}")
        out += _check_focus(cam.get("focus"), where)
        if "lens_mm" not in cam:
            out.append(f"{where}.lens_mm: missing")
    return out


MAPPINGS = ("triplanar", "uv", "planar")


def _vec3(v: Any) -> list[float] | None:
    ok = isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v)
    return [float(x) for x in v] if ok else None


def _check_mapping(where: str, mat: dict) -> list[str]:
    """mapping is one of MAPPINGS; "planar" needs a non-zero planar_axis and a planar_up not parallel to it."""
    mapping = mat.get("mapping", "triplanar")
    if mapping not in MAPPINGS:
        return [f"{where}.mapping: {mapping!r} is not one of {', '.join(MAPPINGS)}"]
    out = [f"{where}.{k}: {mat[k]!r} is not 3 numbers" for k in ("planar_axis", "planar_origin", "planar_up")
           if mat.get(k) is not None and _vec3(mat[k]) is None]
    if out or mapping != "planar":
        return out
    ax, up = _vec3(mat.get("planar_axis", [0, 0, 1])), _vec3(mat.get("planar_up"))
    if sum(a * a for a in ax) < 1e-12:
        return [f"{where}.planar_axis must not be zero"]
    if up is not None:
        cross = (ax[1] * up[2] - ax[2] * up[1], ax[2] * up[0] - ax[0] * up[2], ax[0] * up[1] - ax[1] * up[0])
        if sum(c * c for c in cross) < 1e-12:
            out.append(f"{where}.planar_up {up} is parallel to planar_axis {ax}")
    return out


def _check_materials(look: dict[str, Any], presets: set[str] | None) -> list[str]:
    out = []
    for name, mat in (look.get("materials") or {}).items():
        preset = mat.get("preset") if isinstance(mat, dict) else None
        if preset is None:
            out.append(f'materials.{name}: missing "preset"')
            continue
        if presets is not None and preset not in presets:
            out.append(f"materials.{name}: unknown preset {preset!r} (known: {', '.join(sorted(presets))})")
        out += _check_mapping(f"materials.{name}", mat)
    return out


def _check_assign(look: dict[str, Any], presets: set[str] | None) -> list[str]:
    out = []
    materials = look.get("materials") or {}
    for i, rule in enumerate(look.get("assign") or []):
        where = f"assign[{i}]"
        if not isinstance(rule, dict) or "match" not in rule or "material" not in rule:
            out.append(f'{where}: needs "match" and "material"')
            continue
        err = _regex_problem(rule["match"], f"{where}.match")
        if err:
            out.append(err)
        mat = rule["material"]
        if mat not in materials and (presets is None or mat not in presets):
            out.append(f"{where}.material: {mat!r} is not defined under materials")
    for i, pattern in enumerate(look.get("hide") or []):
        err = _regex_problem(pattern, f"hide[{i}]")
        if err:
            out.append(err)
    return out


def _check_bevels(look: dict[str, Any]) -> list[str]:
    out = []
    for i, rule in enumerate(look.get("bevels") or []):
        where = f"bevels[{i}]"
        if not isinstance(rule, dict) or "match" not in rule or "bevel_m" not in rule:
            out.append(f'{where}: needs "match" and "bevel_m"')
            continue
        err = _regex_problem(rule["match"], f"{where}.match")
        if err:
            out.append(err)
        width = rule["bevel_m"]
        if not isinstance(width, (int, float)) or not 0 < width < 0.01:
            out.append(f"{where}.bevel_m: {width!r} is not a width in (0, 0.01) m")
    return out


def _skin_bodies(skin: dict[str, Any], root: Path) -> list[str]:
    """Body labels a skin follows, read from its weights file (JSON or npz)."""
    path = root / skin["weights"]
    if path.suffix == ".npz":
        import numpy as np

        with np.load(path) as data:
            return [str(x) for x in data["labels"]]
    return list(json.loads(path.read_text())["bodies"])


def _check_skins(look: dict[str, Any], bundle: Any, presets: set[str] | None) -> list[str]:
    out = []
    root = Path(getattr(bundle, "root", "."))
    labels = set(bundle.labels().get("bodies", []))
    for name, skin in (look.get("skins") or {}).items():
        out += _check_skin(f"skins.{name}", skin, root, labels, look, presets)
    return out


def _check_skin(where: str, skin: dict, root: Path, labels: set[str], look: dict,
                presets: set[str] | None) -> list[str]:
    """One skin: a mesh (obj/glb/gltf) that follows a weights file or, rigidly, one ``body``."""
    if "mesh" not in skin or not ("weights" in skin or "body" in skin):
        return [f'{where}: needs "mesh" and either "weights" or "body"']
    out = [f"{where}.{k}: {skin[k]!r} not found under the bundle root {root}"
           for k in ("mesh", "weights") if k in skin and not (root / skin[k]).is_file()]
    if Path(skin["mesh"]).suffix.lower() not in (".obj", ".glb", ".gltf"):
        out.append(f"{where}.mesh: expected .obj, .glb or .gltf, got {skin['mesh']!r}")
    if skin.get("local") and "body" not in skin:
        out.append(f'{where}.local: needs "body"')
    rep = skin.get("replicas", "proxy")
    if isinstance(rep, dict) and rep.get("mesh"):
        if not (root / rep["mesh"]).is_file():
            out.append(f"{where}.replicas.mesh: {rep['mesh']!r} not found under the bundle root")
        if "body" not in skin:
            out.append(f'{where}.replicas.mesh: needs "body"')
    elif rep == "skin" and "body" not in skin:
        out.append(f'{where}.replicas "skin": needs "body"')
    elif rep not in ("proxy", "skin", "none") and not isinstance(rep, dict):
        out.append(f'{where}.replicas: expected "proxy", "skin", "none" or {{"mesh": file}}, got {rep!r}')
    if "body" in skin and skin["body"] not in labels:
        out.append(f"{where}.body: no body labelled {skin['body']!r} in the bundle")
    defined = set(look.get("materials") or {}) | (presets or set())
    refs = [skin.get("material")] + list((skin.get("materials") or {}).values())
    refs += list((skin.get("material_map") or {}).values())
    out += [f"{where}: material {m!r} is not defined under materials" for m in refs if m and m not in defined]
    if "weights" in skin and (root / skin["weights"]).is_file():
        absent = [b for b in _skin_bodies(skin, root) if b not in labels]
        if absent:
            out.append(f"{where}.weights: bodies not in the bundle: {absent[:5]}")
    return out


def _skinned_bodies(look: dict[str, Any], bundle: Any) -> set[str]:
    """Body labels whose own shapes a skin hides (``hide_bodies``, default true)."""
    root = Path(getattr(bundle, "root", "."))
    hidden: set[str] = set()
    for skin in (look.get("skins") or {}).values():
        if not skin.get("hide_bodies", True):
            continue
        if "body" in skin:
            hidden.add(skin["body"])
        if (root / skin.get("weights", "")).is_file():
            hidden.update(_skin_bodies(skin, root))
    return hidden


def _check_liquid_contacts(look: dict[str, Any]) -> list[str]:
    """``liquids.<name>.contact``: container rules that snap the liquid surface onto the glass."""
    out = []
    for name, cfg in (look.get("liquids") or {}).items():
        for i, rule in enumerate((cfg or {}).get("contact") or []):
            where = f"liquids.{name}.contact[{i}]"
            if not isinstance(rule, dict) or "match" not in rule:
                out.append(f'{where}: needs "match" (container shape or body label regex)')
                continue
            err = _regex_problem(rule["match"], f"{where}.match")
            if err:
                out.append(err)
            for key, hi in (("distance_m", 0.02), ("overlap_m", 0.002)):
                v = rule.get(key)
                if v is not None and not (isinstance(v, (int, float)) and 0 <= v <= hi):
                    out.append(f"{where}.{key}: {v!r} is not in [0, {hi}] m")
    return out


def _check_files_and_glare(look: dict[str, Any], bundle: Any) -> list[str]:
    """``file:`` textures exist under the bundle root; ``render.glare`` names a known type."""
    root = Path(getattr(bundle, "root", "."))
    out = []
    for name, mat in (look.get("materials") or {}).items():
        tex = mat.get("texture") if isinstance(mat, dict) else None
        if isinstance(tex, str) and tex.startswith("file:") and not (root / tex[5:]).is_file():
            out.append(f"materials.{name}.texture: {tex[5:]!r} not found under the bundle root {root}")
    glare = (look.get("render") or {}).get("glare")
    if glare is not None and (not isinstance(glare, dict) or glare.get("type") not in ("fog_glow", "bloom")):
        out.append(f'render.glare: expected {{"type": "fog_glow" | "bloom", ...}}, got {glare!r}')
    return out


def _shape_candidates(bundle: Any) -> list[tuple[str, list[str]]]:
    """(display name, [labels to try in order]) for each shape: shape label, then its body label."""
    scene = getattr(bundle, "scene", None) or {}
    body_label = {b["id"]: b.get("label", "") for b in scene.get("bodies", [])}
    shapes = scene.get("shapes", [])
    if not shapes:
        return [(n, [n]) for n in bundle.labels().get("shapes", [])]
    result = []
    for s in shapes:
        if s.get("curve"):
            continue  # a body-chain curve's tube replaces these shapes
        names = [s.get("label", "")]
        if s.get("body", -1) in body_label:
            names.append(body_label[s["body"]])
        result.append((s.get("label") or f"shape#{s.get('id')}", names))
    return result


def _entity_names(bundle: Any, kind: str) -> list[str]:
    return list(bundle.labels().get(kind, []))


def _check_references(look: dict[str, Any], bundle: Any) -> list[str]:
    out = []
    for section, kind in (("liquids", "points"), ("points", "points"), ("instances", "instancers"),
                          ("curves", "curves"), ("surfaces", "surfaces")):
        known = set(_entity_names(bundle, kind))
        for name in look.get(section) or {}:
            if name not in known:
                out.append(f"{WARN}{section}.{name}: bundle has no {kind[:-1]} with that name")
    return out


def unmatched_labels(look: dict[str, Any], bundle: Any) -> list[str]:
    """Bundle labels that no assign rule (or explicit per-entity config) covers."""
    rules = [re.compile(r["match"]) for r in look.get("assign") or [] if isinstance(r, dict) and "match" in r]
    hide = [re.compile(p) for p in look.get("hide") or []]
    missing = []
    skinned = _skinned_bodies(look, bundle)
    for name, candidates in _shape_candidates(bundle):
        if any(h.search(c) for h in hide for c in candidates) or skinned.intersection(candidates[1:]):
            continue
        if not any(r.search(c) for r in rules for c in candidates):
            missing.append(f"shape {name}")
    explicit = {"points": look.get("liquids", {}) | look.get("points", {}), "instancers": look.get("instances", {}),
                "curves": look.get("curves", {}), "surfaces": look.get("surfaces", {})}
    for kind, configured in explicit.items():
        for name in _entity_names(bundle, kind):
            if name in configured or any(r.search(name) for r in rules):
                continue
            missing.append(f"{kind[:-1] if kind != 'surfaces' else 'surface'} {name}")
    return missing


def validate_look(look: dict[str, Any], bundle: Any) -> list[str]:
    """Problems with the look (errors) and unmatched labels or dangling names (prefixed 'warning: ')."""
    presets = preset_names()
    problems = _check_render(look) + _check_cameras(look) + _check_materials(look, presets)
    problems += _check_assign(look, presets) + _check_bevels(look)
    if problems:
        return problems
    problems += _check_skins(look, bundle, presets) + _check_props(look, bundle)
    problems += _check_files_and_glare(look, bundle) + _check_liquid_contacts(look)
    problems += _check_references(look, bundle)
    problems += [f"{WARN}no assign rule matches {label}" for label in unmatched_labels(look, bundle)]
    return problems


def parse_frames_arg(arg: str | None) -> tuple[int | None, int | None]:
    """Parse 'A:B' (A inclusive, B exclusive, either may be empty) or a single 'K' (= K:K+1)."""
    if not arg:
        return None, None
    try:
        if ":" not in arg:
            return int(arg), int(arg) + 1
        lo, hi = arg.split(":", 1)
        return (int(lo) if lo else None), (int(hi) if hi else None)
    except ValueError as exc:
        raise LookError(f"--frames expects A:B with integer ends, got {arg!r}") from exc


def frame_list(look: dict[str, Any], bundle: Any, frames_arg: str | None = None) -> list[int]:
    """Bundle frame indices: look frames start/end (both inclusive, negative counts from the end)/step, then
    the --frames A:B window (B exclusive), matching the Blender side."""
    spec = look.get("frames") or {}
    count = bundle.frame_count
    start = int(spec.get("start", 0))
    start = max(start + count if start < 0 else start, 0)
    end = int(spec.get("end", -1))
    end = min(end + count if end < 0 else end, count - 1)
    step = int(spec.get("step", 1))
    if step < 1:
        raise LookError(f"frames.step must be >= 1, got {step}")
    lo, hi = parse_frames_arg(frames_arg)
    frames = list(range(start, end + 1, step))
    return [k for k in frames if (lo is None or k >= lo) and (hi is None or k < hi)]


def camera_resolution(look: dict[str, Any], camera: str | None) -> tuple[int, int]:
    """Resolution of a camera (its override, else render.resolution)."""
    cam = (look.get("cameras") or {}).get(camera or "", {})
    res = cam.get("resolution") or (look.get("render") or {}).get("resolution") or DEFAULT_RESOLUTION
    return int(res[0]), int(res[1])


def estimate_cost(n_frames: int, resolution: tuple[int, int], samples: int,
                  usd_per_hour: float = SPOT_USD_PER_HOUR) -> CostEstimate:
    """Scale the 45 s/frame reference (1080x1920, 512 spp) by pixels and samples."""
    per_frame = REF_SECONDS * (resolution[0] * resolution[1] / REF_PIXELS) * (samples / REF_SAMPLES)
    hours = n_frames * per_frame / 3600.0
    return CostEstimate(n_frames, per_frame, hours, hours * usd_per_hour)
