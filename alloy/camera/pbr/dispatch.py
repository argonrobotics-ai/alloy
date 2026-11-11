# SPDX-License-Identifier: Apache-2.0
"""Generate one Warp module with every material of a scene: per-material shader-input functions (``codegen``) plus
the dispatchers the path tracer calls.

The module defines
    Images        struct with one wp.array3d(float) per texture the materials use (row 0 = bottom, /255 or /65535)
    mat_params(mid, p_obj, n_obj, uv, col, rnd, im) -> pr.PrincipledParams   (all Principled inputs except the normals)
    mat_height(mid, p_obj, n_obj, uv, col, rnd, im) -> float                  (Bump height; 0 without a Bump node)
    MAT_PARAMS[k], MAT_HEIGHT[k]: material k only (same signature), for material-specialised kernels.
Materials with a ``generic`` block (see ``generic.py``) share one data-driven branch (``mat_generic``) and add the
fields gen_kind / gen_params / tex_meta / tex_data to ``Images``; a module of generic materials only is the same text for
any number of them. Material index = position in ``names``; index len(names) is Cycles' default surface (grey 0.8 diffuse) for empty
slots. Generated files are cached by content hash under ``$ALLOY_CACHE/camera/gen`` (default ~/.cache/alloy).

Example::

    mod, info = generate(pack.static, pack.static["materials_used_by_meshes"])
    kernel = make_kernel(mod)          # alloy.camera.pbr.integrator
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

from ..errors import CameraError
from .codegen import gen_material
from .generic import DISPATCH_SOURCE as GENERIC_SOURCE
from .volume import absorption
from .generic import is_generic

# Principled input name -> (PrincipledParams field, kind)
CONST_INPUTS = {
    "Base Color": ("base_color", "vec3"), "Metallic": ("metallic", "float"), "Roughness": ("roughness", "float"),
    "IOR": ("ior", "float"), "Specular IOR Level": ("specular_ior_level", "float"),
    "Specular Tint": ("specular_tint", "vec3"), "Anisotropic": ("anisotropic", "float"),
    "Anisotropic Rotation": ("anisotropic_rotation", "float"), "Sheen Weight": ("sheen_weight", "float"),
    "Sheen Roughness": ("sheen_roughness", "float"), "Sheen Tint": ("sheen_tint", "vec3"),
    "Coat Weight": ("coat_weight", "float"), "Coat Roughness": ("coat_roughness", "float"),
    "Coat IOR": ("coat_ior", "float"), "Coat Tint": ("coat_tint", "vec3"),
    "Transmission Weight": ("transmission_weight", "float"), "Subsurface Weight": ("subsurface_weight", "float"),
    "Subsurface Radius": ("subsurface_radius", "vec3"), "Subsurface Scale": ("subsurface_scale", "float"),
    "Subsurface Anisotropy": ("subsurface_anisotropy", "float"),
}
UNSUPPORTED_IF_NONZERO = ("Emission Strength", "Thin Film Thickness")
DEFAULT_SURFACE = ["    q.base_color = wp.vec3(0.8, 0.8, 0.8)", "    q.specular_ior_level = 0.0"]
SIG = "(mid: int, p: wp.vec3, n: wp.vec3, uv: wp.vec3, col: wp.vec3, rnd: float, im: Images)"
HEADER = ["# generator 2: uv and col arguments (a new source text = a new cache file)", "import warp as wp", "from alloy.camera.pbr import nodes as nd", "from alloy.camera.pbr.bsdf import principled as pr",
          "from alloy.camera.pbr import generic as gn", ""]


def cache_dir() -> Path:
    """Where generated material modules are written (``$ALLOY_CACHE/camera/gen``, default ~/.cache/alloy)."""
    root = Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy"))
    return root / "camera" / "gen"


def _lit(kind: str, v) -> str:
    if kind == "vec3":
        return f"wp.vec3({float(v[0])!r}, {float(v[1])!r}, {float(v[2])!r})"
    return repr(float(v))


def _call(fn: dict, img_field: dict) -> str:
    args = ", ".join(["p", "n", "uv", "col", "rnd"] + [f"im.{img_field[x]}" for x in fn["images"]])
    return f"{fn['name']}({args})"


def is_transmissive(mat: dict) -> bool:
    """True when a material's Principled Transmission Weight is linked or not 0 (glass, water, clear plastic)."""
    if "generic" in mat:
        return float(mat["generic"].get("transmission", 0.0)) != 0.0
    tw = mat.get("principled", {}).get("inputs", {}).get("Transmission Weight", {"linked": False, "value": 0.0})
    return bool(tw["linked"]) or float(tw["value"]) != 0.0


def _generic_volume(mat: dict) -> dict | None:
    """A generic material's absorption (sigma per metre, rgb) in ``volume.absorption``'s form, or None."""
    sigma = mat["generic"].get("absorption")
    return None if sigma is None else {"sigma_a": [float(x) for x in sigma]}


def _check_material(nm: str, mat: dict) -> None:
    pin = mat["principled"]["inputs"]
    if mat["principled"].get("distribution", "MULTI_GGX") not in ("MULTI_GGX", "GGX"):
        raise CameraError(f"material {nm}: distribution {mat['principled']['distribution']}",
                          "use the GGX or Multiscatter GGX distribution")
    for bad in UNSUPPORTED_IF_NONZERO:
        if bad in pin and (pin[bad]["linked"] or float(pin[bad]["value"]) != 0.0):
            raise CameraError(f"material {nm} uses {bad}",
                              f"set {bad} to 0 (emission and thin film are not supported yet)")
    sss = pin.get("Subsurface Weight", {"linked": False, "value": 0.0})
    method = mat["principled"].get("subsurface_method", "RANDOM_WALK")
    if (sss["linked"] or float(sss["value"]) != 0.0) and method != "RANDOM_WALK":
        raise CameraError(f"material {nm}: subsurface method {method}",
                          "use the Random Walk subsurface method (Christensen-Burley and Skin are not supported)")
    if pin.get("Tangent", {}).get("linked"):
        raise CameraError(f"material {nm} has a linked Tangent", "unlink the Principled Tangent input")


def _param_body(nm: str, mat: dict, fns: dict, img_field: dict) -> list[str]:
    """Statements setting q.<field> for one material (4-space indent)."""
    pin = mat["principled"]["inputs"]
    out = ["    q.single_scatter_ggx = 1"] if mat["principled"].get("distribution") == "GGX" else []
    for inp, (field, kind) in CONST_INPUTS.items():
        if inp not in pin:
            continue
        key = inp.replace(" ", "_")
        if pin[inp]["linked"]:
            if key not in fns:
                raise CameraError(f"material {nm}: linked {inp} could not be compiled", "simplify its node graph")
            out.append(f"    q.{field} = {_call(fns[key], img_field)}")
        else:
            out.append(f"    q.{field} = {_lit(kind, pin[inp]['value'])}")
    return out


GENERIC_BRANCH = ["    if im.gen_kind[mid] != 0:  # every generic slot: its rows in Images",
                  "        return mat_generic(mid, uv, im)"]


def _dispatchers(bodies: list[list[str]], heights: list[str | None], generic: list[bool]) -> list[str]:
    """mat_params / mat_height over every material (if / elif chains), then per-material copies. Generic slots share
    one branch; a module of generic slots only has no per-material functions (its text does not depend on their number)."""
    lines = ["", "", "@wp.func", f"def mat_params{SIG} -> pr.PrincipledParams:", "    q = pr.principled_default()"]
    if any(generic):
        lines += GENERIC_BRANCH
    first = True
    for k, body in enumerate(bodies):
        if generic[k]:
            continue
        lines.append(f"    {'if' if first else 'elif'} mid == {k}:")
        first = False
        lines += ["    " + ln for ln in body] or ["        pass"]
    lines += ["    else:  # Cycles' default surface", *("    " + ln for ln in DEFAULT_SURFACE), "    return q", "", "",
              "@wp.func", f"def mat_height{SIG} -> float:"]
    for k, h in enumerate(heights):
        if h:
            lines.append(f"    if mid == {k}:\n        return {h}")
    lines.append("    return 0.0")
    if all(generic):
        return lines + ["", "", "MAT_PARAMS = []", "MAT_HEIGHT = []"]
    for k, body in enumerate(bodies + [DEFAULT_SURFACE]):
        h = heights[k] if k < len(heights) else None
        if k < len(generic) and generic[k]:
            body = ["    q = mat_generic(mid, uv, im)"]
        lines += ["", "", "@wp.func", f"def mat_params_{k}{SIG} -> pr.PrincipledParams:", "    q = pr.principled_default()",
                  *body, "    return q", "", "", "@wp.func", f"def mat_height_{k}{SIG} -> float:",
                  f"    return {h}" if h else "    return 0.0"]
    n = len(bodies) + 1
    lines += ["", "", f"MAT_PARAMS = [{', '.join(f'mat_params_{k}' for k in range(n))}]",
              f"MAT_HEIGHT = [{', '.join(f'mat_height_{k}' for k in range(n))}]"]
    return lines


def generate_source(static: dict, names: list[str], uv_names=(), col_names=()) -> tuple[str, dict]:
    """(module source, info) with info = {images: [file], srgb: [bool], bump: [dict|None], bevel: [dict|None],
    volume: [dict|None], transmissive: [bool]} (volume: the material's absorption, ``volume.absorption``).
    ``uv_names`` / ``col_names``: the meshes' UV layer and colour attribute names (Attribute nodes of those names)."""
    codes, metas = [], []
    generic = [is_generic(static["materials"][nm]) for nm in names]
    for nm, gen in zip(names, generic):
        if gen:
            codes.append("")
            metas.append({"functions": {}, "images": [], "bump": None, "bevel": None})
            continue
        _check_material(nm, static["materials"][nm])
        code, meta = gen_material(nm, static["materials"][nm], static["textures"], uv_names, col_names)
        codes.append(code)
        metas.append(meta)
    images: list[dict] = []
    for meta in metas:
        images += [im for im in meta["images"] if im["key"] not in [x["key"] for x in images]]
    img_field = {im["key"]: f"i{k}" for k, im in enumerate(images)}
    lines = HEADER + [c for c in codes if c] + ["", "", "@wp.struct", "class Images:"]
    lines += [f"    {img_field[im['key']]}: wp.array3d(dtype=float)" for im in images]
    if any(generic):
        lines += ["    gen_kind: wp.array(dtype=wp.int32)", "    gen_params: wp.array(dtype=wp.vec4)",
                  "    tex_meta: wp.array(dtype=wp.vec4i)", "    tex_data: wp.array(dtype=float)"]
    lines += ["    dummy: int"] if not images and not any(generic) else []
    if any(generic):
        lines += GENERIC_SOURCE.splitlines()
    bodies = [_param_body(nm, static["materials"][nm], meta["functions"], img_field) if not gen else []
              for nm, meta, gen in zip(names, metas, generic)]
    heights = [_call(m["functions"]["height"], img_field) if "height" in m["functions"] else None for m in metas]
    lines += _dispatchers(bodies, heights, generic)
    info = {"images": [im["file"] for im in images], "srgb": [im["srgb"] for im in images],
            "bump": [m["bump"] for m in metas], "bevel": [m["bevel"] for m in metas],
            "has_height": any(heights), "generic": generic,
            "volume": [_generic_volume(static["materials"][nm]) if gen else absorption(static["materials"][nm], nm)
                       for nm, gen in zip(names, generic)],
            "transmissive": [is_transmissive(static["materials"][nm]) for nm in names]}
    return "\n".join(lines) + "\n", info


def generate(static: dict, names: list[str], uv_names=(), col_names=()) -> tuple[object, dict]:
    """Write (once) and import the material module for ``names`` of a pack's static data; returns (module, info)."""
    src, info = generate_source(static, names, uv_names, col_names)
    tag = hashlib.sha1(src.encode()).hexdigest()[:12]
    name = f"alloy_camera_materials_{tag}"
    if name in sys.modules:
        return sys.modules[name], info
    path = cache_dir() / f"{name}.py"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(f'"""GENERATED by alloy.camera.pbr.dispatch for materials {names}. Do not edit."""\n' + src)
        os.replace(tmp, path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, info
