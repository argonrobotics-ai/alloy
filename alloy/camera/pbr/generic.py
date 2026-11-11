# SPDX-License-Identifier: Apache-2.0
"""The data-driven "generic" material: Principled inputs from per-slot rows, not from generated source.

A pack material ``{"generic": {"base_color": [r, g, b], "roughness": 0.5, "metallic": 0.0, "specular": 0.5,
"texture": <key of pack textures> | None, "texture_tint": False, "transmission": 0.0, "ior": 1.5,
"absorption": [r, g, b] | None}}`` is
rendered by ONE branch of the generated dispatch that reads its slot's rows from the module's ``Images`` struct, so a
scene made only of generic materials always produces byte-identical module source: another object, colour or
texture is new data, never new code (no codegen, no recompile).

Rows (``Images.gen_params``, three ``wp.vec4`` per slot): ``[r, g, b, roughness]``, ``[metallic, texture id or -1,
specular IOR level, texture tint 0 | 1]``, ``[transmission, IOR, 0, 0]``. ``absorption`` (Beer-Lambert sigma per
metre, as a Volume Absorption node) goes to the renderer's volume table (``dispatch.generate_source``), not to
the rows. Every texture
of every generic slot lives in one flat atlas: ``Images.tex_data`` (float, rows bottom first, channels interleaved)
and ``Images.tex_meta`` (``wp.vec4i``: offset, width, height, channels).
The texture replaces the base colour (sRGB decoded, bilinear, REPEAT, Blender UV convention: origin bottom-left);
with ``texture_tint`` it is multiplied by it instead (a glTF baseColorFactor times its baseColorTexture).

Example::

    arrays = build_generic_arrays(pack.static, names, textures, "cuda:0")   # dict of Images field -> wp.array
"""
from __future__ import annotations

import numpy as np
import warp as wp

GENERIC_FIELDS = {"gen_kind": wp.int32, "gen_params": wp.vec4, "tex_meta": wp.vec4i, "tex_data": float}

# Module source shared by every generated module that contains generic materials (``Images`` is defined before it).
DISPATCH_SOURCE = '''
@wp.func
def mat_generic(mid: int, uv: wp.vec3, im: Images) -> pr.PrincipledParams:
    q = pr.principled_default()
    a = im.gen_params[3 * mid]
    b = im.gen_params[3 * mid + 1]
    t = im.gen_params[3 * mid + 2]
    c = wp.vec3(a[0], a[1], a[2])
    tid = int(b[1])
    if tid >= 0:
        tex = nd.srgb_to_linear3(gn.atlas_bilinear(im.tex_data, im.tex_meta[tid], uv[0], uv[1]))
        if b[3] > 0.5:
            tex = wp.cw_mul(tex, c)
        c = tex
    q.base_color = c
    q.roughness = a[3]
    q.metallic = b[0]
    q.specular_ior_level = b[2]
    q.transmission_weight = t[0]
    q.ior = t[1]
    return q
'''


def is_generic(mat: dict) -> bool:
    """True for a pack material that carries a ``generic`` parameter block instead of a node tree."""
    return "generic" in mat


@wp.func
def _wrap(i: int, n: int) -> int:
    return ((i % n) + n) % n


@wp.func
def _texel(data: wp.array(dtype=float), meta: wp.vec4i, ix: int, iy: int) -> wp.vec3:
    w = meta[1]
    h = meta[2]
    nc = meta[3]
    base = meta[0] + (_wrap(iy, h) * w + _wrap(ix, w)) * nc
    r = data[base]
    if nc >= 3:
        return wp.vec3(r, data[base + 1], data[base + 2])
    return wp.vec3(r, r, r)


@wp.func
def atlas_bilinear(data: wp.array(dtype=float), meta: wp.vec4i, u: float, v: float) -> wp.vec3:
    """Bilinear sample with REPEAT wrap of the atlas image ``meta`` at (u, v), v up (row 0 = bottom)."""
    x = u * float(meta[1]) - 0.5
    y = v * float(meta[2]) - 0.5
    fx = wp.floor(x)
    fy = wp.floor(y)
    tx = x - fx
    ty = y - fy
    ix = int(fx)
    iy = int(fy)
    a = _texel(data, meta, ix, iy) * (1.0 - tx) + _texel(data, meta, ix + 1, iy) * tx
    b = _texel(data, meta, ix, iy + 1) * (1.0 - tx) + _texel(data, meta, ix + 1, iy + 1) * tx
    return a * (1.0 - ty) + b * ty


def _atlas(images: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(flat float32 data, (n, 4) int32 meta) of images shaped (H, W, C), row 0 = bottom."""
    meta, parts, off = [], [], 0
    for im in images:
        h, w, c = im.shape
        meta.append((off, w, h, c))
        parts.append(np.ascontiguousarray(im, np.float32).reshape(-1))
        off += h * w * c
    data = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    return data, np.asarray(meta, np.int32).reshape(-1, 4) if meta else np.zeros((1, 4), np.int32)


def build_generic_arrays(static: dict, names: list[str], load_images, device: str) -> dict:
    """Images fields for the scene's materials (``names`` order; one extra default slot). ``load_images(files)`` ->
    list of float32 numpy arrays (H, W, C), row 0 = bottom (``shade_data.read_textures`` with the pack root bound)."""
    kind = np.zeros(len(names) + 1, np.int32)
    rows = np.zeros((3 * (len(names) + 1), 4), np.float32)
    rows[2::3, 1] = 1.5  # IOR of the slots without a generic block (principled_default)
    files, tex_id = [], {}
    for k, nm in enumerate(names):
        g = static["materials"][nm].get("generic")
        if g is None:
            continue
        key = g.get("texture")
        if key is not None and key not in tex_id:
            tex_id[key] = len(files)
            files.append(static["textures"][key]["file"])
        kind[k] = 1
        rows[3 * k] = (*[float(x) for x in g["base_color"][:3]], float(g["roughness"]))
        rows[3 * k + 1] = (float(g.get("metallic", 0.0)), float(tex_id[key]) if key is not None else -1.0,
                           float(g.get("specular", 0.5)), 1.0 if g.get("texture_tint") and key is not None else 0.0)
        rows[3 * k + 2] = (float(g.get("transmission", 0.0)), float(g.get("ior", 1.5)), 0.0, 0.0)
    imgs = list(load_images(files)) if files else []
    data, meta = _atlas(imgs)
    return {"gen_kind": wp.array(kind, dtype=wp.int32, device=device),
            "gen_params": wp.array(rows, dtype=wp.vec4, device=device),
            "tex_meta": wp.array(meta, dtype=wp.vec4i, device=device),
            "tex_data": wp.array(data, dtype=float, device=device)}
