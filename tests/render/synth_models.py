# SPDX-License-Identifier: Apache-2.0
"""Display models for the synthetic RJ45 scene, written with numpy only: a glTF 2.0 .glb plug (clear housing, gold
blades, 8 wires, 3 PBR materials), a switch-port .obj with usemtl groups and UVs, and a jacket-print PNG with alpha.
Also a separate glass-of-water bundle (write_glass, glass_look) for the liquid contact A/B.
"""

from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

import numpy as np

# Plug in its body frame (metres): +Y is the insertion direction, Z up.
PLUG_W, PLUG_H, PLUG_L = 0.0117, 0.008, 0.021
PITCH = 0.00102


def box(lo, hi) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flat-shaded box: 24 vertices (4 per face), normals, UVs (0..1 per face) and 12 triangles."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    verts, norms, uvs, tris = [], [], [], []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        for sign in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sign
            quad = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.where(n > 0, hi, lo).astype(float)
                p[u] = (lo[u], hi[u])[a]
                p[v] = (lo[v], hi[v])[b]
                quad.append(p)
            base = len(verts)
            outward = np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), n) > 0
            order = [0, 1, 2, 0, 2, 3] if outward else [0, 2, 1, 0, 3, 2]
            verts += quad
            norms += [n] * 4
            uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
            tris += [base + i for i in order]
    return np.asarray(verts), np.asarray(norms), np.asarray(uvs, float), np.asarray(tris, np.int64).reshape(-1, 3)


def cylinder_y(x: float, z: float, r: float, y0: float, y1: float, n: int = 16):
    """Smooth tube along +Y (no caps)."""
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ring = np.stack([x + r * np.cos(a), np.zeros(n), z + r * np.sin(a)], 1)
    verts = np.concatenate([ring + [0, y0, 0], ring + [0, y1, 0]])
    norms = np.concatenate([np.stack([np.cos(a), np.zeros(n), np.sin(a)], 1)] * 2)
    uvs = np.concatenate([np.stack([a / (2 * math.pi), np.zeros(n)], 1), np.stack([a / (2 * math.pi), np.ones(n)], 1)])
    i = np.arange(n)
    j = (i + 1) % n
    tris = np.concatenate([np.stack([i, j, j + n], 1), np.stack([i, j + n, i + n], 1)])
    return verts, norms, uvs, tris


def _merge(parts: list) -> tuple:
    v, n, uv, t, base = [], [], [], [], 0
    for pv, pn, puv, pt in parts:
        v.append(pv), n.append(pn), uv.append(puv), t.append(pt + base)
        base += len(pv)
    return np.concatenate(v), np.concatenate(n), np.concatenate(uv), np.concatenate(t)


def rj45_parts() -> dict[str, tuple]:
    """Material name -> (verts, normals, uvs, tris) of the plug, in the plug body's local frame."""
    housing = box([-PLUG_W / 2, -PLUG_L / 2, -PLUG_H / 2], [PLUG_W / 2, PLUG_L / 2, PLUG_H / 2])
    xs = (np.arange(8) - 3.5) * PITCH
    top = PLUG_H / 2 - 0.0002
    blades = _merge([box([x - 0.000175, PLUG_L / 2 - 0.0042, top - 0.0009], [x + 0.000175, PLUG_L / 2 - 0.0006, top])
                     for x in xs])
    wires = _merge([cylinder_y(x, -0.0012 + 0.0004 * (k % 2), 0.00045, -PLUG_L / 2 - 0.004, PLUG_L / 2 - 0.0045)
                    for k, x in enumerate(xs)])
    return {"pc_clear": housing, "gold_blade": blades, "wire": wires}


MATERIALS = {
    "pc_clear": {"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0.0,
                                          "roughnessFactor": 0.05},
                 "extensions": {"KHR_materials_transmission": {"transmissionFactor": 1.0},
                                "KHR_materials_ior": {"ior": 1.585}}},
    "gold_blade": {"pbrMetallicRoughness": {"baseColorFactor": [1.0, 0.77, 0.34, 1], "metallicFactor": 1.0,
                                            "roughnessFactor": 0.15}},
    "wire": {"pbrMetallicRoughness": {"baseColorFactor": [0.9, 0.45, 0.2, 1], "metallicFactor": 0.0,
                                      "roughnessFactor": 0.4}},
}


def write_glb(path: Path, parts: dict[str, tuple], materials: dict[str, dict]) -> None:
    """glTF 2.0 binary: one mesh, one primitive per material. Blender Z-up is stored as glTF Y-up (x, z, -y)."""
    to_gltf = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float64)
    blob, views, accessors, prims, mats = bytearray(), [], [], [], []

    def add(arr: np.ndarray, comp: int, kind: str, target: int, minmax: bool = False) -> int:
        data = arr.astype(np.float32 if comp == 5126 else np.uint32).tobytes()
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": kind}
        if minmax:
            acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
        accessors.append(acc)
        return len(accessors) - 1

    for name, (v, n, uv, t) in parts.items():
        attrs = {"POSITION": add(v @ to_gltf.T, 5126, "VEC3", 34962, True), "NORMAL": add(n @ to_gltf.T, 5126, "VEC3",
                                                                                         34962),
                 "TEXCOORD_0": add(np.stack([uv[:, 0], 1.0 - uv[:, 1]], 1), 5126, "VEC2", 34962)}
        prims.append({"attributes": attrs, "indices": add(t.ravel(), 5125, "SCALAR", 34963), "material": len(mats)})
        mats.append({"name": name, **materials[name]})
    doc = {"asset": {"version": "2.0", "generator": "alloy synth"}, "scene": 0, "scenes": [{"nodes": [0]}],
           "nodes": [{"mesh": 0, "name": "plug"}], "meshes": [{"name": "plug", "primitives": prims}],
           "materials": mats, "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
           "extensionsUsed": ["KHR_materials_transmission", "KHR_materials_ior"]}
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)


def switch_port_obj(path: Path, centre) -> None:
    """Port housing (5 walls round an opening facing -Y), metal shield on top and a green LED: usemtl groups."""
    cx, cy, cz = centre
    w, h, d, t = 0.016, 0.014, 0.022, 0.0016
    walls = [([-w / 2, 0, -h / 2], [-w / 2 + t, d, h / 2]), ([w / 2 - t, 0, -h / 2], [w / 2, d, h / 2]),
             ([-w / 2, 0, -h / 2], [w / 2, d, -h / 2 + t]), ([-w / 2, d - t, -h / 2], [w / 2, d, h / 2])]
    groups = {"housing": [box(lo, hi) for lo, hi in walls],
              "shield": [box([-w / 2 - 0.0003, -0.0003, h / 2 - t], [w / 2 + 0.0003, d, h / 2 + 0.0003])],
              "led": [box([w / 2 - 0.004, -0.0006, h / 2 - t - 0.0022], [w / 2 - 0.0022, 0.0002, h / 2 - t - 0.0008])]}
    lines, nv, nt = [], 0, 0
    for name, boxes in groups.items():
        lines.append(f"usemtl {name}")
        for v, _, uv, tris in boxes:
            lines += [f"v {cx + p[0]:.6f} {cy + p[1]:.6f} {cz + p[2]:.6f}" for p in v]
            lines += [f"vt {a:.4f} {b:.4f}" for a, b in uv]
            lines += ["f " + " ".join(f"{nv + i + 1}/{nt + i + 1}" for i in tri) for tri in tris]
            nv, nt = nv + len(v), nt + len(uv)
    path.write_text("\n".join(lines) + "\n")


GLYPHS = {  # 3x5 bitmap font, rows top to bottom
    "C": ["111", "100", "100", "100", "111"], "A": ["010", "101", "111", "101", "101"],
    "T": ["111", "010", "010", "010", "010"], "6": ["111", "100", "111", "101", "111"],
    "U": ["101", "101", "101", "101", "111"], "P": ["111", "101", "111", "100", "100"],
    "2": ["111", "001", "111", "100", "111"], "4": ["101", "101", "111", "001", "001"],
    "W": ["101", "101", "101", "111", "101"], "G": ["111", "100", "101", "101", "111"],
    "E": ["111", "100", "111", "100", "111"], " ": ["000"] * 5, "-": ["000", "000", "111", "000", "000"],
}


def jacket_print_png(path: Path, text: str = "CAT6 UTP 24AWG - ", px: int = 4, band: int = 10) -> None:
    """RGBA print: one line of white letters on transparency, 1/band of the height (a stripe along one side of the
    cable), repeating along U (the cable length)."""
    cols = sum(4 for _ in text)
    img = np.zeros((7 * px * band // 1, cols * px, 4), np.uint8)
    x = 0
    for ch in text:
        for r, row in enumerate(GLYPHS.get(ch, GLYPHS[" "])):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                for top in (0, img.shape[0] // 2):  # two lines on opposite sides of the jacket
                    img[top + (r + 1) * px:top + (r + 2) * px, (x + c) * px:(x + c + 1) * px] = (235, 235, 230, 255)
        x += 4
    write_png(path, img)


def write_png(path: Path, rgba: np.ndarray) -> None:
    h, w = rgba.shape[:2]
    raw = b"".join(b"\0" + rgba[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


PORT_CENTRE = (-0.12, -0.02, 0.007)


def plug_poses(t: np.ndarray) -> np.ndarray:
    """[T, 7] plug pose: eases +Y into the port (at rest at t = 0) and clicks home at t = 0.1 s."""
    x = np.clip(t / 0.1, 0.0, 1.0)
    ramp = x * x * (3.0 - 2.0 * x)
    out = np.zeros((len(t), 7))
    out[:, 0], out[:, 1], out[:, 2], out[:, 6] = PORT_CENTRE[0], -0.045 + 0.03 * ramp, PORT_CENTRE[2], 1.0
    return out


def rj45_cable(t: np.ndarray, k: int = 20) -> np.ndarray:
    """[T, k, 3] jacket centreline leaving the plug's back and drooping to the table."""
    plug = plug_poses(t)
    s = np.arange(k) * 0.006
    out = np.empty((len(t), k, 3), np.float32)
    for i in range(len(t)):
        x, y, z = plug[i, :3]
        droop = np.maximum(z - 1.2 * s ** 1.4, 0.003)
        out[i] = np.stack([x + 0.004 * np.sin(s * 40), y - PLUG_L / 2 - 0.003 - s, droop], 1)
    return out


def add_rj45(root: Path, t: np.ndarray, bodies: list, shapes: list, body_q: np.ndarray, shape_fn) -> np.ndarray:
    """Plug and port bodies (box proxies the skins hide), the cable curve, and the display files under extra/."""
    (root / "extra").mkdir(exist_ok=True)
    write_glb(root / "extra" / "rj45_plug.glb", rj45_parts(), MATERIALS)
    switch_port_obj(root / "extra" / "switch_port.obj", PORT_CENTRE)
    jacket_print_png(root / "extra" / "jacket_print.png")
    still = np.tile([*PORT_CENTRE, 0, 0, 0, 1], (len(t), 1))
    for label, pose, half in (("rj45_plug", plug_poses(t), [PLUG_W / 2, PLUG_L / 2, PLUG_H / 2]),
                              ("switch_body", still, [0.008, 0.011, 0.007])):
        bodies.append({"id": len(bodies), "label": label, "world": 0})
        shapes.append(shape_fn(len(shapes), f"{label}_proxy", len(bodies) - 1, 0, "box", half,
                               [0, 0, 0, 0, 0, 0, 1], [0.5, 0.5, 0.5]))
        body_q = np.concatenate([body_q, pose[:, None, :].astype(np.float32)], axis=1)
    np.save(root / "frames" / "curve_rj45_cable.npy", rj45_cable(t))
    return body_q


def rj45_look(look: dict) -> dict:
    """Look entries for the RJ45 scene: glb + obj skins, file texture, keyed LED, glare and a macro camera."""
    look["materials"].update({
        "plug_pc": {"preset": "polycarbonate_clear", "imperfection": "ambientcg:Fingerprints002",
                    "imperfection_scale_m": 0.03, "imperfection_strength": 0.35},
        "port_led": {"preset": "led_green", "keys": [{"t": 0.0, "emission_strength": 0.0},
                                                     {"t": 0.1, "emission_strength": 10.0}]},
        "jacket": {"preset": "pvc_jacket_matte", "texture": "file:extra/jacket_print.png", "texture_alpha": True}})
    look.setdefault("skins", {}).update({
        "plug": {"mesh": "extra/rj45_plug.glb", "body": "rj45_plug", "local": True,
                 "material_map": {"pc_clear": "plug_pc", "gold_blade": "gold_plated"}},
        "port": {"mesh": "extra/switch_port.obj", "body": "switch_body",
                 "materials": {"housing": "abs_switch_housing", "shield": "steel_brushed", "led": "port_led"}}})
    look["curves"]["rj45_cable"] = {"material": "jacket", "uv_scale_m": 0.04, "resolution": 4}
    look["render"]["glare"] = {"type": "fog_glow", "threshold": 1.0, "size": 6}
    look["cameras"]["rj45"] = {"resolution": [1080, 1920], "lens_mm": 100, "fstop": 5.6,
                               "focus": {"point": [-0.117, -0.039, 0.011]},
                               "keys": [{"t": 0.0, "eye": [-0.068, -0.094, 0.05], "target": [-0.12, -0.034, 0.006]}]}
    look["cameras"]["port"] = {"resolution": [1080, 1920], "lens_mm": 100, "fstop": 5.6,
                               "focus": {"point": [-0.112, -0.021, 0.0115]},
                               "keys": [{"t": 0.0, "eye": [-0.07, -0.085, 0.04], "target": [-0.116, -0.02, 0.009]}]}
    look["cameras"]["jacket"] = {"resolution": [1080, 1920], "lens_mm": 100, "fstop": 5.6,
                                 "focus": {"point": [-0.12, -0.045, 0.005]},
                                 "keys": [{"t": 0.0, "eye": [-0.075, -0.085, 0.045], "target": [-0.12, -0.045, 0.004]}]}
    return look


# Glass of water (metres): thick-walled tumbler, water particles kept GLASS_GAP from the walls as a solver would.
GLASS_RO, GLASS_RI, GLASS_H, GLASS_BASE, WATER_TOP = 0.030, 0.028, 0.085, 0.005, 0.060
WATER_R = 0.0008
GLASS_GAP = 2.6 * WATER_R  # with radius_scale 1.6 the un-snapped surface stops ~1 particle radius short


def revolve(profile: list, n: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Closed surface of revolution about +Z from an (r, z) polyline; solid on the left of travel (outward faces)."""
    verts, rings = [], []
    ang = np.arange(n) * 2 * math.pi / n
    for r, z in profile:
        rings.append(list(range(len(verts), len(verts) + (1 if r == 0 else n))))
        verts += [[0.0, 0.0, z]] if r == 0 else [[r * math.cos(a), r * math.sin(a), z] for a in ang]
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        for k in range(n):
            q = [a[k % len(a)], a[(k + 1) % n % len(a)], b[(k + 1) % n % len(b)], b[k % len(b)]]
            faces += [f for f in ([q[0], q[1], q[2]], [q[0], q[2], q[3]]) if len(set(f)) == 3]
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


def glass_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Tumbler: flat base, straight walls, rounded rim."""
    w, rm = (GLASS_RO - GLASS_RI) / 2, (GLASS_RO + GLASS_RI) / 2
    rim = [(rm + w * math.cos(a), GLASS_H + w * math.sin(a)) for a in np.linspace(0, math.pi, 9)]
    return revolve([(0.0, 0.0), (GLASS_RO, 0.0)] + rim + [(GLASS_RI, GLASS_BASE), (0.0, GLASS_BASE)])


def water_points(seed: int = 3) -> np.ndarray:
    """Particles on concentric rings, outermost GLASS_GAP from the wall (a solver's uniform wall layer), in layers
    from GLASS_GAP above the base to WATER_TOP, with small jitter for particle-scale bumps."""
    rng = np.random.default_rng(seed)
    step = 2 * WATER_R
    rings = [np.zeros((1, 2))]
    for r in np.arange(GLASS_RI - GLASS_GAP, 0.5 * step, -step):
        n = max(int(round(2 * math.pi * r / step)), 1)
        a = (np.arange(n) + 0.5 * (len(rings) % 2)) * 2 * math.pi / n
        rings.append(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    disc = np.concatenate(rings)
    zs = np.arange(GLASS_BASE + GLASS_GAP, WATER_TOP - WATER_R + 1e-9, step)
    p = np.concatenate([np.column_stack([disc, np.full(len(disc), z)]) for z in zs])
    p += rng.uniform(-0.1, 0.1, p.shape) * step
    r = np.hypot(p[:, 0], p[:, 1])
    p[:, :2] *= np.minimum(1.0, (GLASS_RI - GLASS_GAP) / np.maximum(r, 1e-9))[:, None]
    p[:, 2] = np.maximum(p[:, 2], GLASS_BASE + GLASS_GAP)
    return p.astype(np.float32)


def glass_slide(t: np.ndarray, t0: float = 0.05, speed: float = 0.1) -> np.ndarray:
    """Glass x offset: still, then sliding at `speed` (the moving-container case, with motion blur)."""
    return speed * np.maximum(t - t0, 0.0)


def write_glass(root: Path, frames: int = 8, fps: float = 60.0) -> dict:
    """Bundle: table, a glass body (mesh shape) and its water particles riding along; returns scene.json."""
    for sub in ("frames", "meshes"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    t = np.arange(frames) / fps
    np.save(root / "frames" / "t.npy", t)
    gv, gf = glass_mesh()
    np.savez(root / "meshes" / "glass.npz", vertices=gv, faces=gf)
    dx = glass_slide(t)
    body_q = np.zeros((frames, 1, 7), np.float32)
    body_q[:, 0, 0], body_q[:, 0, 6] = dx, 1.0
    np.save(root / "frames" / "body_q.npy", body_q)
    water = water_points()
    pts = np.repeat(water[None], frames, 0)
    pts[:, :, 0] += dx[:, None]
    vel = np.zeros_like(pts)
    vel[:, :, 0] = np.gradient(dx, t)[:, None] if frames > 1 else 0.0
    np.save(root / "frames" / "pts_water.npy", pts.astype(np.float32))
    np.save(root / "frames" / "vel_water.npy", vel.astype(np.float32))
    base = {"world": 0, "scale": [1.0, 1.0, 1.0], "color": [0.8, 0.8, 0.8], "opacity": 1.0, "replica_of": None}
    shapes = [{**base, "id": 0, "label": "table", "body": -1, "type": "box", "size": [0.25, 0.25, 0.005],
               "xform": [0, 0.05, -0.005, 0, 0, 0, 1], "mesh": None},
              {**base, "id": 1, "label": "glass", "body": 0, "type": "mesh", "size": [], "mesh": "glass",
               "xform": [0, 0, 0, 0, 0, 0, 1]}]
    scene = {"arb": 1, "meta": {"clip": "synth_glass", "alloy_commit": "synthetic", "created": "synthetic",
                                "device": "cpu", "world_count": 1},
             "frames": {"count": frames, "fps": fps}, "worlds": [0], "world_offsets": [[0.0, 0.0, 0.0]],
             "hero_world": 0, "bodies": [{"id": 0, "label": "glass_body", "world": 0}], "shapes": shapes,
             "meshes": {"glass": {"file": "meshes/glass.npz", "verts": len(gv), "faces": len(gf), "uv": False}},
             "points": [{"name": "water", "world": 0, "count": int(len(water)), "radius": WATER_R}]}
    (root / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


def glass_look(samples: int = 128, contact: bool = True) -> dict:
    """Side-on 100 mm look through the wall (no depth of field, so walls front and side are both sharp)."""
    water = {"material": "water", "radius_scale": 1.6, "smooth": 2, "voxel_m": 0.0004}
    if contact:
        water["contact"] = [{"match": "^glass$"}]
    eye_z = WATER_TOP + 0.008
    return {
        "look": 1,
        "render": {"resolution": [1080, 1920], "samples": samples, "motion_blur": {"shutter_s": 1 / 60}},
        "frames": {"start": 0, "end": -1},
        "cameras": {
            "side": {"lens_mm": 100, "sensor_mm": 36, "fstop": 0,
                     "keys": [{"t": 0.0, "eye": [0.0, -0.30, eye_z], "target": [0.0, 0.0, WATER_TOP - 0.012]}]},
            "close": {"resolution": [1920, 1080], "lens_mm": 100, "sensor_mm": 36, "fstop": 0,
                      "keys": [{"t": 0.0, "eye": [0.012, -0.20, eye_z - 0.002],
                                "target": [0.012, 0.0, WATER_TOP - 0.004]}]}},
        "world": {"hdri": "polyhaven:photo_studio_loft_hall", "strength": 0.8, "rotation_deg": 40, "show_hdri": True},
        "lights": [
            {"type": "area", "eye": [-0.25, -0.30, 0.25], "target": [0, 0, 0.05], "size_m": 0.4, "watts": 6,
             "kelvin": 5600},
            {"type": "area", "eye": [0.15, 0.35, 0.12], "target": [0, 0, 0.05], "size_m": 0.3, "watts": 5,
             "kelvin": 6500}],
        "materials": {"glass": {"preset": "glass_clear", "ior": 1.5}, "table": {"preset": "wood_oak", "scale_m": 0.5}},
        "assign": [{"match": "^glass$", "material": "glass"}, {"match": "^table$", "material": "table"}],
        "liquids": {"water": water},
    }
