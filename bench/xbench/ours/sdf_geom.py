# SPDX-License-Identifier: Apache-2.0
"""Geometry of the P3-2 mesh-SDF law packs (A8.push, A9.thread_rundown): OBJ meshes and the pack's static checks.

A8.push: Alloy collides triangle meshes (``alloy.testing.contact``: the 96-segment CON-46 peg and blind hole; the
texture SDFs are baked from them on CUDA). The OBJs are the model's own ``shape_source`` triangles in the body frame
(the static hole in the world frame): Alloy's float32 vertices written exactly (17 significant digits).

A9.thread_rundown: Alloy's thread contact is analytic (``alloy.contact.thread_sdf``: the ISO 68-1 basic profile with
the ISO 965-1 fundamental deviations, 6g bolt, 6H nut); its MuJoCo cylinders are filtered from collision. The OBJs are
Alloy's own mesher of the same solids (``thread_sdf.thread_mesh``): vertices on the analytic surface, rows along the
helix, profile corners on mesh edges, chord sag <= r (1 - cos(pi / n)) between vertices along the helix. The bolt is
cut to ``BOLT_PACK_Z``: Alloy's analytic bolt spans ``thread_twins.BOLT_Z`` (+-25 mm), and the nut ([-4, 4] mm,
advancing ~1.7 mm) stays more than 2.5 pitches inside the cut for the whole run, so the contact geometry is the
same and an engine's SDF resolution is spent on the part the nut reaches. The nut is the full DIN 934 thread on
[-4, 4] mm, solid out to its 8.5 mm outer radius.

Checks (``check_*``; ``sdf_pack.pack`` refuses a pack that fails them):
- A8: OBJ vertices and triangles against Alloy's shapes in the body frame. The per-world initial gap of the peg's
  vertices to the hole mesh is computed twice, from the pack (OBJ + cond.json poses) and from Alloy (model meshes +
  state), and both must agree. The 1e-6 m lateral perturbations live in those poses.
- A9: OBJ vertices on Alloy's analytic surfaces; facet sag within the chord bound; at the initial pose, the 6g/6H
  flank gap from the nut's vertices to the bolt's analytic surface (clearance / 4 = 7.9 um normal to the flank) and
  the OBJ-to-OBJ gaps both ways. A float32 or meshing slip would lose details at this scale.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import warp as wp

log = logging.getLogger("xbench.ours")
THREAD_SEGMENTS = 96                 # thread_mesh default: chord sag 2.7 um at the bolt crest, 4.5 um on the nut's rim
BOLT_PACK_Z = (-0.010, 0.010)        # m, bolt thread cut for the pack (module docstring)
OBJ_TOL_M = 1.0e-6                   # pack refusal: OBJ vertex vs Alloy, and pack vs Alloy initial gaps
SURFACE_TOL_M = 1.0e-9               # thread OBJ vertices on the analytic surface (9 significant digits)


# ---------------------------------------------------------------- OBJ io --------------------------------------------

def write_obj(path: Path, verts: np.ndarray, tris: np.ndarray, header: list[str], digits: int = 9) -> None:
    """Triangles with outward winding; vertices in metres with ``digits`` significant digits (9: float32 values
    round-trip through a float32 parser; 17: exact for float64 parsers too)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {h}" for h in header]
    lines += [f"v {x:.{digits}g} {y:.{digits}g} {z:.{digits}g}" for x, y, z in np.asarray(verts, np.float64)]
    lines += [f"f {a} {b} {c}" for a, b, c in np.asarray(tris, np.int64) + 1]
    path.write_text("\n".join(lines) + "\n")


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(float64 vertices [N, 3], int64 triangles [M, 3], 0-based) of a triangle OBJ."""
    v, f = [], []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("v "):
            v.append(ln.split()[1:4])
        elif ln.startswith("f "):
            f.append([int(x.split("/")[0]) - 1 for x in ln.split()[1:4]])
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


# ---------------------------------------------------------------- poses and mesh queries ----------------------------

def transform_points(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Points ``v`` [N, 3] (body frame) placed at the pose ``q`` = x y z qx qy qz qw (float64)."""
    q = np.asarray(q, np.float64)
    u, w = q[3:6], q[6]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t) + q[:3]


@wp.kernel
def _signed_distance(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid()
    r = wp.mesh_query_point_sign_winding_number(mesh, pts[i], 1.0e3)
    out[i] = r.sign * wp.length(pts[i] - wp.mesh_eval_position(mesh, r.face, r.u, r.v))


def signed_distance(verts: np.ndarray, tris: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance [m] of ``pts`` to the closed mesh (negative inside; winding-number sign; float32, CPU)."""
    with wp.ScopedDevice("cpu"):
        mesh = wp.Mesh(wp.array(np.asarray(verts, np.float32), dtype=wp.vec3),
                       wp.array(np.asarray(tris, np.int32).reshape(-1), dtype=wp.int32), support_winding_number=True)
        p = wp.array(np.asarray(pts, np.float32), dtype=wp.vec3)
        out = wp.zeros(len(pts), dtype=float)
        wp.launch(_signed_distance, dim=len(pts), inputs=[mesh.id, p, out])
        return out.numpy().astype(np.float64)


# ---------------------------------------------------------------- A8: Alloy's meshes -------------------------------

def shape_meshes(model, world: int = 0) -> list[dict]:
    """Mesh shapes of ``world``: name, Newton body (-1 static), vertices in the body frame (float32), triangles."""
    starts = model.shape_world_start.numpy()
    xf, body, types = model.shape_transform.numpy(), model.shape_body.numpy(), model.shape_type.numpy()
    out = []
    for s in range(int(starts[world]), int(starts[world + 1])):
        src = model.shape_source[s]
        if src is None or not hasattr(src, "indices"):
            continue
        v = np.asarray(src.vertices, np.float32) * model.shape_scale.numpy()[s]
        v = transform_points(xf[s].astype(np.float64), v.astype(np.float64)).astype(np.float32)
        out.append({"shape": s, "name": str(model.shape_label[s]).rsplit("/", 1)[-1], "body": int(body[s]),
                    "type": int(types[s]), "verts": v, "tris": np.asarray(src.indices, np.int64).reshape(-1, 3)})
    return out


def check_meshes(cond_dir: Path, shapes: list[dict], meshes: list[dict]) -> dict:
    """Worst body-frame vertex distance (m) per OBJ against Alloy's shape, and whether the triangles match."""
    out = {}
    for sh, m in zip(shapes, meshes, strict=True):
        v, f = read_obj(cond_dir / sh["obj"])
        same = v.shape == m["verts"].shape and np.array_equal(f, m["tris"])
        out[sh["name"]] = {"vertex_err_m": float(np.abs(v - m["verts"]).max()) if same else math.inf,
                           "triangles_equal": bool(same), "vertices": int(len(v)), "triangles": int(len(f))}
    return out


def posed_gaps(sampled: tuple[np.ndarray, np.ndarray], partner: tuple[np.ndarray, np.ndarray],
               poses: list[np.ndarray]) -> list[float]:
    """Per pose: min signed distance (m) of the sampled mesh's vertices, posed, to the static partner mesh."""
    return [float(signed_distance(*partner, transform_points(q, sampled[0])).min()) for q in poses]


# ---------------------------------------------------------------- A9: threads ---------------------------------------

def thread_features():
    """(bolt, nut) ``ThreadFeature`` of Alloy's A9 pair (``thread_twins.thread_pairs``)."""
    from alloy.contact.thread_twins import thread_pairs

    pair = thread_pairs(1.0)[0]
    return pair.partner, pair.sampled


def thread_params(feat):
    """Alloy's analytic SDF parameters of ``feat`` (as ``analytic_pair`` builds them)."""
    from alloy.contact.thread_sdf import parse_thread_id, thread_sdf_params

    return thread_sdf_params(parse_thread_id(feat.analytic_id), z0=feat.z0, z1=feat.z1, outer_radius=feat.outer_radius)


def thread_meshes() -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """{"bolt_thread", "nut_thread"}: (float64 vertices, triangles, OBJ header) from ``thread_sdf.thread_mesh``."""
    from alloy.contact.thread_sdf import parse_thread_id, thread_mesh

    bolt, nut = thread_features()
    z0, z1 = BOLT_PACK_Z
    vb, tb = thread_mesh(parse_thread_id(bolt.analytic_id), z1 - z0, THREAD_SEGMENTS, z0=z0)
    vn, tn = thread_mesh(parse_thread_id(nut.analytic_id), nut.z1 - nut.z0, THREAD_SEGMENTS, z0=nut.z0,
                         outer_radius=nut.outer_radius)
    note = f"alloy.contact.thread_sdf.thread_mesh, {THREAD_SEGMENTS} segments per turn"
    return {"bolt_thread": (vb, tb, f"{bolt.analytic_id} on z [{z0:g}, {z1:g}] m (world frame); {note}"),
            "nut_thread": (vn, tn, f"{nut.analytic_id} on z [{nut.z0:g}, {nut.z1:g}] m, outer radius "
                                   f"{nut.outer_radius:g} m (nut body frame); {note}")}


def chord_bound(feat) -> float:
    """Largest chord sag (m) of ``thread_mesh`` on ``feat``: the outermost radius at THREAD_SEGMENTS per turn."""
    from alloy.contact.thread_sdf import parse_thread_id

    r = feat.outer_radius or parse_thread_id(feat.analytic_id).r_major
    return r * (1.0 - math.cos(math.pi / THREAD_SEGMENTS))


def _surface_errors(v: np.ndarray, f: np.ndarray, feat, cap_r: float) -> dict:
    """|Alloy SDF| at the OBJ vertices and at the facet centroids (both away from the pack's bolt caps)."""
    from alloy.contact.thread_sdf import thread_sdf_np

    prm = thread_params(feat)
    on = np.hypot(v[:, 0], v[:, 1]) > cap_r
    cen = v[f].mean(axis=1)
    keep = np.all(on[f], axis=1)
    return {"surface_err_m": float(np.abs(thread_sdf_np(prm, v[on])).max()),
            "facet_sag_m": float(np.abs(thread_sdf_np(prm, cen[keep])).max()), "chord_bound_m": chord_bound(feat)}


def check_threads(cond_dir: Path, shapes: list[dict], nut_pose: np.ndarray) -> dict:
    """A9 checks (module docstring) on the written OBJs, the nut posed at the pack's initial pose."""
    from alloy.contact.thread_sdf import parse_thread_id, thread_sdf_np

    bolt, nut = thread_features()
    objs = {s["name"]: read_obj(cond_dir / s["obj"]) for s in shapes}
    vb, fb = objs["bolt_thread"]
    vn, fn = objs["nut_thread"]
    r_cap = 0.5 * parse_thread_id(bolt.analytic_id).r_minor     # the bolt's cut caps: only the two centre vertices
    out = {"bolt_thread": _surface_errors(vb, fb, bolt, r_cap), "nut_thread": _surface_errors(vn, fn, nut, 0.0)}
    clearance = parse_thread_id(nut.analytic_id).deviation - parse_thread_id(bolt.analytic_id).deviation
    vn_w = transform_points(nut_pose, vn)
    out["fit"] = {"clearance_diametral_m": clearance, "flank_gap_expected_m": 0.25 * clearance,
                  "flank_gap_vertices_m": float(thread_sdf_np(thread_params(bolt), vn_w).min()),
                  "nut_vertices_to_bolt_obj_m": float(signed_distance(vb, fb, vn_w).min()),
                  "bolt_vertices_to_nut_obj_m": float(signed_distance(vn_w, fn, vb).min()),
                  "note": "6g/6H: the bolt's diameters sit `clearance` below the nut's; normal to a 60 deg flank that "
                          "is clearance / 4. OBJ-to-OBJ gaps include the facet sag of the other mesh"}
    return out


def thread_failures(chk: dict) -> list[str]:
    """Reasons a thread check fails (empty when it passes)."""
    bad = [f"{k}: OBJ vertices {v['surface_err_m']:.3g} m off Alloy's surface" for k, v in chk.items()
           if k != "fit" and v["surface_err_m"] > SURFACE_TOL_M]
    bad += [f"{k}: facet sag {v['facet_sag_m']:.3g} m > chord bound" for k, v in chk.items()
            if k != "fit" and v["facet_sag_m"] > v["chord_bound_m"] + SURFACE_TOL_M]
    fit = chk["fit"]
    if abs(fit["flank_gap_vertices_m"] - fit["flank_gap_expected_m"]) > 0.1 * fit["flank_gap_expected_m"]:
        bad.append(f"6g/6H flank gap {fit['flank_gap_vertices_m']:.4g} m, expected {fit['flank_gap_expected_m']:.4g}")
    if min(fit["nut_vertices_to_bolt_obj_m"], fit["bolt_vertices_to_nut_obj_m"]) <= 0.0:
        bad.append("the thread OBJs overlap at the initial pose")
    return bad
