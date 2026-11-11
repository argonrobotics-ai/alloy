# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS-10: `collision: decompose` emits precomputed convex hulls from the `<stem>.hulls.obj` sidecar.

Mass rule (coordinator ruling): a watertight card mesh gives the body mass (= card_mass) and inertia, hull shapes carry
density 0; an open card mesh scales the hull density so the body mass still equals card_mass (inertia from hulls).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from alloy._vendor import newton

from alloy.cards.geometry import MeshData, read_obj
from alloy.cards.mass import mesh_mass
from alloy.cards.types import MeshSource, ObjectCard, ResolvedObject
from alloy.core.errors import AssemblyError
from alloy.route.types import Route, RouteDecision
from alloy.scene.emitters.rigid import emit_rigid
from alloy.testing import route as T

S = 0.05                     # m per unit: the L is two 2x1x1 / 1x1x1 unit boxes
RHO = T.WOOD["density"]
L_POLY = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]


def _l_mesh(drop_side: int | None = None) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Closed L prism (outward winding), z in [0, 1]; ``drop_side`` removes one side quad (open mesh)."""
    n = len(L_POLY)
    verts = np.array([(x, y, 0.0) for x, y in L_POLY] + [(x, y, 1.0) for x, y in L_POLY], float) * S
    r = 3                                                    # reflex vertex (1, 1): a fan from it covers the L
    fan = [(r, (r + k) % n, (r + k + 1) % n) for k in range(1, n - 1)]
    faces = [(a, c, b) for a, b, c in fan] + [(a + n, b + n, c + n) for a, b, c in fan]
    for i in range(n):
        if i == drop_side:
            continue
        j = (i + 1) % n
        faces += [(i, j, j + n), (i, j + n, i + n)]
    return verts, faces


def _box(lo, hi) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    lo, hi = np.asarray(lo, float) * S, np.asarray(hi, float) * S
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v, f


def _obj_text(parts, header: str = "") -> str:
    lines, base = ([header] if header else []), 0
    for k, (v, f) in enumerate(parts):
        lines.append(f"o hull_{k}")
        lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in v.tolist()]
        lines += [f"f {a + base + 1} {b + base + 1} {c + base + 1}" for a, b, c in f]
        base += len(v)
    return "\n".join(lines) + "\n"


def _card(tmp_path, mesh, hulls, *, stale: bool = False, max_parts: int = 4, sidecar: bool = True):
    path = tmp_path / "ell.obj"
    path.write_text(_obj_text([mesh]).replace("o hull_0\n", ""))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sidecar:
        head = f"# source_sha256: {'0' * 64 if stale else sha}"
        (tmp_path / "ell.hulls.obj").write_text(_obj_text(hulls, head))
    card = ObjectCard("ell", MeshSource(str(path), sha, collision="decompose", max_parts=max_parts), "wood")
    mat = T.material("wood", **T.WOOD)
    return ResolvedObject(card, mat, {}, "test", (2 * S, 2 * S, S), {}, ()), path


def _emit(obj):
    b = newton.ModelBuilder()
    d = RouteDecision(Route.RIGID, "mujoco", {}, 1.0, False, ())
    emit_rigid(b, obj, d)
    return b.finalize(device="cpu")


def _analytic_l(rho: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, COM and inertia about the COM of the L (two non-overlapping unit boxes, scaled by S)."""
    boxes = [((0, 0, 0), (2, 1, 1)), ((0, 1, 0), (1, 2, 1))]
    ms, cs, Is = [], [], []
    for lo, hi in boxes:
        d = (np.asarray(hi, float) - np.asarray(lo, float)) * S
        m = rho * d.prod()
        ms.append(m)
        cs.append((np.asarray(lo, float) + np.asarray(hi, float)) * 0.5 * S)
        Is.append(np.diag([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2]) * m / 12.0)
    M = sum(ms)
    com = sum(m * c for m, c in zip(ms, cs, strict=True)) / M
    inertia = np.zeros((3, 3))
    for m, c, i in zip(ms, cs, Is, strict=True):
        r = c - com
        inertia += i + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return M, com, inertia


def test_watertight_card_mesh_gives_mass_and_inertia_overlapping_hulls_add_nothing(tmp_path):
    mesh = _l_mesh()
    hulls = [_box((0, 0, 0), (2, 1, 1)), _box((0, 0.5, 0), (1, 2, 1))]          # overlap [0,1]x[0.5,1]
    obj, path = _card(tmp_path, mesh, hulls)
    model = _emit(obj)
    M, com, inertia = _analytic_l(RHO)
    card_mass = mesh_mass(read_obj(path), RHO)
    assert model.body_count == 1 and model.shape_count == 3          # two hulls and the display mesh (SCR OSS-9-5)
    assert model.body_mass.numpy()[0] == pytest.approx(card_mass, rel=1e-6)
    assert card_mass == pytest.approx(M, rel=1e-5)                              # float32 mesh vertices
    np.testing.assert_allclose(model.body_com.numpy()[0], com, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(model.body_inertia.numpy()[0], inertia, rtol=1e-4, atol=1e-10)


def test_open_card_mesh_scales_hull_density_to_the_card_mass(tmp_path):
    mesh = _l_mesh(drop_side=1)                                                 # x = 2 side face missing
    hulls = [_box((0, 0, 0), (2, 1, 1)), _box((0, 1, 0), (1, 2, 1))]            # exact partition of the L
    obj, path = _card(tmp_path, mesh, hulls)
    v, f = mesh
    card_mass = mesh_mass(MeshData(v, np.asarray(f)), RHO)
    model = _emit(obj)
    assert model.body_mass.numpy()[0] == pytest.approx(card_mass, rel=1e-6)
    M, com, inertia = _analytic_l(RHO)
    np.testing.assert_allclose(model.body_com.numpy()[0], com, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(model.body_inertia.numpy()[0], inertia * card_mass / M, rtol=1e-4, atol=1e-10)


def test_stale_sidecar_names_the_fix(tmp_path):
    obj, _ = _card(tmp_path, _l_mesh(), [_box((0, 0, 0), (2, 1, 1))], stale=True)
    with pytest.raises(AssemblyError, match=r"stale.*python -m alloy\.importers\.<library> hulls"):
        _emit(obj)


def test_missing_sidecar_names_the_fix(tmp_path):
    obj, _ = _card(tmp_path, _l_mesh(), [], sidecar=False)
    with pytest.raises(AssemblyError, match=r"ell\.hulls\.obj.*CoACD never runs inside assembly"):
        _emit(obj)


def test_more_hulls_than_max_parts_raises(tmp_path):
    boxes = [_box((0, 0, 0), (1, 1, 1)), _box((1, 0, 0), (2, 1, 1)), _box((0, 1, 0), (1, 2, 1))]
    obj, _ = _card(tmp_path, _l_mesh(), boxes, max_parts=2)
    with pytest.raises(AssemblyError, match="3 hulls, need 1..max_parts=2"):
        _emit(obj)


def test_hull_with_more_than_64_vertices_raises(tmp_path):
    ang = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    ring = np.c_[np.cos(ang), np.sin(ang)]
    v = np.r_[np.c_[ring, np.zeros(40)], np.c_[ring, np.ones(40)]]              # 80-vertex prism, side faces only
    f = [(i, (i + 1) % 40, 40 + (i + 1) % 40) for i in range(40)]
    obj, _ = _card(tmp_path, _l_mesh(), [(v, f)])
    with pytest.raises(AssemblyError, match="80 vertices > 64"):
        _emit(obj)


@pytest.mark.parametrize("edit", ["empty", "stray_g", "loose_faces"])
def test_malformed_hull_groups_raise(tmp_path, edit):
    obj, _ = _card(tmp_path, _l_mesh(), [_box((0, 0, 0), (2, 1, 1))])
    side = tmp_path / "ell.hulls.obj"
    head, body = side.read_text().split("\n", 1)
    if edit == "empty":
        body += "o hull_1\n"
    elif edit == "stray_g":
        lines = body.splitlines()
        body = "\n".join(lines[:12] + ["g lid"] + lines[12:]) + "\n"
    else:                                                                      # first face before the hull block
        lines = body.splitlines()                                              # o hull_0, 8 v, 12 f
        body = "\n".join(lines[1:10] + ["o hull_0"] + lines[10:]) + "\n"
    side.write_text(head + "\n" + body)
    with pytest.raises(AssemblyError, match="non-empty 'o hull_<k>' blocks owning every face"):
        _emit(obj)


def test_decompose_with_part_materials_raises(tmp_path):
    obj, _ = _card(tmp_path, _l_mesh(), [_box((0, 0, 0), (2, 1, 1))])
    obj = ResolvedObject(obj.card, obj.material, {"lid": obj.material}, "test", obj.extents, {}, ())
    with pytest.raises(AssemblyError, match="part_materials is not supported yet"):
        _emit(obj)
