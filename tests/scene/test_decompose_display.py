# SPDX-License-Identifier: Apache-2.0
"""SCR OSS-9-5 item 4: a ``collision: decompose`` card draws its card mesh, not its convex hulls (CPU).

The emitter adds the card mesh as a display-only shape (density 0, no shape or particle collision, VISIBLE) and clears
VISIBLE on the hulls, as SOFT-3's robot visual layer does. Physics must not change: a rigid drop (MuJoCo, hulls on
hulls) and a cloth draped over a static decompose object (VBD particle contacts) step bit for bit as with the display
shape left out.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import alloy
import alloy.scene.emitters.rigid as rigid
from alloy._vendor import newton
from alloy.camera.from_newton import scene_from_model

S = 0.05                                                   # [m] per unit: an L of a 2x1x1 and a 1x1x1 box
L_POLY = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
HULLS = [((0, 0, 0), (2, 1, 1)), ((0, 1, 0), (1, 2, 1))]   # an exact partition of the L
VISIBLE, COLLIDE = int(newton.ShapeFlags.VISIBLE), int(newton.ShapeFlags.COLLIDE_SHAPES)
PARTICLES = int(newton.ShapeFlags.COLLIDE_PARTICLES)


def _l_mesh() -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    n = len(L_POLY)
    v = np.array([(x, y, 0.0) for x, y in L_POLY] + [(x, y, 1.0) for x, y in L_POLY], float) * S
    fan = [(3, (3 + k) % n, (3 + k + 1) % n) for k in range(1, n - 1)]      # from the reflex vertex (1, 1)
    f = [(a, c, b) for a, b, c in fan] + [(a + n, b + n, c + n) for a, b, c in fan]
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, j + n), (i, j + n, i + n)]
    return v, f


def _box(lo, hi) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    lo, hi = np.asarray(lo, float) * S, np.asarray(hi, float) * S
    v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    f = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    return v, f


def _obj(parts, header: str = "") -> str:
    lines, base = ([header] if header else []), 0
    for k, (v, f) in enumerate(parts):
        lines += [f"o hull_{k}"] if header else []
        lines += [f"v {x!r} {y!r} {z!r}" for x, y, z in v.tolist()]
        lines += [f"f {a + base + 1} {b + base + 1} {c + base + 1}" for a, b, c in f]
        base += len(v)
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    """A card root with one decompose object, ``ell_test`` (ABS, watertight L, two box hulls)."""
    root = tmp_path_factory.mktemp("decompose_display")
    (root / "meshes").mkdir()
    (root / "objects").mkdir()
    mesh = root / "meshes" / "ell_test.obj"
    mesh.write_text(_obj([_l_mesh()]))
    sha = hashlib.sha256(mesh.read_bytes()).hexdigest()
    (root / "meshes" / "ell_test.hulls.obj").write_text(_obj([_box(*h) for h in HULLS], f"# source_sha256: {sha}"))
    (root / "objects" / "ell_test.yaml").write_text(
        "schema: alloy.object/1\nname: ell_test\nshape:\n  mesh:\n    path: meshes/ell_test.obj\n"
        f"    sha256: {sha}\n    collision: decompose\n    max_parts: 2\nmaterial: abs\n")
    return root


def _drop(root: Path) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, roots=[str(root)])
    scene.add_ground()
    scene.add_object("base", "ell_test", pos=(0.0, 0.0, 0.0), static=True)
    scene.add_object("ell", "ell_test", pos=(0.01, 0.01, 0.08), rpy=(0.3, 0.2, 0.1))
    return scene


def _drape(root: Path) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, roots=[str(root)])
    scene.add_ground()
    scene.add_object("base", "ell_test", pos=(0.0, 0.0, 0.0), static=True)
    scene.add_cloth("sheet", size=(0.12, 0.12), pos=(0.05, 0.05, 0.07), resolution=6, mass=0.01)
    return scene


def _run(scene: alloy.Scene, periods: int) -> dict[str, np.ndarray]:
    sim = alloy.build(scene, device="cpu")
    sim.step(periods)
    assert sim.check()["ok"], sim.check()["problems"]
    return {k: np.asarray(v) for k, v in sim.state(world=0).items()}


def _no_display(monkeypatch) -> None:
    monkeypatch.setattr(rigid, "_add_display_mesh", lambda b, body, mesh, cfg, name, hulls: -1)


def test_the_card_mesh_is_drawn_and_the_hulls_are_not(root):
    sim = alloy.build(_drop(root), device="cpu")
    m = sim.engine.model
    labels, flags = [str(x) for x in m.shape_label], m.shape_flags.numpy()
    for name in ("base", "ell"):
        hulls = [s for s, lbl in enumerate(labels) if lbl.startswith(f"{name}/hull_")]
        shown = [s for s, lbl in enumerate(labels) if lbl == f"{name}/visual"]
        assert len(hulls) == len(HULLS) and len(shown) == 1, labels
        assert all(flags[s] & COLLIDE and not flags[s] & VISIBLE for s in hulls)
        (d,) = shown
        assert flags[d] & VISIBLE and not flags[d] & (COLLIDE | PARTICLES)
        assert int(m.shape_type.numpy()[d]) == int(newton.GeoType.MESH)
        assert len(m.shape_source[d].vertices) == len(_l_mesh()[0])
    drawn = scene_from_model(m).inst_name
    assert not [n for n in drawn if "/hull_" in n] and {"base/visual", "ell/visual"} <= set(drawn)


def test_mass_and_surface_are_the_cards(root, monkeypatch):
    on = alloy.build(_drop(root), device="cpu").engine.model
    _no_display(monkeypatch)
    off = alloy.build(_drop(root), device="cpu").engine.model
    for k in ("body_mass", "body_com", "body_inertia", "body_inv_mass"):
        np.testing.assert_array_equal(getattr(on, k).numpy(), getattr(off, k).numpy(), err_msg=k)
    labels = [str(x) for x in on.shape_label]
    d = labels.index("ell/visual")
    h = labels.index("ell/hull_0")
    for k in ("shape_material_mu", "shape_material_restitution", "shape_material_mu_torsional"):
        assert getattr(on, k).numpy()[d] == getattr(on, k).numpy()[h], k          # the readback checks every shape


def _on_the_base(s: dict[str, np.ndarray]) -> bool:
    """The contacts happened: the dropped L rests on the base's top (z 0.05), or the cloth over it does: something
    rests on the top and nothing sank through it (a sheet soft as its fabric rucks a little there, WP11's R2-E1)."""
    if "ell.pos" in s:
        return bool(s["ell.pos"][2] > 0.045)
    p = s["sheet.points"]
    over = (p[:, 0] > 0.005) & (p[:, 0] < 0.095) & (p[:, 1] > 0.005) & (p[:, 1] < 0.045)
    return bool(over.any() and np.all(p[over, 2] > 0.049) and np.any(p[over, 2] < 0.053))


@pytest.mark.parametrize("scene, periods", [(_drop, 40), (_drape, 20)], ids=["rigid_drop", "cloth_drape"])
def test_physics_is_bitwise_unchanged(root, monkeypatch, scene, periods):
    """A colliding display mesh changes both runs (checked on 2026-11-06), so this compares contacts, not free fall."""
    with_display = _run(scene(root), periods)
    assert _on_the_base(with_display)
    _no_display(monkeypatch)
    without = _run(scene(root), periods)
    assert with_display.keys() == without.keys()
    for k, v in with_display.items():
        np.testing.assert_array_equal(v, without[k], err_msg=k)
