# SPDX-License-Identifier: Apache-2.0
"""Law: a scanned object that simulates as a SOFT body shows the same photo texture as its rigid twin (SCAN-TEXTURES).

Fixture: ``scan_fixture.ball_card``, a 4 cm sphere whose visual.glb is a latitude-longitude sphere with an 8 x 4
checker; the same card with ``silicone_40a`` routes to ``soft_vbd`` (a particle surface from tetrahedra, UVs from the
nearest glb vertex: ``alloy.camera.scan_soft``) and with ``abs`` stays rigid (the glb mesh itself). Both are rendered
at the build pose, head-on, under a uniform white sky; the albedo images are compared over the ball (an analytic
mask, 90 % of its radius, so the silhouettes' different facets do not count), in sRGB 0-255:

- front view (the checker, away from the texture's seam): mean |soft - rigid| <= ``MEAN_MAX``, and at most
  ``EDGE_MAX`` of the pixels differ by more than ``BAD`` (checker edges a facet off);
- seam view (looking at the u = 0 / 1 meridian, where the soft surface's triangles straddle the seam and interpolate
  across the whole image: a streak): at most ``STREAK_MAX`` of the pixels differ by more than ``BAD``;
- the soft surface's triangles that straddle a seam (``scan_soft.seam_fraction``) are at most ``SEAM_TRIS_MAX``;
- planted mutant: UVs zeroed (every vertex reads one texel) misses the front view by at least ``MUTANT_MIN``.

Measured on the CPU (fast / full tracer): front mean 3.2 / 3.8, front share over BAD 1.8 % / 1.6 %, seam-view
streak 12.2 % / 12.0 %.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera import scan_soft
from tests.camera.scan_fixture import BALL, BALL_RADIUS, ball_card

# the soft ball is a mesh card filled with tetrahedra at build: that needs the `mesh` extra
pytest.importorskip("pytetwild", reason="the soft scan law needs the mesh extra (pip install 'alloy-engine[mesh]')")
SIZE, FOV, SPP = 96, 30.0, 64
BAD = 40.0           # a pixel "differs" past this many sRGB levels on its worst channel
MEAN_MAX = 6.0       # front view: mean |soft - rigid|
EDGE_MAX = 0.04      # front view: share of pixels past BAD
STREAK_MAX = 0.18    # seam view: share of pixels past BAD (the seam streak)
SEAM_TRIS_MAX = 0.05
MUTANT_MIN = 30.0
VIEWS = {"front": (0.0, -0.25, 0.04), "seam": (0.25, 0.0, 0.04)}
RIGID = "scan_ball_rigid"

pytestmark = [pytest.mark.timeout(900), pytest.mark.law]


def _sim(root, card: str):
    scene = alloy.Scene(dt=0.01, roots=[root])
    scene.add_ground()  # the camera needs one shape besides the soft body
    scene.add_object("ball", card, pos=(0.0, 0.0, 0.0))
    return alloy.build(scene, device="cpu")


def _albedo(sim, tracer: str, eye) -> np.ndarray:
    """The albedo image (linear, (H, W, 3)) of world 0 from ``eye`` towards the ball's centre."""
    from alloy.camera import Camera, camera_scene
    from alloy.camera.scene import Lighting
    white = Lighting([], (1.0, 1.0, 1.0), 1.0)
    cam = Camera("law", look_at=(eye, (0.0, 0.0, BALL_RADIUS)), width=SIZE, height=SIZE, fov=FOV, spp=SPP)
    if tracer == "fast":
        camera_scene(sim, lighting=white, device="cpu", quality="fast")
        out = cam.render(sim, output="all", seed=0)
    else:
        sc = camera_scene(sim, lighting=white, device="cpu", quality="full")
        out = cam.render(sc, state=sim.engine.state, output="all", seed=0)
    return out.albedo.numpy()[0].astype(np.float64)


def ball_mask(eye) -> np.ndarray:
    """Pixels whose ray meets the ball shrunk to 90 % of its radius (a pinhole camera of ``FOV``, square image)."""
    eye, c = np.asarray(eye, np.float64), np.array([0.0, 0.0, BALL_RADIUS])
    f = (c - eye) / np.linalg.norm(c - eye)
    r = np.cross(f, (0.0, 0.0, 1.0))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    t = np.tan(np.radians(FOV / 2))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    d = f + ((2 * (xx + 0.5) / SIZE - 1) * t)[..., None] * r - ((2 * (yy + 0.5) / SIZE - 1) * t)[..., None] * u
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    oc = eye - c
    b = (d * oc).sum(-1)
    return b ** 2 - (oc @ oc - (0.9 * BALL_RADIUS) ** 2) > 0


def _srgb(lin: np.ndarray) -> np.ndarray:
    c = np.clip(lin, 0.0, 1.0)
    return 255.0 * np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def compare(soft: np.ndarray, rigid: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """(mean worst-channel |soft - rigid|, share of pixels past BAD) over ``mask``, sRGB 0-255."""
    d = np.abs(_srgb(soft) - _srgb(rigid)).max(-1)[mask]
    return float(d.mean()), float((d > BAD).mean())


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    root = tmp_path_factory.mktemp("soft_scan")
    ball_card(root)
    ball_card(root, name=RIGID, material="abs")
    return root


@pytest.fixture(scope="module")
def images(roots):
    """{(kind, tracer, view): albedo} for the soft and the rigid ball."""
    out = {}
    for kind, card in (("soft", BALL), ("rigid", RIGID)):
        sim = _sim(roots, card)
        route = sim.routes()["ball"]["route"]
        assert route == ("soft_vbd" if kind == "soft" else "rigid"), route
        for tracer in ("fast", "full"):
            for view, eye in VIEWS.items():
                out[(kind, tracer, view)] = _albedo(sim, tracer, eye)
    return out


@pytest.mark.parametrize("tracer", ["fast", "full"])
def test_soft_scan_matches_its_rigid_twin_away_from_the_seam(images, tracer):
    mean, share = compare(images[("soft", tracer, "front")], images[("rigid", tracer, "front")],
                          ball_mask(VIEWS["front"]))
    assert mean <= MEAN_MAX and share <= EDGE_MAX, (mean, share)


@pytest.mark.parametrize("tracer", ["fast", "full"])
def test_the_seam_streak_is_bounded(images, tracer):
    _, share = compare(images[("soft", tracer, "seam")], images[("rigid", tracer, "seam")], ball_mask(VIEWS["seam"]))
    assert share <= STREAK_MAX, share


def test_few_soft_triangles_straddle_a_seam(roots):
    from alloy.camera.deform import deform_objects
    (obj,) = deform_objects(_sim(roots, BALL)).objects
    assert obj.uv is not None and obj.image is not None
    assert scan_soft.seam_fraction(obj.uv, obj.tris) <= SEAM_TRIS_MAX


@pytest.mark.parametrize("tracer", ["fast", "full"])
def test_zeroed_uvs_fail_the_law(roots, images, monkeypatch, tracer):
    real = scan_soft.surface_uvs
    monkeypatch.setattr(scan_soft, "surface_uvs", lambda *a: np.zeros_like(real(*a)))
    soft = _albedo(_sim(roots, BALL), tracer, VIEWS["front"])
    mean, _ = compare(soft, images[("rigid", tracer, "front")], ball_mask(VIEWS["front"]))
    assert mean >= MUTANT_MIN, mean
