# SPDX-License-Identifier: Apache-2.0
"""The scan-texture law: a scanned library object is drawn with its own photo texture, by both tracers.

The fixture is a real scan (the front panel of the YCB 004 sugar box, ``data/scan_fixture``, ATTRIBUTION.md there),
placed with ``scene.add_object`` as users do and rendered head-on under a uniform white sky (no sun, no ground: a
surface facing the camera receives the same light everywhere). The reference is the same glb drawn by an independent
numpy rasterizer through the camera's pinhole (``scan_reference``). Per channel, a gain and an offset fitted over the
panel's pixels stand for the light (the full tracer's Principled BSDF adds a little specular). The law then bounds
the gain and offset (so a texture drawn 0.4x dark fails), the mean absolute error and the worst 4 x 4 region, in
sRGB 0-255 units. Mutants planted in copies of the fixture must fail it: the texture flattened to its mean colour,
and the UVs flipped in v. The albedo image (the denoiser's guide) must carry the texture too.

Measured on the CPU at 64 spp (camera-scan-textures c92d36de6), as gain / offset / mean / worst region:
- the fixture: fast tracer 0.94-0.99 / 0.00-0.03 / 3.0 / 4.1; full tracer 0.92-0.97 / 0.02-0.05 / 3.2 / 4.7;
- the fetched scans: sugar box fast 3.1 / 3.8, full 3.2 / 4.7; mug fast 2.6 / 2.6, full 3.2 / 3.0;
- the mutants (mean): flat texture fast 71, full 122; v flipped fast 100, full 100.

``ALLOY_SCAN_LAW_DEVICE=cuda:0`` runs it on a GPU; ``ALLOY_CAMERA_BACKEND=optix`` with the OptiX tracer.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from tests.camera import scan_reference as sr

log = logging.getLogger(__name__)

FIXTURE = Path(__file__).resolve().parent / "data" / "scan_fixture"
CARD = "ycb_004_sugar_box_front"
DEVICE = os.environ.get("ALLOY_SCAN_LAW_DEVICE", "cpu")
WIDTH, HEIGHT, SPP = 96, 128, 64
FILL = 0.8              # the object's height (or width) fills this share of the image
GAIN = (0.85, 1.05)     # fitted light gain per channel under the white sky
OFFSET = (-0.02, 0.10)  # fitted offset (linear): the full tracer's specular layer adds about 0.05
MEAN_MAX = 6.0          # mean |rendered - reference| over the panel, sRGB 0-255
REGION_MAX = 10.0       # worst 4 x 4 region's mean colour difference, sRGB 0-255
MUTANT_MIN = 30.0       # a mutant must miss by at least this mean error
TRACERS = ("fast", "full")
pytestmark = pytest.mark.compile_heavy


@dataclass
class Score:
    gain: np.ndarray
    offset: np.ndarray
    mean: float
    region: float

    def ok(self) -> bool:
        return (bool(np.all((self.gain >= GAIN[0]) & (self.gain <= GAIN[1])))
                and bool(np.all((self.offset >= OFFSET[0]) & (self.offset <= OFFSET[1])))
                and self.mean <= MEAN_MAX and self.region <= REGION_MAX)

    def __str__(self) -> str:
        return (f"gain {np.round(self.gain, 3).tolist()} offset {np.round(self.offset, 3).tolist()} "
                f"mean {self.mean:.2f} region {self.region:.2f}")


def score(img: np.ndarray, ref: sr.Reference, mask: np.ndarray) -> Score:
    """Fit the light (gain, offset per channel), then compare in sRGB 0-255 over the mask."""
    g, o = sr.affine_fit(img, ref.rgb, mask)
    a8 = sr.linear_to_srgb8((img - o) / np.where(np.abs(g) > 1e-6, g, 1e-6))
    r8 = sr.linear_to_srgb8(ref.rgb)
    return Score(g, o, float(np.abs(a8 - r8)[mask].mean()), float(sr.region_errors(a8, r8, mask).max()))


def fixture_root(tmp: Path, mesh: sr.ScanMesh | None = None, look: tuple | None = None,
                 material_look: tuple | None = None) -> Path:
    """A card root holding the fixture; ``mesh`` replaces its glb (a mutant), ``look`` gives the card an object look
    of that linear base colour, ``material_look`` its material a default look."""
    root = tmp / "cards"
    shutil.copytree(FIXTURE, root, ignore=shutil.ignore_patterns("*.py", "*.md"))
    if mesh is not None:
        sr.write_scan(root / "meshes" / CARD / "visual.glb", mesh)
    if look is not None:
        (root / "looks" / "objects").mkdir(parents=True)
        (root / "looks" / "objects" / f"{CARD}.json").write_text(json.dumps(
            {"schema": "alloy.look/1", "object": CARD, "assign": [], "default": "body",
             "materials": {"body": {"preset": "plastic_matte", "base_color": list(look), "roughness": 0.8}}}))
    if material_look is not None:
        (root / "looks" / "materials").mkdir(parents=True, exist_ok=True)
        (root / "looks" / "materials" / f"{CARD}_eff.json").write_text(json.dumps(
            {"schema": "alloy.look/1", "material": f"{CARD}_eff", "preset": "plastic_matte",
             "base_color": list(material_look), "roughness": 0.8}))
    return root


def build(root: Path | None, card: str = CARD):
    """A one-world sim of ``card`` at the origin, named by its card (so the camera's card looks find it), no ground."""
    import alloy
    scene = alloy.Scene(dt=0.01, **({"roots": [str(root)]} if root is not None else {}))
    scene.add_object(card, card, pos=(0.0, 0.0, 0.0))
    return alloy.build(scene, device=DEVICE)


def as_fetched(monkeypatch, root: Path) -> Path:
    """Make every default card library read ``root`` as it reads a fetched scan library (the camera's card looks
    use the default library, not the scene's roots)."""
    import alloy.cards.library as library
    orig = library.fetched_roots
    monkeypatch.setattr(library, "fetched_roots", lambda found: [*orig(found), Path(root)])
    return root


def head_on(sim, mesh: sr.ScanMesh, ss: int = 4, side=(1.0, 0.0, 0.0)) -> tuple[sr.View, sr.Reference]:
    """A view from the body's ``side`` (+x: the box's printed front) that the object fills to FILL, and the
    reference image there."""
    st, name = sim.state(world=0), sim.names()["bodies"][0]
    pos, quat = st.get(f"{name}.body_pos", st[f"{name}.pos"]), st[f"{name}.quat_xyzw"]
    rot = sr.quat_xyzw_matrix(quat)
    pts = mesh.points @ rot.T + np.asarray(pos, np.float64)
    c = 0.5 * (pts.min(0) + pts.max(0))
    out, up = rot @ np.asarray(side, np.float64), rot[:, 2]
    right = np.cross(-out, up)
    extent = max(float(np.ptp((pts - c) @ up)), float(np.ptp((pts - c) @ right)) * HEIGHT / WIDTH)
    dist = float(((pts - c) @ out).max()) + 0.5 * extent / FILL / np.tan(0.5 * np.radians(60.0))
    view = sr.View(tuple(c + out * dist), tuple(c), WIDTH, HEIGHT, up=tuple(up))
    return view, sr.render_reference(mesh, pos, quat, view, ss=ss)


def render(sim, view: sr.View, tracer: str, look: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(linear RGB, albedo) of world 0 from ``view`` under a uniform white sky, by the fast or the full tracer
    (``look``: a scene look file's dict, full tracer only)."""
    from alloy.camera import Camera, camera_scene
    from alloy.camera.scene import Lighting
    white = Lighting([], (1.0, 1.0, 1.0), 1.0)
    cam = Camera("law", look_at=(view.eye, view.target, view.up), width=view.width, height=view.height,
                 fov=view.fov, spp=SPP)
    if tracer == "fast":
        camera_scene(sim, lighting=white, device=DEVICE, quality="fast")  # cached: the camera binds this scene
        out = cam.render(sim, output="all", seed=0)
    else:
        sc = camera_scene(sim, lighting=white, device=DEVICE, quality="full", look=look)
        out = cam.render(sc, state=sim.engine.state, output="all", seed=0)
    return out.linear.numpy()[0].astype(np.float64), out.albedo.numpy()[0].astype(np.float64)


def measure(root: Path | None, tracer: str, mesh: sr.ScanMesh, card: str = CARD, ss: int = 4,
            side=(1.0, 0.0, 0.0)) -> tuple[Score, Score]:
    """(colour score, albedo score) of ``card`` from ``root`` against ``mesh``'s reference, seen from ``side``."""
    sim = build(root, card)
    view, ref = head_on(sim, mesh, ss, side)
    mask = sr.law_mask(ref)
    assert mask.sum() > 800, f"only {mask.sum()} law pixels: the view misses the object's front"
    lin, alb = render(sim, view, tracer)
    rgb, albedo = score(lin, ref, mask), score(alb, ref, mask)
    log.info("%s %s: colour %s; albedo %s", card, tracer, rgb, albedo)
    return rgb, albedo


@pytest.fixture(scope="module")
def scan() -> sr.ScanMesh:
    return sr.load_scan(FIXTURE / "meshes" / CARD / "visual.glb")


def flat_in(lin: np.ndarray, mask: np.ndarray, color: tuple, tracer: str) -> None:
    """The panel is one flat colour with ``color``'s dominant channel (a look drew it, not the texture)."""
    px = lin[mask]
    assert px.std(0).max() < 0.03, f"{tracer}: the look did not replace the texture (std {px.std(0)})"
    k = int(np.argmax(color))
    assert px.mean(0)[k] > 3.0 * np.delete(px.mean(0), k).max(), f"{tracer}: not the look's colour {px.mean(0)}"


@pytest.mark.parametrize("tracer", TRACERS)
def test_scan_is_drawn_with_its_own_texture(tmp_path, monkeypatch, scan, tracer):
    """The law. The card resolves as a fetched scan does: its material has no look, so the camera's card look is the
    grey fallback, which must not cover the photo."""
    rgb, albedo = measure(as_fetched(monkeypatch, fixture_root(tmp_path)), tracer, scan)
    assert rgb.ok(), f"{tracer} tracer: the scan's texture is not drawn as its photo: {rgb}"
    assert albedo.mean <= MEAN_MAX and albedo.region <= REGION_MAX, f"{tracer}: the albedo guide misses it: {albedo}"


@pytest.mark.parametrize("tracer", TRACERS)
@pytest.mark.parametrize("mutant", ("flat", "vflip"))
def test_planted_mutants_fail_the_law(tmp_path, monkeypatch, scan, tracer, mutant):
    """The texture flattened to its mean colour (as a renderer that drops it draws), or v flipped (glTF's v runs down
    from the image's top, Blender's up from its bottom): the same law must fail, by a clear margin."""
    if mutant == "flat":
        bad = replace(scan, image=np.broadcast_to(scan.image.reshape(-1, 3).mean(0).round().astype(np.uint8),
                                                  scan.image.shape).copy())
    else:
        bad = replace(scan, uv=np.stack([scan.uv[:, 0], 1.0 - scan.uv[:, 1]], axis=1))
    rgb, _ = measure(as_fetched(monkeypatch, fixture_root(tmp_path, mesh=bad)), tracer, scan)
    assert not rgb.ok() and rgb.mean >= MUTANT_MIN, f"{tracer}: the {mutant} mutant passes the law: {rgb}"


@pytest.mark.parametrize("tracer", TRACERS)
def test_a_loader_that_drops_the_image_fails_the_law(tmp_path, monkeypatch, scan, tracer):
    """Planted in the code: the glb's image does not decode (as without Pillow), so the scan keeps a flat colour."""
    import alloy.camera.scan_textures as scan_textures
    monkeypatch.setattr(scan_textures, "decode_image", lambda data: None)
    rgb, _ = measure(as_fetched(monkeypatch, fixture_root(tmp_path)), tracer, scan)
    assert not rgb.ok() and rgb.mean >= MUTANT_MIN, f"{tracer}: a dropped image passes the law: {rgb}"


@pytest.mark.parametrize("tracer", TRACERS)
def test_a_factor_other_than_trimesh_default_scales_the_texture(tmp_path, monkeypatch, scan, tracer):
    """The fixture's baseColorFactor is trimesh's default 0.4 (read as 1, so the photo shows at full strength: the
    law's gain band); any other factor multiplies the texture, as glTF says."""
    half, _ = measure(fixture_root(tmp_path, mesh=replace(scan, factor=(0.5, 0.5, 0.5))), tracer, scan)
    full, _ = measure(fixture_root(tmp_path / "one", mesh=replace(scan, factor=(1.0, 1.0, 1.0))), tracer, scan)
    assert full.ok(), f"{tracer}: factor 1: {full}"
    assert np.allclose(half.gain / full.gain, 0.5, atol=0.05) and half.mean <= MEAN_MAX, f"{tracer}: {half} vs {full}"


@pytest.mark.parametrize("tracer", TRACERS)
def test_the_cards_own_look_overrides_the_texture(tmp_path, monkeypatch, scan, tracer):
    """A look written for the card (looks/objects/<card>.json) still wins: the panel is drawn flat in its colour."""
    green = (0.05, 0.45, 0.05)
    sim = build(as_fetched(monkeypatch, fixture_root(tmp_path, look=green)))
    view, ref = head_on(sim, scan)
    flat_in(render(sim, view, tracer)[0], sr.law_mask(ref), green, tracer)


def test_a_scene_look_assigned_to_the_scan_overrides_the_texture(tmp_path, monkeypatch, scan):
    """A scene look (``look=``) that assigns the object a material replaces its texture (full tracer)."""
    blue = (0.05, 0.05, 0.5)
    look = {"assign": [{"match": CARD, "material": "paint"}],
            "materials": {"paint": {"preset": "plastic_matte", "base_color": list(blue), "roughness": 0.8}}}
    sim = build(as_fetched(monkeypatch, fixture_root(tmp_path)))
    view, ref = head_on(sim, scan)
    flat_in(render(sim, view, "full", look=look)[0], sr.law_mask(ref), blue, "full")


@pytest.mark.parametrize("tracer", TRACERS)
def test_a_material_default_look_does_not_cover_the_photo(tmp_path, monkeypatch, scan, tracer):
    """A default look of the card's material (a colour for every object of that material) is below the scan's own
    photo: only a look written for the object, a scene look or an item look replaces it."""
    sim = build(as_fetched(monkeypatch, fixture_root(tmp_path, material_look=(0.6, 0.1, 0.1))))
    view, ref = head_on(sim, scan)
    mask = sr.law_mask(ref)
    lin, _ = render(sim, view, tracer)
    rgb = score(lin, ref, mask)
    assert rgb.ok(), f"{tracer}: the material's look covered the photo: {rgb}"


def _fetched(card: str) -> Path | None:
    cache = Path(os.environ.get("ALLOY_CACHE", Path.home() / ".cache" / "alloy")) / "imports" / "ycb_gso"
    glb = cache / "meshes" / card / "visual.glb"
    return glb if glb.is_file() else None


@pytest.mark.parametrize("tracer", TRACERS)
@pytest.mark.parametrize(("card", "side"), [("ycb_004_sugar_box", (1.0, 0.0, 0.0)),   # the printed front
                                          ("ycb_025_mug", (0.0, -1.0, 0.0))])       # the handle to one side
def test_fetched_scans_follow_the_law(tracer, card, side):
    """The whole fetched scans (4,096 px textures, every chart), when they are on disk (``alloy fetch <card>``)."""
    glb = _fetched(card)
    if glb is None:
        pytest.skip(f"{card} is not fetched (alloy fetch {card})")
    rgb, _ = measure(None, tracer, sr.load_scan(glb), card=card, ss=8, side=side)
    assert rgb.ok(), f"{card} {tracer}: {rgb}"
