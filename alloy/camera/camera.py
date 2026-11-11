# SPDX-License-Identifier: Apache-2.0
"""``Camera``: a path-traced camera sensor, batched over every world of a simulation, outputs on the GPU.

Example::

    from alloy._vendor import newton; import alloy.camera as ac
    b = newton.ModelBuilder(); b.add_ground_plane()
    b.add_shape_box(b.add_body(xform=((0, 0, 0.2), (0, 0, 0, 1))), hx=0.1, hy=0.1, hz=0.1)
    model = b.finalize()
    cam = ac.Camera("front", look_at=((1.0, -1.0, 0.8), (0.0, 0.0, 0.1)), width=128, height=128, fov=50, spp=16)
    img = cam.render(model, model.state())        # wp.array uint8 (worlds, 128, 128, 3), row 0 = top
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from .errors import CameraError
from .item_fast import with_item_looks
from .pose import as_poses, image_extent, look_at
from .scan_textures import with_scan_textures
from .scene import Lighting, SceneData
from .settings import RenderSettings
from .tracer.gpu_scene import GpuScene
from .tracer.renderer import Renderer
from .view import STANDARD, View, clipped_fraction, to_display, view_from_settings

log = logging.getLogger(__name__)
_SCENES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_PACKS: dict = {}
OUTPUTS = ("uint8", "linear", "all")
CLIP_WARN = 0.2  # warn when more than this share of a first image's pixels clip to white


@wp.kernel
def attach_poses(body_q: wp.array(dtype=wp.transform), cam_body: wp.array(dtype=wp.int32),
                 cam_local: wp.array(dtype=wp.transform), eye: wp.array(dtype=wp.vec3),
                 rot: wp.array(dtype=wp.mat33)):
    """Camera pose of image i = body pose x camera pose in the body frame (cam_body < 0: the pose is in world)."""
    i = wp.tid()
    t = cam_local[i]
    b = cam_body[i]
    if b >= 0:
        t = wp.transform_multiply(body_q[b], t)
    eye[i] = wp.transform_get_translation(t)
    rot[i] = wp.quat_to_matrix(wp.transform_get_rotation(t))


@dataclass
class CameraOutput:
    """Render result on the device: ``rgb`` (uint8 display or linear), plus linear guide buffers when asked, plus
    the depth / id channels of ``alloy.camera.aov`` a tuple ``output`` names (fields not asked for are None)."""

    rgb: wp.array | None
    linear: wp.array | None = None  # denoised when the camera denoises
    albedo: wp.array | None = None
    normal: wp.array | None = None
    noisy: wp.array | None = None  # the raw path-traced colour (same as linear without a denoiser)
    depth: wp.array | None = None  # float32 metres along the optical axis, inf = nothing hit (aov.py)
    segmentation: wp.array | None = None  # int32: 0 nothing, 1 not on a body, 2 + k = k-th body of its world
    body_id: wp.array | None = None  # int32 index into state.body_q; -1 not on a body, -2 nothing hit
    world_id: wp.array | None = None  # int32; -1 shared between worlds, -2 nothing hit
    instance_id: wp.array | None = None  # int32: 0 nothing, 1 + k = k-th drawn shape of its world
    semantic: wp.array | None = None  # int32: 0 nothing, 1 no class, 2 + k = k-th class (aov.set_semantics)
    flow: wp.array | None = None  # float32 (..., 2) pixels moved since the last flow render; NaN = not tracked


def default_device() -> str:
    """'cuda:0' when a CUDA GPU is present, else 'cpu' (slow but works)."""
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def camera_scene(source, lighting: Lighting | None = None, device: str | None = None, look=None,
                 quality: str = "auto", items="auto"):
    """The device scene for a source, built once per (source, device, look) and shared by cameras.

    Sources: a newton.Model, an ``alloy.camera.SceneData``, a scene pack directory (see :func:`load_pack`) or an
    already built scene. A model renders with the fast diffuse tracer (shape colours, ``lighting``) unless a
    ``look`` (look-file dict or path: materials, lights, HDRI world) is given or ``quality="full"``; then it renders
    with Blender's materials and lights (full-material tracer; the first render compiles for about a minute).
    A Sim's scene item looks (scene variants) render on the full-material tracer, or as colours and tiled colour maps
    with ``quality="fast"`` (``item_fast``): ``items`` is that layer (``items_look``; "auto": the source's; None: none).
    Example: ``camera_scene(model, look="my.look.json")`` before creating cameras.
    """
    if isinstance(source, GpuScene) or _is_pbr(source):
        return source
    if items == "auto":
        from .item_looks import items_look
        items = items_look(source)
    from .deform import has_deformables
    from .films import has_films
    from .liquid_scene import has_liquids
    sim, dyn = source, (has_deformables(source), has_liquids(source), has_films(source))  # + SurfaceFlow films
    source = unwrap_sim(source)[0]
    if quality not in ("auto", "fast", "full"):
        raise CameraError(f"quality {quality!r}", "use 'auto', 'fast' or 'full'")
    dev = str(device or default_device())
    if isinstance(source, (str, Path)):
        return load_pack(source, device=dev)
    per = _SCENES.setdefault(source, {}) if _weakrefable(source) else {}
    if (look is not None or quality == "full" or (items and quality != "fast")) and not isinstance(source, SceneData):
        key = (dev, "full", _look_key(look), _look_key(items), id(lighting), dyn)
        if key not in per:
            per[key] = _model_pbr(source, look, lighting, dev, items, _layouts(sim, dyn), sim)
        return per[key]
    key = (dev, dyn, sim is not source, _look_key(items) if items else None)  # a Sim: scan textures (and item looks)
    if key not in per or lighting is not None:
        sd = source if isinstance(source, SceneData) else with_item_looks(
            with_scan_textures(_scene_from(source, lighting), sim), items)
        if lighting is not None and isinstance(source, SceneData):
            sd.lighting = lighting
        per[key] = _fast_scene(sd, dev, _layouts(sim, dyn), source)
    return per[key]


def _layouts(sim, dyn: tuple) -> tuple:
    """(DeformLayout, LiquidLayout, engine, FilmLayout) of a sim, each None when it has none (films: SurfaceFlow)."""
    from .deform import deform_objects
    from .films import film_objects
    from .liquid_scene import liquid_objects
    return (deform_objects(sim) if dyn[0] else None, liquid_objects(sim) if dyn[1] else None,
            getattr(sim, "engine", None), film_objects(sim) if dyn[2] else None)


def _forget(sc) -> None:
    """Drop a device scene from the cache (the next camera_scene builds it again)."""
    for per in _SCENES.values():
        for k in [k for k, v in per.items() if v is sc]:
            del per[k]


def _fast_scene(sd: SceneData, device: str, layouts: tuple, model) -> GpuScene:
    """The diffuse tracer's device scene, with the deformables', liquids' and films' per-world meshes (``_layouts``)."""
    layout, liquid, engine, films = layouts
    if layout is None and liquid is None and films is None:
        return GpuScene(sd, device=device)
    from .deform_gpu import DeformSet
    from .films import Dynamic, FilmSet
    from .liquid_scene import LiquidSet
    ds = DeformSet(layout, len(sd.meshes), device, model) if layout is not None else None
    sd = ds.scene_data(sd) if ds is not None else sd
    ls = LiquidSet(liquid, len(sd.meshes), device, engine) if liquid is not None else None
    sd = ls.scene_data(sd) if ls is not None else sd
    fs = FilmSet(films, len(sd.meshes), device) if films is not None else None
    sd = fs.scene_data(sd) if fs is not None else sd
    pts = Dynamic(ds, fs).points() if ds is not None or fs is not None else None
    gs = GpuScene(sd, device=device, points=pts, liquid=ls.buffers() if ls else None)
    for dyn in (ds, ls, fs):
        if dyn is not None:
            dyn.attach(gs)
    return gs


def scene_view(sc) -> View:
    """The display transform of a device scene: a full-material scene's ``settings["view_settings"]`` (model scenes:
    the look's ``render`` view, AgX by default, as Blender renders), else the Standard view."""
    pack = getattr(sc, "pack", None)
    vs = (pack.static.get("settings") or {}).get("view_settings") if pack is not None else None
    return view_from_settings(vs) if vs else STANDARD


def _warn_if_clipped(disp: wp.array, name: str, stops: float, view: View) -> None:
    """Log a warning when more than CLIP_WARN of the pixels are blown out (a camera's first batch, up to 16 worlds)."""
    frac = clipped_fraction(disp[: min(16, disp.shape[0])].numpy())
    if frac > CLIP_WARN:
        log.warning("camera %s: %.0f%% of pixels clip to white (%s view, exposure %+.1f stops); lower the exposure "
                    "(render(..., exposure=-1) or the look's render.exposure) or dim the lights", name, 100 * frac,
                    "AgX" if view.transform == "agx" else "Standard", stops)


def _look_key(look) -> str:
    import json
    return look if isinstance(look, (str, Path)) else json.dumps(look, sort_keys=True, default=str)


def _model_pbr(model, look, lighting, device: str, items: dict | None = None, layouts=(None,) * 4, sim=None):
    """A newton.Model as a full-material scene (look materials and lights, item looks; see model_pack), with the
    deformables, liquids and films of ``layouts`` (``_layouts``) as per-world meshes refreshed every render."""
    from .pbr import PbrScene
    from .scan_pack import pack_with_scans
    layout, liquid, engine, films = layouts
    pk, frame, binding = pack_with_scans(model, look, lighting, items, layout, liquid, sim)
    n_liq = 0 if liquid is None else liquid.worlds * len(liquid.objects)
    ds = ls = None
    if layout is not None:
        from .deform_gpu import DeformSet
        ds = DeformSet(layout, len(frame.meshes) - n_liq - layout.worlds * len(layout.objects), device, model)
    if liquid is not None:
        from .liquid_scene import LiquidSet
        ls = LiquidSet(liquid, len(frame.meshes) - n_liq, device, engine)
    if films is not None:  # after the deformables and liquids: their first meshes are counted without the films
        from .films import Dynamic, add_to_pack
        binding, fs = add_to_pack(pk, frame, binding, films, device)
        ds = Dynamic(ds, fs)
    return PbrScene(pk, frame, binding[0].shape[1], device, binding=binding, deform=ds, liquid=ls)


def load_pack(path, frame: int | None = None, num_worlds: int = 1, device: str | None = None, static=None,
              cache: bool = True):
    """A scene pack ready to render, cached (``cache=False`` builds a fresh scene the cache does not keep).
    v2 (``pack.json``): Blender materials as node graphs, area lights, an
    HDRI world and the camera, rendered by the full-material tracer. v1 (``scene.json``): coloured meshes, a sun and
    a sky, rendered by the diffuse tracer. ``static``: mesh-name prefixes shared by every world (one BVH copy).

    Example: ``scene = load_pack("c3_pack", frame=420, num_worlds=64); Camera(width=128).render(scene)``.
    """
    p = Path(path).expanduser().resolve()
    dev = str(device or default_device())
    key = (str(p), frame, int(num_worlds), dev, tuple(static or ()))
    if key in _PACKS:
        return _PACKS[key]
    if (p / "pack.json").exists():
        from .pbr import PbrScene
        scene = PbrScene.from_pack(p, frame, num_worlds, dev, static)
    elif (p / "scene.json").exists():
        from .pack_v1 import scene_from_pack_v1
        scene = GpuScene(scene_from_pack_v1(p, num_worlds, static), device=dev)
    else:
        raise CameraError(f"{p} is not a scene pack", "pass a folder with pack.json (v2) or scene.json (v1)")
    if cache:
        _PACKS[key] = scene
    return scene


def _is_pbr(obj) -> bool:
    return type(obj).__name__ == "PbrScene"


def unwrap_sim(source, state=None):
    """(model, state) of an ``alloy.Sim`` (``sim.engine.model`` / ``sim.engine.state``); other sources unchanged."""
    engine = getattr(source, "engine", None)
    if engine is not None and hasattr(engine, "model"):
        return engine.model, (state if state is not None else getattr(engine, "state", None))
    if hasattr(source, "plugins") and hasattr(source, "model") and hasattr(source, "state"):  # an engine simulation
        return source.model, (state if state is not None else source.state)
    return source, state


def _weakrefable(obj) -> bool:
    try:
        return weakref.ref(obj) is not None
    except TypeError:
        return False


def _scene_from(source, lighting: Lighting | None) -> SceneData:
    if hasattr(source, "shape_type") and hasattr(source, "shape_body"):
        from .from_newton import scene_from_model
        return scene_from_model(source, lighting)
    raise CameraError(f"cannot render a {type(source).__name__}",
                      "pass a newton.Model, an alloy.camera.SceneData or a scene from camera_scene()")


class Camera:
    """A camera over every world: fixed pose, per-world poses, or attached to a body (pose in the body frame)."""

    def __init__(self, name: str = "camera", pose=None, look_at=None, attach_to=None, width: int = 128,
                 height: int = 128, fov: float = 60.0, spp: int = 16, settings: RenderSettings | None = None,
                 clip: tuple[float, float] = (0.01, 1000.0), fit: str = "VERTICAL", denoise=None,
                 backend: str | None = None):
        """``denoise``: None, "auto", "learned", "oidn" or a :class:`~alloy.camera.denoise.DenoiseSpec` (weights,
        tta); "auto" lets 4-8 spp look like hundreds of samples. ``backend``: "warp", "optix", "auto" or
        "wavefront-warp" (see :class:`RenderSettings`); None keeps the settings' own (``ALLOY_CAMERA_BACKEND``, else
        "warp")."""
        if pose is not None and look_at is not None:
            raise CameraError("both pose and look_at given", "pass one of them")
        if int(spp) < 1:
            raise CameraError(f"spp is {spp}", "use at least 1 sample per pixel (16 is a good start)")
        self.name, self.attach_to, self.spp = name, attach_to, int(spp)
        self.width, self.height, self.fov, self.fit = int(width), int(height), float(fov), fit
        self.extent = image_extent(self.fov, self.width, self.height, fit)
        self.clip = (float(clip[0]), float(clip[1]))
        self._settings = settings  # None: Cycles defaults, or the pack's own settings for a scene pack
        self.settings = settings or RenderSettings()
        self._pose = _initial_pose(pose, look_at)
        self._bound: tuple | None = None
        self._dirty = True
        self._clip_check = True
        from .denoise import as_spec
        self.denoise, self._denoiser = as_spec(denoise), None
        self.backend = backend

    # ------------------------------------------------------------------ pose
    def set_pose(self, pose=None, look_at=None) -> None:
        """New pose (7,) or per-world (worlds, 7); or ``look_at=(eye, target)``. In the body frame when attached."""
        self._pose = _initial_pose(pose, look_at)
        self._dirty = True

    # ------------------------------------------------------------------ render
    def render(self, source, state=None, output: str = "uint8", exposure: float = 0.0, seed: int | None = None):
        """Render every world of ``source`` at ``state``: an ``alloy.Sim`` (its current state), a newton.Model, a
        SceneData, a scene pack folder or a ``camera_scene``. Example: ``batch = cam.render(sim)``.

        Returns a wp.array: uint8 (worlds, H, W, 3) for ``output="uint8"``, float32 linear for ``"linear"``, or a
        :class:`CameraOutput` with rgb, linear, albedo and normal for ``"all"``. The arrays are the camera's own
        buffers: the next render overwrites them, so copy what you keep. ``seed`` changes the noise of this render.
        ``output="depth"`` (metres), ``"segmentation"``, ``"body_id"`` or ``"world_id"`` gives that image, (worlds,
        H, W), from the pixel-centre ray with no colour traced; a tuple such as ``("rgb", "depth")`` a
        :class:`CameraOutput` with those too (``alloy.camera.aov``; fast camera only).
        """
        from .aov import keep_fields, parse_output, render_aovs
        want = parse_output(output, OUTPUTS)  # None: a colour-only output, the unchanged path
        _not_capturing(source, self.name)
        from .item_looks import items_look
        items = items_look(source)
        sim = source
        source, state = unwrap_sim(source, state)
        sc, r = self._bind(sim, items)
        dyn = _dynamic(sc)
        if any(d.stale() for d in dyn):
            log.warning("camera %s: the deformables' topology changed (cut or torn); rebuilding the scene", self.name)
            _forget(sc)
            sc, r = self._bind(sim, items)
            dyn = _dynamic(sc)
        if state is not None:
            for d in dyn:
                d.update(state)  # before the top-level refit below
        sc.update(getattr(state, "body_q", None) if state is not None else None)
        if self._dirty:
            self._upload_pose(r)
        if self._attached:
            self._attach(state)
        aovs = render_aovs(r, want.aovs) if want is not None else {}
        if want is not None and not want.color:
            return aovs[want.aovs[0]] if want.single else CameraOutput(None, **aovs)
        noisy, alb, nrm = r.render(self.spp, seed=seed)
        rgb = self._denoised(noisy, alb, nrm, r)
        if output == "linear":
            return rgb
        view = scene_view(sc)
        disp = to_display(rgb, exposure, view=view)
        if self._clip_check:
            self._clip_check = False
            _warn_if_clipped(disp, self.name, exposure + view.exposure, view)
        if output == "uint8":
            return disp
        out = CameraOutput(disp, rgb, alb, nrm, noisy, **aovs)
        return out if want is None else keep_fields(out, want.names)

    def _denoised(self, rgb: wp.array, alb: wp.array, nrm: wp.array, r) -> wp.array:
        if self.denoise.kind in (None, "off"):
            return rgb
        if self._denoiser is None:
            from .denoise import make_denoiser
            self._denoiser = make_denoiser(self.denoise, self.width, self.height, r.sc.device) or False
        return self._denoiser(rgb, alb, nrm, self.spp) if self._denoiser else rgb

    def render_torch(self, source, state=None, output: str = "uint8", **kw):
        """Same as :meth:`render`, as a torch tensor on the same device (needs PyTorch); a tuple ``output`` gives a
        dict of tensors by channel name."""
        out = self.render(source, state, output, **kw)
        if isinstance(out, CameraOutput) and output == "all":
            raise CameraError("render_torch with output='all'", "use render(...) and wp.to_torch on each field")
        try:
            if isinstance(out, CameraOutput):
                from .aov import as_torch
                return as_torch(out)
            return wp.to_torch(out)
        except ImportError as exc:
            raise CameraError("PyTorch is not installed", "pip install torch, or use render() for warp arrays") from exc

    # ------------------------------------------------------------------ internals
    def _bind(self, source, items: dict | None = None):
        model = unwrap_sim(source)[0]
        sc = camera_scene(source, device=getattr(model, "device", None), items=items)
        if self._bound is None or self._bound[0] is not sc:
            if _is_pbr(sc):
                from .pbr import PbrRenderer
                r = PbrRenderer(sc, self.width, self.height, self._with_backend(self._settings or sc.settings))
            else:
                r = Renderer(sc, self.width, self.height, self._with_backend(self.settings))
            self._bound = (sc, r)
            self._resolve_attach(sc, source)
            self._dirty = True
            self._clip_check = True
        return self._bound

    def _with_backend(self, settings: RenderSettings) -> RenderSettings:
        return settings if self.backend is None else replace(settings, backend=self.backend)

    def _resolve_attach(self, sc: GpuScene, source=None) -> None:
        n = sc.num_worlds
        self._attached = self.attach_to is not None
        bodies = np.full(n, -1, np.int32)
        if self._attached:
            bodies = _sim_bodies(source, self.attach_to)
            bodies = _attach_bodies(sc, self.attach_to, _sim_names(source)) if bodies is None else bodies
        self._cam_body = wp.array(bodies, dtype=wp.int32, device=sc.device)

    def _upload_pose(self, r: Renderer) -> None:
        n = r.n_img
        if self._pose is None and hasattr(r, "ps") and not r.ps.from_model:
            self._dirty = False  # keep the pack's own camera (lens, focus, field of view)
            return
        own = getattr(r.sc.sd, "camera", None)
        if self._pose is None and own is not None:  # the scene's own view (v1 packs)
            self._pose = own["pose"]
            self.fov, self.fit, self.clip = own["fov_deg"], own["fit"], own["clip"]
            self.extent = image_extent(self.fov, self.width, self.height, self.fit)
        if self._pose is None:  # auto-frame what the first image's world shows now
            from .pose import auto_frame
            self._pose = auto_frame(*r.sc.world_bounds(int(r.worlds[0])))
        poses = as_poses(self._pose, n)
        self._cam_local = wp.array(poses, dtype=wp.transform, device=r.sc.device)
        from .scene import quat_to_matrix
        r.set_cameras(poses[:, :3], np.stack([quat_to_matrix(q) for q in poses[:, 3:]]),
                      np.tile(self.extent, (n, 1)), np.tile(self.clip, (n, 1)))
        self._dirty = False

    def _attach(self, state) -> None:
        if state is None or getattr(state, "body_q", None) is None:
            raise CameraError(f"camera {self.name!r} is attached to a body but got no state",
                              "pass the simulation state: cam.render(model, state)")
        r = self._bound[1]
        wp.launch(attach_poses, dim=r.n_img, inputs=[state.body_q, self._cam_body, self._cam_local],
                  outputs=[r.cams.eye, r.cams.rot], device=r.sc.device)


def _not_capturing(source, name: str) -> None:
    """Raise when the source's device is capturing a CUDA graph: a render reads results back to the host and builds
    acceleration structures, which must not be recorded into (or run inside) a simulation's graph."""
    dev = getattr(unwrap_sim(source)[0], "device", None)
    try:
        capturing = dev is not None and wp.get_device(dev).is_capturing
    except (TypeError, ValueError, RuntimeError):
        capturing = False
    if capturing:
        raise CameraError(f"camera {name}: render() was called while {dev} captures a CUDA graph",
                          "render between graph launches (after sim.step), not from inside a captured step")


def _dynamic(sc) -> list:
    """The DeformSet, LiquidSet and FilmSet of a device scene (GpuScene, or a PbrScene's), those it has."""
    gs = getattr(sc, "sc", sc)
    return [d for d in (getattr(gs, n, None) for n in ("deform", "liquid", "film")) if d is not None]


def _initial_pose(pose, look_at_arg):
    if look_at_arg is not None:
        eye, target = look_at_arg[0], look_at_arg[1]
        up = look_at_arg[2] if len(look_at_arg) > 2 else (0.0, 0.0, 1.0)
        return look_at(eye, target, up)
    if pose is None:
        return None
    if isinstance(pose, (tuple, list)) and len(pose) == 2 and np.size(pose[0]) == 3:
        return np.concatenate([np.asarray(pose[0], np.float64), np.asarray(pose[1], np.float64)])
    return np.asarray(pose, np.float64)


def _sim_names(source) -> list[str]:
    """Body names of an ``alloy.Sim`` (``sim.names()["bodies"]``); [] for any other source."""
    return list(getattr(getattr(source, "index", None), "bodies", None) or [])


def _sim_bodies(source, attach_to) -> np.ndarray | None:
    """Body index per world of an ``alloy.Sim`` body name (any name ``sim.names()["bodies"]`` lists, e.g.
    ``"arm/link1"``); None when the source is not a sim or the name is not one of its bodies."""
    index = getattr(source, "index", None)
    if not isinstance(attach_to, str) or attach_to not in (getattr(index, "bodies", None) or {}):
        return None
    return np.asarray(index.per_world(index.bodies[attach_to], "body"), np.int32)


def _attach_bodies(sc: GpuScene, attach_to, names: list[str] | None = None) -> np.ndarray:
    """Body index per world for ``attach_to``: a body label, a body index (world 0's body; its copies in the other
    worlds are found by label), or one body index per world. ``names``: the sim's own body names, for the error."""
    n = sc.num_worlds
    if np.ndim(attach_to) == 1:
        out = np.asarray(attach_to, np.int32)
        if out.shape != (n,):
            raise CameraError(f"attach_to has {out.size} bodies for {n} worlds", "give one body index per world")
        return out
    model = next((m for m, per in _SCENES.items() if sc in per.values()), None)
    labels = list(getattr(model, "body_label", None) or getattr(model, "body_key", None) or []) if model else []
    if not labels:
        if isinstance(attach_to, (int, np.integer)):
            return np.full(n, int(attach_to), np.int32)
        raise CameraError(f"attach_to={attach_to!r} needs a newton.Model source with body labels",
                          "render a newton.Model, or pass body indices instead of a label")
    world = model.body_world.numpy() if getattr(model, "body_world", None) is not None else np.zeros(len(labels))
    if isinstance(attach_to, (int, np.integer)):
        attach_to = labels[int(attach_to)]
    hits = [i for i, lb in enumerate(labels) if lb == attach_to or lb.endswith("/" + str(attach_to))]
    out = np.full(n, -1, np.int32)
    for i in hits:
        w = int(world[i])
        if 0 <= w < n and out[w] < 0:
            out[w] = i
    for i in hits:  # a body outside any world (world -1) serves every world that has no own copy
        if int(world[i]) < 0:
            out[out < 0] = i
    if (out < 0).any():
        raise CameraError(f"no body {attach_to!r} in world {int(np.flatnonzero(out < 0)[0])}",
                          f"use one of the body names: {names}" if names else
                          f"use one of the body labels, e.g. {sorted(set(labels))[:6]}")
    return out
