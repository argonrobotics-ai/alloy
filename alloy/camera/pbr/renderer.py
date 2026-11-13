# SPDX-License-Identifier: Apache-2.0
"""Full-material renderer: a scene pack v2 frame (Blender's materials, lights, world and camera) on the GPU, with
Principled BSDFs, shader nodes, Bump, Bevel, area lights, an HDRI world and a thin-lens camera (Cycles 4.5).

Example::

    from alloy.camera.pbr.renderer import PbrScene, PbrRenderer
    ps = PbrScene.from_pack("c3_pack", frame=420, num_worlds=64, device="cuda:0")
    r = PbrRenderer(ps, 128, 128)                    # the pack's camera in every world
    rgb, albedo, normal = r.render(spp=16)           # wp.array(vec3) (64, 128, 128), linear, row 0 = top

``settings.backend = "optix"`` casts the rays on RT cores through the wavefront (``alloy.camera.wavefront``): same
shading code, same sampler, same outputs; build the OptiX helper once with ``python -m alloy.camera.optix build``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import warp as wp

from ..scene import IDENTITY_XF, Lighting, MeshGeom, SceneData
from ..settings import RenderSettings
from ..tracer.gpu_scene import GpuScene
from ..tracer.renderer import image_worlds, make_params
from ..tracer.tables import SOBOL_BURLEY_TABLE, filter_table
from .dispatch import generate
from .integrator import BumpTab, CamsV2, ParamsV2, make_kernel
from .pack_io import load_pack
from .corner_attrs import select_attributes
from .generic import build_generic_arrays
from .shade_data import build_lights, build_shade_data, camera_params, load_textures, read_textures

log = logging.getLogger(__name__)
_KERNELS: dict = {}


def settings_from_pack(static: dict, **over) -> RenderSettings:
    """RenderSettings from the pack's Cycles settings (bounces, clamps, filter, seed), with overrides."""
    c = static["settings"]["cycles"]
    st = RenderSettings(filter_type=c.get("pixel_filter_type", "BLACKMAN_HARRIS"),
                        filter_width=c.get("filter_width", 1.5), max_bounces=c["max_bounces"],
                        diffuse_bounces=c["diffuse_bounces"], glossy_bounces=c["glossy_bounces"],
                        transmission_bounces=c["transmission_bounces"], clamp_direct=c["sample_clamp_direct"],
                        clamp_indirect=c["sample_clamp_indirect"], min_light_bounces=c.get("min_light_bounces", 0),
                        seed=c.get("seed", 0))
    for k, v in over.items():
        setattr(st, k, v)
    return st


class PbrScene:
    """One frame of a scene pack on a device: geometry (GpuScene), shading data, lights, materials, textures."""

    def __init__(self, pack, frame, num_worlds: int = 1, device: str = "cuda:0", static_names=None, binding=None,
                 deform=None, liquid=None):
        """``binding``: optional (body (meshes, worlds), local (meshes, worlds, 7)[, present (meshes, worlds)]) from
        model_pack: meshes in body frames that follow ``state.body_q`` (see :meth:`update`), drawn where present.
        ``deform``: the :class:`~alloy.camera.deform_gpu.DeformSet` of the frame's last (per-world) meshes, followed
        by those of ``liquid`` (:class:`~alloy.camera.liquid_scene.LiquidSet`) when given."""
        self.pack, self.frame, self.device = pack, frame, str(device)
        self.from_model = binding is not None  # no camera of its own: cameras auto-frame like model scenes
        self.mat_names = list(pack.static["materials_used_by_meshes"])
        self.scene_data = (_scene_data(frame, num_worlds, static_names) if binding is None
                           else _bound_scene_data(frame, *binding))
        self.sc = GpuScene(self.scene_data, device=self.device, points=deform.points() if deform else None,
                           liquid=liquid.buffers() if liquid else None)
        uv_names, col_names = select_attributes(frame, pack.static, self.mat_names)
        self.shade = build_shade_data(frame, self.mat_names, self.sc, (col_names or [None])[0])
        self.lights = build_lights(pack, frame, num_worlds, self.device)
        self.disp, self.info = generate(pack.static, self.mat_names, uv_names, col_names)
        self.sc.set_transmissive(_transmissive_meshes(frame, self.mat_names, self.info["transmissive"]))
        self.images = self._images(pack)
        from .bsdf import principled as pr
        self.lut = pr.load_tables(self.device)
        self.bump = _bump_table(self.info, self.device)
        self.settings = settings_from_pack(pack.static)
        if deform is not None:
            deform.attach(self.sc, self.shade)
        if liquid is not None:
            liquid.attach(self.sc, self.shade)

    def _images(self, pack):
        """The material module's ``Images``: node-tree textures (i0, ...) and the generic materials' rows + atlas."""
        im, arrays = self.disp.Images(), pack.static.get("texture_arrays")
        for k, a in enumerate(load_textures(pack.root, self.info["images"], self.device, arrays)):
            setattr(im, f"i{k}", a)
        if any(self.info["generic"]):
            def read(files):
                return read_textures(pack.root, files, arrays)
            for field, a in build_generic_arrays(pack.static, self.mat_names, read, self.device).items():
                setattr(im, field, a)
        return im

    @staticmethod
    def from_pack(path, frame: int | None = None, num_worlds: int = 1, device: str = "cuda:0",
                  static_names=None) -> PbrScene:
        """Load ``frame`` (default: the first) of the pack at ``path``."""
        pack = load_pack(Path(path).expanduser())
        return PbrScene(pack, pack.frame(frame), num_worlds, device, static_names)

    @property
    def num_worlds(self) -> int:
        return self.sc.num_worlds

    def update(self, body_q: wp.array | None) -> None:
        """Move body-bound meshes to ``body_q`` and refit (see :meth:`GpuScene.update`)."""
        self.sc.update(body_q)


def _scene_data(frame, num_worlds: int, static_names) -> SceneData:
    """The frame's meshes (frame world space) as SceneData: one instance per world, or shared when static."""
    meshes = [MeshGeom(m.meta["name"], m.v, m.tri) for m in frame.meshes]
    static = [bool(static_names) and any(m.name.startswith(s) for s in static_names) for m in meshes]
    rows = [(k, -1) for k, s in enumerate(static) if s]
    rows += [(k, w) for w in range(num_worlds) for k, s in enumerate(static) if not s]
    mesh_idx, group = (np.array(x) for x in zip(*rows))
    n = len(rows)
    return SceneData(meshes, mesh_idx, group, np.full(n, -1), np.tile(IDENTITY_XF, (n, 1)), np.full((n, 3), 0.8),
                     num_worlds, [meshes[k].name for k in mesh_idx], Lighting([], (0.0, 0.0, 0.0)))


def _transmissive_meshes(frame, mat_names: list[str], flags: list[bool]) -> list[bool]:
    """Per frame mesh: True when one of its triangles uses a transmissive material (dispatch.is_transmissive)."""
    out = []
    for m in frame.meshes:
        slots = m.meta["materials"]
        used = {slots[k] for k in np.unique(np.asarray(m.mat, np.int64))}
        out.append(any(s is not None and flags[mat_names.index(s)] for s in used))
    return out


def _bound_scene_data(frame, body: np.ndarray, local: np.ndarray, present: np.ndarray | None = None) -> SceneData:
    """One instance of mesh k per world w on body ``body[k, w]`` (or fixed at ``local[k, w]``), where
    ``present[k, w]`` (default: everywhere)."""
    meshes = [MeshGeom(m.meta["name"], m.v, m.tri) for m in frame.meshes]
    k_n, w_n = body.shape
    keep = np.ones(k_n * w_n, bool) if present is None else np.asarray(present, bool).T.reshape(-1)
    idx = np.tile(np.arange(k_n), w_n)[keep]
    grp = np.repeat(np.arange(w_n), k_n)[keep]
    b = body.T.reshape(-1)[keep]
    loc = local.transpose(1, 0, 2).reshape(-1, 7)[keep]
    return SceneData(meshes, idx, grp, b, loc, np.full((len(idx), 3), 0.8), w_n, [meshes[k].name for k in idx],
                     Lighting([], (0.0, 0.0, 0.0)))


def _camera_background(static: dict):
    """The plain colour camera rays see (world_nodes.camera_background), None for a plain world or a Python world."""
    tree = (static.get("world") or {}).get("tree")
    if not tree:
        return None
    from .world_nodes import camera_background
    bg = camera_background(tree)
    return None if bg is None else [float(x) for x in bg]


def _bump_table(info: dict, device: str) -> BumpTab:
    """Per-material Bump, Bevel and absorption-volume parameters (last entry: the default surface, none)."""
    bump, bev = info["bump"] + [None], info["bevel"] + [None]
    vol = list(info.get("volume") or [None] * len(info["bump"])) + [None]

    def arr(vals, dtype):
        return wp.array(np.asarray(vals), dtype=dtype, device=device)

    bt = BumpTab()
    bt.on = arr([int(b is not None) for b in bump], wp.int32)
    bt.strength = arr([b["strength"] if b else 0.0 for b in bump], float)
    bt.distance = arr([b["distance"] if b else 0.0 for b in bump], float)
    bt.filter_width = arr([b["filter_width"] if b else 0.0 for b in bump], float)
    bt.invert = arr([int(b["invert"]) if b else 0 for b in bump], wp.int32)
    bt.bevel_on = arr([int(b is not None) for b in bev], wp.int32)
    bt.bevel_radius = arr([b["radius"] if b else 0.0 for b in bev], float)
    bt.bevel_samples = arr([b["samples"] if b else 0 for b in bev], wp.int32)
    bt.vol_on = arr([int(v is not None) for v in vol], wp.int32)
    bt.sigma_a = arr([v["sigma_a"] if v else [0.0, 0.0, 0.0] for v in vol], wp.vec3)
    return bt


class PbrRenderer:
    """Megakernel path tracer over a :class:`PbrScene`; image i looks into world ``worlds[i]``."""

    def __init__(self, ps: PbrScene, width: int, height: int, settings: RenderSettings | None = None, worlds=None,
                 block_dim: int = 32, wavefront=None):
        """``wavefront``: optional :class:`~alloy.camera.wavefront.knobs.WavefrontOptions` (backend != "warp")."""
        self.ps, self.sc, self.w, self.h = ps, ps.sc, int(width), int(height)
        self.st = settings or ps.settings
        self.block_dim = int(block_dim)
        self.worlds = image_worlds(worlds, ps.num_worlds)
        self.n_img = len(self.worlds)
        key = ps.disp.__name__
        if key not in _KERNELS:
            _KERNELS[key] = make_kernel(ps.disp)
        self.kernel = _KERNELS[key]
        self.cams = self._pack_cameras()
        c = ps.pack.static["settings"]["cycles"]
        self.p2 = ParamsV2()
        self.p2.max_glossy = self.st.glossy_bounces + 1
        self.p2.blur_glossy = float(c.get("blur_glossy", 1.0))
        self.p2.caustics_reflective = int(bool(c.get("caustics_reflective", True)))
        self.p2.caustics_refractive = int(bool(c.get("caustics_refractive", True)))
        self.p2.max_transmission = self.st.transmission_bounces + 1
        self.p2.lights_visible_camera = 0
        cam_bg = _camera_background(ps.pack.static)
        self.p2.cam_bg_on = int(cam_bg is not None)
        self.p2.cam_bg = wp.vec3(*(cam_bg if cam_bg is not None else (0.0, 0.0, 0.0)))
        dev = self.sc.device
        self.ftab = wp.array(filter_table(self.st.filter_type, self.st.filter_width), dtype=float, device=dev)
        self.sobol = wp.array(SOBOL_BURLEY_TABLE.reshape(-1), dtype=wp.uint32, device=dev)
        n = self.n_img * self.w * self.h
        self.rgb, self.albedo, self.normal = (wp.zeros(n, dtype=wp.vec3, device=dev) for _ in range(3))
        self.wf = self._make_wavefront(wavefront)

    def _make_wavefront(self, options):
        """The wavefront renderer when ``settings.backend`` asks for one ("optix", "wavefront-warp", "auto"), else None."""
        from ..wavefront.tracers import make_tracer, resolve_backend
        backend = resolve_backend(self.st.backend, self.sc.device)
        if backend == "warp":
            return None
        from ..wavefront.knobs import WavefrontOptions
        from ..wavefront.pbr import PbrWavefront
        return PbrWavefront(self, make_tracer(self.sc, backend), options or WavefrontOptions.from_env())

    def _pack_cameras(self) -> CamsV2:
        """Every image starts with the pack frame's camera (thin lens when the pack uses depth of field)."""
        cp, n, dev = camera_params(self.ps.frame, self.w, self.h), self.n_img, self.sc.device

        def tile(v, dtype):
            return wp.array(np.tile(np.asarray(v, np.float32), (n,) + (1,) * np.ndim(v)), dtype=dtype, device=dev)

        cams = CamsV2()
        cams.world = wp.array(self.worlds, dtype=wp.int32, device=dev)
        cams.eye, cams.rot = tile(cp["eye"], wp.vec3), tile(cp["rot"], wp.mat33)
        cams.extent, cams.clip = tile(cp["extent"], wp.vec2), tile(cp["clip"], wp.vec2)
        cams.aperture = wp.array(np.full(n, cp["aperture"], np.float32), dtype=float, device=dev)
        cams.focus = wp.array(np.full(n, cp["focus"], np.float32), dtype=float, device=dev)
        return cams

    def set_cameras(self, eye, rot, extent, clip) -> None:
        """Per-image pinhole cameras (host arrays, as tracer.Renderer.set_cameras); the pack's lens is dropped."""
        self.cams.eye.assign(np.asarray(eye, np.float32).reshape(self.n_img, 3))
        self.cams.rot.assign(np.asarray(rot, np.float32).reshape(self.n_img, 3, 3))
        self.cams.extent.assign(np.asarray(extent, np.float32).reshape(self.n_img, 2))
        self.cams.clip.assign(np.asarray(clip, np.float32).reshape(self.n_img, 2))
        self.cams.aperture.zero_()

    def render(self, spp: int, sample_offset: int = 0, total_spp: int | None = None,
               seed: int | None = None) -> tuple[wp.array, ...]:
        """(rgb, albedo, normal) means over samples [sample_offset, sample_offset + spp), (n_images, H, W) vec3."""
        prm = make_params(self.st, self.w, self.h, spp, sample_offset, total_spp or spp, seed)
        ps = self.ps
        if self.wf is not None:
            self.wf.render(spp, prm)
        else:
            wp.launch(self.kernel, dim=self.n_img * self.w * self.h,
                      inputs=[self.sc.geo, ps.shade, ps.bump, ps.lights, self.cams, prm, self.p2, self.ftab,
                              self.sobol, ps.lut, ps.images], outputs=[self.rgb, self.albedo, self.normal],
                      device=self.sc.device, block_dim=self.block_dim)
        shp = (self.n_img, self.h, self.w)
        return tuple(a.reshape(shp) for a in (self.rgb, self.albedo, self.normal))

    def render_converged(self, total_spp: int, chunk: int = 256) -> tuple[np.ndarray, ...]:
        """Mean of ``total_spp`` samples in chunks (float64 on the host), numpy (n_images, H, W, 3)."""
        shp = (self.n_img, self.h, self.w, 3)
        acc = [np.zeros(shp, np.float64) for _ in range(3)]
        done = 0
        while done < total_spp:
            k = min(chunk, total_spp - done)
            for a, o in zip(acc, self.render(k, done, total_spp)):
                a += o.numpy().reshape(shp).astype(np.float64) * k
            done += k
        return tuple(a / total_spp for a in acc)
