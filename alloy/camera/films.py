# SPDX-License-Identifier: Apache-2.0
"""SurfaceFlow films in the camera: paint coats, concrete paste, rinse water and soil, drawn as a look layer.

A film is drawn as a skin of quads, one per chart cell, a fraction of a millimetre above the cell (its chart frame
follows the body it rides on). The look sorts every cell into a few appearance levels ("bins"), each one mesh per
world with its own colour and roughness; a cell is drawn in the bin its state calls for and collapsed to a point
(not drawn) in every other. One kernel per film and render reads the plug-in's per-cell state on the device
(``alloy.plugins.surface_flow.render_view``: no host copy, no writes), then the bins' meshes are refitted like the
deformables' (``alloy.camera.deform_gpu``). A sim without a ``SurfaceFlow`` plug-in builds no film, so its images are
unchanged bit for bit.

Looks (``FilmLook``; ``kind="auto"`` picks from the plug-in): ``paint`` (a drying stage or a "solids" species): the
paint's colour over the substrate by dry film thickness, opaque from the hiding thickness, glossy while wet;
``soil`` (a soil stage): the soil's colour by the share of the start load left, darker and glossy where wet;
``wet`` (anything else: concrete paste, water): darkening and sheen growing with the film depth.

Example::

    from alloy.camera.films import FilmLook, set_film_look
    set_film_look(flow, FilmLook(color=(0.6, 0.05, 0.04)))      # a red coat (linear RGB)
    img = alloy.Camera("c", width=320, height=240).render(sim)    # the coat as the plug-in has it now
"""
from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from .errors import CameraError

log = logging.getLogger(__name__)

PAINT, SOIL, WET = 0, 1, 2
KINDS = {"paint": PAINT, "soil": SOIL, "wet": WET}
OFFSET_M = 2.0e-4         # the skin's height above the cell: over a paint coat, under any tool that touches it
MAX_TRIS = 1 << 26        # drawn film triangles over all bins and worlds of one camera scene
HIDE_FRAC = 0.98          # opacity at the hiding thickness (paint_metrics' hiding fraction)
PAINT_Q = (0.15, 0.4, 0.75, 1.0)     # dft / hide_dft drawn by each paint level (< 0.25, < 0.5, < 1, >= 1)
SOIL_A = (0.0, 0.35, 0.7, 1.0)       # soil colour share of each soil level (clean, < 10 %, < 50 %, more)
WET_DARK = (0.85, 0.72, 0.62)        # darkening of each wet level (depth < 0.2 mm, < 1 mm, more)
WET_ROUGH = (0.35, 0.2, 0.08)
DEFAULT_BASE = (0.6, 0.6, 0.6)
DEFAULT_COLOR = {PAINT: (0.05, 0.13, 0.42), SOIL: (0.32, 0.12, 0.05)}   # a blue coat, tomato-paste soil


@dataclass(frozen=True)
class FilmLook:
    """How one film is drawn. Colours are linear RGB; ``base`` None takes the chart's shape colour in the scene.
    ``painted_dft`` / ``clean_soil``: thinner coats and lighter soil are not drawn; ``hide_dft`` None: the drying
    stage's hiding thickness (else 33 um); ``soil_ref`` None: the largest start load."""

    kind: str = "auto"
    color: tuple | None = None
    base: tuple | None = None
    painted_dft: float = 1.0e-6
    hide_dft: float | None = None
    clean_soil: float = 1.0e-6
    soil_ref: float | None = None


_LOOKS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def set_film_look(flow, look: FilmLook) -> None:
    """Draw the ``SurfaceFlow`` plug-in ``flow`` with ``look`` (kept beside the plug-in; nothing is written to it).
    Set it before the first render: a camera scene keeps the look it was built with."""
    if look.kind not in ("auto", *KINDS):
        raise CameraError(f"film look kind {look.kind!r}", f"use 'auto' or one of {sorted(KINDS)}")
    _LOOKS[flow] = look


@wp.struct
class LookParams:
    mode: int
    wet_depth: float
    lo: float           # paint: painted_dft; soil: clean_soil
    ref: float          # paint: hide_dft; soil: soil_ref
    phi_m: float        # paint: solids fraction of a dry coat (0: wet whenever deeper than wet_depth)
    solids: int


@dataclass(frozen=True)
class FilmObject:
    """One film to draw: ``corners`` (C, 4, 3) are its cells' corners in their chart frames, lifted by
    :data:`OFFSET_M`; ``cell_chart`` (C,) the chart of each cell; ``base`` the substrate colour per chart (None: from
    the scene); ``bins`` (colour, roughness, name) per appearance level."""

    name: str
    view: object
    look: FilmLook
    params: LookParams
    corners: np.ndarray
    normals: np.ndarray
    cell_chart: np.ndarray
    shapes: tuple

    @property
    def cells(self) -> int:
        return len(self.cell_chart)

    @property
    def nbins(self) -> int:
        return {PAINT: 8, SOIL: 7, WET: 3}[self.params.mode]

    def rest(self) -> np.ndarray:
        """(4 C, 3) vertices before the first render: every cell collapsed to its chart-frame centre."""
        return np.repeat(self.corners.mean(1), 4, axis=0).astype(np.float32)

    def tris(self) -> np.ndarray:
        q = 4 * np.arange(self.cells, dtype=np.int32)[:, None]
        return np.concatenate([q + [0, 1, 2], q + [0, 2, 3]], 1).reshape(-1, 3).astype(np.int32)


@dataclass(frozen=True)
class FilmLayout:
    worlds: int
    objects: tuple


def _host(sim):
    plugins = getattr(sim, "plugins", None)
    return plugins if plugins is not None and not callable(plugins) else None


def film_views(sim) -> list:
    """The ``render_view.FilmView`` of each SurfaceFlow plug-in of ``sim`` (``sim.plugins``), [] without any."""
    host = _host(sim)
    if host is None or not isinstance(getattr(host, "plugins", None), dict):
        return []
    try:
        from alloy.plugins.surface_flow.render_view import film_plugins, film_view  # noqa: PLC0415
    except ImportError:
        return []
    return [film_view(p) for p in film_plugins(host)]


def has_films(sim) -> bool:
    """True for a sim (``alloy.Sim`` or engine simulation) with a SurfaceFlow plug-in (cheap: no device reads)."""
    host = _host(sim)
    plugins = getattr(host, "plugins", None) if host is not None else None
    if not isinstance(plugins, dict):
        return False
    return any(type(p).__name__ == "SurfaceFlow" for p in plugins.values())


def film_objects(sim) -> FilmLayout | None:
    """The films of ``sim`` to draw (None without any). Raises CameraError when the film skins would not fit."""
    host = _host(sim)
    views = film_views(sim)
    if not views:
        return None
    flows = {v.label: p for v, p in zip(views, _flows(host))}
    objs = tuple(_object(v, _LOOKS.get(flows.get(v.label), FilmLook())) for v in views)
    worlds = views[0].worlds
    tris = worlds * sum(2 * o.cells * o.nbins for o in objs)
    if tris > MAX_TRIS:
        raise CameraError(f"the films need {tris:,} triangles over {worlds} worlds (limit {MAX_TRIS:,})",
                          "render fewer worlds at once, or use coarser chart cells")
    log.info("camera: %d films (%s), %d drawn triangles per world", len(objs),
             ", ".join(f"{o.name}: {['paint', 'soil', 'wet'][o.params.mode]}" for o in objs), tris // worlds)
    return FilmLayout(worlds, objs)


def _flows(host) -> list:
    from alloy.plugins.surface_flow.render_view import film_plugins  # noqa: PLC0415
    return film_plugins(host)


def _kind(view, look: FilmLook) -> int:
    if look.kind != "auto":
        return KINDS[look.kind]
    if view.solids >= 0 or "solids" in view.species:
        return PAINT
    return SOIL if view.soil is not None else WET


def _object(view, look: FilmLook) -> FilmObject:
    mode = _kind(view, look)
    solids = view.solids if view.solids >= 0 else (view.species.index("solids") if "solids" in view.species else -1)
    if mode == PAINT and solids < 0:
        raise CameraError(f"film {view.label!r}: a paint look needs a 'solids' species", "use FilmLook(kind='wet')")
    if mode == SOIL and view.soil is None:
        raise CameraError(f"film {view.label!r}: a soil look needs the plug-in's soil stage", "use kind='wet'")
    p = LookParams()
    p.mode, p.wet_depth, p.solids = mode, float(view.wet_depth), int(solids)
    p.phi_m = float(view.phi_m or 0.0)
    if mode == PAINT:
        p.lo, p.ref = float(look.painted_dft), float(look.hide_dft or view.hide_dft or 3.3e-5)
    elif mode == SOIL:
        ref = look.soil_ref or float(np.max(view.soil_start.numpy(), initial=0.0)) or 0.2   # bind-time read
        p.lo, p.ref = float(look.clean_soil), float(ref)
    else:
        p.lo, p.ref = 0.0, 1.0
    corners = np.concatenate([g.corners + OFFSET_M * g.normals[:, None, :] for g in view.charts]).astype(np.float32)
    normals = np.concatenate([g.normals for g in view.charts]).astype(np.float32)
    cell_chart = np.concatenate([np.full(g.nx * g.ny, g.index, np.int32) for g in view.charts])
    return FilmObject(view.label, view, look, p, corners, normals, cell_chart, tuple(g.shape for g in view.charts))


def bin_looks(o: FilmObject, base: list) -> list:
    """[(linear rgb, roughness, name)] of each appearance level of ``o`` over substrate colours ``base`` (per chart:
    the first chart's colour is used for all; films cover one substrate in the tasks)."""
    s = np.asarray(o.look.base if o.look.base is not None else (base[0] if base else DEFAULT_BASE), np.float64)
    mode = o.params.mode
    c = np.asarray(o.look.color if o.look.color is not None else DEFAULT_COLOR.get(mode, (0, 0, 0)), np.float64)
    out = []
    if mode == PAINT:
        for lvl, q in enumerate(PAINT_Q):
            a = 1.0 - (1.0 - HIDE_FRAC) ** q
            rgb = s * (1.0 - a) + c * a
            out += [(tuple(rgb), 0.55, f"paint{lvl}_dry"), (tuple(0.9 * rgb), 0.12, f"paint{lvl}_wet")]
    elif mode == SOIL:
        for lvl, a in enumerate(SOIL_A):
            rgb = s * (1.0 - a) + c * a
            if lvl:
                out.append((tuple(rgb), 0.7, f"soil{lvl}_dry"))
            out.append((tuple(0.72 * rgb), 0.1, f"soil{lvl}_wet"))
    else:
        out = [(tuple(k * s), r, f"wet{lvl}") for lvl, (k, r) in enumerate(zip(WET_DARK, WET_ROUGH))]
    return out


@wp.func
def film_bin(p: LookParams, h: float, dft: float, soil: float) -> int:
    """The appearance level of one cell (-1: not drawn); see :func:`bin_looks` for the order."""
    wet = h > p.wet_depth
    if p.mode == 0:
        if dft <= p.lo:
            return -1
        q = dft / p.ref
        lvl = 3
        if q < 0.25:
            lvl = 0
        elif q < 0.5:
            lvl = 1
        elif q < 1.0:
            lvl = 2
        if wet and (p.phi_m <= 0.0 or dft < 0.999 * p.phi_m * h):   # solvent left (no drying stage: any depth)
            return 2 * lvl + 1
        return 2 * lvl
    if p.mode == 1:
        lvl = 0
        if soil > p.lo:
            lvl = 3
            if soil < 0.1 * p.ref:
                lvl = 1
            elif soil < 0.5 * p.ref:
                lvl = 2
        if lvl == 0:
            if wet:
                return 0
            return -1
        if wet:
            return 2 * lvl
        return 2 * lvl - 1
    if not wet:
        return -1
    if h < 2.0e-4:
        return 0
    if h < 1.0e-3:
        return 1
    return 2


@wp.kernel(enable_backward=False)
def film_skin(depth: wp.array2d(dtype=wp.float64), base: wp.array2d(dtype=wp.float32),
              mass: wp.array3d(dtype=wp.float64),
              soil: wp.array2d(dtype=wp.float64), p: LookParams, cell_chart: wp.array(dtype=wp.int32),
              cbody: wp.array2d(dtype=wp.int32), clocal: wp.array2d(dtype=wp.transform),
              body_q: wp.array(dtype=wp.transform), corners: wp.array2d(dtype=wp.vec3),
              cnrm: wp.array(dtype=wp.vec3), nbins: int,
              out: wp.array3d(dtype=wp.vec3), nrm: wp.array3d(dtype=wp.vec3)):
    """Place cell c of world w in its bin's mesh (its 4 corners in world, on the film's surface: base + depth along
    the cell normal) and collapse it in the others."""
    w, c = wp.tid()
    k = cell_chart[c]
    X = clocal[w, k]
    if cbody[w, k] >= 0:
        X = wp.transform_multiply(body_q[cbody[w, k]], X)
    dft = float(0.0)
    if p.solids >= 0:
        dft = float(mass[w, c, p.solids])
    s = float(0.0)
    if p.mode == 1:
        s = float(soil[w, c])
    h = float(depth[w, c])
    b = film_bin(p, h, dft, s)
    up = cnrm[c] * (base[w, c] + h)
    a = wp.transform_point(X, corners[c, 0] + up)
    e1 = wp.transform_point(X, corners[c, 1] + up)
    e2 = wp.transform_point(X, corners[c, 2] + up)
    e3 = wp.transform_point(X, corners[c, 3] + up)
    mid = 0.25 * (a + e1 + e2 + e3)
    n = wp.normalize(wp.cross(e1 - a, e2 - a))
    for m in range(nbins):
        if m == b:
            out[m, w, 4 * c + 0] = a
            out[m, w, 4 * c + 1] = e1
            out[m, w, 4 * c + 2] = e2
            out[m, w, 4 * c + 3] = e3
        else:
            for q in range(4):
                out[m, w, 4 * c + q] = mid
        for q in range(4):
            nrm[m, w, 4 * c + q] = n


def base_colours(sd, shapes: tuple) -> list:
    """The scene colour of each chart's shape (world 0's instance named like it), DEFAULT_BASE when not found."""
    names = [str(n) for n in getattr(sd, "inst_name", [])]
    out = []
    for s in shapes:
        hit = [i for i, n in enumerate(names) if s and (n == s or n.endswith(f"/{s}") or n.split("#")[0] == s)]
        out.append(tuple(float(v) for v in sd.inst_color[hit[0]]) if hit else DEFAULT_BASE)
    return out


class FilmSet:
    """The films of one camera scene: bin b of film f in world w is scene mesh ``first[f] + b * worlds + w``.
    Same interface as ``deform_gpu.DeformSet`` (scene_data, points, attach, stale, update)."""

    def __init__(self, layout: FilmLayout, first_mesh: int, device: str):
        self.layout, self.device, W = layout, str(device), layout.worlds
        self.first, k = [], int(first_mesh)
        for o in layout.objects:
            self.first.append(k)
            k += o.nbins * W
        dev = self.device
        self.buffers = [wp.array(np.ascontiguousarray(np.broadcast_to(o.rest(), (o.nbins, W, 4 * o.cells, 3))),
                                 dtype=wp.vec3, device=dev) for o in layout.objects]
        self.normals = [wp.zeros((o.nbins, W, 4 * o.cells), dtype=wp.vec3, device=dev) for o in layout.objects]
        self.tris = [wp.array(o.tris(), dtype=wp.int32, device=dev) for o in layout.objects]
        self.corners = [wp.array(o.corners, dtype=wp.vec3, device=dev) for o in layout.objects]
        self.cnrm = [wp.array(o.normals, dtype=wp.vec3, device=dev) for o in layout.objects]
        self.cell_chart = [wp.array(o.cell_chart, dtype=wp.int32, device=dev) for o in layout.objects]
        self.none2 = wp.zeros((1, 1), dtype=wp.float64, device=dev)
        self.none3 = wp.zeros((1, 1, 1), dtype=wp.float64, device=dev)
        self.no_bodies = wp.zeros(1, dtype=wp.transform, device=dev)
        self.gs = self.shade = None

    def mesh_ids(self) -> list[int]:
        return [f + i for f, o in zip(self.first, self.layout.objects) for i in range(o.nbins * self.layout.worlds)]

    def points(self) -> dict:
        """{scene mesh index: its vertex buffer (a row of the film's (bins, worlds, V) buffer)} for GpuScene."""
        W = self.layout.worlds
        return {f + b * W + w: buf[b][w] for f, o, buf in zip(self.first, self.layout.objects, self.buffers)
                for b in range(o.nbins) for w in range(W)}

    def scene_data(self, sd):
        """``sd`` plus each film's bin meshes, one per world, instanced in that world with the bin's colour."""
        from .scene import IDENTITY_XF, MeshGeom, SceneData
        meshes, rows, W = list(sd.meshes), [], self.layout.worlds
        for o in self.layout.objects:
            looks, rest, tris = bin_looks(o, base_colours(sd, o.shapes)), o.rest(), o.tris()
            for b, (rgb, _, tag) in enumerate(looks):
                for w in range(W):
                    rows.append((len(meshes), w, rgb, f"{o.name}:{tag}#{w}" if w else f"{o.name}:{tag}"))
                    meshes.append(MeshGeom(f"{o.name}:{tag}@{w}", rest, tris))
        k, grp, col, names = zip(*rows)
        n = len(rows)
        return SceneData(meshes, np.r_[sd.inst_mesh, k], np.r_[sd.inst_group, grp], np.r_[sd.inst_body, np.full(n, -1)],
                         np.r_[sd.inst_local, np.tile(IDENTITY_XF, (n, 1))], np.r_[sd.inst_color, np.asarray(col)],
                         sd.num_worlds, list(sd.inst_name) + list(names), sd.lighting, sd.camera,
                         sd.textures, np.r_[sd.inst_textured, np.ones(n, bool)])  # scan textures kept

    def attach(self, gs, shade=None) -> None:
        """Bind to the built GpuScene (and a full-material scene's ShadeData, whose corner normals then follow)."""
        self.gs, self.shade = gs, shade
        gs.film = self

    def stale(self) -> bool:
        """Charts never change shape after setup."""
        return False

    def update(self, state) -> None:
        """Draw every film as its plug-in has it now (device reads only) and refit the bin meshes."""
        if self.gs is None:
            return
        from .deform_gpu import corner_normals, mesh_bounds, reset_bounds
        bq = getattr(state, "body_q", None) if state is not None else None
        bq = self.no_bodies if bq is None or not len(bq) else bq
        W, dev = self.layout.worlds, self.device
        for i, o in enumerate(self.layout.objects):
            v = o.view
            ins = [v.depth, v.base, v.species_mass if v.species_mass is not None else self.none3,
                   v.soil if v.soil is not None else self.none2, o.params, self.cell_chart[i], v.cbody, v.clocal, bq,
                   self.corners[i], self.cnrm[i], o.nbins]
            wp.launch(film_skin, dim=(W, o.cells), inputs=ins, outputs=[self.buffers[i], self.normals[i]], device=dev)
            for b in range(o.nbins):
                first = self.first[i] + b * W
                wp.launch(reset_bounds, dim=W, inputs=[first, self.gs.mesh_lo, self.gs.mesh_hi], device=dev)
                wp.launch(mesh_bounds, dim=(W, 4 * o.cells), inputs=[self.buffers[i][b], first, self.gs.mesh_lo,
                                                                      self.gs.mesh_hi], device=dev)
                if self.shade is not None:
                    wp.launch(corner_normals, dim=(W, 2 * o.cells, 3),
                              inputs=[self.normals[i][b], self.buffers[i][b], self.tris[i], self.shade.mesh_tri_off,
                                      first], outputs=[self.shade.tri_n], device=dev)
        for k in self.mesh_ids():
            self.gs.meshes[k].refit()
        if self.gs.optix is not None:
            for f, o in zip(self.first, self.layout.objects):
                self.gs.optix.refit_gas(f, f + o.nbins * W)


def film_pack_meshes(layout: FilmLayout):
    """(meshes, {material: pack material}, {texture key: entry}, present (len(meshes), worlds) bool) of the films for
    a full-material pack (``model_pack._add_dynamic``): bin b of film f in world w, present in world w only."""
    from .looks import material_tree
    from .pbr.pack_io import Mesh
    meshes, specs, W, eye = [], {}, layout.worlds, np.eye(4).tolist()
    for o in layout.objects:
        rest, tris = o.rest(), o.tris()
        n = np.broadcast_to(np.array([0.0, 0.0, 1.0], np.float32), (len(tris), 3, 3)).copy()
        for rgb, rough, tag in bin_looks(o, []):
            mat = f"film_{o.name}_{tag}"
            specs[mat] = {"preset": "plastic_matte", "base_color": [float(c) for c in rgb], "roughness": rough,
                          "hard": False}
            for w in range(W):
                meta = {"name": f"{o.name}:{tag}@{w}", "object_name": f"{o.name}:{tag}", "materials": [mat],
                        "object_to_world": eye, "dynamic": True, "kind": "film"}
                meshes.append(Mesh(meta, rest, tris, n, np.zeros(len(tris), np.int32)))
    present = np.zeros((len(meshes), W), bool)
    present[np.arange(len(meshes)), np.arange(len(meshes)) % W] = True
    mats, tex = {}, {}
    for name, spec in specs.items():
        mats[name] = material_tree(name, spec)
        tex.update(mats[name].pop("textures", {}))
    return meshes, mats, tex, present


def add_to_pack(pk, frame, binding: tuple, layout: FilmLayout, device: str) -> tuple:
    """Append the films' bin meshes and materials to a full-material pack (after its other meshes, as model_pack's
    dynamic meshes); returns (the extended binding, the FilmSet of those meshes)."""
    from .scene import IDENTITY_XF
    extra, mats, tex, present = film_pack_meshes(layout)
    first = len(frame.meshes)
    frame.meshes.extend(extra)
    st = pk.static
    st["materials"].update({k: v for k, v in mats.items() if k not in st["materials"]})
    st["textures"].update(tex)
    st["materials_used_by_meshes"] = sorted({m.meta["materials"][0] for m in frame.meshes})
    n, w = present.shape
    binding = (np.concatenate([binding[0], np.full((n, w), -1, np.int32)]),
               np.concatenate([binding[1], np.tile(IDENTITY_XF, (n, w, 1)).astype(np.float32)]),
               np.concatenate([binding[2] if len(binding) > 2 else np.ones(binding[0].shape, bool), present]))
    return binding, FilmSet(layout, first, device)


class Dynamic:
    """Dynamic sets passed together where one is expected (a PbrScene's ``deform``): one points dict, one attach."""

    def __init__(self, *sets):
        self.sets = [s for s in sets if s is not None]

    def points(self) -> dict:
        return {k: v for s in self.sets for k, v in s.points().items()}

    def attach(self, gs, shade=None) -> None:
        for s in self.sets:
            s.attach(gs, shade)


__all__ = ["Dynamic", "FilmLayout", "FilmLook", "FilmObject", "FilmSet", "add_to_pack", "base_colours", "bin_looks",
           "film_objects", "film_pack_meshes", "film_views", "has_films", "set_film_look"]
