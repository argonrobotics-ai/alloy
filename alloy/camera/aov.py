# SPDX-License-Identifier: Apache-2.0
"""Depth, id and optical-flow images from the camera's primary ray (opt-in; the colour render is not touched).

One ray per pixel, through the pixel centre, against the camera scene's own Warp BVH, in a kernel of its own that is
launched only when one of these outputs is asked for. So it is exact and repeatable (no jitter: independent of ``spp``
and ``seed``), the colour kernels and their images stay bit for bit, and an output without a colour channel costs no
path tracing at all.

Example::

    cam = alloy.Camera("front", look_at=((0.5, -0.5, 0.4), (0, 0, 0.05)), width=64, height=64, spp=4)
    depth = cam.render(sim, output="depth")                          # wp float32 (worlds, 64, 64), metres
    out = cam.render(sim, output=("rgb", "depth", "segmentation"))   # CameraOutput: .rgb .depth .segmentation
    names = segment_labels(sim)                                      # names[id] for each segmentation id

Channels (each ``(worlds, H, W)``, row 0 = top, the camera's own buffers: the next render overwrites them):

- ``depth`` float32: distance in metres from the camera along its optical axis (not along the ray) to the first
  surface; ``inf`` where the ray meets nothing between the camera's clip distances.
- ``segmentation`` int32: 0 nothing hit, 1 a surface that is not on a body (ground, static objects, cloth, liquid),
  2 + k the k-th body of its world, so one id means the same body in every world (:func:`segment_labels`).
- ``body_id`` int32: the body's index into ``state.body_q``; -1 a surface not on a body, -2 nothing hit.
- ``world_id`` int32: the world of the surface; -2 nothing hit; -1 an instance the model shares between worlds
  (a scene built with ``alloy.Scene`` has none: each world has its own ground and static objects, which read that
  world's index; use ``segmentation == 1`` or ``body_id == -1`` for surfaces that are not on a body).

- ``instance_id`` int32: 0 nothing hit, 1 + k the k-th drawn shape of its world (:func:`instance_labels`), so two
  static objects, or two shapes of one body, read different ids, the same in every world.
- ``semantic`` int32: 0 nothing hit, 1 a shape with no class, 2 + k the k-th class given to :func:`set_semantics`
  (which must be called first).
- ``flow`` float32 ``(worlds, H, W, 2)``: (dx, dy) in pixels, x to the right and y down: the surface point at pixel
  (col, row) was at (col - dx, row - dy) in the previous render of this camera that asked for ``flow`` (the first
  one reads 0). It follows rigid bodies (robot links, kinematic objects), static shapes and the camera's own
  motion; NaN where nothing is hit, on cloth, soft bodies, films and liquid (their motion is not tracked), and
  for a point that was behind the camera.

Covered: the fast camera (model and ``SceneData`` scenes, v1 packs) on the Warp and OptiX backends (this pass always
traces the Warp BVH). Not covered: full-material scenes (``quality="full"``, looks, v2 packs), and a live liquid on
the OptiX backend (its surface exists only in OptiX there: use ``backend="warp"``); both raise ``CameraError``. Every
surface counts as opaque (glass and liquid surfaces are hit, not seen through), and a pixel
takes the surface at its centre: no blending along edges.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass

import numpy as np
import warp as wp

from .errors import CameraError
from .tracer.geometry import Geo, closest_hit
from .tracer.integrator import Cams, Params, camera_ray

AOVS = ("depth", "segmentation", "body_id", "world_id", "instance_id", "semantic", "flow")
COLOR = ("rgb", "linear", "albedo", "normal")
NO_BODY, NO_HIT = -1, -2  # body_id / world_id values that are not an index
SEG_BACKGROUND, SEG_NO_BODY, SEG_FIRST_BODY = 0, 1, 2
SEM_UNLABELED, SEM_FIRST_CLASS = 1, 2
_DTYPES = {name: wp.int32 for name in AOVS} | {"depth": wp.float32, "flow": wp.vec2}
_BIT = {name: 1 << k for k, name in enumerate(AOVS)}
_TABLES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # GpuScene -> IdTables
_SEMANTICS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # model or SceneData -> Semantics


@wp.struct
class Ids:
    """Per instance, in the device scene's instance order."""

    body: wp.array(dtype=wp.int32)
    world: wp.array(dtype=wp.int32)
    seg: wp.array(dtype=wp.int32)
    instance: wp.array(dtype=wp.int32)
    semantic: wp.array(dtype=wp.int32)
    rigid: wp.array(dtype=wp.int32)  # 1: the instance moves as one piece (flow is defined), 0: a deforming mesh


@wp.func
def pixel_of(cams: Cams, img: wp.int32, p: wp.vec3, prm: Params) -> wp.vec2:
    """Image position (column, row; row 0 = top, pixel centres at .5) of world point ``p``; NaN behind the camera."""
    r = cams.rot[img]
    v = p - cams.eye[img]
    z = -(r[0, 2] * v[0] + r[1, 2] * v[1] + r[2, 2] * v[2])
    if z <= 1.0e-9:
        return wp.vec2(wp.nan, wp.nan)
    ext = cams.extent[img]
    x = (r[0, 0] * v[0] + r[1, 0] * v[1] + r[2, 0] * v[2]) / (z * ext[0])
    y = (r[0, 1] * v[0] + r[1, 1] * v[1] + r[2, 1] * v[2]) / (z * ext[1])
    return wp.vec2(0.5 * (x + 1.0) * float(prm.width), 0.5 * (1.0 - y) * float(prm.height))


@wp.kernel
def aov_kernel(geo: Geo, cams: Cams, prm: Params, ids: Ids, prev: Cams, prev_xf: wp.array(dtype=wp.transform),
               want: wp.int32, miss_depth: float, out_depth: wp.array(dtype=wp.float32),
               out_seg: wp.array(dtype=wp.int32), out_body: wp.array(dtype=wp.int32),
               out_world: wp.array(dtype=wp.int32), out_instance: wp.array(dtype=wp.int32),
               out_semantic: wp.array(dtype=wp.int32), out_flow: wp.array(dtype=wp.vec2)):
    """First surface on the pixel-centre ray of every pixel; writes the channels whose bit is set in ``want``."""
    tid = wp.tid()
    npix = prm.width * prm.height
    img = tid // npix
    pix = tid - img * npix
    row = pix // prm.width
    px = pix - row * prm.width
    pyr = prm.height - 1 - row  # raster y counts from the bottom row, as in the colour kernel
    o, d, tmax = camera_ray(cams, img, float(px) + 0.5, float(pyr) + 0.5, prm)
    inst, t, _n, _face = closest_hit(geo, cams.world[img], o, d, tmax)
    hit = inst >= 0
    inst = wp.max(inst, 0)
    if (want & 1) != 0:
        r = cams.rot[img]
        back = wp.vec3(r[0, 2], r[1, 2], r[2, 2])  # the camera looks along -z
        out_depth[tid] = wp.where(hit, -wp.dot(o + d * t - cams.eye[img], back), miss_depth)
    if (want & 2) != 0:
        out_seg[tid] = wp.where(hit, ids.seg[inst], 0)
    if (want & 4) != 0:
        out_body[tid] = wp.where(hit, ids.body[inst], -2)
    if (want & 8) != 0:
        out_world[tid] = wp.where(hit, ids.world[inst], -2)
    if (want & 16) != 0:
        out_instance[tid] = wp.where(hit, ids.instance[inst], 0)
    if (want & 32) != 0:
        out_semantic[tid] = wp.where(hit, ids.semantic[inst], 0)
    if (want & 64) != 0:
        flow = wp.vec2(wp.nan, wp.nan)
        if hit and ids.rigid[inst] != 0:
            local = wp.transform_point(geo.inst_inv[inst], o + d * t)
            before = pixel_of(prev, img, wp.transform_point(prev_xf[inst], local), prm)
            flow = wp.vec2(float(px) + 0.5, float(row) + 0.5) - before
        out_flow[tid] = flow


@dataclass(frozen=True)
class Request:
    """What one ``render(output=...)`` asks for: the depth / id channels, whether colour is traced too, and whether
    the answer is one array (a channel name) or a ``CameraOutput`` (a tuple of names)."""

    aovs: tuple[str, ...]
    color: bool
    single: bool
    names: tuple[str, ...] = ()  # a tuple output: the fields the CameraOutput keeps


def parse_output(output, legacy: tuple[str, ...]) -> Request | None:
    """None for the colour-only outputs in ``legacy`` (the camera's unchanged path); else the :class:`Request`.
    Raises CameraError for an unknown name. Example: ``parse_output(("rgb", "depth"), OUTPUTS).aovs == ("depth",)``."""
    if isinstance(output, str):
        if output in legacy:
            return None
        if output in AOVS:
            return Request((output,), False, True)
        raise CameraError(f"output {output!r}", f"use one of {legacy + AOVS}, or a tuple such as ('rgb', 'depth')")
    names = tuple(output)
    bad = [n for n in names if n not in AOVS + COLOR]
    if bad or not names:
        raise CameraError(f"output {output!r}", f"a tuple takes names from {COLOR + AOVS}")
    return Request(tuple(n for n in AOVS if n in names), any(n in COLOR for n in names), False, names)


def keep_fields(out, names: tuple[str, ...]):
    """``out`` (a CameraOutput) with every field that ``names`` does not ask for set to None."""
    for field in vars(out):
        if field not in names:
            setattr(out, field, None)
    return out


def body_local_index(body_world: np.ndarray) -> np.ndarray:
    """Per body: its rank among the bodies of its own world (bodies shared by every world, world -1, come after the
    largest world's). Example: ``body_local_index(np.array([0, 0, 1, 1]))`` is ``[0, 1, 0, 1]``."""
    body_world = np.asarray(body_world, np.int64).reshape(-1)
    local = np.zeros(len(body_world), np.int32)
    per_world = 0
    for w in np.unique(body_world[body_world >= 0]):
        idx = np.flatnonzero(body_world == w)
        local[idx] = np.arange(len(idx))
        per_world = max(per_world, len(idx))
    shared = np.flatnonzero(body_world < 0)
    local[shared] = per_world + np.arange(len(shared))
    return local


def _scene_model(sc):
    """The newton.Model a device scene was built from (None for SceneData and pack scenes)."""
    from .camera import _SCENES
    return next((m for m, per in _SCENES.items() if any(v is sc for v in per.values())), None)


def _body_world(sc) -> np.ndarray:
    """World of every body: the model's ``body_world``, else the world of the instances drawn on it."""
    model = _scene_model(sc)
    bw = getattr(model, "body_world", None) if model is not None else None
    if bw is not None:
        return np.asarray(bw.numpy(), np.int64)
    body, group = sc.sd.inst_body, sc.sd.inst_group
    out = np.full(int(body.max()) + 1 if len(body) and body.max() >= 0 else 0, -1, np.int64)
    out[body[body >= 0]] = group[body >= 0]
    return out


def group_rank(group: np.ndarray) -> np.ndarray:
    """Per instance: its rank among the instances of its own world, in scene order (shared instances, world -1,
    come after the largest world's). Example: ``group_rank(np.array([0, 0, 1, 1, -1]))`` is ``[0, 1, 0, 1, 2]``."""
    return body_local_index(group)


@dataclass
class Semantics:
    """The classes of one source: glob patterns in the caller's order, each with its class index, and the Alloy
    names of every body (by its rank in its world) that a pattern may match besides the shape's own label."""

    patterns: list[tuple[str, int]]
    labels: list[str]
    body_names: list[list[str]]

    def ids(self, sd, local_body: np.ndarray) -> np.ndarray:
        """Semantic id of every instance of ``sd``: the first pattern that matches its label or its body's names."""
        from fnmatch import fnmatchcase
        out = np.full(sd.num_instances, SEM_UNLABELED, np.int32)
        for i, (name, body) in enumerate(zip(sd.inst_name, sd.inst_body)):
            rank = local_body[body] if 0 <= body < len(local_body) else -1
            names = [name] + (self.body_names[rank] if 0 <= rank < len(self.body_names) else [])
            hit = next((c for pat, c in self.patterns if any(fnmatchcase(n, pat) for n in names)), None)
            if hit is not None:
                out[i] = SEM_FIRST_CLASS + hit
        return out


class IdTables:
    """Per-instance ids of a device scene, in its instance order (:class:`Ids`); ``semantic`` follows the last
    :func:`set_semantics` of the scene's source."""

    def __init__(self, sc):
        order, dev = sc.order, sc.device
        body = np.asarray(sc.sd.inst_body, np.int32)[order]
        world = np.where(sc.inst_group >= sc.num_worlds, -1, sc.inst_group).astype(np.int32)
        self.local = body_local_index(_body_world(sc))
        seg = np.full(len(body), SEG_NO_BODY, np.int32)
        on_body = body >= 0
        seg[on_body] = SEG_FIRST_BODY + self.local[body[on_body]]
        moving = list(getattr(sc, "dynamic", ())) + list(getattr(sc, "rebuilt", {}))
        rigid = ~np.isin(np.asarray(sc.sd.inst_mesh)[order], moving)
        self.ids, self._sem = Ids(), None
        self.ids.body = wp.array(body, dtype=wp.int32, device=dev)
        self.ids.world = wp.array(world, dtype=wp.int32, device=dev)
        self.ids.seg = wp.array(seg, dtype=wp.int32, device=dev)
        self.ids.instance = wp.array(1 + group_rank(sc.sd.inst_group)[order], dtype=wp.int32, device=dev)
        self.ids.semantic = wp.zeros(len(body), dtype=wp.int32, device=dev)
        self.ids.rigid = wp.array(rigid.astype(np.int32), dtype=wp.int32, device=dev)

    def with_semantics(self, sc) -> Ids:
        """The ids with the semantic column of the source's current classes; CameraError when none were set."""
        sem = _SEMANTICS.get(_source_key(sc))
        if sem is None:
            raise CameraError("the 'semantic' output needs classes",
                              "call alloy.camera.aov.set_semantics(sim, {'<name or pattern>': '<class>', ...}) first")
        if sem is not self._sem:
            self.ids.semantic = wp.array(sem.ids(sc.sd, self.local)[sc.order], dtype=wp.int32, device=sc.device)
            self._sem = sem
        return self.ids


class AovPass:
    """The depth / id / flow pass of one renderer (its scene, cameras and image size); buffers are made on first
    use. For ``flow`` it keeps the instance poses and cameras of the previous render that asked for it. A scene
    that is rebuilt (a cloth cut or torn) gets a new renderer and pass, so its next flow render reads 0 again
    ("0" is float32 round-off, about 1e-5 px, not an exact zero)."""

    def __init__(self, renderer):
        self._r = weakref.ref(renderer)  # the renderer keeps this pass: no strong reference back
        self.sc = getattr(renderer, "sc", None)
        if not hasattr(self.sc, "geo") or not hasattr(getattr(renderer, "cams", None), "extent"):  # not a tracer.Renderer
            raise CameraError("depth, id and flow images are not available on the full-material camera yet",
                              "render them from the fast camera: no look, quality='fast', a model or SceneData scene")
        self.prm = Params()
        self.prm.width, self.prm.height = int(renderer.w), int(renderer.h)
        self.bufs: dict[str, wp.array] = {}
        self._dummy = {t: wp.zeros(1, dtype=t, device=self.sc.device) for t in set(_DTYPES.values())}
        self._prev: tuple[Cams, wp.array] | None = None  # cameras and instance poses of the last flow render

    def _tables(self) -> IdTables:
        if self.sc not in _TABLES:
            _TABLES[self.sc] = IdTables(self.sc)
        return _TABLES[self.sc]

    def _buffer(self, name: str, want: bool) -> wp.array:
        if not want:
            return self._dummy[_DTYPES[name]]
        if name not in self.bufs:
            r = self._r()
            n = r.n_img * r.w * r.h
            self.bufs[name] = wp.zeros(n, dtype=_DTYPES[name], device=self.sc.device)
        return self.bufs[name]

    def _previous(self, cams: Cams) -> tuple[Cams, wp.array]:
        """The cameras and instance poses of the last flow render (the current ones before the first)."""
        prev = self._prev
        stale = prev is None or prev[0].eye.shape != cams.eye.shape or prev[1].shape != self.sc.xf.shape
        if stale:
            prev = Cams()
            for field in ("world", "eye", "rot", "extent", "clip"):
                setattr(prev, field, wp.clone(getattr(cams, field)))
            self._prev = (prev, wp.clone(self.sc.xf))
        return self._prev

    def _remember(self, cams: Cams) -> None:
        prev, xf = self._prev
        for field in ("world", "eye", "rot", "extent", "clip"):
            wp.copy(getattr(prev, field), getattr(cams, field))
        wp.copy(xf, self.sc.xf)

    def render(self, names: tuple[str, ...]) -> dict[str, wp.array]:
        """The channels ``names`` at the scene's current poses, each ``(images, H, W)`` (``flow``: one more axis
        of 2). Example: ``AovPass(renderer).render(("depth",))["depth"]``."""
        r, tab = self._r(), self._tables()
        if getattr(self.sc, "liquid", None) is not None and getattr(self.sc, "optix", None) is not None:
            raise CameraError("depth, id and flow images of a live liquid are not available on the OptiX backend",
                              "use the Warp backend for this camera: Camera(..., backend='warp')")
        ids = tab.with_semantics(self.sc) if "semantic" in names else tab.ids
        prev, prev_xf = self._previous(r.cams) if "flow" in names else (r.cams, self.sc.xf)
        want = sum(_BIT[n] for n in names)
        outs = [self._buffer(n, n in names) for n in AOVS]
        wp.launch(aov_kernel, dim=r.n_img * r.w * r.h,
                  inputs=[self.sc.geo, r.cams, self.prm, ids, prev, prev_xf, want, float("inf")],
                  outputs=outs, device=self.sc.device)
        if "flow" in names:
            self._remember(r.cams)
        return {n: self.bufs[n].reshape((r.n_img, r.h, r.w)) for n in names}


def render_aovs(renderer, names: tuple[str, ...]) -> dict[str, wp.array]:
    """Depth / id channels of ``renderer``'s current cameras and scene poses (the pass lives on the renderer)."""
    aov = getattr(renderer, "_aov", None)
    if aov is None:
        aov = renderer._aov = AovPass(renderer)
    return aov.render(names)


def as_host(out):
    """A render result as numpy: an array, or a dict of the fields a ``CameraOutput`` carries."""
    if isinstance(out, wp.array):
        return out.numpy()
    return {k: v.numpy() for k, v in vars(out).items() if v is not None}


def as_torch(out) -> dict:
    """The fields a ``CameraOutput`` carries as torch tensors on its device (needs PyTorch)."""
    return {k: wp.to_torch(v) for k, v in vars(out).items() if v is not None}


def segment_labels(source) -> list[str]:
    """Name of every ``segmentation`` id of ``source`` (an ``alloy.Sim``, a newton.Model or a ``SceneData``):
    ``["background", "no_body", <body 0 of a world>, ...]``. An ``alloy.Sim`` gives its own body names
    (``sim.names()["bodies"]``, robot parts as ``"<robot>/<body>"``); a model its body labels of world 0.

    Example: ``segment_labels(sim)[int(seg[0, 32, 32])]`` is the name of what world 0's centre pixel shows."""
    from .camera import unwrap_sim
    model = unwrap_sim(source)[0]
    labels = ["background", "no_body"]
    bw = getattr(model, "body_world", None)
    if bw is None:  # a SceneData: bodies have no names
        body = np.asarray(getattr(model, "inst_body", np.zeros(0, np.int32)))
        group = np.asarray(getattr(model, "inst_group", np.zeros(0, np.int32)))
        world = np.full(int(body.max()) + 1 if len(body) and body.max() >= 0 else 0, -1, np.int64)
        world[body[body >= 0]] = group[body >= 0]
        n = int(body_local_index(world).max()) + 1 if len(world) else 0
        return labels + [f"body_{k}" for k in range(n)]
    local = body_local_index(bw.numpy())
    names = [""] * (int(local.max()) + 1 if len(local) else 0)
    raw = [str(x) for x in (getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])]
    for b in range(len(local) - 1, -1, -1):  # the lowest body index (world 0's copy) names the id
        names[local[b]] = raw[b] if b < len(raw) else f"body_{local[b]}"
    index = getattr(source, "index", None)
    for name, b in sorted(getattr(index, "bodies", {}).items(), key=lambda kv: len(kv[0])):
        names[local[b]] = name  # the longest Alloy name wins: "arm/link1" over "arm"
    return labels + names



def _source_key(sc):
    """What :func:`set_semantics` files a device scene's classes under: its model, else its SceneData."""
    model = _scene_model(sc)
    return sc.sd if model is None else model


def _fast_scene_data(source):
    """The SceneData the fast camera draws for ``source`` (a scene that already exists, else a new one)."""
    from .camera import _SCENES, camera_scene, unwrap_sim
    model = unwrap_sim(source)[0]
    if hasattr(model, "inst_name"):
        return model
    built = [v for v in _SCENES.get(model, {}).values() if hasattr(v, "geo")]
    return (built[0] if built else camera_scene(source, quality="fast")).sd


def instance_labels(source) -> list[str]:
    """Name of every ``instance_id`` of ``source`` (an ``alloy.Sim``, a newton.Model or a ``SceneData``):
    ``["background", <shape 0 of a world>, ...]``, the shapes' own labels (``"ground"``, ``"table"``, ...).

    Example: ``instance_labels(sim)[int(inst[0, 32, 32])]`` is the shape world 0's centre pixel shows."""
    sd = _fast_scene_data(source)
    rank = group_rank(sd.inst_group)
    names = [""] * (int(rank.max()) + 1 if len(rank) else 0)
    for i in range(sd.num_instances - 1, -1, -1):  # the first instance (world 0's copy) names the id
        names[rank[i]] = str(sd.inst_name[i])
    return ["background"] + names


def _body_names(source) -> list[list[str]]:
    """Every name of each body, by its rank in its world: the model's label and the Alloy names of an ``alloy.Sim``."""
    from .camera import unwrap_sim
    model = unwrap_sim(source)[0]
    bw = getattr(model, "body_world", None)
    if bw is None:
        return []
    local = body_local_index(bw.numpy())
    names: list[list[str]] = [[] for _ in range(int(local.max()) + 1 if len(local) else 0)]
    raw = [str(x) for x in (getattr(model, "body_label", None) or getattr(model, "body_key", None) or [])]
    for b in range(min(len(raw), len(local))):
        names[local[b]].append(raw[b])
    for name, b in getattr(getattr(source, "index", None), "bodies", {}).items():
        names[local[b]].append(name)
    return names


def set_semantics(source, classes: dict[str, str]) -> list[str]:
    """Give the ``semantic`` output of ``source`` (an ``alloy.Sim``, a newton.Model or a ``SceneData``) its
    classes, and return the name of every id: ``["background", "unlabeled", <class>, ...]``, classes in the order
    they first appear. A key is a name or a ``fnmatch`` pattern (``*``, ``?``, case-sensitive); it matches a shape by
    its own label (:func:`instance_labels`) or by a name of the body it is on (``sim.names()["bodies"]``), and the
    first key that matches wins. Calling it again replaces the classes for every camera of that source.

    Example: ``set_semantics(sim, {"block": "cube", "arm/*": "robot", "bin_*": "container"})``."""
    from .camera import unwrap_sim
    labels: list[str] = []
    patterns = []
    for pattern, name in classes.items():
        if not isinstance(pattern, str) or not isinstance(name, str) or not pattern or not name:
            raise CameraError(f"semantic class {pattern!r}: {name!r}", "map name patterns to class names, both str")
        if name not in labels:
            labels.append(name)
        patterns.append((pattern, labels.index(name)))
    _SEMANTICS[unwrap_sim(source)[0]] = Semantics(patterns, labels, _body_names(source))
    return ["background", "unlabeled"] + labels


def semantic_labels(source) -> list[str]:
    """Name of every ``semantic`` id of ``source``, as the last :func:`set_semantics` returned it."""
    from .camera import unwrap_sim
    sem = _SEMANTICS.get(unwrap_sim(source)[0])
    return ["background", "unlabeled"] + (list(sem.labels) if sem is not None else [])
