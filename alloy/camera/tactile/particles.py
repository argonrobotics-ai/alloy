# SPDX-License-Identifier: Apache-2.0
"""How deep deformables press into the gel: cloth and soft-body surfaces, MPM grains and liquid particles.

The rigid press map (``alloy.camera.tactile.depth``) traces collision shapes. Deformables live as particles in
``state.particle_q``; this module adds them to the same press map, in the same sensor frame and with the same rule
(the first surface along a ray from ``max_depth_mm`` behind the gel, measured from the undeformed gel surface):
- cloth and soft bodies: their surface triangles, thickened by the particles' radius (a cloth's half thickness);
- MPM grains, paste and liquid, and SPH liquid: one sphere of the particle's radius per particle.
With an ``alloy.Sim`` the objects come from the camera's layouts (``alloy.camera.deform.deform_objects``,
``alloy.camera.liquid_scene.liquid_objects``); with a bare Newton model, from ``model.tri_indices`` (triangles) and
every other active particle (spheres).

Each triangle or particle is one thread that writes its footprint with ``wp.atomic_max`` (order-independent), into a
buffer that is then merged with the rigid map; where a deformable presses deeper, the pixel's touched shape is -1
(the stick-slip drag follows rigid bodies only).

Example::

    pp = ParticlePress(model, num_images=1, height=60, width=80, mm_per_px=0.2, max_depth_mm=4.0, sim=sim)
    press = press_map.compute(state.body_q)
    pp.add(state, press_map.gel_xf, press, press_map.touched)     # cloth / grains / liquid now press too
"""
from __future__ import annotations

import logging

import numpy as np
import warp as wp

log = logging.getLogger(__name__)
ACTIVE = wp.constant(1)  # newton.ParticleFlags.ACTIVE


@wp.func
def ray_press(z_low: float, z_high: float, x: float, y: float, max_depth: float, gel_r: float) -> float:
    """Press [mm] of a surface spanning z_low..z_high on the ray at (x, y): 0 behind the ray start or off the gel,
    a full ``max_depth`` when the ray starts inside it."""
    sag = float(0.0)
    if gel_r > 0.0:
        sag = gel_r - wp.sqrt(wp.max(gel_r * gel_r - x * x - y * y, 0.0))
    zg = -sag
    z0 = zg - max_depth
    if z_high < z0:
        return 0.0
    if z_low <= z0:
        return max_depth
    return wp.clamp(zg - z_low, 0.0, max_depth)


@wp.func
def pixel_xy(r: int, c: int, h: int, w: int, mm: float) -> wp.vec2:
    return wp.vec2((float(c) + 0.5 - 0.5 * float(w)) * mm, (float(r) + 0.5 - 0.5 * float(h)) * mm)


@wp.func
def gel_mm(gel_xf: wp.transform, p: wp.vec3) -> wp.vec3:
    """A world point [m] in the gel frame [mm]."""
    return wp.transform_point(wp.transform_inverse(gel_xf), p) * 1000.0


@wp.func
def image_range(w: int, world_image: wp.array(dtype=wp.int32), n_images: int) -> wp.vec2i:
    """Images a particle of world w presses: its world's image, or every image for a shared particle (w < 0)."""
    if w < 0:
        return wp.vec2i(0, n_images)
    if w >= world_image.shape[0]:
        return wp.vec2i(0, 0)
    i = world_image[w]
    if i < 0:
        return wp.vec2i(0, 0)
    return wp.vec2i(i, i + 1)


@wp.kernel
def splat_spheres(pq: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float), flags: wp.array(dtype=wp.int32),
                  ids: wp.array(dtype=wp.int32), pworld: wp.array(dtype=wp.int32),
                  world_image: wp.array(dtype=wp.int32), gel_xf: wp.array(dtype=wp.transform), mm: float,
                  max_depth: float, gel_r: float, out: wp.array3d(dtype=float)):
    """One particle: its sphere's press on every pixel of its footprint (order-independent max)."""
    p = ids[wp.tid()]
    if (flags[p] & ACTIVE) == 0:
        return
    span = image_range(pworld[p], world_image, out.shape[0])
    for i in range(span[0], span[1]):
        sphere_pixels(gel_mm(gel_xf[i], pq[p]), radius[p] * 1000.0, i, mm, max_depth, gel_r, out)


@wp.func
def sphere_pixels(c: wp.vec3, rad: float, i: int, mm: float, max_depth: float, gel_r: float,
                  out: wp.array3d(dtype=float)):
    if c[2] - rad > 0.0 or c[2] + rad < -max_depth - gel_r:
        return
    h = out.shape[1]
    wd = out.shape[2]
    c0 = wp.max(int(wp.floor((c[0] - rad) / mm + 0.5 * float(wd))), 0)
    c1 = wp.min(int(wp.ceil((c[0] + rad) / mm + 0.5 * float(wd))), wd - 1)
    r0 = wp.max(int(wp.floor((c[1] - rad) / mm + 0.5 * float(h))), 0)
    r1 = wp.min(int(wp.ceil((c[1] + rad) / mm + 0.5 * float(h))), h - 1)
    for r in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            xy = pixel_xy(r, col, h, wd, mm)
            d2 = (xy[0] - c[0]) * (xy[0] - c[0]) + (xy[1] - c[1]) * (xy[1] - c[1])
            if d2 < rad * rad:
                dz = wp.sqrt(rad * rad - d2)
                v = ray_press(c[2] - dz, c[2] + dz, xy[0], xy[1], max_depth, gel_r)
                if v > 0.0:
                    wp.atomic_max(out, i, r, col, v)


@wp.kernel
def splat_tris(pq: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float), tris: wp.array2d(dtype=wp.int32),
               pworld: wp.array(dtype=wp.int32), world_image: wp.array(dtype=wp.int32),
               gel_xf: wp.array(dtype=wp.transform), mm: float, max_depth: float, gel_r: float,
               out: wp.array3d(dtype=float)):
    """One surface triangle, thickened by its vertices' mean radius: its press on the pixels it covers."""
    k = wp.tid()
    ia = tris[k, 0]
    ib = tris[k, 1]
    ic = tris[k, 2]
    t = (radius[ia] + radius[ib] + radius[ic]) * (1000.0 / 3.0)
    span = image_range(pworld[ia], world_image, out.shape[0])
    for i in range(span[0], span[1]):
        xf = gel_xf[i]
        a = gel_mm(xf, pq[ia])
        b = gel_mm(xf, pq[ib])
        c = gel_mm(xf, pq[ic])
        area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if wp.min(wp.min(a[2], b[2]), c[2]) - t <= 0.0 and wp.abs(area) >= 1.0e-9:
            tri_pixels(a, b, c, t, area, i, mm, max_depth, gel_r, out)


@wp.func
def tri_pixels(a: wp.vec3, b: wp.vec3, c: wp.vec3, t: float, area: float, i: int, mm: float, max_depth: float,
               gel_r: float, out: wp.array3d(dtype=float)):
    h = out.shape[1]
    wd = out.shape[2]
    c0 = wp.max(int(wp.floor(wp.min(wp.min(a[0], b[0]), c[0]) / mm + 0.5 * float(wd))), 0)
    c1 = wp.min(int(wp.ceil(wp.max(wp.max(a[0], b[0]), c[0]) / mm + 0.5 * float(wd))), wd - 1)
    r0 = wp.max(int(wp.floor(wp.min(wp.min(a[1], b[1]), c[1]) / mm + 0.5 * float(h))), 0)
    r1 = wp.min(int(wp.ceil(wp.max(wp.max(a[1], b[1]), c[1]) / mm + 0.5 * float(h))), h - 1)
    for r in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            xy = pixel_xy(r, col, h, wd, mm)
            wa = ((b[0] - xy[0]) * (c[1] - xy[1]) - (c[0] - xy[0]) * (b[1] - xy[1])) / area
            wb = ((c[0] - xy[0]) * (a[1] - xy[1]) - (a[0] - xy[0]) * (c[1] - xy[1])) / area
            wc = 1.0 - wa - wb
            if wa >= 0.0 and wb >= 0.0 and wc >= 0.0:
                z = wa * a[2] + wb * b[2] + wc * c[2]
                v = ray_press(z - t, z + t, xy[0], xy[1], max_depth, gel_r)
                if v > 0.0:
                    wp.atomic_max(out, i, r, col, v)


@wp.kernel
def merge_press(soft: wp.array3d(dtype=float), press: wp.array3d(dtype=float), touched: wp.array3d(dtype=wp.int32)):
    """Rigid and deformable press: the deeper one; a deformable-led pixel has no touched shape."""
    i, r, c = wp.tid()
    v = soft[i, r, c]
    if v > press[i, r, c]:
        press[i, r, c] = v
        touched[i, r, c] = -1


def _sim_layout(sim, n_worlds: int) -> tuple[np.ndarray, np.ndarray]:
    """(triangles (T, 3), sphere particle ids) of an ``alloy.Sim``'s deformables in every world (global indices)."""
    from ..deform import deform_objects
    from ..liquid_scene import liquid_objects
    tris, spheres = [], []
    layout = deform_objects(sim)
    for o in (layout.objects if layout is not None else ()):
        for w in range(layout.worlds):
            base = o.start + w * layout.stride
            if o.kind == "surface":
                tris.append(np.asarray(o.tris, np.int64) + base)
            else:
                spheres.append(np.arange(base, base + o.count))
    liquids = liquid_objects(sim)
    for o in (liquids.objects if liquids is not None else ()):
        for w in range(liquids.worlds):
            base = o.start + w * liquids.stride
            spheres.append(np.arange(base, base + o.count))
    return _stack(tris, (0, 3)), _stack(spheres, (0,))


def _model_layout(model) -> tuple[np.ndarray, np.ndarray]:
    """A bare Newton model: its triangles, and every particle no triangle uses as a sphere."""
    n = int(model.particle_count)
    tris = model.tri_indices.numpy().reshape(-1, 3) if getattr(model, "tri_indices", None) is not None else None
    tris = np.zeros((0, 3), np.int64) if tris is None else tris.astype(np.int64)
    used = np.zeros(n, bool)
    used[tris.reshape(-1)] = True
    return tris, np.flatnonzero(~used)


def _stack(parts: list, empty: tuple) -> np.ndarray:
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(empty, np.int64)


class ParticlePress:
    """Adds the deformables of a model (or an ``alloy.Sim``) to press maps of ``num_images`` gels.

    ``image_world[i]`` is the world of image i (default: image i = world i); ``sim``: an ``alloy.Sim`` whose layouts
    say which particles are cloth / soft surfaces and which are grains or liquid (None: the bare-model rule)."""

    def __init__(self, model, num_images: int, height: int, width: int, mm_per_px: float, max_depth_mm: float = 4.0,
                 gel_radius_mm: float = 0.0, image_world: np.ndarray | None = None, sim=None):
        self.model, self.device = model, str(model.device)
        self.shape = (int(num_images), int(height), int(width))
        self.mm, self.max_depth, self.gel_r = float(mm_per_px), float(max_depth_mm), float(gel_radius_mm)
        tris, spheres = (np.zeros((0, 3), np.int64), np.zeros(0, np.int64)) if int(model.particle_count) == 0 \
            else self._layout(model, sim)
        self.num_tris, self.num_spheres = len(tris), len(spheres)
        if not self.active:
            return
        n_worlds = max(int(getattr(model, "world_count", 1) or 1), 1)
        iw = np.arange(self.shape[0]) if image_world is None else np.asarray(image_world, np.int64)
        world_image = np.full(n_worlds, -1, np.int32)
        for i, w in enumerate(iw):
            if 0 <= w < n_worlds and world_image[w] < 0:
                world_image[w] = i
        dev = self.device
        self._tris = wp.array(tris.astype(np.int32).reshape(-1, 3), dtype=wp.int32, device=dev)
        self._spheres = wp.array(spheres.astype(np.int32), dtype=wp.int32, device=dev)
        self._world_image = wp.array(world_image, dtype=wp.int32, device=dev)
        self._pworld = model.particle_world if getattr(model, "particle_world", None) is not None else \
            wp.zeros(int(model.particle_count), dtype=wp.int32, device=dev)
        self.soft = wp.zeros(self.shape, dtype=float, device=dev)
        log.info("tactile: %d deformable triangles and %d particles can press the gel", self.num_tris,
                 self.num_spheres)

    @staticmethod
    def _layout(model, sim) -> tuple[np.ndarray, np.ndarray]:
        if sim is not None and getattr(sim, "index", None) is not None:
            from ..errors import CameraError
            try:
                return _sim_layout(sim, int(getattr(model, "world_count", 1) or 1))
            except CameraError as exc:  # an unusual particle layout: fall back to the bare-model rule
                log.warning("tactile: %s; deformables press as their triangles and particles instead", exc)
        return _model_layout(model)

    @property
    def active(self) -> bool:
        """True when some deformable can press the gel."""
        return self.num_tris + self.num_spheres > 0

    def add(self, state, gel_xf: wp.array, press: wp.array, touched: wp.array) -> None:
        """Merge the deformables' press at ``state`` (``state.particle_q``) into ``press`` / ``touched``."""
        pq = getattr(state, "particle_q", None)
        if not self.active or pq is None:
            return
        m, dev = self.model, self.device
        self.soft.zero_()
        args = [self.mm, self.max_depth, self.gel_r]
        if self.num_spheres:
            wp.launch(splat_spheres, dim=self.num_spheres, device=dev,
                      inputs=[pq, m.particle_radius, m.particle_flags, self._spheres, self._pworld, self._world_image,
                              gel_xf, *args], outputs=[self.soft])
        if self.num_tris:
            wp.launch(splat_tris, dim=self.num_tris, device=dev,
                      inputs=[pq, m.particle_radius, self._tris, self._pworld, self._world_image, gel_xf, *args],
                      outputs=[self.soft])
        wp.launch(merge_press, dim=self.shape, inputs=[self.soft], outputs=[press, touched], device=dev)
