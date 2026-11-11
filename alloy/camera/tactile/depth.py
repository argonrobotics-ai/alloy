# SPDX-License-Identifier: Apache-2.0
"""How deep each touching object presses into the gel: a depth image in the sensor frame, every world at once.

Sensor frame: origin at the centre of the gel surface, +Z out of the gel (toward the touched object), +X along image
columns (right), +Y along image rows (down): the OpenCV frame of the camera behind the gel. For each pixel a ray
starts ``max_depth_mm`` behind the gel surface and runs along +Z through the collision shapes of the world (not the
sensor's own body); the press is how far the first surface lies on the camera side of the undeformed gel.

Shapes are intersected exactly (Newton's ray-primitive routines; meshes through their ``wp.Mesh``). A cull pass keeps,
per world, only the shapes whose bounds reach the gel box, so the per-pixel loop sees a handful of shapes.

Example::

    from alloy._vendor import newton; import numpy as np; import warp as wp
    from alloy.camera.tactile.depth import PressMap
    b = newton.ModelBuilder()
    pad = b.add_body(xform=wp.transform((0, 0, 0), wp.quat_identity()))
    b.add_shape_box(pad, hx=0.01, hy=0.01, hz=0.002)
    ball = b.add_body(xform=wp.transform((0, 0, 0.0115), wp.quat_identity()))
    b.add_shape_sphere(ball, radius=0.01)                       # 0.5 mm into a gel at z = 2 mm, normal +Z
    model = b.finalize(device="cpu")
    pm = PressMap(model, sensor_body=np.array([pad]), height=60, width=80, mm_per_px=0.2, max_depth_mm=4.0)
    pm.set_local_pose(np.array([0, 0, 0.002, 0, 0, 0, 1.0]))   # gel surface centre in the pad frame
    press = pm.compute(model.state().body_q)                     # (1, 60, 80) mm, peak about 0.5
"""
from __future__ import annotations

import logging

import numpy as np
import warp as wp

from ..errors import CameraError

log = logging.getLogger(__name__)
MAX_CANDIDATES = 32  # shapes near one sensor in one world
PLANE, MESH, CONVEX_MESH, HFIELD = 1, 8, 10, 2  # newton.GeoType values (checked in _check_geo_types)
INSIDE_PROBE_M = 1.0  # mesh rays run this far [m] to tell a start inside the mesh (back face first)


def _newton_ray():
    try:
        from alloy._vendor.newton._src.geometry.raycast import map_ray_to_local, ray_intersect_shape
    except ImportError as exc:  # Newton moved its ray routines
        raise CameraError("this Newton version has no newton._src.geometry.raycast",
                          "install the Newton version Alloy pins (scripts/newton_env.sh)") from exc
    return map_ray_to_local, ray_intersect_shape


_map_ray_to_local, _ray_intersect_shape = _newton_ray()


@wp.kernel
def sensor_poses(body_q: wp.array(dtype=wp.transform), sensor_body: wp.array(dtype=wp.int32),
                 local: wp.array(dtype=wp.transform), out: wp.array(dtype=wp.transform)):
    """World pose of each image's gel frame = body pose x gel pose in the body frame."""
    i = wp.tid()
    b = sensor_body[i]
    t = local[i]
    if b >= 0:
        t = wp.transform_multiply(body_q[b], t)
    out[i] = t


@wp.func
def shape_pose(body_q: wp.array(dtype=wp.transform), shape_body: wp.array(dtype=wp.int32),
               shape_xf: wp.array(dtype=wp.transform), s: int) -> wp.transform:
    b = shape_body[s]
    if b >= 0:
        return wp.transform_multiply(body_q[b], shape_xf[s])
    return shape_xf[s]


@wp.kernel
def cull_shapes(body_q: wp.array(dtype=wp.transform), shape_body: wp.array(dtype=wp.int32),
                shape_xf: wp.array(dtype=wp.transform), shape_type: wp.array(dtype=wp.int32),
                aabb_lo: wp.array(dtype=wp.vec3), aabb_hi: wp.array(dtype=wp.vec3),
                world_start: wp.array(dtype=wp.int32), world_shapes: wp.array(dtype=wp.int32),
                image_world: wp.array(dtype=wp.int32), gel_xf: wp.array(dtype=wp.transform), box_hi: wp.vec3,
                box_lo_z: float, count: wp.array(dtype=wp.int32), cand: wp.array2d(dtype=wp.int32)):
    """Keep shape j of image i's world when its bounds reach the gel box [-hx, hx] x [-hy, hy] x [lo_z, hz]."""
    i, j = wp.tid()
    w = image_world[i]
    k = world_start[w] + j
    if k >= world_start[w + 1]:
        return
    s = world_shapes[k]
    inv = wp.transform_inverse(gel_xf[i])
    hit = bool(shape_type[s] == PLANE)
    if not hit:
        t = wp.transform_multiply(inv, shape_pose(body_q, shape_body, shape_xf, s))
        lo = aabb_lo[s]
        hi = aabb_hi[s]
        blo = wp.vec3(1.0e30)
        bhi = wp.vec3(-1.0e30)
        for c in range(8):
            p = wp.vec3(wp.where((c & 1) != 0, hi[0], lo[0]), wp.where((c & 2) != 0, hi[1], lo[1]),
                        wp.where((c & 4) != 0, hi[2], lo[2]))
            q = wp.transform_point(t, p)
            blo = wp.min(blo, q)
            bhi = wp.max(bhi, q)
        hit = (blo[0] <= box_hi[0] and bhi[0] >= -box_hi[0] and blo[1] <= box_hi[1] and bhi[1] >= -box_hi[1]
               and blo[2] <= box_hi[2] and bhi[2] >= box_lo_z)
    if hit:
        slot = wp.atomic_add(count, i, 1)
        if slot < cand.shape[1]:
            cand[i, slot] = s


@wp.func
def ray_shape(body_q: wp.array(dtype=wp.transform), shape_body: wp.array(dtype=wp.int32),
              shape_xf: wp.array(dtype=wp.transform), shape_type: wp.array(dtype=wp.int32),
              shape_scale: wp.array(dtype=wp.vec3), shape_mesh: wp.array(dtype=wp.uint64), s: int, o: wp.vec3,
              d: wp.vec3, tmax: float) -> float:
    """Distance along the unit ray to shape s: -1 on a miss beyond tmax, 0 when the ray starts inside the shape
    (its first surface is a back face / the exit of a convex primitive), so deep presses read as full presses."""
    xf = shape_pose(body_q, shape_body, shape_xf, s)
    gt = shape_type[s]
    if gt == MESH or gt == CONVEX_MESH or gt == HFIELD:
        ol, dl = _map_ray_to_local(xf, o, d, shape_scale[s])
        q = wp.mesh_query_ray(shape_mesh[s], ol, dl, INSIDE_PROBE_M)
        if not q.result:
            return -1.0
        if q.sign < 0.0 and gt != HFIELD:
            return 0.0
        return wp.where(q.t <= tmax, q.t, -1.0)
    t, n = _ray_intersect_shape(xf, shape_scale[s], gt, o, d, False)
    if t >= 0.0 and gt != PLANE and wp.dot(n, d) > 0.0:
        return 0.0
    return wp.where(t <= tmax, t, -1.0)


@wp.kernel
def press_kernel(body_q: wp.array(dtype=wp.transform), shape_body: wp.array(dtype=wp.int32),
                 shape_xf: wp.array(dtype=wp.transform), shape_type: wp.array(dtype=wp.int32),
                 shape_scale: wp.array(dtype=wp.vec3), shape_mesh: wp.array(dtype=wp.uint64),
                 count: wp.array(dtype=wp.int32), cand: wp.array2d(dtype=wp.int32),
                 gel_xf: wp.array(dtype=wp.transform), mm_per_px: float, max_depth_mm: float, gel_radius_mm: float,
                 press: wp.array3d(dtype=float), touched: wp.array3d(dtype=wp.int32)):
    """Press [mm] of every pixel of every image, and the touched shape (-1 = none)."""
    i, r, c = wp.tid()
    n = wp.min(count[i], cand.shape[1])
    press[i, r, c] = 0.0
    touched[i, r, c] = -1
    if n == 0:
        return
    x = (float(c) + 0.5 - 0.5 * float(press.shape[2])) * mm_per_px
    y = (float(r) + 0.5 - 0.5 * float(press.shape[1])) * mm_per_px
    sag = float(0.0)
    if gel_radius_mm > 0.0:
        sag = gel_radius_mm - wp.sqrt(wp.max(gel_radius_mm * gel_radius_mm - x * x - y * y, 0.0))
    xf = gel_xf[i]
    o = wp.transform_point(xf, wp.vec3(x, y, -sag - max_depth_mm) * 0.001)
    d = wp.transform_vector(xf, wp.vec3(0.0, 0.0, 1.0))
    tmax = max_depth_mm * 0.001
    best = tmax
    best_s = int(-1)
    for k in range(n):
        s = cand[i, k]
        t = ray_shape(body_q, shape_body, shape_xf, shape_type, shape_scale, shape_mesh, s, o, d, best)
        if t >= 0.0 and (t < best or (t == best and s < best_s)):  # ties: lowest shape id (candidate order varies)
            best = t
            best_s = s
    if best_s >= 0:
        press[i, r, c] = (tmax - best) * 1000.0
        touched[i, r, c] = best_s


def world_shape_lists(model, skip_bodies: set[int], num_images: int, image_world: np.ndarray):
    """CSR of the collision shapes each image's world can touch (its own world's plus the global ones)."""
    from alloy._vendor import newton
    flags = model.shape_flags.numpy()
    body, world = model.shape_body.numpy(), model.shape_world.numpy()
    collide = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    keep = collide & ~np.isin(body, sorted(skip_bodies) or [-2])
    n_worlds = int(max(image_world.max() + 1, 1))
    shared = np.flatnonzero(keep & (world < 0))
    lists = [np.concatenate([np.flatnonzero(keep & (world == w)), shared]) for w in range(n_worlds)]
    start = np.concatenate([[0], np.cumsum([len(lst) for lst in lists])]).astype(np.int32)
    flat = np.concatenate(lists).astype(np.int32) if start[-1] else np.zeros(1, np.int32)
    return start, flat, int(max((len(lst) for lst in lists), default=0))


def _check_geo_types() -> None:
    from alloy._vendor import newton
    gt = newton.GeoType
    if (int(gt.PLANE), int(gt.MESH), int(gt.CONVEX_MESH), int(gt.HFIELD)) != (PLANE, MESH, CONVEX_MESH, HFIELD):
        raise CameraError("newton.GeoType numbering changed", "update alloy.camera.tactile.depth's type constants")


class PressMap:
    """Press images (images, H, W) [mm] of one gel per image: image i sits on body ``sensor_body[i]``."""

    def __init__(self, model, sensor_body: np.ndarray, height: int, width: int, mm_per_px: float,
                 max_depth_mm: float = 4.0, gel_radius_mm: float = 0.0, image_world: np.ndarray | None = None,
                 skip_bodies: tuple[int, ...] = ()):
        _check_geo_types()
        self.model, self.device = model, str(model.device)
        self.sensor_body = np.asarray(sensor_body, np.int32).reshape(-1)
        n = len(self.sensor_body)
        self.image_world = np.arange(n, dtype=np.int32) if image_world is None else np.asarray(image_world, np.int32)
        self.shape = (n, int(height), int(width))
        self.mm_per_px, self.max_depth_mm, self.gel_radius_mm = float(mm_per_px), float(max_depth_mm), float(gel_radius_mm)
        skip = {int(b) for b in self.sensor_body if b >= 0} | {int(b) for b in skip_bodies}
        start, flat, self.max_list = world_shape_lists(model, skip, n, self.image_world)
        dev = self.device
        self._start = wp.array(start, dtype=wp.int32, device=dev)
        self._flat = wp.array(flat, dtype=wp.int32, device=dev)
        self._img_world = wp.array(self.image_world, dtype=wp.int32, device=dev)
        self._body = wp.array(self.sensor_body, dtype=wp.int32, device=dev)
        self._shape_type = wp.array(model.shape_type.numpy().astype(np.int32), dtype=wp.int32, device=dev)
        self._shape_body = wp.array(model.shape_body.numpy().astype(np.int32), dtype=wp.int32, device=dev)
        self._local = wp.zeros(n, dtype=wp.transform, device=dev)
        self.gel_xf = wp.zeros(n, dtype=wp.transform, device=dev)
        self.count = wp.zeros(n, dtype=wp.int32, device=dev)
        self.cand = wp.zeros((n, MAX_CANDIDATES), dtype=wp.int32, device=dev)
        self.press = wp.zeros(self.shape, dtype=float, device=dev)
        self.touched = wp.zeros(self.shape, dtype=wp.int32, device=dev)
        self.set_local_pose(np.array([0, 0, 0, 0, 0, 0, 1.0]))

    def set_local_pose(self, pose: np.ndarray) -> None:
        """Gel frame in the body frame: (7,) for every image or (images, 7), (px, py, pz, qx, qy, qz, qw) [m]."""
        from ..pose import as_poses
        self._local.assign(as_poses(pose, self.shape[0]))

    def compute(self, body_q: wp.array | None) -> wp.array:
        """Press map for these body poses (``state.body_q``); also fills ``gel_xf`` and ``touched``."""
        m, dev, n = self.model, self.device, self.shape[0]
        if body_q is None:
            body_q = wp.zeros(1, dtype=wp.transform, device=dev)
        wp.launch(sensor_poses, dim=n, inputs=[body_q, self._body, self._local], outputs=[self.gel_xf], device=dev)
        self.count.zero_()
        if self.max_list > 0:
            hx, hy = 0.5e-3 * self.shape[2] * self.mm_per_px, 0.5e-3 * self.shape[1] * self.mm_per_px
            sag = 1e-3 * self._max_sag(hx, hy)
            wp.launch(cull_shapes, dim=(n, self.max_list),
                      inputs=[body_q, self._shape_body, m.shape_transform, self._shape_type,
                              m.shape_collision_aabb_lower, m.shape_collision_aabb_upper, self._start, self._flat,
                              self._img_world, self.gel_xf, wp.vec3(hx, hy, 1e-4), -1e-3 * self.max_depth_mm - sag],
                      outputs=[self.count, self.cand], device=dev)
        wp.launch(press_kernel, dim=self.shape,
                  inputs=[body_q, self._shape_body, m.shape_transform, self._shape_type, m.shape_scale,
                          m.shape_source_ptr, self.count, self.cand, self.gel_xf, self.mm_per_px, self.max_depth_mm,
                          self.gel_radius_mm], outputs=[self.press, self.touched], device=dev)
        return self.press

    def _max_sag(self, hx: float, hy: float) -> float:
        r = self.gel_radius_mm
        d2 = (1e3 * hx) ** 2 + (1e3 * hy) ** 2
        return 0.0 if r <= 0 else r - float(np.sqrt(max(r * r - d2, 0.0)))

    def overflowed(self) -> bool:
        """True when some image had more nearby shapes than MAX_CANDIDATES (extra shapes were not traced)."""
        return bool((self.count.numpy() > MAX_CANDIDATES).any())
