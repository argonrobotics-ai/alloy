# SPDX-License-Identifier: Apache-2.0
"""Renderer-neutral scene description: unique meshes, their instances in each world, colours and lights.

A :class:`SceneData` says *what* to draw; ``alloy.camera.from_newton`` builds one from a Newton model, and
``alloy.camera.pack`` from a scene pack on disk. Instances that follow a rigid body store the body index and the
mesh pose in the body frame; every render reads the body poses from the simulation state.

Example::

    import numpy as np
    from alloy.camera.scene import MeshGeom, SceneData, Sun
    tri = MeshGeom("tri", np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32), np.array([[0, 1, 2]], np.int32))
    sd = SceneData.single_world([tri], colors=[(0.8, 0.2, 0.2)])
    assert sd.num_instances == 1 and sd.inst_group[0] == 0
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .errors import CameraError

IDENTITY_XF = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)  # (px, py, pz, qx, qy, qz, qw)


@dataclass
class MeshGeom:
    """One unique triangle mesh in its own frame (metres)."""

    name: str
    points: np.ndarray  # (V, 3) float32
    tris: np.ndarray  # (T, 3) int32, counter-clockwise seen from outside
    uv: np.ndarray | None = None  # (V, 2) float32 per vertex, glTF: u right, v down, (0, 0) = image top-left; repeat
    texture: int = -1  # index into SceneData.textures; -1 = untextured (flat instance colour)

    def __post_init__(self) -> None:
        self.points = np.ascontiguousarray(self.points, np.float32).reshape(-1, 3)
        self.tris = np.ascontiguousarray(self.tris, np.int32).reshape(-1, 3)
        if self.uv is not None:
            self.uv = np.ascontiguousarray(self.uv, np.float32).reshape(-1, 2)
            if len(self.uv) != len(self.points):
                raise CameraError(f"mesh {self.name!r}: {len(self.uv)} UVs for {len(self.points)} points",
                                  "pass one (u, v) per point, or uv=None")
        self.texture = int(self.texture)
        if self.texture >= 0 and self.uv is None:
            raise CameraError(f"mesh {self.name!r} has a texture but no UVs", "pass uv with the texture, or texture=-1")
        if len(self.tris) and (self.tris.min() < 0 or self.tris.max() >= len(self.points)):
            raise CameraError(f"mesh {self.name!r}: triangle index outside 0..{len(self.points) - 1}",
                              "pass tris as (T, 3) indices into points")


@dataclass
class Sun:
    """A distant light (Blender sun): direction toward the sun, irradiance [W/m^2] per channel, apex angle [rad]."""

    direction_to_sun: np.ndarray
    strength: tuple[float, float, float]
    angle: float = math.radians(0.526)

    @staticmethod
    def from_euler(euler_xyz: tuple[float, float, float], energy: float, color=(1.0, 1.0, 1.0),
                   angle: float | None = None) -> Sun:
        """A Blender sun object: it shines along its local -Z, rotated by Euler XYZ angles [rad].

        Example: ``Sun.from_euler((0.6, 0.3, 0.8), 3.0)`` is the default studio sun.
        """
        to_sun = euler_xyz_to_matrix(euler_xyz) @ np.array([0.0, 0.0, 1.0])
        return Sun(to_sun / np.linalg.norm(to_sun), tuple(float(c) * energy for c in color),
                   math.radians(0.526) if angle is None else float(angle))


@dataclass
class Lighting:
    """Suns plus a constant world colour (the sky seen by every ray that leaves the scene)."""

    suns: list[Sun] = field(default_factory=list)
    world_color: tuple[float, float, float] = (0.6, 0.65, 0.7)
    world_strength: float = 1.0

    @staticmethod
    def studio() -> Lighting:
        """The default: one sun (energy 3, Euler XYZ 0.6, 0.3, 0.8 rad) and a pale blue-grey sky."""
        return Lighting([Sun.from_euler((0.6, 0.3, 0.8), 3.0)], (0.6, 0.65, 0.7), 1.0)


@dataclass(eq=False)
class SceneData:
    """Instances of unique meshes, grouped by world (-1 = shared by every world), with linear RGB colours."""

    meshes: list[MeshGeom]
    inst_mesh: np.ndarray  # (I,) int32 index into meshes
    inst_group: np.ndarray  # (I,) int32 world index, -1 = global (static, shared by all worlds)
    inst_body: np.ndarray  # (I,) int32 index into state.body_q, -1 = fixed
    inst_local: np.ndarray  # (I, 7) float32 body-from-mesh transform (world-from-mesh when inst_body = -1)
    inst_color: np.ndarray  # (I, 3) float32 linear RGB albedo
    num_worlds: int = 1
    inst_name: list[str] = field(default_factory=list)
    lighting: Lighting = field(default_factory=Lighting.studio)
    camera: dict | None = None  # the scene's own view: {"pose": (7,), "fov_deg", "fit", "clip"} (scene packs)
    textures: list = field(default_factory=list)  # uint8 (H, W, 3) sRGB images, row 0 = top (MeshGeom.texture)
    inst_textured: np.ndarray | None = None  # (I,) bool, default True; False = flat inst_color (a card look)

    def __post_init__(self) -> None:
        self.inst_mesh = np.asarray(self.inst_mesh, np.int32).reshape(-1)
        n = len(self.inst_mesh)
        self.inst_group = np.asarray(self.inst_group, np.int32).reshape(n)
        self.inst_body = np.asarray(self.inst_body, np.int32).reshape(n)
        self.inst_local = np.asarray(self.inst_local, np.float32).reshape(n, 7)
        self.inst_color = np.asarray(self.inst_color, np.float32).reshape(n, 3)
        if not self.inst_name:
            self.inst_name = [f"inst_{i}" for i in range(n)]
        self.inst_textured = (np.ones(n, bool) if self.inst_textured is None
                              else np.asarray(self.inst_textured, bool).reshape(n))
        self.validate()

    @property
    def num_instances(self) -> int:
        return len(self.inst_mesh)

    def validate(self) -> None:
        """Raise CameraError on inconsistent arrays (the message says which field to fix)."""
        if self.num_worlds < 1:
            raise CameraError(f"num_worlds is {self.num_worlds}", "use at least one world")
        if self.num_instances == 0:
            raise CameraError("the scene has nothing to draw", "add visible shapes or meshes before rendering")
        if self.inst_mesh.min() < 0 or self.inst_mesh.max() >= len(self.meshes):
            raise CameraError("inst_mesh points outside meshes", f"use indices 0..{len(self.meshes) - 1}")
        bad = (self.inst_group < -1) | (self.inst_group >= self.num_worlds)
        if bad.any():
            raise CameraError(f"inst_group {int(self.inst_group[bad][0])} is not a world",
                              f"use -1 (shared) or 0..{self.num_worlds - 1}")
        if not np.isfinite(self.inst_local).all() or not np.isfinite(self.inst_color).all():
            raise CameraError("non-finite instance pose or colour", "check inst_local and inst_color for NaN/inf")
        for m in self.meshes:
            if m.texture >= len(self.textures):
                raise CameraError(f"mesh {m.name!r} uses texture {m.texture} of {len(self.textures)}",
                                  "append the image to SceneData.textures, or set texture=-1")
        for k, img in enumerate(self.textures):
            if not (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3):
                raise CameraError(f"texture {k} is not a uint8 (H, W, 3) image", "decode it to sRGB uint8 RGB")

    @property
    def textured(self) -> bool:
        """True when any mesh carries an image texture."""
        return any(m.texture >= 0 for m in self.meshes)

    @staticmethod
    def single_world(meshes: list[MeshGeom], colors, lighting: Lighting | None = None,
                     textures: list | None = None) -> SceneData:
        """One fixed instance of each mesh, placed as given, in a single world (``textures``: their images)."""
        n = len(meshes)
        return SceneData(meshes, np.arange(n), np.zeros(n), np.full(n, -1), np.tile(IDENTITY_XF, (n, 1)),
                         np.asarray(colors, np.float32).reshape(n, 3), 1, [m.name for m in meshes],
                         lighting or Lighting.studio(), textures=list(textures or []))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space (lo, hi) of every instance at its stored pose (bodies at their stored local pose only)."""
        lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
        for i in range(self.num_instances):
            pts = transform_points(self.inst_local[i], self.meshes[self.inst_mesh[i]].points)
            if len(pts):
                lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
        return lo, hi


def euler_xyz_to_matrix(e) -> np.ndarray:
    """Blender Euler XYZ [rad] -> rotation matrix R = Rz @ Ry @ Rx."""
    cx, sx, cy, sy, cz, sz = (math.cos(e[0]), math.sin(e[0]), math.cos(e[1]), math.sin(e[1]),
                              math.cos(e[2]), math.sin(e[2]))
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quat_to_matrix(q) -> np.ndarray:
    """(qx, qy, qz, qw) -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit (qx, qy, qz, qw)."""
    tr = np.trace(r)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        q = np.array([(r[2, 1] - r[1, 2]) * s, (r[0, 2] - r[2, 0]) * s, (r[1, 0] - r[0, 1]) * s, 0.25 / s])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[j] = (r[j, i] + r[i, j]) / s
        q[k] = (r[k, i] + r[i, k]) / s
        q[3] = (r[k, j] - r[j, k]) / s
    return q / np.linalg.norm(q)


def transform_points(xf: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a (px, py, pz, qx, qy, qz, qw) transform to (V, 3) points."""
    return np.asarray(pts, np.float64) @ quat_to_matrix(xf[3:7]).T + np.asarray(xf[:3], np.float64)
