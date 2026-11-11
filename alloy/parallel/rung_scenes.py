# SPDX-License-Identifier: Apache-2.0
"""Raw-Newton world templates for the speed ladder (parallel §4.9): arm (L0), box pile (L1), hull pile (L2),
cantilever joint-chain cable (L4).

These are the PAR-1 baseline ("raw solver") and run before the card path (WP1/WP2) lands; the ``alloy_*`` ladder
configs build the same scenes from SceneCards. Each function returns one world's ``ModelBuilder`` (replicated by the
caller) plus the facts the quality gate needs. SI units; Z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.defaults import GRAVITY, RIGID_GAP_M

BOX_HALF = 0.025  # [m] 5 cm cubes
BOX_GAP = 1e-3  # [m] horizontal gap between pile neighbours
STACK_GAP = 5e-4  # [m] vertical clearance before settling
HULL_VERTS = 32
CABLE_L, CABLE_R, CABLE_RHO, CABLE_Z = 0.5, 3e-3, 1200.0, 1.0  # [m], [m], [kg/m^3], [m]
CABLE_SAG_FRACTION = 0.05  # small-deflection design: tip drop = 5 % of L under self weight


@dataclass(frozen=True)
class SceneFacts:
    """What a quality gate needs about one world template."""

    kind: str
    n_dynamic: int  # dynamic bodies per world
    tip_body: int = -1  # L4: last link (local index)
    tip_drop_ref: float = 0.0  # L4: small-deflection elastica tip drop [m]
    arm_dofs: int = 0  # L0: driven DOFs


def pyramid_layers(count: int) -> list[tuple[int, int]]:
    """(side, boxes) per layer of a half-offset pyramid (b², (b−1)², ...) holding ``count`` boxes, bottom first."""
    if count < 1:
        raise ValueError(f"pile needs at least one object, got {count}")
    b = 1
    while sum(i * i for i in range(1, b + 1)) < count:
        b += 1
    layers, left = [], count
    for side in range(b, 0, -1):
        take = min(left, side * side)
        layers.append((side, take))
        left -= take
        if left == 0:
            break
    return layers


def pyramid_positions(count: int, half: float, gap: float) -> list[tuple[float, float, float]]:
    """Centres of a pre-settled pile: layer k is a side² grid, offset half a pitch from the layer below."""
    pitch = 2.0 * half + gap
    out = []
    for k, (side, n_layer) in enumerate(pyramid_layers(count)):
        z = half + k * (2.0 * half + STACK_GAP) + STACK_GAP
        off = -0.5 * (side - 1) * pitch
        for i in range(n_layer):
            out.append((off + (i % side) * pitch, off + (i // side) * pitch, z))
    return out


def box_pile(count: int, mu: float = 0.6) -> tuple[newton.ModelBuilder, SceneFacts]:
    """L1: ``count`` free 5 cm boxes in a pre-settled half-offset pyramid on a ground plane."""
    b = _builder()
    b.add_ground_plane()
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=500.0)
    for x, y, z in pyramid_positions(count, BOX_HALF, BOX_GAP):
        body = b.add_body(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()))
        b.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)
    return b, SceneFacts(kind="pile", n_dynamic=count)


def hull_points(seed: int, n: int = HULL_VERTS, radius: float = BOX_HALF) -> np.ndarray:
    """``n`` points on a randomly stretched ellipsoid (Fibonacci lattice), deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    pts = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return (pts * radius * rng.uniform(0.7, 1.0, size=3)).astype(np.float32)


def hull_mesh(points: np.ndarray) -> newton.Mesh:
    from scipy.spatial import ConvexHull  # SPEC §1.3 dependency

    hull = ConvexHull(points)
    faces = hull.simplices.copy()
    centre = points.mean(axis=0)
    for f in faces:  # outward winding
        a, b_, c = points[f]
        if np.dot(np.cross(b_ - a, c - a), a - centre) < 0.0:
            f[1], f[2] = f[2], f[1]
    return newton.Mesh(points, faces.reshape(-1).astype(np.int32), compute_inertia=True)


def hull_pile(count: int, mu: float = 0.6, seed: int = 0) -> tuple[newton.ModelBuilder, SceneFacts]:
    """L2: ``count`` 32-vertex convex hulls (8 distinct shapes) dropped as a loose pyramid onto a plane."""
    b = _builder()
    b.add_ground_plane()
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=500.0)
    meshes = [hull_mesh(hull_points(seed + k)) for k in range(min(count, 8))]
    for i, (x, y, z) in enumerate(pyramid_positions(count, BOX_HALF, 4.0 * BOX_GAP)):
        body = b.add_body(xform=wp.transform(wp.vec3(x, y, z + 2.0 * STACK_GAP), wp.quat_identity()))
        b.add_shape_convex_hull(body, mesh=meshes[i % len(meshes)], cfg=cfg)
    return b, SceneFacts(kind="pile", n_dynamic=count)


def cable_props(links: int) -> dict[str, float]:
    """Discretised rod: EI sized for a 5 % tip drop, k_bend = EI/l, k_twist = GJ/l with ν = 0.5 (GJ/EI ≈ 0.67)."""
    w = CABLE_RHO * math.pi * CABLE_R**2 * GRAVITY  # weight per length [N/m]
    ei = w * CABLE_L**4 / (8.0 * CABLE_SAG_FRACTION * CABLE_L)
    l = CABLE_L / links
    return {"l": l, "w": w, "EI": ei, "k_bend": ei / l, "k_twist": ei / 1.5 / l,
            "mass": CABLE_RHO * math.pi * CABLE_R**2 * l, "drop_ref": w * CABLE_L**4 / (8.0 * ei)}


def cable_chain(links: int, damping_time: float = 0.05) -> tuple[newton.ModelBuilder, SceneFacts]:
    """L4: horizontal cantilever of ``links`` capsule links joined by 3-axis angular D6 joints (two bend + one
    twist hinge, D7); the joint at x = 0 is the clamp (2x bend stiffness). Adjacent links do not collide."""
    b = _builder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    p = cable_props(links)
    l, m = p["l"], p["mass"]
    it, ia = m * (3 * CABLE_R**2 + l * l) / 12.0, 0.5 * m * CABLE_R**2
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)  # capsule axis z -> x
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)  # link mass/inertia set explicitly (rod section)
    parent, joints = -1, []
    for i in range(links):
        body = b.add_link(xform=wp.transform(wp.vec3((i + 0.5) * l, 0.0, CABLE_Z), wp.quat_identity()), mass=m,
                          inertia=wp.mat33(ia, 0, 0, 0, it, 0, 0, 0, it))
        b.add_shape_capsule(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot), radius=CABLE_R,
                            half_height=max(0.5 * l - CABLE_R, 0.25 * l), cfg=massless)
        scale = 2.0 if i == 0 else 1.0
        pxf = wp.transform(wp.vec3(0.0, 0.0, CABLE_Z) if parent < 0 else wp.vec3(0.5 * l, 0.0, 0.0),
                           wp.quat_identity())
        joints.append(b.add_joint_d6(parent, body, angular_axes=_cable_axes(p, scale, damping_time), parent_xform=pxf,
                                     child_xform=wp.transform(wp.vec3(-0.5 * l, 0.0, 0.0), wp.quat_identity())))
        parent = body
    b.add_articulation(joints)
    return b, SceneFacts(kind="cable", n_dynamic=links, tip_body=parent, tip_drop_ref=p["drop_ref"])


def _cable_axes(p: dict[str, float], scale: float, tau: float) -> list:
    cfg = newton.ModelBuilder.JointDofConfig
    kb, kt = p["k_bend"] * scale, p["k_twist"] * scale
    return [cfg(axis=(0.0, 1.0, 0.0), target_ke=kb, target_kd=kb * tau),
            cfg(axis=(0.0, 0.0, 1.0), target_ke=kb, target_kd=kb * tau),
            cfg(axis=(1.0, 0.0, 0.0), target_ke=kt, target_kd=kt * tau)]


def _builder() -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    b.rigid_gap = RIGID_GAP_M  # SPEC §4.2: never Newton's 0.1 m default
    return b
