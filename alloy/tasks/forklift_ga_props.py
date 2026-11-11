# SPDX-License-Identifier: Apache-2.0
"""GA's loads and rack for the forklift task runs (SCR-WP8-12 F5), built in GA's facility frame.

- ``add_pallet``: GA's GMA pallet (F/loads.py:26-121) as one free body: three stringers, seven top and five bottom
  deck boards, and an optional wrapped unit load fused on top, with the composite mass and inertia GA computes. The
  body frame has its origin at the pallet's bottom face centre, x along the stringers (the forks enter from +-x).
- ``carried_at``: where the carried pallet sits seated on the truck's forks for a given truck placement and mast
  pose (GA builds it there, F/scene.py:228-248).
- ``add_rack_row``: a selective rack row as GA's ``rack_row`` (F/facility.py:71-105): uprights, beam pairs at 1.60 and
  3.20 m and column guards, labelled as GA's geoms ('<row><frame>_beam<bay-1>_<level>_<x>'), all static.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.plugins.harness.scene import body_xform
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_site import IN, SITE

LEAD, INNER = 5.5 * IN, 3.5 * IN               # F/params.py:966 (GMA deck boards)
MU_FORK_PALLET, MU_PALLET_FLOOR, MU_CARTON = 0.35, 0.50, 0.30      # F/params.py:595-597
FORK_GAP = 0.01                                # m between the backrest face and the carried pallet
POCKET = 0.03           # m of lift that leaves the lowered blades mid-pocket in a pallet resting on the floor, clear
                        # of its top deck (2.4 cm) and its bottom boards (2.4 cm): set down there, nothing drags
REST_GAP = 2.0e-4       # m a resting load starts above its support: it lands at 0.06 m/s (GA's set-down limit 0.10)


@dataclass(frozen=True)
class UnitLoad:
    """A wrapped load on a pallet: mass (kg) and full size (m)."""

    mass: float
    size: tuple[float, float, float]


def gma_boxes() -> list[tuple[np.ndarray, np.ndarray, str]]:
    """[(centre, half extents, kind)] of the pallet's boards in its frame (F/loads.py:26-46)."""
    g = SITE.pallet
    L, W, H, t, sw = g.L, g.W, g.H, g.deck, g.stringer_w
    hs = H - 2 * t
    out = [(np.array([0.0, y, t + hs / 2]), np.array([L / 2, sw / 2, hs / 2]), "stringer")
           for y in (-(W / 2 - sw / 2), 0.0, W / 2 - sw / 2)]
    for widths, z, kind in (([LEAD] + [INNER] * 5 + [LEAD], H - t / 2, "top_deck"),
                            ([LEAD] + [INNER] * 3 + [LEAD], t / 2, "bottom_deck")):
        xs = np.linspace(-L / 2 + widths[0] / 2, L / 2 - widths[-1] / 2, len(widths))
        out += [(np.array([x, 0.0, z]), np.array([w / 2, W / 2, t / 2]), kind) for x, w in zip(xs, widths,
                                                                                                strict=True)]
    return out


def _composite(parts: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, centre of mass and inertia about it of solid boxes (centre, half extents, mass)."""
    mass = sum(p[2] for p in parts)
    com = sum(p[2] * p[0] for p in parts) / mass
    inertia = np.zeros((3, 3))
    for c, h, m in parts:
        a, b, cc = 2 * h
        box = np.diag([m * (b * b + cc * cc), m * (a * a + cc * cc), m * (a * a + b * b)]) / 12
        d = c - com
        inertia += box + m * (d @ d * np.eye(3) - np.outer(d, d))
    return mass, com, inertia


def pallet_load(load: UnitLoad | None) -> tuple[np.ndarray, np.ndarray] | None:
    """The unit load's box (centre, half extents) in the pallet frame."""
    if load is None:
        return None
    return np.array([0.0, 0.0, SITE.pallet.H + load.size[2] / 2]), np.asarray(load.size) / 2


def add_pallet(b: newton.ModelBuilder, label: str, xform: wp.transform, load: UnitLoad | None = None) -> int:
    """A free pallet body ``label`` at ``xform`` (its bottom face centre), with ``load`` fused on top."""
    g = SITE.pallet
    boxes = gma_boxes()
    volume = sum(8 * float(np.prod(h)) for _, h, _ in boxes)
    parts = [(c, h, g.mass * 8 * float(np.prod(h)) / volume) for c, h, _ in boxes]
    top = pallet_load(load)
    if top is not None:
        parts.append((top[0], top[1], load.mass))
    mass, com, inertia = _composite(parts)
    body = b.add_body(xform=xform, mass=float(mass), com=wp.vec3(*com), inertia=wp.mat33(*inertia.ravel()),
                      lock_inertia=True, label=label)
    wood = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU_PALLET_FLOOR)
    for k, (c, h, kind) in enumerate(boxes):
        b.add_shape_box(body, xform=body_xform(*c), hx=h[0], hy=h[1], hz=h[2], cfg=wood, label=f"{label}:{kind}{k}")
    if top is not None:
        carton = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU_CARTON)
        b.add_shape_box(body, xform=body_xform(*top[0]), hx=top[1][0], hy=top[1][1], hz=top[1][2], cfg=carton,
                        label=f"{label}:wrapped")
    return body


def add_box_load(b: newton.ModelBuilder, label: str, xform: wp.transform, size: tuple[float, float, float],
                 mass: float, mu: float) -> int:
    """A free box (a bale, a dunnage strip) ``label`` at ``xform`` (its bottom face centre), uniform."""
    hx, hy, hz = (0.5 * v for v in size)
    inertia = mass / 3.0 * np.diag([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])
    body = b.add_body(xform=xform, mass=float(mass), com=wp.vec3(0.0, 0.0, hz), inertia=wp.mat33(*inertia.ravel()),
                      lock_inertia=True, label=label)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu)
    b.add_shape_box(body, xform=body_xform(0.0, 0.0, hz), hx=hx, hy=hy, hz=hz, cfg=cfg, label=label)
    return body


def carried_at(truck: wp.transform, pose: ft.Pose) -> wp.transform:
    """The carried pallet's frame seated on the forks: its top deck's underside on the blades, its near face
    ``FORK_GAP`` ahead of the fork face, x along the truck's heading."""
    g = SITE.pallet
    local = np.array([ft.FORK_FACE + FORK_GAP + g.L / 2 - ft.PIVOT[0], 0.0,
                      ft.FORK_BOTTOM + ft.FORK_THICK - (g.H - g.deck) - ft.PIVOT[2]])
    p = ft.to_truck(local, "sideshift", pose)
    rel = wp.transform(wp.vec3(*p), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pose.tilt))
    return wp.transform_multiply(truck, rel)


def add_rack_row(b: newton.ModelBuilder, row: str, bays: range) -> list[str]:
    """Static uprights, beams and guards of rack row ``row`` ('A', 'B' or 'C') for ``bays`` (1-based), labelled as
    GA's geoms. Returns every label."""
    from alloy.tasks.forklift_site import _row_frames  # noqa: PLC0415

    k = SITE.rack
    x0, double, _ = _row_frames(SITE)[row]
    frames = [(x0, x0 + k.frame_depth)]
    if double:
        frames.append((x0 + k.frame_depth + k.frame_gap, x0 + 2 * k.frame_depth + k.frame_gap))
    steel = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5)
    labels: list[str] = []

    def box(label: str, centre: tuple, half: tuple) -> None:
        b.add_shape_box(-1, xform=body_xform(*centre), hx=half[0], hy=half[1], hz=half[2], cfg=steel, label=label)
        labels.append(label)
    for fi, (fa, fb) in enumerate(frames):
        for j in range(bays.start - 1, bays.stop):
            y = k.y0 + j * k.bay_pitch
            for xx in (fa + k.upright / 2, fb - k.upright / 2):
                box(f"{row}{fi}_up{j}_{xx:.2f}", (xx, y, k.upright_h / 2), (k.upright / 2, k.upright / 2,
                                                                             k.upright_h / 2))
        for j in range(bays.start - 1, bays.stop - 1):
            yc = k.y0 + (j + 0.5) * k.bay_pitch
            for lv, z in enumerate(k.beam_levels, start=1):
                for xx in (fa + k.beam_w / 2, fb - k.beam_w / 2):
                    box(f"{row}{fi}_beam{j}_{lv}_{xx:.2f}", (xx, yc, z - k.beam_h / 2),
                        (k.beam_w / 2, k.beam / 2, k.beam_h / 2))
        for yy in (k.y0 - 0.12, k.y0 + k.n_bays * k.bay_pitch + 0.12):
            box(f"{row}{fi}_guard_{yy:.1f}", ((fa + fb) / 2, yy, 0.25), (k.frame_depth / 2, 0.06, 0.25))
    return labels


def yaw_xform(x: float, y: float, z: float, yaw: float) -> wp.transform:
    return wp.transform(wp.vec3(x, y, z), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))


__all__ = ["FORK_GAP", "MU_FORK_PALLET", "POCKET", "REST_GAP", "UnitLoad", "add_box_load", "add_pallet",
           "add_rack_row", "carried_at", "gma_boxes", "pallet_load", "yaw_xform"]
