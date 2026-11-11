# SPDX-License-Identifier: Apache-2.0
"""Peel rig for the bonds laws and the tape task. One world holds:
- a static steel plate (top at z = 0);
- a PET tape strip lying on it along +x (SolverFilm, the 3M 7818 face stock);
- a grip box above the strip's tab, carried by a two-axis MuJoCo gantry (prismatic x and z on position drives).
The film is a second entry, coupled lagged through ``alloy.plugins.simulation`` (Newton's stock proxy).

The grip box collides with nothing; the bonds hold the tab to it. ``free_x`` leaves the x axis undriven: the arm's
pull then keeps the grip above the front, which is the sliding fixture of a 90° peel test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import body_xform
from alloy.solvers.film.config import FilmConfig
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import FilmExtras, FilmSpec
from alloy.solvers.film.rigs import grid_mesh

PET_7818 = FilmSpec(youngs=4.0e9, poisson=0.38, density=1390.0, thickness=8.0e-5, yield_stress=1.0e8,
                    extras=FilmExtras(air=0.0, bend_damping=0.0))
"""3M 7818 face stock: 0.08 mm PET (sheet). E 4 GPa, ν 0.38 and ρ 1390 kg/m³ are textbook PET (ESTIMATES); the strain
cap is SC8's yield/E. Air and hinge damping are off for the laws (DC5 review R6)."""
BOPP_60 = FilmSpec(youngs=2.0e9, poisson=0.4, density=905.0, thickness=6.0e-5, yield_stress=3.0e7,
                   extras=FilmExtras(air=0.0, bend_damping=0.0))
"""60 µm BOPP label face stock (ESTIMATES: E 2 GPa, ν 0.4, ρ 905 kg/m³, yield 30 MPa), as in the B2 resolution check."""
GRIP_GAP = 0.5e-3                     # m between the film's contact surface and the grip box
LAND_GAP = 1.0e-4                     # m between a tube roller and the label's edge at the start
CONTACT_KE = 2.0e4
"""N/m, film-to-shape contact stiffness (model soft_contact_ke and the shapes' ke). At 1e5 Pa on a 2.5 mm cell a
vertex carries 0.6 N: 30 µm, 6 % of the 0.5 mm contact radius."""


@dataclass(frozen=True)
class PeelRig:
    length: float = 0.100             # m, strip along +x from x = 0
    width: float = 0.025              # m, across y (centred)
    dx: float = 2.5e-3                # m, mesh spacing
    tab: float = 5.0e-3               # m of the strip's start held by the grip
    film: FilmSpec = PET_7818
    plate_half: tuple[float, float, float] = (0.075, 0.03, 0.005)
    link_mass: float = 0.2            # kg per gantry link
    drive_ke: float = 2.0e4           # N/m
    free_x: bool = False              # 90° fixture: no x drive (the grip follows the front)
    effort: float = 0.0               # N, z drive force limit (0: none); a weak robot
    dt: float = 1.0e-3                # s, exchange step
    control_dt: float = 0.01          # s
    iterations: int = 10              # VBD iterations
    film_substeps: int = 1            # film entry substeps per exchange (the API picks 1..64 at runtime)
    film_config: FilmConfig = field(default_factory=FilmConfig)

    @property
    def z_film(self) -> float:
        return self.film.extras.contact_radius

    @property
    def grip_z(self) -> float:
        """Height of the grip box's centre at the start (its bottom GRIP_GAP above the film's contact surface)."""
        return 2.0 * self.z_film + GRIP_GAP + 1.0e-3


def _gantry(b: newton.ModelBuilder, rig: Any, start_x: float, start_z: float, half: tuple[float, float, float],
            label: str, touches_film: bool, roller: Any = None, prefix: str = "gantry", at_y: float = 0.0) -> int:
    """A two-axis gantry (prismatic x, z: links and joints ``<prefix>_x``, ``<prefix>_z``) carrying a box ``at_y``
    off its axis (or the roller) labelled ``label``; returns the z link."""
    start = body_xform(start_x, 0.0, start_z)
    kd = 2.0 * math.sqrt(rig.drive_ke * 2.0 * rig.link_mass)
    inertia = ((1e-4, 0, 0), (0, 1e-4, 0), (0, 0, 1e-4))
    links, joints = [], []
    for k, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))):
        name = f"{prefix}_" + "xz"[k]
        link = b.add_link(xform=start, mass=rig.link_mass, inertia=inertia, label=name)
        free = (k == 0 and rig.free_x) or (k == 1 and getattr(rig, "dead_weight", False))
        effort = rig.effort if (k == 1 and rig.effort > 0.0) else 1.0e6
        joints.append(b.add_joint_prismatic(links[-1] if links else -1, link, axis=axis,
                                            parent_xform=start if not links else body_xform(0, 0, 0),
                                            target_ke=0.0 if free else rig.drive_ke, target_kd=1.0 if free else kd,
                                            effort_limit=effort, label=name))
        links.append(link)
    b.add_articulation(joints, label=prefix)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0 if roller else 0.5, ke=CONTACT_KE,
                                          has_shape_collision=False, has_particle_collision=touches_film)
    if roller is None:
        b.add_shape_box(links[-1], xform=body_xform(0.0, at_y, 0.0), hx=half[0], hy=half[1], hz=half[2], cfg=cfg,
                        label=label)
        return links[-1]
    across = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)      # the capsule's z along world y
    b.add_shape_capsule(links[-1], xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), across), radius=roller.radius,
                        half_height=0.5 * roller.width, cfg=cfg, label=label)       # frictionless: it rolls
    return links[-1]


@dataclass(frozen=True)
class PressRig(PeelRig):
    """A free film patch on the plate under a pad on the same gantry (no grip): the formation laws. The pad's z drive
    is force-limited at ``effort`` and its target is set below the plate, so the pad presses with ``effort`` plus its
    weight."""

    length: float = 0.020
    width: float = 0.020
    pad_half: tuple[float, float, float] = (0.0125, 0.0125, 0.002)
    link_mass: float = 1.0            # kg: the pad's contact reaction is lagged; k·dt²/m ~ 0.2 at 81 cells
    dt: float = 5.0e-4

    @property
    def pad_z(self) -> float:
        return 2.0 * self.z_film + GRIP_GAP + self.pad_half[2]


@dataclass(frozen=True)
class RollRig(PressRig):
    """A free film patch on the plate under a rubber roller (``wetout.Roller``, a frictionless capsule across the
    strip, colliding with the film only) on the gantry. It starts ``lead`` before the strip and the script rolls it
    along x (wet-out laws). It presses with the z servo's ``effort`` plus the z link's weight, or, ``dead_weight``,
    rides a free vertical slide and presses with its weight alone (``link_mass``: the card's 2.04 kg)."""

    roller: Any = None                 # wetout.Roller (default: the standard ASTM D3330 roller)
    lead: float = 6.0e-3               # m before the strip's start
    effort: float = 5.0                # N
    dead_weight: bool = False

    @property
    def card(self) -> Any:
        from .wetout import STANDARD_ROLLER  # noqa: PLC0415

        return self.roller or STANDARD_ROLLER

    @property
    def roller_z(self) -> float:
        return 2.0 * self.z_film + GRIP_GAP + self.card.radius


@dataclass(frozen=True)
class AirRig(PressRig):
    """A label pre-bonded on a plate around a recess, the trapped-air pocket (laws T-1..T-3). The recess floor is the
    gantry's pad (``piston``), driven up under the pocket. A static lid rests on the label over the recess (contact
    stiffness ``lid_ke``) and keeps it flat, so the pocket's volume is the recess's. ``open_end``: the recess runs out
    past the label's end, and the pocket vents through the label's edge. No gravity: the piston's servo then holds its
    target to the nanometre (no load), and the label lies flat."""

    length: float = 0.030
    width: float = 0.020
    film: FilmSpec = BOPP_60
    recess: tuple[float, float] = (0.014, 0.012)     # m, along x and y, centred on the label
    depth: float = 5.0e-4                            # m, the floor below the plate's top at the start
    open_end: bool = False
    lid: bool = True
    lid_ke: float = 1.0e8                            # N/m
    link_mass: float = 0.05
    dt: float = 1.0e-3
    gravity: float = 0.0

    @property
    def recess_x(self) -> tuple[float, float]:
        cx = 0.5 * self.length
        return cx - 0.5 * self.recess[0], (self.length + 5.0e-3) if self.open_end else cx + 0.5 * self.recess[0]


def build_air_world(b: newton.ModelBuilder, rig: AirRig) -> None:
    """The plate as frame boxes around the recess (substrates plate, frame_e (closed recess only), frame_s,
    frame_n), the piston under it, the label on top and the lid over the recess."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, ke=CONTACT_KE)
    x0, x1, y1, t = -5.0e-3, rig.length + 5.0e-3, 0.5 * rig.width + 5.0e-3, 5.0e-3
    lo, hi = rig.recess_x
    ry = 0.5 * rig.recess[1]

    def box(label: str, xa: float, xb: float, ya: float, yb: float) -> None:
        b.add_shape_box(-1, xform=body_xform(0.5 * (xa + xb), 0.5 * (ya + yb), -0.5 * t), hx=0.5 * (xb - xa),
                        hy=0.5 * (yb - ya), hz=0.5 * t, cfg=cfg, label=label)

    box("plate", x0, lo, -y1, y1)
    if not rig.open_end:
        box("frame_e", hi, x1, -y1, y1)
    box("frame_s", lo, hi, -y1, -ry)
    box("frame_n", lo, hi, ry, y1)
    nx, ny = int(round(rig.length / rig.dx)), int(round(rig.width / rig.dx))
    v, tris = grid_mesh(nx, ny, rig.dx)
    add_film_mesh(b, v, tris, rig.film, pos=(0.0, -0.5 * rig.width, rig.z_film), label="label")
    if rig.lid:
        lid = newton.ModelBuilder.ShapeConfig(mu=0.0, ke=rig.lid_ke)
        top = min(hi, rig.length + 1.0e-3)
        b.add_shape_box(-1, xform=body_xform(0.5 * (lo + top), 0.0, 2.0 * rig.z_film + 1.0e-3),
                        hx=0.5 * (top - lo) + 1e-3, hy=ry + 1.0e-3, hz=1.0e-3, cfg=lid, label="lid")
    half = (0.5 * (hi - lo) - 0.5e-3, ry - 0.5e-3, 2.0e-3)
    _gantry(b, rig, 0.5 * (lo + hi), -rig.depth - half[2], half, "piston", False)


@dataclass(frozen=True)
class EdgeRig(PressRig):
    """An edges-first application (trapped-air test): a 20 mm label floats ``lift0`` above the plate (no gravity).
    A frame pad on one gantry (``frame_x``/``frame_z``) presses its ``band`` wide edge band; a centre pad on another
    (``pad_x``/``pad_z``) presses the rest inside it. Each z drive is force-limited at ``ratio`` x p_form over its
    own area (``efforts``)."""

    film: FilmSpec = BOPP_60
    dx: float = 2.0e-3
    lift0: float = 3.0e-4                            # m: the air under the label at the start
    band: float = 3.0e-3                             # m: GA's edge band
    ratio: float = 2.0                               # x p_form
    p_form: float = 1.0e5                            # Pa (set from the adhesive card)
    link_mass: float = 0.5
    gravity: float = 0.0

    @property
    def efforts(self) -> tuple[float, float]:
        """(frame, pad) drive force limits [N]."""
        inner = self.length - 2.0 * self.band
        return (self.ratio * self.p_form * (self.length * self.width - inner * inner),
                self.ratio * self.p_form * (inner - 2.0e-3) ** 2)


def build_edge_world(b: newton.ModelBuilder, rig: EdgeRig) -> None:
    """The plate, the floating label, the frame pad (four boxes) and the centre pad."""
    from dataclasses import replace  # noqa: PLC0415

    b.add_shape_box(-1, xform=body_xform(0.5 * rig.length, 0.0, -0.005), hx=0.02, hy=0.02, hz=0.005,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=CONTACT_KE), label="plate")
    nx, ny = int(round(rig.length / rig.dx)), int(round(rig.width / rig.dx))
    v, t = grid_mesh(nx, ny, rig.dx)
    add_film_mesh(b, v, t, rig.film, pos=(0.0, -0.5 * rig.width, rig.z_film + rig.lift0), label="label")
    z0 = 2.0 * rig.z_film + rig.lift0 + GRIP_GAP + 1.0e-3
    cx, outer, inner = 0.5 * rig.length, 0.5 * rig.length + 2.0e-3, 0.5 * rig.length - rig.band
    frame, pad = (replace(rig, effort=e) for e in rig.efforts)
    side = 0.5 * (outer - inner)
    link = _gantry(b, frame, cx, z0, (outer, side, 1.0e-3), "frame", True, prefix="frame", at_y=inner + side)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, ke=CONTACT_KE, has_shape_collision=False)
    b.add_shape_box(link, xform=body_xform(0.0, -(inner + side), 0.0), hx=outer, hy=side, hz=1.0e-3, cfg=cfg)
    for sx in (-1.0, 1.0):
        b.add_shape_box(link, xform=body_xform(sx * (inner + side), 0.0, 0.0), hx=side, hy=inner, hz=1.0e-3, cfg=cfg)
    h = inner - 1.0e-3
    _gantry(b, pad, cx, z0, (h, h, 1.0e-3), "pad", True, prefix="pad")


@dataclass(frozen=True)
class TubeRig(RollRig):
    """A label on a static tube (radius ``tube_radius``, axis along y, top at z = 0 under the label's middle) under the
    roller (law C-1). The label is built flat and posed wrapped on the tube (``pose``): its rest layout stays flat."""

    tube_radius: float = 0.025
    lead: float = 15.0e-3              # m: the contact sweeps from 2 mm before the label to 2 mm past it

    @property
    def reach(self) -> float:
        """m between the tube's axis and the roller's when they touch through the label."""
        return self.tube_radius + 2.0 * self.z_film + self.card.radius

    @property
    def nip_ratio(self) -> float:
        """The contact's speed along the label over the roller's along x, at the top: (R_s + r)/(R_s + 2r + R)."""
        return (self.tube_radius + self.z_film) / self.reach

    @property
    def roller_z(self) -> float:
        """The roller's start height: touching the label's leading edge (its x = 0 row, posed) from x = -lead, plus
        LAND_GAP. It used to start where it would rest on a label at x = -lead, where there is none, and fell ~1.8 mm
        onto the edge: that knock slid a 1.25 mm-cell label around the tube and off it (WP5's red, 2026-11-10)."""
        rad = self.tube_radius + self.z_film
        th = -0.5 * self.length / rad
        ex, ez = 0.5 * self.length + rad * math.sin(th), -self.tube_radius + rad * math.cos(th)
        touch, dx = self.card.radius + self.z_film, -self.lead - ex
        return ez + math.sqrt(touch * touch - dx * dx) + LAND_GAP

    def pose(self, m: newton.Model) -> None:
        """Wrap every film vertex onto the tube: arc length along x from the top, at the contact radius above it."""
        q = m.particle_q.numpy()
        rad = self.tube_radius + self.z_film
        th = (q[:, 0] - 0.5 * self.length) / rad
        q[:, 0] = 0.5 * self.length + rad * np.sin(th)
        q[:, 2] = -self.tube_radius + rad * np.cos(th)
        m.particle_q.assign(q)


def build_tube_world(b: newton.ModelBuilder, rig: TubeRig) -> None:
    """The tube (substrate ``tube``), the flat label above its top, the roller on its gantry."""
    along_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
    b.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0.5 * rig.length, 0.0, -rig.tube_radius), along_y),
                         radius=rig.tube_radius, half_height=0.03,
                         cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=CONTACT_KE), label="tube")
    nx, ny = int(round(rig.length / rig.dx)), int(round(rig.width / rig.dx))
    v, t = grid_mesh(nx, ny, rig.dx)
    add_film_mesh(b, v, t, rig.film, pos=(0.0, -0.5 * rig.width, rig.z_film), label="label")
    _gantry(b, rig, -rig.lead, rig.roller_z, rig.pad_half, "roller", True, roller=rig.card)


def build_press_world(b: newton.ModelBuilder, rig: PressRig) -> None:
    hx, hy, hz = rig.plate_half
    b.add_shape_box(-1, xform=body_xform(0.5 * rig.length, 0.0, -hz), hx=hx, hy=hy, hz=hz,
                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, ke=CONTACT_KE), label="plate")
    nx, ny = int(round(rig.length / rig.dx)), int(round(rig.width / rig.dx))
    v, t = grid_mesh(nx, ny, rig.dx)
    add_film_mesh(b, v, t, rig.film, pos=(0.0, -0.5 * rig.width, rig.z_film), label="label")
    if isinstance(rig, AirRig):
        raise TypeError("an AirRig builds with build_air_world")
    if isinstance(rig, RollRig):
        _gantry(b, rig, -rig.lead, rig.roller_z, rig.pad_half, "roller", True, roller=rig.card)
        return
    _gantry(b, rig, 0.5 * rig.length, rig.pad_z, rig.pad_half, "pad", True)


def build_world(b: newton.ModelBuilder, rig: PeelRig) -> None:
    hx, hy, hz = rig.plate_half
    plate = newton.ModelBuilder.ShapeConfig(mu=0.5, ke=CONTACT_KE)
    b.add_shape_box(-1, xform=body_xform(0.5 * rig.length, 0.0, -hz), hx=hx, hy=hy, hz=hz, cfg=plate, label="plate")
    nx, ny = int(round(rig.length / rig.dx)), int(round(rig.width / rig.dx))
    v, t = grid_mesh(nx, ny, rig.dx)
    add_film_mesh(b, v, t, rig.film, pos=(0.0, -0.5 * rig.width, rig.z_film), label="tape")
    _gantry(b, rig, 0.5 * rig.tab, rig.grip_z, (0.5 * rig.tab + 0.25e-3, 0.5 * rig.width + 1e-3, 1.0e-3), "grip",
            False)


def model(rig: PeelRig, worlds: int, device: str = "cpu") -> newton.Model:
    from alloy.solvers.film.config import register_custom_attributes  # noqa: PLC0415

    g = (0.0, 0.0, float(getattr(rig, "gravity", -9.81)))
    tmpl = newton.ModelBuilder(gravity=g)
    newton.solvers.SolverMuJoCo.register_custom_attributes(tmpl)
    register_custom_attributes(tmpl)
    build = build_world
    if isinstance(rig, PressRig):
        build = build_air_world if isinstance(rig, AirRig) else build_edge_world if isinstance(rig, EdgeRig) \
            else build_tube_world if isinstance(rig, TubeRig) else build_press_world
    build(tmpl, rig)
    root = newton.ModelBuilder(gravity=g)
    newton.solvers.SolverMuJoCo.register_custom_attributes(root)
    register_custom_attributes(root)
    root.replicate(tmpl, worlds)
    root.color()
    m = root.finalize(device=device)
    m.soft_contact_ke = CONTACT_KE
    if isinstance(rig, TubeRig):
        rig.pose(m)                                  # built flat, posed wrapped: the rest layout stays flat
    return m


def built(m: newton.Model, rig: PeelRig) -> Any:
    """Two entries: MuJoCo owns the gantry, the film entry every particle; one lagged pair MuJoCo -> film."""
    from alloy.coupling.types import PairPolicy  # noqa: PLC0415
    from alloy.scene.types import CouplerChoice, EntrySpec  # noqa: PLC0415
    from alloy.testing.integration import built_from_model  # noqa: PLC0415

    mj = EntrySpec("mujoco", "mujoco", tuple(range(m.body_count)), (), tuple(range(m.joint_count)), (), 1, rig.dt,
                   solver_kwargs={"nconmax": 64, "njmax": 128})
    film = EntrySpec("film", "film", (), tuple(range(m.particle_count)), (), (), int(rig.film_substeps), rig.dt,
                     max_substeps=int(rig.film_substeps),
                     solver_kwargs={"film": rig.film_config, "iterations": rig.iterations})
    pair = PairPolicy(source="mujoco", destination="film", mode="lagged")
    return built_from_model(m, [mj, film], control_dt=rig.control_dt, coupler=CouplerChoice("proxy", (pair,)))


def simulation(rig: PeelRig, plugins: list, worlds: int = 2, device: str = "cpu", graph: bool = False) -> Any:
    """``alloy.plugins.simulation`` of the rig with ``plugins`` (the Bonds plug-in among them); ``graph``: an
    IntegratedSimulation replaying each period as a captured graph."""
    from alloy.plugins.host import simulation as plugin_simulation  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

    register_builtin_solvers()
    m = model(rig, worlds, device)
    if not graph:
        return plugin_simulation(built(m, rig), plugins)
    from alloy.integration.simulation import IntegratedSimulation  # noqa: PLC0415

    return plugin_simulation(built(m, rig), plugins, cls=IntegratedSimulation, use_graph=True, cpu_graph=True)


def gantry_columns(sim: Any) -> np.ndarray:
    """joint_q indices [W, 2] of the gantry's x and z joints."""
    m = sim.model
    labels, starts = m.joint_label, m.joint_q_start.numpy()
    cols = [[starts[j] for j, lab in enumerate(labels) if lab.endswith(f"gantry_{a}")] for a in "xz"]
    return np.asarray(cols, np.int64).T


def set_gantry(sim: Any, cols: np.ndarray, x: float, z: float) -> None:
    """Gantry targets (m, relative to the start) in every world."""
    tgt = sim.control.joint_target_q.numpy()
    tgt[cols[:, 0]], tgt[cols[:, 1]] = x, z
    sim.control.joint_target_q.assign(tgt.astype(np.float32))


__all__ = ["BOPP_60", "PET_7818", "AirRig", "EdgeRig", "PeelRig", "PressRig", "RollRig", "TubeRig",
           "build_air_world", "build_edge_world", "build_press_world", "build_tube_world", "build_world", "built",
           "gantry_columns", "model", "set_gantry", "simulation"]
