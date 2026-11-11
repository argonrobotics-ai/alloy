# SPDX-License-Identifier: Apache-2.0
"""Deposit-cans rig (WP13, the Gym-Anything deposit-cans gap; docs/notes/film-grip-c16.md): a recycling bag held by its
two straps in a gantry's pinch heads, standing on a counter, and aluminium cans above its open mouth.

- **Bag.** A T-shirt bag: two LLDPE 30 um panels (alloy.solvers.film LLDPE_30UM; Gym-Anything's bag is 38.1 um,
  E 200 MPa) welded at the sides and bottom. Two straps rise from the sides. The top ``band`` of each strap is welded
  2-ply, and that is where the pads grip. Away from the welds the panels open to ``gap`` each side (a lens rest shape,
  as C16's bag mesh), so the mouth starts open (2·gap at the centre).
  The band is one sheet: its cells are emitted once, so it carries one ply's mass (about 0.07 g short per strap).
  Where its lower edge meets the strap's two plies, 3 edges per strap have 3 triangles (Y-junctions, a seal seam).
  Newton warns "Detected non-manifold edge" and keeps one hinge per edge, band to the + ply; the - ply has no bending
  across the junction, which is negligible at 30 um.
- **Robot.** A gantry carriage on x and z servos carries two pinch heads. Each head is two pads on y servos, pushed
  closed with ``grip_n`` (a force-limited servo, like a parallel gripper's force mode). The pads start on the band at
  the squeeze that carries ``grip_n``.
- **Cans.** Empty 12 oz aluminium cans (Ø 66 mm, 122 mm, 14 g: ESTIMATES from typical values, not a spec sheet), rigid
  thin shells (inertia of a cylindrical shell with ends), released at rest above the mouth.
- **Coupling.** MuJoCo owns the gantry, pads and cans; the film entry (SolverFilm, grip terms on) owns every particle;
  one lagged pair MuJoCo -> film, as the bonds rig (alloy.plugins.bonds.rig).
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
from alloy.solvers.film.materials import LLDPE_30UM, FilmSpec

CONTACT_KE = 2.0e4                    # N/m, film-to-shape contact stiffness (model soft ke and every shape's ke)
MU_FILM = 0.4                         # film on pads, counter and cans (model value; PE on rubber/steel 0.2-0.5)


@dataclass(frozen=True)
class BagSpec:
    width: float = 0.36               # flat width (x): the mouth between the straps takes 3 cans
    height: float = 0.26              # body below the mouth
    handle_w: float = 0.06            # strap width
    handle_h: float = 0.10            # strap height above the mouth
    band: float = 0.04                # welded 2-ply grip band at each strap top
    cell: float = 0.02                # mesh spacing; divides every length above
    gap: float = 0.05                 # half the panels' separation away from the welds (the mouth: 2·gap)
    ramp: float = 0.04                # distance from a weld over which the panels open to ``gap``
    film: FilmSpec = LLDPE_30UM

    @property
    def top(self) -> float:
        return self.height + self.handle_h


@dataclass(frozen=True)
class CanSpec:
    radius: float = 0.033             # ESTIMATE: 12 oz can body Ø 66 mm
    height: float = 0.122             # ESTIMATE: 122 mm
    mass: float = 0.014               # ESTIMATE: empty 12 oz aluminium can, typically 13-15 g

    def inertia(self) -> tuple[tuple[float, float, float], ...]:
        """Thin cylindrical shell plus two end discs, mass split by area."""
        r, h, m = self.radius, self.height, self.mass
        a_side, a_end = 2.0 * math.pi * r * h, math.pi * r * r
        ms, me = m * a_side / (a_side + 2 * a_end), m * a_end / (a_side + 2 * a_end)
        iz = ms * r * r + 2 * me * r * r / 2.0
        ix = ms * (r * r / 2.0 + h * h / 12.0) + 2 * me * (r * r / 4.0 + h * h / 4.0)
        return ((ix, 0.0, 0.0), (0.0, ix, 0.0), (0.0, 0.0, iz))


@dataclass(frozen=True)
class DepositRig:
    bag: BagSpec = BagSpec()
    can: CanSpec = CanSpec()
    cans: int = 3
    can_lift: float = 0.02            # m between a can's bottom and the mouth at release
    grip_n: float = 20.0              # N per pinch head
    pad_kp: float = 1.0e4             # N/m pad servo
    pad_mass: float = 0.1             # kg per pad
    link_mass: float = 1.0            # kg per gantry link
    drive_ke: float = 4.0e4           # N/m gantry servos
    dt: float = 1.0e-3                # s, exchange step
    control_dt: float = 0.01
    iterations: int = 10
    film_substeps: int = 1
    film_config: FilmConfig = field(default_factory=FilmConfig)

    @property
    def z_bag(self) -> float:
        """Height of the bag's bottom weld (resting on the counter at z = 0)."""
        return self.bag.film.extras.contact_radius

    @property
    def band_z(self) -> float:
        return self.z_bag + self.bag.top - 0.5 * self.bag.band

    def can_x(self) -> list[float]:
        """Release x of each can: centred on the mouth, one diameter + 2 mm apart."""
        step = 2.0 * self.can.radius + 2.0e-3
        return [(k - 0.5 * (self.cans - 1)) * step for k in range(self.cans)]


def _weld(s: BagSpec, x: float, y: float) -> bool:
    e = 1e-9
    strap = y > s.height + e
    band = strap and y >= s.top - s.band - e
    return x <= e or x >= s.width - e or y <= e or band


def _weld_distance(s: BagSpec, x: float, y: float) -> float:
    d = min(x, s.width - x, y)
    if y > s.height:
        d = min(d, s.top - s.band - y)
    return max(d, 0.0)


def _inside(s: BagSpec, x: float, y: float) -> bool:
    if y <= s.height + 1e-9:
        return True
    return x <= s.handle_w + 1e-9 or x >= s.width - s.handle_w - 1e-9


def _inside_vertex(s: BagSpec, x: float, y: float) -> bool:
    return y <= s.height + 1e-9 or x <= s.handle_w + 1e-9 or x >= s.width - s.handle_w - 1e-9


def _diag_ac(i: int, j: int, quad: tuple[int, int, int, int], welded: set[int]) -> bool:
    """Alternating diagonals, except never between two weld vertices: both panels would share that edge and it
    would have four triangles (non-manifold; C16's bag_mesh rule)."""
    a, b, c, d = quad
    ac = (i + j) % 2 == 0
    if ac and a in welded and c in welded:
        return False
    if not ac and b in welded and d in welded:
        return True
    return ac


def bag_mesh(s: BagSpec) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(vertices [N, 3], triangles [T, 3], groups) in the bag frame: bottom centre at the origin, x across, y the
    panel normal, z up. Groups: ``band_l`` / ``band_r`` (grip-band vertices of the -x / +x strap), ``body`` (below the
    mouth), ``mouth`` (open edge between the straps)."""
    for name in ("width", "height", "handle_w", "handle_h", "band"):
        n = getattr(s, name) / s.cell
        if abs(n - round(n)) > 1e-6:
            raise ValueError(f"BagSpec.cell {s.cell} must divide {name} {getattr(s, name)}")
    nx, ny = int(round(s.width / s.cell)), int(round(s.top / s.cell))
    xs, ys = np.linspace(0.0, s.width, nx + 1), np.linspace(0.0, s.top, ny + 1)
    index: dict[tuple[int, int, int], int] = {}
    verts: list[tuple[float, float, float]] = []

    def vid(i: int, j: int, side: int) -> int:
        w = _weld(s, xs[i], ys[j])
        key = (i, j, 0 if w else side)
        if key not in index:
            off = 0.0 if w else side * s.gap * min(1.0, _weld_distance(s, xs[i], ys[j]) / s.ramp)
            index[key] = len(verts)
            verts.append((xs[i] - 0.5 * s.width, off, ys[j]))
        return index[key]

    tris: list[tuple[int, int, int]] = []
    welded = {vid(i, j, 1) for i in range(nx + 1) for j in range(ny + 1) if _weld(s, xs[i], ys[j])
              and _inside_vertex(s, xs[i], ys[j])}
    for side in (+1, -1):
        for i in range(nx):
            for j in range(ny):
                cy = 0.5 * (ys[j] + ys[j + 1])
                if not _inside(s, 0.5 * (xs[i] + xs[i + 1]), cy):
                    continue
                if side < 0 and cy > s.top - s.band:
                    continue                    # the welded band is one sheet: its cells once, not twice
                a, b, c, d = vid(i, j, side), vid(i + 1, j, side), vid(i + 1, j + 1, side), vid(i, j + 1, side)
                quad = [(a, b, c), (a, c, d)] if _diag_ac(i, j, (a, b, c, d), welded) else [(a, b, d), (b, c, d)]
                tris += quad if side > 0 else [(p, r, q) for p, q, r in quad]
    v, t = np.asarray(verts), np.asarray(tris, np.int64)
    used = np.unique(t)
    if len(used) != len(v):
        raise ValueError("bag_mesh: unreferenced vertices")
    z, x = v[:, 2], v[:, 0]
    band = z >= s.top - s.band - 1e-9
    groups = {"band_l": np.flatnonzero(band & (x < 0)), "band_r": np.flatnonzero(band & (x > 0)),
              "body": np.flatnonzero(z <= s.height + 1e-9),
              "mouth": np.flatnonzero((np.abs(z - s.height) < 1e-9) & (np.abs(x) < 0.5 * s.width - s.handle_w))}
    return v, t, groups


def _jaw_geometry(rig: DepositRig, sign: float) -> tuple[tuple[float, float, float], float]:
    """(pad half sizes, pad centre y) for the jaw on side ``sign``: its face sign·(r - pen) from the band's plane,
    pen the squeeze that carries ``grip_n`` on the band's vertices at the start."""
    s, r = rig.bag, rig.bag.film.extras.contact_radius
    half = (0.5 * s.handle_w + 5.0e-3, 3.0e-3, 0.5 * s.band)
    n_band = (int(round(s.handle_w / s.cell)) + 1) * (int(round(s.band / s.cell)) + 1)
    pen = rig.grip_n / (n_band * CONTACT_KE)
    return half, sign * (r - pen + half[1])


def _pad_cfg() -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU_FILM, ke=CONTACT_KE, has_shape_collision=False)


def _head(b: newton.ModelBuilder, carriage: int, x: float, rig: DepositRig, side: str) -> int:
    """A vise: the +y jaw fixed on the carriage, the -y jaw on a y slider pushed closed with ``grip_n`` (a
    force-limited servo). Two force-limited jaws would float sideways together (measured ±1.5 mm)."""
    half, y = _jaw_geometry(rig, 1.0)
    b.add_shape_box(carriage, xform=body_xform(x, y, 0.0), hx=half[0], hy=half[1], hz=half[2], cfg=_pad_cfg(),
                    label=f"jaw_{side}_fixed")
    half, y = _jaw_geometry(rig, -1.0)
    m, inertia = rig.pad_mass, ((1e-5, 0, 0), (0, 1e-5, 0), (0, 0, 1e-5))
    pad = b.add_link(xform=body_xform(x, y, rig.band_z), mass=m, inertia=inertia, label=f"jaw_{side}")
    j = b.add_joint_prismatic(carriage, pad, axis=(0.0, 1.0, 0.0), parent_xform=body_xform(x, y, 0.0),
                              target_ke=rig.pad_kp, target_kd=2.0 * math.sqrt(rig.pad_kp * m),
                              effort_limit=rig.grip_n, label=f"jaw_{side}_slide")
    b.add_shape_box(pad, hx=half[0], hy=half[1], hz=half[2], cfg=_pad_cfg(), label=f"jaw_{side}")
    return j


def _gantry(b: newton.ModelBuilder, rig: DepositRig) -> None:
    start = body_xform(0.0, 0.0, rig.band_z)
    kd = 2.0 * math.sqrt(rig.drive_ke * 2.0 * rig.link_mass)
    inertia = ((1e-3, 0, 0), (0, 1e-3, 0), (0, 0, 1e-3))
    links, joints = [], []
    for k, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))):
        name = "gantry_" + "xz"[k]
        link = b.add_link(xform=start, mass=rig.link_mass, inertia=inertia, label=name)
        joints.append(b.add_joint_prismatic(links[-1] if links else -1, link, axis=axis,
                                            parent_xform=start if not links else body_xform(0, 0, 0),
                                            target_ke=rig.drive_ke, target_kd=kd, label=name))
        links.append(link)
    hx = 0.5 * (rig.bag.width - rig.bag.handle_w)
    for x, side in ((-hx, "l"), (hx, "r")):
        joints.append(_head(b, links[-1], x, rig, side))
    b.add_articulation(joints, label="gantry")


def _cans(b: newton.ModelBuilder, rig: DepositRig) -> None:
    c = rig.can
    z = rig.z_bag + rig.bag.height + rig.can_lift + 0.5 * c.height
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3, ke=CONTACT_KE)
    for k, x in enumerate(rig.can_x()):
        body = b.add_body(xform=body_xform(x, 0.0, z), mass=c.mass, inertia=c.inertia(), label=f"can{k}")
        b.add_shape_cylinder(body, radius=c.radius, half_height=0.5 * c.height, cfg=cfg, label=f"can{k}")


def build_world(b: newton.ModelBuilder, rig: DepositRig) -> dict[str, np.ndarray]:
    """Counter (static box, top at z = 0), the bag, the gantry with its jaws, the cans. Returns the bag's groups as
    world-local particle ids (the bag is the world's only particle object)."""
    counter = newton.ModelBuilder.ShapeConfig(mu=MU_FILM, ke=CONTACT_KE)
    b.add_shape_box(-1, xform=body_xform(0.1, 0.0, -0.02), hx=0.4, hy=0.25, hz=0.02, cfg=counter, label="counter")
    v, t, groups = bag_mesh(rig.bag)
    add_film_mesh(b, v, t, rig.bag.film, pos=(0.0, 0.0, rig.z_bag), label="bag")
    _gantry(b, rig)
    _cans(b, rig)
    return groups


def model(rig: DepositRig, worlds: int, device: str = "cpu") -> tuple[newton.Model, dict[str, np.ndarray]]:
    from alloy._vendor.newton.solvers import SolverMuJoCo  # noqa: PLC0415

    from alloy.solvers.film.config import register_custom_attributes  # noqa: PLC0415

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMuJoCo.register_custom_attributes(b)
    register_custom_attributes(b)
    groups: dict[str, np.ndarray] = {}
    for _ in range(worlds):
        b.begin_world()
        groups = build_world(b, rig)
        b.end_world()
    b.color()
    m = b.finalize(device=device)
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = CONTACT_KE, 0.0, MU_FILM
    return m, groups


def _split(labels: list[str], key: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    can = tuple(i for i, lab in enumerate(labels) if key in lab.split("/")[-1])
    return tuple(i for i in range(len(labels)) if i not in set(can)), can


def built(m: newton.Model, rig: DepositRig) -> Any:
    """MuJoCo owns the gantry and jaws; the film entry owns every particle and the cans (VBD rigid bodies): a 14 g can
    on the film through a lagged pair is an explicit spring at k·dt²/m ~ 1.4, unstable (stable needs <= 0.02), so the
    can-film contact is solved inside the film's VBD. One lagged pair MuJoCo -> film (the jaws)."""
    from alloy.coupling.types import PairPolicy  # noqa: PLC0415
    from alloy.scene.types import CouplerChoice, EntrySpec  # noqa: PLC0415
    from alloy.testing.integration import built_from_model  # noqa: PLC0415

    gb, cb = _split(list(m.body_label), "can")
    gj, cj = _split(list(m.joint_label), "can")
    mj = EntrySpec("mujoco", "mujoco", gb, (), gj, (), 1, rig.dt, solver_kwargs={"nconmax": 64, "njmax": 256})
    film = EntrySpec("film", "film", cb, tuple(range(m.particle_count)), cj, (), int(rig.film_substeps), rig.dt,
                     max_substeps=int(rig.film_substeps),
                     solver_kwargs={"film": rig.film_config, "iterations": rig.iterations,
                                    "rigid_body_particle_contact_buffer_size": 4096})
    pair = PairPolicy(source="mujoco", destination="film", mode="lagged")
    return built_from_model(m, [mj, film], control_dt=rig.control_dt, coupler=CouplerChoice("proxy", (pair,)))


def simulation(rig: DepositRig, worlds: int = 1, device: str = "cpu", graph: bool = False) -> tuple[Any, dict]:
    """The coupled simulation (alloy.plugins.host.simulation, no plug-ins) and the bag's groups."""
    from alloy.plugins.host import simulation as plugin_simulation  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

    register_builtin_solvers()
    m, groups = model(rig, worlds, device)
    if not graph:
        return plugin_simulation(built(m, rig), []), groups
    from alloy.integration.simulation import IntegratedSimulation  # noqa: PLC0415

    return plugin_simulation(built(m, rig), [], cls=IntegratedSimulation, use_graph=True, cpu_graph=True), groups


__all__ = ["BagSpec", "CanSpec", "DepositRig", "bag_mesh", "build_world", "built", "model", "simulation"]
