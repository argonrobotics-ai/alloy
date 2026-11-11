# SPDX-License-Identifier: Apache-2.0
"""Soil (cohesive granular on implicit MPM) laws: scenes and reference values (docs/notes/soil-design.md).

A soil card is a granular material (``plastic.friction_angle``) with a Mohr-Coulomb cohesion (``plastic.cohesion``),
which the MPM emitter passes to Newton as the pressure-independent ``yield_stress``. The laws here check that mapping
and the flow:

- **L1, free-standing column.** A column of height H stands under its own weight while the stress at its base stays
  under the unconfined compressive strength q_u = 2 c tan(45° + φ/2), i.e. for H < H_c = q_u / (ρ g), and slumps
  above it.
- **L2, axisymmetric dry collapse.** Lube, Huppert, Sparks and Hallworth 2004 (J. Fluid Mech. 508, 175-199,
  abstract): r_inf = r_i (1 + 1.24 a) for a = h_i / r_i < 1.7, r_i (1 + 1.6 a^0.5) above, the grains at rest after
  t_inf = 3 (h_i / g)^0.5.
- **L3, bearing capacity (Prandtl).** A rigid strip footing on φ = 0 soil rests under q_ult = (π + 2) c and sinks
  above it (quasi-2D slab between smooth walls, as CARD-9; the footing is a MuJoCo body coupled to the MPM).

Scenes are 1 world on a rough static floor (Coulomb μ = tan φ), built from code cards (``alloy.testing.route``), so a
law can sweep φ and c without library cards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

G = 9.81
RHO = 2650.0                 # kg/m^3, grain density of the law scenes (bulk 0.6 x this, mpm_density)
VOXEL = 5.0e-3               # the router's mpm_voxel_max, which these scenes select


@dataclass(frozen=True)
class Soil:
    """Mohr-Coulomb soil of the law scenes."""

    phi_deg: float = 30.0
    cohesion: float = 0.0    # Pa
    rho: float = RHO         # grain density, the card's

    @property
    def bulk(self) -> float:
        """Continuum density the MPM fills at (``alloy.scene.emitters.deformable.mpm_density``)."""
        from alloy.route.checks import GRAIN_PACKING  # noqa: PLC0415

        return GRAIN_PACKING * self.rho

    def material(self) -> Any:
        """Code card: friction angle and, if positive, ``plastic.cohesion`` (the test helper has no cohesion)."""
        import dataclasses  # noqa: PLC0415

        from alloy.testing import route as T  # noqa: PLC0415

        mat = T.material("soil", density=self.rho, E=1e7, nu=0.3, friction_angle=math.radians(self.phi_deg))
        if self.cohesion <= 0.0:
            return mat
        return dataclasses.replace(mat, plastic=dataclasses.replace(mat.plastic, cohesion=T.q(self.cohesion)))


def column_hc(soil: Soil) -> float:
    """L1: the height [m] at which a free column's base stress reaches q_u = 2 c tan(45° + φ/2)."""
    qu = 2.0 * soil.cohesion * math.tan(math.radians(45.0 + 0.5 * soil.phi_deg))
    return qu / (soil.bulk * G)


def lube_axisymmetric(a: float) -> float:
    """L2: Lube et al. 2004 normalized runout (r_inf - r_i) / r_i."""
    return 1.24 * a if a < 1.7 else 1.6 * math.sqrt(a)


def _floor(soil: Soil, half: float) -> Any:
    from alloy.testing import route as T  # noqa: PLC0415

    rough = T.material("rough", density=600.0, E=1e10, mu=math.tan(math.radians(soil.phi_deg)))
    return T.box("floor", rough, half, half, 0.01, motion="static")


def column_scene(soil: Soil, height: float, width: float = 0.04) -> tuple[Any, Any]:
    """(card, lib): a square column, ``width`` wide and ``height`` tall, standing on the floor."""
    from alloy.cards.types import ParticleSource  # noqa: PLC0415
    from alloy.testing import route as T  # noqa: PLC0415
    from alloy.testing import scene as S  # noqa: PLC0415

    w = 0.5 * width
    src = ParticleSource(((-w, -w, 0.0), (w, w, height)), 0.5 * VOXEL, grain_diameter=3e-4)
    objs = [_floor(soil, 0.15), T.obj("soil", src, soil.material())]
    return S.scene(objs, [S.place("floor", z=-0.01), S.place("soil", z=0.0005)], worlds=1)


def cylinder_scene(soil: Soil, aspect: float, radius: float) -> tuple[Any, Any]:
    """(card, lib): a cylinder of ``radius`` and height ``aspect * radius`` on the floor (L2)."""
    from alloy.cards.types import Primitive  # noqa: PLC0415
    from alloy.testing import route as T  # noqa: PLC0415
    from alloy.testing import scene as S  # noqa: PLC0415

    h = aspect * radius
    objs = [_floor(soil, 0.4), T.obj("soil", Primitive("cylinder", (radius, 0.5 * h)), soil.material())]
    return S.scene(objs, [S.place("floor", z=-0.01), S.place("soil", z=0.5 * h + 0.0005)], worlds=1)


def prandtl_qult(soil: Soil) -> float:
    """L3: strip footing on a purely cohesive soil, q_ult = (π + 2) c [Pa]."""
    return (math.pi + 2.0) * soil.cohesion


@dataclass(frozen=True)
class Strip:
    """L3 geometry [m]: footing width, bed length and depth; the slab is 2 voxels wide between smooth walls."""

    width: float = 0.04
    length: float = 0.2
    depth: float = 0.06
    wall: float = 0.01


def footing_scene(soil: Soil, pressure: float, g: Strip = Strip()) -> tuple[Any, Any]:
    """(card, lib): a rigid strip footing pressing ``pressure`` [Pa] on a soil bed (module docstring, L3)."""
    from alloy.cards.types import ParticleSource  # noqa: PLC0415
    from alloy.testing import route as T  # noqa: PLC0415
    from alloy.testing import scene as S  # noqa: PLC0415

    w, hz = 2.0 * VOXEL, 0.01                    # slab half-width; footing half-height
    fw = w - 0.5e-3                              # 0.5 mm clearance to each side wall
    rho_f = pressure / (G * 2.0 * hz)            # its weight over its footprint is the pressure
    src = ParticleSource(((-g.length / 2, -w, 0.0), (g.length / 2, w, g.depth)), 0.5 * VOXEL, grain_diameter=3e-4)
    rough = T.material("rough", density=600.0, E=1e10, mu=0.6)
    smooth = T.material("smooth", density=600.0, E=1e10, mu=0.0)
    objs = [T.box("floor", rough, g.length / 2 + g.wall, w + g.wall, g.wall, motion="static"),
            T.box("side_p", smooth, g.length / 2, g.wall, g.depth, motion="static"),
            T.box("side_n", smooth, g.length / 2, g.wall, g.depth, motion="static"),
            T.box("end_p", rough, g.wall, w + g.wall, g.depth, motion="static"),
            T.box("end_n", rough, g.wall, w + g.wall, g.depth, motion="static"),
            T.obj("soil", src, soil.material()),
            T.box("foot", T.material("foot", density=rho_f, E=1e10, mu=0.6), g.width / 2, fw, hz)]
    places = [S.place("floor", z=-g.wall), S.place("side_p", y=w + g.wall, z=g.depth),
              S.place("side_n", y=-w - g.wall, z=g.depth), S.place("end_p", x=g.length / 2 + g.wall, z=g.depth),
              S.place("end_n", x=-g.length / 2 - g.wall, z=g.depth), S.place("soil", z=0.0005),
              S.place("foot", z=g.depth + 0.001 + hz)]
    return S.scene(objs, places, worlds=1)


def footing_sink(card: Any, lib: Any, seconds: float) -> float:
    """L3: how far the footing sinks [m] from the end of the first control period to ``seconds``."""
    from alloy.core.registry import isolated_registry  # noqa: PLC0415
    from alloy.integration import build_simulation  # noqa: PLC0415
    from alloy.testing.integration import assemble  # noqa: PLC0415

    with isolated_registry(empty=True):
        built = assemble(card, lib, "cpu", 1)
        sim = build_simulation(built, "standard", graph="off")
        foot = int(built.handles["foot"].template.body.start)
        sim.step_period()                        # the footing lands on the bed
        z0 = float(sim.state.body_q.numpy()[foot, 2])
        for _ in range(int(round(seconds / built.control_dt)) - 1):
            sim.step_period()
        return z0 - float(sim.state.body_q.numpy()[foot, 2])


def run(card: Any, lib: Any, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build on the CPU (graphs off) and run ``seconds``: (start positions, end positions, end velocities)."""
    from alloy.core.registry import isolated_registry  # noqa: PLC0415
    from alloy.integration import build_simulation  # noqa: PLC0415
    from alloy.testing.integration import assemble  # noqa: PLC0415

    with isolated_registry(empty=True):
        built = assemble(card, lib, "cpu", 1)
        sim = build_simulation(built, "standard", graph="off")
        p0 = sim.state.particle_q.numpy().copy()
        for _ in range(int(round(seconds / built.control_dt))):
            sim.step_period()
        return p0, sim.state.particle_q.numpy().copy(), sim.state.particle_qd.numpy().copy()


def top_drop(p0: np.ndarray, p1: np.ndarray) -> float:
    """L1: fall of the column's top (99th percentile of z) [m]."""
    return float(np.percentile(p0[:, 2], 99) - np.percentile(p1[:, 2], 99))
