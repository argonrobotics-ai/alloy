# SPDX-License-Identifier: Apache-2.0
"""PluginName: a parametric dumbbell (two spheres on a bar, one rigid body).

TEMPLATE: a complete, passing object plug-in. Replace build() and keep the contract: add one copy to the builder at
``xform``, prefix labels with ``label``, return the new body indices. The harness drops it on the ground and checks
mass, inertia, determinism and that it comes to rest; laws() add your own checks.
Check it with:  python -m alloy.plugins check <this file>
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins import Law, LawResult, ObjectPlugin, body_xform


class PluginName(ObjectPlugin):
    """A dumbbell: two spheres of ``radius`` at the ends of a square bar of ``length`` [m], density [kg/m³].

    Example::

        PluginName(radius=0.03, length=0.2).build(builder, wp.transform_identity(), "dumbbell")
    """

    name = "plugin_name"
    summary = "parametric dumbbell (template)"
    drop_height = 0.1                 # m: the default drop test starts the body frame here

    @dataclass
    class Config:
        radius: float = 0.03          # m, end spheres
        length: float = 0.2           # m, centre to centre
        bar: float = 0.01             # m, bar half-thickness
        density: float = 7800.0       # kg/m³ (steel)

        def __post_init__(self) -> None:
            if min(self.radius, self.length, self.bar, self.density) <= 0 or self.bar >= self.radius:
                raise ValueError("radius, length, bar, density must be > 0 and bar < radius")

    def build(self, builder: newton.ModelBuilder, xform: wp.transform, label: str) -> list[int]:
        c = self.config
        cfg = newton.ModelBuilder.ShapeConfig(density=c.density, mu=0.6)
        body = builder.add_body(xform=xform, label=f"{label}_body")
        for side in (-0.5, 0.5):
            builder.add_shape_sphere(body, xform=body_xform(side * c.length, 0.0, 0.0), radius=c.radius, cfg=cfg)
        builder.add_shape_box(body, hx=0.5 * c.length, hy=c.bar, hz=c.bar, cfg=cfg)
        return [body]

    def laws(self) -> list[Law]:
        return [Law("mass_from_density", _check_mass, None, "mass = density x shape volumes"),
                Law("rests_on_both_spheres", _check_rest, None, "lies level at height = radius")]


def _rows(run) -> np.ndarray:
    return np.nonzero(run.model.body_world.numpy() == 0)[0]


def _check_mass(run) -> LawResult:
    c = run.plugin.config                                     # the object under test
    want = c.density * (2 * 4.0 / 3.0 * math.pi * c.radius**3 + c.length * (2 * c.bar) ** 2)
    got = float(run.model.body_mass.numpy()[_rows(run)].sum())
    return LawResult.within((got - want) / want, 0.01, f"mass {got:.4g} kg vs {want:.4g} kg",
                            "give every shape the material density")


def _check_rest(run) -> LawResult:
    c = run.plugin.config
    q = run.trace.body_q[-1, _rows(run)[0]]
    return LawResult.within(float(q[2]) - c.radius, 0.002, f"rest height {q[2]:.4f} m vs radius {c.radius} m",
                            "spheres at the bar ends, centred on the body frame")
