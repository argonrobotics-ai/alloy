# SPDX-License-Identifier: Apache-2.0
"""PluginName: linear air drag F = -coefficient · v on one body.

TEMPLATE: this file is a complete, passing force plug-in. Replace the physics (the kernel, Config, laws) and keep
the structure: setup() finds bodies and allocates, apply() launches kernels that ADD into forces.body_f, laws()
state what must be true. Check it with:  python -m alloy.plugins check <this file>
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton

from alloy.plugins import ForcePlugin, Forces, Law, LawResult, TestScene, body_xform
from alloy.plugins.funcs import add_wrench


@wp.kernel
def _drag_kernel(body_qd: wp.array[wp.spatial_vector], body: wp.array[wp.int32], coefficient: wp.float32,
                 body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()                                   # one thread per world
    b = body[w]                                    # this world's body
    qd = body_qd[b]
    f = -coefficient * wp.vec3(qd[0], qd[1], qd[2])                      # COM linear velocity -> drag force
    add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0))   # body_f[b] = body_f[b] + wrench


class PluginName(ForcePlugin):
    """Linear air drag on one body: F = -coefficient · v at the centre of mass [N, N·s/m, m/s].

    Example::

        drag = PluginName(body="box", coefficient=10.0)
        sim = alloy.plugins.simulation(built, [drag])
    """

    name = "plugin_name"
    summary = "linear drag on one body (template)"

    @dataclass
    class Config:
        body: str = "box"              # label of the body to drag
        coefficient: float = 10.0      # N·s/m

        def __post_init__(self) -> None:
            if self.coefficient < 0:
                raise ValueError("coefficient must be >= 0 N·s/m")

    def setup(self, ctx) -> None:
        self.body = ctx.bodies(self.config.body)          # int32 [W]: the body in every world

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        wp.launch(_drag_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_qd, self.body, float(self.config.coefficient)],
                  outputs=[forces.body_f], device=self.ctx.device)

    def test_scene(self) -> TestScene:
        """A 1 kg box falling in air (no ground) for 1 s: ten drag time constants m/c."""
        return TestScene(build=_falling_box(self.config.body), ground=False, periods=50)

    def laws(self) -> list[Law]:
        return [Law("terminal_velocity", _check_terminal, None, "falling speed approaches m·g / c")]


def _falling_box(label: str):
    def build(b: newton.ModelBuilder) -> None:
        box = b.add_body(xform=body_xform(0.0, 0.0, 10.0), label=label)
        b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=1000.0))
    return build


def _check_terminal(run) -> LawResult:
    plugin = run.plugin                                      # this run's copy of the plug-in under test
    b = run.bodies(plugin.config.body)[0]                    # world 0's body
    m = float(run.model.body_mass.numpy()[b])
    want = m * 9.81 / plugin.config.coefficient
    got = -float(run.trace.body_qd[-1, b, 2])                # downward speed at the end
    return LawResult.within((got - want) / want, 0.01, f"end speed {got:.4g} m/s vs m·g/c {want:.4g} m/s",
                            "the drag force must be -coefficient x velocity")
