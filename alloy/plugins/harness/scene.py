# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws for the plug-in harness, and how a scene becomes a running simulation.

A ``TestScene`` is a world template (a function that adds bodies to a ``newton.ModelBuilder``) plus how to run it.
The harness replicates the template into ``worlds`` worlds, adds a ground plane, wraps the model as an Alloy scene
with one solver entry and runs it in ``alloy.sim.Simulation`` with the plug-ins installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError

if TYPE_CHECKING:
    from alloy.plugins.base import Plugin
    from alloy.plugins.harness.run import Run

log = logging.getLogger("alloy.plugins")

GRAVITY = (0.0, 0.0, -9.81)


@dataclass
class TestScene:
    """One world's template and how to run it.

    Example::

        def build(b: newton.ModelBuilder) -> None:
            box = b.add_body(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()), label="box")
            b.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)

        TestScene(build=build, periods=20)
    """

    __test__ = False                    # not a pytest class

    build: Callable[[newton.ModelBuilder], None]
    worlds: int = 3                     # >= 2: world independence perturbs worlds 1.. and watches world 0
    periods: int = 25                   # control periods per run
    control_dt: float = 0.02            # s
    dt: float = 2.0e-3                  # target exchange step [s]; the stepper picks control_dt / 2^k <= dt
    solver: str = "mujoco"              # registered solver capability ("mujoco", "xpbd", "vbd", ...)
    solver_kwargs: Mapping[str, Any] = field(default_factory=dict)
    ground: bool = True
    ground_mu: float | None = None      # ground friction (MuJoCo mixes pairs by max); None = Newton's default 1.0
    gravity: tuple[float, float, float] = GRAVITY
    companions: Callable[[], Sequence[Plugin]] | None = None   # other plug-ins the scene needs (fresh per run)
    init: Callable[[Any], None] | None = None                  # init(sim) after build: set inputs, velocities
    make: Callable[[Sequence[Plugin], str, int, bool], Any] | None = None
    """Builds the simulation itself, ``make(plugins, device, worlds, graph)``, for scenes the template cannot express
    (e.g. a MuJoCo entry coupled to a film entry); ``build`` is then unused. ``init`` still runs."""

    def __post_init__(self) -> None:
        if self.worlds < 1 or self.periods < 1:
            raise PluginError(f"TestScene needs worlds >= 1 and periods >= 1, got {self.worlds}, {self.periods}")


@dataclass
class LawResult:
    """The outcome of one law: ``passed`` if ``|value| <= bound``. ``fix`` says what to change when it fails.

    Example: ``LawResult.within(energy_drift, 0.02, "relative energy drift over 0.5 s")``.
    """

    passed: bool
    value: float
    bound: float
    detail: str = ""
    fix: str = ""

    @staticmethod
    def within(value: float, bound: float, detail: str = "", fix: str = "") -> LawResult:
        """Pass when ``abs(value) <= bound`` (NaN fails)."""
        v = float(value)
        return LawResult(bool(abs(v) <= bound), v, float(bound), detail, fix)


@dataclass
class Law:
    """A physics law checked on its own scene. ``check(run)`` reads ``run.trace`` and returns a ``LawResult``.
    The run holds a fresh copy of the plug-in under test, with ``config`` overrides applied when given.

    Example::

        Law("terminal_velocity", scene=TestScene(build=drop_box, ground=False, periods=100),
            check=lambda run: LawResult.within(run.trace.body_qd[-1, 0, 2] + m * g / c, 0.01 * m * g / c))
    """

    name: str
    check: Callable[[Run], LawResult]
    scene: TestScene | None = None      # None: the plug-in's test_scene()
    doc: str = ""
    config: Mapping[str, Any] | None = None   # Config overrides for this law's copy of the plug-in under test


def _template(scene: TestScene) -> newton.ModelBuilder:
    b = newton.ModelBuilder(gravity=scene.gravity)
    scene.build(b)
    if b.body_count == 0 and b.particle_count == 0:
        raise PluginError("TestScene.build added no bodies or particles: add the bodies your plug-in acts on")
    return b


def build_model(scene: TestScene, device: str, worlds: int | None = None) -> newton.Model:
    """The scene's model: the template replicated into ``worlds`` worlds (default ``scene.worlds``), plus ground."""
    root = newton.ModelBuilder(gravity=scene.gravity)
    root.replicate(_template(scene), worlds or scene.worlds)
    if scene.ground:
        cfg = None if scene.ground_mu is None else newton.ModelBuilder.ShapeConfig(mu=scene.ground_mu)
        root.add_ground_plane(cfg=cfg)
    return root.finalize(device=device)


def build_simulation(scene: TestScene, plugins: Sequence[Plugin], device: str, worlds: int | None = None, *,
                     graph: bool = False) -> Any:
    """``alloy.sim.Simulation`` of ``scene`` with ``plugins`` (plus the scene's companions) installed; with ``graph``
    an ``IntegratedSimulation`` that captures each control period as a graph and replays it (CPU graphs too)."""
    from alloy.plugins.host import simulation  # noqa: PLC0415
    from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415
    from alloy.testing.integration import built_from_model, entry_spec  # noqa: PLC0415

    register_builtin_solvers()
    extra = list(scene.companions()) if scene.companions is not None else []
    if scene.make is not None:
        sim = scene.make([*extra, *plugins], device, worlds or scene.worlds, graph)
        if scene.init is not None:
            scene.init(sim)
        return sim
    model = build_model(scene, device, worlds)
    entry = entry_spec(model, scene.solver, scene.dt, **dict(scene.solver_kwargs))
    built = built_from_model(model, [entry], control_dt=scene.control_dt)
    if graph:
        from alloy.integration.simulation import IntegratedSimulation  # noqa: PLC0415

        sim = simulation(built, [*extra, *plugins], cls=IntegratedSimulation, use_graph=True, cpu_graph=True)
    else:
        sim = simulation(built, [*extra, *plugins])
    if scene.init is not None:
        scene.init(sim)
    return sim


def body_xform(x: float, y: float, z: float, q: Sequence[float] | None = None) -> wp.transform:
    """``wp.transform`` at (x, y, z) with quaternion ``q`` (x, y, z, w; identity by default)."""
    return wp.transform(wp.vec3(x, y, z), wp.quat(*q) if q is not None else wp.quat_identity())


__all__ = ["GRAVITY", "Law", "LawResult", "TestScene", "body_xform", "build_model", "build_simulation"]
