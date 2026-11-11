# SPDX-License-Identifier: Apache-2.0
"""Plug-in base classes: ``ForcePlugin``, ``SensorPlugin``, ``ObjectPlugin`` and ``SolverPlugin``.

A plug-in is a class with a ``name``, a ``Config`` dataclass (its JSON parameters) and a few hooks. Force and sensor
plug-ins run inside the simulation step, once per exchange (solver step):

    apply(state, forces, dt)          ForcePlugin: ADD forces into forces.body_f / forces.particle_f
    <solver step>
    update(state_in, state_out, dt)   ForcePlugin (optional): advance the plug-in's own state
    measure(state_in, state_out, dt)  SensorPlugin: write readings into the plug-in's arrays

Rules every hook follows (the harness checks them):

- Launch Warp kernels only: no ``.numpy()``, no allocation, no Python branching on device values. The step may be
  captured in a CUDA graph and replayed.
- Every array the next step reads is registered with ``add_state`` and is world-major (first dimension = worlds),
  so snapshots, ``reset``, rollbacks and determinism digests cover it.
- Deterministic: each output element is written by exactly one thread (no float ``atomic_add``; ``a[i] += x`` in a
  kernel is an atomic too, write ``a[i] = a[i] + x``).
- World ``w`` reads and writes only world ``w``'s items.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError

if TYPE_CHECKING:
    from alloy.core.registry import SolverCapability
    from alloy.plugins.context import PluginContext
    from alloy.plugins.harness.scene import Law, TestScene

log = logging.getLogger("alloy.plugins")

KINDS = ("force", "sensor", "object", "solver")


def make_config(cls: type, config: Any, params: Mapping[str, Any]) -> Any:
    """Build a ``cls`` dataclass from an instance, a dict (JSON) or keyword params; unknown keys raise with the valid
    names. Example: ``make_config(Magnet.Config, {"moments": [0.1, 0.1]}, {})``."""
    if not dataclasses.is_dataclass(cls):
        raise PluginError(f"{cls.__qualname__} must be a @dataclass so its parameters can be read from JSON")
    if config is None:
        values: dict[str, Any] = {}
    elif isinstance(config, cls):
        values = dataclasses.asdict(config)
    elif isinstance(config, Mapping):
        values = dict(config)
    else:
        raise PluginError(f"config must be a {cls.__qualname__}, a dict or None, got {type(config).__name__}")
    values.update(params)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise PluginError(f"{cls.__qualname__} has no field(s) {unknown}; valid fields: {sorted(names)}")
    try:
        return cls(**values)
    except (TypeError, ValueError) as err:
        raise PluginError(f"{cls.__qualname__}: {err}") from err


class Plugin:
    """Base of every plug-in kind. Subclass ``ForcePlugin``, ``SensorPlugin``, ``ObjectPlugin`` or ``SolverPlugin``.

    Example::

        class Drag(ForcePlugin):
            name = "drag"
            summary = "linear air drag on one body"

            @dataclass
            class Config:
                body: str = "box"
                coefficient: float = 0.5          # N·s/m

            def setup(self, ctx):
                self.body = ctx.bodies(self.config.body)

    ``Drag(coefficient=2.0)``, ``Drag({"coefficient": 2.0})`` and ``Drag(Drag.Config(coefficient=2.0))`` are equal.
    """

    name: ClassVar[str] = ""
    kind: ClassVar[str] = ""
    summary: ClassVar[str] = ""
    speed_budget: ClassVar[float] = 2.0      # harness: step time with the plug-in <= budget x without it

    @dataclass
    class Config:
        """No parameters."""

    def __init__(self, config: Any = None, *, label: str | None = None, **params: Any) -> None:
        if not type(self).name:
            raise PluginError(f"{type(self).__qualname__} needs a class attribute name = '...' (registry key)")
        self.config = make_config(type(self).Config, config, params)
        self.label = label or type(self).name
        self.ctx: PluginContext | None = None
        self._state: dict[str, wp.array] = {}
        self._inputs: dict[str, wp.array] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(label={self.label!r}, config={self.config!r})"

    # --- lifecycle ---------------------------------------------------------------------------------------------

    def setup(self, ctx: PluginContext) -> None:
        """Find bodies, allocate every array (``ctx.zeros``) and register state. Runs once, before the first step."""

    def check_dt(self, dt: float) -> None:
        """Raise ``PluginError`` if this plug-in is unstable at exchange step ``dt`` [s] (say which knob to change).
        Called on the host once per distinct ``dt`` before stepping with it."""

    # --- arrays ------------------------------------------------------------------------------------------------

    def add_state(self, name: str, array: wp.array) -> wp.array:
        """Register ``array`` as state the next step reads: world-major, snapshotted, reset and digested.
        Example: ``self.volume = self.add_state("volume", ctx.zeros(ctx.world_count, wp.float64))``."""
        return self._register(self._state, "state", name, array)

    def add_input(self, name: str, array: wp.array) -> wp.array:
        """Register a world-major command array the user sets between periods with ``set_input``.
        Example: ``self.flow = self.add_input("pump_flow", ctx.zeros(ctx.world_count))``."""
        return self._register(self._inputs, "input", name, array)

    def set_input(self, name: str, values: Any) -> None:
        """Copy host ``values`` (one per world, or a scalar for all worlds) into input ``name``.
        Example: ``sim.plugins["hydraulic"].set_input("pump_flow", 2e-6)``."""
        if name not in self._inputs:
            raise PluginError(f"{self.label}: no input {name!r}; inputs: {sorted(self._inputs)}")
        dst = self._inputs[name]
        src = np.broadcast_to(np.asarray(values, dtype=dst.numpy().dtype), dst.numpy().shape)
        dst.assign(np.ascontiguousarray(src))

    def state_arrays(self) -> dict[str, wp.array]:
        """Every registered state array by name."""
        return dict(self._state)

    def inputs(self) -> dict[str, wp.array]:
        """Every registered input array by name."""
        return dict(self._inputs)

    def read(self) -> dict[str, np.ndarray]:
        """Host copies of the state arrays, ``[W, ...]`` each. Override to add derived values (call ``super()``)."""
        return {k: v.numpy().copy() for k, v in self._state.items()}

    def _register(self, table: dict[str, wp.array], what: str, name: str, array: wp.array) -> wp.array:
        if self.ctx is None:
            raise PluginError(f"{self.label}: add_{what}({name!r}) outside setup(); register arrays in setup(ctx)")
        if not isinstance(array, wp.array):
            raise PluginError(f"{self.label}: {what} {name!r} must be a wp.array (use ctx.zeros), got {type(array)}")
        worlds = self.ctx.world_count
        if array.ndim == 0 or array.shape[0] != worlds:
            raise PluginError(f"{self.label}: {what} {name!r} has shape {array.shape}; its first dimension must be "
                              f"the world count {worlds} (allocate ctx.zeros((ctx.world_count, ...)))")
        if name in table:
            raise PluginError(f"{self.label}: {what} {name!r} registered twice")
        table[name] = array
        return array

    # --- testing -----------------------------------------------------------------------------------------------

    def test_scene(self) -> TestScene:
        """The scene the harness runs this plug-in in (crash, determinism, world independence, speed)."""
        raise PluginError(f"{type(self).__name__} has no test_scene(): return alloy.plugins.TestScene(build=...) "
                          f"with a world template that exercises the plug-in")

    def laws(self) -> list[Law]:
        """Physics laws the harness checks, each with its own scene (``alloy.plugins.Law``)."""
        return []


@dataclass
class Forces:
    """The force buffers force plug-ins ADD into; zeroed by the host before every exchange.

    ``body_f``: ``[body_count]`` ``wp.spatial_vector`` (force xyz [N], torque xyz [N·m]), world frame, at the body's
    centre of mass (Newton's ``State.body_f`` convention). ``particle_f``: ``[particle_count]`` ``wp.vec3`` [N].
    """

    body_f: wp.array
    particle_f: wp.array


class ForcePlugin(Plugin):
    """Task physics: adds forces every exchange (``apply``) and/or advances its own state (``update``), e.g. a
    magnet (forces only), ink deposition (state only) or a hydraulic piston (both). Implement at least one.

    Set ``internal = True`` when every force has an equal and opposite partner inside the scene (magnets, springs
    between bodies): the harness then checks that the net force and net moment of each world are zero."""

    kind: ClassVar[str] = "force"
    internal: ClassVar[bool] = False

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        """Launch kernels that ADD this plug-in's forces into ``forces`` (read ``state``, never write it)."""

    @property
    def applies_forces(self) -> bool:
        """True if the subclass overrides ``apply`` (the host skips force bookkeeping otherwise)."""
        return type(self).apply is not ForcePlugin.apply

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Optional: advance the plug-in's own state after the solver step (``state_out`` is the new state)."""


class SensorPlugin(Plugin):
    """Measures after every exchange into its own arrays; never changes the physics."""

    kind: ClassVar[str] = "sensor"

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        """Launch kernels that write this sensor's readings (registered with ``add_state``) from the step."""
        raise PluginError(f"{type(self).__name__}.measure(state_in, state_out, dt) is not implemented")


class ObjectPlugin(Plugin):
    """Builds a parametric object (bodies, shapes, joints) into one world's ``newton.ModelBuilder``."""

    kind: ClassVar[str] = "object"
    drop_height: ClassVar[float] = 0.2       # m: where the default test scene drops the object's frame

    def build(self, builder: newton.ModelBuilder, xform: wp.transform, label: str) -> list[int]:
        """Add one copy at ``xform``; prefix every body/shape label with ``label``; return the new body indices."""
        raise PluginError(f"{type(self).__name__}.build(builder, xform, label) is not implemented")

    def test_scene(self) -> TestScene:
        """Default: drop one copy from ``drop_height`` onto the ground and let it settle for 1.5 s."""
        from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

        return TestScene(build=lambda b: self.build(b, body_xform(0.0, 0.0, self.drop_height), "object"),
                         periods=75)


class SolverPlugin(Plugin):
    """A new solver (SPEC §3.8): ``capability()`` returns the ``SolverCapability`` that builds it. The harness
    registers it, then runs ``test_scene()`` with ``solver=<capability name>``."""

    kind: ClassVar[str] = "solver"

    def capability(self) -> SolverCapability:
        """The ``alloy.core.registry.SolverCapability`` to register (name, factory, routes, owns, ...)."""
        raise PluginError(f"{type(self).__name__}.capability() is not implemented")


__all__ = ["KINDS", "ForcePlugin", "Forces", "ObjectPlugin", "Plugin", "SensorPlugin", "SolverPlugin",
           "make_config"]
