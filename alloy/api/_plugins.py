# SPDX-License-Identifier: Apache-2.0
"""Plug-ins inside the API's simulation: ``alloy.build(scene, plugins=[...])`` and the scene's heat.

``alloy.plugins.simulation(built, plugins)`` builds its own simulation, which an ``alloy.Sim`` cannot wrap. Here the
plug-in host is installed on the stepper that ``alloy.integration.wiring.wire`` built for the profile, before the
``IntegratedSimulation`` exists (it snapshots the stepper's state arrays when it is made, so ``reset`` and health
rollbacks then restore plug-in state too). The construction mirrors ``wiring.build_simulation``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from alloy.api.errors import AlloyAPIError, BuildError

log = logging.getLogger("alloy.api")
_FIX = ("change the setting the message names: heat via scene.set_heat / set_heater / set_temperature, your own "
        "plug-ins via their config")


def build_engine(built: Any, profile: str, *, seed: int, plugins: Sequence[Any] = (), **overrides: Any) -> Any:
    """The engine simulation of ``built``; ``plugins`` are plug-in objects or functions of the built scene that
    return one (``alloy.plugins``). Without plug-ins this is ``wiring.build_simulation``. With plug-ins and nothing
    dynamic (a film on a static plate) the scene steps the plug-ins alone (``alloy.api._static``).

    Example: ``engine = build_engine(built, "standard", seed=0, plugins=[lambda b: Drag(body="cube", c=0.1)])``."""
    from alloy.integration import wiring  # noqa: PLC0415

    if not plugins:
        return wiring.build_simulation(built, profile, seed=seed, **overrides)
    from alloy.integration.simulation import IntegratedSimulation  # noqa: PLC0415
    from alloy.plugins.errors import PluginError  # noqa: PLC0415
    from alloy.plugins.host import PluginHost  # noqa: PLC0415

    from alloy.api._static import stepper_for  # noqa: PLC0415

    p = wiring.profile(profile, **overrides)
    parts = wiring.wire(built, p, stepper=stepper_for(built))       # plug-ins only (a film on a static plate)
    try:
        made = [_make(x, parts.built) for x in plugins]
        host = PluginHost(parts.built.model, made, control_dt=parts.built.control_dt)
        host.install(parts.stepper)
    except AlloyAPIError:
        raise
    except (PluginError, ValueError, TypeError) as exc:
        raise BuildError(f"plug-in setup failed: {exc}", _FIX, field="plugins") from exc
    use_graph, cpu_graph = _graphs(wiring, parts.built, p)
    sim = IntegratedSimulation(parts.built, precision=parts.precision, contact_layer=parts.contact_layer,
                               gap_monitor=parts.gap_monitor, health=parts.health, topology=parts.topology,
                               stepper=parts.stepper, use_graph=use_graph, cpu_graph=cpu_graph, seed=seed,
                               contact_adjusters=parts.contact_adjusters)
    sim.parts, sim.plugins = parts, host
    log.info("plug-ins in the simulation: %s", sorted(host.plugins))
    return sim


def check(plugins: Sequence[Any]) -> list[Any]:
    """``plugins`` as a list, each a plug-in or a function of the built scene; raises before anything is built.

    Example: ``check([Magnet(bodies=("a", "b"), moments=(1.0, 1.0))])``."""
    from alloy.plugins.base import Plugin  # noqa: PLC0415

    out = list(plugins)
    bad = [x for x in out if not isinstance(x, Plugin) and not callable(x)]
    if bad:
        raise BuildError(f"plugins {bad!r}: not a plug-in or a function of the built scene that returns one",
                         "pass plug-in objects (alloy.plugins.library, python -m alloy.plugins new ...) or "
                         "functions built -> plug-in", field="plugins")
    return out


def _make(x: Any, built: Any) -> Any:
    from alloy.plugins.base import Plugin  # noqa: PLC0415

    if isinstance(x, Plugin):
        return x
    if callable(x):
        made = x(built)
        if isinstance(made, Plugin):
            return made
        raise TypeError(f"{x!r} returned {type(made).__name__}, not a plug-in")
    raise TypeError(f"{x!r} is not a plug-in or a function of the built scene that returns one")


def _graphs(wiring: Any, built: Any, p: Any) -> tuple[bool, bool]:
    """CUDA / CPU graph use as ``build_simulation`` decides it (scene effects can turn graphs off)."""
    import warp as wp  # noqa: PLC0415

    decide = getattr(wiring, "_scene_graphs", None)
    if decide is not None:
        return decide(built, p)
    return p.wants_graph(wp.get_device(built.model.device).is_cuda)
