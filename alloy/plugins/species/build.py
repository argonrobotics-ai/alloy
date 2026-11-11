# SPDX-License-Identifier: Apache-2.0
"""``build(scene, plugins, ...)``: ``alloy.build`` with plug-ins running inside the engine's own stepper.

Kept for the species tasks' callers: it is ``alloy.build(scene, plugins=plugins, **kwargs)`` (OSS-3's pass-through,
on the stepper ``alloy.integration.wiring`` wires: SPH with rigid two-way, pours, health). The host goes in before the
simulation snapshots its state arrays, so reset, rollback and graph capture cover the plug-ins' state; the host is
``sim.plugins`` (``sim.plugins["species"]``). Until WP12's ``plugins=`` pass-through (68d2edd53) this module patched
``wiring.make_stepper`` for the length of the call.

Example::

    sim = build(scene, [Species(species=("red", "blue"), regions=...)], device="cpu")
    sim.run(2.0)
    print(sim.plugins.read()["species"]["conservation_error"])
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alloy.plugins.base import Plugin


def build(scene: Any, plugins: Sequence[Plugin], **kwargs: Any) -> Any:
    """``alloy.build(scene, plugins=plugins, **kwargs)`` (module doc); ``sim.plugins`` is their host."""
    import alloy  # noqa: PLC0415

    return alloy.build(scene, plugins=list(plugins), **kwargs)
