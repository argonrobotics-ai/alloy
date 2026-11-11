# SPDX-License-Identifier: Apache-2.0
"""Route → emitter dispatch. Every stock capability shares one `RouteEmitter` (SolverCapability.emitter); Alloy
plug-ins (SPH) bring their own emitter and are looked up through the registry by the assembler."""

from __future__ import annotations

from collections.abc import Callable

from alloy._vendor import newton

from alloy.cards.types import ResolvedObject
from alloy.core.errors import AssemblyError
from alloy.route.types import Route, RouteDecision

EmitFn = Callable[[newton.ModelBuilder, ResolvedObject, RouteDecision], None]


def _table() -> dict[Route, EmitFn]:
    from alloy.scene.emitters.chain import emit_chain
    from alloy.scene.emitters.deformable import emit_cloth, emit_mpm, emit_soft
    from alloy.scene.emitters.panels import emit_panels
    from alloy.scene.emitters.rigid import emit_grains, emit_rigid
    from alloy.scene.emitters.rod_vbd import emit_rod_vbd

    return {
        Route.STATIC: emit_rigid, Route.RIGID: emit_rigid, Route.RIGID_COMPLIANT: emit_rigid,
        Route.RIGID_PILE: emit_rigid, Route.GRAIN_RIGID: emit_grains, Route.CHAIN: emit_chain,
        Route.PANELS: emit_panels, Route.CLOTH_VBD: emit_cloth, Route.CLOTH_STYLE3D: emit_cloth,
        Route.SOFT_VBD: emit_soft, Route.GRAIN_MPM: emit_mpm, Route.PASTE_MPM: emit_mpm, Route.LIQUID_MPM: emit_mpm,
        Route.ROD_VBD: emit_rod_vbd,
    }


class RouteEmitter:
    """`Emitter` protocol implementation dispatching on `RouteDecision.route` (public builder methods only)."""

    def __init__(self) -> None:
        self._fns: dict[Route, EmitFn] | None = None

    def emit(self, b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
        if self._fns is None:
            self._fns = _table()
        fn = self._fns.get(d.route)
        if fn is None:
            raise AssemblyError(f"{obj.card.name}: no stock emitter for route {d.route} (LIQUID_SPH is emitted by "
                                "the SPH plug-in)")
        fn(b, obj, d)


ROUTE_EMITTER = RouteEmitter()
