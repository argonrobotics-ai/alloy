# SPDX-License-Identifier: Apache-2.0
"""Stock Newton solver construction for bench laws (single-solver worlds, not `Simulation`).

Laws state their solver settings as data (`SolverSettings`), so every report shows the kwargs that ran. Defaults
that would contaminate an analytic law are overridden here and documented:
- Featherstone and SemiImplicit default `angular_damping=0.05` (newton@ac7f7830:newton/_src/solvers/featherstone/
  solver_featherstone.py:160, semi_implicit/solver_semi_implicit.py:83); laws run them undamped.
- XPBD defaults to 2 iterations (solvers/xpbd/solver_xpbd.py:135); laws use 10, as Newton's friction tests do.
- MuJoCo uses its own contacts unless `newton_contacts` is set (solvers/mujoco/solver_mujoco.py:4323, 4414).
- VBD needs a coloured builder (solver_vbd.py:1191).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from alloy._vendor import newton

from alloy.core.errors import LawError

log = logging.getLogger("alloy.bench")

BASELINE_SOLVERS: tuple[str, ...] = ("mujoco", "xpbd", "featherstone")
RIGID_SOLVERS: tuple[str, ...] = ("mujoco", "xpbd", "featherstone", "semi_implicit")
TET_SOLVERS: tuple[str, ...] = ("vbd", "xpbd", "semi_implicit")

_DEFAULT_KWARGS: dict[str, dict[str, Any]] = {
    "mujoco": {},
    "xpbd": {"iterations": 10, "angular_damping": 0.0},
    "featherstone": {"angular_damping": 0.0},
    "semi_implicit": {"angular_damping": 0.0},
    "vbd": {"iterations": 20},
}


@dataclass(frozen=True)
class SolverSettings:
    """One solver configuration used by a law. `kwargs` override the bench defaults above."""

    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    mujoco_solimp: tuple[float, float, float, float, float] | None = None  # written to every geom (flat imp, D22)
    newton_contacts: bool = False  # MuJoCo only: use Newton's CollisionPipeline instead of MuJoCo collision

    def merged_kwargs(self) -> dict[str, Any]:
        out = dict(_DEFAULT_KWARGS.get(self.name, {}))
        out.update(self.kwargs)
        if self.name == "mujoco" and self.newton_contacts:
            out["use_mujoco_contacts"] = False
        return out

    def describe(self) -> dict[str, Any]:
        """JSON-friendly record of what ran (goes into report provenance)."""
        return {"solver": self.name, "kwargs": self.merged_kwargs(), "mujoco_solimp": self.mujoco_solimp,
                "newton_contacts": self.uses_pipeline()}

    def uses_pipeline(self) -> bool:
        """True when contacts come from a Newton `CollisionPipeline` (everything except native MuJoCo)."""
        return self.name != "mujoco" or self.newton_contacts


def _solver_class(name: str) -> type:
    S = newton.solvers
    table = {"mujoco": S.SolverMuJoCo, "xpbd": S.SolverXPBD, "featherstone": S.SolverFeatherstone,
             "semi_implicit": S.SolverSemiImplicit, "vbd": S.SolverVBD}
    if name not in table:
        raise LawError(f"no stock bench factory for solver {name!r}; known: {sorted(table)}")
    return table[name]


def prepare_builder(name: str, builder: newton.ModelBuilder) -> None:
    """Register solver custom attributes before any world is added (MuJoCo solimp/solref live there)."""
    if name == "mujoco":
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)


def before_finalize(name: str, builder: newton.ModelBuilder) -> None:
    """Solver-specific builder work after all worlds are added (VBD graph colouring)."""
    if name == "vbd" and builder.particle_count > 0:
        builder.color()


def make_solver(model: newton.Model, settings: SolverSettings) -> newton.solvers.SolverBase:
    """Construct the stock solver; writes the flat MuJoCo solimp first when the law asks for it."""
    if settings.mujoco_solimp is not None:
        if settings.name != "mujoco":
            raise LawError(f"mujoco_solimp given for solver {settings.name!r}")
        attrs = model.mujoco.geom_solimp
        values = attrs.numpy()
        values[:] = settings.mujoco_solimp
        attrs.assign(values)
    kwargs = settings.merged_kwargs()
    log.debug("solver %s kwargs=%s", settings.name, kwargs)
    if settings.name == "mujoco":
        return _routed_mujoco(model, kwargs)
    return _solver_class(settings.name)(model, **kwargs)


def _routed_mujoco(model: newton.Model, kwargs: dict[str, Any]) -> newton.solvers.SolverBase:
    """SolverMuJoCo built as Alloy's route builds it: the `mujoco` capability factory (alloy/scene/builtin_plugins.py
    `_mujoco`; rk4 entries get `SyncedMuJoCo`, which skips Newton's per-step float32 state push, WP2 3e803f3)."""
    try:
        from alloy.scene.builtin_plugins import builtin_capabilities
    except ImportError as exc:  # the bench must still run if the scene package cannot import
        log.warning("mujoco capability factory unavailable, building stock SolverMuJoCo (%s)", exc)
        return newton.solvers.SolverMuJoCo(model, **kwargs)
    cap = next(c for c in builtin_capabilities() if c.name == "mujoco")
    return cap.factory(model, kwargs)


def make_pipeline(model: newton.Model, settings: SolverSettings) -> newton.CollisionPipeline | None:
    """A Newton collision pipeline when the solver consumes Newton contacts, else None."""
    if not settings.uses_pipeline() or model.shape_count == 0:
        return None
    return newton.CollisionPipeline(model)


__all__ = ["SolverSettings", "make_solver", "make_pipeline", "prepare_builder", "before_finalize",
           "BASELINE_SOLVERS", "RIGID_SOLVERS", "TET_SOLVERS"]
