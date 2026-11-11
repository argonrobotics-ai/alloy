# SPDX-License-Identifier: Apache-2.0
"""Fit-ready R4 drop-and-bounce twin: WP6's `BounceTwin` with card values per (virtual) condition.

Two restitution paths:
- ``"damping"`` (default, SPEC §3.4): e reaches the solver as per-contact force-space damping written by WP2's
  `PairTable.apply` (kd = 2 zeta(e) sqrt(K m_eff), alloy.scene.pairs) after every collide, from a per-world table
  (one candidate per world). Needs a solver that reads `rigid_contact_stiffness/damping`: SolverMuJoCo with Newton
  contacts (newton@ac7f7830:newton/_src/solvers/mujoco/kernels.py:663-685) or SemiImplicit. `contact_stiffness`
  (per material, series with the ground's, D22) is fittable on this path.
- ``"native"``: `ShapeConfig.restitution` on both shapes, which only XPBD consumes (builder.py:1009-1017).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.restitution import MU
from alloy.bench.solvers import SolverSettings
from alloy.bench.twins.bounce import LAW, RADIUS, BounceTwin
from alloy.bench.types import Condition, RigCard, RigTrial
from alloy.fit.errors import FitError
from alloy.fit.twins.base import FitValues

DEFAULT_K = 2.0e6        # N/m per material (D22, the library default)
GROUND, BALL = 0, 1      # alloy:shape_material ids in the twin's pair table
Path = Literal["damping", "native"]


class DampedPipeline:
    """A Newton CollisionPipeline whose contacts carry per-contact properties, rewritten by `apply` after collide."""

    def __init__(self, inner: newton.CollisionPipeline, model: newton.Model,
                 apply: Callable[[newton.Contacts, Any], None]) -> None:
        self.inner, self.model, self.apply = inner, model, apply

    def contacts(self) -> newton.Contacts:
        return newton.Contacts(self.inner.rigid_contact_max, self.inner.soft_contact_max, device=self.model.device,
                               per_contact_shape_properties=True,
                               requested_attributes=self.model.get_requested_contact_attributes())

    def collide(self, state: Any, contacts: newton.Contacts) -> None:
        self.inner.collide(state, contacts)
        self.apply(contacts, state)


class FitBounceTwin(BounceTwin):
    """R4 twin whose restitution (and, on the damping path, contact stiffness) comes from bound card values."""

    law_id = LAW.id
    settings = {"xpbd": SolverSettings("xpbd", {"enable_restitution": True}),
                # elliptic: under the pyramidal default Newton scales the normal gain by mu^2 (1 + mu^2) / (condim - 1)
                # (newton@ac7f7830:newton/_src/solvers/mujoco/kernels.py:217-220); at R4's mu = 0.01 the contact
                # is ~1e-4 as stiff as K (the ball sinks 14 cm, measured), so the damping path needs the elliptic cone
                "mujoco": SolverSettings("mujoco", {"cone": "elliptic"}, newton_contacts=True),
                "semi_implicit": SolverSettings("semi_implicit")}

    def __init__(self, e: float | None = None, path: Path = "damping", k_ground: float = DEFAULT_K) -> None:
        super().__init__()
        if path not in ("damping", "native"):
            raise FitError(f"unknown restitution path {path!r} (damping | native)")
        self.path: Path = path
        self.k_ground = float(k_ground)
        self.values = FitValues({} if e is None else {"restitution": float(e)})
        self.fit_fields = frozenset({"restitution", "contact_stiffness"} if path == "damping" else {"restitution"})
        self.table: Any = None

    def bind_values(self, cond_id: str, values: Mapping[str, float]) -> None:
        self.values.bind(cond_id, values)

    def fix_values(self, values: Mapping[str, float]) -> None:
        self.values.fix(values)

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        if self.values.fixed is None and not self.values.defaults:
            super().bind_calib(card, calib)
            self.values.defaults["restitution"] = self.e_fit
            return
        self.monitors["e_fixed"] = dict(self.values.fixed or self.values.defaults)

    def initial_values(self, calib_means: Mapping[str, float]) -> dict[str, float]:
        """Closed-form start: e = mean calib restitution (WP6's bind_calib rule); K has no closed form here."""
        e = calib_means.get("restitution")
        return {"restitution": float(e)} if e is not None and np.isfinite(e) else {}

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        if self.path == "damping":
            from alloy.scene.template import register_alloy_attributes  # WP2

            register_alloy_attributes(b)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        e = self.values.get(cond.id, "restitution")
        cfg = newton.ModelBuilder.ShapeConfig(mu=MU, restitution=e if self.path == "native" else 0.0)
        plane = b.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0, cfg=cfg)
        z = RADIUS + cond.params["drop_height"]
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), label="ball")
        ball = b.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        if self.path == "damping":
            mats = b.custom_attributes["alloy:shape_material"].values
            mats[plane], mats[ball] = GROUND, BALL

    def world_values(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-world (e, series K) in world order."""
        e = np.array([self.values.get(c.id, "restitution") for c in self.worlds])
        k_ball = np.array([self.for_k(c.id) for c in self.worlds])
        return e, k_ball * self.k_ground / (k_ball + self.k_ground)

    def for_k(self, cond_id: str) -> float:
        return self.values.for_condition(cond_id).get("contact_stiffness", DEFAULT_K)

    def on_model(self, model: newton.Model) -> None:
        if self.path == "damping":
            self.table = _world_pair_table(model, *self.world_values(), mujoco=self.solver_name == "mujoco")

    def make_pipeline(self, model: newton.Model, solver: str) -> Any:
        inner = super().make_pipeline(model, solver)
        if self.path != "damping":
            return inner
        if inner is None:
            raise FitError(f"R4 damping path needs Newton contacts; solver {solver!r} uses its own")
        from alloy.scene.pairs import apply_pair_table  # WP2

        return DampedPipeline(inner, model, lambda c, s: apply_pair_table(self.table, c, s, precision_external=False))


def _world_pair_table(model: newton.Model, e: np.ndarray, k: np.ndarray, *, mujoco: bool) -> Any:
    """WP2 PairTable over (ground, ball) with per-world e and K (alloy.scene.pairs, randomize.ensure_world_tables)."""
    from alloy.scene import pairs as pt  # WP2
    from alloy.scene.finish import shape_imp
    from alloy.scene.randomize import ensure_world_tables

    table4 = np.zeros((2, 2, 4))
    table4[..., 0] = table4[..., 1] = MU
    stiff = np.full((2, 2), float(np.median(k)))
    built = pt.TableBuild(table4, stiff, [])
    imp = shape_imp(model) if mujoco else None   # MuJoCo rows: kd / sqrt(1 - imp) (derive.contact_damping)
    entries = [_Entry("ground"), _Entry("ball")]
    table = pt.make_pair_table(entries, built, model, mu_mix_rule=pt.MIX_MAX, shape_imp=imp)  # type: ignore[arg-type]
    ensure_world_tables(table, model.world_count)
    wt, ws = table.world_table.numpy(), table.world_stiffness.numpy()
    wt[:, GROUND, BALL, 2] = wt[:, BALL, GROUND, 2] = e
    ws[:, GROUND, BALL] = ws[:, BALL, GROUND] = k
    table.world_table.assign(wt)
    table.world_stiffness.assign(ws)
    return table


class _Entry:
    """The one MaterialEntry field `make_pair_table` reads (`key`); the twin's table is built from numbers."""

    def __init__(self, key: str) -> None:
        self.key = key


__all__ = ["FitBounceTwin", "DampedPipeline", "DEFAULT_K"]
