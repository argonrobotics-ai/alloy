# SPDX-License-Identifier: Apache-2.0
"""Opt-in fast route profile for precision scenes (SCR-WP3-8, coordinator ruling 2026-11-02; SPEC §3.5 creep rule).

By default a MuJoCo entry with precision pairs holds them with impratio 300 (`entries.PRECISION_IMPRATIO`). The fast
profile, `SceneAssembler(library, router, route_profile="fast")`, uses the creep rule's noslip branch instead
(SCR-WP3-8 item 3): impratio 2 plus Alloy's scoped noslip, 50 iterations at tolerance 1e-10, on the robot rows and
the precision pairs' contacts (`alloy.contact.noslip`, `NoslipSpec.pairs`). If the entry runs CG because of an
equality attachment on a chain (`alloy.scene.attachments`), the profile switches it to MuJoCo Warp's Newton solver
and sets chain joint armature to at least 1e-5 kg·m². Newton factors its Hessian in float32, and a cable dof's pivot
is about armature / H_jj of the diagonal. At 1e-6 that pivot was 2e-6, at float32 rounding for T09's 115 dofs, and
1 of 2,048 seated worlds went non-finite in 2 of WP12's 3 runs and 1 of WP2's 5 (WP12-T09-FAST-NAN; cause isolated
by WP3, wp3-results.md). At 1e-5 it is 2.3e-5. D7 holds at 1e-5 (wp2-results.md).

This is WP3's measured T09 setting with that armature. The plug holds at 0.0000 mm per 10 s. The cable creeps
0.99-1.06 mm per 10 s (0.73-0.94 at armature 1e-6), against 0.13-0.22 on the default route; that is the CHAIN creep
known gap. The seated hold runs about 3 % slower than at 1e-6 (WP2, L4: 5.50x against 5.68x at 2,048 worlds); the
profile's speed rows are WP12's. The default route is unchanged. Every report of this profile's speed must print the
cable creep beside it (`FAST_CAVEAT`, added to the route report's warnings).

The profile does nothing, and logs why, when the scene has no precision pair or its MuJoCo entry runs `discrete`
(MuJoCo runs no noslip there).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from alloy._vendor import newton

from alloy.route.types import Route
from alloy.scene.attachments import CG_EFFECT
from alloy.scene.template import TemplateObject

log = logging.getLogger("alloy.scene")

ROUTE_PROFILES = ("default", "fast")
FAST_EFFECT = "mujoco:profile=fast"
NEWTON_EFFECT = "mujoco:solver=newton"      # replaces the chain attachments' CG (`entries.solver_kwargs`)
FAST_IMPRATIO = 2.0                         # SCR-WP3-8 item 3: impratio >= 1.5 with noslip, 2 when routed
FAST_NOSLIP = (50, 1e-10)                   # iterations, tolerance (WP3's measured setting)
CHAIN_ARMATURE = 1e-5                       # kg·m², chain joint dofs under MuJoCo Warp's Newton solver (pivot margin)
FAST_CAVEAT = ("fast route profile (SCR-WP3-8): impratio 2 + scoped noslip on the precision pairs, chain armature "
               "1e-5. Measured on T09 (L4, 3 s settle + 4 s hold): plug 0.0000 mm per 10 s, but cable creep 0.99-1.06 "
               "mm per 10 s against 0.13-0.22 on the default route (CHAIN known gap, CREEP_GENERAL 0.1 mm). Print the "
               "cable creep beside any speed from this profile. It also fails A5.speed (sliding force -94 % of mu N "
               "at 0.4-4 mm/s): impratio 2 makes sliding friction rate-dependent and noslip does not touch steady "
               "sliding (SCR-WP3-10)")

ShapePairs = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FastChoice:
    on: bool
    reason: str                     # logged
    newton: bool = False            # CG -> MuJoCo Warp Newton, with CHAIN_ARMATURE on the chain dofs
    pairs: ShapePairs = ()          # precision pairs as template shape ids (the noslip scope)

    def effects(self) -> tuple[str, ...]:
        if not self.on:
            return ()
        return (FAST_EFFECT, NEWTON_EFFECT) if self.newton else (FAST_EFFECT,)


OFF = FastChoice(False, "default route profile")


def choose_fast(objs: Sequence[TemplateObject], pairs: ShapePairs, integrator: str) -> FastChoice:
    """Whether the profile applies to the scene's MuJoCo entry. `integrator` is that entry's integrator."""
    mj = [t for t in objs if t.decision.entry == "mujoco"]
    if not mj:
        return FastChoice(False, "fast profile: no MuJoCo entry, route unchanged")
    if not pairs:
        return FastChoice(False, "fast profile: no precision pair (no impratio 300 to trade), route unchanged")
    if integrator == "discrete":
        return FastChoice(False, "fast profile: the MuJoCo entry runs discrete (no noslip), route unchanged")
    newton_ = any(CG_EFFECT in t.decision.scene_effects for t in mj)
    return FastChoice(True, f"fast profile: impratio {FAST_IMPRATIO:g} + scoped noslip on {len(pairs)} precision "
                            f"pairs{'; CG -> Newton + chain armature' if newton_ else ''} (SCR-WP3-8)",
                      newton_, tuple(pairs))


def fast_noslip(scene_request, pairs: ShapePairs) -> tuple[int, float, ShapePairs]:
    """Noslip request for the factory: FAST_NOSLIP, tightened by the scene's own MJCF request if it has one, scoped
    to the robot rows and `pairs` (`noslip_route.install_entry_noslip`)."""
    n, tol = FAST_NOSLIP
    if scene_request is not None:
        n, tol = max(n, scene_request.iterations), min(tol, scene_request.tolerance)
    return n, tol, tuple(pairs)


def raise_chain_armature(b: newton.ModelBuilder, objs: Sequence[TemplateObject],
                         value: float = CHAIN_ARMATURE) -> int:
    """Raise armature to at least `value` on every non-free joint dof of CHAIN objects in the template. Returns the
    number of dofs raised. Same values as WP12's `throughput.chain_armature` applied to the finalized model."""
    raised = 0
    for t in objs:
        if t.decision.route != Route.CHAIN:
            continue
        for j in t.ranges.joint:
            if b.joint_type[j] == newton.JointType.FREE:
                continue
            for d in _joint_dofs(b, j):
                if b.joint_armature[d] < value:
                    b.joint_armature[d] = value
                    raised += 1
    return raised


def _joint_dofs(b: newton.ModelBuilder, j: int) -> range:
    start = b.joint_qd_start[j]
    return range(start, start + sum(b.joint_dof_dim[j]))


__all__ = ["CHAIN_ARMATURE", "FAST_CAVEAT", "FAST_EFFECT", "FAST_IMPRATIO", "FAST_NOSLIP", "NEWTON_EFFECT", "OFF",
           "ROUTE_PROFILES", "FastChoice", "choose_fast", "fast_noslip", "raise_chain_armature"]
