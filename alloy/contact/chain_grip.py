# SPDX-License-Identifier: Apache-2.0
"""CHAIN grip: the contact rule for a CHAIN cable beside a moving body in the MuJoCo entry (SCR-WP3-CHAINGRIP).

A CHAIN link weighs about a gram, so its contacts are soft in two ways, and a gripped cable slipped. WP9's R4 rig on
CHAIN (the cable's end held between two sliding pads squeezing 4.2 N, the wrist rolled 0.3 rad over 0.5 s): the end
followed half the roll and lost 92 % of that in a 0.7 s hold, with the friction at 2-3 % of its cone, and the pads sank
1.5 mm into the 3 mm cable ([internal note], scr/WP3.md "CHAINGRIP").
- **Normal.** MuJoCo mixes the two geoms' solref by solmix (1:1 by default). The link's native solref (3.73 ms at the
  1.25 ms step, `alloy.scene.native_gains`) averaged with a robot pad's MJCF default (20 ms) gives 11.9 ms.
- **Friction.** The route holds a CHAIN on the Newton solver at MuJoCo's impratio 1 (``entries.CHAIN_NEWTON_HOLD``:
  MuJoCo Warp's float32 Newton Hessian went non-finite at 30-300 on T09, at 100-300 on this rig). At 1 the friction rows
  are as soft as the normal row, and friction below the cone is viscous: the cable slides back under its own torque.

**The rule.** It covers a CHAIN in the MuJoCo entry that is not a chain attachment (no CG effect from
`alloy.scene.attachments`, no tree weld from `alloy.scene.tree_weld`, not a chain the SCR-DC1-2 rule moved off CG),
beside a non-CHAIN body that moves (a robot, or a dynamic or kinematic object). For such a chain:
- its link shapes get ``mujoco:geom_solmix`` = SOLMIX, so the link's own native solref governs a contact with a geom
  at the default solmix (robot links, objects). Static shapes are near 0 already (`alloy.scene.mujoco_pairs`).
- The entry runs Newton at impratio IMPRATIO on MuJoCo C and WARP_IMPRATIO on MuJoCo Warp, over the hold and over a
  file's lower value (the shipped arm files set 1). The option rides under KEY in the entry kwargs; the MuJoCo
  capability factory merges it (`alloy.scene.builtin_plugins`).
- MuJoCo Warp builds the Newton Hessian in float32. At impratio 300 a gripped chain's comes out indefinite by
  rounding alone (SCR-WP3-CHAINGRIP-GPU, R4 rig: eigenvalues -1.9e-7 .. 98; the links' twist curvature is the 1e-6
  armature floor, the friction rows' float32 error ~6e-6), and float32 Newton does not converge on a cable resting
  beside a robot (the 100-iteration cap in 97 % of periods; damping H did not help: 18-25x on an L4). At 30 it
  converges (1.3-1.6x on an L4). Should its float32 factor still fail, `alloy.precision.mjw_chol` redoes it in
  float64 and, on the entry's model (`mark_pivot_floor`), floors the pivots at PIVOT_FLOOR * nv * eps32 * max diag H,
  not mjMINVAL (which made the step inf).

Every other entry is unchanged, bit for bit.

Measured on the R4 rig, 1 world:
- MuJoCo C: ramp end 1.0000, follow 0.9997 at the end of the 0.7 s hold (was 0.498 / 0.081), and 0.9931 after a 5 s
  hold; pad depth 0.10 mm (was 1.54).
- MuJoCo Warp (an L4) at 30: 0.9976 / 0.9828 (was 0.81 / 0.45 with the solmix alone); WP9's api_speed robot scene
  1.62x / 1.14x / 1.10x at 64 / 1,024 / 4,096 worlds.
- Wall per simulated second: +33 % with a gripped cable (WP9's api_speed robot scene, 0.947 -> 1.257 s). The Panda
  scene with a cable on its table showed no measurable cost (1.53 / 1.47 s).

Known gap: docs/known_gaps/rod-chain-grip-slips.md.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from alloy.route.types import Route

log = logging.getLogger("alloy.contact")

KEY = "alloy:chain_grip"                  # entry kwarg: the rule's solver options (builtin_plugins pops it)
ATTACHED_EFFECT = "mujoco:chain_attached"  # a chain attachment the SCR-DC1-2 rule moved off CG (newton_rule)
SOLMIX = 1000.0                           # the link's solref weight against a geom at MuJoCo's default 1
IMPRATIO = 300.0                          # the route's fill-in (entries.FILL_IMPRATIO), on MuJoCo C
WARP_IMPRATIO = 30.0                      # MuJoCo Warp: float32 Newton converges here (SCR-WP3-CHAINGRIP-GPU)
PIVOT_FLOOR = 10.0                        # MuJoCo Warp: the float64 redo's pivot floor, x nv * eps32 * max diag H
MOVING = ("dynamic", "kinematic")
_CG_EFFECT = "mujoco:solver=cg"           # alloy.scene.attachments.CG_EFFECT (not imported: no scene -> contact cycle)
_TREE_NUMBER = "tree_root"                # alloy.scene.tree_weld.TREE_NUMBER


def attached(t: Any) -> bool:
    """A chain attachment: a weld or joint (CG effect), a tree weld, or one the SCR-DC1-2 rule moved to Newton."""
    d = t.decision
    fx = getattr(d, "scene_effects", ())
    return _CG_EFFECT in fx or ATTACHED_EFFECT in fx or _TREE_NUMBER in getattr(d, "numbers", {})


def gripped(objs: Sequence[Any]) -> list[Any]:
    """The MuJoCo entry's CHAIN objects the rule covers (module docstring); empty when no moving body is beside them."""
    mj = [t for t in objs if _entry(t) == "mujoco"]
    chains = [t for t in mj if t.decision.route == Route.CHAIN and not attached(t)]
    if not chains or partner(mj) is None:
        return []
    return chains


def partner(objs: Sequence[Any]) -> str | None:
    """The first non-CHAIN body in the MuJoCo entry that moves: a robot, or a dynamic or kinematic object."""
    for t in objs:
        if _entry(t) != "mujoco" or t.decision.route == Route.CHAIN:
            continue
        if t.obj is None or getattr(t.obj.card, "motion", "dynamic") in MOVING:
            return getattr(t, "name", "a robot")
    return None


def _entry(t: Any) -> str:
    """The object's entry; "mujoco" when the decision does not say."""
    return getattr(t.decision, "entry", "mujoco")


def reason(near: str) -> str:
    return (f"chain grip beside {near} (SCR-WP3-CHAINGRIP): the links' own contact solref governs (solmix "
            f"{SOLMIX:g}); Newton at impratio {IMPRATIO:g} on MuJoCo C, where a held twist still creeps (known gap "
            f"rod-chain-grip-slips), and {WARP_IMPRATIO:g} with a stick anchor on MuJoCo Warp, where it holds")


def apply_solmix(b: Any, objs: Sequence[Any], report: Any = None) -> int:
    """Write SOLMIX on the covered chains' link shapes and add the rule to their route reasons (and to `report`'s
    decisions, which ``Sim.routes()`` and ``alloy.estimate`` print); returns the shapes written. A builder without
    the ``mujoco:geom_solmix`` attribute (no MuJoCo solver registered) is left alone."""
    chains = gripped(objs)
    attr = getattr(b, "custom_attributes", {}).get("mujoco:geom_solmix")
    if not chains or attr is None:
        return 0
    near, written = partner(objs), 0
    for t in chains:
        for s in t.ranges.shape:
            attr.values[s] = SOLMIX
            written += 1
        t.decision = replace(t.decision, reasons=(*t.decision.reasons, reason(near)))
        if report is not None and t.name in report.decisions:
            report.decisions[t.name] = t.decision
    log.info("chain grip: %d CHAIN shapes beside %s take solmix %g", written, near, SOLMIX)
    return written


def overrides(objs: Sequence[Any], impratio: float, has_precision: bool,
              newton: bool = True) -> dict[str, tuple[object, str]]:
    """The rule's MuJoCo entry option, key -> (value, reason) as `entries.mujoco_overrides` gives them: KEY, the
    impratio both routes run, when the rule covers the entry, the entry's solver is Newton (``newton``; a CG entry, a
    chain attachment's, keeps its own) and its own `impratio` (after the route's other options) is below IMPRATIO; {}
    otherwise, and with precision pairs (they set the impratio themselves)."""
    if has_precision or not newton or impratio >= IMPRATIO or not gripped(objs):
        return {}
    return {KEY: ({"impratio": IMPRATIO}, f"chain grip beside {partner(objs)} (SCR-WP3-CHAINGRIP): Newton at impratio "
                                          f"{IMPRATIO:g} on MuJoCo C, {WARP_IMPRATIO:g} on MuJoCo Warp (known gap "
                                          f"rod-chain-grip-slips)")}


def route_kwargs(kw: dict[str, object], grip: Mapping[str, object] | None) -> dict[str, object]:
    """The entry kwargs a route builds with: ``kw`` with the rule's options (KEY) merged in."""
    if not grip:
        return kw
    log.info("chain grip: %s", dict(grip))
    return {**kw, **grip}


def warp_options(grip: Mapping[str, object] | None, kw: Mapping[str, object]) -> Mapping[str, object] | None:
    """The rule's options for a MuJoCo Warp solver built with ``kw``: impratio max(the entry's own, WARP_IMPRATIO), so
    the rule only raises it, when `alloy.precision.mjw_chol` guards MuJoCo Warp's Newton Cholesky (its pivot floor is
    the safety net there); else None (the entry keeps its own impratio; logged)."""
    if not grip:
        return None
    from alloy.precision import mjw_chol  # noqa: PLC0415

    if mjw_chol.install():
        return {**grip, "impratio": max(float(kw.get("impratio", 1.0)), WARP_IMPRATIO)}
    log.warning("chain grip: mjw_chol is not installed (mujoco_warp moved off its pin), so MuJoCo Warp keeps the "
                "entry's impratio (no pivot floor, no impratio %g; the route reason still names it)", WARP_IMPRATIO)
    return None


def mark_pivot_floor(solver: Any, grip: Mapping[str, object] | None) -> None:
    """On a MuJoCo Warp solver the rule covers (``grip`` from `warp_options`): its model's float64 Cholesky redo floors
    pivots at float32 rounding (`alloy.precision.mjw_chol.set_pivot_floor`, on the model and its Data)."""
    if not grip:
        return
    from alloy.precision import mjw_chol  # noqa: PLC0415

    mjw_chol.set_pivot_floor(solver.mjw_model, PIVOT_FLOOR, solver.mjw_data)
    log.info("chain grip: MuJoCo Warp Newton at impratio %g, pivot floor %g x nv * eps32", WARP_IMPRATIO, PIVOT_FLOOR)


def install_anchor(solver: Any, grip: Mapping[str, object] | None) -> bool:
    """On a MuJoCo Warp solver the rule covers (``grip`` from `warp_options`): the stick anchor
    (`alloy.contact.mjw_stick_anchor`) on the contacts of the rule's link geoms, the ones that carry SOLMIX. A held
    cable then stays with its grip instead of creeping back (2 % per second without it)."""
    if not grip:
        return False
    import numpy as np  # noqa: PLC0415

    from alloy.contact.mjw_stick_anchor import install_stick_anchor  # noqa: PLC0415

    links = np.flatnonzero(np.asarray(solver.mj_model.geom_solmix) == SOLMIX)
    return install_stick_anchor(solver, links)


__all__ = ["ATTACHED_EFFECT", "IMPRATIO", "KEY", "PIVOT_FLOOR", "SOLMIX", "WARP_IMPRATIO", "apply_solmix", "attached",
           "gripped", "install_anchor", "mark_pivot_floor", "overrides", "partner", "reason", "route_kwargs",
           "warp_options"]
