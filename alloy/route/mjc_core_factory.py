# SPDX-License-Identifier: Apache-2.0
"""The ``mujoco`` entry factory's CPU branch (SCR-WP8-6): build :class:`~alloy.route.mjc_core.MjcCoreSolver` for an
entry that asked for the CPU route (:mod:`alloy.route.mjc_core_config`), or say why not.

Refused here, with one info line naming the reason; those entries are built as before (MuJoCo Warp on CPU):
- a model that is not on the CPU (the GPU route is unchanged; no line is logged for it);
- an entry that sets ``use_mujoco_cpu=True`` itself (it wants Newton's own MuJoCo C step, e.g. ``newton_rule``'s
  probe reading ``solver.mj_data``);
- ``use_mujoco_contacts=False``: Newton's collision pipeline feeds the solver (VBD, cloth and MPM couplers, SDF /
  precision-pair contacts), which the MuJoCo C core does not take;
- sleeping (MuJoCo Warp only in Newton) and speculative contacts;
- kinematic bodies (Newton drives them through its per-step push, which the route does not repeat).

What the branch installs, in the GPU branch's order: the model-level plug-ins (tree weights, pair rows, light-body
rule; box-box routing and pair order are MuJoCo Warp fixes that the C model does not need and they say so), then the
per-step plug-ins as the route's config: the scene's noslip request becomes Alloy's scoped noslip, scope "robot" plus
its precision pairs; the static hold (``alloy:noslip_all``) scope "all"; a pyramidal cone keeps MuJoCo's own noslip on
every row (scoped noslip needs the elliptic cone); rk4 and the discrete integrator run neither (as on the GPU route).
The coordinate carry is not installed (float64 qpos); an explicit request for it is logged and ignored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from alloy._vendor import newton

from alloy.contact.noslip import ALL, ROBOT, NoslipSpec
from alloy.route.mjc_core_config import NCON_MAX, RouteConfig

log = logging.getLogger("alloy.route")


@dataclass(frozen=True)
class EntryRequests:
    """What the factory popped from the entry kwargs before building."""

    noslip: Any = None          # alloy.scene.noslip_route request (NoslipRequest or (n, tol[, pairs]))
    hold: tuple[int, float] | None = None
    carry: bool | None = None
    line: bool = True
    light: bool | None = None
    gains: bool | None = None   # F6 R1: the entry's gains are rk4's (light_body.GAINS_KEY)


def refusal(view: Any, kw: dict[str, Any]) -> str | None:
    """Why this entry cannot take the CPU route (module docstring), or None."""
    if str(view.device) != "cpu":
        return f"model on {view.device}"
    if kw.get("use_mujoco_cpu"):
        return "use_mujoco_cpu=True: the entry asks for Newton's own MuJoCo C step"
    if kw.get("use_mujoco_contacts") is False:
        return "use_mujoco_contacts=False: Newton's collision pipeline (couplers, SDF / precision-pair contacts)"
    if kw.get("enable_sleeping"):
        return "sleeping"
    if kw.get("use_speculative_contacts"):
        return "speculative contacts"
    if view.body_count and np.any(view.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)):
        return "kinematic bodies"
    return None


def build_cpu_route(view: Any, kw: dict[str, Any], req: EntryRequests) -> Any:
    """The route's solver for this entry, or None (refused: build it as before)."""
    why = refusal(view, kw)
    if why is not None:
        if not why.startswith("model on"):
            log.info("CPU route refused (%s): this entry is built as before (MuJoCo Warp on CPU, or Newton's own C "
                     "step when it asks for use_mujoco_cpu)", why)
        return None
    from alloy.contact.light_body import install_light_body
    from alloy.route.mjc_core import MjcCoreSolver
    from alloy.scene.builtin_plugins import _pair_fit, _pair_order, _primitive_boxbox
    from alloy.scene.tree_invweight import install_tree_invweight

    solver = MjcCoreSolver(view, route=RouteConfig(line=bool(req.line), ncon_max=int(kw.get("nconmax") or NCON_MAX)),
                           **kw)
    solver = _pair_fit(install_tree_invweight(_pair_order(_primitive_boxbox(solver, req.line))))
    install_light_body(solver, None if req.light is None else bool(req.light), rk4_gains=req.gains)
    if req.carry:
        log.info("CPU route: coordinate carry requested but not needed (MuJoCo C keeps float64 qpos)")
    solver.route = solver.route.replace(noslip=noslip_spec(solver, req))
    return solver.finalize_route()


def noslip_spec(solver: Any, req: EntryRequests) -> NoslipSpec | None:
    """The route's scoped noslip for the entry's hold or noslip request (module docstring); None = none (MuJoCo's own
    noslip, if any, stays in the model)."""
    from alloy.contact.noslip import pair_geoms
    from alloy.core.newton_compat import enable_mujoco_noslip
    from alloy.scene.noslip_route import _unpack

    if req.hold is None and req.noslip is None:
        return None
    opt = solver.mj_model.opt
    integ = mujoco.mjtIntegrator(int(opt.integrator))
    if integ == mujoco.mjtIntegrator.mjINT_DISCRETE or (req.hold is not None
                                                       and integ == mujoco.mjtIntegrator.mjINT_RK4):
        log.info("CPU route: noslip requested on a %s entry: off", integ.name)
        return None
    n, tol, pairs = (*req.hold, ()) if req.hold is not None else _unpack(req.noslip)
    if int(opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        enable_mujoco_noslip(solver, n, tol)
        log.info("CPU route: noslip %d (tol %g): MuJoCo's own on every row (pyramidal cone)", n, tol)
        return None
    if req.hold is not None:
        return NoslipSpec(iterations=n, tolerance=tol, scope=ALL)
    return NoslipSpec(iterations=n, tolerance=tol, scope=ROBOT, pairs=pair_geoms(solver, pairs))


__all__ = ["EntryRequests", "build_cpu_route", "noslip_spec", "refusal"]
