# SPDX-License-Identifier: Apache-2.0
"""Opt-in static hold: noslip over every contact of a MuJoCo entry (coordinator ruling 2026-11-04 01:36Z).

MuJoCo's soft friction is viscous below the cone, so a friction-held part creeps under a steady push. The A5 box
rig on the route's impratio 300 creeps 1.06 / 2.07 um/s at 0.25 / 0.5 mu N. DC3's seated QSFP plug creeps 109 um/s
under the latch push (about 0.25 mu N); physically it should hold.

Alloy's scoped noslip with scope "all" (restart off, `alloy.contact.noslip`) holds both (CPU, 1 world):
- A5 rig: 0.000 um/s;
- QSFP: 0.13 um/s, with the control period +21 % (MuJoCo's own noslip: 0.12 um/s, +15 %).
The scene's robot-scope request (`alloy.scene.noslip_route`) does not reach a robot-less scene's contacts.

A scene opts in through its MuJoCo entry kwargs: `{HOLD_KEY: True}` (50 iterations, tol 1e-10) or
`{HOLD_KEY: (iterations, tolerance)}`; the entry factory pops the key (`alloy.scene.builtin_plugins`). Where it
installs:
- elliptic MuJoCo Warp without sleeping: scoped noslip over all contacts;
- pyramidal cone, sleeping or the MuJoCo C path: MuJoCo's own noslip, which covers every row too;
- rk4 entries (SCR-WP6-4) and the discrete integrator: nothing. They keep the creep, and the request is logged.
The default route stays unchanged (SB-6 fails with scope all, ruling 2026-11-04).

Limit (coordinator ruling 2026-11-04 12:15Z): the hold keeps an already-settled contact state. From a start out of
equilibrium the part slides while the normal forces settle, and noslip stops that slip only at each friction row's
reference damping, B = 2/(dmax tc). On the explicit pair rows' 0.02 s friction reference that is 105 /s, about
9.5 ms. DC3's QSFP start at impratio 300 (pads 0.72 mm into the card, slider off its stop) loads in 87.8 um; started
at the pads' and slider's loaded positions it loads in 3.4 um and creeps 0.024 um/s. Start preloaded parts loaded.
"""

from __future__ import annotations

from typing import Any

from alloy.core.log import get_logger

log = get_logger("contact.noslip_hold")

HOLD_KEY = "alloy:noslip_all"
HOLD_ITERATIONS = 50
HOLD_TOLERANCE = 1e-10


def hold_request(value: Any) -> tuple[int, float] | None:
    """(iterations, tolerance) of an entry's HOLD_KEY value: True -> the defaults, an int or (n[, tol]) -> those;
    None, False or 0 iterations -> None."""
    if value is None or value is False:
        return None
    if value is True:
        return HOLD_ITERATIONS, HOLD_TOLERANCE
    n, tol = (value, HOLD_TOLERANCE) if isinstance(value, int) else (value[0], value[1] if len(value) > 1 else
                                                                        HOLD_TOLERANCE)
    return (int(n), float(tol)) if int(n) > 0 else None


def install_hold(solver: Any, iterations: int = HOLD_ITERATIONS, tolerance: float = HOLD_TOLERANCE) -> str:
    """Install the hold on a built SolverMuJoCo (before any step or graph capture); returns the mode used
    ("scoped-all", "mujoco" or "off")."""
    import mujoco  # noqa: PLC0415

    from alloy.contact.noslip import NoslipSpec, install_noslip  # noqa: PLC0415
    from alloy.core.newton_compat import enable_mujoco_noslip  # noqa: PLC0415

    opt = solver.mj_model.opt
    integ = int(opt.integrator)
    if integ in (int(mujoco.mjtIntegrator.mjINT_RK4), int(mujoco.mjtIntegrator.mjINT_DISCRETE)):
        log.info("noslip scope all requested on a %s entry: not installed, friction-held contacts creep below the "
                 "cone (SCR-WP6-4 rk4 entries keep it)", "rk4" if integ == int(mujoco.mjtIntegrator.mjINT_RK4)
                 else "discrete")
        return "off"
    elliptic = int(opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    sleeping = bool(int(opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
    c_path = getattr(solver, "use_mujoco_cpu", False) or getattr(solver, "mjw_model", None) is None
    if c_path or not elliptic or sleeping:
        enable_mujoco_noslip(solver, iterations, tolerance)
        log.info("static hold: MuJoCo's own noslip %d (tol %g) on every row", iterations, tolerance)
        return "mujoco"
    install_noslip(solver, NoslipSpec(iterations=iterations, tolerance=tolerance, scope="all"))
    log.info("static hold: Alloy scoped noslip %d (tol %g), scope all", iterations, tolerance)
    return "scoped-all"


__all__ = ["HOLD_ITERATIONS", "HOLD_KEY", "HOLD_TOLERANCE", "hold_request", "install_hold"]
