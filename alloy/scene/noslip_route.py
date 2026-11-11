# SPDX-License-Identifier: Apache-2.0
"""Noslip for routed scenes (SCR-WP3-4, SPEC §3.5 creep rule): a scene any of whose MJCF files (robots, and since
SCR-SC6-3 its MJCF object cards: benches, fixtures) sets `<option noslip_iterations="n">` with n > 0 (the T11, T12,
T14, H1 and YAM recordings) runs its MuJoCo entry with noslip; Newton's MJCF importer drops the option, so the
assembler reads it here and the MuJoCo capability factory
(`alloy.scene.builtin_plugins`) installs it right after building `SolverMuJoCo`, before any step or graph capture:

- MuJoCo Warp, elliptic cone, no sleeping: Alloy's scoped noslip, `alloy.contact.noslip.install_noslip(solver,
  NoslipSpec(n, tol, scope="robot"))` (frictional rows that touch a robot tree only);
- pyramidal cone, sleeping on, or the MuJoCo C path: MuJoCo's own, `newton_compat.enable_mujoco_noslip(solver, n, tol)`;
- `discrete` integrator: no noslip (MuJoCo has none there), logged.
The request rides in the entry's solver kwargs under NOSLIP_KEY, which the factory pops before the constructor:
(iterations, tolerance), plus the precision pairs as template shape-id pairs when the fast route profile scopes noslip
to them too (`alloy.scene.fast_profile`, SCR-WP3-8; MuJoCo's own noslip covers every row, those pairs included).
Code that builds SolverMuJoCo itself (coupled clip sims) calls `install_file_noslip(solver, files)` with the MJCF
files it loaded, so a file's own option is never silently dropped.
Where no file asks, a scene with a robot gripper and a free rigid object gets HELD's noslip (100, 1e-6;
`alloy.scene.held_route`, coordinator ruling scr/SC9.md 2026-11-06T18:14Z).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from alloy.cards.types import RobotRef

log = logging.getLogger("alloy.scene")

NOSLIP_KEY = "alloy:noslip"
MJ_NOSLIP_TOLERANCE = 1e-6          # MuJoCo's default <option noslip_tolerance>
HELD_SOURCE = "held_route"          # NoslipRequest.source of HELD's default (no file asked)


@dataclass(frozen=True)
class NoslipRequest:
    iterations: int
    tolerance: float
    source: str                     # the MJCF that asked for it, or held_route
    reason: str = ""                # held_route: why the scene takes HELD's noslip


def mjcf_noslip(path: str) -> NoslipRequest | None:
    """The `<option noslip_iterations/noslip_tolerance>` of an MJCF file, None when it sets none > 0 (or is missing)."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        log.warning("noslip: cannot read %s (%s); no noslip from it", path, exc)
        return None
    best: NoslipRequest | None = None
    for opt in root.iter("option"):
        n = int(float(opt.get("noslip_iterations", "0")))
        if n > 0 and (best is None or n > best.iterations):
            best = NoslipRequest(n, float(opt.get("noslip_tolerance", MJ_NOSLIP_TOLERANCE)), path)
    return best


def files_noslip(paths: Sequence[str]) -> NoslipRequest | None:
    """The request of a set of MJCF files: the most iterations, then the tightest tolerance; None when none asks."""
    reqs = [r for r in (mjcf_noslip(str(p)) for p in paths) if r is not None]
    if not reqs:
        return None
    return max(reqs, key=lambda r: (r.iterations, -r.tolerance))


def scene_noslip(robots: Sequence[RobotRef], asset: Callable[[str], str],
                 objects: Sequence = ()) -> NoslipRequest | None:
    """The scene's request over every MJCF it loads (SCR-SC6-3): its MJCF robots, then its MJCF object cards
    (benches, fixtures), the files `alloy.scene.mjcf_options.option_files` lists for the other `<option>` values.
    Where no file asks: HELD's noslip when a robot grips and a free rigid object is present (`held_route`)."""
    from alloy.scene.held_route import HELD, held_reason
    from alloy.scene.mjcf_options import option_files

    req = files_noslip(option_files(robots, objects, asset))
    if req is not None:
        return req
    why = held_reason(robots, asset, objects)
    return None if why is None else NoslipRequest(HELD.noslip_iterations, HELD.noslip_tolerance, HELD_SOURCE, why)


def held_effects(req: NoslipRequest | None) -> tuple[str, ...]:
    """The scene effects of a request: ``held_route.HELD_EFFECT`` for HELD's default (the MuJoCo entry raises a
    file's impratio below 3), none for a file's request or no request."""
    from alloy.scene.held_route import HELD_EFFECT

    return (HELD_EFFECT,) if req is not None and req.source == HELD_SOURCE else ()


def report_line(req: NoslipRequest) -> str:
    """The route-report line of a file's request, or of HELD's (`RouteReport.entry_options`)."""
    if req.source == HELD_SOURCE:
        return (f"mujoco noslip_iterations={req.iterations} noslip_tolerance={req.tolerance:g}: HELD default, "
                f"{req.reason}; installed by the MuJoCo entry (alloy.scene.held_route)")
    name = Path(req.source).parent.name + "/" + Path(req.source).name
    return (f"mujoco noslip_iterations={req.iterations} noslip_tolerance={req.tolerance:g} from {name}: installed "
            f"by the MuJoCo entry (Newton's MJCF importer drops it; alloy.scene.noslip_route)")


def install_file_noslip(solver: Any, mjcf_files: Sequence[str]) -> str:
    """For code that builds SolverMuJoCo itself instead of through the MuJoCo entry (coupled clip sims): install the
    noslip the MJCF files it loaded ask for, by the route's rule (`install_entry_noslip`), before the first step or
    graph capture. Returns the mode used, "none" when no file asks."""
    req = files_noslip(mjcf_files)
    if req is None:
        return "none"
    mode = install_entry_noslip(solver, req)
    log.info("noslip %d from %s on a directly built SolverMuJoCo: %s", req.iterations, req.source, mode)
    return mode


class _PairShapes(NamedTuple):
    """Template shape ids of a precision pair, as `alloy.contact.noslip.pair_geoms` reads them."""
    shape_a: int
    shape_b: int


def _unpack(req: NoslipRequest | tuple) -> tuple[int, float, tuple[_PairShapes, ...]]:
    if isinstance(req, NoslipRequest):
        return req.iterations, req.tolerance, ()
    pairs = tuple(_PairShapes(int(a), int(b)) for a, b in (req[2] if len(req) > 2 else ()))
    return int(req[0]), float(req[1]), pairs


def install_entry_noslip(solver: Any, req: NoslipRequest | tuple) -> str:
    """Install the request on a built SolverMuJoCo; returns the mode used ("scoped", "mujoco", "off")."""
    import mujoco

    from alloy.contact.noslip import NoslipSpec, install_noslip, pair_geoms
    from alloy.core.newton_compat import enable_mujoco_noslip

    n, tol, pairs = _unpack(req)
    opt = solver.mj_model.opt
    if int(opt.integrator) == int(mujoco.mjtIntegrator.mjINT_DISCRETE):
        log.info("noslip %d requested but the entry runs the discrete integrator: noslip off", n)
        return "off"
    # the MuJoCo C entry keeps an mjw_model too: only use_mujoco_cpu says which path steps
    mjw = None if getattr(solver, "use_mujoco_cpu", False) else getattr(solver, "mjw_model", None)
    elliptic = int(opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    sleeping = bool(int(opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP))
    if mjw is None or not elliptic or sleeping:
        enable_mujoco_noslip(solver, n, tol)
        log.info("noslip %d (tol %g): MuJoCo's own (%s)", n, tol,
                 "C path" if mjw is None else ("pyramidal cone" if not elliptic else "sleeping on"))
        return "mujoco"
    install_noslip(solver, NoslipSpec(iterations=n, tolerance=tol, scope="robot", pairs=pair_geoms(solver, pairs)))
    log.info("noslip %d (tol %g): Alloy scoped noslip, scope robot + %d precision pairs", n, tol, len(pairs))
    return "scoped"


__all__ = ["HELD_SOURCE", "NOSLIP_KEY", "NoslipRequest", "files_noslip", "held_effects", "install_entry_noslip",
           "install_file_noslip", "mjcf_noslip", "report_line", "scene_noslip"]
