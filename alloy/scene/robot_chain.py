# SPDX-License-Identifier: Apache-2.0
"""CG for a robot file that is a long chain of light bodies (SAMPLE-CLASSIC C1, WP0 / physics-host1 2026-11-11).

MuJoCo Warp's Newton solver factorises H = M + JᵀDJ in float32. On a ball chain, a necklace or a rope of rigid links
written as nested bodies in a robot file, that factorisation breaks down at the first contact: the world goes
NONFINITE_Q at the landing, on a GPU and on the CPU's Warp route (MuJoCo C, float64, steps it). The SCR-DC1-2 margin
(`alloy.scene.newton_rule`) measures it: eps32·|S·JᵀDJ·S| / λ_min(S·M·S), S = diag(M)^-1/2. A chain makes λ_min small
(its joints' inertias are nearly dependent), and the contacts' curvature grows with the impratio (the elliptic cone's
friction rows), so the rule here screens the robot file before the build:

    CG when λ_min(S·M·S) at the file's pose < LAM_AT_FILL · max(impratio, IMP_FLOOR) / FILL_IMPRATIO

λ_min is unit-free and does not change when every mass is scaled (beads 100x heavier fail too). Measured on the
sample's 8 mm steel beads ([internal note] notes/wp2-chain: margin.py, mjw_beads.jsonl, robots.jsonl), dropped 1 cm, on
MuJoCo Warp (CPU, `ALLOY_CPU_ROUTE=warp`) at the route's impratio 300:
- 5 beads, λ_min 1.22e-3: OK. 6, 7, 8, 10 beads, λ_min 6.1e-4 / 3.5e-4 / 2.2e-4 / 9.9e-5: INVALID at the landing.
- 10 beads at impratio 1: OK; 30 beads (2.6e-6) at impratio 1 go non-finite (WP0, MuJoCo Warp alone).
- Shipped robots, λ_min 0.012 (G1) to 0.25 (Go2, FR3): above the bound by 12x or more, so they keep Newton.
A file that sets `<option solver=...>` keeps its own solver. The screen loads the file in MuJoCo C (MJCF or URDF; a
USD file, or one MuJoCo cannot load, is not screened) in 2-45 ms, once per file and modification time.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from alloy.cards.types import RobotRef
from alloy.scene import mjcf_options as MO
from alloy.scene.entries import FILL_IMPRATIO, ROBOT_CG_EFFECT
from alloy.scene.newton_rule import dof_name, margin
from alloy.scene.template import TemplateObject

log = logging.getLogger("alloy.scene")

LAM_AT_FILL = 1.0e-3        # λ_min(S·M·S) bound at the route's impratio (FILL_IMPRATIO); between 5 and 6 beads
IMP_FLOOR = 10.0            # the bound stops shrinking below this impratio (30 beads fail at impratio 1)
GAP_PAGE = "docs/known_gaps/robot-long-chain-first-contact.md"
_SCREENED = ("mjcf", "urdf")
_MEMO: dict[tuple[str, int, int], Screen | None] = {}


@dataclass(frozen=True)
class Screen:
    lam: float              # λ_min(S·M·S) of the file's mass matrix at its own pose
    dof: str                # the dof with the largest weight in λ_min's eigenvector
    nv: int


def bound(impratio: float) -> float:
    """The λ_min(S·M·S) below which a robot's MuJoCo entry steps on CG, for the entry's impratio."""
    return LAM_AT_FILL * max(float(impratio), IMP_FLOOR) / FILL_IMPRATIO


def screen(path: str) -> Screen | None:
    """λ_min(S·M·S) of the robot file at `path` (MuJoCo C, its own pose), or None when MuJoCo cannot load it or it has
    no dof with mass. Memoised by path, modification time and size."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    if key not in _MEMO:
        _MEMO[key] = _screen(path)
    return _MEMO[key]


def _screen(path: str) -> Screen | None:
    import mujoco  # noqa: PLC0415 - loaded only when a scene has a robot file

    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as exc:  # noqa: BLE001 - a file MuJoCo cannot load is not screened; Newton's importer decides
        log.debug("robot chain screen skipped for %s: %s: %s", path, type(exc).__name__, exc)
        return None
    if m.nv == 0:
        return None
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    if not np.all(np.diag(M) > 0.0):
        return None
    g = margin(M, np.zeros(0))
    return Screen(g.lam, dof_name(m, g.dof), int(m.nv))


def mark(plan, asset: Callable[[str], str]) -> list[TemplateObject]:
    """Add `ROBOT_CG_EFFECT` to each robot in the MuJoCo entry whose file screens below `bound` (in place, with a
    route-report line and a warning); returns `plan.objects`. A file's own `<option solver=...>` stands."""
    robots = [t for t in plan.objects if _screened(t)]
    if not robots:
        return plan.objects
    opts = MO.scene_options(MO.option_files(plan.scene.robots, plan.objects, asset))
    if "solver" in opts.values:
        return plan.objects
    impratio = float(opts.values.get("impratio", FILL_IMPRATIO))
    for t in robots:
        s = screen(asset(t.placement.path))
        if s is None or s.lam >= bound(impratio):
            continue
        t.decision = replace(t.decision, scene_effects=(*t.decision.scene_effects, ROBOT_CG_EFFECT))
        if ROBOT_CG_EFFECT not in plan.report.scene_effects:
            plan.report.scene_effects.append(ROBOT_CG_EFFECT)
        line = report_line(t.name, s, impratio)
        plan.report.entry_options.append(line)
        log.warning("%s", line)
    return plan.objects


def report_line(name: str, s: Screen, impratio: float) -> str:
    return (f"mujoco solver: CG for robot {name}, a long chain of light bodies: lambda_min(S·M·S) = {s.lam:.3g} at the "
            f"file's pose (worst dof {s.dof}, {s.nv} dofs) is below {bound(impratio):.3g} for impratio {impratio:g}, "
            f"where MuJoCo Warp's float32 Newton solver goes non-finite at contact; set <option solver=\"Newton\"/> "
            f"in the file to keep Newton ({GAP_PAGE}, SAMPLE-CLASSIC C1)")


def _screened(t: TemplateObject) -> bool:
    """A robot from a file in the MuJoCo entry, not yet on CG. A robot built in code (``RobotRef`` path "", e.g. the
    UQD halves of ``alloy.tasks.dc_cool_uqd``) has no file to screen: its empty path resolves to the card roots."""
    p = t.placement
    return (isinstance(p, RobotRef) and bool(p.path) and p.kind in _SCREENED and t.decision.entry == "mujoco"
            and ROBOT_CG_EFFECT not in t.decision.scene_effects)


__all__ = ["IMP_FLOOR", "LAM_AT_FILL", "Screen", "bound", "mark", "report_line", "screen"]
