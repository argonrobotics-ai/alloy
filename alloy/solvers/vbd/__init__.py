# SPDX-License-Identifier: Apache-2.0
"""Alloy's VBD path: Newton's SolverVBD with an exact coarse translation step (SCR-WP11-2) on a base that keeps
self-contacting particles out of static shapes (SCR-SC8-3); owner WP11.

`make_vbd` is the `vbd` capability's factory (alloy/scene/builtin_plugins.py). The entry kwarg `coarse_step` picks
SolverVBDMomentum (True) or SolverVBDAlloy (False: the base alone, bitwise stock SolverVBD without self-contact);
without it, `COARSE_DEFAULT` decides. The default is on since
the ruling's acceptance passed ([internal note], 2026-11-04 05:53Z; docs/notes/wp11-vbd-momentum.md §5).
Scope: A17 covers free components; contacting components keep stock VBD (the step is skipped for them).
This package registers no capability of its own: the `vbd` capability (routes cloth_vbd, soft_vbd, rod_vbd) builds
through `make_vbd`, so its adapters, estimator and determinism facts (registered for SolverVBD) cover the subclass.
On import it registers the SB-7 bully target `cloth_box_press` (`press_target`; WP7's cloth class).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton

from .base import SolverVBDAlloy
from .components import ComponentLayout, build_layout
from .contact_list import list_size as contact_list_size
from .reach import SELF_CONTACT_SPEED, SPEED_KEY, with_reach
from .self_contact_rows import pop_rows, with_rows
from .solver import SolverVBDMomentum
from .exit_speed import EXIT_KEY, pop_exit_speed, with_exit_speed
from .stick import STICK_KEY, pop_stick, with_stick

from . import press_target as _press_target  # registers the SB-7 bully target cloth_box_press

log = logging.getLogger("alloy.solvers.vbd")

COARSE_KEY = "coarse_step"
COARSE_CONTACT_KEY = "coarse_contact"    # opt-in (SCR-WP11-6 step 3): touching components take the coarse step too
COARSE_CONTACT_DEFAULT = False
COARSE_DEFAULT = True


PINNED_PIECE_START = True
SELF_CONTACT_FILTER_RINGS = 1          # self contact skips only pairs that share a triangle ring (Newton's default 2)
BODY_CONTACT_LOG_BARRIER = True        # body-particle normal contact on the self-contact log barrier (Newton: off)


def make_vbd(view: Any, kwargs: Mapping[str, Any]) -> newton.solvers.SolverVBD:
    """Build the `vbd` entry's solver: SolverVBDMomentum or SolverVBDAlloy (module docstring).

    The entry starts a pinned piece's iterations from x + h·v (Newton's particle_pinned_piece_start; CARD-8, ENGINE
    item 1): Newton's inertial start settles a pinned stiff sheet 20-30x past its static sag at the entry's 10
    iterations. Other pieces keep the inertial start (exact free fall; a draping piece still falls: XB4's D03 drape
    stayed flat when contact pieces got it). Solvers built directly (SolverPuncture, the bully targets) keep stock.

    With particle self contact on, pairs within one ring of each other are skipped instead of Newton's two
    (``particle_topological_contact_filter_threshold``, laundry CT-13): a squeezed cloth ridge folds sharper than two
    cells, and the elements two rings apart across its tip passed through each other untested (C7's pinch rig: 2-5
    crossings, 0 with one ring; docs/notes/laundry-results.md). At rest those pairs are a cell apart, far beyond the
    contact distance.

    With ``alloy_self_contact_speed`` (the route's `vbd_contact` sets it), the self-contact gap, Newton's detection-only
    slack beyond the margin, comes from that motion bound at each step's dt (`reach`). The gap Newton's truncation
    needs to let a particle move at the bound for one step is gap = max(0, v dt / 0.85 - margin). With gap = margin
    (the route before 2026-11-08) a 0.6 mm towel fell at a constant 0.82 m/s at 1.25 ms. An explicit
    ``particle_self_contact_gap`` wins.

    Body-particle normal contact uses the C2 log barrier that self contact already uses
    (``rigid_soft_contact_use_log_barrier``; Newton's default is the legacy linear penalty), so a rigid face pressed
    on thin cloth cannot push a particle through it (sewing ply pinch, WP3, coordinator 2026-11-06): pads pressed at
    50-100 N on two 0.61 mm muslin plies latched PEN_FAIL with the penalty and hold with the barrier
    (docs/notes/sewing-ply-pinch.md). An explicit kwarg wins.

    An entry whose model has Newton ROD joints (the ROD_VBD route, SCR-WP9-ROD) gets the chosen class with
    `alloy.solvers.rod.SolverVBDRod` mixed in front, which solves each rod chain directly. Its overrides are
    rigid-side only, so particles step as before."""
    kw = dict(kwargs)
    kw.setdefault("particle_pinned_piece_start", PINNED_PIECE_START)
    kw.setdefault("rigid_soft_contact_use_log_barrier", BODY_CONTACT_LOG_BARRIER)
    kw.setdefault("rigid_body_particle_contact_buffer_size", contact_list_size(view))   # SCR-WP10-6
    if kw.get("particle_enable_self_contact"):
        kw.setdefault("particle_topological_contact_filter_threshold", SELF_CONTACT_FILTER_RINGS)
    coarse = bool(kw.pop(COARSE_KEY, COARSE_DEFAULT))
    contact = bool(kw.pop(COARSE_CONTACT_KEY, COARSE_CONTACT_DEFAULT))
    if contact and not coarse:
        raise ValueError(f"{COARSE_CONTACT_KEY}=True needs the momentum coarse step ({COARSE_KEY}=True)")
    if contact:
        kw[COARSE_CONTACT_KEY] = True
    speed = _reach_speed(kw)
    stick = pop_stick(kw)                  # stick friction for held cloth (opt-in; stick.py)
    exit_speed = pop_exit_speed(kw)        # cap on the speed a contact's depth gives (opt-in; exit_speed.py)
    cls = SolverVBDMomentum if coarse else SolverVBDAlloy
    from alloy.solvers.rod import has_rod_chains, with_rod_solve  # noqa: PLC0415 - ROD_VBD entries only

    if has_rod_chains(view):
        cls = with_rod_solve(cls)
    if pop_rows(kw):                       # rows sized to the scene (opt-in); behind the reach mixin: it sets the radius
        cls = with_rows(cls)
    if speed is not None:
        cls, kw[SPEED_KEY] = with_reach(cls), speed
    if exit_speed is not None:
        cls = with_exit_speed(cls, exit_speed)
    if stick:
        cls = with_stick(cls)
    return cls(view, **kw)


def _reach_speed(kw: dict[str, Any]) -> float | None:
    """Pop the self-contact motion bound (`reach`); None when self contact is off or the gap is given explicitly.
    With the bound, the gap starts at 0 and `ReachMixin` sets it from the first step's dt."""
    speed = kw.pop(SPEED_KEY, None)
    if speed is None or not kw.get("particle_enable_self_contact"):
        return None
    if kw.get("particle_self_contact_gap") is not None:
        log.debug("make_vbd: explicit particle_self_contact_gap %s; the motion bound %s is not used",
                  kw["particle_self_contact_gap"], speed)
        return None
    kw["particle_self_contact_gap"] = 0.0
    return float(speed)


_press_target.register()

__all__ = ["BODY_CONTACT_LOG_BARRIER", "COARSE_CONTACT_DEFAULT", "COARSE_CONTACT_KEY", "COARSE_DEFAULT", "COARSE_KEY",
           "EXIT_KEY",
           "PINNED_PIECE_START", "SELF_CONTACT_FILTER_RINGS", "SELF_CONTACT_SPEED", "SPEED_KEY", "STICK_KEY",
           "ComponentLayout", "SolverVBDAlloy", "SolverVBDMomentum", "build_layout", "make_vbd"]
