# SPDX-License-Identifier: Apache-2.0
"""Liquid laws registered by WP9 (SPEC §5.3): A13 hydrostatics / Archimedes / volume / resting jitter, A14 dam break,
R7 pour twin.

Imported by ``alloy.bench.law.discover`` (CATALOGUE module ``alloy.solvers.sph.laws``); registration happens on
import. Tolerances are SPEC §5.3's amendments (A13: wall force 3 %, draft 5 %, volume loss <= 1 % over 5 s with
H/(2r) >= 50 and draft/(2r) >= 10; A14: Ritter bound, Martin & Moyce NRMSE <= 0.1 once digitised) and bench §4 R7;
A13.resting_jitter is the coordinator's SCR-SC-C2 ruling (settled liquid, default settings, joins SB-12);
A13.sealed_column and A13.piston_displacement are SCR-SC10-1's (seal_twins.py; ambient pressure, ambient.py).
Time steps sit on the D4 grid (0.02 s / 2^k).
"""

from __future__ import annotations

from alloy.bench.law import register_law
from alloy.bench.types import LawSpec, Tolerance

from .pour import POUR_DT, PourTwin
from .seal_twins import PistonDisplacementTwin, SealedColumnTwin
from .twins import ArchimedesTwin, DamBreakTwin, HydrostaticTwin, RestingJitterTwin

HYDRO_DT = 0.02 / 16  # 1.25 ms: resting column (CFL is not active)
DRAFT_DT = 0.02 / 32  # 0.625 ms: explicit fluid-body exchange of the free box (D5 monolithic)
DAM_DT = 0.02 / 256  # 78 us: CFL 0.2 at the Ritter front speed 2 sqrt(g h0) for 2r = 2 mm, h0 = 0.2 m
JITTER_DT = 0.02 / 16  # 1.25 ms: the route's D4 step at r = 2 mm (SPH_CFL 2r / 0.5 m/s = 1.6 ms, snapped down)
SEAL_DT = 0.02 / 512  # 39 us: r = 0.075 mm, CFL 0.2 up to 0.77 m/s, under csf's capillary step (86 us)

A13_HYDROSTATIC = LawSpec(
    id="A13.hydrostatic", suite="analytic",
    metrics={"wall_force": Tolerance(abs=0.03, source="SPEC §5.3 A13 / §5.2 LIQ-4: wall force within 3 % (relative)"),
             "volume_loss": Tolerance(abs=0.01, source="SPEC §5.3 A13: volume loss <= 1 % over 5 s")},
    stat={"wall_force": "mean", "volume_loss": "mean"}, dt=HYDRO_DT, substeps=1, duration_s=5.0,
    worlds_per_condition=1, solvers=("sph",), producer="WP9", quick={"worlds": 1, "duration_s": 0.3},
)

A13_ARCHIMEDES = LawSpec(
    id="A13.archimedes", suite="analytic",
    metrics={"draft": Tolerance(abs=0.05, source="SPEC §5.3 A13 / §5.2 LIQ-4: draft within 5 % (relative), "
                                                  "draft/(2r) >= 10")},
    stat={"draft": "mean"}, dt=DRAFT_DT, substeps=1, duration_s=5.0, worlds_per_condition=1, solvers=("sph",),
    producer="WP9", quick={"worlds": 1, "duration_s": 1.0},  # >= 2 heave periods (~0.2 s) in the averaged half
)

A13_RESTING_JITTER = LawSpec(
    id="A13.resting_jitter", suite="analytic",
    metrics={"rms_speed": Tolerance(abs=3.0e-3, source="SCR-SC-C2 ruling (coordinator 2026-11-03): settled water at "
                                                       "30 / 100 mm depth, default settings, per-particle velocity RMS "
                                                       "<= 3 mm/s after a 2 s settle"),
             "growth": Tolerance(abs=0.1, source="SCR-SC-C2 ruling: no growth over the next 10 s (last-2 s RMS <= "
                                                 "first-2 s RMS x 1.1)")},
    stat={"rms_speed": "mean", "growth": "mean"}, dt=JITTER_DT, substeps=1, duration_s=12.0,
    worlds_per_condition=1, solvers=("sph",), producer="WP9", quick={"worlds": 1, "duration_s": 4.0},  # 2 s settle
)

A14_DAM_BREAK = LawSpec(
    id="A14.dam_break", suite="analytic",
    metrics={"ritter_excess": Tolerance(abs=0.025, source="bench §4 A14: no front faster than Ritter 2 sqrt(g h0); "
                                                          "allowance r/a at a = 20 spacings")},
    stat={"ritter_excess": "mean"}, dt=DAM_DT, substeps=1, duration_s=0.22, worlds_per_condition=1,
    solvers=("sph",), producer="WP9", quick={"worlds": 1, "duration_s": 0.11},
)

_SEAL = "SCR-SC10-1 (coordinator 2026-11-04 18:30Z, WP0 contract and amendment): "

A13_SEALED_COLUMN = LawSpec(
    id="A13.sealed_column", suite="analytic",
    metrics={"pressure": Tolerance(abs=0.10, source=_SEAL + "gauge pressure under the static sealing piston "
                                                            "-rho g H within 10 % (90 deg walls, flat meniscus)"),
             "drift": Tolerance(abs=0.1, source=_SEAL + "column drift < 0.1 particle spacing over 1 s"),
             "void": Tolerance(abs=0.01, source=_SEAL + "no void: liquid fills the bore under the piston (1 %)"),
             "control": Tolerance(abs=0.5, source=_SEAL + "control: unsealed, the column drains as today "
                                                          "(0 = moved >= 1 spacing)")},
    stat={"pressure": "mean", "drift": "mean", "void": "mean", "control": "mean"}, dt=SEAL_DT, substeps=1,
    duration_s=1.0, worlds_per_condition=1, solvers=("sph",), producer="WP9", quick={"worlds": 2, "duration_s": 0.05},
)

A13_PISTON_DISPLACEMENT = LawSpec(
    id="A13.piston_displacement", suite="analytic",
    metrics={"outflow": Tolerance(abs=0.01, source=_SEAL + "a piston moving down at U pushes out the swept volume: "
                                                           "outflow within 1 % of A U t")},
    stat={"outflow": "mean"}, dt=SEAL_DT, substeps=1, duration_s=1.0, worlds_per_condition=1, solvers=("sph",),
    producer="WP9", quick={"worlds": 1, "duration_s": 0.05},
)

R7_POUR = LawSpec(
    id="R7.pour", suite="rig",
    metrics={"onset_angle": Tolerance(abs=1.5, source="bench §4 R7: onset angle +/- 1.5 deg"),
             "m_poured": Tolerance(abs=0.05, source="bench §4 R7: m(theta) NRMSE <= 0.05"),
             "poured_mass": Tolerance(rel=0.05, source="bench §4 R7 / SPEC §5.2 LIQ-7: poured mass 5 %"),
             "spill": Tolerance(abs=0.002, source="bench §4 R7 / SPEC §5.2 LIQ-7: residual and spill +/- 2 g")},
    stat={"onset_angle": "mean", "m_poured": "curve", "poured_mass": "mean", "spill": "mean"}, dt=POUR_DT,
    substeps=1, duration_s=14.0, worlds_per_condition=1, solvers=("sph",), producer="WP9",
    quick={"worlds": 3, "duration_s": 4.0},
)

register_law(A13_HYDROSTATIC, lambda: HydrostaticTwin(HYDRO_DT))
register_law(A13_ARCHIMEDES, lambda: ArchimedesTwin(DRAFT_DT))
register_law(A13_RESTING_JITTER, lambda: RestingJitterTwin(JITTER_DT))
register_law(A14_DAM_BREAK, lambda: DamBreakTwin(DAM_DT))
register_law(R7_POUR, lambda: PourTwin(POUR_DT))
register_law(A13_SEALED_COLUMN, lambda: SealedColumnTwin(SEAL_DT))
register_law(A13_PISTON_DISPLACEMENT, lambda: PistonDisplacementTwin(SEAL_DT))

__all__ = ["A13_HYDROSTATIC", "A13_ARCHIMEDES", "A13_PISTON_DISPLACEMENT", "A13_RESTING_JITTER", "A13_SEALED_COLUMN",
           "A14_DAM_BREAK", "R7_POUR"]
