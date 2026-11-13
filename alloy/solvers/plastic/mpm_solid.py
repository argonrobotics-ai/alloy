# SPDX-License-Identifier: Apache-2.0
"""A ``plastic.model: j2`` solid on Newton's implicit MPM, as a von Mises solid (GAP #6 slice 1, opt-in).

The route is chosen by the object card's route override (``route: paste_mpm``), never by the router itself: without
the override a j2 volume still goes to soft_vbd. Mapping of the card to the per-particle ``mpm:*`` attributes
(``j2_mpm_values``):

* ``young_modulus``, ``poisson_ratio``: the card's.
* ``yield_stress`` = sigma_y / sqrt(3). Newton limits the deviatoric stress norm |r_T| = sqrt(J2) to ``yield_stress``
  (rheology_solver_kernels.py:91-107, ``shear_yield_stress``), and von Mises is sqrt(J2) = sigma_y / sqrt(3)
  (Johnson, Contact Mechanics, ch. 6; ``[internal note]``).
* ``friction`` 0: the shear limit is then constant (no Drucker-Prager pressure term), a pure J2 surface.
* ``tensile_yield_ratio`` 1: Newton's default 0 forbids hydrostatic tension (rheology_solver_kernels.py:84-87), which a
  metal carries.
* ``yield_pressure``: not capped. Newton's default is 1e15 Pa (solver_implicit_mpm.py:1436-1441, the attribute's
  ``default=1.0e15``); Alloy's ``mpm_values`` leaves it at that default unless the card sets one, and 1e15 is
  what ``j2_mpm_values`` returns.
* ``hardening`` 0 (perfect plasticity; Newton's hardening is a Jp law, not a linear modulus, so a card with a
  ``hardening_modulus`` is refused), ``viscosity``, ``dilatancy``, ``damping`` 0.

Measured (law U1, ``[internal note]`` and ``route_slice1.md``: a 12 mm block between a frictionless
floor and a kinematic plate, E 3 MPa, dense grid, CPU, through this route): plateau 9,949 Pa of 10,000 at 1.4 mm/s and
9,934 at 0.7 mm/s (-0.5 % and -0.7 %, 0.15 % apart); elastic slope 3.00 of 3.00 MPa. The tolerance matters: at Alloy's
default 3e-4 the slower press never reached yield (14 % low); 1e-5 with 1,000 iterations read 9,991 Pa (-0.09 %).
A punch needs more than a squeeze (slice 2, ``[internal note]``): a 5 mm ball pressed 1 mm into a
4,000-particle block (E 3 MPa, yield 50 kPa) read 0.84 of its converged force at slice 1's 1e-4 and 250 iterations,
0.99 at ``TOLERANCE`` 1e-5 and ``MAX_ITERATIONS`` 1,000 (1.35x the wall time on a CPU; 1.4-1.5x on an L4 at 1,024
worlds, 3.31 -> 3.86 N in every world). The whole entry runs at this tolerance and cap, and with the Delassus
regularization ``DELASSUS_REGULARIZATION`` (slice 2b; vendored patch 0015): every MPM material beside the
j2 solid pays for it (``install_notes`` says so in ``sim.routes()``). Stiff solids do not converge at any cap a run
can afford: the sweeps a step needs grow as (c dt / h)^2 (``sweeps_needed``), so a real metal reads its force a few %
of the truth (``stiffness_warning``). The entry runs on the dense grid (``grid_type``).
"""

from __future__ import annotations

import math
from typing import Any

from alloy.cards.types import MaterialCard
from alloy.core.errors import AssemblyError

__all__ = ["DELASSUS_REGULARIZATION", "EFFECT", "MAX_ITERATIONS", "NO_PRESSURE_CAP", "SQRT3", "SWEEPS_PER_CFL2",
           "TOLERANCE", "TYPICAL_STEP", "grid_type", "holds_j2", "install_notes", "j2_mpm_values", "stiffness_warning",
           "sweeps_needed"]

SQRT3 = math.sqrt(3.0)
TOLERANCE = 1e-5                 # the MPM rheology tolerance of an entry holding a j2 solid (module docstring)
NO_PRESSURE_CAP = 1.0e15         # Pa: Newton's default mpm:yield_pressure
MAX_ITERATIONS = 1000            # the entry's rheology sweeps per step (Newton's default 250: a punch read 15 % low)
DELASSUS_REGULARIZATION = 1e-12  # the entry's Delassus regularization (Newton's 1e-6 swamps a mass-scaled metal's
#                                  1e-9..1e-7 diagonal: vendored patch 0015; soft cards read the same to 5 digits)
SWEEPS_PER_CFL2 = 0.2            # sweeps per (c dt / h)^2: a rough fit to the converged rows (E 3-300 MPa: 0.05-0.2)
TYPICAL_STEP = 0.00125           # s: the MPM step at a 10 ms or 5 ms control period (1/600 s snapped to the grid)
PUNCH = (" A punch's force is not verified against the indentation laws yet "
         "(docs/known_gaps/plasticity-mpm-solid-punch-force.md).")
GRID_LIMITS = {         # what each grid cannot do, appended to the route note
    "sparse": "On the sparse grid a block resting on the ground drifts (0.5 mm in 0.4 s on a CPU) and a punch pushes "
              "on it from 2 mm above its surface (34 % low at 1 mm deep on an L4).",
    "dense": "Floors thinner than a voxel leak on the dense grid (docs/known_gaps/mpm-sparse-floor-offset.md), so "
             "sand or dough beside the solid in a thin-floored container changes with the solid's grid.",
}
EFFECT = "mpm_j2_solid"          # scene effect of a decision: the entry's solver runs at TOLERANCE


def j2_mpm_values(material: MaterialCard) -> dict[str, float]:
    """``mpm:*`` per-particle values of a j2 solid (module docstring). Example: sigma_y 1e4 Pa gives
    ``yield_stress`` 5773.5. Raises ``AssemblyError`` for a card without elastic or yield data, or with a hardening
    modulus."""
    p, e = material.plastic, material.elastic
    if p is None or p.model != "j2" or p.yield_stress is None or e is None:
        raise AssemblyError(f"material {material.name}: a j2 solid on MPM needs plastic.model j2, plastic.yield_stress "
                         "and an elastic block")
    if p.hardening_modulus is not None and float(p.hardening_modulus.value) != 0.0:
        raise AssemblyError(f"material {material.name}: plastic.hardening_modulus "
                         f"{float(p.hardening_modulus.value):g} Pa is not supported on the MPM route (Newton's "
                         "hardening is another law). Fix: use hardening_modulus 0, or leave the route override off "
                         "(soft_vbd)")
    return {"young_modulus": float(e.youngs.value), "poisson_ratio": float(e.poisson.value),
            "yield_stress": float(p.yield_stress.value) / SQRT3, "friction": 0.0, "tensile_yield_ratio": 1.0,
            "yield_pressure": NO_PRESSURE_CAP, "hardening": 0.0, "viscosity": 0.0, "dilatancy": 0.0,
            "damping": 0.0}


def sweeps_needed(youngs: float, density: float, voxel: float, dt: float = TYPICAL_STEP) -> float:
    """Gauss-Seidel sweeps one MPM step needs to converge a j2 solid: SWEEPS_PER_CFL2 (c dt / h)^2, c = sqrt(E /
    rho) the wave speed, h = voxel / 2 the particle spacing (the fit was made with that h; the grid cell is the
    voxel). Read off WP7 slice 2's punch rows (5 mm ball, voxel 3 mm, 1.25 ms): E 3 MPa (387) converged at 1,000
    sweeps, 30 MPa (3,870) at 4,000, 300 MPa (38,700) reached 0.98 of its force at 16,000; nothing stiffer
    converged, so for brass (97 GPa, 8,500 kg/m3, voxel 3 mm) its 1.6 million extrapolates 3-4 decades past the
    rows: read it as "far over any cap"."""
    h = 0.5 * voxel
    return SWEEPS_PER_CFL2 * (youngs / density) * (dt / h) ** 2


def stiffness_warning(name: str, material: MaterialCard, voxel: float, mass_scale: float = 1.0) -> str | None:
    """The build warning for a j2 solid too stiff for its entry's solve (``sweeps_needed`` over ``MAX_ITERATIONS`` at
    the typical step, at the density times ``mass_scale``: ``quasi_static``), or None."""
    youngs, density = float(material.elastic.youngs.value), float(material.density.value)
    need = sweeps_needed(youngs, density * mass_scale, voxel)
    if need <= MAX_ITERATIONS:
        return None
    shown = float(f"{need:.2g}")                   # two significant figures: an estimate
    scaled = "" if mass_scale == 1.0 else f", with quasi-static mass scaling x{mass_scale:,.0f},"
    return (f"{name}: {material.name} (E {youngs:.3g} Pa, {density:.4g} kg/m3){scaled} is too stiff for implicit "
            f"MPM's stress solve: at a {TYPICAL_STEP * 1e3:g} ms step and {voxel * 1e3:g} mm voxels it needs far "
            f"more sweeps per step than the entry's {MAX_ITERATIONS:,} (about {shown:,.0f} by a rule fitted at E "
            "3-300 MPa), so forces on a tool read low (a brass punch: 3 % of the force at 4,000 sweeps). A tool that "
            "is moved still leaves its dent. Metals are supported on this route only with quasi-static mass scaling. "
            "Fix: for a solid that rests under a tool, scene.set_quasi_static_mass_scale(name, tool_speed=...) "
            "(quasi-static only); else a card with a smaller E / density (about 7,000 m2/s2 converges at 3 mm "
            "voxels), or leave the route override off (docs/known_gaps/plasticity-mpm-solid-punch-force.md)")


# --- the entry that holds a j2 solid ---------------------------------------------------------------------------

def holds_j2(model: Any) -> bool:
    """Whether ``model`` (a Model or ModelView) has a particle with a tensile yield ratio: only a j2 solid sets one."""
    t = getattr(getattr(model, "mpm", None), "tensile_yield_ratio", None)
    return t is not None and bool((t.numpy() > 0.0).any())


def grid_type(view: Any, default: str) -> str:
    """The grid of an entry: ``"dense"`` for a j2 solid unless the user chose one (``ALLOY_MPM_GRID`` or
    ``grid_override``), else ``default``, on any device. The sparse grid on the CPU spreads a block resting on the
    ground by 0.5 mm in 0.4 s (U1.md; docs/known_gaps/mpm-sparse-floor-offset.md), and on an L4 a punch pushed
    on it from 2 mm above its surface and read 34 % low at 1 mm deep, where the dense grid read the CPU's force
    to 0.1 % (WP7 slice 2). Graphs need the sparse grid, so a scene holding a j2 solid runs without them
    (``EFFECT`` is one of ``graph_budget.EAGER_EFFECTS``)."""
    from alloy.integration import mpm_grid as G  # noqa: PLC0415

    return "dense" if holds_j2(view) and not G.chosen() else default


def install_notes(solver: Any, view: Any, grid: str) -> None:
    """The route note of an MPM entry that holds a j2 solid: what its rheology solve runs at, read live from the
    solver, and the grid (M1 of WP12's review: the natural MPM reason names the environment's tolerance, not this)."""
    if not holds_j2(view):
        return

    def notes() -> dict[str, str]:
        reg = getattr(solver, "delassus_regularization", None)          # None: a solver before patch 0015
        named = "" if reg is None else f", Delassus regularization {reg:g}"
        return {"mpm_j2_solid": f"this entry runs at tolerance {solver.tolerance:g}, at most {solver.max_iterations} "
                                f"iterations{named}, {grid} grid, because a j2 solid is in it; every MPM material "
                                "in the entry pays for it (the entry's voxel is also the smallest of its objects'). "
                                + GRID_LIMITS.get(grid, "") + PUNCH}
    from alloy.integration.mpm_solve import attach_notes  # noqa: PLC0415

    attach_notes(solver, "j2", notes)
