# SPDX-License-Identifier: Apache-2.0
"""A cap on the pull of MuJoCo Warp's elliptic friction rows (RIGID-DEFAULT (b), WP3; known gap rigid-sliding-box-hops).

MuJoCo's soft friction rows ask for a reference acceleration ``aref_t = -b v_slip`` with ``b = 2 / (dmax tau)``
(about 105 /s at the default solref). At metres per second that is hundreds of m/s2, far outside the friction cone,
and the elliptic cone's projection turns the excess into normal force: a sliding box rides up on its contacts and
rocks, more the faster it slides, and more under implicitfast (one contact solve per step) than under rk4. Pitch peak
to peak, a 0.3 m cube pushed at 1.5 mu m g for 2 s beside 8 resting cubes (CPU, Warp route, 5 ms): 3.86 deg under rk4,
19.2 under implicitfast.

The cap clamps the two sliding rows of every elliptic contact together, right after ``make_constraint``:

    aref_t <- aref_t * min(1, cap / |aref_t|)

so the pull stops at ``cap`` (m/s2) instead of growing with the slip speed. Torsion and rolling rows, normal rows and
pyramidal contacts are untouched, and so is every pull under the cap: stiction and slow slides keep MuJoCo's rows bit
for bit. At the default ``CAP_DEFAULT`` (20 m/s2) it bites above about 0.2 m/s of slip. The sliding force stays
Coulomb (A5.speed force / mu N 1.0000; a coast at 2.5 m/s, mu 0.5, decelerates at 1.000 mu g). The same case under
the cap: 0.41 deg under implicitfast, not growing (0.32 over the last 0.5 s). A first form, aref_t / (1 + |aref_t| /
cap) (09e8a110c), softened sub-cap pulls too: 27-cube stacks settled up to 3.40 mm off (768 worlds, R1, CPU), against
2.23 mm clamped, 1.65 mm uncapped and 3.26 mm on today's rk4 (WP7's L4 rows); the median stays 0.19 mm.

Where (coordinator ruling 2026-11-09 16:32Z, RIGID-DEFAULT): the fast rigid mode (``entries.mujoco_overrides`` sets
``CAP_KEY`` under ``RIGID_FAST_EFFECT``) and F6 R1 entries on implicitfast (the MuJoCo factory, ``builtin_plugins``,
takes ``default_cap()`` for an entry with R1's gains flag and no rk4), on MuJoCo Warp. Every other entry, and the
MuJoCo C route, keep MuJoCo's rows.

Installed per solver (``install_friction_cap``): the solver's MuJoCo Warp step runs with ``capped_make_constraint`` in
place of ``constraint.make_constraint`` (``forward.fwd_position`` calls it through the module attribute, so rk4's
stages and graph capture take it too), then MuJoCo Warp's own is put back. Not on the MuJoCo C route.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import warp as wp

log = logging.getLogger(__name__)

CAP_KEY = "alloy:friction_cap"     # MuJoCo entry kwarg (entries.mujoco_overrides): the cap in m/s2, or None / 0 = off
CAP_DEFAULT = 20.0                 # m/s2: 4 mu g at mu 0.5 (module docstring)
ENV = "ALLOY_FRICTION_CAP"         # "0" turns the cap off everywhere (A/B runs)
_vec5 = wp.types.vector(length=5, dtype=wp.float32)


@wp.kernel
def cap_friction_aref(nacon: wp.array[int], dim: wp.array[int], world: wp.array[int], addr: wp.array2d[int],
                      etype: wp.array2d[int], aref: wp.array2d[float], elliptic: int, cap: float):
    """The two sliding rows of each elliptic contact, clamped together to |aref_t| <= cap."""
    c = wp.tid()
    if c >= nacon[0]:
        return
    w = world[c]
    a0 = addr[c, 0]
    if dim[c] < 3 or a0 < 0:
        return
    if etype[w, a0] != elliptic:
        return
    e1 = addr[c, 1]
    e2 = addr[c, 2]
    r = wp.sqrt(aref[w, e1] * aref[w, e1] + aref[w, e2] * aref[w, e2])
    if r <= cap:
        return
    k = cap / r
    aref[w, e1] = aref[w, e1] * k
    aref[w, e2] = aref[w, e2] * k


def launch_cap(d: Any, cap: float) -> None:
    """The cap on MuJoCo Warp data ``d``, after its ``make_constraint``."""
    from alloy._vendor.mujoco_warp._src.types import ConstraintType  # noqa: PLC0415

    wp.launch(cap_friction_aref, dim=d.naconmax, inputs=[d.nacon, d.contact.dim, d.contact.worldid,
              d.contact.efc_address, d.efc.type, d.efc.aref, int(ConstraintType.CONTACT_ELLIPTIC), float(cap)])


def capped_make_constraint(inner: Any, cap: float) -> Any:
    """``inner`` (MuJoCo Warp's ``make_constraint``) followed by the cap."""

    def make_constraint(m: Any, d: Any) -> None:
        inner(m, d)
        launch_cap(d, cap)

    make_constraint._alloy_friction_cap = cap
    return make_constraint


def default_cap() -> float:
    """The cap an R1 entry (implicitfast on rk4's gains, ``integrator_route``) takes when it names none."""
    return CAP_DEFAULT


def cap_value(request: Any) -> float | None:
    """The cap an entry asks for (``CAP_KEY``'s value), or None when it is off (None, 0 or ``ALLOY_FRICTION_CAP=0``)."""
    if request is None or os.environ.get(ENV, "1") == "0":
        return None
    cap = float(request)
    if cap < 0.0:
        raise ValueError(f"{CAP_KEY} must be a cap in m/s2 (>= 0, 0 = off), got {cap}")
    return cap or None


def install_friction_cap(solver: Any, request: Any) -> bool:
    """Cap this solver's elliptic friction rows (``cap_value(request)`` m/s2) on its MuJoCo Warp step. Call before the
    first step or graph capture. False when the cap is off or the solver has no MuJoCo Warp step (MuJoCo C)."""
    cap = cap_value(request)
    if cap is None or getattr(solver, "use_mujoco_cpu", False):
        return False
    orig = getattr(solver, "_mujoco_warp_step", None)
    if orig is None:
        return False
    have = getattr(solver, "_alloy_friction_cap", None)
    if have is not None:
        if have != cap:
            raise RuntimeError(f"this solver's friction rows are already capped at {have} m/s2, not {cap}")
        return True
    from alloy._vendor.mujoco_warp._src import constraint  # noqa: PLC0415

    def step(*args: Any, **kwargs: Any) -> Any:
        prev = constraint.make_constraint
        constraint.make_constraint = capped_make_constraint(prev, cap)
        try:
            return orig(*args, **kwargs)
        finally:
            constraint.make_constraint = prev

    solver._mujoco_warp_step = step
    solver._alloy_friction_cap = cap
    log.info("mjw_friction_cap: elliptic friction rows' pull capped at %g m/s2 on this solver", cap)
    return True


__all__ = ["CAP_DEFAULT", "CAP_KEY", "ENV", "cap_friction_aref", "cap_value", "capped_make_constraint",
           "default_cap", "install_friction_cap", "launch_cap"]
