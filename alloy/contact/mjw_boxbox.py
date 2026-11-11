# SPDX-License-Identifier: Apache-2.0
"""Box-box pairs on MuJoCo Warp's primitive collider (BUL-0011).

MuJoCo C collides box-box pairs with its primitive ``mjc_BoxBox``. MuJoCo Warp (pin 86e2fae) sends them through its
general convex path instead (GJK, EPA and the box multicontact), and uses its port of ``mjc_BoxBox`` only when
``NATIVECCD`` is disabled or a box-box pair has a margin (``collision_driver._narrowphase``). On a thin box resting on
a slab, inside a band of about 0.3 um around its resting depth at a 2e-4 rad tilt, the convex path returns the
contacts flipped: normal -z at the two boxes' summed thickness (-27.7 mm for T12's block on the plywood), and the
solver kicks the block to ~3 m/s (SB-7's last S1 on task_t12). At the same frames the primitive collider returns
what MuJoCo C does (two contacts at -0.009 / -0.007 mm, +z).

``use_primitive_boxbox(solver)`` sets ``NATIVECCD`` in the Warp model's disable flags. In MuJoCo Warp that bit only
selects the box-box collider: every other pair keeps its collider, and the MuJoCo C model is untouched. Call it after
building ``SolverMuJoCo`` and before its first step or graph capture (the narrowphase reads the flag when it launches).

``use_primitive_boxbox(solver, line=True)`` also gives that solver Alloy's line contacts (``alloy.contact.
mjw_boxbox_line``, BUL-0014): MuJoCo's contacts, except that a single edge contact for an edge lying on a face (or
along a parallel edge) becomes the overlap segment's two endpoints.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("alloy.contact")


def box_box_pairs(mjw_model: Any) -> int:
    """Number of box-box geom pairs the MuJoCo Warp model collides (its own pair-type count)."""
    from alloy._vendor.mujoco_warp import GeomType

    n, b = len(GeomType), int(GeomType.BOX.value)
    return int(mjw_model.geom_pair_type_count[(b * (2 * n - b - 1)) // 2 + b])  # io.put_model's geom_trid_index


def set_primitive_boxbox(mjw_model: Any) -> bool:
    """Route ``mjw_model``'s box-box pairs to the primitive collider; False when it has none (nothing changed)."""
    from alloy._vendor.mujoco_warp import DisableBit

    if box_box_pairs(mjw_model) == 0:
        return False
    mjw_model.opt.disableflags = int(mjw_model.opt.disableflags) | int(DisableBit.NATIVECCD)
    return True


def use_primitive_boxbox(solver: Any, line: bool = False, reduce4: bool = False) -> str | None:
    """Box-box pairs of a built ``SolverMuJoCo`` on MuJoCo Warp's primitive collider (before its first step or
    capture), with ``line`` the line contacts on them too, and with ``reduce4`` (the fast rigid mode) face pairs cut
    to 4 points (``alloy.contact.mjw_boxbox_reduce``; only with ``line``). None = switched, else why nothing changed."""
    if getattr(solver, "use_mujoco_cpu", False):
        if reduce4:
            log.info("mjw_boxbox: MuJoCo C keeps mjc_BoxBox's face contacts (up to 8); the 4-point pick is "
                     "MuJoCo Warp's")
        return "MuJoCo C path (mjc_BoxBox already)"
    model = getattr(solver, "mjw_model", None)
    if model is None:
        return "no MuJoCo Warp model"
    if not set_primitive_boxbox(model):
        return "no box-box pairs"
    from alloy.contact.mjw_boxbox_line import enable_line_boxbox

    if reduce4 and not line:   # WP13's review: say why the fast mode keeps up to 8
        log.info("mjw_boxbox: line contacts are off on this solver, so the 4-point pick is off too (up to 8 per face)")
    on = line and enable_line_boxbox(solver, reduce4=reduce4)
    log.info("mjw_boxbox: %d box-box pairs on the primitive collider (line contacts %s%s)", box_box_pairs(model),
             "on" if on else "off", ", face pairs at 4 points" if on and reduce4 else "")
    return None


__all__ = ["box_box_pairs", "set_primitive_boxbox", "use_primitive_boxbox"]
