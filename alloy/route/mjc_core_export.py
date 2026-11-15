# SPDX-License-Identifier: Apache-2.0
"""The CPU route's contact copy, sized from MuJoCo C's own count.

MuJoCo C solves every contact a world makes: its arena has no fixed contact cap. The route copies them out for
reporting only (health's penetration and start-overlap checks, Episode's touch and penetration queries, the viewer,
the plug-in host's ``CONTACT_F`` sensors, the bully's observers). Nothing that steps reads the copy: the line rule and
the scoped noslip read MjData. So the copy follows MuJoCo C's count instead of an entry's ``nconmax``:

- each world's export buffer is the route's own (only ``views.gather`` reads it). When a world has more contacts than
  it holds, ``refit`` grows it and copies that call's contacts again from the MjData, so no world drops one there;
- the pooled mirror in ``mjw_data`` is what consumers size their own buffers from, once, when they are built
  (``get_max_contact_count``: the plug-in host's and the bully's ``Contacts``). ``grow_mirror`` sizes it once, in
  the solver's constructor, before any consumer exists, to ``start_cap`` or the route's cap, whichever is more:
  ``MARGIN`` times the start's contacts when they outgrew the entry's ``nconmax`` (Newton raised it), else the
  entry's own size.
  Every consumer's arrays stay the ones it was built with. Beyond that pool the mirror still raises
  CONTACT_OVERFLOW (fail closed: its readers would miss those contacts), as does a full MuJoCo C contact arena.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

import mujoco
import numpy as np

log = logging.getLogger(__name__)

MARGIN = 1.5            # x the start's contacts: DC5's ziptie rig started with 1,081 and reached 1,088 in one period


def start_cap(resolved: int, asked: int) -> int:
    """Per-world contacts the copy holds from the start (before the route's own cap): the entry's ``nconmax`` as
    Newton resolved it, and ``MARGIN`` x that when Newton had to raise the asked one (``asked`` > 0) to the start's
    contacts, which a resting scene adds to as it settles. An entry whose own size held the start keeps it."""
    resolved, asked = max(int(resolved), 0), max(int(asked), 0)
    return int(math.ceil(MARGIN * resolved)) if 0 < asked < resolved else resolved


def grow_mirror(d: Any, per_world: int, nworld: int) -> int:
    """``mjw_data`` (``d``) with room for ``per_world`` contacts per world in its pooled contact arrays (never
    shrunk). New rows repeat row 0, which before the first step holds each field's initial fill (-1 for
    ``efc_address``). Returns the pooled capacity."""
    old, new = int(d.naconmax), int(per_world) * int(nworld)
    if old <= 0 or new <= old:
        return old
    c = d.contact
    for f in dataclasses.fields(c):
        a = getattr(c, f.name)
        if getattr(a, "ndim", 0) < 1 or a.shape[0] != old:
            continue                                    # e.g. flex / elem / vert: empty without flexes
        host = a.numpy()
        grown = np.concatenate([host, np.repeat(host[:1], new - old, axis=0)])
        setattr(c, f.name, type(a)(grown, dtype=a.dtype, device=a.device))
    d.naconmax = new
    return new


def copy_contacts(m: mujoco.MjModel, d: mujoco.MjData, out: Any, n: int) -> None:
    """Contacts ``0..n-1`` of ``d`` into ``out``'s buffers: geoms, pos, frame, dist and ``mj_contactForce``."""
    c = d.contact
    out.geom[:n] = c.geom[:n]
    out.pos[:n] = c.pos[:n]
    out.frame[:n] = c.frame[:n]
    out.dist[:n] = c.dist[:n]
    for i in range(n):
        mujoco.mj_contactForce(m, d, i, out.force[i])


def refit(w: int, out: Any, m: mujoco.MjModel, d: mujoco.MjData) -> int:
    """After world ``w``'s call: if its export held fewer contacts than MuJoCo C made, grow ``out`` (to the count, at
    least ``MARGIN`` x what it held), copy them all, and take the shortfall back out of its overflow counter. Returns
    the contacts the export had been short (0: it held them all)."""
    n, cap = int(d.ncon), int(out.ncon_max)
    short = n - int(out.ncon)
    if cap <= 0 or short <= 0:
        return 0
    size = max(n, int(math.ceil(MARGIN * cap)))
    log.info("CPU route: world %d made %d contacts, its contact export held %d: grown to %d", w, n, cap, size)
    out.grow_contacts(size)
    copy_contacts(m, d, out, n)
    out.set_ncon(n)
    out.uncount(short)
    return short


__all__ = ["MARGIN", "copy_contacts", "grow_mirror", "refit", "start_cap"]
