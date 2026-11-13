# SPDX-License-Identifier: Apache-2.0
"""Keep Isaac Lab 3.0.0's own MuJoCo Warp path working beside Alloy (a compatibility fix for exactly one array).

Alloy needs MuJoCo 3.17 (its bundled MuJoCo Warp 3.17 reads 3.17 structures). Isaac Lab 3.0.0's Newton 1.6.1 brings
MuJoCo Warp 3.12, built for MuJoCo 3.12. Between the two, one size changed: equality-constraint data grew from 11 to
12 numbers per row (``mjNEQDATA``). The first 11 keep their meaning; the 12th is a new flag that only MuJoCo 3.13+
code reads. Unfixed, ``physics=newton_mjwarp`` then fails on any robot with an equality constraint (Isaac Lab's
Franka, whose fingers are coupled, among them) with "cannot reshape array of size 12 into shape (1,1,11)".

:func:`install` hands MuJoCo Warp 3.12 the first 11 numbers of each row, which is the data it was written for. It
acts only when it finds that exact pairing (MuJoCo with 12-number rows and a MuJoCo Warp whose rows hold 11), and only
on that array shape; with matching versions it does nothing. Isaac Lab ``main`` (MuJoCo Warp 3.17) needs none.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

log = logging.getLogger("alloy.isaaclab")

OLD_NEQDATA, NEW_NEQDATA = 11, 12
MODULE = "mujoco_warp._src.io"   # Isaac Lab's own MuJoCo Warp (the pip package), never Alloy's bundled copy
_MARK = "_alloy_neqdata_fix"


def _needs_fix(io: Any) -> bool:
    import mujoco

    eq_spec = getattr(getattr(io, "types", None), "Model", None)
    field = getattr(eq_spec, "__dataclass_fields__", {}).get("eq_data")
    dtype = getattr(getattr(field, "type", None), "dtype", None)
    return getattr(mujoco, "mjNEQDATA", None) == NEW_NEQDATA and getattr(dtype, "_length_", None) == OLD_NEQDATA


def _patch(io: Any) -> bool:
    if getattr(io._create_array, _MARK, False) or not _needs_fix(io):
        return False
    create = io._create_array

    @functools.wraps(create)
    def create_array(data, spec, *args, **kwargs):
        length = getattr(getattr(spec, "dtype", None), "_length_", None)
        if length == OLD_NEQDATA and data is not None and getattr(data, "shape", (0,))[-1:] == (NEW_NEQDATA,):
            data = data[..., :OLD_NEQDATA]
        return create(data, spec, *args, **kwargs)

    setattr(create_array, _MARK, True)
    io._create_array = create_array
    log.info("MuJoCo Warp 3.12 under MuJoCo 3.13+: equality data passed as its first 11 numbers (alloy.isaaclab.compat)")
    return True


def install() -> None:
    """Apply the fix when Isaac Lab's MuJoCo Warp loads (at once if it already has); a no-op where versions match."""
    from alloy.api._hooks import after_import

    def apply() -> None:
        import sys

        try:
            _patch(sys.modules[MODULE])
        except Exception as err:  # never break Isaac Lab's import over a compatibility fix
            log.warning("alloy.isaaclab.compat: could not check MuJoCo Warp's equality data size (%s)", err)

    after_import(MODULE, apply)
