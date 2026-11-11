# SPDX-License-Identifier: Apache-2.0
"""MuJoCo Warp's own determinism switch for SolverMuJoCo (P3-1, SPEC §8 open question 13).

MuJoCo Warp has ``Option.deterministic``, a ``DeterminismType`` bitmask (mujoco_warp@86e2fae0:
mujoco_warp/_src/types.py:267-282, 1025-1039). The bits are CONTACTS (contact sort), CONSTRAINT (canonical
constraint rows) and ATOMICS (Warp's deterministic lowering on the kernels MuJoCo Warp marks). ``put_model`` sets
NONE (io.py:444) and Newton never sets it. MuJoCo Warp reads the bitmask on every call, so ``set_mjw_determinism``
can write it on a live solver before its first step or capture.

Measured at the pin (docs/notes/wp5-mjw-determinism.md): no mode makes T12 repeatable.
- CONTACTS and CONSTRAINT call functions whose body is a TODO ``pass`` (collision_driver.py:1064,
  constraint.py:4788). They are still stubs at upstream main d41a47c (2026-10-31). Contact slots and constraint
  rows come from ``atomic_add`` counters (collision_core.py:271, constraint.py:298 and others), so their order
  changes from run to run.
- ATOMICS alone leaves R1 failing. ``smooth._subtree_com_acc`` still differs run to run under the lowering;
  upstream replaced it with range sums in #1872. ATOMICS also costs far more than M7 allows, and it does not capture
  into a CUDA graph (numbers in the note).

``stub_bits`` finds the bits whose implementation is still a no-op, so a pin bump that fills them in shows up
(``tests/precision/test_mjw_det.py``). ``effective_determinism`` never upgrades SolverMuJoCo for bits set by hand
here; it reports ``downgrade_reason`` instead. The upgrade is Alloy's ordered route (P3-1 round 5,
``alloy.precision.mjw_route``), which sets the sort bits itself on covered paths.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from alloy.core.errors import PrecisionError

log = logging.getLogger("alloy.precision")

NONE, CONTACTS, CONSTRAINT, ATOMICS = 0, 1, 2, 4
ALL = CONTACTS | CONSTRAINT | ATOMICS
MODE_BITS: dict[str, int] = {"none": NONE, "contacts": CONTACTS, "constraint": CONSTRAINT, "atomics": ATOMICS,
                             "all": ALL}
# bit -> (module, function) that implements it; read-only introspection, registered via SCR-WP5-8
SORT_HOOKS: dict[int, tuple[str, str]] = {
    CONTACTS: ("alloy._vendor.mujoco_warp._src.collision_driver", "_sort_contacts"),
    CONSTRAINT: ("alloy._vendor.mujoco_warp._src.constraint", "_sort_constraints"),
}
MEASURED = ("P3-1 measured T12 on an L4 at mujoco_warp@86e2fae0: no DeterminismType mode alone repeats bitwise; "
            "use Alloy's ordered route (mjw_route.enable_route) (docs/notes/wp5-mjw-determinism.md)")


def has_determinism_option() -> bool:
    """True when the installed mujoco_warp has ``DeterminismType`` (absent before 86e2fae0)."""
    try:
        mjw = importlib.import_module("alloy._vendor.mujoco_warp")
    except ImportError:
        return False
    return hasattr(mjw, "DeterminismType")


def _is_noop(func: Any) -> bool:
    """A function whose body is only a docstring and ``pass``: it references no names and holds no other constant."""
    code = getattr(func, "__code__", None)
    if code is None:
        return False
    return not code.co_names and all(c is None or isinstance(c, str) for c in code.co_consts)


def stub_bits() -> int:
    """Bits whose implementing function is a no-op in the installed mujoco_warp; a missing function counts as a stub."""
    from alloy.precision.mjw_order import original

    out = 0
    for bit in SORT_HOOKS:
        try:
            func = original(bit)
        except ImportError:
            func = None
        if func is None or _is_noop(func):
            out |= bit
    return out


def _option(solver: Any) -> Any:
    model = getattr(solver, "mjw_model", None)
    opt = getattr(model, "opt", None)
    if opt is None or not hasattr(opt, "deterministic"):
        raise PrecisionError(f"{type(solver).__name__} has no mjw_model.opt.deterministic (MuJoCo Warp backend with "
                             "DeterminismType needed, mujoco_warp >= 86e2fae0)")
    return opt


def mjw_bits(solver: Any) -> int:
    """The ``DeterminismType`` bits a SolverMuJoCo runs with (0 when it has no MuJoCo Warp backend)."""
    opt = getattr(getattr(solver, "mjw_model", None), "opt", None)
    return int(getattr(opt, "deterministic", 0) or 0)


def set_mjw_determinism(solver: Any, bits: int | str, *, fill_stubs: bool = True) -> int:
    """Set ``mjw_model.opt.deterministic`` on a built SolverMuJoCo (before its first step or graph capture).

    With ``fill_stubs`` (route c), CONTACTS and CONSTRAINT get Alloy's sorts (``mjw_order``) where mujoco_warp
    still has stubs."""
    value = MODE_BITS[bits.lower()] if isinstance(bits, str) else int(bits)
    if value & ~ALL:
        raise PrecisionError(f"not a DeterminismType bitmask: {bits!r}")
    _option(solver).deterministic = value
    filled = 0
    if fill_stubs and value & (CONTACTS | CONSTRAINT):
        from alloy.precision.mjw_order import install_order_hooks

        filled = install_order_hooks()
    stubs = value & stub_bits() & ~filled
    log.info("SolverMuJoCo: mujoco_warp DeterminismType %d set%s%s", value,
             f" (Alloy fills bits {value & filled})" if value & filled else "",
             f" (bits {stubs} are stubs at this pin)" if stubs else "")
    return value


def filled_bits() -> int:
    """Bits whose stub Alloy has replaced in this process (route c)."""
    from alloy.precision.mjw_order import _INSTALLED

    return sum(_INSTALLED)


def downgrade_reason(solver: Any) -> str | None:
    """Why the MuJoCo Warp bits on ``solver`` do not make it repeatable; None when no bit is set."""
    bits = mjw_bits(solver)
    if not bits:
        return None
    filled = filled_bits()
    stubs = stub_bits() & ~filled
    missing = ALL & ~bits
    parts = [f"mujoco_warp DeterminismType {bits} set"]
    if filled:
        parts.append(f"bits {filled} filled by Alloy's contact/row sorts (route c)")
    if stubs:
        parts.append(f"bits {stubs} are unimplemented stubs (contact and row order race)")
    if missing:
        parts.append(f"bits {missing} off")
    parts.append(MEASURED)
    return "SolverMuJoCo: " + "; ".join(parts)
