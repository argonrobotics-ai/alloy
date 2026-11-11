# SPDX-License-Identifier: Apache-2.0
"""MuJoCo Warp's EPA with the first six commits of google-deepmind/mujoco_warp PR #1845 (vendored:
``alloy.contact.vendor.mjw_pr1845_epa``).

Why: WP7's SB-7 S1s emb_e8_go2_route 100039:1758 and emb_e2_panda_route 100124:832 are OverflowType.EPA_HORIZON at
ccd_iterations 35.
- MuJoCo Warp 3.16.0 builds EPA's horizon by adding every visible face's three edges in face order, so it holds
  transient edges. A deep polytope overflows the fixed 24-edge buffer, and the world exports no contacts that substep
  (health: capacity_overflow).
- MuJoCo C has the same 24 edges but walks the boundary, and keeps the contact.
- PR #1845 stores EPA's faces as outward planes (its first commit, 8cffc23), then fixes what that exposes: the
  polytope builders, the closest face when the polytope cannot grow, winding orientation, deleting only the visible
  faces connected to the closest one, a triangle start. With its first six commits the contacts match MuJoCo C
  float64 on every case checked: one contact at -4.21e-5 m on upstream package 14's pair (C the same), 31 per world on
  e2's overflow poses (C 31), BUL-0011's slab / block frame (-8.7 / -7.3 um, where the stock EPA flips them to
  -27.7 mm), and DC5's touching ziptie pair.
- Not carried: its seventh commit 54f56ca turns that touching cylinder / box pair (MuJoCo C +1.5e-8 m) into two
  contacts at -1.113 mm, and a3b060a (multicontact's four-point pick) is not part of the EPA fix. The first commit
  alone left DC5's ziptie unsnipped in one world per route. Sizing the buffer to 3V - 6 removed the overflow but still
  missed contacts (0 and 26). Coordinator ruling 2026-11-06 04:49Z; rows in [internal note] notes/wp3-epa.

How: ``install()``, called from ``alloy/__init__``, swaps the vendored functions into MuJoCo Warp's two EPA users,
``collision_convex`` and ``collision_flex``. It runs when they are imported, or at once if they already are.
- The vendored part is what the commits change plus what uses it: its Polytope struct, the polytope builders, EPA's
  loop, ``epa_phase`` and ``ccd``. The rest stays the pinned ``collision_gjk``, so the users need only ``ccd`` and
  ``epa_phase`` rebound.
- Their kernel builders look up those names in the module's globals when they first build, so the swap must come
  first. If any of their kernels is already built, ``install`` raises rather than mix the two EPAs.
- Importing a vendored module first imports MuJoCo Warp, and so its users, while that module is half-built. Those
  users wait (``state()`` says pending) and are patched as soon as the vendored modules have loaded; no kernel can
  build in between.
- The commits only rename the face buffers the users pass (``face_pr`` / ``face_norm2`` -> ``face_normal`` /
  ``face_dist``: same positions, shapes and dtypes), so the stock call sites fit.
- It acts only on the stock files the commits were cherry-picked onto (``BASE_SHA256``, the f34c6dd5 pin). It steps
  aside, saying why in ``state()``, under ``ALLOY_MJW_EPA=stock`` (A/B runs) or once the pinned MuJoCo Warp has the
  fix (drop this module and the vendored files then).
"""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import inspect
import logging
import os
import sys
from types import ModuleType
from typing import Any

log = logging.getLogger(__name__)

PR = "google-deepmind/mujoco_warp#1845"
OPT_OUT = "ALLOY_MJW_EPA"               # "stock": keep MuJoCo Warp's own EPA
GJK = "alloy._vendor.mujoco_warp._src.collision_gjk"
# The names each EPA user imports from collision_gjk that the PR changes (mujoco_warp@f34c6dd5, v3.17.0:
# collision_convex.py:28-33, collision_flex.py:25; the other four it imports are unchanged)
USERS: dict[str, tuple[str, ...]] = {
    "alloy._vendor.mujoco_warp._src.collision_convex": ("ccd", "epa_phase"),
    "alloy._vendor.mujoco_warp._src.collision_flex": ("ccd",),
}
# sha256 of the stock files at the pin (alloy.core.newton_compat.MJWARP_PIN) that the vendored copy was merged onto,
# as vendored in alloy/_vendor/mujoco_warp (import lines rewritten by scripts/vendor_engines.py; code unchanged)
# The MuJoCo Warp commit the PR copies were cherry-picked onto (v3.17.0; re-merged 2026-11-09, the copies unchanged);
# scripts/vendor_engines.py re-keys BASE_SHA256 only
# for a vendored MuJoCo Warp at this commit (another pin: re-merge the copies onto it first).
BASE_PIN = "f34c6dd52813852d0f1a2ab6d1f4d21ef1f0829f"
BASE_SHA256: dict[str, str] = {
    GJK: "39e9fa896c30e66417930b95e7a5819154ce4ee1b11b3604720040d1cd596e7f",
    "alloy._vendor.mujoco_warp._src.collision_convex": "7f1bd3d1e8278defaeb0eb7811fc09d12d38700388bcb11cf29081ecec402e37",
    "alloy._vendor.mujoco_warp._src.collision_flex": "30dc203ea09709caaa9b4e2f5b6b80c68bb666dbcd285cf4bfbf35086f7eacbc",
}
VENDORED = "vendored"
PENDING = "pending: the vendored modules are loading"
VENDOR = ("alloy.contact.vendor.mjw_pr1845_polytope", "alloy.contact.vendor.mjw_pr1845_epa")
_state: dict[str, str] = {}             # EPA user -> VENDORED, PENDING, or why it kept the stock EPA
_pending: list[ModuleType] = []         # users imported while a vendored module was loading
_loading = 0                            # vendored modules being executed right now


def state() -> dict[str, str]:
    """Each imported EPA user: ``"vendored"``, or why it kept MuJoCo Warp's own EPA."""
    return dict(_state)


def has_pr(gjk: ModuleType) -> bool:
    """Whether a collision_gjk module already has the PR (its ``ccd`` takes ``face_normal``)."""
    return "face_normal" in inspect.signature(gjk.ccd.func).parameters


def _sha256(module: ModuleType | None) -> str | None:
    path = getattr(module, "__file__", None)
    if path is None:
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def built_kernels(name: str) -> int:
    """MuJoCo Warp kernels already built by module ``name``'s cached builders (``warp_util.cache_kernel``)."""
    from alloy._vendor.mujoco_warp._src.warp_util import _KERNEL_CACHE  # noqa: PLC0415

    return sum(1 for key in _KERNEL_CACHE if key[-2] == name)


def refusal(user: ModuleType) -> str | None:
    """None when the vendored EPA can go into ``user``, else why it keeps the stock EPA."""
    if os.environ.get(OPT_OUT, "") == "stock":
        return f"{OPT_OUT}=stock"
    gjk = sys.modules.get(GJK)
    if gjk is not None and has_pr(gjk):
        return f"the pinned MuJoCo Warp has {PR}: drop alloy.contact.mjw_epa and its vendored copy"
    for module in (gjk, user):
        name = getattr(module, "__name__", GJK)
        if _sha256(module) != BASE_SHA256[name]:
            return f"{name} is not the {BASE_PIN[:8]} file the vendored {PR} commits were cherry-picked onto"
    return None


def patch(user: ModuleType) -> str:
    """Put the vendored EPA into one EPA user (``USERS``); returns its ``state()`` entry."""
    n = built_kernels(user.__name__)
    if n:
        raise RuntimeError(f"{user.__name__} built {n} kernels with MuJoCo Warp's own EPA before alloy could install "
                           f"{PR}; import alloy before running MuJoCo Warp")
    why = refusal(user)
    if why is None and _loading:
        # a vendored module's own imports loaded MuJoCo Warp: it is half-built, so finish once it has loaded
        _pending.append(user)
        why = PENDING
    elif why is None:
        pr = importlib.import_module(VENDOR[1])
        for name in USERS[user.__name__]:
            setattr(user, name, getattr(pr, name))
        why = VENDORED
        log.info("%s: EPA from vendored %s", user.__name__, PR)
    else:
        log.warning("%s keeps MuJoCo Warp's own EPA: %s", user.__name__, why)
    _state[user.__name__] = why
    return why


def _finish_pending() -> None:
    while _pending:
        patch(_pending.pop(0))


class _Loader(importlib.abc.Loader):
    """Runs a module's own loader, then ``patch`` on an EPA user, or the pending patches once a vendored module (and
    every vendored module it imports) has loaded."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def create_module(self, spec):  # noqa: ANN001, ANN201
        return self._inner.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        global _loading
        if module.__name__ in USERS:
            self._inner.exec_module(module)
            patch(module)
            return
        _loading += 1
        try:
            self._inner.exec_module(module)
        finally:
            _loading -= 1
        if not _loading:
            _finish_pending()

    def __getattr__(self, name: str) -> Any:     # get_source / get_filename / is_package for inspect and linecache
        return getattr(self._inner, name)


class _Finder(importlib.abc.MetaPathFinder):
    """Gives the EPA users (``USERS``) and the vendored modules (``VENDOR``) ``_Loader``."""

    def find_spec(self, fullname, path, target=None):  # noqa: ANN001, ANN201
        if fullname not in USERS and fullname not in VENDOR:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    """The vendored EPA in both EPA users: now for those already imported, the rest on import. Idempotent.

    The finder goes just before Python's path finder, not first: post-import hooks in front of it (``alloy.api._hooks``,
    e.g. SCR-WP8-5's fast path on collision_convex) ask the finders after them and wrap the loader they return, so
    both patches apply. First in line, this finder would answer for collision_convex and those hooks would never run.
    """
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        at = next((i for i, f in enumerate(sys.meta_path) if f is importlib.machinery.PathFinder), len(sys.meta_path))
        sys.meta_path.insert(at, _Finder())
    for name in USERS:
        module = sys.modules.get(name)
        if module is not None and name not in _state:
            patch(module)
