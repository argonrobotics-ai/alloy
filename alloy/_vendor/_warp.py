# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the vendored engines under names of their own (SCR-WP0-COEXIST; coordinator 2026-11-08 13:21Z, the
amendment in [internal note] scr/WP0.md).

``scripts/vendor_engines.py`` (``alloy._vendor._warp_calls``) rewrites the vendored code so that:
- every ``wp.kernel(...)`` call that passes ``module=`` is ``_alloy_warp.kernel(...)`` (Warp's ``kernel`` reads no
  caller frame, so a wrapper is exact);
- every ``wp.func(..., module=X)`` and ``wp.get_module(X)`` reads ``_alloy_warp.module_name(X)``, the call itself
  left to Warp (``func`` finds overloads in its caller's locals, which a wrapper would hide; WP8's review).
Everything else in the vendored code uses ``warp`` itself (Warp's code generator tells builtins such as ``wp.tid()``
by ``caller is warp``).

Why: Warp keeps two kinds of modules by name, not by Python module.
- ``module="unique"``: the kernel's module is named ``f"{key}_{content hash}"``, ``key`` being the function's qualified
  name, and an existing module of that name is reused.
- ``module="<name>"``: one Warp module per name.
A user's own ``newton`` or ``mujoco_warp`` imported beside Alloy has the same qualified names, contents and module
names, so without the rewrite both copies share those kernels (OSS-2: a PyPI newton 1.8.0rc1 took over 17 named
modules from Alloy's copy, in either import order). Each shared kernel is bound to the struct types of
whichever copy defined it first, and launches from the other copy fail ("expects an array with dtype=<Struct A> but
passed array has dtype=<Struct B>", WP10's robot + towel), or silently run the other copy's code. Here a unique
kernel's key, and a named module's name, get the prefix ``alloy_vendor__``; a name built from ``__name__``
(``alloy._vendor....``) is distinct already and stays. The compiled code is the same; only names (and so
kernel-cache entries) differ.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import warp

PREFIX = "alloy_vendor__"
OWN = ("alloy._vendor.", PREFIX)     # module names that are Alloy's copies' own


def unique_key(f: Callable, name: str | None = None) -> str:
    """The key of a vendored ``module="unique"`` kernel: PREFIX + ``name``, or + Warp's own key for ``f``
    (``warp._src.codegen.make_full_qualified_name``: the qualified name, dots as ``__``, other symbols dropped)."""
    key = name if name is not None else re.sub("[^0-9a-zA-Z_]+", "", f.__qualname__.replace(".", "__"))
    return key if key.startswith(PREFIX) else PREFIX + key


def module_name(module: Any) -> Any:
    """A named Warp module of the vendored engines: PREFIX + its name unless it is Alloy's own already (``OWN``).
    "unique", None and Module objects pass (``kernel`` checks Module objects)."""
    if isinstance(module, str) and module != "unique" and not module.startswith(OWN):
        return PREFIX + module
    return module


def _own_module(module: Any) -> Any:
    """A Module object must be one of Alloy's (from ``wp.get_module(_alloy_warp.module_name(...))`` or named after
    a vendored Python module): a shared one would bring the clash back unseen (WP8's review)."""
    name = getattr(module, "name", None)
    if isinstance(module, warp._src.context.Module) and not str(name).startswith(OWN):
        raise ValueError(f"vendored kernel in Warp module {name!r}: not one of Alloy's own (alloy._vendor._warp)")
    return module


def kernel(f: Callable | None = None, *, name: str | None = None, module: Any = None, **options: Any) -> Any:
    """``warp.kernel`` for the vendored engines (module docstring); same arguments, same result otherwise."""
    if module != "unique":
        return warp.kernel(f, name=name, module=_own_module(module_name(module)), **options)
    if f is None:
        return lambda g: warp.kernel(g, name=unique_key(g, name), module=module, **options)
    return warp.kernel(f, name=unique_key(f, name), module=module, **options)


__all__ = ["OWN", "PREFIX", "kernel", "module_name", "unique_key"]
