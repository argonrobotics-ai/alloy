# SPDX-License-Identifier: Apache-2.0
"""Run a function right after a module is first imported (a post-import hook), using only the standard library.

``alloy/__init__.py`` uses it so that engine set-up that needs Newton runs when Newton is imported, not at
``import alloy`` (which must stay under one second). Rules:
- if the module is already imported, the callback runs at once;
- otherwise a one-shot ``sys.meta_path`` finder wraps the module's loader; after the module body has executed it
  removes itself, then runs each waiting callback once, in order;
- if the module's import fails, nothing runs (the finder stays for a retry);
- a callback's exception propagates to the importer (a failed set-up fails loudly).
"""

from __future__ import annotations

import importlib.abc
import sys
import threading
from collections.abc import Callable

_LOCK = threading.Lock()


def after_import(module: str, fn: Callable[[], None]) -> None:
    """Call ``fn()`` once ``module`` has been imported; at once if it already is.

    Example::

        from alloy.api._hooks import after_import
        after_import("json", lambda: print("json is loaded"))   # prints now: json is already imported
    """
    with _LOCK:
        if module not in sys.modules:
            finder = _finder_for(module)
            if finder is None:
                finder = _OneShotFinder(module)
                sys.meta_path.insert(0, finder)
            finder.callbacks.append(fn)
            return
    fn()


def pending(module: str) -> int:
    """How many callbacks still wait for ``module``. Example: ``pending("newton") == 0`` once newton is imported."""
    finder = _finder_for(module)
    return 0 if finder is None else len(finder.callbacks)


def _finder_for(module: str) -> _OneShotFinder | None:
    return next((f for f in sys.meta_path if isinstance(f, _OneShotFinder) and f.module == module), None)


class _OneShotFinder(importlib.abc.MetaPathFinder):
    """Finds nothing itself: asks the finders after it for ``module`` and wraps the loader of the spec they return."""

    def __init__(self, module: str) -> None:
        self.module = module
        self.callbacks: list[Callable[[], None]] = []

    def find_spec(self, fullname, path, target=None):  # noqa: ANN001, ANN201 - importlib protocol
        if fullname != self.module:
            return None
        spec = _spec_after(self, fullname, path, target)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "exec_module"):
            spec.loader = _HookLoader(spec.loader, self)
        return spec

    def fire(self) -> None:
        """Remove this finder, then run every callback once (exceptions propagate)."""
        with _LOCK:
            if self in sys.meta_path:
                sys.meta_path.remove(self)
            callbacks, self.callbacks = self.callbacks, []
        for fn in callbacks:
            fn()


def _spec_after(finder: _OneShotFinder, fullname: str, path, target):  # noqa: ANN001, ANN202
    for other in sys.meta_path:
        if other is finder or not hasattr(other, "find_spec"):
            continue
        spec = other.find_spec(fullname, path, target)
        if spec is not None:
            return spec
    return None


class _HookLoader(importlib.abc.Loader):
    """The real loader, plus the finder's callbacks after the module body ran."""

    def __init__(self, inner: importlib.abc.Loader, finder: _OneShotFinder) -> None:
        self.inner, self.finder = inner, finder

    def create_module(self, spec):  # noqa: ANN001, ANN201
        return self.inner.create_module(spec)

    def exec_module(self, module) -> None:  # noqa: ANN001
        self.inner.exec_module(module)       # raises on a failed import: no callbacks, the finder stays
        self.finder.fire()

    def __getattr__(self, item: str):  # noqa: ANN204 - get_data, get_filename, is_package, ... of the real loader
        return getattr(self.inner, item)
