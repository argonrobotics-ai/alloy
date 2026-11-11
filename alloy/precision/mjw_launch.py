# SPDX-License-Identifier: Apache-2.0
"""Scoped kernel substitution inside MuJoCo Warp stages (P3-1 round 5; private names in PRIVATE_HOOKS, SCR-WP5-9).

MuJoCo Warp builds its kernels in ``@cache_kernel`` factories and launches them with ``wp.launch`` from its stage
functions. Alloy's ordered and Fixed64 replacements need scratch buffers the original launches do not pass, so
they cannot be dropped in as a kernel with the same signature. Instead:

- ``substitute(module, factory, handler)`` wraps a factory; each kernel it returns is remembered with the factory's
  arguments.
- ``scope(module, stage)`` wraps a stage function (``smooth.com_pos``, ``solver.solve`` ...) under every name it is
  bound to in the package (the public ``mujoco_warp.com_pos`` that Newton calls included). While it runs for a
  model ``enable`` registered, a ``wp.launch`` of a remembered kernel calls ``handler(factory_args, launch)``
  instead of launching it.
- ``warp.launch`` is wrapped once. Every other launch, and every launch for a model not enabled, passes through
  unchanged. Handlers launch with ``original_launch``.

A CUDA graph captured inside a stage records the replacement launches. ``uninstall`` restores every name.
"""

from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import warp as wp

from alloy.core.errors import PrecisionError

log = logging.getLogger("alloy.precision")


@dataclass(frozen=True)
class Launch:
    """One intercepted ``wp.launch`` call, normalized."""
    dim: Any
    inputs: list
    outputs: list
    block_dim: Any  # None = Warp's default
    kwargs: dict = field(default_factory=dict)
    names: tuple[str, ...] = ()  # the kernel's argument names, inputs then outputs
    kernel: Any = None  # the intercepted MuJoCo Warp kernel

    def arg(self, name: str) -> Any:
        """The launch argument bound to the kernel parameter ``name``."""
        try:
            return (self.inputs + self.outputs)[self.names.index(name)]
        except ValueError as e:
            raise PrecisionError(f"kernel has no argument {name!r} (has {self.names})") from e


Handler = Callable[[tuple, Launch], None]


@dataclass(frozen=True)
class _Entry:
    handler: Handler
    args: tuple
    kernel: Any  # held so the id stays unique
    factory: str  # module.factory, for ``substituted_calls``


_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("alloy_mjw_active", default=False)
_KERNELS: dict[int, _Entry] = {}
_ENABLED: dict[int, weakref.ref] = {}
_PATCHED: list[tuple[Any, str, Any]] = []  # (module, name, original), in install order
_STATE: dict[str, Any] = {"launch": None, "sig": None}  # sig: (launch, its inspect.signature), set at install
_CALLS: dict[str, int] = {}  # module.factory -> launches replaced (Python side, counted at record time)


def original_launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
    """``warp.launch`` as Warp defines it (for handlers)."""
    return (_STATE["launch"] or wp.launch)(kernel, *args, **kwargs)


def _launch_signature() -> inspect.Signature:
    """``inspect.signature`` of the ``wp.launch`` this install wraps (``_STATE["launch"]``), computed once per install
    (41 us per call, 36 calls per T12 step on Warp CPU: SCR-WP8-5). Keyed on that callable's identity, so a stale
    entry is recomputed rather than used; never taken from ``wp.launch`` at import time."""
    launch = _STATE["launch"]
    cached = _STATE["sig"]
    if cached is None or cached[0] is not launch:
        cached = (launch, inspect.signature(launch))
        _STATE["sig"] = cached
    return cached[1]


def _bind(kernel: Any, args: tuple, kwargs: dict) -> Launch:
    b = _launch_signature().bind(kernel, *args, **kwargs)
    b.apply_defaults()
    a = b.arguments
    rest = {k: v for k, v in a.items() if k not in ("kernel", "dim", "inputs", "outputs", "block_dim")}
    names = tuple(v.label for v in getattr(getattr(kernel, "adj", None), "args", ()))
    return Launch(a["dim"], list(a["inputs"] or []), list(a["outputs"] or []), a["block_dim"], rest, names,
                  kernel)


def launch_as_is(launch: Launch) -> Any:
    """Run the intercepted kernel unchanged (a handler that only adds work after it)."""
    return _STATE["launch"](launch.kernel, dim=launch.dim, inputs=launch.inputs, outputs=launch.outputs,
                            block_dim=launch.block_dim, **launch.kwargs)


def _launch(kernel: Any, *args: Any, **kwargs: Any) -> Any:
    if _ACTIVE.get():
        entry = _KERNELS.get(id(kernel))
        if entry is not None:
            _CALLS[entry.factory] = _CALLS.get(entry.factory, 0) + 1
            return entry.handler(entry.args, _bind(kernel, args, kwargs))
    return _STATE["launch"](kernel, *args, **kwargs)


def _install_launch() -> None:
    if _STATE["launch"] is None:
        _STATE["launch"] = wp.launch
        _STATE["sig"] = (wp.launch, inspect.signature(wp.launch))
        _patch(wp, "launch", _launch)


def _patch(owner: Any, name: str, value: Any) -> None:
    _PATCHED.append((owner, name, getattr(owner, name)))
    setattr(owner, name, value)


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise PrecisionError(f"mujoco_warp module {name} not importable: {e}") from e


def _attr(mod: Any, name: str) -> Any:
    if not hasattr(mod, name):
        raise PrecisionError(f"{mod.__name__}.{name} missing: the PRIVATE_HOOKS contract no longer holds")
    return getattr(mod, name)


def _rebind(orig: Any, value: Any, packages: tuple[str, ...]) -> None:
    """Point every module attribute bound to ``orig`` (``packages`` and their loaded submodules, e.g. the public
    ``mujoco_warp.com_pos`` alias of ``mujoco_warp._src.smooth.com_pos``) at ``value``."""
    import sys

    for name, mod in list(sys.modules.items()):
        if mod is None or not any(name == p or name.startswith(p + ".") for p in packages):
            continue
        for attr, v in list(vars(mod).items()):
            if v is orig:
                _patch(mod, attr, value)


def substitute(module: str, factory: str, handler: Handler) -> None:
    """Wrap ``module.factory`` so a launch of any kernel it returns calls ``handler`` inside an enabled scope."""
    mod = _module(module)
    orig = _attr(mod, factory)

    @functools.wraps(orig)
    def wrapped(*args: Any) -> Any:
        kernel = orig(*args)
        _KERNELS[id(kernel)] = _Entry(handler, args, kernel, f"{module.rsplit('.', 1)[-1]}.{factory}")
        return kernel

    _install_launch()
    _rebind(orig, wrapped, (module.split(".")[0],))


def scope(module: str, stage: str, on_enter: Callable[..., None] | None = None) -> None:
    """Wrap ``module.stage(m, d, ...)``: substitutions apply while it runs for an enabled model; ``on_enter(m, d)``
    runs first for an enabled model (e.g. to mark per-solve scratch stale)."""
    mod = _module(module)
    orig = _attr(mod, stage)

    @functools.wraps(orig)
    def scoped(m: Any, *args: Any, **kwargs: Any) -> Any:
        on = is_enabled(m)
        if on and on_enter is not None:
            on_enter(m, *args, **kwargs)
        token = _ACTIVE.set(on)
        try:
            return orig(m, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    _rebind(orig, scoped, (module.split(".")[0],))


def substituted_calls() -> dict[str, int]:
    """Launches replaced so far, by ``module.factory`` (a graph replay does not count again)."""
    return dict(_CALLS)


def installed() -> bool:
    return bool(_PATCHED)


def enable(model: Any) -> None:
    _ENABLED[id(model)] = weakref.ref(model)


def disable(model: Any) -> None:
    _ENABLED.pop(id(model), None)


def is_enabled(model: Any) -> bool:
    ref = _ENABLED.get(id(model))
    return ref is not None and ref() is model


def uninstall() -> None:
    """Restore every wrapped name (reverse order) and forget kernels and models."""
    while _PATCHED:
        owner, name, value = _PATCHED.pop()
        setattr(owner, name, value)
    _STATE["launch"] = None
    _STATE["sig"] = None
    _KERNELS.clear()
    _ENABLED.clear()
    _CALLS.clear()
    log.info("mjw_launch: substitutions removed")
