# SPDX-License-Identifier: Apache-2.0
"""Whether Isaac Lab is importable here, and the error every ``alloy.isaaclab`` name raises when it is not.

Every module of ``alloy.isaaclab`` imports without Isaac Lab (so ``import alloy.isaaclab`` and Alloy's own import
checks never fail on it); using a name that needs Isaac Lab raises ``FeatureUnavailableError`` with the install line.
"""

from __future__ import annotations

from collections.abc import Callable

from alloy.api.errors import FeatureUnavailableError

INSTALL = ("install Isaac Lab 3.0 (kit-less is enough), then Alloy in the same environment: `uv pip install "
           "isaaclab==3.0.0 --index https://pypi.nvidia.com --torch-backend cu130` and `uv pip install alloy-engine` "
           "(docs/guide/isaaclab.md)")


def _probe() -> ImportError | None:
    try:
        import isaaclab.utils  # noqa: F401
        import isaaclab_newton.physics  # noqa: F401
    except ImportError as err:
        return err
    return None


ISAACLAB_ERROR: ImportError | None = _probe()


def _solver_api() -> bool:
    """True on Isaac Lab after 3.0 (``main``): a ``NewtonSolver`` class per solver; False on 3.0's per-solver
    ``NewtonManager`` subclasses (and without Isaac Lab)."""
    if ISAACLAB_ERROR is not None:
        return False
    import importlib.util

    return importlib.util.find_spec("isaaclab_newton.physics.newton_solver") is not None


NEWTON_SOLVER_API: bool = _solver_api()


def missing(name: str) -> FeatureUnavailableError:
    return FeatureUnavailableError(f"alloy.isaaclab.{name} needs Isaac Lab 3.0 with its Newton backend in this "
                                   f"environment ({ISAACLAB_ERROR})", INSTALL)


def require(name: str) -> None:
    """Raise :func:`missing` for ``name`` when Isaac Lab is not importable here."""
    if ISAACLAB_ERROR is not None:
        raise missing(name)


def missing_getattr(module: str, names: tuple[str, ...]) -> Callable[[str], object]:
    """A module ``__getattr__`` for a module whose names need Isaac Lab: those names raise ``missing``."""
    def __getattr__(attr: str) -> object:
        if attr in names:
            raise missing(attr)
        raise AttributeError(f"module {module!r} has no attribute {attr!r}")
    return __getattr__
