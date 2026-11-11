# SPDX-License-Identifier: Apache-2.0
"""Importer errors: every failure says which object and what to change."""

from __future__ import annotations

from alloy.core.errors import AlloyError


class ImporterError(AlloyError):
    """An import step failed for one object. ``.obj`` names the object, ``.step`` the pipeline step.

    Example::

        raise ImporterError("ycb_025_mug", "repair", "mesh has 0 faces; check the source path")
    """

    def __init__(self, obj: str, step: str, message: str) -> None:
        super().__init__(f"{obj}: {step}: {message}")
        self.obj = obj
        self.step = step
        self.message = message


class MissingExtraError(ImporterError):
    """A mesh-processing dependency is not installed (they live in the optional ``import`` extra)."""

    def __init__(self, module: str) -> None:
        super().__init__("<importer>", "deps", f"module {module!r} is missing: pip install 'alloy[import]' "
                                               f"(or pip install {module})")


def need(module: str):
    """Import an optional dependency or raise ``MissingExtraError`` naming the install command.

    Example::

        trimesh = need("trimesh")
    """
    import importlib  # noqa: PLC0415

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(module) from exc
