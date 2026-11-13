# SPDX-License-Identifier: Apache-2.0
"""Errors raised by alloy.io; every message says what to change, and ``fix`` (when set) is the command or edit.

Example::

    from alloy.io.errors import ImportFileError
    try:
        import_file("part.step", "lib")
    except ImportFileError as e:
        print(e.message)    # "part.step: unknown suffix '.step'; use one of [...]"
        print(e.fix)        # "pass an MJCF (.xml), URDF (.urdf) or USD (...) file"
"""

from __future__ import annotations

from alloy.api.errors import AlloyAPIError

USD_PACKAGES = '"usd-core>=25.5,<26.9" newton-usd-schemas'     # the pins of Newton's own USD importer


class ExportError(AlloyAPIError, RuntimeError):
    """An export could not be written or did not load back. ``fix`` says what to change."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message, fix or "export the other formats (formats=[...]), and check the object the "
                                         "message names with alloy.check")


class ImportFileError(AlloyAPIError, ValueError):
    """A file could not be imported. ``message`` says what went wrong, ``fix`` what to run or change."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(message, fix or "check the file the message names; a robot file also loads directly: "
                                         "scene.add_robot(name, path)")


def usd_fix() -> str:
    """The command that makes USD files readable: the ``usd`` extra when this install declares it, else the two
    packages. Example: ``usd_fix()`` -> ``pip install "alloy-engine[usd]"``."""
    from importlib import metadata  # noqa: PLC0415

    try:
        extras = metadata.metadata("alloy-engine").get_all("Provides-Extra") or []
    except metadata.PackageNotFoundError:
        extras = []
    if "usd" in extras:
        return 'pip install "alloy-engine[usd]"'
    return f"pip install {USD_PACKAGES}"
