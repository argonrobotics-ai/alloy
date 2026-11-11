# SPDX-License-Identifier: Apache-2.0
"""Errors raised by alloy.io; every message says what to change, and ``fix`` (when set) is the command or edit.

Example::

    from alloy.io.errors import ImportFileError
    try:
        import_file("part.step", "lib")
    except ImportFileError as e:
        print(e)            # "part.step: unknown suffix '.step'; use one of [...]"
"""

from __future__ import annotations

USD_PACKAGES = '"usd-core>=25.5,<26.9" newton-usd-schemas'     # the pins of Newton's own USD importer


class ExportError(RuntimeError):
    """An export could not be written or did not load back."""


class ImportFileError(ValueError):
    """A file could not be imported. ``fix`` is what to run or change (also part of the message)."""

    def __init__(self, message: str, fix: str | None = None) -> None:
        super().__init__(f"{message}; fix: {fix}" if fix else message)
        self.fix = fix


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
