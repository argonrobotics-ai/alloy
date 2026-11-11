# SPDX-License-Identifier: Apache-2.0
"""Errors raised by the plug-in kit. Every message says what to change."""

from __future__ import annotations

from alloy.core.errors import AlloyError


class PluginError(AlloyError):
    """A plug-in is malformed, misconfigured or failed to load. ``fix`` (optional) is the change to make, kept apart
    from the message for tools (``python -m alloy.plugins`` prints it as the reply's ``"fix"``).

    Example::

        raise PluginError("magnet: Config has no field 'strength'", fix="use one of: bodies, moments, axes")
    """

    def __init__(self, message: str = "", *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix


__all__ = ["PluginError"]
