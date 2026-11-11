# SPDX-License-Identifier: Apache-2.0
"""User-facing warnings of the public API: logged to ``alloy.api``, and printed to stderr when nothing would show
them (Alloy's package logger carries a NullHandler, which stops Python's last-resort output, so without a configured
handler a warning meant for the user, such as a frozen world or a soft-body route, would vanish)."""

from __future__ import annotations

import logging
import sys

import alloy.core.log  # noqa: F401 - the package NullHandler: else Python's last-resort handler prints the
#                        warning too, unprefixed, whenever no engine module that imports it has loaded yet

log = logging.getLogger("alloy.api")


def _shown(logger: logging.Logger) -> bool:
    """True when a handler other than a NullHandler would receive WARNING records from ``logger``."""
    lg: logging.Logger | None = logger
    while lg is not None:
        if any(not isinstance(h, logging.NullHandler) and h.level <= logging.WARNING for h in lg.handlers):
            return True
        lg = lg.parent if lg.propagate else None
    return False


def warn(fmt: str, *args: object) -> None:
    """Log a warning for the user; print it to stderr as ``alloy: ...`` if no log handler shows it.
    Example: ``warn("%s: the liquid is %.4g ml", "water", 38.7)``."""
    log.warning(fmt, *args)
    if log.isEnabledFor(logging.WARNING) and not _shown(log):
        print("alloy: " + (fmt % args if args else fmt), file=sys.stderr)
