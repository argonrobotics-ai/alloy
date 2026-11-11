# SPDX-License-Identifier: Apache-2.0
"""Logger helper: every Alloy module logs through ``logging.getLogger("alloy.<pkg>")`` (SPEC §1.3 rule 7)."""

from __future__ import annotations

import logging

logging.getLogger("alloy").addHandler(logging.NullHandler())


def get_logger(pkg: str) -> logging.Logger:
    """Return the ``alloy.<pkg>`` logger (``pkg`` may be dotted, e.g. ``"contact.monitor"``)."""
    return logging.getLogger(f"alloy.{pkg}" if pkg else "alloy")
