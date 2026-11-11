# SPDX-License-Identifier: Apache-2.0
"""Errors of the Objaverse importer. Every message says what to change."""

from __future__ import annotations

from alloy.core.errors import AlloyError


class ImportRejected(AlloyError):
    """An object was rejected at one pipeline stage; ``.uid``, ``.stage`` and ``.reason`` say which and why.

    Rejections are expected (most of Objaverse fails the licence or quality filter); batch runs record them as rows,
    they do not stop the batch.

    Example:
        >>> e = ImportRejected("abc", "licence", "non-commercial (CC BY-NC)")
        >>> e.as_row()
        {'uid': 'abc', 'stage': 'licence', 'reason': 'non-commercial (CC BY-NC)'}
    """

    def __init__(self, uid: str, stage: str, reason: str) -> None:
        super().__init__(f"{uid}: rejected at {stage}: {reason}")
        self.uid = uid
        self.stage = stage
        self.reason = reason

    def as_row(self) -> dict[str, str]:
        """JSON row for a batch log."""
        return {"uid": self.uid, "stage": self.stage, "reason": self.reason}


class ImporterConfigError(AlloyError, ValueError):
    """A category table, option or path given to the importer is invalid; the message names the field to fix."""
