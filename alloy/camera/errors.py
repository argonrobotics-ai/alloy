# SPDX-License-Identifier: Apache-2.0
"""Camera errors: every message says what went wrong and what to change.

Example::

    from alloy.camera.errors import CameraError
    err = CameraError("width is 0", "use a width of at least 1 pixel")
    assert "use a width" in str(err) and err.to_json()["fix"].startswith("use")
"""
from __future__ import annotations


class CameraError(ValueError):
    """A camera or render request that cannot run; ``fix`` says what to change."""

    def __init__(self, problem: str, fix: str) -> None:
        super().__init__(f"{problem}. Fix: {fix}.")
        self.problem = problem
        self.fix = fix

    def to_json(self) -> dict:
        """{"error": problem, "fix": what to change} for JSON tools."""
        return {"error": self.problem, "fix": self.fix}
