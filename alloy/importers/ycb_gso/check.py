# SPDX-License-Identifier: Apache-2.0
"""Checks on ``imports/ycb_gso`` (card validators + drop test): ``alloy.importers.common.checks`` for this library.

Example::

    from alloy.importers.ycb_gso.check import run_checks
    run_checks(Path("alloy-assets/imports/ycb_gso"), workers=3)
"""

from __future__ import annotations

from pathlib import Path

from alloy.importers.common import checks
from alloy.importers.common.checks import check_all, check_one, library

LIB = "ycb_gso"
__all__ = ["check_all", "check_one", "library", "run_checks"]


def run_checks(root: Path, names: list[str] | None = None, workers: int = 2) -> dict:
    """``common.checks.run_checks`` on the YCB + GSO library.

    Example::

        run_checks(Path("alloy-assets/imports/ycb_gso"), ["ycb_025_mug"], workers=1)
    """
    return checks.run_checks(root, LIB, names, workers)
