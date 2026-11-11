# SPDX-License-Identifier: Apache-2.0
"""Drilling plug-in (engine item 10, material removal part 2): ``Drill`` and its feed-controlled spindle ``FeedHead``."""

from alloy.plugins.drilling.head import FeedHead
from alloy.plugins.drilling.plugin import Drill

__all__ = ["Drill", "FeedHead"]
