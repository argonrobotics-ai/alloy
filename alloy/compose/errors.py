# SPDX-License-Identifier: Apache-2.0
"""The scene builder's error: what cannot be built as asked, and what to change.

Example::

    >>> err = ComposeError("no object matches 'unicorn'", "use a word from alloy.compose.search(...)")
    >>> err.to_dict()["fix"]
    'use a word from alloy.compose.search(...)'
"""

from __future__ import annotations

from alloy.api.errors import AlloyAPIError


class ComposeError(AlloyAPIError, ValueError):
    """The scene request cannot be built as asked; ``message`` says why, ``fix`` what to change.

    Example: ``ComposeError("no object matches 'unicorn'", "use a word from alloy.compose.search(...)")``."""


__all__ = ["ComposeError"]
