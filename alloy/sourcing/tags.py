# SPDX-License-Identifier: Apache-2.0
"""Read the sourcing tag of a card quantity that was written before this package existed.

Card quantities carry ``source`` and a free-text ``note``. The alloy-assets convention puts the tag in the note
(``ESTIMATE``, ``UNVERIFIED``, ``[SNIPPET]``, ``[DOC]``). The rules, weakest marker first:

- ``source: default`` -> ``MISSING`` (nobody sourced it);
- ``ESTIMATE`` in the note, or ``source: model`` -> ``ESTIMATE``;
- ``UNVERIFIED`` in the note -> ``UNVERIFIED``; ``SNIPPET`` -> ``SNIPPET``;
- ``catalog`` / ``measured`` / ``fit`` with a URL or DOI in the note -> ``DOC``;
- the same without a URL -> ``UNVERIFIED`` (a cited handbook nobody re-read here; DOC needs a checkable link).

Example:
    >>> from alloy.core.quantity import Q
    >>> tag_of(Q(0.5, "catalog", note="Serway Table 5.1"))
    'UNVERIFIED'
    >>> tag_of(Q(0.5, "catalog", note="DOC: 'mu 0.5' https://example.org"))
    'DOC'
"""

from __future__ import annotations

import re
from typing import Final

from alloy.core.quantity import Q

_MARKERS: Final[tuple[str, ...]] = ("ESTIMATE", "UNVERIFIED", "SNIPPET")
_LINK = re.compile(r"https?://|\bdoi[: ]\s*10\.|\b10\.\d{4,}/", re.IGNORECASE)


def has_link(note: str) -> bool:
    """True if a note cites a URL or DOI (checkable evidence).

    Example:
        >>> has_link("Ensinger datasheet https://example.org/abs.pdf")
        True
    """
    return bool(_LINK.search(note))


def tag_of(q: Q | None) -> str:
    """``DOC``, ``SNIPPET``, ``UNVERIFIED``, ``ESTIMATE`` or ``MISSING`` for one card quantity (rules above)."""
    if q is None or q.source == "default":
        return "MISSING"
    for marker in _MARKERS:
        if re.search(rf"\b{marker}\b", q.note):
            return marker
    if q.source == "model":
        return "ESTIMATE"
    return "DOC" if has_link(q.note) else "UNVERIFIED"
