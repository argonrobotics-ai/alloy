# SPDX-License-Identifier: Apache-2.0
"""Module caches keyed by an object's ``id()`` or an array's device address, dropped when that owner is freed.

CPython reuses an object's id, and Warp's allocators reuse a device address, once the owner is gone. A cache entry
that outlives its owner is then found for a different object of other sizes: stale sort scratch, tree sizes or qLD
groups that index past the new arrays. WP4 hit this as ``test_mjw_order_gaps`` failing only inside the full suite
(2026-11-03). ``remember`` ties each entry to its owner with ``weakref.finalize``, which runs before the owner's memory
(and so its id or address) can be handed out again.
"""

from __future__ import annotations

import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any


def _drop(cache: MutableMapping, key: Hashable, value: Any) -> None:
    if cache.get(key) is value:  # a later entry under the same key (another view of a live array) stays
        del cache[key]


def remember(cache: MutableMapping, key: Hashable, owner: Any, value: Any) -> Any:
    """``cache[key] = value`` for as long as ``owner`` lives; returns ``value``."""
    cache[key] = value
    weakref.finalize(owner, _drop, cache, key, value)
    return value


__all__ = ["remember"]
