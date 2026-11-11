# SPDX-License-Identifier: Apache-2.0
"""``alloy create``: your mesh, a photo or a text description in; a validated, physics-sourced object card out.

Example::

    >>> import alloy.create
    >>> r = alloy.create.create(mesh="part.stl", units="mm", material="steel", out="my_objects")   # doctest: +SKIP
    >>> r.ok, r.estimates, r.verify[:1]                                                         # doctest: +SKIP
    (True, ['mass'], ['mass: weigh it; a hollow object weighs less than this'])
    >>> r = alloy.create.create(text="a red ceramic coffee mug", size=0.11)                    # doctest: +SKIP

Command line (JSON on stdout; exit 0 = card written and drop test passed, 1 = check failed, 2 = bad input):
``python -m alloy.create --mesh part.stl --units mm`` (or ``alloy create ...`` once registered). Use the card with
``alloy.compose.compose("... on a table", roots=["my_objects"])`` or ``Scene.add_object``.
"""

from __future__ import annotations

from typing import Any

from alloy.create.request import CreateError, CreateRequest

__all__ = ["CreateError", "CreateRequest", "create"]


def create(request: Any = None, **kwargs: Any) -> Any:
    """Make one object card; see ``alloy.create.build.create``. Example: ``create(mesh="mug.obj", size=0.11)``."""
    from alloy.create.build import create as _create  # noqa: PLC0415 - keeps `import alloy.create` light

    return _create(request, **kwargs)
