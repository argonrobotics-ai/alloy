# SPDX-License-Identifier: Apache-2.0
"""Live viewer for an Alloy simulation: Newton's viewers (browser, desktop window, Rerun, USD file) on a ``Sim``,
with pause, single steps, world picking and a contact overlay. Opt-in: ``import alloy.viewer``; each backend's
packages are optional (``alloy.viewer.backends()`` says which are installed and the pip line for the rest).

Example::

    import alloy
    import alloy.viewer

    sim = alloy.load("demo_peg_d12", worlds=4)
    with alloy.viewer.launch(sim, backend="viser", worlds=[0, 3]) as view:
        print(view.url)                  # open in a browser
        view.run(seconds=10.0)           # Pause / Step in the browser, or view.pause() / view.step(n)

Command line: ``python -m alloy.viewer <scene> --backend viser --worlds 4 --show 0,3``.
"""

from __future__ import annotations

from typing import Any

from alloy.viewer._backends import backends
from alloy.viewer._session import Viewer, launch


def contacts(sim: Any, world: int = 0) -> dict[str, Any]:
    """The contacts of one world of ``sim`` as plain data. Example: ``alloy.viewer.contacts(sim, world=0)``."""
    from alloy.viewer._session import contacts_module  # noqa: PLC0415

    return contacts_module().contacts(sim, world)


def force_on(sim: Any, name: str, world: int = 0) -> Any:
    """The net contact force on one object, (3,) newtons. Example: ``alloy.viewer.force_on(sim, "block")``."""
    from alloy.viewer._session import contacts_module  # noqa: PLC0415

    return contacts_module().force_on(sim, name, world)


def __getattr__(name: str) -> Any:
    """``alloy.viewer.MouseDrag`` loads on first use (it brings NumPy; ``import alloy.viewer`` stays light)."""
    if name == "MouseDrag":
        from alloy.viewer._drag import MouseDrag  # noqa: PLC0415

        return MouseDrag
    raise AttributeError(f"module 'alloy.viewer' has no attribute {name!r}")


__all__ = ["MouseDrag", "Viewer", "backends", "contacts", "force_on", "launch"]
