# SPDX-License-Identifier: Apache-2.0
"""Contact-driven deposition plug-in: markers lay ink on a board and wipers lift it (whiteboards, labelling, painting
by contact), into a per-world colour map with an exact ledger.

Example::

    from alloy.plugins.deposition import Deposition
    board = Deposition(tools=("pen", "eraser"), kinds=("marker", "wiper"), board="whiteboard")
    report = alloy.plugins.check(board, quick=True)        # crash + laws

It is off by default: nothing changes unless a simulation installs it.
"""

from alloy.plugins.deposition.plugin import Deposition

__all__ = ["Deposition"]
