# SPDX-License-Identifier: Apache-2.0
"""The step graph ran out of device memory: say so once, and keep a record a program can read (SPEED-FIX D1).

``IntegratedSimulation._out_of_memory`` halves a CUDA step graph that does not fit, then steps without graphs: the
same results, usually slower (a Panda at 4,096 worlds on one L4: 60x; MPM scenes can be faster).
Its log lines never reach stderr (``alloy.core.log`` puts a NullHandler on the
package logger), so the first step down also raises one RuntimeWarning (a ResourceWarning is hidden by default).
Every step down is kept in ``sim.report()["graph"]["memory_notes"]``; ``sim.check()["ok"]`` is not changed by it.
"""

from __future__ import annotations

import warnings

FIRST = ("alloy: the CUDA step graph does not fit in device memory, so stepping falls back to smaller graphs, then to "
         "none: same results, usually slower (rigid scenes up to 60x: a Panda at 4,096 worlds on one L4). Use fewer "
         "worlds, a smaller alloy.build(scene, contact_capacity=n), fewer mesh collision shapes (CCD scratch) or fewer "
         "MPM particles; see sim.report()['graph']['memory_notes']. A device this full can still fail a later "
         "allocation and stop the run (seen with 64 worlds of MPM grains on a 23 GB GPU): then use fewer worlds")


def note_graph_memory(notes: list[str], chunk: int, why: str) -> None:
    """Append one plain line for a step down to ``notes``; the first one also warns on stderr.

    Example::

        >>> notes: list[str] = []
        >>> note_graph_memory(notes, 4, "out of memory")  # doctest: +SKIP
        >>> notes
        ['a 4-exchange step graph did not fit in device memory (out of memory)']
    """
    notes.append(f"a {chunk}-exchange step graph did not fit in device memory ({why})")
    if len(notes) == 1:
        warnings.warn(FIRST, RuntimeWarning, stacklevel=2)
