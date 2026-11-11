# SPDX-License-Identifier: Apache-2.0
"""Material removal plug-in (engine item 10, v1): sanding, polishing, deburring and wiping with a compliant pad.

Example::

    from alloy.plugins.removal import Removal
    sander = Removal(tool="sander", workpiece="plate", pad_center=(0.0, 0.0, -0.03))
    report = alloy.plugins.check(sander, quick=True)       # crash + laws R-1..R-3

It is off by default: nothing changes unless a simulation installs it.
"""

from alloy.plugins.removal.plugin import Removal

__all__ = ["Removal"]
