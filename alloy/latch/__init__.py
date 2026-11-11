# SPDX-License-Identifier: Apache-2.0
"""Generic sprung latch (DC1): tabs, clips, sleeves and pads on spring hinges or slides, with catches and stops.

One data spec (``LatchSpec``) and two back ends:
- ``latch_mjcf(spec)``: an MJCF piece for an object card (alloy-assets ``mjcf/<area>/``), routed by the assembler;
- ``add_latch(builder, spec, xform)``: the same mechanism on a ``newton.ModelBuilder`` (rigs, twins).
Latch state comes from geometry (``alloy.latch.state``); closed-form checks live in ``alloy.latch.design``.
API notes: docs/notes/dc1-latch-api.md.
"""

from alloy.latch.spec import (ROOT, TOUCHED, Box, Catch, ContactPair, Coupling, LatchSpec, LatchSpecError, Member,
                               Spring)

__all__ = ["ROOT", "TOUCHED", "Box", "Catch", "ContactPair", "Coupling", "LatchSpec", "LatchSpecError", "Member",
           "Spring",           "add_latch", "latch_mjcf"]


def __getattr__(name: str):  # lazy: the back ends import warp / newton
    if name == "add_latch":
        from alloy.latch.builder import add_latch
        return add_latch
    if name == "latch_mjcf":
        from alloy.latch.mjcf import latch_mjcf
        return latch_mjcf
    raise AttributeError(name)
