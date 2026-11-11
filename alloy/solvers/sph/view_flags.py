# SPDX-License-Identifier: Apache-2.0
"""SPH particle flags back to the full model when SPH steps on a coupler's model view (SCR-OSS22-6).

Beside a rigid entry (or as a one-way follower) SolverDFSPH runs on a Newton ``ModelView``. The view gets its own
copy of ``particle_flags`` (``ModelView.disable_particles`` copies on first write), so what the emitters, sinks and the
escape box change (ACTIVE on for a poured particle, off for a retired one) stays in that copy. Newton copies only
positions and velocities back to the full state. Everything that reads the full model's flags then sees the
build-time flags: the health monitor's particle checks and energy (a pour's particles stayed outside the energy
sum while their energy was booked as external work), liquid clip recording and the API's ``.active``.

``ViewFlagMirror`` copies the flags of the solver's own particles (phase >= spare, plus retired) into every distinct
``particle_flags`` array up the view's ``parent`` chain after each step and each reset. Other particles (cloth, soft
bodies) are left alone: their entries own them. Newton keeps particle indices global in entry views (its compaction
keeps the particle projection identity), so index i is the same particle in both arrays; a view with a different
particle count is not mirrored and says so once.
"""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from .config import PHASE_NONE

log = logging.getLogger("alloy.solvers.sph")

_NONE = wp.constant(PHASE_NONE)


@wp.kernel(enable_backward=False)
def mirror_flags_kernel(phase: wp.array[wp.int8], src: wp.array[wp.int32], dst: wp.array[wp.int32]):
    """dst[i] = src[i] for the SPH solver's particles (liquid, spare, retired); others untouched."""
    i = wp.tid()
    if wp.int32(phase[i]) != _NONE:
        dst[i] = src[i]


def parent_flag_arrays(model: Any) -> list[wp.array]:
    """``particle_flags`` arrays of ``model``'s parent chain that are not the model's own (by device pointer),
    each once; [] for a plain Model or a view that shares its parent's flags."""
    own = model.particle_flags
    seen = {own.ptr}
    out: list[wp.array] = []
    node = getattr(model, "parent", None)
    while node is not None:
        if int(node.particle_count) != int(model.particle_count):
            log.warning("SolverDFSPH: the parent model has %d particles, the SPH view %d: emitted and retired "
                        "particles' flags are not copied to it (SCR-OSS22-6)", node.particle_count,
                        model.particle_count)
            break
        arr = node.particle_flags
        if arr is not None and arr.ptr not in seen:
            seen.add(arr.ptr)
            out.append(arr)
        node = getattr(node, "parent", None)
    return out


class ViewFlagMirror:
    """Copies the SPH particles' flags from the solver's view into its parents' flags (capture-safe launches)."""

    def __init__(self, model: Any, phase: wp.array, n: int, device: wp.context.Device):
        self.targets = parent_flag_arrays(model) if n else []
        self._src, self._phase, self._n, self._device = model.particle_flags, phase, n, device
        if self.targets:
            log.info("SolverDFSPH: on a model view; SPH particle flags are copied to %d parent model(s) each step",
                     len(self.targets))

    def __call__(self) -> None:
        for dst in self.targets:
            wp.launch(mirror_flags_kernel, dim=self._n, inputs=[self._phase, self._src, dst], device=self._device)
