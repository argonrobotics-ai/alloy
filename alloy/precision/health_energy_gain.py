# SPDX-License-Identifier: Apache-2.0
"""ENERGY_JUMP's gain tier: a passive world whose energy rises with nothing adding it (PHYSA-1, 2026-11-09).

The two older bounds (D20, ``health_kernels.energy_flags``) compare the net energy E - W_ext with its highest value
since the reset: more than 1 % of E_scale above it in one step is ENERGY_JUMP, and more than 50 % above the start is
INVALID. A steady pump passes both. A free 0.5 kg mesh cube with ``collision="sdf"``, started 0.2 mm above the
ground, gained under 1 % per step and reached 44 % of E_scale (127 mm up, 1.0 m/s) in 3 s with health OK. And after a
fall the highest value since the reset is the start, so a body that pumps itself back up reads clean until it is
above where it began.

The gain tier keeps, per world, the highest net energy of each ``BLOCK_S`` of simulated time, and a floor: the lowest
of those block maxima since the reset. A passive world only loses energy, so every later value is at or below every
earlier block's maximum; a value more than ``GAIN_REL`` of E_scale above the floor is energy from nowhere:

    gain = (E - W_ext) - floor  >  GAIN_REL * E_scale     ->  ENERGY_JUMP, DEGRADED, sticky until the world is reset

The block maximum, not the lowest sample, is the reference because a contact stores energy health does not count: a
bouncing ball's net energy dips for the milliseconds it is pressed into the floor and comes back. A dip that hides a
whole block needs a contact slower than 1 / BLOCK_S = 4 Hz, which sinks g / (2 pi 4 Hz)^2 = 1.5 cm at rest: that is
PEN_FAIL already. Some worlds keep their first block's maximum as the floor (it is never lowered), because they
have a store E does not count that can hold energy for longer than a block (``_floor_lowers``): worlds with
particles (cloth, soft-body and rod strain energy), and rigid worlds with passive stiffness the energy sum leaves
out (a spring on a ball, free or rod joint, a tendon spring: WP5's review, 2026-11-10). A ball-joint spring pendulum
at 1 Hz holds its swing in the spring for half a second; with a floor that followed it, the swing back read as a
22 % gain.

The first block after a reset only sets the floor; nothing is judged in it. What a start releases at once (a start
overlap pushed apart, a pre-strained sheet) is START_PEN's and the older bounds' to report, as before. After that the
current block is not in the floor, so a fast pump is seen at once and a slow one when it has added GAIN_REL. The
floor restarts with the energy reference (``HealthMonitor.reset_reference``: a reset, a teleport, a topology
change); the flag and the peak gain are cleared by the world's reset (the source's ``reset``). Without step
lengths (``update(dt=None)``) no block ever ends and the tier stays off.

**Outside work.** The tier judges a world only while no outside work has been booked in it since the energy
reference was set (``W_ext`` exactly 0): "a passive world only loses energy" needs books that are exact, and a
booker's are only as exact as its formula. The SFP insertion rig books its crosshead's work as f . v dt; at the
latch's snap that is off by 18 % of E_scale in one start in four, which the tier read as a gain and turned
main's ensemble law red (2026-11-10). Where work is booked (a rig's ``add_external_work``, a force plug-in, a held
``apply_wrench``, a pour) the two older bounds apply as before.

**Rollbacks.** The flag and the peak are in the monitor's ``state_arrays``, so a world that a rollback restores
gets them back as they were at the snapshot: a pass that spiked and was discarded leaves no flag on the kept
trajectory (WP5's review). The block clock, block maximum and floor are not restored. After a discarded pass they
can only be stale toward a miss: a discarded high value can raise a block maximum, never lower the floor.

**The bound.** GAIN_REL = 10 % of E_scale, from the sweep of 2026-11-10 (L4 and CPU; scr/WP1.md): the sdf cube
cases read 33 to 57 % within 3 s; 113 passive scenes of the test suite read at most 0.23 %; the library's worst is
``demo_peg_d12``, 1.1 to 5.8 % in worlds where the peg misses its hole, falls over and rolls on the block: it speeds
up by a tenth on its own before it rolls off (a small real pump in the rolling contact, reported to the contact
owners). 10 % is 1.7 times that and a third of the smallest sdf reading.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.precision.health_adapters import CapacitySource
from alloy.precision.health_kernels import EnergyRef
from alloy.precision.health_setup import _attr, _core
from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.precision")

GAIN_REL = 0.10         # of E_scale: the gain over the floor that flags (module docstring, The bound)
BLOCK_S = 0.25          # s of simulated time per block (module docstring: slower contacts are PEN_FAIL at rest)


@wp.struct
class GainState:
    t: wp.array[wp.float64]            # [W] simulated time since the reference was reset
    block: wp.array[wp.int32]          # [W] the current block's index
    block_max: wp.array[wp.float64]    # [W] the highest net energy of the current block
    floor: wp.array[wp.float64]        # [W] the lowest block maximum since the reset (particle worlds: the first)
    ready: wp.array[wp.int32]          # [W] 1 once the first block has set the floor
    lowers: wp.array[wp.int32]         # [W] 1: the floor follows the block maxima (``_floor_lowers``)
    peak: wp.array[wp.float64]         # [W] the largest gain so far, of E_scale
    tripped: wp.array[wp.int32]        # [W] 1 once the gain passed the bound (sticky)


@wp.kernel
def _gain_step(energy: wp.array[wp.float64], ref: EnergyRef, dt: wp.float64, block_s: wp.float64, bound: wp.float64,
               st: GainState):
    w = wp.tid()
    if ref.w_cum[w] != wp.float64(0.0):       # outside work is booked here: not this tier's to judge (docstring)
        return
    net = energy[w]
    st.t[w] = st.t[w] + dt
    b = wp.int32(st.t[w] / block_s)
    if b != st.block[w]:
        if st.ready[w] == 0 or (st.lowers[w] != 0 and st.block_max[w] < st.floor[w]):
            st.floor[w] = st.block_max[w]
        st.ready[w] = 1
        st.block[w] = b
        st.block_max[w] = net
    elif net > st.block_max[w]:
        st.block_max[w] = net
    if st.ready[w] == 0:
        return
    gain = (net - st.floor[w]) / ref.e_scale[w]
    if gain > st.peak[w]:
        st.peak[w] = gain
    if gain > bound:
        st.tripped[w] = 1


@wp.kernel
def _gain_rebase(mask: wp.array[wp.int32], ref: EnergyRef, st: GainState):
    """The masked worlds start again from the energy reference's E0 (``init_energy_ref`` ran before)."""
    w = wp.tid()
    if mask[w] == 0:
        return
    st.t[w] = wp.float64(0.0)
    st.block[w] = 0
    st.block_max[w] = ref.e0[w]
    st.floor[w] = ref.e0[w]
    st.ready[w] = 0


@wp.kernel
def _gain_clear(mask: wp.array[wp.int32], every_world: int, st: GainState):
    w = wp.tid()
    if every_world != 0 or mask[w] != 0:
        st.peak[w] = wp.float64(0.0)
        st.tripped[w] = 0


def energy_ref(world_count: int, device: Any) -> EnergyRef:
    """A zeroed energy reference for ``world_count`` worlds (``HealthMonitor.reset_reference`` fills it)."""
    ref = EnergyRef()
    for name in ("z_ref", "e0", "e_max", "w_cum", "e_scale"):
        setattr(ref, name, wp.zeros(world_count, dtype=wp.float64, device=device))
    return ref


class EnergyGain:
    """The gain tier of one HealthMonitor (module docstring). ``launch`` after the world energies are summed;
    ``rebase`` with the energy reference; ``source`` raises ENERGY_JUMP from ``tripped`` and clears it at a reset."""

    def __init__(self, model: Any, world_count: int, *, bound: float = GAIN_REL, block_s: float = BLOCK_S) -> None:
        self.device, self.world_count = model.device, int(world_count)
        self.bound, self.block_s = float(bound), float(block_s)
        f64 = {"dtype": wp.float64, "device": self.device}
        i32 = {"dtype": wp.int32, "device": self.device}
        st = GainState()
        st.t, st.block_max, st.floor, st.peak = (wp.zeros(self.world_count, **f64) for _ in range(4))
        st.block, st.tripped, st.ready = (wp.zeros(self.world_count, **i32) for _ in range(3))
        st.lowers = wp.array(_floor_lowers(model, self.world_count), **i32)
        self.state = st

    def launch(self, energy: wp.array, ref: EnergyRef, dt: float | None) -> None:
        """One check: fold the worlds' energies in (one kernel; ``dt`` None keeps the blocks where they are)."""
        wp.launch(_gain_step, dim=self.world_count,
                  inputs=[energy, ref, wp.float64(dt or 0.0), wp.float64(self.block_s), wp.float64(self.bound),
                          self.state], device=self.device)

    def rebase(self, mask: wp.array, ref: EnergyRef) -> None:
        wp.launch(_gain_rebase, dim=self.world_count, inputs=[mask, ref, self.state], device=self.device)

    def clear(self, world_mask: wp.array | None = None) -> None:
        """Forget the masked worlds' flag and peak (all when None): a reset starts a new run."""
        mask = world_mask if world_mask is not None else self.state.tripped
        wp.launch(_gain_clear, dim=self.world_count, inputs=[mask, int(world_mask is None), self.state],
                  device=self.device)

    def source(self) -> CapacitySource:
        return CapacitySource(self.state.tripped, "nonzero", int(HealthFlags.ENERGY_JUMP), per_world=True,
                              label="energy gain", reset=self.clear)

    def state_arrays(self) -> dict[str, wp.array]:
        """What a world's rollback restores (module docstring, Rollbacks): the flag and the peak, [W] each."""
        return {"health.energy_gain_tripped": self.state.tripped, "health.energy_gain_peak": self.state.peak}

    def report_fields(self, on: bool) -> dict[str, Any]:
        """``HealthReport`` fields: each world's largest gain so far (of E_scale; None with the energy check off)."""
        return {"energy_gain": self.state.peak.numpy().astype(np.float64) if on else None,
                "energy_gain_limit": self.bound}


def _floor_lowers(model: Any, world_count: int) -> np.ndarray:
    """[W] int32: 1 for a world whose stores are all in E or shorter than a block, so its floor may follow the block
    maxima; 0 for a world with particles, with passive stiffness on a joint that is not a hinge or slide (the energy
    sum leaves it out: ``health_setup._joint_springs``) or with a tendon spring. An item in world -1 counts for
    every world."""
    out = np.ones(world_count, np.int32)
    for worlds in (_particle_worlds(model), _uncounted_spring_worlds(model)):
        if (worlds < 0).any():
            out[:] = 0
        out[np.unique(worlds[(worlds >= 0) & (worlds < world_count)])] = 0
    if not out.all():
        log.info("energy gain: %d of %d worlds keep their first floor (particles or uncounted springs)",
                 int(world_count - out.sum()), world_count)
    return out


def _particle_worlds(model: Any) -> np.ndarray:
    return model.particle_world.numpy() if int(model.particle_count) else np.zeros(0, np.int32)


def _uncounted_spring_worlds(model: Any) -> np.ndarray:
    """The world of every joint with stiffness that is not a hinge or slide, and of every tendon with stiffness."""
    out = [np.zeros(0, np.int32)]
    k = _core(model, "joint_stiffness")
    if k is not None and int(model.joint_count) and np.any(k > 0):
        jtype, qds = model.joint_type.numpy(), model.joint_qd_start.numpy()
        one_dof = (int(newton.JointType.REVOLUTE), int(newton.JointType.PRISMATIC))
        sprung = np.array([bool(np.any(k[qds[j]:qds[j + 1]] > 0)) and int(jtype[j]) not in one_dof
                           for j in range(int(model.joint_count))])
        out.append(model.joint_world.numpy()[sprung])
    kt = _attr(model, "tendon_stiffness")
    if kt is not None and np.any(kt > 0):
        tw = _attr(model, "tendon_world")
        out.append(tw[kt > 0] if tw is not None and tw.shape[0] == kt.shape[0] else np.full(1, -1, np.int32))
    return np.concatenate(out).astype(np.int32)


__all__ = ["BLOCK_S", "GAIN_REL", "EnergyGain", "GainState", "energy_ref"]
