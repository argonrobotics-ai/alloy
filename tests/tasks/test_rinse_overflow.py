# SPDX-License-Identifier: Apache-2.0
"""The rinse tasks' contact-overflow watch (``rinse_hand.OverflowWatch``): every substep's capacity bits, not the
period's last (DC5's review of JOB3's rinse-sponge: Newton zeroes MuJoCo Warp's word before each step)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from alloy.tasks.rinse_hand import OverflowWatch, contact_overflow, overflow_watch, period_overflow


def _watch(worlds: int) -> tuple[OverflowWatch, wp.array]:
    ctx = SimpleNamespace(world_count=worlds, device="cpu",
                          zeros=lambda shape, dtype: wp.zeros(shape, dtype=dtype, device="cpu"))
    watch = OverflowWatch()
    watch.ctx = ctx
    watch.setup(ctx)
    bits = wp.zeros(worlds, dtype=wp.int32, device="cpu")
    return watch.attach(SimpleNamespace(mjw_data=SimpleNamespace(overflow=bits))), bits


def test_the_watch_keeps_every_substeps_capacity_bits():
    """Substep bits OR into the record: a capacity bit in a middle substep stays when the last is clean (the
    period-end read misses it); iteration bits alone never count; ``take`` clears the record."""
    from alloy._vendor.mujoco_warp import OverflowType

    watch, bits = _watch(4)
    narrow, iters = int(OverflowType.NARROWPHASE), int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)
    for word in ([0, narrow, 0, iters], [0, 0, 0, iters], [0, 0, narrow | iters, 0], [0, 0, 0, 0]):
        bits.assign(np.array(word, np.int32))
        watch.measure(None, None, 1.0e-3)
    solver = SimpleNamespace(mjw_data=SimpleNamespace(overflow=bits))
    assert contact_overflow(solver).tolist() == [False, False, False, False]          # the last substep only
    assert period_overflow(watch, solver).tolist() == [False, True, True, False]
    assert watch.take().tolist() == [False, False, False, False]                        # taken: cleared


def test_the_watch_attaches_to_a_solver_or_a_steppers_solver():
    """No MuJoCo solver: the watch is unattached (None) and ``period_overflow`` reads the solver's word, or None. A
    stepper (what the plug-in host passes at install) attaches it to the MuJoCo Warp solver among its solvers."""
    watch, bits = _watch(2)
    assert watch.attach(None).take() is None
    assert period_overflow(watch, None) is None
    assert overflow_watch(SimpleNamespace(stepper=None)) is None
    solver = SimpleNamespace(mjw_data=SimpleNamespace(overflow=bits))
    stepper = SimpleNamespace(solvers=lambda: [SimpleNamespace(), solver])                # a stepper: its MuJoCo solver
    assert watch.attach(stepper).bits is bits
    assert watch.attach(SimpleNamespace(solvers=lambda: [SimpleNamespace()])).bits is None


def test_a_starved_scene_flags_the_overflow_through_the_real_step():
    """Eight boxes on the ground with room for 2 contacts per world (MuJoCo Warp raises it to 32, still short): the
    watch, attached to the built sim's solver by its plug-in host at install, flags both worlds; with room (256
    contacts) nothing is flagged."""
    from alloy._vendor import newton

    from alloy.plugins.harness.scene import TestScene, build_simulation
    from alloy.tasks.rinse_hand import mjw_solver

    def build(b: newton.ModelBuilder) -> None:
        b.rigid_gap = 0.005
        for k in range(8):
            body = b.add_body(xform=wp.transform((0.06 * k, 0.0, 0.0105), wp.quat_identity()))
            b.add_shape_box(body, hx=0.02, hy=0.02, hz=0.01)

    def flagged(kwargs: dict) -> list[bool]:
        scene = TestScene(build=build, worlds=2, ground=True, control_dt=0.02, solver_kwargs=kwargs)
        watch = OverflowWatch()
        sim = build_simulation(scene, [watch], "cpu")
        assert watch.bits is not None                          # attached by the plug-in host at install
        assert overflow_watch(sim) is watch and watch.bits is mjw_solver(sim).mjw_data.overflow
        sim.step_period()
        return period_overflow(watch, mjw_solver(sim)).tolist()

    assert flagged({"nconmax": 2, "njmax": 24}) == [True, True]
    assert flagged({"nconmax": 256, "njmax": 1024}) == [False, False]
