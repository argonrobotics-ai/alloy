# SPDX-License-Identifier: Apache-2.0
"""SCR-SC11-1 / SCR-SC11-2: the public per-colour callback of Alloy's VBD path (alloy/solvers/vbd/base.py).

- called once per colour, colours in order, in every particle iteration; at colour 0 the iteration's force and
  Hessian buffers are already zeroed (pins the hook order against Newton's colour loop);
- a force added by the callback for each colour's particles acts like the same force given as `particle_f` (on a
  sheet whose membrane is negligible next to m/dt^2, so each vertex solve is exact and the two routes agree to
  rounding; on a stiff sheet a force through the sweep converges slowly, which is the leak A17 measures);
- a deterministic Warp mode with contacts: raises while Newton's contact gather overwrites particle_forces; with a
  Newton whose gather accumulates (the pinned fix), the callback runs there too;
- a callback that adds nothing leaves the step bitwise unchanged (SolverVBDMomentum with its coarse step);
- CUDA: a captured step with a callback replays bitwise equal to eager;
- the after_accumulate slot (SCR-SC11-2): once per colour in order; it sees a colliding particle's self-contact
  Hessian, which the default slot does not; a no-op callback there is bitwise; the source tripwire rejects a loop
  whose flag read is not between the last accumulation and the solve.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd import SolverVBDAlloy, SolverVBDMomentum
from alloy.solvers.vbd.base import GATHER_OVERWRITES, after_accumulate_hook_ok
from alloy.solvers.vbd import scenes as SC

wp.config.quiet = True
STEPS, ITERS = 6, 4


@wp.kernel
def _add_force(group: wp.array[wp.int32], load: wp.array[wp.vec3], forces: wp.array[wp.vec3]):
    i = group[wp.tid()]
    forces[i] = forces[i] + load[i]


def _solver(cls, sc, iterations=ITERS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return cls(sc.model, **SC.solver_kwargs(sc, iterations))


def _load_callback(load: wp.array):
    def fn(solver, colour, state, contacts):
        group = solver.model.particle_color_groups[colour]
        wp.launch(_add_force, dim=group.size, inputs=[group, load], outputs=[solver.particle_forces],
                  device=solver.device)
    return fn


def test_called_once_per_colour_in_order_after_the_zeroing():
    sc = SC.build("cap_load")
    solver = _solver(SolverVBDMomentum, sc)
    seen = []

    def record(s, colour, state, contacts):
        if colour == 0:
            assert not s.particle_forces.numpy().any() and not s.particle_hessians.numpy().any()
        seen.append(colour)

    solver.add_colour_callback(record, external=False)
    SC.run(sc, solver, STEPS)
    colours = len(sc.model.particle_color_groups)
    assert colours >= 2 and seen == list(range(colours)) * ITERS * STEPS


def _loose_sheet(load_on: bool, n: int = 6):
    """A sheet with a negligible membrane; a random per-particle load in `particle_f` (load_on) or returned."""
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(), rot=wp.quat_identity(), vel=wp.vec3(), dim_x=n, dim_y=n, cell_x=0.01, cell_y=0.01,
                     mass=1.0e-3, tri_ke=1.0e-6, tri_ka=0.0, tri_kd=0.0, edge_ke=0.0, edge_kd=0.0)
    b.color()
    m = b.finalize(device="cpu")
    load = wp.array(np.random.default_rng(1).normal(0.0, 1.0e-3, (m.particle_count, 3)).astype(np.float32),
                    dtype=wp.vec3, device=m.device)
    return SC.CheckScene("loose", m, None, load if load_on else None, [(0, m.particle_count)]), load


def test_callback_force_acts_like_particle_f():
    ref_sc, _ = _loose_sheet(True)
    ref = SC.run(ref_sc, SolverVBDAlloy(ref_sc.model, iterations=ITERS), STEPS).particle_q.numpy()
    sc, load = _loose_sheet(False)
    solver = SolverVBDAlloy(sc.model, iterations=ITERS)
    solver.add_colour_callback(_load_callback(load))
    got = SC.run(sc, solver, STEPS).particle_q.numpy()
    unloaded = SC.run(sc, SolverVBDAlloy(sc.model, iterations=ITERS), STEPS).particle_q.numpy()
    assert len(sc.model.particle_color_groups) >= 2
    assert np.abs(got - ref).max() < 1e-2 * np.abs(unloaded - ref).max()       # float32 rounding of y: 1.4e-3


def test_deterministic_mode_with_contacts():
    sc = SC.build("sheet_rest")                         # a sheet on the ground: contacts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        solver = SolverVBDAlloy(sc.model, **SC.solver_kwargs(sc, ITERS), deterministic=wp.DeterministicMode.RUN_TO_RUN)
    seen = []
    solver.add_colour_callback(lambda s, colour, *args: seen.append(colour))
    if GATHER_OVERWRITES:
        with pytest.raises(RuntimeError, match="overwrites particle_forces"):
            SC.run(sc, solver, 1)
    else:
        SC.run(sc, solver, 1)
        assert seen == list(range(len(sc.model.particle_color_groups))) * ITERS


def test_no_op_callback_is_bitwise():
    a_sc, b_sc = SC.build("cap_load"), SC.build("cap_load")
    a = SC.run(a_sc, _solver(SolverVBDMomentum, a_sc), STEPS).particle_q.numpy()
    solver = _solver(SolverVBDMomentum, b_sc)
    solver.add_colour_callback(lambda *args: None, external=False)
    assert np.array_equal(a, SC.run(b_sc, solver, STEPS).particle_q.numpy())


@pytest.mark.gpu
@pytest.mark.parametrize("when", ["before_accumulate", "after_accumulate"])
def test_capture_with_callback_replays_bitwise(device, when):
    if not device.startswith("cuda"):
        pytest.skip("needs CUDA (ALLOY_DEVICE=cuda:0)")

    def build():
        sc = SC.build("cap_load", device, 4)
        load, sc.force = sc.force, None
        solver = _solver(SolverVBDAlloy, sc, 10)
        solver.add_colour_callback(_load_callback(load), when=when)
        return sc, solver

    sc, solver = build()
    eager = SC.run(sc, solver, 10).particle_q.numpy()
    sc, solver = build()
    s0, s1, ctrl = sc.model.state(), sc.model.state(), sc.model.control()
    with wp.ScopedCapture(device=device) as cap:
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, 1.0e-3)
        wp.copy(s0.particle_q, s1.particle_q)
        wp.copy(s0.particle_qd, s1.particle_qd)
        wp.copy(s0.particle_q_lo, s1.particle_q_lo)   # the sub-ulp remainder is State data too (SCR-WP11-F32)
    for _ in range(10):
        wp.capture_launch(cap.graph)
    assert np.array_equal(s1.particle_q.numpy(), eager)


def _upper_sheet_hessians(when: str, self_contact: bool = True) -> list[float]:
    """Per colour visit: max |particle_hessians| over that colour's particles of the upper `stack` sheet, seen from
    slot `when`. The lower sheet is pinned 0.5 mm below, 5 cm above the ground: the upper sheet's only accumulated
    terms are self-contact (no springs, attachments or body contacts)."""
    sc = SC.build("stack")
    kw = SC.solver_kwargs(sc, ITERS)
    kw["particle_enable_self_contact"] = self_contact
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        solver = SolverVBDAlloy(sc.model, **kw)
    lo, hi = sc.sheets[1]
    seen = []

    def probe(s, colour, state, contacts):
        group = s.model.particle_color_groups[colour].numpy()
        ids = group[(group >= lo) & (group < hi)]
        seen.append(float(np.abs(s.particle_hessians.numpy()[ids]).max()) if ids.size else 0.0)

    solver.add_colour_callback(probe, external=False, when=when)
    SC.run(sc, solver, 2)
    assert len(seen) == len(sc.model.particle_color_groups) * ITERS * 2
    return seen


def test_after_accumulate_slot_sees_the_self_contact_hessian():
    assert max(_upper_sheet_hessians("after_accumulate")) > 0.0
    assert max(_upper_sheet_hessians("before_accumulate")) == 0.0
    assert max(_upper_sheet_hessians("after_accumulate", self_contact=False)) == 0.0


def test_after_accumulate_runs_once_per_colour_in_order_with_both_slots():
    sc = SC.build("cap_load")
    solver = _solver(SolverVBDMomentum, sc)
    seen = []
    solver.add_colour_callback(lambda s, c, *a: seen.append(("before", c)), external=False)
    solver.add_colour_callback(lambda s, c, *a: seen.append(("after", c)), external=False, when="after_accumulate")
    SC.run(sc, solver, STEPS)
    per_iteration = [(slot, c) for c in range(len(sc.model.particle_color_groups)) for slot in ("before", "after")]
    assert seen == per_iteration * ITERS * STEPS


def test_no_op_after_accumulate_callback_is_bitwise():
    a_sc, b_sc = SC.build("stack"), SC.build("stack")
    a = SC.run(a_sc, _solver(SolverVBDAlloy, a_sc), STEPS).particle_q.numpy()
    solver = _solver(SolverVBDAlloy, b_sc)
    solver.add_colour_callback(lambda *args: None, external=False, when="after_accumulate")
    assert np.array_equal(a, SC.run(b_sc, solver, STEPS).particle_q.numpy())


def test_after_accumulate_tripwire_and_slot_names(monkeypatch):
    loop = ("for color in range(n):\n    launch(kernel=accumulate_spring)\n    if self.use_particle_tile_solve:\n"
            "        launch(kernel=tiled)\n    else:\n        launch(kernel=solve_elasticity)\n")
    assert after_accumulate_hook_ok() and after_accumulate_hook_ok(loop)
    moved = loop.replace("    launch(kernel=accumulate_spring)\n", "").replace("        launch(kernel=tiled)\n",
                                                                          "        launch(kernel=accumulate_spring)\n")
    assert not after_accumulate_hook_ok(moved)
    sc = SC.build("cap_load")
    solver = _solver(SolverVBDAlloy, sc)
    with pytest.raises(ValueError, match="expected one of"):
        solver.add_colour_callback(lambda *a: None, when="after_solve")
    monkeypatch.setattr("alloy.solvers.vbd.base.after_accumulate_hook_ok", lambda: False)
    with pytest.raises(RuntimeError, match="use_particle_tile_solve"):
        solver.add_colour_callback(lambda *a: None, when="after_accumulate")
