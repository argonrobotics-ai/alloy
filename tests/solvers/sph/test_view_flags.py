# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS22-6: SPH on a coupler's model view copies its particles' flags (poured, retired) to the full model."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.config import PHASE_LIQUID, PHASE_NONE, PHASE_RETIRED, PHASE_SPARE
from alloy.solvers.sph.view_flags import ViewFlagMirror, parent_flag_arrays

ACTIVE = 1


def _model(flags, parent=None):
    return SimpleNamespace(particle_count=len(flags), parent=parent,
                           particle_flags=wp.array(np.asarray(flags, np.int32), dtype=wp.int32, device="cpu"))


def _phase(values):
    return wp.array(np.asarray(values, np.int8), dtype=wp.int8, device="cpu")


def test_mirror_copies_only_the_sph_particles():
    full = _model([1, 0, 1, 1, 1])
    view = _model([0, 1, 1, 0, 0], parent=full)      # view copy: particle 0 disabled (another entry's)
    phase = _phase([PHASE_NONE, PHASE_SPARE, PHASE_LIQUID, PHASE_RETIRED, PHASE_NONE])
    ViewFlagMirror(view, phase, 5, "cpu")()
    # spare poured (0 -> 1), liquid kept, retired cleared; the two non-SPH particles keep the full model's flags
    assert full.particle_flags.numpy().tolist() == [1, 1, 1, 0, 1]


def test_no_target_on_a_plain_model_or_a_shared_array():
    plain = _model([1, 1])
    assert parent_flag_arrays(plain) == []
    shared = SimpleNamespace(particle_count=2, parent=plain, particle_flags=plain.particle_flags)
    assert parent_flag_arrays(shared) == []           # the view never copied its flags: nothing to mirror


def test_each_distinct_parent_array_once():
    root = _model([0, 0, 0])
    middle = SimpleNamespace(particle_count=3, parent=root, particle_flags=root.particle_flags)   # shares root's
    leaf = _model([1, 1, 1], parent=middle)
    assert [a.ptr for a in parent_flag_arrays(leaf)] == [root.particle_flags.ptr]


def test_a_view_with_another_particle_count_is_not_mirrored(caplog):
    view = _model([1, 1], parent=_model([0, 0, 0]))
    with caplog.at_level(logging.WARNING, logger="alloy.solvers.sph"):
        assert parent_flag_arrays(view) == []
    assert "not copied" in caplog.text


@pytest.mark.compile_heavy
def test_pour_beside_a_free_body_reaches_the_full_model():
    """A pour into a cup with a free box elsewhere (MuJoCo + SPH entries: SPH steps on a view): the full model's
    flags count every poured particle, the energy check (which reads them) stays on and passes, and the liquid is in
    the cup."""
    from alloy._vendor import newton

    import alloy.api as alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_cylinder("cup", inner_radius=0.03, depth=0.05, wall=0.004)
    scene.add_pour("water", 6e-6, pos=(0, 0, 0.12), rate=2e-5)              # 94 particles over 0.3 s
    scene.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.2, 0, 0.02), mass=0.1)
    sim = alloy.build(scene, device="cpu")
    sph = next(s for s in sim.engine.stepper._entries.values() if hasattr(s.solver, "add_emitter")).solver
    assert sph.model is not sim.engine.model and sph._mirror_flags.targets   # it does run on a view
    assert sim.engine.health._energy_on
    sim.run(0.5)
    act = int(newton.ParticleFlags.ACTIVE)
    full = sim.engine.model.particle_flags.numpy() & act != 0
    view = sph.model.particle_flags.numpy() & act != 0
    phase = sph.layout.phase.numpy()
    ids = np.flatnonzero(phase != PHASE_NONE)
    assert int(full[ids].sum()) == int(view[ids].sum()) == 94
    assert sim.check()["ok"], sim.check()["problems"]
    z = sim.state(["water"], world=0)["water.points"][:, 2]
    assert 0.0 < float(np.median(z)) < 0.05                                   # inside the 5 cm cup
    sim.close()
