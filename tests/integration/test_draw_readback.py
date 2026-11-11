# SPDX-License-Identifier: Apache-2.0
"""SCR-DC3-8 (WP12 part): a BuiltScene rebuilt with other entry kwargs keeps its draw readback (``randomized``,
``randomized_seed``, ``_plan``), in ``with_entry_kwargs`` and inside ``build_simulation``; reset redraws at the seed
the scene was drawn at (``draw_seed``), not at the simulation's seed (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.integration import build_simulation
from alloy.integration.built_copy import ASSEMBLY_DRAW_SEED, draw_seed, replace_built
from alloy.integration.throughput import with_entry_kwargs
from alloy.scene.randomize import draw
from alloy.scene.readback import readback
from alloy.scene.readback_checks import per_world
from alloy.testing import integration as I
from alloy.testing import scene as S

DENSITY = ("box", "material.density")
POSE_X = ("box", "pose.x")


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):      # assembly registers the builtin solvers in it
        yield


def _built(worlds: int = 3):
    card, lib = {n: (c, lb) for n, c, lb in S.mixed_scenes()}["randomized"]
    return I.assemble(card, lib, "cpu", worlds)


def _box_mass(built) -> np.ndarray:
    h = built.handles["box"]
    stride = len(h.world_stride.body)
    return built.model.body_mass.numpy()[[h.template.body.start + w * stride for w in range(built.world_count)]]


def test_replace_built_keeps_extras_by_reference():
    built = _built()
    built.randomized_seed = 0
    out = replace_built(built, entries=list(built.entries))
    assert out is not built and out.model is built.model
    assert out.randomized is built.randomized and out._plan is built._plan and out.randomized_seed == 0


def test_readback_after_with_entry_kwargs_returns_the_draws():
    """The DC3 C13 case: draws read back after ``with_entry_kwargs`` (they came back empty: SCR-DC3-8)."""
    built = _built()
    cap = built.entries[0].capability
    out = with_entry_kwargs(built, {"alloy:readback_probe": None}, cap)       # a new BuiltScene, same kwargs
    assert out is not built and out.randomized is built.randomized and out._plan is built._plan
    t = next(o for o in out._plan.objects if o.name == "box")
    np.testing.assert_array_equal(per_world(out, t, "material.density", 0.0), built.randomized[DENSITY])
    assert readback(out) == []                             # needs _plan; without the draws it reads nominal values
    assert len(set(out.randomized[DENSITY].round(6))) == out.world_count


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_reset_redraws_at_the_scene_draw_seed(monkeypatch):
    """build_simulation(seed=7) on a with_entry_kwargs copy: sim.built keeps the draws, and reset draws at the
    scene's draw seed (0 from the assembler; 3 after a redraw at 3), not at 7: the draws and the drawn poses return.
    Those draws are the values the model holds, so the reset rewrites nothing (TRAIN-FIX 3a): no apply_randomization
    call, where a draw at 7 would differ and be written."""
    import alloy.scene.randomize as R

    real, seeds = R.apply_randomization, []
    monkeypatch.setattr(R, "apply_randomization", lambda b, seed, *a, **k: seeds.append(seed) or real(b, seed, *a, **k))
    built = with_entry_kwargs(_built(), {"alloy:readback_probe": None}, "mujoco")
    assert draw(7, built.global_world_id(1), "box.pose.x", -0.05, 0.05) != built.randomized[POSE_X][1]
    for seed in (ASSEMBLY_DRAW_SEED, 3):
        if seed:
            real(built, seed)
            built.randomized_seed = seed
        drawn, seeds[:] = built.randomized[POSE_X].copy(), []
        sim = build_simulation(built, "standard", seed=7, graph="off")
        assert sim.built.randomized is built.randomized and sim.built._plan is built._plan
        assert draw_seed(sim.built) == seed and sim.seed == 7
        sim.step_period()
        sim.reset()
        assert seeds == []
        np.testing.assert_array_equal(sim.built.randomized[POSE_X], drawn)
        np.testing.assert_allclose(_box_x(sim), drawn, atol=1e-6)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_reset_keeps_drawn_masses():
    """A reset redraw keeps the drawn masses (WP2 127ad72a: a density / youngs redraw scales by new / previous draw)."""
    built = _built()
    mass = _box_mass(built)
    sim = build_simulation(built, "standard", graph="off")
    sim.reset()
    np.testing.assert_allclose(_box_mass(sim.built), mass, rtol=1e-6)


def _box_x(sim) -> np.ndarray:
    """Each world's box root x coordinate in the state (its pose.x draw after a reset)."""
    m, h = sim.built.model, sim.built.handles["box"]
    qs, stride = m.joint_q_start.numpy(), len(h.world_stride.joint)
    jq = sim.state.joint_q.numpy()
    return np.asarray([jq[qs[h.template.joint.start + w * stride]] for w in range(sim.built.world_count)])
