# SPDX-License-Identifier: Apache-2.0
"""P3-2 law scorer (CPU): engine npz named like the pack's cond.json (MJCF names) score as if named like Newton."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import score_laws  # noqa: E402
from law_pack import RIGID, RODS, law_twin  # noqa: E402


def _scorer_model(law_id: str):
    from alloy.bench.runner import RunContext, build_model, plan_worlds

    spec, twin = law_twin(law_id)
    ctx = RunContext(spec, "cpu", 0, "mujoco", False)
    worlds = plan_worlds(spec, twin.conditions(False), ctx, getattr(twin, "expand", None))
    return build_model(spec, twin, worlds, ctx), worlds


@pytest.mark.parametrize("law_id", RIGID + RODS)
def test_pack_body_order_is_newtons_within_a_world(law_id):
    """Each condition's first world: the k-th Newton body starts where the pack's k-th body starts."""
    from alloy._vendor import newton

    model, worlds = _scorer_model(law_id)
    s0 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pos, by_world, seen = s0.body_q.numpy()[:, :3], score_laws._world_bodies(model), set()
    for g, w in enumerate(worlds):
        if w.condition.id in seen:
            continue
        seen.add(w.condition.id)
        cond = json.loads((OURS / "packs" / "laws" / law_id / w.condition.id / "cond.json").read_text())
        pack = np.array([cond["initial"][0]["bodies"][b["name"]]["q"][:3] for b in cond["bodies"]])
        newton_pos = pos[[b for b, _ in by_world[g]]]
        assert pack.shape == newton_pos.shape and np.allclose(pack, newton_pos, atol=1e-6), w.condition.id


def test_npz_bodies_match_by_newton_label_else_by_pack_order():
    """A4 tumble (world 1 of the scorer's model is Newton's body_5..; the pack calls it body_0..)."""
    model, worlds = _scorer_model("A4.momentum")
    g = next(i for i, w in enumerate(worlds) if w.condition.id == "tumble")
    bodies = score_laws._world_bodies(model)[g]
    pack = score_laws._pack_names("A4.momentum", "tumble")
    assert [n for _, n in bodies] != pack
    newton = [n for _, n in bodies][::-1]
    assert score_laws._npz_index({"body_names": np.array(newton)}, bodies, "A4.momentum", "tumble") == \
        list(range(len(bodies)))[::-1]
    assert score_laws._npz_index({"body_names": np.array(pack)}, bodies, "A4.momentum", "tumble") == \
        list(range(len(bodies)))
    with pytest.raises(ValueError, match="neither"):
        score_laws._npz_index({"body_names": np.array(pack[::-1])}, bodies, "A4.momentum", "tumble")
