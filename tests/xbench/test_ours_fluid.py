# SPDX-License-Identifier: Apache-2.0
"""P3-2 SPH law packs (CPU): Alloy's own runs scored through the engine npz path give Alloy's direct verdicts and
errors exactly, and the packs hold exactly the initial state Alloy simulates.

Runs are the twins' quick conditions, shortened to fit the 60 s timeout: A13.hydrostatic its quick 0.3 s (240 steps,
4 x 4 x 50 column), A13.archimedes 0.1 s of its 1.0 s (160 steps, 16 x 16 x 10 pool, 8-cell box), A14.dam_break
0.05 s of its 0.11 s (640 steps, a = 12). The equality does not depend on the length (same code path per column).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import fluid_pack  # noqa: E402

SHORT = {"A13.hydrostatic": None, "A13.archimedes": 0.1, "A14.dam_break": 0.05}


@pytest.mark.parametrize("strided", [False, True], ids=["stride1", "strided"])
@pytest.mark.parametrize("law_id", fluid_pack.LAWS)
def test_fluid_scores_from_npz_equal_alloys_direct_verdicts(tmp_path, monkeypatch, law_id, strided):
    """``strided``: the twin's sample cap 2000 -> 36 gives the full runs' structure on short runs (record stride 7 /
    5 / 18 for hydrostatic / archimedes / dam break; 7 and 18 do not divide the step count; dam break frames every
    18 steps)."""
    if strided:
        from alloy.solvers.sph import twins

        monkeypatch.setattr(twins, "MAX_SAMPLES", 36)
    direct = fluid_pack.alloy_law(law_id, tmp_path, "cpu", quick=True, duration_s=SHORT[law_id])
    scored = fluid_pack.score_law(law_id, tmp_path, "cpu", quick=True)
    stride = fluid_pack.law_twin(law_id)[1].record_stride(direct["steps"])
    assert (stride > 1 and (direct["steps"] % stride != 0 or law_id == "A13.archimedes")) if strided else stride == 1
    assert scored["port_check"]["ok"] and scored["port_check"]["worlds"][0]["max_dev_m"] == 0.0
    assert scored["status"] == direct["status"]
    assert [(m["name"], m["error"]) for m in scored["metrics"]] == [(m["name"], m["error"]) for m in direct["metrics"]]


def _cells(law_id: str, p: dict) -> int:
    """Liquid lattice cells the twin's parameters ask for (archimedes: the pool minus the cells under the box)."""
    if law_id == "A14.dam_break":
        return p["a"] * p["depth"] * 2 * p["a"]
    under = p["bx"] ** 2 * round(p["bz"] * p["rho_b"] / 1000.0) if law_id == "A13.archimedes" else 0
    return p["nx"] * p["ny"] * p["nz"] - under


def test_fluid_packs_hold_alloys_initial_state(tmp_path):
    """The pack = the model Alloy steps (whose frame 0 the equality test's port check matches at 0.0 m)."""
    fluid_pack.pack(tmp_path, quick=True)
    for law_id in fluid_pack.LAWS:
        _, _, _, worlds, model, _, s0 = fluid_pack._build(law_id, quick=True, cond_index=0)
        d = tmp_path / law_id / worlds[0].condition.id
        rec = json.loads((d / "fluid.json").read_text())
        q = np.load(d / "particles.npz")["particle_q"]
        ref = model.particle_q.numpy()[model.particle_world.numpy() == 0]
        assert q.dtype == np.float32 and np.array_equal(q, ref) and rec["particles"]["count"] == len(q)
        assert len(q) == _cells(law_id, worlds[0].condition.params)  # no cell lost or gained at a rounding tie
        assert rec["particles"]["lattice_offset_max_m"] < 1e-8  # float32 rounding of the 2r lattice, nothing else
        if law_id == "A13.archimedes":
            assert rec["bodies"][0]["q"] == s0.body_q.numpy()[0].tolist()
            assert rec["bodies"][0]["mass"] == float(model.body_mass.numpy()[0])
        assert rec["tank"]["inner_lo"] == [0.0, 0.0, 0.0]
        assert rec["steps"] * rec["dt"] == pytest.approx(rec["duration_s"])


def test_fluid_port_check_accepts_a_permutation_and_rejects_a_moved_particle(tmp_path):
    law_id = "A14.dam_break"
    direct = fluid_pack.alloy_law(law_id, tmp_path, "cpu", quick=True, duration_s=0.01)
    (f,) = (tmp_path / law_id).glob("*.npz")
    z = dict(np.load(f))
    perm = np.random.default_rng(1).permutation(z["particle_q"].shape[2])
    np.savez(f, **dict(z, particle_q=z["particle_q"][:, :, perm]))
    permuted = fluid_pack.score_law(law_id, tmp_path, "cpu", quick=True)
    assert permuted["port_check"]["worlds"][0]["order"] == "permuted"
    assert [m["error"] for m in permuted["metrics"]] == [m["error"] for m in direct["metrics"]]  # the front is a max
    moved = z["particle_q"].copy()
    moved[:, :, 7, 0] += 2.0e-6
    np.savez(f, **dict(z, particle_q=moved))
    assert fluid_pack.score_law(law_id, tmp_path, "cpu", quick=True)["status"] == "port_mismatch"


def test_fluid_missing_condition_is_not_run(tmp_path):
    assert fluid_pack.score_law("A13.hydrostatic", tmp_path, "cpu")["status"] == "not_run"
