# SPDX-License-Identifier: Apache-2.0
"""P3-2 mesh-SDF law packs (``bench/xbench/ours/sdf_pack.py``) on CPU.

- A9: Alloy's own run, exported and scored through the npz path, gives exactly Alloy's direct verdicts and errors.
- A9: the pose-only fallback stays close, and a port bug in ``joint_q`` is reported as a port error.
- Both packs reproduce Alloy's scene (OBJ vertices, initial state, the um-scale perturbations and the 6g/6H fit).
- The A8 scorer measures the pack geometry on scripted poses.
A8's direct run needs CUDA (texture SDFs): its npz equality test is ``gpu``-marked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import sdf_geom  # noqa: E402
import sdf_pack  # noqa: E402
import sdf_scene  # noqa: E402

PUSH, THREAD = sdf_scene.PUSH, sdf_scene.THREAD
# Shortened runs for the 60 s budget: A9 runs the 0.25 s settle + 0.1 s of run-down (the full 2.3 s run gives the
# same exact equality by hand on CPU: both pass, advance error 1.017e-6 m, torque error 5.18e-5 N m); A8 (GPU) runs
# the 0.1 s settle + 0.1 s of the ramp.
A9_SHORT_S, A8_SHORT_S = 0.35, 0.2


def _errors(res: dict) -> list[tuple[str, float]]:
    return [(m["name"], m["error"]) for m in res["metrics"]]


@pytest.fixture(scope="module")
def a9_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("a9")
    return out, sdf_pack.alloy_law(THREAD, out, "cpu", duration_s=A9_SHORT_S)


def test_a9_scores_from_npz_equal_alloys_direct_verdicts(a9_run):
    out, direct = a9_run
    scored = sdf_pack.score_law(THREAD, out, "cpu")
    assert scored["status"] == direct["status"]
    assert _errors(scored) == _errors(direct)
    # WP3's coordinate carry (SCR-SC10-3, default-on) puts float32 rounding back into joint_q after Newton's in-step
    # FK, so exported joint_q runs up to one spacing ahead of eval_ik(body_q): 1.46e-11 m here (WP8 OK'd the bound)
    assert scored["port_check"]["ok"] and scored["port_check"]["max_dz_m"] <= 1e-9


def _rewrite(src: Path, dst: Path, edit) -> None:
    (dst / THREAD).mkdir(parents=True)
    for f in (src / THREAD).glob("*.npz"):
        z = dict(np.load(f))
        edit(z)
        np.savez(dst / THREAD / f.name, **z)


def test_a9_pose_fallback_and_joint_port_error(a9_run, tmp_path):
    out, direct = a9_run
    _rewrite(out, tmp_path / "poses", lambda z: z.pop("joint_q"))
    poses = sdf_pack.score_law(THREAD, tmp_path / "poses", "cpu")
    assert poses["port_check"]["joint_q"].startswith("from body_q") and poses["status"] == direct["status"]
    for (_, a), (_, b), m in zip(_errors(poses), _errors(direct), direct["metrics"], strict=True):
        assert abs(a - b) < 1e-3 * m["bound"]

    def flip(z):                                      # an engine whose spin axis is reversed
        z["joint_q"][..., 1] *= -1.0

    _rewrite(out, tmp_path / "flip", flip)
    bad = sdf_pack.score_law(THREAD, tmp_path / "flip", "cpu")
    assert bad["status"] == "port_error" and not bad["port_check"]["ok"]
    _rewrite(out, tmp_path / "notau", lambda z: z.pop("drive_tau"))
    assert sdf_pack.score_law(THREAD, tmp_path / "notau", "cpu")["status"] == "error"


def test_a9_pack_reproduces_alloys_scene(tmp_path):
    d = sdf_scene.write_law(THREAD, tmp_path)
    law = json.loads((d / "law.json").read_text())
    assert [c["id"] for c in law["conditions"]] == ["mu0.14", "mu0.1"]
    for c in law["conditions"]:
        cond = json.loads((d / c["id"] / "cond.json").read_text())
        chk = cond["checks"]
        assert chk["initial_err"] == 0.0 and not sdf_geom.thread_failures(chk["threads"])
        fit = chk["threads"]["fit"]
        assert fit["flank_gap_vertices_m"] == pytest.approx(fit["flank_gap_expected_m"], rel=0.01)   # 7.9 um
        assert min(fit["nut_vertices_to_bolt_obj_m"], fit["bolt_vertices_to_nut_obj_m"]) > 0.95 * fit[
            "flank_gap_expected_m"]
        assert cond["contact_pairs"][0]["mu"] == pytest.approx(c["params"]["mu"]) and cond["gravity"] == [0, 0, 0]
        spin = cond["joints"][0]["dofs"][1]
        assert spin["drive"]["mode"] == "VELOCITY" and spin["drive"]["kd"] == 10.0
        drive = np.load(d / c["id"] / "drive.npz")
        assert drive["target_qd"][-1, 0] == pytest.approx(c["params"]["omega"])
        assert np.all(drive["joint_f"][:, 0, 0] == -c["params"]["F"])


def test_a8_pack_matches_alloys_scene_and_scorer_measures_it(tmp_path):
    d = sdf_scene.write_law(PUSH, tmp_path / "packs") / "push50N"
    cond = json.loads((d / "cond.json").read_text())
    chk = cond["checks"]
    assert chk["initial_err"] == 0.0 and chk["gaps"]["max_diff_m"] == 0.0
    assert all(m["vertex_err_m"] == 0.0 and m["triangles_equal"] for m in chk["meshes"].values())
    lateral = [i["bodies"]["peg"]["q"][0] for i in cond["initial"]]      # the um-scale perturbations survive
    assert cond["worlds"] == 16 and lateral == [float(np.float32(s["lateral_m"])) for s in cond["samples"]]
    q0 = np.array([i["bodies"]["peg"]["q"] for i in cond["initial"]], np.float32)
    frames = np.repeat(q0[None, :, None], 4, axis=0)                    # [T, W, 1, 7]: settle, then the ramp step
    frames[2:, 0, 0, 2] -= np.float32(chk["gaps"]["gap_pack_m"][0] + 60e-6)   # world 0: 60 um into the bore bottom
    run = tmp_path / "run" / PUSH
    run.mkdir(parents=True)
    meta = {"engine": "scripted", "dt": 0.05, "record_every": 1}          # ramp step = 0.1 s / 0.05 s = frame 2
    np.savez(run / "push50N.npz", meta=np.array(json.dumps(meta)), body_names=np.array(["peg"]), t=np.arange(4) * 0.05,
             body_q=frames, body_qd=np.zeros((4, 16, 1, 6), np.float32))
    res = sdf_pack.score_law(PUSH, tmp_path / "run", "cpu")
    world = res["monitors"]["world_max_m"]
    assert world[0] == pytest.approx(60e-6, abs=1e-8) and max(world[1:]) == 0.0
    assert res["status"] == "fail" and res["metrics"][0]["error"] == pytest.approx(world[0])


def test_a8_alloy_beside_needs_cuda(tmp_path):
    res = sdf_pack.alloy_law(PUSH, tmp_path, "cpu")
    assert res["status"] == "not_run" and "CUDA" in res["reason"]


@pytest.mark.gpu
@pytest.mark.timeout(600)
def test_a8_scores_from_npz_equal_alloys_direct_verdicts(tmp_path):
    direct = sdf_pack.alloy_law(PUSH, tmp_path, "cuda:0", duration_s=A8_SHORT_S)
    scored = sdf_pack.score_law(PUSH, tmp_path, "cuda:0")
    assert scored["status"] == direct["status"]
    assert _errors(scored) == _errors(direct)
    assert scored["monitors"]["world_max_m"] == direct["monitors"]["world_max_m"]


def test_missing_condition_is_not_run(tmp_path):
    assert sdf_pack.score_law(THREAD, tmp_path, "cpu")["status"] == "not_run"


def test_a8_guard_fails_a_peg_that_left_its_support():
    """pen_max_m only sees sampled pairs: a peg that fell through scores 0, so the sink guard decides."""
    assert sdf_pack.support_guard({"sink_m": [7.5e-5, 7.4e-5]}) is None
    assert sdf_pack.support_guard({"sink_m": [-6e-4]}) is None
    g = sdf_pack.support_guard({"sink_m": [137.9, 0.0]})
    assert g["name"] == "support_lost" and g["sink_m"] == 137.9
