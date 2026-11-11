# SPDX-License-Identifier: Apache-2.0
"""P3-2 scorers (CPU): replaying Alloy's own runs through the exported-trajectory path gives Alloy's own numbers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import alloy_laws  # noqa: E402
import alloy_repeat  # noqa: E402
import alloy_t12  # noqa: E402
import digest  # noqa: E402
import report  # noqa: E402
import score_laws  # noqa: E402
import score_repeat  # noqa: E402
import score_s1  # noqa: E402
import sb7_watch  # noqa: E402
import soft_pack  # noqa: E402


@pytest.mark.parametrize("law_id", ["A1.freefall", "A6.restitution"])
def test_law_scores_from_npz_equal_alloys_direct_verdicts(tmp_path, law_id):
    direct = alloy_laws.run_law(law_id, tmp_path, "cpu")
    scored = score_laws.score_law(law_id, tmp_path, "cpu")
    assert scored["status"] == direct["status"]
    assert [(m["name"], m["error"]) for m in scored["metrics"]] == [(m["name"], m["error"]) for m in direct["metrics"]]


def test_d1_scores_zero_ulp_on_two_cpu_runs(tmp_path):
    alloy_laws.run_d1(tmp_path, "cpu")
    res = score_laws.score_d1(tmp_path)
    assert res["status"] == "pass" and res["metrics"][0]["error"] == 0.0


def test_d1_meta_records_the_route_in_force_not_the_one_asked():
    assert alloy_laws._route_state(False, []) == "off"
    assert alloy_laws._route_state(True, []) == "on"
    assert alloy_laws._route_state(True, ["dense Jacobian (only sparse proven)"]) == \
        "refused: dense Jacobian (only sparse proven)"


def test_repeat_law_scene_records_the_route_in_force(monkeypatch):
    """The law scenes run a dense Jacobian (and, before F6 R2, RK4), outside WP5's proven coverage: ``--det on`` must
    say refused."""
    run = alloy_repeat._law_run("A6.restitution", 0, 1, "cpu", True)
    assert run["refused"] and "dense Jacobian" in " ".join(run["refused"])
    monkeypatch.setattr(alloy_repeat, "_law_run", lambda *a: dict(run, refused=list(run["refused"])))
    rec = alloy_repeat.run_law("A6.restitution", 0, 1, "cpu", True)
    assert rec["route"].startswith("refused: dense Jacobian") and rec["det"] == "ordered route (WP5)"


def test_missing_condition_is_not_run(tmp_path):
    assert score_laws.score_law("A1.freefall", tmp_path, "cpu")["status"] == "not_run"


@pytest.fixture(scope="module")
def t12_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("t12") / "batch_100000_2.npz"
    arrays = alloy_t12.run(100_000, 2, "cpu", 1, horizon_s=0.2)
    np.savez(path, **arrays)
    return path


def test_s1_replay_matches_alloys_online_monitors(t12_run):
    res = score_s1.score(t12_run, "cpu")
    for k in ("max_speed", "teleport", "nonfinite", "impossible_speed", "joint_sep", "nontarget_disp"):
        assert np.allclose(res["monitors"][k], res["online_monitors"][k], rtol=1e-5, atol=1e-6), k
    assert res["severity"] == res["online_severity"]


def test_union_touch_pass_runs_and_only_adds_robot_excuses(t12_run):
    pre, union = (score_s1.score(t12_run, "cpu", touch) for touch in ("pre", "union"))
    assert union["severity"] == pre["severity"] == [0, 0]
    assert np.all(np.asarray(union["monitors"]["e_gain_cum"]) <= np.asarray(pre["monitors"]["e_gain_cum"]) + 1e-12)
    assert union["monitors"]["max_speed"] == pre["monitors"]["max_speed"]


def test_sb7_watch_flags_an_injected_teleport(t12_run, tmp_path):
    z = dict(np.load(t12_run))
    meta = json.loads(str(z["meta"]))
    q = z["body_q"].copy()
    q[200:, 1, list(z["body_names"]).index("block_3"), 2] += 1.0
    z.update(body_q=q, meta=np.array(json.dumps(meta)))
    batch = tmp_path / "batch_100000_2.npz"
    np.savez(batch, **{k: v for k, v in z.items() if not k.startswith("alloy_s1")})
    batch.with_suffix(".done").touch()
    (tmp_path / "STOP").touch()
    sb7_watch.watch(tmp_path, "cpu", keep=False)
    res = json.loads(batch.with_suffix(".s1.json").read_text())
    assert res["episodes"] == [[100_000, 1]] and res["s1_causes"] == {"1": "teleport"}
    assert (tmp_path / "trips" / "trip_100000_1.npz").exists() and not batch.exists()
    s = sb7_watch.summary(tmp_path)
    assert s["episodes"] == 2 and s["s1"] == 1 and s["causes"] == {"teleport": 1}


def test_digests_are_per_world_and_bit_sensitive():
    rng = np.random.default_rng(0)
    raw = {"pos": rng.standard_normal((4, 3, 7)).astype(np.float32), "vel": rng.standard_normal((4, 3, 6))}
    d = digest.world_digests(raw)
    alone = digest.world_digests({k: v[2:3] for k, v in raw.items()})
    assert alone[0] == d[2] and len(set(d)) == 4
    raw["pos"].view(np.uint32)[1, 0, 0] ^= 1
    d2 = digest.world_digests(raw)
    assert d2[1] != d[1] and [d2[i] for i in (0, 2, 3)] == [d[i] for i in (0, 2, 3)]


def test_repeat_shares():
    proc = {"engine": "e", "scene": "s", "det": "x", "worlds": 3,
            "runs": {"a": {"ours": ["a", "b", "c"]}, "b_rerun": {"ours": ["a", "b", "z"]}, "n1": {"ours": ["a"]}}}
    other = json.loads(json.dumps(proc))
    other["runs"]["a"]["ours"] = ["a", "y", "c"]
    other["runs"]["b_rerun"]["ours"] = ["a", "y", "c"]
    other["runs"]["n1"]["ours"] = ["q"]
    s = score_repeat.shares([proc, other])
    assert s["r1_share"] == 5 / 6 and s["r2_share"] == 2 / 3 and s["r4_equal"] == "1/2"
    assert s["sb11_share"] == (5 + 2) / (6 + 3)


def test_a15_scores_same_nodes_exactly_and_own_mesh_by_faces(tmp_path):
    direct = soft_pack.alloy_run(tmp_path, "cpu")
    exact = soft_pack.score(tmp_path, "cpu")
    assert exact["path"].startswith("twin record") and exact["metrics"][0]["error"] == direct["metrics"][0]["error"]
    for f in (tmp_path / soft_pack.LAW_ID).glob("*.npz"):   # an engine with its own node order: faces path
        z = dict(np.load(f))
        perm = np.random.default_rng(1).permutation(z["particle_q"].shape[2])
        z["particle_q"] = z["particle_q"][:, :, perm]
        z["particle_rest"] = z["particle_q"][0, 0]
        np.savez(f, **z)
    faces = soft_pack.score(tmp_path, "cpu")
    assert faces["path"].startswith("engine faces")
    assert abs(faces["metrics"][0]["error"] - direct["metrics"][0]["error"]) < 1e-6


def test_scorer_contact_buffer_grows_instead_of_dropping_rows(t12_run):
    z = score_s1.load(t12_run)
    rep = score_s1.Replay(2, "cpu", "pre", "task_t12", contacts_per_world=4)
    idx = score_s1._order([str(n) for n in z["body_names"]], rep.names)
    small = rep.run(z["body_q"][:, :, idx], z["body_qd"][:, :, idx], z["t"].astype(np.float64))
    st = rep.pm.contact_stats()
    assert st["per_world"] > 4 and st["peak_rows"] <= st["per_world"] * 2      # grew, then fit
    full = score_s1.score(t12_run, "cpu")["monitors"]
    for k in ("max_pen", "pen_s1_ratio", "e_gain_cum", "impossible_speed"):
        assert np.allclose(small[k], full[k], rtol=1e-6, atol=1e-9), k


def test_in_process_scorer_on_the_hull_twin_scores_as_the_replay(t12_run):
    """``alloy_sb7``'s path (the live sim's model, its mesh geoms MESH; contacts from ``target_model``'s hull twin,
    driven by the live states) gives the replay's numbers."""
    from packs_io import target_model

    z = score_s1.load(t12_run)
    rep = score_s1.Replay(2, "cpu")
    idx = score_s1._order([str(n) for n in z["body_names"]], rep.names)
    q, qd, t = z["body_q"][:, :, idx], z["body_qd"][:, :, idx], z["t"].astype(np.float64)
    want = rep.run(q, qd, t)
    live = target_model("task_t12", 2, "cpu", hulls=False)
    rep.pm = score_s1.PipelineMonitors(live, collision_model=rep.model)
    rep.prev, rep.cur = live.state(), live.state()
    got = rep.run(q, qd, t)
    for k in want:
        assert np.array_equal(got[k], want[k]), k


def test_primitive_only_scene_scores_without_mesh_counters():
    """Go2 (E8) has no mesh, so Newton makes no triangle-pair counter and no contact reducer (was a null read on
    the GPU)."""
    rep = score_s1.Replay(1, "cpu", "pre", "emb_e8_go2")
    assert all(c is not None and c.shape == (1,) for c in rep.pm._counters)
    q0, qd0 = rep.prev.body_q.numpy(), rep.prev.body_qd.numpy()
    q, qd = np.broadcast_to(q0, (3, 1, *q0.shape)), np.broadcast_to(qd0, (3, 1, *qd0.shape))
    out = rep.run(q, qd, np.arange(3) * 0.02)
    assert np.isfinite(out["max_pen"]).all() and rep.pm.contact_stats()["peak_triangle_pairs"] == 0


def test_report_headline_and_law_matrix_count_not_supported_apart(tmp_path):
    laws = tmp_path / "eng" / "laws"
    laws.mkdir(parents=True)
    ok = {"status": "pass", "metrics": [{"name": "e", "error": 0.0, "bound": 1.0}]}
    (laws / "laws_scored.json").write_text(json.dumps([dict(ok, law="A1.freefall"), dict(ok, law="A2.pendulum",
                                                                                          status="fail")]))
    (laws / "not_supported.json").write_text(json.dumps({"A13.hydrostatic": "no SPH"}))
    sb7 = tmp_path / "eng" / "sb7" / "recommended"
    sb7.mkdir(parents=True)
    (sb7 / "summary.json").write_text(json.dumps({"upper95": 3e-5, "passes": False, "episodes": 100_000, "s1": 1,
                                                  "causes": {}, "s1_raw": 1, "causes_raw": {}, "rate": 1e-5}))
    (tmp_path / "eng" / "settings.json").write_text(json.dumps({"actuator_clamp": "yes"}))
    rows = report.engine_rows(tmp_path, "eng")
    head = report.headline(rows, ["eng"])
    assert "1/2 (1 n.s., 18 not run)" in head and "recommended 1e-05 (up95 3e-05, N 100,000)" in head
    m = {ln.split(" | ")[0][2:]: ln.split(" | ")[1] for ln in report.law_matrix(rows, ["eng"]).splitlines()[2:]}
    assert (m["A1.freefall"], m["A2.pendulum"], m["A13.hydrostatic"], m["A8.push"]) == ("pass", "FAIL", "n.s.", "not run")
    doc = tmp_path / "doc.md"
    doc.write_text("intro\n<!-- report.py:laws -->\nold\n<!-- /report.py:laws -->\nend\n")
    report.fill_doc(doc, {"laws": "new\n"})
    assert doc.read_text() == "intro\n<!-- report.py:laws -->\nnew\n<!-- /report.py:laws -->\nend\n"
    doc.write_text("no blocks\n")
    with pytest.raises(ValueError):
        report.fill_doc(doc, {"laws": "new\n"})


def _sb7_summary(d: Path, episodes: int, s1: int) -> None:
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"upper95": 3.0 / episodes, "passes": False, "episodes": episodes,
                                                "s1": s1, "causes": {}, "s1_raw": s1, "causes_raw": {},
                                                "rate": s1 / episodes}))


def test_report_followup_package_overlays_laws_and_sb7_rows(tmp_path):
    ok = {"status": "pass", "metrics": [{"name": "e", "error": 0.0, "bound": 1.0}]}
    for pkg, scored, not_run in (("eng", [dict(ok, law="A1.freefall"), dict(ok, law="A10.capstan", status="fail")],
                                  None),
                                 ("eng_followup", [dict(ok, law="A10.capstan")],
                                  {"A1.freefall": "late", "A13.hydrostatic": "no port"})):
        (tmp_path / pkg / "laws").mkdir(parents=True)
        (tmp_path / pkg / "laws" / "laws_scored.json").write_text(json.dumps(scored))
        if not_run:
            (tmp_path / pkg / "laws" / "not_run.json").write_text(json.dumps(not_run))
    _sb7_summary(tmp_path / "eng" / "sb7" / "tuned", 6_144, 0)
    _sb7_summary(tmp_path / "eng" / "sb7" / "recommended", 36_864, 9_452)
    _sb7_summary(tmp_path / "eng_followup" / "sb7" / "tuned", 16_384, 10)
    rows = {(r["test"], r["row"]): r for r in report.engine_rows(tmp_path, "eng") if r["engine"] == "eng"}
    assert rows[("SB-12:A1.freefall", "recommended")]["verdict"] == "pass"      # a late not_run never erases a verdict
    assert rows[("SB-12:A10.capstan", "recommended")]["verdict"] == "pass"     # the follow-up's verdict wins
    assert (rows[("SB-12:A13.hydrostatic", "recommended")]["verdict"],
            rows[("SB-12:A13.hydrostatic", "recommended")]["notes"]) == ("not_run", "no port")
    assert rows[("SB-7:t12", "tuned")]["n"] == 16_384 and rows[("SB-7:t12", "recommended")]["n"] == 36_864
    assert sum(t == "SB-7:t12" for t, _ in rows) == 2


def test_report_tags_t12_rows_and_compares_only_clamped_ones(tmp_path):
    """XB4-4: an unclamped port's T12 rows stay in the record, labelled, and leave the headline; a ``_clamp``
    package's rows sit beside them; Genesis and Alloy beside count as clamped unless settings.json says otherwise."""
    _sb7_summary(tmp_path / "eng" / "sb7" / "recommended", 36_864, 9_452)
    _sb7_summary(tmp_path / "eng" / "sb7_alloy", 16_384, 4)
    _sb7_summary(tmp_path / "genesis" / "sb7" / "recommended", 16_384, 50)
    rows = report.engine_rows(tmp_path, "eng") + report.engine_rows(tmp_path, "genesis")
    tags = {(r["engine"], r["row"]): r["actuator_clamp"] for r in rows if r["test"] == "SB-7:t12"}
    assert tags == {("eng", "recommended"): "no", ("alloy@eng", "alloy"): "yes", ("genesis", "recommended"): "yes"}
    assert "unclamped port, not compared" in report.headline(rows, ["eng", "genesis"])
    _sb7_summary(tmp_path / "eng_clamp" / "sb7" / "recommended", 16_384, 20)
    (tmp_path / "eng_clamp" / "settings.json").write_text(json.dumps({"actuator_clamp": "yes"}))
    rows = report.engine_rows(tmp_path, "eng")
    head = report.headline(rows, ["eng"])
    assert "N 16,384" in head and "N 36,864" not in head and "unclamped port" not in head
    assert "| SB-7:t12 | eng | recommended | no |" in report.markdown(rows)         # the old row stays, labelled


def test_published_pages_carry_the_sb7_counted_population_line():
    """Beside every published S1 rate stands the counted-population sentence, pointing at the table (audit
    AB-laws-7). The pages keep the count of the day they were written (report.py rewrites its own blocks), so this
    checks the sentence is there, not that its count is today's: WP6's table grows, and an exact match with
    ``population_line()`` broke main once."""
    from alloy.bench.sb7 import population_line

    note = report.population_note()
    assert population_line() in note and "POPULATION" in note
    xbench = OURS.parent
    if not (xbench / "OUR_TESTS.md").exists():             # the two pages are internal: not in a release tree
        return
    for page, places in (("OUR_TESTS.md", 5), ("CROSS_ENGINE.md", 1)):
        text = " ".join((xbench / page).read_text().split())
        assert text.lower().count("counted population") >= 2 * places, page    # the lead-in and the table's name
        head = text.split("<!-- report.py:headline -->")[1].split("<!-- /report.py:headline -->")[0]
        assert "SB-7 counted population" in head and "bench/SCOREBOARD.md" in head, page
