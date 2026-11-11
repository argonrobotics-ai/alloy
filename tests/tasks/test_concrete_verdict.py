# SPDX-License-Identifier: Apache-2.0
"""The concrete ports' verdicts by the AB-gym rule (audit AB-gym-7): a criterion a port cannot judge is None and never
a pass, success counts the judged criteria and Alloy's own gates, and every verdict says "k of n" and how its scene
differs from Gym-Anything's. Hand-made end states, no simulation."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from alloy.tasks import concrete_drywall as DW
from alloy.tasks import concrete_drywall_checks as CK
from alloy.tasks import concrete_drywall_start as DS
from alloy.tasks import concrete_joints as J
from alloy.tasks import concrete_metrics as MX
from alloy.tasks import concrete_patch as PATCH
from alloy.tasks import concrete_ports as CP
from alloy.tasks import concrete_sealant as SEAL
from alloy.tasks import concrete_sealant_metrics as SM
from alloy.tasks import concrete_sealant_start as SS
from alloy.tasks import concrete_verdict as CV
from alloy.tasks import pad_edge as PE
from alloy.tasks import pad_metrics as PM
from alloy.tasks import pad_run as PR
from alloy.tasks import pad_window as PW
from alloy.tasks.concrete_pad_start import pad_fields
from tests.tasks.test_concrete_drywall import _done, _start
from tests.tasks.test_concrete_metrics import good_joint, good_patch, mutate
from tests.tasks.test_concrete_sealant import _tooled
from tests.tasks.test_pad_metrics import _grid, _ideal

PAD_NAMES = ("forms_undisturbed", "tools_back", "arms_clear")


def _verdict(task: str, **passed: bool) -> MX.Verdict:
    """A hand-made verdict: criteria by name, passed or not."""
    crits = [MX.Criterion(k, 1.0, 1.0, ok) for k, ok in passed.items()]
    return MX.Verdict(task, crits, all(passed.values()), 1.0, None, 1.0 if all(passed.values()) else 0.3)


def _says_how_it_differs(d: dict, *heads: str) -> None:
    """The verdict lists its scene's differences, the actor first, and folds their heads into the summary."""
    assert d["scene_differs"] and d["scene_differs"][0].startswith("actor:")
    assert "ASSISTED" in d["scene_differs"][0] and "robot-only" in d["scene_differs"][0]
    found = [x.split(":", 1)[0] for x in d["scene_differs"]]
    assert all(h in found for h in heads), found
    assert d["summary"].endswith("the scene differs from Gym-Anything's in: " + ", ".join(found))


def test_a_part_judged_criterion_is_false_or_none_never_true() -> None:
    assert CV.partly(True, None) is None and CV.partly(False, None) is False and CV.partly(True, True) is True
    assert CV.partly(None, False) is False and CV.partly() is True


def test_success_counts_the_judged_criteria_and_alloys_gates_only() -> None:
    """A criterion the port does not judge is None whatever its default says; a judged failure fails; nothing judged
    is no success; one of Alloy's gates open fails and is named as Alloy's."""
    port = CV.Port("t", not_judged={"b": "why not"}, scene_differs=("actor: x",))
    d = CV.verdict_dict(_verdict("t", a=True, b=True), port)
    assert d["criteria"]["b"]["passed"] is None and d["criteria"]["b"]["note"] == "why not"
    assert d["success"] and d["judged"] == [1, 2] and d["not_judged"] == ["b"]
    assert d["summary"] == ("1 of 2 Gym-Anything criteria judged; not judged: b; the scene differs from "
                            "Gym-Anything's in: actor")
    assert CV.verdict_dict(_verdict("t", a=True, b=False), port)["success"]          # None does not fail
    assert not CV.verdict_dict(_verdict("t", a=False, b=True), port)["success"]      # a judged failure does
    assert not CV.verdict_dict(_verdict("t", b=True), port)["success"]               # nothing judged
    shut = CV.verdict_dict(_verdict("t", a=True), port, gates=CV.alloy_gates("Alloy's", books_closed=False))
    assert not shut["success"] and shut["judged"] == [1, 1] and "books_closed" not in shut["criteria"]
    assert "Alloy's own gates, not Gym-Anything criteria: books_closed" in shut["summary"]
    assert CV.verdict_dict(_verdict("t", a=True), port, gates=CV.alloy_gates("Alloy's", books_closed=True))["success"]


@pytest.fixture(scope="module")
def edged():
    start = pad_fields("pad-edge", 0)
    return start, _ideal(start), _grid(start)


def test_pad_edge_judges_five_of_eight(edged) -> None:
    """Forms (cells, no force) and arms (no robot) are None; tools_back is the edger alone: None when it is back,
    False when it is not. A forms force nobody measures cannot fail the port, a gouge does."""
    start, done, g0 = edged
    v = PM.pad_edge(done, g0, start["x"], start["y"], MX.PadScalars(tools_off=0, tools_total=1))
    d = PR.verdict_dict(v)
    c = d["criteria"]
    assert [c[k]["passed"] for k in PAD_NAMES] == [None, None, None]
    assert [c[k]["measured"] for k in PAD_NAMES] == [False, "partly", False] and c["tools_back"]["judged_part"]
    assert "edger alone" in c["tools_back"]["note"] and "privileged carriage" in c["tools_back"]["note"]
    assert d["success"] and d["judged"] == [5, 8] and d["not_judged"] == list(PAD_NAMES)
    assert d["summary"].startswith("5 of 8 Gym-Anything criteria judged; not judged: forms_undisturbed, tools_back "
                                   "(judged in part), arms_clear")
    _says_how_it_differs(d, "forms", "tools", "site")
    off = PR.verdict_dict(PM.pad_edge(done, g0, start["x"], start["y"], MX.PadScalars(tools_off=1, tools_total=1)))
    assert off["criteria"]["tools_back"]["passed"] is False and not off["success"] and off["judged"] == [6, 8]
    pushed = PM.pad_edge(done, g0, start["x"], start["y"], MX.PadScalars(forms_peak_n=200.0, arms_over=2))
    assert not pushed.success and PR.verdict_dict(pushed)["success"]                 # defaults are not judged
    deep = mutate(done, h=np.where(done.paste_mask & (PM.pad_distance(start["x"], start["y"]) < 0.05), -0.02, done.h))
    assert not PR.verdict_dict(PM.pad_edge(deep, g0, start["x"], start["y"]))["success"]
    corner = PR.verdict_dict(v, differs=PE.window_differs((0.37, 0.37)))
    assert corner["scene_differs"][-1].startswith("window: a 0.37 x 0.37 m") and not PE.window_differs(None)


def test_pad_finish_and_pad_darby_do_not_judge_forms_or_the_robot() -> None:
    """pad-finish: forms and robot_off_pad None, the four tools in part, Alloy's gates named and counted; pad-darby
    as pad-edge."""
    v = NS(score=1.0, criteria=[MX.Criterion("smooth_hard", 0.99, 0.95, True),
                                *(c for c in MX._pad_shared(None, MX.PadScalars(tools_total=4))
                                  if c.name != "arms_clear"), MX.Criterion("robot_off_pad", 0.0, 0.0, True)])
    ok = dict(ride_ok=True, books_ok=True, within_limit=True, stages_at_rest=True)
    d = PR.verdict_dict(v, CP.PAD_FINISH, gates=CV.alloy_gates(CP.FINISH_GATES, **ok))
    assert [d["criteria"][k]["passed"] for k in ("forms_undisturbed", "tools_back", "robot_off_pad")] == [None] * 3
    assert d["success"] and d["judged"] == [1, 4] and list(d["alloy_gates"]) == list(ok)
    late = PR.verdict_dict(v, CP.PAD_FINISH, gates=CV.alloy_gates(CP.FINISH_GATES, **{**ok, "within_limit": False}))
    assert not late["success"] and CV.judged_pass(late)
    assert late["score"] is None and late["score_before_judging"] == 1.0 and d["score"] == 1.0   # never False at 1
    darby = MX.Verdict("pad-darby", [MX.Criterion("stone_down", 0.99, 0.98, True),
                                     *MX._pad_shared(None, MX.PadScalars(tools_off=1, tools_total=2))], False, 0.5)
    d = PR.verdict_dict(darby)
    assert d["criteria"]["tools_back"]["passed"] is False and not d["success"]
    assert d["criteria"]["arms_clear"]["passed"] is None and "darby" in d["criteria"]["tools_back"]["note"]


def test_pad_trowel_keeps_its_ruling_and_a_window_skips_what_is_not_judged() -> None:
    """pad-trowel (WP15's ruling): the shared code leaves tools_back a judged pass, so its proxy's success rule reads
    booleans; a window passes over None criteria and fails on a False one."""
    v = MX.Verdict("pad-trowel", [MX.Criterion("smooth_hard", 0.99, 0.95, True),
                                  *MX._pad_shared(None, MX.PadScalars(tools_total=2))], True, 1.0, None, 1.0)
    c = PR.verdict_dict(v)["criteria"]
    assert c["tools_back"]["passed"] is True and "note" in c["tools_back"]
    assert c["forms_undisturbed"]["passed"] is None and c["arms_clear"]["passed"] is None
    score = PW.WindowScore(0.96, 0.95, 2.0, 2.0, {k: c[k] for k in PAD_NAMES})
    assert PW.window_passed(score, 0)
    assert not PW.window_passed(PW.WindowScore(0.96, 0.95, 2.0, 2.0, {"surface_wc": {"passed": False}}), 0)


def test_patch_flush_says_its_grid() -> None:
    """All seven judged; at the default 5 mm the verdict says the grid is not their 1 mm and which criteria that
    eases, at 1 mm it does not."""
    v = MX.patch_flush(good_patch())
    d = PATCH.verdict_dict(v, PATCH.PatchFlush().cell)
    assert d["success"] and d["judged"] == [7, 7] and d["not_judged"] == []
    assert d["grid"] == "5 mm cells, coarser than their 1 mm (lenient)"
    _says_how_it_differs(d, "bucket", "site", "grid")
    assert "5 mm cells, Gym-Anything's are 1 mm" in d["scene_differs"][-1] and "no_lip" in d["scene_differs"][-1]
    theirs = PATCH.verdict_dict(v, 0.001)
    assert theirs["grid"] == "theirs (1 mm)" and not any(x.startswith("grid") for x in theirs["scene_differs"])
    other = "7 of 7 Gym-Anything criteria judged; judged by another measure than Gym-Anything's: "
    assert d["summary"].startswith(other + "no_lip, no_hard_lip, against_the_cut, no_feather_edge; the scene")
    assert theirs["summary"].startswith(other + "no_feather_edge; the scene")
    assert d["criteria"]["no_lip"]["note"] == "read on 5 mm cells, not their 1 mm"
    lipped = mutate(good_patch())
    lipped.h[40, 30:90] = 0.004                                        # a 4 mm lip along the cut
    assert not PATCH.verdict_dict(MX.patch_flush(lipped), 0.005)["success"]


def _fake_flow(w: J.Wall, read: dict):
    C = w.cells[0] * w.cells[1]
    worlds = len(read["spilled_volume"])
    flow = NS(h=NS(numpy=lambda: np.zeros((worlds, C))), base=NS(numpy=lambda: np.zeros((worlds, C), np.float32)),
              read=lambda: read)
    layer = NS(read=lambda: {k: np.zeros((worlds, C)) for k in ("C", "tau_w", "worked")})
    return flow, layer


def test_tool_joints_books_mortar_off_the_wall_and_does_not_count_its_pass() -> None:
    """The off-wall catch is read from the film's books (what left the trowel or the chart; the drains' take is the
    board's): over 2e-8 m³ smear fails and so does the port. At 0 the face part passes but smear stays None: the
    lifted trowel's stick limit is over its capacity on the flat wall, so their gob cannot fall here."""
    read = {"spilled_volume": np.array([0.0, 1e-9, 0.0]), "drained_total_volume": np.array([4e-6, 4e-6, 4e-6]),
            "tool_dropped": np.array([[0.0, 0.0], [0.0, 3e-8], [0.0, 0.0]]),
            "tool_released": np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.5e-8]]),
            "tool_dripped": np.zeros((3, 2))}
    assert J.off_wall(read) == pytest.approx([0.0, 3.1e-8, 1.5e-8], rel=1e-12, abs=0)
    with pytest.raises(KeyError):                                       # a renamed book fails, it does not read 0
        J.off_wall({k: x for k, x in read.items() if k != "tool_dropped"})
    assert J.off_wall({**read, J.OFF_BANDS: np.full((3, 2), 1e-9)})[0] == pytest.approx(2e-9, rel=1e-12, abs=0)
    w = J.wall(J.ToolJoints(window=(4.10, 4.22, 0.80, 0.835)))
    assert [g.off_m3 for g in J.grids(w, *_fake_flow(w, read))] == pytest.approx([0.0, 3.1e-8, 1.5e-8], abs=0)
    clean = CV.verdict_dict(MX.tool_joints(good_joint()), CP.TOOL_JOINTS)
    assert clean["criteria"]["smear"]["passed"] is None and clean["criteria"]["smear"]["judged_part"]
    assert clean["success"] and clean["judged"] == [4, 5] and clean["not_judged"] == ["smear"]
    assert clean["summary"].startswith("4 of 5 Gym-Anything criteria judged; not judged: smear (judged in part)")
    _says_how_it_differs(clean, "wall", "mortar board", "site")
    for off, fails in ((3.1e-8, True), (1.5e-8, False)):                 # their tolerance: 2e-8 m³
        d = CV.verdict_dict(MX.tool_joints(mutate(good_joint(), off_m3=off)), CP.TOOL_JOINTS)
        assert (d["criteria"]["smear"]["passed"] is False) == fails and d["success"] != fails, off
    smeared = mutate(good_joint(), t=np.where(good_joint().paste_mask, 0.003, 0.001))
    assert not CV.verdict_dict(MX.tool_joints(smeared), CP.TOOL_JOINTS)["success"]   # the face part is judged
    win = J.window_differs((4.10, 4.22, 0.80, 0.835))
    assert win[0].startswith("window: x 4.1-4.22 m") and not J.window_differs(None)


def test_the_joints_trowel_cannot_drop_a_gob_on_the_flat_wall() -> None:
    """Why a pass of the off-wall catch is not counted: a lifted blade keeps ``stick_volume`` (sole down, the film's
    one yield), here over three times what the trowel can carry."""
    from alloy.plugins.surface_flow.stores_lift import stick_volume

    task = J.ToolJoints(window=(4.10, 4.22, 0.80, 0.835))
    _, flow, _, _, _ = J.plugins(task, J.wall(task))
    trowel = flow.config.tools[1]
    assert trowel.body == "trowel" and trowel.store.capacity == J.TROWEL_CAP
    assert stick_volume(trowel, flow.config.material, (0.0, 0.0, -9.81)) > 3.0 * J.TROWEL_CAP


def test_tool_sealant_judges_all_five_and_says_its_wall_and_rag() -> None:
    j = SS.joint_region(0)
    skin = SS.skin_s()
    start = SM.start_grid(j, skin)
    d = CV.verdict_dict(SM.tool_sealant(_tooled(j, skin), start), CP.TOOL_SEALANT,
                        differs=SEAL.window_differs((0.3, 0.5)))
    assert d["success"] and d["judged"] == [5, 5] and d["task"] == "tool-sealant"
    _says_how_it_differs(d, "wall", "rag", "site", "window")
    assert not CV.verdict_dict(SM.tool_sealant(start, start), CP.TOOL_SEALANT)["success"]
    assert not SEAL.window_differs(None)


def test_patch_drywall_names_its_books_gate_as_alloys() -> None:
    """Five judged; the books gate is Alloy's (not counted among Gym-Anything's criteria) and still gates success;
    a coarser grid says so."""
    board = DS.on_grid(DS.wallboard(0), 0.005)
    v = CK.patch_drywall(_done(board), _start(board))
    d = DW._verdict_dict(v, 0.5e-9, 0.005)
    assert d["success"] and d["judged"] == [5, 5] and d["alloy_gates"]["books_closed"]["passed"]
    assert "stricter than Gym-Anything" in d["alloy_gates"]["books_closed"]["note"]
    assert d["grid"].startswith("smoke: 5 mm") and d["scene_differs"][-1].startswith("grid: 5 mm cells")
    _says_how_it_differs(d, "site", "grid")
    assert "kinematic knives" in d["scene_differs"][0]
    open_ = DW._verdict_dict(v, 2e-9)
    assert not open_["success"] and CV.judged_pass(open_) and open_["judged"] == [5, 5]
    assert open_["grid"] == "theirs (1 mm): the verdict" and len(open_["scene_differs"]) == 2
    holes = CK.patch_drywall(_start(board), _start(board))
    assert not DW._verdict_dict(holes, 0.0)["success"]


def test_the_env_reads_a_bool_from_every_ports_verdict() -> None:
    from alloy.envs.ports import concrete as ENV

    d = CV.verdict_dict(MX.tool_joints(good_joint()), CP.TOOL_JOINTS)
    r = NS(verdicts=[d, {**d, "success": False, "score": 0.4}])
    assert ENV._success(r, 0) is True and ENV._success(r, 1) is False
    assert ENV._score(r, 0) == 1.0 and ENV._score(r, 1) == 0.4
