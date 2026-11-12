# SPDX-License-Identifier: Apache-2.0
"""The tasks the engine must keep doing, each with its small check and the tolerance stated for each metric.

Sources: [internal note] (the paused programs' finished tasks, coordinator 2026-11-06)
plus T12. A tolerance comes from the task's own spec or a measured spread and is written with its reason (`why`); it
is never loosened to make a run pass. Last recorded values live in last_results.json (re-taken with `rebaseline`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from alloy.bench.task_regression.types import MetricCheck, TaskCheck

_A = "alloy.bench.task_regression.adapters"
_NO_INVALID = "a physics-INVALID world is never a pass (every task's pass rule)"

T12_REPLAY = TaskCheck(
    id="T12.replay_2s", title="T12: 2 s replay of the recorded trace on Alloy's route", owner="WP3", part="cpu",
    runner=f"{_A}.t12:replay_2s", seeds=(), est_s=40.0, device="cpu",
    command="pytest tests/tasks/test_replay.py::test_two_second_replay_within_gate[t12]",
    source="tests/tasks/test_replay.py (SPEC §7 WP13 acceptance item 2); bench/traces/t12",
    metrics=(
        MetricCheck("max_point_err_mm", "band", hi=1.0, unit="mm", why="the replay gate GATE_M = 1 mm "
                    "(tests/tasks/test_replay.py:19)"),
        MetricCheck("pos_err_max_mm", "abs", tol=0.03, unit="mm", why="test_replay.py:82 pins the replay's position "
                    "error at 0.100 mm +- 0.03 mm (Newton zeroes the pair margins, 0.1 mm)"),
        MetricCheck("overflow", "band", hi=0.0, why="no contact or constraint buffer overflow"),
        MetricCheck("noslip_iterations", "band", lo=50.0, hi=50.0, why="the recording ran noslip 50 "
                    "(test_recorded_noslip_comes_from_the_vendored_scene)")))

T12_SB7_CPU = TaskCheck(
    id="T12.sb7_cpu", title="T12: one SB-7 batch of 16 random-legal 2 s rollouts on CPU (blow-up check)", owner="WP3",
    part="cpu", runner=f"{_A}.t12:sb7_cpu", seeds=(100000,), est_s=240.0, device="cpu",
    command="python -m alloy.bully sb7 --target task_t12 --device cpu --worlds 16 --episodes 16 --horizon 2.0 "
            "--seed 100000",
    source="docs/notes/wp7-results.md (SB-7 task class: task_t12, 0 S1 in 303,104 on aeac0ab1); SPEC §4.3",
    note="S2 (arrival penetration on T12, coordinator 2026-11-05 14:58Z) is reported, not judged",
    metrics=(
        MetricCheck("s1_trips", "band", hi=0.0, why="an S1 in a legal-action rollout is a failure (SPEC §4.3; T12's "
                    "counting campaign has 0 S1 in 303,104)"),
        MetricCheck("episodes", "band", lo=16.0, hi=16.0, why="the batch ran all 16 episodes")))

DC3_C13 = TaskCheck(
    id="dc3.c13_forces", title="DC3 C13 latch_d22 plug: insertion and withdrawal force at mu 0.5 (route impratio)",
    owner="WP3", part="cpu", runner=f"{_A}.dc3:c13_forces", seeds=(), est_s=90.0, device="cpu",
    command="python bench/dc/conn/c13/c13_rows.py insertion_mu0.5_route OUT; ... withdrawal_mu0.5_route OUT",
    source="docs/notes/dc3-c13-results.md:58-59, 85-87; coordinator ruling 2026-11-05 13:08Z (scr/WP3.md)",
    metrics=(
        MetricCheck("insertion_mu0.5_N", "rel", tol=0.02, unit="N", why="the plug's own wedge law (25.35 N) is the "
                    "acceptance; CPU 25.27 and L4 25.45 N sit within 0.4 % of it (coordinator 13:08Z); 2 % is about "
                    "3x the CPU-L4 spread, so a change of mechanism shows"),
        MetricCheck("withdrawal_mu0.5_N", "rel", tol=0.05, unit="N", why="withdrawal is dt-converged within 5 % "
                    "(dc3-c13-results.md:85-87)"),
        MetricCheck("withdrawal_mu0.5_N", "band", lo=10.0, hi=50.0, unit="N", why="IEC 60320-1 withdrawal band"),
        MetricCheck("invalid_worlds", "band", hi=0.0, why=_NO_INVALID)))

DC3_SFP = TaskCheck(
    id="dc3.sfp_ensemble", title="DC3 SFP+: insertion ensemble, 16 starts at rung 1 (mu 0.53, impratio 300)",
    owner="WP3", part="cpu", runner=f"{_A}.dc3:sfp_ensemble", seeds=(), est_s=60.0, device="cpu",
    command="pytest tests/tasks/test_dc_conn_sfp_forces.py -k 'ensemble_vs_f64 and 1'",
    source="tests/tasks/test_dc_conn_sfp_forces.py:99-139 (coordinator ruling 2026-11-05 23:00Z, DC3 agreed)",
    metrics=(
        MetricCheck("failed_starts", "band", hi=6.0, why="wedged or unlatched starts within MuJoCo C float64's "
                    "binomial band at rung 1 (INSERTION_F64_BAND, test_dc_conn_sfp_forces.py:105)"),
        MetricCheck("unhealthy_starts", "band", hi=2.0, why="ENERGY_JUMP or INVALID starts within float64's health "
                    "band (same line)"),
        MetricCheck("median_peak_N", "band", hi=18.0, unit="N", why="SFF-8432's insertion limit on the median peak "
                    "(test_dc_conn_sfp_forces.py:106)")))

SC_C8 = TaskCheck(
    id="SC.C8_w0", title="SC C8: AA cell into a sprung holder (YAM), world 0 on CPU (MuJoCo only, 11 s)",
    owner="WP3", part="cpu", runner=f"{_A}.sc:c8_world", seeds=(0,), est_s=180.0, device="cpu",
    command="python -c 'from alloy.clips.fold_take import take; from alloy.clips.fold_clips import c8_clip; "
            "print(take(c8_clip(1), \"cpu\", [0])[\"summary\"])'",
    source="[internal note] (15/16, one fail at 0.51 mm); alloy/clips/fold_clips.py aa_success",
    note="the batch's one failing world sits at 0.51 mm; world 0 is the check, the batch is the nightly GPU's",
    metrics=(
        MetricCheck("succeeded", "band", lo=1.0, why="seated: on the holder axis within 1.5 mm, flat within 3 deg "
                    "(fold_clips.aa_success); world 0 passes in the 15/16 take"),
        MetricCheck("valid", "band", lo=1.0, why=_NO_INVALID),
        MetricCheck("worst_pen_mm", "band", hi=0.5, unit="mm", why="the kit's penetration limit, 0.5 mm "
                    "(alloy/clips/kit/take.py:23, SC.md rule 3)")))

SC_C14 = TaskCheck(
    id="SC.C14_w0", title="SC C14: AA cell into a sprung holder (UR5e + 2F-85), world 0 on CPU (MuJoCo only, 11 s)",
    owner="WP3", part="cpu", runner=f"{_A}.sc:c14_world", seeds=(0,), est_s=240.0, device="cpu",
    command="python -c 'from alloy.clips.fold_take import take; from alloy.clips.ur_aa import c14_clip; "
            "print(take(c14_clip(1), \"cpu\", [0])[\"summary\"])'",
    source="[internal note] (16/16, pen 0.32 mm); alloy/clips/fold_clips.py aa_success",
    metrics=(
        MetricCheck("succeeded", "band", lo=1.0, why="seated: on the holder axis within 1.5 mm, flat within 3 deg "
                    "(fold_clips.aa_success); C14 took 16/16"),
        MetricCheck("valid", "band", lo=1.0, why=_NO_INVALID),
        MetricCheck("worst_pen_mm", "band", hi=0.5, unit="mm", why="the kit's penetration limit, 0.5 mm "
                    "(alloy/clips/kit/take.py:23, SC.md rule 3)")))

_SOFT_SRC = ("[internal note] SOFT-1 row 1 (6/6 ok, 0 errors, 0 warnings on CPU, 2 worlds "
             "x 0.5 s, 16-51 s each; SOFT-1 first batch, main 7dd864b6)")
_SOFT_CMD = "python -m alloy.testing.soft_film --device cpu --budget 0 --only {}"


def _soft_route(route: str) -> tuple[str, ...]:
    """SOFT-1's objects on one route, read from alloy.testing.soft_film.OBJECTS, so a library object added there joins
    its task (and its count) without a registry edit."""
    from alloy.testing.soft_film import OBJECTS  # noqa: PLC0415  (stdlib-only module)

    return tuple(name for name, r in OBJECTS.items() if r == route)


def _soft_metrics(n: int) -> tuple[MetricCheck, ...]:
    """The SOFT-1 pass rule: every object ok (0 errors; alloy.check's cards, overlap, rest drift, health, penetration
    and energy checks). Warnings are reported, not judged (not in the rule)."""
    return (MetricCheck("failed", "band", hi=0.0, why="every object ok (SOFT-1 pass rule)"),
            MetricCheck("errors", "band", hi=0.0, why="0 alloy.check errors (SOFT-1 pass rule)"),
            MetricCheck("objects", "band", lo=float(n), hi=float(n), why=f"all {n} objects checked"))


SOFT1_FILM = TaskCheck(
    id="SOFT1.film_drop", title=f"SOFT-1: the {len(_soft_route('cloth_film'))} film objects (plastic sheets, a grocery "
    "bag) dropped on the table top (cloth_film), CPU",
    owner="WP4", part="cpu", runner=f"{_A}.soft:film_drop", seeds=(), est_s=220.0, device="cpu",
    command=_SOFT_CMD.format(" ".join(_soft_route("cloth_film"))), source=_SOFT_SRC,
    note="owner: the film plug-in (alloy/solvers/film, WP4 in scripts/owners.py)",
    metrics=_soft_metrics(len(_soft_route("cloth_film"))))

SOFT1_CLOTH = TaskCheck(
    id="SOFT1.cloth_drop", title=f"SOFT-1: the {len(_soft_route('cloth_vbd'))} cloth objects (towels, cloth sheets) "
    "dropped on the table top (cloth_vbd), CPU",
    owner="WP11", part="cpu", runner=f"{_A}.soft:cloth_drop", seeds=(), est_s=190.0, device="cpu",
    command=_SOFT_CMD.format(" ".join(_soft_route("cloth_vbd"))), source=_SOFT_SRC,
    note="owner: VBD cloth (alloy/solvers/vbd, WP11 in scripts/owners.py)",
    metrics=_soft_metrics(len(_soft_route("cloth_vbd"))))

_FILL_SRC = ("[internal note] SOFT-2 row 1 (CPU 1 world x 0.3 s: 7/7 ok, warnings only "
             "rest_drift on the SPH fills; cards as merged in 882baf4a, main feb0a5b2d)")
_FILL_CMD = "python [internal note] <fill> 0.3 cpu  (each of: {})"

SOFT2_SPH = TaskCheck(
    id="SOFT2.sph_fills", title="SOFT-2: water, glycerol 80 %, olive oil resting in the open box (liquid_sph), CPU",
    owner="WP9", part="cpu", runner=f"{_A}.soft:sph_fills", seeds=(), est_s=420.0, device="cpu",
    command=_FILL_CMD.format("water_box_200ml glycerol_80_box_100ml oil_olive_box_150ml"), source=_FILL_SRC,
    note="owner: SPH (alloy/solvers/sph, WP9); the source's rest_drift warnings (WCSPH settling) are reported",
    metrics=_soft_metrics(3))

SOFT2_PASTE = TaskCheck(
    id="SOFT2.paste_fills", title="SOFT-2: honey and ketchup resting in the open box (paste_mpm), CPU",
    owner="WP12", part="cpu", runner=f"{_A}.soft:paste_fills", seeds=(), est_s=190.0, device="cpu",
    command=_FILL_CMD.format("honey_box_150ml sauce_ketchup_box_150ml"), source=_FILL_SRC,
    note="owner: MPM (alloy/solvers/mpm, WP12 until engine item 2 merges; WP9 reviews)", metrics=_soft_metrics(2))

SOFT2_GRAIN = TaskCheck(
    id="SOFT2.grain_fills", title="SOFT-2: sand and rice resting in the open box (grain_mpm), CPU",
    owner="WP12", part="nightly", runner=f"{_A}.soft:grain_fills", seeds=(), est_s=920.0, device="cpu",
    command=_FILL_CMD.format("sand_box_200ml rice_box_200ml"), source=_FILL_SRC,
    note="a CPU task in the nightly part: 909 s on the test host (sand 572 s + rice 457 s in the source's run)",
    metrics=_soft_metrics(2))

SOFT2_VISCO = TaskCheck(
    id="SOFT2.viscoplastic_fills", title="SOFT-2: mayonnaise and six shear-thinning food pastes resting in the open "
    "box (paste_mpm, five on the Herschel-Bulkley solver), CPU",
    owner="SOFT-2", part="nightly", runner=f"{_A}.soft:viscoplastic_fills", seeds=(), est_s=490.0, device="cpu",
    command=_FILL_CMD.format("sauce_mayonnaise_box_150ml batter_bread_box_150ml batter_cake_box_150ml "
                             "puree_fruit_box_150ml tomato_concentrate_box_150ml yogurt_box_150ml "
                             "peanut_butter_box_150ml"),
    source="[internal note] SOFT-2 row 1 (added 2026-11-06, main 442896119; the paste "
           "laws tests/solvers/mpm_rheology run in the gate)",
    note="a CPU task in the nightly part (481 s on the test host, 65-70 s a fill); owner: the rheology MPM solver "
         "(alloy/solvers/mpm_rheology, SOFT-2 in scripts/owners.py); objects: adapters/soft.py PASTES",
    metrics=_soft_metrics(7))

_FOOD_SRC = ("[internal note] SOFT-3 row (check_object drop tests: 8/8 ok on one L4 at 64 "
             "worlds x 2 s, dough_ball 4 worlds, main e2b52a25; CPU at the 25 s budget in notes/soft3/check_cpu_*; "
             "sponge_cellulose added 11-06, main 4e55481a, CPU ok with 0 warnings)")
_FOOD_CMD = ("python [internal note] cpu {}  (at check_object's 25 s budget; the task runs "
             "in full)")
_FOOD_NOTE = "objects: adapters/soft.py FOODS (SOFT-3 messages WP6 when it adds one); warnings reported, not judged"

SOFT3_VBD = TaskCheck(
    id="SOFT3.soft_vbd_drops", title="SOFT-3: the 8 soft food / tissue objects (cucumber, fruit, tissue pads, gel, "
    "foam, sponge) through alloy.check.check_object (soft_vbd), CPU",
    owner="WP11", part="cpu", runner=f"{_A}.soft:soft_vbd_drops", seeds=(), est_s=350.0, device="cpu",
    command=_FOOD_CMD.format("cucumber_slicing fruit_tomato fruit_peach tissue_pad_skin tissue_pad_fat "
                             "gel_block_ballistic foam_pad_pu sponge_cellulose"), source=_FOOD_SRC,
    note="owner: VBD soft bodies (alloy/solvers/vbd, WP11 in scripts/owners.py); " + _FOOD_NOTE,
    metrics=_soft_metrics(8))

SOFT3_DOUGH = TaskCheck(
    id="SOFT3.dough_drop", title="SOFT-3: dough_ball through alloy.check.check_object (paste_mpm), CPU",
    owner="WP12", part="nightly", runner=f"{_A}.soft:dough_drops", seeds=(), est_s=530.0, device="cpu",
    command=_FOOD_CMD.format("dough_ball"), source=_FOOD_SRC,
    note="a CPU task in the nightly part (519 s on the test host; paste_mpm is in the CPU part through "
    "SOFT2.paste_fills); owner: MPM (alloy/solvers/mpm, WP12; WP9 reviews); " + _FOOD_NOTE, metrics=_soft_metrics(1))

SOFT3_VISUALS = TaskCheck(
    id="SOFT3.visuals_bitwise", title="SOFT-3 visuals: 11 arm targets on the CPU route, 2 s with vs without the "
    "pack visual layer, bit for bit", owner="SOFT-3", part="cpu", runner=f"{_A}.visuals:layer_bitwise", seeds=(),
    est_s=150.0, device="cpu", command="python [internal note] OUT <targets>  (VIS=1, "
    "against a pre-layer commit; the task compares ALLOY_ROBOT_VISUALS=1 vs 0 on one commit)",
    source="[internal note] row SOFT-3 visuals (11 of 11 bitwise, main d3fd6060)",
    note="the MuJoCo C half of the source run is the gate test tests/tasks/test_emb_visuals.py",
    metrics=(MetricCheck("differing", "band", hi=0.0, why="the layer adds display geometry only: every period's "
                         "state hashes the same with and without it (source: 11 of 11 bitwise)"),
             MetricCheck("targets", "min", tol=0.0, why="every listed target ran (a removed target is a regression; "
                         "an added one is not)"),
             MetricCheck("attached", "min", tol=0.0, why="targets whose layer attached (more shapes with it on): "
                         "fewer would make the check vacuous")))

# Nightly part: GPU checks and long CPU checks, run in the reconciliation pass (coordinator ruling 2026-11-06, end of
# scr/XB4-rulings.md: the task-regression GPU part is deferred there, the nightly GPU run paused). Each GPU check is
# baselined on current main by a GPU run (`rebaseline` on an L4) before it gates anything; until then it is reported
# as unbaselined. A GPU MuJoCo run is NONE (coordinator 2026-11-06 06:23Z): it repeats bit for bit only on Alloy's
# ordered route when asked for (WP5, main 670c092b). `route=True` marks a task whose tool asks for it: T12's SB-7
# target does (alloy/bully/tasks.py, TaskTargetSpec.deterministic; covered path: nv 64 sparse) and its check fails if
# the route stays off. The others cannot ask yet, so their verdict is a sample and each tolerance covers run-to-run
# spread (docs/notes/wp5-mjw-determinism.md, "Which library scenes and clips"): DC2's Ethernet swap and DC3's QSFP
# starts run the DISCRETE integrator; JOB1's slice has a dense Jacobian and bench/jobs/slice/run.py has no route
# option; C6 and C12 are clips (kit gravcomp through the actuators, dense Jacobian: the route refuses every clip).
# est_s: wall seconds measured on one L4 (g2-standard-8, rebaseline runs 2026-11-06, kernels compiled cold) or on
# the test host's CPU (one thread). GPU checks run one at a time; the CPU ones run beside them (run.run_tasks).

T12_SB7_GPU = TaskCheck(
    id="T12.sb7_gpu", title="T12: one SB-7 batch of 2,048 random-legal 2 s rollouts (the counting campaign's first)",
    owner="WP3", part="nightly", runner=f"{_A}.t12:sb7_gpu", seeds=(100000,), est_s=620.0, device="cuda:0",
    command="python -m alloy.bully sb7 --target task_t12 --device cuda:0 --worlds 2048 --episodes 2048 --horizon 2.0 "
            "--seed 100000",
    source="docs/notes/wp7-results.md (SB-7 task class: task_t12, 0 S1 in 303,104 on aeac0ab1; batch 345 s on an L4)",
    note="S2 (arrival penetration, coordinator 2026-11-05 14:58Z) is reported, not judged", route=True,
    metrics=(
        MetricCheck("s1_trips", "band", hi=0.0, why="an S1 in a legal-action rollout is a failure (SPEC §4.3); the "
                    "counting campaign had 0 in 303,104"),
        MetricCheck("episodes", "band", lo=2048.0, hi=2048.0, why="the batch ran all 2,048 episodes"),
        MetricCheck("route_off", "band", hi=0.0, why="the target asks for WP5's ordered route (coordinator "
                    "2026-11-06 06:23Z: the regression tasks request it); off, the GPU rows stop repeating")))

DC2_ETH = TaskCheck(
    id="dc2.eth_swap_nocable", title="DC2 Ethernet swap without the cable (port 1 -> 3, v2): 64 seed-0 worlds",
    owner="WP2", part="nightly", runner=f"{_A}.dc2:eth_swap_nocable", seeds=(0,), est_s=1080.0, device="cuda:0",
    command="python bench/dc/episode/eth_batch.py --parts real --no-cable --route-profile default --nconmax 400 "
            "--njmax 300 --layouts random --worlds 1024 --seed 0 --solver v2 --max-sim-s 300 --only 0,...,63 --out DIR",
    source="docs/notes/dc2-eth-results.md:96-120 (945/1,024, 0 invalid, dc2 b1a0c790 on main aeec5640); "
           "[internal note] DC 1",
    metrics=(
        MetricCheck("success_rate", "min", tol=0.07, why="two binomial standard deviations of a 64-world rate at p "
                    "0.92 (0.068): GPU episodes are not bitwise repeatable and single worlds flip "
                    "(dc2-eth-results.md:190)"),
        MetricCheck("invalid_worlds", "band", hi=0.0, why="pass rule: 0 physics-invalid (task-regression-sources.md "
                    "DC 1)")))

DC3_QSFP = TaskCheck(
    id="dc3.qsfp_starts64", title="DC3 QSFP DAC: the first 64 starts of the seed-0 batch (pull and re-seat)",
    owner="WP3", part="nightly", runner=f"{_A}.dc3:qsfp_starts", seeds=(0,), est_s=950.0, device="cuda:0",
    command="python -m alloy.tasks.dc_conn_qsfp_bench --device cuda --starts 64 --offset 0 --seed 0",
    source="docs/notes/dc3-qsfp-results.md:105-115 (1,024/1,024, 0 INVALID, main cd43da5c)",
    metrics=(
        MetricCheck("success_rate", "min", tol=1.0 / 64.0, why="one failure in 64 allowed: the 1,024-start run had "
                    "none, so a second failure in 64 is a regression at p < 0.2 %"),
        MetricCheck("invalid_worlds", "band", hi=0.0, why=_NO_INVALID)))

JOB1_PRESS = TaskCheck(
    id="job1.first_press_w0", title="JOB1 first press: world 0 of seed 1000 to 8 s (cucumber in VBD, BladeContact)",
    owner="WP10", part="nightly", runner=f"{_A}.job1:first_press_w0", seeds=(1000,), est_s=750.0, device="cuda:0",
    command="python bench/jobs/slice/run.py --worlds 1 --seed 1000 --first 0 --max-sim-s 8 --plan clear=fingers "
            "--out DIR (assets b2956b8+)",
    source="docs/notes/wp10-blade-contact.md:123-150 (required check 2 for blade_contact changes, main 91fa7243)",
    metrics=(
        MetricCheck("invalid_worlds", "band", hi=0.0, why=_NO_INVALID),
        MetricCheck("max_pen_mm", "band", hi=0.5, unit="mm", why="no PEN_FAIL: the cut-front growth limit 0.5 mm "
                    "(wp10-blade-contact.md:137-146: passing reference max pen 0.220 mm, failing 0.53-0.56 mm)"),
        MetricCheck("cut_front_pen_mm", "band", hi=1.3, unit="mm", why="the 1.3 mm cut-front cap "
                    "(wp10-blade-contact.md:150, the required check's sweep replays)")))

SC_C12 = TaskCheck(
    id="SC.C12_batch", title="SC C12: thread through a 1 mm needle eye (2x FR3), the 16-world batch (9 s)",
    owner="WP3", part="nightly", runner=f"{_A}.sc:c12_batch", seeds=(0,), est_s=680.0, device="cuda:0",
    command="python -m alloy.clips.franka_needle --worlds 16 --seconds 9 --seed 0 --device cuda:0 --out DIR; "
            "python -m alloy.clips.precise_judge c6 DIR",
    source="[internal note] (16/16, tip 23.0-23.4 mm past; batch 547 s on an L4, C12.md:73)",
    metrics=(
        MetricCheck("succeeded", "min", tol=1.0, why="C12 took 16/16; a GPU batch is a sample (coordinator 06:23Z) "
                    "and the same command differs in detail run to run (C6.md:72), so one world may flip; two is a "
                    "regression"),
        MetricCheck("worlds", "band", lo=16.0, hi=16.0, why="the batch ran all 16 worlds"),
        MetricCheck("overlap_max_mm", "band", hi=0.1, unit="mm", why="the clip's thread/needle overlap limit "
                    "(precise_judge.judge_c6, SC.md rule 3)")))

_C6_FIXED = ("WP3 closed the known gap contact-needle-eye (f1d0fcd6a, 2026-11-07: thread/needle impedance 0.99 -> 0.999; "
             "MuJoCo C 9 starts + dt/2 0.009-0.028 mm, the L4 batch 16/16 at 0.011-0.040 mm, wp3-results C6 row)")
_C6_RULE = "the clip's thread/needle overlap limit (precise_judge.judge_c6, SC.md rule 3), back since " + _C6_FIXED

SC_C6 = TaskCheck(
    id="SC.C6_batch", title="SC C6: thread through a 1 mm needle eye (2x YAM), the 16-world batch (9 s)",
    owner="WP3", part="nightly", runner=f"{_A}.sc:c6_batch", seeds=(0,), est_s=680.0, device="cuda:0",
    command="python -m alloy.clips.precise_needle_run --worlds 16 --seconds 9 --seed 0 --device cuda:0 --out DIR; "
            "python -m alloy.clips.precise_judge c6 DIR",
    source="[internal note]:25-45 (c6take2, 16/16; take world 7: overlap 0.055 mm, MuJoCo contacts 0.125 mm); "
           "docs/notes/wp3-results.md C6 row (fixed 2026-11-07: 16/16 at 0.011-0.040 mm on an L4)",
    note="its baseline is taken in the reconciliation pass (reconcile_queue.md, alloy-dev); until then the 0.1 mm rule "
         "and the 16 worlds are checked, succeeded is not",
    metrics=(
        MetricCheck("succeeded", "min", tol=1.0, why="worlds the clip's own judge passes (overlap < 0.1 mm among its "
                    "rules) must not drop by more than one: a GPU batch is a sample (coordinator 06:23Z) and the same "
                    "command differs in detail run to run (C6.md:72)"),
        MetricCheck("worlds", "band", lo=16.0, hi=16.0, why="the batch ran all 16 worlds"),
        MetricCheck("overlap_max_mm", "band", hi=0.1, unit="mm", why=_C6_RULE)))

SC_C6_W0 = TaskCheck(
    id="SC.C6_w0", title="SC C6: world 0 alone on CPU (9 s)",
    owner="WP3", part="nightly", runner=f"{_A}.sc:c6_world", seeds=(0,), est_s=1250.0, device="cpu",
    command="python -m alloy.clips.precise_needle_run --worlds 1 --seconds 9 --seed 0 --device cpu --out DIR; "
            "python -m alloy.clips.precise_judge c6 DIR",
    source="docs/notes/wp6-results.md C6-BISECT (world 0 on CPU: 0.08975 mm before the carry, 0.10755 mm after, at "
           "impedance 0.99); docs/notes/wp3-results.md C6 row (MuJoCo Warp on CPU, carry on, 0.015 / 0.021 mm at 0.999)",
    note="a CPU task in the nightly part (~20 CPU-min, one thread); deterministic on a build, so any change of the "
         "peak names a commit",
    metrics=(
        MetricCheck("overlap_max_mm", "band", hi=0.1, unit="mm", why=_C6_RULE),
        MetricCheck("worlds", "band", lo=1.0, hi=1.0, why="world 0 ran"),
        MetricCheck("mj_pen_max_mm", "band", hi=0.5, unit="mm", why="the clip's MuJoCo penetration limit, 0.5 mm "
                    "(precise_judge.judge_c6)")))

DC1_SCENE = TaskCheck(
    id="dc1.eth_scene", title="DC1 Ethernet scene on CPU: start latched, 89 N hold, release, insert (S4), latch, pen",
    owner="WP2", part="nightly", runner=f"{_A}.dc1:eth_scene", seeds=(), est_s=1100.0, device="cpu",
    command="python bench/dc/eth/checks.py scene  (rows only: the adapter calls scene_rows('cpu'), no merge)",
    source="docs/notes/dc1-results.md:47-53 (S1-S6, pass); [internal note] DC 8",
    note="a CPU task in the nightly part: ~18 CPU-min, too long for the post-merge run; it runs beside the GPU lane",
    metrics=(
        MetricCheck("rows_failed", "band", hi=0.0, why="every S row passes its own rule (bench/dc/eth/checks.py "
                    "scene_metric_rows: latched, slip <= 1.5 mm, released < 20 N, insert <= 20 N at <= 0.1 m/s, "
                    "latched in port 3, overlap <= 1.0 mm, 0 INVALID)"),
        MetricCheck("insert_max_speed_mps", "band", hi=0.1, unit="m/s", why="the no-slam guard (S4)"),
        MetricCheck("insert_peak_N", "rel", tol=0.10, unit="N", why="the S4 peak moves 9 % from dt to dt/2 (1.38 / "
                    "1.50 N, dc1-results.md:51); a larger move at the same dt is a model change"),
        MetricCheck("invalid_worlds", "band", hi=0.0, why="pass rule: 0 INVALID (HealthMonitor)")))

SC_C17 = TaskCheck(
    id="SC.C17_w0", title="SC C17: bone screw into a locking plate (UR5e + driver), world 0 on CPU (22 s)",
    owner="WP3", part="nightly", runner=f"{_A}.sc:c17_world", seeds=(0,), est_s=2150.0, device="cpu",
    command="python -m alloy.clips.bone_run --worlds 1 --seconds 22 --seed 0 --device cpu --out DIR",
    source="[internal note]:18-26 (c17f 16/16: flush 0.0070-0.0093 mm, thread overlap 1.80-3.48 um)",
    note="a CPU task in the nightly part: ~35 CPU-min; world 0 alone on main 53706c9b gave the take's events "
         "(land 13.433 s, caught 16.100 s, flush 19.383 s)",
    metrics=(
        MetricCheck("works", "band", lo=1.0, why="flush and the thread pair under its max_penetration "
                    "(bone_judge.world_results)"),
        MetricCheck("lost", "band", hi=0.0, why="the screw stays on the driver"),
        MetricCheck("h_end_mm_max", "band", hi=0.02, unit="mm", why="flush <= 0.02 mm (the SC list's C17 rule; the "
                    "depth stop, C17.md:19)"),
        MetricCheck("thread_pen_max_um", "band", hi=5.12, unit="um", why="the pair's max_penetration: 6H/6g "
                    "clearance / 4 (bone_judge.MAX_PEN, SC.md rule 3)")))

TASKS: tuple[TaskCheck, ...] = (T12_REPLAY, T12_SB7_CPU, DC3_C13, DC3_SFP, SC_C8, SC_C14, SOFT1_FILM, SOFT1_CLOTH,
                                SOFT2_SPH, SOFT2_PASTE, SOFT3_VBD, SOFT3_VISUALS,
                                T12_SB7_GPU, DC2_ETH, DC3_QSFP, JOB1_PRESS, SC_C12, SC_C6, SC_C6_W0, DC1_SCENE, SC_C17,
                                SOFT2_GRAIN, SOFT3_DOUGH, SOFT2_VISCO)


NO_ADAPTERS = ("no task-regression adapters are installed in this copy: the registered task checks are built from "
               "tasks that are not distributed yet; public ones will be added under "
               "alloy/bench/task_regression/adapters/")


def available(tasks: Sequence[TaskCheck] = TASKS) -> tuple[TaskCheck, ...]:
    """The tasks whose adapter module is installed; a copy without a task's adapter skips that task cleanly.

    >>> all(t.runner.partition(":")[0].startswith("alloy.") for t in available())
    True
    """
    return tuple(t for t in tasks if importlib.util.find_spec(t.runner.partition(":")[0]) is not None)


def get(task_id: str) -> TaskCheck:
    for t in TASKS:
        if t.id == task_id:
            return t
    raise KeyError(f"no task-regression task {task_id!r} (known: {', '.join(t.id for t in TASKS)})")


__all__ = ["NO_ADAPTERS", "TASKS", "available", "get"]
