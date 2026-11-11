# SPDX-License-Identifier: Apache-2.0
"""Task-regression suite (coordinator 2026-11-06 05:55Z): tolerances judged against the last result, the store's
reason rule, the report's green flag and owners, and the registry's own rules (no tolerance without a source)."""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from alloy.bench.task_regression import registry, run, store
from alloy.bench.task_regression import one as ONE
from alloy.bench.task_regression.types import LastResult, MetricCheck, TaskCheck, judge


def _task(*metrics: MetricCheck, part: str = "cpu") -> TaskCheck:
    return TaskCheck(id="X.t", title="t", owner="WP9", part=part, runner="m:f", seeds=(0,), metrics=metrics,
                     est_s=1.0, device="cpu", command="c", source="s")


def test_each_rule_bounds_the_value_around_the_last_result():
    last = LastResult({"rate": 0.9, "pen_mm": 1.0, "n": 25.0}, "abc", "d", "s")
    t = _task(MetricCheck("rate", "min", tol=0.05), MetricCheck("pen_mm", "max", tol=0.1),
              MetricCheck("n", "rel", tol=0.02), MetricCheck("pen_mm", "band", hi=1.5))
    assert judge(t, last, {"rate": 0.86, "pen_mm": 1.1, "n": 25.4}).verdict == "pass"
    assert judge(t, last, {"rate": 2.0, "pen_mm": 0.0, "n": 24.6}).verdict == "pass"     # better is never a fail
    for bad in ({"rate": 0.84}, {"pen_mm": 1.11}, {"n": 25.6}):
        got = judge(t, last, {"rate": 0.9, "pen_mm": 1.0, "n": 25.0, **bad})
        assert got.verdict == "fail", bad
    assert judge(t, last, {"rate": 0.9, "pen_mm": 1.0}).verdict == "fail"               # a missing metric fails
    assert judge(t, last, {"rate": math.nan, "pen_mm": 1.0, "n": 25.0}).verdict == "fail"


def test_a_task_without_a_last_result_is_reported_not_gated():
    t = _task(MetricCheck("rate", "min", tol=0.05), MetricCheck("pen_mm", "band", hi=1.5))
    out = judge(t, None, {"rate": 0.0, "pen_mm": 0.2})
    assert out.verdict == "unbaselined" and not out.gates
    assert judge(t, None, {"rate": 0.0, "pen_mm": 9.0}).verdict == "fail"               # a band still fails


def test_recording_needs_a_reason_and_keeps_the_history(tmp_path):
    path = tmp_path / "last.json"
    with pytest.raises(ValueError):
        store.record("X.t", {"a": 1.0}, build="b1", source="s", reason="", path=path)
    store.record("X.t", {"a": 1.0}, build="b1", source="s", reason="first", path=path)
    store.record("X.t", {"a": 2.0}, build="b2", source="s", reason="ruled change", path=path)
    assert store.last_result("X.t", path).values == {"a": 2.0}
    hist = store.load(path)["history"]
    assert [h["reason"] for h in hist] == ["first", "ruled change"] and hist[1]["replaced"] == {"a": 1.0}


def test_report_is_green_only_without_fail_or_error_and_names_owners(tmp_path, monkeypatch):
    t1 = _task(MetricCheck("rate", "band", lo=0.5))
    t2 = TaskCheck(**{**t1.__dict__, "id": "X.u", "owner": "WP3"})
    monkeypatch.setattr(run, "TASKS", (t1, t2))
    monkeypatch.setattr(run, "available", lambda tasks: tuple(tasks))  # fakes have no adapter module
    monkeypatch.setattr(run.store, "last_result", lambda task_id: None)
    results = {"X.t": ({"rate": 0.9}, 1.0, ""), "X.u": (None, 2.0, "timed out after 300 s")}
    monkeypatch.setattr(run, "measure", lambda task, work: results[task.id])
    rep = run.run_part("cpu", out_dir=tmp_path)
    assert not rep["green"] and rep["counts"] == {"pass": 1, "fail": 0, "error": 1, "unbaselined": 0}
    assert rep["regressions"][0]["task"] == "X.u" and rep["regressions"][0]["owner"] == "WP3"
    assert json.loads((tmp_path / "report.json").read_text())["schema"] == run.SCHEMA
    assert "| X.u | WP3 | error |" in (tmp_path / "report.md").read_text()
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert rows == rep["rows"] and rows[0]["metric"] == "rate" and rows[0]["tolerance"] == [0.5, None]
    assert rows[1] == {**rows[1], "task": "X.u", "owner_wp": "WP3", "metric": None, "pass": False, "device": "cpu"}
    results["X.u"] = ({"rate": 0.7}, 2.0, "")
    assert run.run_part("cpu", out_dir=tmp_path)["green"]


def test_one_writes_metrics_or_the_error(tmp_path, monkeypatch):
    t = _task(MetricCheck("rate", "band", lo=0.5))
    monkeypatch.setattr(ONE, "get", lambda task_id: t)
    monkeypatch.setattr(ONE, "resolve", lambda runner: lambda seeds, device: {"rate": 0.75})
    assert ONE.main(["--task", "X.t", "--json", str(tmp_path / "a.json")]) == 0
    assert json.loads((tmp_path / "a.json").read_text())["metrics"] == {"rate": 0.75}

    def boom(seeds, device):
        raise RuntimeError("solver blew up")
    monkeypatch.setattr(ONE, "resolve", lambda runner: boom)
    assert ONE.main(["--task", "X.t", "--json", str(tmp_path / "b.json")]) == 2
    assert "solver blew up" in json.loads((tmp_path / "b.json").read_text())["error"]


def test_every_registered_task_states_where_each_tolerance_comes_from():
    ids = [t.id for t in registry.TASKS]
    assert len(ids) == len(set(ids)) and ids
    for t in registry.TASKS:
        assert t.part in ("cpu", "nightly") and t.source and t.command and ":" in t.runner, t.id
        assert t.metrics and all(m.why for m in t.metrics), t.id
        assert t.part == "nightly" or t.est_s <= 600, (t.id, "a CPU task must stay minutes-sized")
    for t in registry.available():
        ONE.resolve(t.runner)                                           # an installed adapter imports


def test_last_results_name_registered_tasks_only():
    data = store.load()
    assert set(data["tasks"]) <= {t.id for t in registry.TASKS}
    for task_id, entry in data["tasks"].items():
        names = {m.name for m in registry.get(task_id).metrics}
        assert names <= set(entry["values"]), (task_id, names - set(entry["values"]))


def test_soft1_tasks_count_every_soft_film_object_on_their_route(monkeypatch):
    from alloy.testing import soft_film as SF

    for task, route in ((registry.SOFT1_FILM, "cloth_film"), (registry.SOFT1_CLOTH, "cloth_vbd")):
        names = [n for n, r in SF.OBJECTS.items() if r == route]
        band = next(m for m in task.metrics if m.name == "objects")
        assert band.lo == band.hi == len(names), (task.id, band, names)
        assert all(n in task.command for n in names), task.command
    monkeypatch.setitem(SF.OBJECTS, "film_new_sheet", "cloth_film")       # a library object added later joins its task
    assert registry._soft_route("cloth_film")[-1] == "film_new_sheet"


def test_soft2_and_soft3_bands_count_the_adapter_lists():
    from alloy.bench.task_regression.adapters import soft

    for task, route, objects in ((registry.SOFT2_SPH, "liquid_sph", soft.FILLS),
                                 (registry.SOFT2_PASTE, "paste_mpm", soft.FILLS),
                                 (registry.SOFT2_GRAIN, "grain_mpm", soft.FILLS),
                                 (registry.SOFT2_VISCO, "paste_mpm", soft.PASTES),
                                 (registry.SOFT3_VBD, "soft_vbd", soft.FOODS),
                                 (registry.SOFT3_DOUGH, "paste_mpm", soft.FOODS)):
        band = next(m for m in task.metrics if m.name == "objects")
        assert band.lo == band.hi == sum(r == route for r in objects.values()), task.id


def test_a_box_fill_fails_when_the_router_moves_it_off_its_listed_route(monkeypatch):
    import alloy.check as AC
    from alloy.bench.task_regression.adapters import soft

    class Report:
        ok = True

        def to_dict(self):
            return {"checks": {}}

        def errors(self):
            return []

        def warnings(self):
            return []

    monkeypatch.setattr(AC, "check", lambda *a, **k: Report())
    routes = {"honey_box_150ml": "paste_mpm", "sauce_ketchup_box_150ml": "liquid_sph"}
    monkeypatch.setattr(soft, "_routed", lambda fill, device: routes[fill])
    assert soft.paste_fills((), "cpu") == {"objects": 2.0, "failed": 1.0, "errors": 1.0, "warnings": 0.0}


def test_visuals_rows_name_the_first_differing_period():
    from alloy.bench.task_regression.adapters import visuals as V

    same = V._compare("t", {"hashes": ["a", "b"], "shapes": 9, "wall_s": 1}, {"hashes": ["a", "b"], "shapes": 7,
                                                                             "wall_s": 1})
    moved = V._compare("t", {"hashes": ["a", "b"], "shapes": 7, "wall_s": 1}, {"hashes": ["a", "c"], "shapes": 7,
                                                                              "wall_s": 1})
    assert (same["identical"], same["first_differing_period"]) == (True, None)
    assert (moved["identical"], moved["first_differing_period"]) == (False, 1)


def test_cli_maps_gpu_to_nightly_and_exits_2_before_running_on_a_bad_environment(tmp_path, monkeypatch):
    from alloy.bench.task_regression import __main__ as CLI

    assert CLI._part("gpu") == CLI._part("reconcile") == "nightly" and CLI._part("cpu") == "cpu"
    monkeypatch.setattr(CLI, "run_part", lambda *a, **k: pytest.fail("ran tasks on a bad environment"))
    monkeypatch.setattr(CLI, "_preflight", lambda: "installed Newton abc is not the pin def")
    assert CLI.main(["run", "--part", "cpu", "--out", str(tmp_path)]) == 2
    monkeypatch.setattr(CLI, "_preflight", lambda: "")
    assert CLI.main(["run", "--part", "gpu", "--only", "NO.such_task", "--out", str(tmp_path)]) == 2


def test_rows_say_how_a_task_repeats_and_a_routed_task_checks_its_route(monkeypatch):
    from alloy.bench import gpu_repeat as GR
    from alloy.bench.task_regression.adapters.t12 import route_on

    t = _task(MetricCheck("rate", "band", lo=0.5), part="nightly")
    routed = TaskCheck(**{**t.__dict__, "id": "X.r", "route": True})
    monkeypatch.setattr(run, "TASKS", (t, routed))
    monkeypatch.setattr(run, "available", lambda tasks: tuple(tasks))  # fakes have no adapter module
    rows = run.flat_rows([judge(x, None, {"rate": 0.9}) for x in (t, routed)], {"X.t": "cuda:0", "X.r": "cuda:0"})
    assert [r["repeats"] for r in rows] == [GR.repeats("cuda:0"), GR.repeats("cuda:0", route_requested=True)]
    info = "alloy.precision INFO mjw_route: SolverMuJoCo enabled (1 worlds, nv 64)\n"   # the scene-info build's line
    assert not route_on(info, 2048) and route_on(info + info.replace("(1 worlds", "(2048 worlds"), 2048)
    for task in registry.TASKS:
        if task.route:
            assert task.device.startswith("cuda") and "route_off" in {m.name for m in task.metrics}, task.id


def test_a_task_keeps_its_per_world_rows_in_its_run_folder(tmp_path, monkeypatch):
    from alloy.bench.task_regression import adapters as AD

    src = tmp_path / "rows.json"
    src.write_text("[]")
    monkeypatch.delenv(AD.WORK_ENV, raising=False)
    AD.keep(src)
    AD.keep_json("w.json", [1])                                         # outside a suite run: nothing is written
    seen: dict[str, str] = {}

    def fake_run(cmd, **kw):
        seen.update(kw["env"])
        raise run.subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(run.subprocess, "run", fake_run)
    assert run.measure(_task(MetricCheck("rate", "band", lo=0.5)), tmp_path / "w")[2].startswith("timed out")
    monkeypatch.setenv(AD.WORK_ENV, seen[AD.WORK_ENV])
    AD.keep(src, "kept.json")
    AD.keep_json("w.json", [{"world": 0}])
    assert sorted(p.name for p in (tmp_path / "w" / "X.t").iterdir()) == ["kept.json", "w.json"]


def test_the_qsfp_bench_can_write_its_row_under_the_adapters_folder(tmp_path, monkeypatch):
    """The bench runs git in ROWS.parents[3]: a shallow temporary ROWS raised IndexError after a 13 min L4 episode."""
    from alloy.bench.task_regression.adapters.dc3 import qsfp_rows
    from alloy.tasks import dc_conn_qsfp_bench as QB

    monkeypatch.setattr(QB, "ROWS", qsfp_rows(tmp_path))
    monkeypatch.setenv("QSFP_COMMIT", "abc12345")
    assert QB.record("starts_chunks", {"success_rate": 1.0})["commit"] == "abc12345"
    monkeypatch.delenv("QSFP_COMMIT")
    QB.record("starts_chunks", {"success_rate": 0.5})                  # git outside a repo: no commit, no error
    assert len((tmp_path / "bench/dc/conn/qsfp/starts_chunks.jsonl").read_text().splitlines()) == 2


def test_gpu_tasks_run_one_at_a_time_and_cpu_tasks_beside_them(tmp_path, monkeypatch):
    import threading

    base = _task(MetricCheck("rate", "band", lo=0.5), part="nightly")
    gpu1, gpu2 = (TaskCheck(**{**base.__dict__, "id": f"G.{k}", "device": "cuda:0"}) for k in (1, 2))
    cpu = TaskCheck(**{**base.__dict__, "id": "C.1", "device": "cpu"})
    lock, active, peak, cpu_ran = threading.Lock(), [0], [0], threading.Event()

    def fake_measure(task, work):
        if task.device == "cpu":
            cpu_ran.set()
            return {"rate": 0.9}, 0.0, ""
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        assert cpu_ran.wait(10), "the CPU task did not run beside the GPU lane"
        with lock:
            active[0] -= 1
        return {"rate": 0.9}, 0.0, ""
    monkeypatch.setattr(run, "measure", fake_measure)
    monkeypatch.setattr(run.store, "last_result", lambda task_id: None)
    outcomes = run.run_tasks([gpu1, gpu2, cpu], tmp_path, jobs=1)
    assert [o.task for o in outcomes] == ["G.1", "G.2", "C.1"] and peak[0] == 1


def test_a_copy_without_adapters_skips_cleanly_and_says_so(monkeypatch, capsys):
    from alloy.bench.task_regression import __main__ as CLI

    gone = dataclasses.replace(registry.TASKS[0], id="X.gone", runner="alloy.bench.task_regression.adapters.gone:f")
    assert registry.available((gone,)) == ()
    monkeypatch.setattr(CLI, "available", lambda tasks=(): ())
    assert CLI.main(["list"]) == 0 and "no task-regression adapters" in capsys.readouterr().out
