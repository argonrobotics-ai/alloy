# SPDX-License-Identifier: Apache-2.0
"""CLI (`python -m alloy.bench run|scoreboard`), solver settings and the quick-suite plumbing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alloy.bench import cli
from alloy.bench import solvers as SV
from alloy.core.errors import LawError


def test_solver_settings_defaults_and_overrides() -> None:
    s = SV.SolverSettings("featherstone")
    assert s.merged_kwargs() == {"angular_damping": 0.0} and s.uses_pipeline()
    m = SV.SolverSettings("mujoco", {"cone": "elliptic"}, newton_contacts=True)
    assert m.merged_kwargs() == {"cone": "elliptic", "use_mujoco_contacts": False} and m.uses_pipeline()
    assert not SV.SolverSettings("mujoco").uses_pipeline()
    assert SV.SolverSettings("xpbd", {"iterations": 3}).describe()["kwargs"]["iterations"] == 3
    with pytest.raises(LawError):
        SV._solver_class("nope")


def test_cli_run_scoreboard_and_baseline(tmp_path, capsys) -> None:
    out, base = tmp_path / "res", tmp_path / "base.json"
    args = ["run", "--suite", "analytic", "--law", "A1", "--solver", "xpbd", "--quick", "--out", str(out),
            "--baseline", str(base), "--add-new-keys"]
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert "A16.selftest" in text and "A1.freefall" in text and "NEW A1.freefall|xpbd|cpu|quick (added)" in text
    entries = json.loads(base.read_text())["entries"]
    assert "A1.freefall|xpbd|cpu|quick" in entries
    assert cli.main(args[:-1]) == 0  # second run: compare only, no regression
    assert "REGRESSION" not in capsys.readouterr().out
    reports = list(out.rglob("*.json"))
    assert any(p.name.startswith("A1.freefall.xpbd") for p in reports)
    assert cli.main(["scoreboard", str(out), "--out", str(tmp_path / "SB.json")]) == 0
    rows = json.loads((tmp_path / "SB.json").read_text())["rows"]
    assert any(r["metric_id"] == "BEN-1" and r["verdict"] == "pass" for r in rows)


def test_cli_seeded_regression_exits_nonzero(tmp_path, capsys) -> None:
    out, base = tmp_path / "res", tmp_path / "base.json"
    run = ["run", "--law", "A1", "--solver", "xpbd", "--quick", "--out", str(out), "--baseline", str(base)]
    assert cli.main(run + ["--update-baseline", "initial"]) == 0
    data = json.loads(base.read_text())
    entry = data["entries"]["A1.freefall|xpbd|cpu|quick"]
    entry["metrics"]["pos_err_ratio"]["error"] = 0.0  # pretend the baseline was perfect
    entry["metrics"]["pos_err_ratio"]["bound"] = 0.1
    base.write_text(json.dumps(data))
    assert cli.main(run) == 1
    assert "REGRESSION A1.freefall|xpbd|cpu|quick: pos_err_ratio error" in capsys.readouterr().out


@pytest.mark.parametrize("broken", [True, False])
def test_the_gate_fails_a_planted_failing_law_that_has_no_key(tmp_path, capsys, broken) -> None:
    """Audit AB-laws-1: the gate's own command (ci_cpu.sh: `run --suite analytic --device cpu --quick --baseline
    bench/goldens/analytic_baseline.json`) on a law the committed baseline has no key for. Planted broken (its
    measured errors tripled, as a halved stiffness would) it must exit 1; the same law unbroken is NEW and passes."""
    import dataclasses

    from alloy.bench import baseline as BL
    from alloy.bench import law as L
    from alloy.bench.analytic import freefall

    class Planted(freefall.FreeFallTwin):
        law_id = "A1.planted"

        def scalars(self, log):
            return {k: (3.0 * v + 1.0 if broken else v) for k, v in super().scalars(log).items()}

    key, committed = "A1.planted|xpbd|cpu|quick", Path(__file__).resolve().parents[2] / BL.DEFAULT_PATH
    assert committed.exists() and key not in BL.load_baseline(committed)["entries"]
    before = committed.read_bytes()
    L.discover()
    L.register_law(dataclasses.replace(freefall.LAW, id="A1.planted", solvers=("xpbd",)), Planted)
    try:
        code = cli.main(["run", "--suite", "analytic", "--device", "cpu", "--quick", "--law", "A1.planted",
                         "--baseline", str(committed), "--out", str(tmp_path)])
    finally:
        L.unregister_law("A1.planted")
    text = capsys.readouterr().out
    assert committed.read_bytes() == before                   # compare mode never writes the baseline
    assert f"NEW {key}: {'fail' if broken else 'pass'}" in text, text
    assert code == (1 if broken else 0), text
    assert (f"REGRESSION {key}: no baseline key and its own verdict is fail" in text) == broken, text


def test_cli_add_new_keys_records_a_known_failure(tmp_path, capsys) -> None:
    """`--add-new-keys` is the explicit way in for a law that fails today: the key is written with its fail verdict
    (reviewed with the baseline file) and later runs compare against it."""
    from alloy.bench.analytic import selftest

    base = tmp_path / "base.json"
    run = ["run", "--law", "A16", "--quick", "--out", str(tmp_path / "res"), "--baseline", str(base)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(selftest, "run_cases", lambda: (0.5, {"x": "got fail want pass"}))
        assert cli.main(run) == 1 and "REGRESSION A16.selftest|none|cpu|quick: no baseline key" in capsys.readouterr().out
        assert not base.exists()
        assert cli.main(run + ["--add-new-keys"]) == 0
        assert "NEW A16.selftest|none|cpu|quick (added): fail" in capsys.readouterr().out
        assert json.loads(base.read_text())["entries"]["A16.selftest|none|cpu|quick"]["verdict"] == "fail"
        assert cli.main(run) == 0                                    # the known failure, under its key


def test_cli_whole_suite_run_fails_when_a_keyed_law_has_no_result(tmp_path, capsys, monkeypatch) -> None:
    """A whole-suite run (the gate's: no --law, no --solver) with a baseline key it has no result for exits 1; a
    filtered run does not look for the laws it left out."""
    from alloy.bench import baseline as BL
    from alloy.bench import law as L

    base, out = tmp_path / "base.json", str(tmp_path / "res")
    assert cli.main(["run", "--law", "A16", "--quick", "--out", out, "--baseline", str(base), "--add-new-keys"]) == 0
    data = json.loads(base.read_text())
    gone = dict(data["entries"]["A16.selftest|none|cpu|quick"], law="A1.gone", solver="xpbd")
    data["entries"]["A1.gone|xpbd|cpu|quick"] = gone
    base.write_text(json.dumps(data))
    assert cli.main(["run", "--law", "A16", "--quick", "--out", out, "--baseline", str(base)]) == 0
    monkeypatch.setattr(L, "registered_laws", lambda suite=None: [
        it for it in L._REGISTRY.values() if it[0].id == "A16.selftest" and suite in (None, it[0].suite)])
    monkeypatch.setattr(L, "missing_laws", lambda suite=None: [])
    capsys.readouterr()
    assert cli.main(["run", "--quick", "--out", out, "--baseline", str(base)]) == 1
    assert "REGRESSION A1.gone|xpbd|cpu|quick: in the baseline (pass) but this run has no result" in capsys.readouterr().out
    BL.drop_keys(["A1.gone|xpbd|cpu|quick"], base, reason="retired (test)")
    assert cli.main(["run", "--quick", "--out", out, "--baseline", str(base)]) == 0


def test_cli_blocks_everything_when_a16_fails(tmp_path, monkeypatch, capsys) -> None:
    from alloy.bench.analytic import selftest

    monkeypatch.setattr(selftest, "run_cases", lambda: (0.5, {"x": "got fail want pass"}))
    assert cli.main(["run", "--law", "A1", "--solver", "xpbd", "--quick", "--out", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "A16.selftest" in text and "A1.freefall                  xpbd           error" in text


def test_unmerged_producer_shows_not_run(tmp_path, capsys, monkeypatch) -> None:
    # a catalogue entry whose producer module does not exist, so the test holds whichever real producers have merged
    from alloy.bench import law as L

    fake = L.CatalogueEntry("A98", "analytic", "WP98", "alloy.never_merged_pkg.laws", "test-only family")
    monkeypatch.setattr(L, "CATALOGUE", L.CATALOGUE + (fake,))
    assert cli.main(["run", "--law", "A98", "--quick", "--out", str(tmp_path)]) == 0
    text = capsys.readouterr().out
    assert "A98                          any            not_run" in text
    rep = json.loads(next(tmp_path.rglob("A98.any.cpu.json")).read_text())
    assert "alloy.never_merged_pkg.laws not merged" in rep["monitors"]["not_run_reason"]


def test_cli_baseline_from_reports(tmp_path) -> None:
    out, base = tmp_path / "res", tmp_path / "base.json"
    assert cli.main(["run", "--law", "A1", "--solver", "featherstone", "--quick", "--out", str(out)]) == 0
    assert cli.main(["baseline", str(out), "--baseline", str(base)]) == 0
    entries = json.loads(base.read_text())["entries"]
    assert "A1.freefall|featherstone|cpu|quick" in entries and "A16.selftest|none|cpu|quick" in entries


def test_command_records_the_cli_arguments(tmp_path) -> None:
    out = tmp_path / "res"
    assert cli.main(["run", "--law", "A16", "--quick", "--out", str(out)]) == 0
    rep = json.loads(next(out.rglob("A16.selftest*.json")).read_text())
    assert rep["command"].startswith("python -m alloy.bench run --law A16 --quick")


def test_mujoco_is_built_as_the_route_builds_it() -> None:
    # rk4 entries get SyncedMuJoCo (no per-step float32 state push, WP2 3e803f3); others stock SolverMuJoCo
    from alloy._vendor import newton
    import warp as wp

    from alloy.bench import default_route as DR
    from alloy.bench import solvers as SV

    b = newton.ModelBuilder()
    SV.prepare_builder("mujoco", b)
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    b.add_shape_box(body, hx=0.05, hy=0.1, hz=0.15)
    model = b.finalize(device="cpu")
    routed = DR.routed_mujoco_settings(model)
    assert routed.kwargs["integrator"] == "rk4"
    assert type(SV.make_solver(model, routed)).__name__ == "SyncedMuJoCo"
    plain = SV.make_solver(model, SV.SolverSettings("mujoco", {**DR.mujoco_default_kwargs()}))
    assert type(plain).__name__ == "SolverMuJoCo"



def test_routed_mujoco_settings_take_the_route_fill_ins(monkeypatch) -> None:
    """SCR-WP3-10: a scene with no MJCF file gets the route's impratio fill-in; the A5 kinetic laws follow it."""
    from alloy.bench import default_route as DR
    from alloy.scene import entries

    calls: list[tuple[object, ...]] = []

    def fake(effects, has_precision, mjcf=None, chain_newton=False):
        calls.append((effects, has_precision, mjcf, chain_newton))
        return {"impratio": (300.0, "fill-in when no file sets one (SCR-WP3-10)")}

    monkeypatch.setattr(entries, "mujoco_overrides", fake)
    DR.route_fill_ins.cache_clear()
    try:
        assert DR.routed_mujoco_settings().kwargs["impratio"] == 300.0
        assert calls == [(frozenset(), False, None, False)]  # no file, no precision pair, no route effect
        assert "impratio 300 where no file sets one" in DR.describe()
    finally:
        DR.route_fill_ins.cache_clear()
    assert DR.routed_mujoco_settings().kwargs.get("impratio") == DR.route_fill_ins().get("impratio")


def test_a_chain_on_newton_gets_the_routes_held_back_fill_ins() -> None:
    """The route holds the impratio fill-in back for a CHAIN on Newton (A10.capstan's rope went NaN with it)."""
    from alloy.bench import default_route as DR
    from alloy.scene import entries

    want = {k: v for k, (v, _) in entries.mujoco_overrides(frozenset(), False, None, chain_newton=True).items()}
    assert DR.route_fill_ins(chain_newton=True) == want
    assert "impratio" not in DR.routed_mujoco_settings(chain_newton=True).kwargs
    assert DR.routed_mujoco_settings().kwargs.get("impratio") == 300.0
