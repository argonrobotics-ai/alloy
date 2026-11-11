# SPDX-License-Identifier: Apache-2.0
"""Audit AB-laws-2: a law's reference must not read its constants back from the engine under test.

Each test plants the engine mistake that used to move the simulation and the reference together (so the law passed)
and checks the law now fails, its reference being the law's own numbers. A1 (gravity) and A2 (bob inertia and arm)
are in test_laws_cpu.py; this file holds the laws of other producers that WP6 pinned (rows: docs/notes/wp6-results.md,
the audit table: [internal note] scr/WP6.md). Fast checks run in every gate; the stepped liquid and thread plants are slow.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from alloy.bench.runner import run_law

pytestmark = pytest.mark.law


def _metric(res, name):
    return next(v for v in res.verdicts if v.name == name)


def test_a8_push_bound_does_not_follow_the_engines_depth_rule(monkeypatch) -> None:
    """A8.push: the 50 um bound is the law's literal (the D2 rule gives the same today). With the rule returning twice
    the depth, the pair's stiffness halves, so the peg sinks deeper, and the bound stays; it used to double with it."""
    from alloy.contact import laws as CL
    from alloy.contact.law_twins import push_pair
    from alloy.contact.types import PrecisionPairSpec

    assert CL._MAX_PEN_M == 5.0e-5 and CL.A8_PUSH.metrics["pen_max_m"].abs == 5.0e-5
    k0 = push_pair().pair_stiffness
    assert push_pair().resolved_max_penetration() == pytest.approx(5.0e-5, rel=1e-12)   # the engine rule, today
    rule = PrecisionPairSpec.resolved_max_penetration
    monkeypatch.setattr(PrecisionPairSpec, "resolved_max_penetration", lambda self: 2.0 * rule(self))
    assert push_pair().pair_stiffness == pytest.approx(0.5 * k0)
    assert CL.A8_PUSH.metrics["pen_max_m"].abs == 5.0e-5 and "resolved_max_penetration()" not in "".join(
        ln.split("#")[0] for ln in open(CL.__file__))             # the bound is no call into that rule


def _plant_pitch(monkeypatch, factor: float) -> None:
    """Every importer of the thread-id parser gets a pitch `factor` times the id's (the thread SDF and the twins)."""
    from alloy.contact import analytic_pair, analytic_samples, thread_sdf, thread_twins

    parse = thread_sdf.parse_thread_id

    def wrong(analytic_id: str):
        spec = parse(analytic_id)
        return dataclasses.replace(spec, pitch=factor * spec.pitch)

    for mod in (thread_sdf, analytic_pair, analytic_samples, thread_twins):
        monkeypatch.setattr(mod, "parse_thread_id", wrong)


def test_a9_reference_is_the_laws_own_thread(monkeypatch) -> None:
    """A9: advance p and the power-screw torque come from the law's M10x1.5 literals, which equal the parsed thread
    today (pitch and basic pitch diameter d - 3/4 H). A parser that reads the pitch 2 % long no
    longer moves the reference."""
    from alloy.contact import thread_twins as tt

    twin, cond = tt.ThreadA9Twin(), None
    cond = twin.conditions(True)[0]
    spec, ref = tt.parse_thread_id(tt.BOLT_ID), twin.reference(cond)
    assert spec.pitch == pytest.approx(tt.A9_PITCH_M, rel=1e-12) and spec.d2 == pytest.approx(tt.A9_D2_M, rel=1e-9)
    assert ref["advance_per_rev"] == 1.5e-3 and tt.A9_D2_M == pytest.approx(9.026e-3, abs=5e-7)   # ISO 724's table
    k, d2, p = cond.params["mu"] / math.cos(math.pi / 6), 10.0e-3 - 0.75 * math.sqrt(3.0) / 2.0 * 1.5e-3, 1.5e-3
    assert ref["rundown_torque"] == pytest.approx(cond.params["F"] * 0.5 * d2 * (p + math.pi * k * d2)
                                                  / (math.pi * d2 - k * p), rel=1e-12)
    _plant_pitch(monkeypatch, 1.02)
    assert tt.parse_thread_id(tt.BOLT_ID).pitch == pytest.approx(1.53e-3)
    assert tt.ThreadA9Twin().reference(cond) == ref


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_a9_fails_on_a_thread_cut_at_the_wrong_pitch(monkeypatch) -> None:
    """The stepped plant: the nut runs down a thread whose pitch is 2 % long, advances 1.53 mm a turn and fails the
    0.5 % law (the reference used to read 1.53 mm from the same parser)."""
    from alloy.contact import thread_twins as tt

    _plant_pitch(monkeypatch, 1.02)
    res = run_law(tt.A9_LAW, tt.ThreadA9Twin(), None, device="cpu", quick=True)
    adv = _metric(res, "advance_per_rev")
    assert adv.verdict == "fail" and adv.error == pytest.approx(3.0e-5, abs=4e-6), (res.verdicts, res.monitors)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("pull_in", [0.0, 0.0125])
def test_a11_catenary_reference_is_the_laws_span(pull_in) -> None:
    """A11.catenary (rod_vbd twin; the MuJoCo chain shares the reference code): sag against the catenary of the law's
    span S with both supports at H0. A rod hung between supports 12.5 mm closer than S sags 3 % more and fails the
    1 % law; the reference used to be recomputed at the realised span and followed it."""
    from alloy.scene import analytic_rods_vbd as RV

    class Planted(RV.CatenaryVBDTwin):
        def build(self, b, cond, sample):
            built = dataclasses.replace(cond, params={**cond.params, "span": cond.params["span"] - pull_in})
            super().build(b, built, sample)

    res = run_law(RV.LAW_CATENARY_VBD, Planted(), None, device="cpu", solver="vbd", quick=True)
    sag = _metric(res, "sag_rel_err")
    assert sag.verdict == ("fail" if pull_in else "pass"), (res.verdicts, res.monitors)
    assert sag.error == pytest.approx(0.030 if pull_in else 0.002, abs=0.005 if pull_in else 0.002)


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_a13_hydrostatic_fails_on_an_overfilled_column() -> None:
    """A13.hydrostatic: the wall force against rho0 g w H^2 / 2 with H the law's nz layers. A column filled 5 layers
    (10 %) too high pushes 21 % harder and fails the 3 % law; the reference used to take H from the column's own SPH
    volume, rose with it, and the law passed (checked here: the force is within 3 % of that old reference)."""
    from alloy.solvers.sph import laws as L

    class Overfilled(L.HydrostaticTwin):
        def build(self, b, cond, cfg):
            super().build(b, dataclasses.replace(cond, params={**cond.params, "nz": cond.params["nz"] + 5}), cfg)

    res = run_law(L.A13_HYDROSTATIC, Overfilled(L.HYDRO_DT), None, device="cpu", quick=True)
    force, mon = _metric(res, "wall_force"), res.monitors
    assert force.verdict == "fail" and force.error == pytest.approx(0.21, abs=0.03), (res.verdicts, mon)
    assert mon["height_m"][0] == pytest.approx(0.110, rel=1e-3)                 # the solver's own column
    old_ref = mon["wall_force_ref_N"][0] * (mon["height_m"][0] / 0.100) ** 2
    assert abs(mon["wall_force_N"][0] - old_ref) / old_ref < 0.03


@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_a13_archimedes_fails_on_a_box_built_too_light() -> None:
    """A13.archimedes: draft against rho_b x box height / rho0 from the law's condition. A box built at 0.9 rho_b
    floats 10 % high and fails the 5 % law; the reference used to read the built box's mass and floated with it."""
    from alloy.solvers.sph import laws as L

    class Light(L.ArchimedesTwin):
        def build(self, b, cond, cfg):
            super().build(b, dataclasses.replace(cond, params={**cond.params, "rho_b": 0.9 * cond.params["rho_b"]}),
                          cfg)

    spec = dataclasses.replace(L.A13_ARCHIMEDES, quick={"worlds": 1, "duration_s": 0.4})
    res = run_law(spec, Light(L.DRAFT_DT), None, device="cpu", quick=True)
    draft = _metric(res, "draft")
    assert draft.verdict == "fail" and draft.error > 0.08, (res.verdicts, res.monitors)
    assert res.monitors["draft_ref_m"][0] == pytest.approx(0.01)                # 800 kg/m3 x 12.5 mm / 1000
    assert res.monitors["draft_built_m"][0] == pytest.approx(0.009, rel=1e-3)   # what the old reference read
