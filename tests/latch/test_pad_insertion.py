# SPDX-License-Identifier: Apache-2.0
"""Preloaded slide pads opened by a blade (DC3's C13 contact pair; ``alloy.testing.latch.PadRig``): the insertion peak
follows the wedge law on the 12 deg lead-in and the carrier seats, with the pads regularised by armature and held by a
stiff closed stop (``stop_stiffness``) and ``snap_step``. Touched faces use the D22 precision contact (the rig's
``cfg_for`` + flat solimp 0.99); on a card route that is the card's ``precision_features``."""

from __future__ import annotations

import dataclasses

import pytest

from alloy.testing.latch import PadRig, RigRun, build_pad_rig, insert


def _insert(p: PadRig, solver_kw: dict) -> dict[str, float]:
    rig = build_pad_rig(p)
    run = RigRun(rig, p.design_dt, solver_kw)
    tr = insert(run, speed=0.010, overtravel=0.003)
    return {"peak": float(tr.peak_before(p.start_gap + p.lead_len + 1.0e-3)[0]), "x_end": float(tr.x[-1, 0]),
            "travel_in": rig.travel_in, "pad_q": float(run.read()["tab"][0]), "finite": tr.finite}


@pytest.mark.parametrize("stop_stiffness", [1.0e5, None])
def test_pads_open_at_the_wedge_force_and_seat(stop_stiffness):
    """2 pads x 7.04 N x (tan 12 deg + 0.5) / (1 - 0.5 tan 12 deg) = 11.3 N; measured 11.3 N (MuJoCo C)."""
    p = dataclasses.replace(PadRig(), stop_stiffness=stop_stiffness)
    r = _insert(p, {"use_mujoco_cpu": True})
    assert r["finite"]
    assert r["peak"] == pytest.approx(p.peak_estimate(), rel=0.15)
    assert r["x_end"] >= r["travel_in"]
    assert r["pad_q"] == pytest.approx(p.opening, abs=2.0e-5)


@pytest.mark.compile_heavy
@pytest.mark.parametrize("impratio", [100.0, 300.0])
def test_pads_on_mujoco_warp(impratio):
    """The same on MuJoCo Warp (float32), at DC3's route impratio 100 and at 300."""
    p = PadRig()
    r = _insert(p, {"impratio": impratio})
    assert r["finite"] and r["x_end"] >= r["travel_in"]
    assert r["peak"] == pytest.approx(p.peak_estimate(), rel=0.15)
