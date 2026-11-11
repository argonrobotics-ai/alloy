# SPDX-License-Identifier: Apache-2.0
"""The forms proxy (``alloy.tasks.pad_forms``; WP15's ruling 2026-11-09 ~05:40Z): the forms are cells, not bodies, so
``forms_undisturbed`` is each tool's lowest sole clearance over the form tops along its whole commanded path."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks import pad_forms as PF
from alloy.tasks import pad_trowel_lifts as LF
from alloy.tasks.concrete_pad_start import FORM_H
from alloy.tasks.concrete_tools import HEIGHT, Key, Stroke

HALF = (0.02, 0.01, 0.001)       # m: a small blade, half length (along yaw), width, thickness


def _f() -> dict:
    """A 0.2 m square of 2 mm cells, the forms the ring outside |x|, |y| < 0.08 m."""
    x = y = np.arange(-0.1, 0.1 + 1e-9, 0.002)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return {"x": x, "y": y, "cell": 0.002, "form_mask": (np.abs(X) > 0.08) | (np.abs(Y) > 0.08)}


def _table(*programs: list[Stroke]) -> tuple[np.ndarray, np.ndarray]:
    K = max(len(p) for p in programs)
    table = np.zeros((len(programs), K), dtype=Key.numpy_dtype())
    for r, p in enumerate(programs):
        for j in range(K):
            s = p[min(j, len(p) - 1)]
            table[r, j] = (s.t, s.pos, s.yaw, s.pitch, s.force, s.mode, s.release)
    return table, np.array([len(p) for p in programs])


def _flight(z0: float, z1: float, x1: float = 0.095, yaw: float = 0.0, pitch: float = 0.0) -> list[Stroke]:
    return [Stroke(0.0, (0.0, 0.0, z0), yaw=yaw, pitch=pitch, mode=HEIGHT),
            Stroke(1.0, (x1, 0.0, z1), yaw=yaw, pitch=pitch, mode=HEIGHT)]


def test_a_sole_over_a_form_top_reads_its_clearance_and_a_dip_below_fails() -> None:
    f = _f()
    up = PF.clearance(*_table(_flight(FORM_H + 0.01, FORM_H + 0.003)), 1, [HALF], f)[0]
    assert up.passed and up.min_m == pytest.approx(0.003 - HALF[2], abs=1e-9) and up.tool == 0
    down = PF.clearance(*_table(_flight(FORM_H + 0.01, FORM_H - 0.0005)), 1, [HALF], f)[0]
    assert not down.passed and down.min_m == pytest.approx(-0.0005 - HALF[2], abs=1e-9)
    assert down.at[0] > 0.08 - HALF[0]                       # where the footprint first reaches the forms


def test_the_whole_swept_path_counts_not_only_the_keys() -> None:
    """A straight flight from a low key inside to a high key past the forms dips over the forms between keys; a
    pitched sole's low corner counts; a path that never reaches a form has no clearance (inf, passes)."""
    f = _f()
    t, n = _table(_flight(FORM_H - 0.002, FORM_H + 0.02))
    c = PF.clearance(t, n, 1, [HALF], f)[0]
    keys_only = min(FORM_H + 0.02 - HALF[2] - FORM_H, math.inf)
    assert c.min_m < keys_only and 0.0 < c.t < 1.0
    flat = PF.clearance(*_table(_flight(FORM_H + 0.005, FORM_H + 0.005)), 1, [HALF], f)[0]
    tilted = PF.clearance(*_table(_flight(FORM_H + 0.005, FORM_H + 0.005, pitch=0.1)), 1, [HALF], f)[0]
    assert tilted.min_m == pytest.approx(flat.min_m - HALF[0] * math.sin(0.1) - HALF[2] * (math.cos(0.1) - 1),
                                         abs=1e-9)
    inside = PF.clearance(*_table(_flight(FORM_H - 0.01, FORM_H - 0.01, x1=0.03)), 1, [HALF], f)[0]
    assert inside.passed and math.isinf(inside.min_m) and inside.tool == -1


def test_a_yawed_blade_reaches_the_forms_by_its_rotated_footprint() -> None:
    """Parked 0.068 m from the centre, at the form tops: along x its 0.02 m half length reaches the forms (0.08 m),
    turned 90 degrees its 0.01 m half width does not."""
    f = _f()
    park = lambda yaw: [Stroke(0.0, (0.068, 0.0, FORM_H), yaw=yaw, mode=HEIGHT),     # noqa: E731
                        Stroke(1.0, (0.068, 0.0, FORM_H), yaw=yaw, mode=HEIGHT)]
    assert not PF.clearance(*_table(park(0.0)), 1, [HALF], f)[0].passed
    assert PF.clearance(*_table(park(math.pi / 2)), 1, [HALF], f)[0].passed


def test_the_verdict_takes_the_proxy_and_arms_clear_is_not_applicable() -> None:
    crit = {"smooth": {"value": 0.9, "threshold": 0.8, "passed": True, "measured": True},
            "forms_undisturbed": {"value": 0.0, "threshold": 50.0, "passed": True, "measured": False},
            "arms_clear": {"value": 0, "threshold": 0, "passed": True, "measured": False}}
    ok = LF._proxied({"success": True, "score": 1.0, "criteria": {k: dict(v) for k, v in crit.items()}},
                     PF.Clearance(0.004, 0, 1.0, (0.0, 0.0)))
    assert ok["success"] and ok["criteria"]["forms_undisturbed"]["measured"] == "proxy"
    assert ok["criteria"]["arms_clear"]["passed"] is None and "not applicable" in ok["criteria"]["arms_clear"]["note"]
    assert ok["judged"] == [2, 3] and ok["not_judged"] == ["arms_clear"]              # the count with the proxy in
    assert ok["summary"].startswith("2 of 3 Gym-Anything criteria judged; not judged: arms_clear; judged by another "
                                    "measure than Gym-Anything's: forms_undisturbed")
    touch = LF._proxied({"success": True, "score": 1.0, "criteria": {k: dict(v) for k, v in crit.items()}},
                        PF.Clearance(-0.0002, 1, 2.0, (0.0, 0.0)))
    assert not touch["success"] and touch["score"] is None and touch["score_before_proxy"] == 1.0
