# SPDX-License-Identifier: Apache-2.0
"""rinse-sponge verdict (alloy/tasks/sponge_rinse_score.py): gates, ledge box and score on synthetic readouts,
anchored to Gym-Anything's report numbers (start 59.3 mL at 0.75 g/L; A5 assisted rinse: 7e-5 g/L, wrung to
41.4 mL)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.rinse_score import HANDS_NOT_JUDGED
from alloy.tasks.rinse_sink import PotSink
from alloy.tasks.can_wash_score import row
from alloy.tasks.sponge_rinse_score import V_FULL, Ledge, SpongeReadout, gates, pore_conc, verdict

LEDGE = Ledge.behind(PotSink(), bowl_back_y=5.2856, wall_y=5.4)      # their frame (layout.py)
ON = (3.30, 5.33, 0.9405)                                            # the sponge's start origin


def _readout(pore_ml: float, c_g_l: float, pos=ON, **kw) -> SpongeReadout:
    base = {"faucet_gpm": 0.0, "held": False, "intact": True, "water_rel": 0.0, "species_rel": 0.0,
            "species_kg": 0.0, "finite": True}
    base |= kw
    v = pore_ml * 1e-6
    return SpongeReadout(pore_soap_kg=np.array([c_g_l * v]), pore_m3=np.array([v]), pos=np.array([pos]),
                         **{k: np.array([x]) for k, x in base.items()})


START = _readout(59.3, 0.75, held=True)


def test_ledge_is_their_deck_behind_the_three_bowls():
    assert np.allclose(LEDGE.lo, (2.6096, 5.2856, 0.935)) and np.allclose(LEDGE.hi, (3.9812, 5.4, 0.995))
    assert V_FULL * 1e6 == pytest.approx(79.1, abs=0.05)


def test_a5_rinse_passes_every_judged_gate_and_the_reset_scores_zero():
    v = verdict(START, _readout(41.4, 7e-5), 0, LEDGE)
    assert v.success and v.score == 1.0, v.failed()
    assert v.judged == (5, 6) and v.not_judged() == ["water_off_and_hands_free"]
    assert verdict(START, START, 0, LEDGE).score == 0.0          # s = 0 at the start concentration


def test_hands_free_never_passes_on_the_scripts_own_grip():
    """Audit AB-gym-3: GA's "nothing held" is a finger pad on a movable body; the privileged hand has no pads, so
    the gate is not judged (None) once the script lets go, and fails while it still grips or the faucet runs. The
    row says how many criteria were judged."""
    free = gates(_readout(41.4, 7e-5), 0, LEDGE)["water_off_and_hands_free"]
    assert free[0] is None and free[1] == {"outlet_off": True, "hands": HANDS_NOT_JUDGED, "faucet_gpm": 0.0}
    held = gates(_readout(41.4, 7e-5, held=True), 0, LEDGE)["water_off_and_hands_free"]
    assert held[0] is False and held[1]["hands"] == "still gripping"
    assert gates(_readout(41.4, 7e-5, faucet_gpm=0.5), 0, LEDGE)["water_off_and_hands_free"][0] is False
    out = row(verdict(START, _readout(41.4, 7e-5), 0, LEDGE), 3.0)
    assert out["success"] is True and out["judged"] == [5, 6] and out["not_judged"] == ["water_off_and_hands_free"]
    assert out["summary"] == "5 of 6 Gym-Anything criteria judged; not judged: water_off_and_hands_free"
    assert out["gates"]["water_off_and_hands_free"]["ok"] is None


@pytest.mark.parametrize("r, gate", [
    (_readout(41.4, 0.03), "no_suds"),
    (_readout(48.0, 0.001), "wrung_out"),                        # 0.607 of 79.1 mL
    (_readout(41.4, 0.001, pos=(3.30, 5.20, 0.9405)), "sponge_on_ledge"),   # over the rinse bowl
    (_readout(41.4, 0.001, pos=(3.30, 5.33, 1.00)), "sponge_on_ledge"),     # held above the deck
    (_readout(41.4, 0.001, held=True), "water_off_and_hands_free"),
    (_readout(41.4, 0.001, faucet_gpm=0.5), "water_off_and_hands_free"),
])
def test_each_gate_fails_alone(r, gate):
    assert [k for k, (ok, _) in gates(r, 0, LEDGE).items() if ok is False] == [gate]
    assert verdict(START, r, 0, LEDGE).failed() == [gate]


def test_empty_pores_read_no_suds():
    r = _readout(0.0005, 0.75)                                    # 0.5 µL left
    assert pore_conc(r, 0) == 0.0 and gates(r, 0, LEDGE)["no_suds"][0]


def test_progress_terms():
    """Half the log reduction, fill 0.70 (w = 0.75), still held: (s + s w)/3 with s = 0.5."""
    c = float(np.sqrt(0.75 * 0.02))
    v = verdict(START, _readout(0.70 * V_FULL * 1e6, c, held=True), 0, LEDGE)
    assert v.terms["rinsed"] == pytest.approx(0.5) and v.terms["wrung"] == pytest.approx(0.75)
    assert v.terms["returned"] == 0.0 and v.progress == pytest.approx((0.5 + 0.375) / 3.0)
    assert v.score == pytest.approx(0.9 * v.progress)
    free = verdict(START, _readout(0.70 * V_FULL * 1e6, c), 0, LEDGE)       # let go: the judged parts count
    assert free.terms["returned"] == 1.0 and free.progress == pytest.approx((0.5 + 0.375 + 0.375) / 3.0)


def test_score_reads_the_live_state_and_gates_the_settled_one():
    """GA reads s and the fill live, before the settle, and the gates after it: wrung to 41.4 mL at 7e-5 g/L live,
    it scores 1 on both terms even if the settle soaked it back to 50 mL at 0.03 g/L, and those gates then fail."""
    live, settled = _readout(41.4, 7e-5), _readout(50.0, 0.03)
    v = verdict(START, settled, 0, LEDGE, live=live)
    assert v.terms["rinsed"] == 1.0 and v.terms["wrung"] == 1.0 and v.failed() == ["no_suds", "wrung_out"]
    assert v.progress == pytest.approx(1.0) and v.score == pytest.approx(0.9)
    off = verdict(START, settled, 0, LEDGE)
    assert off.terms["rinsed"] < 1.0 and off.terms["wrung"] < 1.0
