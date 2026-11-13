# SPDX-License-Identifier: Apache-2.0
"""Laws of the OpenTac board reading (ours: OpenTac states none), on the tactile law pad (``tactile/laws.py``).

The board is ``opentac_square_9x9_60pin`` centred on the 6 x 6 cm pad; noise is off. The laws:
- **opentac_force_sum:** the board's taxel loads add up to the engine's normal force on the pad, within 1e-4 (a box
  at rest, its contacts inside the board). The bump split conserves each bump's load by construction.
- **opentac_force_sum_plain:** the same on a plain skin (no bump sheet), foam squares one pitch wide, within 1e-3.
  With OpenTac's own squares the sum reads ``model.foam_box_factor`` (1.15 at 4.2 mm) times the force instead.
- **opentac_round_trip:** a 3 x 3 x 1 mm puck on one bump, loaded at 0.3 g tangential: the contact shear, through
  the split, v3 counts (whole counts) and the decoder, comes back within 2 %.
- **opentac_round_trip_tip:** the same at 0.1 g with ``opentac_tip``: the decoder returns what the split put in,
  F_t + 2 (first moment) / H, within 2 %. The puck creeps (MuJoCo's soft friction: ~0.6 mm in 0.5 s here, ~1.9 mm at
  0.3 g), and the first moment of a 1 N load 0.6 mm off the bump centre decodes as ~0.6 N of shear, 5x the contact
  shear: with the tip term, OpenTac's decoder reads off-centre load as shear.
- **opentac_counts:** a heavy box with the v2 profile (0.31 N per taxel saturation) and the tip term (so loads
  spread from zero to past saturation): counts match the reference readout of the loads within 1 count, never fall
  as the load rises, and clip at the ADC limit.
- **opentac_null:** nothing on the pad: every count is 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.tactile import laws as TL
from alloy.plugins.tactile.opentac import model as M
from alloy.plugins.tactile.opentac import readout as R

BOARD = "opentac_square_9x9_60pin"
BUMP_AT = (0.0021, 0.0021)       # a full 2x2 block's bump centre on this board (m, pad frame)
PUCK = 0.0015                    # puck half extent (m)
PUCK_H = 0.0005                  # puck half height (m): low, so it barely tips
PUCK_MASS = 0.1                  # kg: 0.98 N on one bump
HEAVY_MASS = 0.3                 # kg: ~0.7 N per corner, past v2's 0.31 N per taxel
CONFIG = {**TL.NOISE_OFF, "opentac": BOARD, "opentac_bumps": True, "opentac_readout": "v3", "opentac_gain": 1.0,
          "opentac_read_noise": False, "opentac_foam_cells": "opentac", "opentac_tip": False, "reach": 0.0,
          "capture_depth": 0.003}


def _build_puck(b: newton.ModelBuilder) -> None:
    TL._pad(b)
    puck = b.add_body(xform=body_xform(BUMP_AT[0], BUMP_AT[1], 2 * TL.PAD_H + PUCK_H + 5e-4), label="box")
    b.add_shape_box(puck, hx=PUCK, hy=PUCK, hz=PUCK_H, cfg=TL._cfg(PUCK_MASS / (8 * PUCK * PUCK * PUCK_H)))


def _build_heavy(b: newton.ModelBuilder) -> None:
    TL._pad(b)
    box = b.add_body(xform=body_xform(0.0, 0.0, 2 * TL.PAD_H + TL.BOX + 5e-4), label="box")
    b.add_shape_box(box, hx=TL.BOX, hy=TL.BOX, hz=TL.BOX, cfg=TL._cfg(HEAVY_MASS / (2 * TL.BOX) ** 3))


def _last(run: Any, name: str) -> np.ndarray:
    return np.asarray(run.trace.plugin(run.plugin.label, name))[-1, 0, 0]


def _board(run: Any):
    return run.plugin._board.board


def _check_force_sum(bound: float):
    def check(run: Any) -> LawResult:
        load, pad = _last(run, "board_load"), _last(run, "pad_force")
        rel = (load.sum() - pad[0]) / max(abs(pad[0]), 1e-12)
        return LawResult.within(rel, bound, f"board taxels {load.sum():.6g} N vs engine {pad[0]:.6g} N",
                                "a captured contact missed the board's taxels: check the pad pose and the reach")
    return check


def _check_round_trip(run: Any) -> LawResult:
    board, w, counts = _board(run), _last(run, "bump_wrench"), _last(run, "counts")
    tip = 2.0 / board.bump_height if run.plugin.config.opentac_tip else 0.0
    put_in = w[:, 1:3] + tip * w[:, 3:5]
    got = M.decode(board, counts, run.plugin.config.opentac_readout)["bump_shear"]
    applied = w[:, 1:3].sum(0)
    err = float(np.linalg.norm((got - put_in).sum(0)) / max(np.linalg.norm(put_in.sum(0)), 1e-12))
    return LawResult.within(err, 0.02, f"decoded shear {got.sum(0)} N vs put in {put_in.sum(0)} N (contact shear "
                            f"{applied} N)", "the split, the readout or the decoder disagree: compare with model.py")


def _check_counts(run: Any) -> LawResult:
    board, load, counts = _board(run), _last(run, "board_load"), _last(run, "counts")
    p = R.profile(run.plugin.config.opentac_readout)
    ref = R.counts(load, board.pitch, p)
    order = np.argsort(load, kind="stable")
    fall = float(np.clip(-np.diff(counts[order]), 0.0, None).max(initial=0.0))
    off = float(np.abs(counts - ref).max())
    clipped, between = int((counts >= p.adc_max).sum()), int(((counts > 0) & (counts < p.adc_max)).sum())
    bad = max(off - 1.0, fall, 0.0) + (0.0 if clipped and between else 1.0)
    return LawResult.within(bad, 0.0, f"max |counts - reference| {off:.2f}, largest fall {fall:.2f}, {clipped} taxels "
                            f"at the ADC limit {p.adc_max:.0f}, {between} between", "the readout kernel is off: "
                            "compare finish_board with readout.counts")


def _check_null(run: Any) -> LawResult:
    v = float(np.abs(_last(run, "counts")).max())
    return LawResult.within(v, 0.0, f"max count {v:.3g} with nothing on the pad", "contacts outside the capture "
                            "depth must not count")


def opentac_laws() -> list[Law]:
    """The laws above, each on its own scene, with noise off."""
    rest = TestScene(build=TL._build_rest, periods=25, ground_mu=1.0)
    puck = TestScene(build=_build_puck, periods=25, ground_mu=1.0, gravity=TL._tilted(0.3, 30.0))
    puck_tip = TestScene(build=_build_puck, periods=25, ground_mu=1.0, gravity=TL._tilted(0.1, 30.0))
    heavy = TestScene(build=_build_heavy, periods=25, ground_mu=1.0)
    null = TestScene(build=TL._build_null, periods=10, ground_mu=1.0)
    plain = {**CONFIG, "opentac_bumps": False, "opentac_foam_cells": "pitch"}
    return [
        Law("opentac_force_sum", _check_force_sum(1e-4), scene=rest, config=CONFIG,
            doc="board taxel loads add up to the engine's pad force"),
        Law("opentac_force_sum_plain", _check_force_sum(1e-3), scene=rest, config=plain,
            doc="plain skin, pitch-wide foam squares: loads add up to the pad force"),
        Law("opentac_round_trip", _check_round_trip, scene=puck, config=CONFIG,
            doc="shear to counts to decoded shear, within 2 %"),
        Law("opentac_round_trip_tip", _check_round_trip, scene=puck_tip, config={**CONFIG, "opentac_tip": True},
            doc="with the tip term: decoded = shear + 2 (first moment) / H, within 2 %"),
        Law("opentac_counts", _check_counts, scene=heavy,
            config={**CONFIG, "opentac_readout": "v2", "opentac_tip": True},
            doc="counts match the readout, rise with load, clip at the ADC limit"),
        Law("opentac_null", _check_null, scene=null, config=CONFIG, doc="no contact, zero counts"),
    ]
