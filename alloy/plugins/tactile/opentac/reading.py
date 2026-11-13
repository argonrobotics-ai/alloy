# SPDX-License-Identifier: Apache-2.0
"""The OpenTac readings a ``Tactile`` adds when its config names a board (``Tactile.Config.opentac``).

Readings, beside the unchanged ``taxels`` and ``pad_force`` (the ground truth):
- ``board_load`` [W, P, T]: N per taxel as the board's model has it (bump split, or foam spread on a plain skin);
- ``counts`` [W, P, T]: 12-bit ADC counts above baseline, as the board streams them (``readout.py``);
- ``bump_wrench`` [W, P, B, 5]: each bump's contact load, the truth for the decoder (bump boards only): F_n, F_x,
  F_y (N, on the pad, pad frame) and the normal load's first moments about the bump centre, sum F_n dx and
  sum F_n dy (N m). The decoder reads back F_t + 2 (first moment) / H, since an off-centre load tips the bump.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.tactile.opentac import kernels as K
from alloy.plugins.tactile.opentac import readout as R
from alloy.plugins.tactile.opentac.board import Board
from alloy.plugins.tactile.opentac.model import FOAM_SIGMA_MM, foam_half_width


def _count_args(board: Board, c: Any) -> K.CountArgs:
    p = R.profile(c.opentac_readout)
    if c.opentac_gain <= 0.0:
        raise ValueError("opentac_gain must be > 0")
    r = K.CountArgs()
    r.a, r.b, r.area_mm2 = R.gain_a(p, board.pitch), p.b, R.taxel_area_mm2(board.pitch)
    r.adc_max, r.gain = p.adc_max, float(c.opentac_gain)
    r.noise = p.noise_counts if c.opentac_read_noise else 0.0
    r.quantize, r.seed = 1, int(c.seed)
    return r


class BoardReading:
    """Device arrays, states and launches of the board reading for one ``Tactile``."""

    def __init__(self, board: Board, config: Any) -> None:
        self.board, self.config = board, config
        self.counts_args = _count_args(board, config)
        foam_half_width(board.pitch[0], config.opentac_foam_cells)  # validates the cells name

    def setup(self, plugin: Any, ctx: Any, P: int) -> None:
        b, W = self.board, ctx.world_count
        T, B = b.n_taxels, b.n_bumps
        self.W, self.P, self.T, self.B = W, P, T, B
        plugin.board_load = plugin.add_state("board_load", ctx.zeros((W, P, T)))
        plugin.counts = plugin.add_state("counts", ctx.zeros((W, P, T)))
        if B:
            plugin.bump_wrench = plugin.add_state("bump_wrench", ctx.zeros((W, P, B, 5)))
        self._load = plugin.board_load.reshape((-1,))
        self._counts = plugin.counts.reshape((-1,))
        self._bump_wrench = plugin.bump_wrench.reshape((-1,)) if B else None
        self._bacc = ctx.zeros(max(W * P * B * 5, 1), wp.int64)
        self._tacc = ctx.zeros(W * P * T, wp.int64)
        self._args = self._board_args(ctx, plugin.layout.reach)
        self._start = ctx.array(b.member_start.astype(np.int32), wp.int32)
        self._members = ctx.array(np.maximum(b.members, 0).astype(np.int32) if B else np.zeros(1, np.int32),
                                  wp.int32)
        self._weights = ctx.array(b.moment_weights.astype(np.float32), wp.vec2)

    def _board_args(self, ctx: Any, reach: float) -> K.BoardArgs:
        b, c = self.board, self.config
        a = K.BoardArgs()
        a.taxel_xy = ctx.array(b.xy.astype(np.float32), wp.vec2)
        a.taxel_bump = ctx.array(b.taxel_bump.astype(np.int32), wp.int32)
        a.bump_xy = ctx.array((b.bump_xy if b.n_bumps else np.zeros((1, 2))).astype(np.float32), wp.vec2)
        a.n, a.nb, a.reach, a.capture = b.n_taxels, max(b.n_bumps, 1), float(reach), float(c.capture_depth)
        a.mode = K.MODE_BUMPS if b.bumps else K.MODE_FOAM
        a.foam_hw = foam_half_width(b.pitch[0], c.opentac_foam_cells) if not b.bumps else 0.0
        a.foam_sigma = FOAM_SIGMA_MM * 1e-3
        return a

    def measure(self, plugin: Any, state_in: Any) -> None:
        ctx = plugin.ctx
        con, force, model, dev = ctx.contacts, ctx.contact_force, ctx.model, ctx.device
        wp.launch(K.bin_board, dim=con.rigid_contact_max, device=dev,
                  inputs=[con.rigid_contact_count, con.rigid_contact_shape0, con.rigid_contact_shape1,
                          con.rigid_contact_point0, con.rigid_contact_point1, force, model.shape_body,
                          state_in.body_q, plugin._pad_of_body, self.P, plugin._origin, plugin._rot, self._args],
                  outputs=[self._bacc, self._tacc])
        if self.B:
            wp.launch(K.split_bumps, dim=self.W * self.P * self.B, device=dev,
                      inputs=[self._bacc, self._start, self._members, self._weights, self.B, self.T,
                              float(self.board.bump_height), 1.0 if self.config.opentac_tip else 0.0],
                      outputs=[self._load, self._bump_wrench])
        wp.launch(K.finish_board, dim=self.W * self.P * self.T, device=dev,
                  inputs=[self._tacc, self._args.taxel_bump, self.T, self.P * self.T, plugin.tick, self.counts_args],
                  outputs=[self._load, self._counts])
