# SPDX-License-Identifier: Apache-2.0
"""Explicit-stability numbers of the deposition tips, per world and tool, read once at setup.

Each tip is removal v1's compliant pad (``alloy.plugins.removal.stability``: the combined normal + 2·tilt bound,
WP2 review N1). With several tools on one floating board, their springs, dampers and viscous friction add on the
board's modes (WP2 review F2), so every tool's bound carries the board terms of all tools:

    ω_t² = k_t·w_tool,t + Σ_u k_u·w_board + 2·tilt_t/I_tool,t + Σ_u 2·tilt_u/I_board        (tilt_u = k_u·R_u²/4)

and the same with ``τ·k`` for the dampers. The friction mobility of tool t is ``m_tool,t + T·w_board``. Because each
of the T tools keeps ``μF/eps·dt·mobility <= 1/2``, their sum on the board does too. For a static board the board
terms vanish, and for one tool this is removal v1's bound. ``w`` are point mobilities (``1/m + r²/I``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.removal.stability import body_of, in_plane_inertia, patch_reach_sq, point_mobility, smallest_inertia
from alloy.plugins.removal.surface import Patch


@dataclass(frozen=True)
class TipResponse:
    """Per (world, tool) body response, ``[W, T]`` each, and per world the board's, ``[W]``."""

    w_tool: np.ndarray          # tool normal mobility at the tip: 1/m + d_perp²/I_plane
    inv_tilt_tool: np.ndarray   # 1 / the tool's smallest inertia about an axis in the tip plane
    friction_tool: np.ndarray   # tool point mobility at the farthest tip point
    w_board: np.ndarray         # board point mobility anywhere on the patch (0 if static)
    inv_tilt_board: np.ndarray  # 1 / the board's smallest principal inertia (0 if static)

    def friction_mobility(self) -> np.ndarray:
        """``[W, T]`` float32 for the force kernel: m_tool,t + T·w_board."""
        tools = self.w_tool.shape[1]
        return (self.friction_tool + tools * self.w_board[:, None]).astype(np.float32)


def tip_response(model: newton.Model, bodies: np.ndarray, patches: list[Patch], specs: list[Any]) -> TipResponse:
    W, T = bodies.shape
    w_tool, inv_tilt, friction = (np.zeros((W, T)) for _ in range(3))
    w_board, inv_board = np.zeros(W), np.zeros(W)
    for w in range(W):
        p = patches[w]
        if p.body >= 0:
            board = body_of(model, p.body, "board")
            w_board[w] = point_mobility(board, patch_reach_sq(board, p))
            inv_board[w] = 1.0 / smallest_inertia(board.inertia)
        for t, s in enumerate(specs):
            tool = body_of(model, int(bodies[w, t]), s.label)
            n = np.asarray(s.normal, np.float64) / np.linalg.norm(s.normal)
            arm = np.asarray(s.center, np.float64) - tool.com
            offset = arm - float(np.dot(arm, n)) * n
            plane = in_plane_inertia(tool.inertia, n)
            w_tool[w, t] = 1.0 / tool.mass + float(np.dot(offset, offset)) / plane
            inv_tilt[w, t] = 1.0 / plane
            friction[w, t] = point_mobility(tool, (float(np.linalg.norm(arm)) + s.radius) ** 2)
    return TipResponse(w_tool, inv_tilt, friction, w_board, inv_board)


def stability_numbers(r: TipResponse, specs: list[Any], dt: float) -> dict[str, tuple[float, float]]:
    """Per tool, the worst world's (ω·dt, damping number) under the bound in the module docstring."""
    k = np.asarray([s.contact_stiffness for s in specs])
    tau = np.asarray([s.damping for s in specs])
    tilt = k * 0.25 * np.asarray([s.radius for s in specs]) ** 2
    board_spring = r.w_board * k.sum() + 2.0 * r.inv_tilt_board * tilt.sum()                  # [W]
    board_damp = r.w_board * (tau * k).sum() + 2.0 * r.inv_tilt_board * (tau * tilt).sum()
    out = {}
    for t, s in enumerate(specs):
        own = k[t] * r.w_tool[:, t] + 2.0 * tilt[t] * r.inv_tilt_tool[:, t]
        omega_sq = float((own + board_spring).max())
        damp = float((tau[t] * own + board_damp).max())
        out[s.label] = (float(np.sqrt(omega_sq)) * dt, damp * dt)
    return out


__all__ = ["TipResponse", "stability_numbers", "tip_response"]
