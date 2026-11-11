# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the E21 and E22 reference environments: a Panda scene file with a firm grip, and a start-pose
solver with steps small enough to converge on hand-down poses."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import panda_kin as kin
from panda_env import panda_file


def strong_grip_file(kp: float = 2000.0) -> Path:
    """The Panda scene file with the finger servo stiffened from kp 100 to ``kp`` N/m: with 100 the squeeze is about
    1 N (kp times the 1 cm the fingers are still apart on a 16 mm bar), too little to hold a drawer's pull or a
    peg's insertion force."""
    src = panda_file()
    out = src.with_name(f"{src.stem}-grip{int(kp)}.xml")
    if out.is_file():
        return out
    text = src.read_text(encoding="utf-8")
    weak = 'gainprm="100.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" biasprm="0.0 -100.0 -10.0'
    if weak not in text:
        raise ValueError(f"finger actuator line not found in {src}")
    strong = f'gainprm="{kp} 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0" biasprm="0.0 -{kp} -{kp / 10}'
    out.write_text(text.replace(weak, strong), encoding="utf-8")
    return out


def solve_start(goal: np.ndarray, facing: np.ndarray, iters: int = 800, tol: float = 5e-3) -> np.ndarray:
    """Joint angles (7,) putting the grasp point at ``goal`` (3,) with the hand frame ``facing`` (3, 3). The loop of
    ``kin.solve_ik`` with smaller steps (its coarse ones stop 2 mm short for some poses); raises beyond ``tol``."""
    q = kin.Q_HOME[None].copy()
    for _ in range(iters):
        q = np.clip(q + kin.ik_step(q, goal[None], facing, max_lin=0.03, max_dq=0.1), kin.Q_LOW, kin.Q_HIGH)
    f = kin.fk(q)
    miss = np.linalg.norm(f.grasp[0] - goal) + np.abs(kin.rot_error(facing, f.hand_rot)).max()
    if miss > tol:
        raise ValueError(f"no start pose for {goal}: residual {miss:.4f}")
    return q[0]
