# SPDX-License-Identifier: Apache-2.0
"""GoalNet laws: the pretensioned net's fundamental mode against its closed form, and a 25 m/s catch.

| Law | Kind | Scene | Reference | Bound |
|---|---|---|---|---|
| net_mode | closed form | 2.88 x 2.04 m net, 1 % pretension, no gravity or losses, (1,1) at 2 mm | ω² below | 1 % |
| net_catch | regression | a 25 m/s ball into the default net (gravity on) | depth, rebound | 1.2 m, 0.45 v0 |

net_mode: ω² = 4T/(m·s)·[sin²(π/2N_u) + sin²(π/2N_v)], with T = EA·strain0 the cord tension, m the node mass, s
the mesh and N the cells across and up. The continuum membrane gives f = ½·sqrt(T/2μ)·sqrt(1/a² + 1/b²); the row
reports both. net_catch has no measured reference: it guards against pass-through and a lost knot friction.
"""

from __future__ import annotations

import math

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.sports import model as M
from alloy.plugins.sports.scene import add_ball, launch

PLANE_X = 1.0
MODE = {"origin": (PLANE_X, -1.44, 0.0), "width": 2.88, "height": 2.04, "strain0": 0.01, "zeta": 0.0,
        "knot_friction": 0.0, "ea": 600.0, "ball": "ball"}
CATCH = {"origin": (PLANE_X, -1.44, 0.0), "width": 2.88, "height": 2.04}
AMP = 2.0e-3
CATCH_SPEED = 25.0
REBOUND_MAX = 0.45      # regression guard on the knot friction (now 0.32 v0); the rebound itself is an ESTIMATE


def _ball_build(x: float, z: float):
    def build(b: newton.ModelBuilder) -> None:
        add_ball(b, (x, 0.0, z))
    return build


def net_scene() -> TestScene:
    """Three identical worlds: a 10 m/s ball into the default net, slightly off centre."""
    starts = [((10.0, 0.3, 1.0), (0.0, 0.0, 0.0))]
    return TestScene(build=_ball_build(0.0, 1.02), worlds=3, periods=20, dt=5.0e-4, ground=False, init=launch(starts))


def _mode_init(sim) -> None:
    net = sim.plugins["goal_net"]
    x = net.nodes.numpy()
    rows, cols = net.net.rows, net.net.cols
    i, j = np.divmod(np.arange(rows * cols), cols)
    shape = np.sin(math.pi * i / (rows - 1)) * np.sin(math.pi * j / (cols - 1))
    x[:, :, 0] += AMP * shape[None, :]
    net.nodes.assign(x)


def mode_frequency(cfg: dict) -> tuple[float, float]:
    """(discrete, continuum) (1,1) frequency [Hz] of the pretensioned net in ``cfg``."""
    s = 0.12
    nu, nv = round(cfg["width"] / s), round(cfg["height"] / s)
    t = cfg["ea"] * cfg["strain0"]                              # cord tension EA·strain0
    m = 6.4e-3 * 2.0 * s
    w2 = 4.0 * t / (m * s) * (math.sin(math.pi / (2 * nu)) ** 2 + math.sin(math.pi / (2 * nv)) ** 2)
    cont = 0.5 * math.sqrt(t / (2.0 * 6.4e-3)) * math.sqrt(1.0 / cfg["width"] ** 2 + 1.0 / cfg["height"] ** 2)
    return math.sqrt(w2) / (2.0 * math.pi), cont


def check_mode(run) -> LawResult:
    nodes = run.trace.plugin("goal_net", "nodes")[:, 0]          # [K, N, 3]
    rows, cols = round(MODE["height"] / 0.12) + 1, round(MODE["width"] / 0.12) + 1
    centre = (rows // 2) * cols + cols // 2
    w = nodes[:, centre, 0].astype(np.float64) - PLANE_X
    t = run.trace.t.astype(np.float64)
    up = np.flatnonzero((w[:-1] < 0.0) & (w[1:] >= 0.0))
    if len(up) < 3:
        return LawResult(False, float("nan"), 0.01, "fewer than 3 oscillations", "check the cord forces")
    tc = t[up] - w[up] * (t[up + 1] - t[up]) / (w[up + 1] - w[up])
    got = (len(tc) - 1) / (tc[-1] - tc[0])
    want, cont = mode_frequency(MODE)
    return LawResult.within(got / want - 1.0, 0.01, f"(1,1) mode {got:.3f} Hz vs {want:.3f} Hz (discrete net; "
                            f"continuum membrane {cont:.3f} Hz)", "check net_kernels.cord and the node mass")


def check_catch(run) -> LawResult:
    b = run.bodies("ball")[0]
    x = run.trace.body_q[:, b, 0].astype(np.float64)
    vx = run.trace.body_qd[:, b, 0].astype(np.float64)
    depth = float(x.max() + M.R_BALL - PLANE_X)
    back = float(-vx.min())
    ok = 0.0 < depth <= 1.2 and 0.0 < back <= REBOUND_MAX * CATCH_SPEED and x[-1] < PLANE_X + 1.2
    return LawResult(bool(ok), depth, 1.2, f"25 m/s ball: net bags {depth:.2f} m, ball comes back at {back:.1f} m/s "
                     f"({100 * (back / CATCH_SPEED) ** 2:.0f} % of its energy; ESTIMATE, no measured net found)",
                     "pass-through: a smaller cell or larger cord_radius; a hard rebound: check the knot slider "
                     "(knot_friction, knot_stiffness); slack, EA and cord damping barely move it")


def net_laws() -> list[Law]:
    mode = TestScene(build=_ball_build(-5.0, 1.0), worlds=1, periods=200, control_dt=0.005, dt=5.0e-4, ground=False,
                     gravity=(0.0, 0.0, 0.0), init=_mode_init)
    catch = TestScene(build=_ball_build(0.0, 1.02), worlds=1, periods=20, dt=5.0e-4, ground=False,
                      init=launch([((CATCH_SPEED, 0.0, 0.0), (0.0, 0.0, 0.0))]))
    return [
        Law("net_mode", check_mode, mode, config=MODE, doc="a pretensioned net rings at its discrete (1,1) mode"),
        Law("net_catch", check_catch, catch, config=CATCH, doc="the slack net stops a 25 m/s ball within 1.2 m"),
    ]


__all__ = ["CATCH", "MODE", "PLANE_X", "mode_frequency", "net_laws", "net_scene"]
