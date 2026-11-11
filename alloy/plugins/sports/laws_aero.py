# SPDX-License-Identifier: Apache-2.0
"""BallAero laws: closed forms of drag and Magnus flight, and a free kick against the float64 reference.

| Law | Scene (no gravity unless said) | Closed form / reference | Bound |
|---|---|---|---|
| drag_deceleration | 25 m/s, no spin, supercritical (C_d 0.17) for 1 s | x = ln(1 + κv0t)/κ, κ = ρAC_d/2m | 1e-3 |
| magnus_circle | 20 m/s, S = 0.25 spin about z, no drag, no spin decay | circle of radius m/(½ρAC_L(S)) | 0.5 % |
| free_kick | 25 m/s at 12°, 8 rev/s sidespin, gravity, full model | ``model.flight`` (RK4, float64) | 0.2 % |
"""

from __future__ import annotations

import math

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene
from alloy.plugins.sports import model as M
from alloy.plugins.sports.scene import add_ball, launch

DT = 5.0e-4
NO_GRAVITY = (0.0, 0.0, 0.0)
V_DRAG, V_CURL = 25.0, 20.0
S_CURL = 0.25
KICK_V = (25.0 * math.cos(math.radians(12.0)), 0.0, 25.0 * math.sin(math.radians(12.0)))
KICK_W = (0.0, 0.0, 8.0 * 2.0 * math.pi)


def _ball_at(z: float):
    def build(b: newton.ModelBuilder) -> None:
        add_ball(b, (0.0, 0.0, z))
    return build


def aero_scene() -> TestScene:
    """Three identical worlds of a spinning ball in flight (gravity on)."""
    starts = [((11.0, 1.0, 3.0), (0.0, 10.0, 20.0))]
    return TestScene(build=_ball_at(1.0), worlds=3, periods=25, dt=DT, ground=False, init=launch(starts))


def _free(periods: int, start: tuple, gravity=NO_GRAVITY) -> TestScene:
    return TestScene(build=_ball_at(0.0 if gravity == NO_GRAVITY else M.R_BALL), worlds=1, periods=periods, dt=DT,
                     ground=False, gravity=gravity, init=launch([start]))


def check_drag(run) -> LawResult:
    b = run.bodies("ball")[0]
    t, x = run.trace.t.astype(np.float64), run.trace.body_q[:, b, 0].astype(np.float64)
    kappa = 0.5 * M.RHO_AIR * M.area() * M.CD_SUPER / M.M_BALL
    want = np.log1p(kappa * V_DRAG * t) / kappa
    err = (x[-1] - x[0]) / want[-1] - 1.0
    v_end = V_DRAG / (1.0 + kappa * V_DRAG * t[-1])
    return LawResult.within(err, 1e-3, f"after {t[-1]:.2f} s: {x[-1] - x[0]:.4f} m vs {want[-1]:.4f} m; speed "
                            f"{v_end:.2f} m/s stays supercritical (C_d {M.CD_SUPER})",
                            "check the drag term in kernels.ball_aero")


def circle_radius(xy: np.ndarray) -> float:
    """Algebraic (Kåsa) circle fit."""
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(a, x * x + y * y, rcond=None)
    return float(math.sqrt(c[2] + 0.25 * (c[0] ** 2 + c[1] ** 2)))


def check_curl(run) -> LawResult:
    b = run.bodies("ball")[0]
    xy = run.trace.body_q[:, b, 0:2].astype(np.float64)
    want = M.M_BALL / (0.5 * M.RHO_AIR * M.area() * float(M.cl(S_CURL)))
    got = circle_radius(xy)
    return LawResult.within(got / want - 1.0, 5e-3, f"radius {got:.2f} m vs m/(½ρA·C_L) = {want:.2f} m at S "
                            f"{S_CURL} (C_L {float(M.cl(S_CURL)):.3f})", "check the Magnus term in kernels.ball_aero")


def check_free_kick(run) -> LawResult:
    b = run.bodies("ball")[0]
    t = run.trace.t.astype(np.float64)
    x = run.trace.body_q[:, b, 0:3].astype(np.float64)
    tr, xr, _ = M.flight(np.array([0.0, 0.0, M.R_BALL]), np.array(KICK_V), np.array(KICK_W), float(t[-1]))
    ref = np.stack([np.interp(t, tr, xr[:, i]) for i in range(3)], axis=1)
    err = float(np.max(np.linalg.norm(x - ref, axis=1)))
    span = float(np.linalg.norm(ref[-1] - ref[0]))
    return LawResult.within(err / span, 2e-3, f"max {err * 1e3:.1f} mm off the float64 flight over {span:.1f} m; "
                            f"curl {ref[-1, 1]:.2f} m, height {ref[-1, 2]:.2f} m at {t[-1]:.2f} s",
                            "the kernels must match model.aero")


def aero_laws() -> list[Law]:
    w_curl = S_CURL * V_CURL / M.R_BALL
    no_drag = {"cd_sub": 0.0, "cd_super": 0.0, "cm": 0.0}
    return [
        Law("drag_deceleration", check_drag, _free(50, ((V_DRAG, 0.0, 0.0), (0.0, 0.0, 0.0))),
            doc="a supercritical ball decelerates as x = ln(1 + κv0t)/κ"),
        Law("magnus_circle", check_curl, _free(60, ((V_CURL, 0.0, 0.0), (0.0, 0.0, w_curl))), config=no_drag,
            doc="with drag off, a spinning ball circles at radius m/(½ρA·C_L(S))"),
        Law("free_kick", check_free_kick, _free(50, (KICK_V, KICK_W), gravity=(0.0, 0.0, -M.G)),
            doc="a curling free kick follows the float64 flight of model.py"),
    ]


__all__ = ["aero_laws", "aero_scene", "circle_radius"]
