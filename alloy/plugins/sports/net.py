# SPDX-License-Identifier: Apache-2.0
"""``GoalNet``: a cable net that catches a ball (Gym-Anything's football net; their sim is a custom cable graph too).

The net is a grid of nodes joined by cords in a rectangle, pinned on chosen edges (the frame and the ground pegs).
- **Cords** carry tension only: EA·strain, a dashpot, and knot friction. The knot friction is a Coulomb slider with
  a pre-slide spring, so the tension gains up to +t_f while a cord lengthens and loses up to t_f while it shortens,
  at any stretch rate.
- **Nodes** stay above the ground, with friction.
- **Stepping:** the plug-in owns the nodes. Its ``update`` advances them in sub-steps short enough for the cords
  (counted from each exchange step). It pushes them out of the ball, which moves straight through each exchange
  between its start and end poses.
- **The ball** gets the nodes' push-back as an equal and opposite impulse in the next exchange (a one-exchange lag,
  0.31 ms in the laws and the task, against a catch of ~50 ms).

Defaults are a 3 mm polypropylene net. Every stiffness and loss is an ESTIMATE (no measured net data found):
- a 120 mm square mesh (EN 748 nets; manufacturer page) and 6.4 g/m of cord (ρ·π·(1.5 mm)², ρ = 910 kg/m³);
- EA 600 N for a knotted cord (Gym-Anything's net uses 2,500 N/m on 0.1 m segments, EA 250 N);
- hung 8 % slack, with 40 N of knot friction per cord (pre-slide stiffness 10 x EA/rest) and cord damping ratio 0.5.

A 25 m/s ball bags a 2.88 x 2.04 m panel 0.69 m deep and comes back at 7.9 m/s (10 % of the energy); a 23 m/s
ball into the task's 7.56 x 2.56 m net comes back with 8 %. The knot friction sets that (sweeps in README.md): with
none, 36-42 % comes back.

Readings (state): ``nodes`` [W, N] positions, row-major from the bottom-left corner, for rendering and checks.

    net = GoalNet(origin=(2.12, -3.84, 0.0), width=7.68, height=2.56)    # the plane x = 2.12, facing -x
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.errors import PluginError
from alloy.plugins.sports import model as M
from alloy.plugins.sports import net_kernels as NK

Vec = tuple[float, float, float]


class GoalNet(ForcePlugin):
    """Cable net (tension-only cords with knot friction on a grid, pinned edges) that a ball pushes into."""

    name = "goal_net"
    summary = "cable net (goal net): tension-only cords on a grid, pinned edges, catches a ball"
    speed_budget = 5.0     # it steps its own 1,430-node net in ~10 sub-steps: on CPU, 2-4x a bare ball scene

    @dataclass
    class Config:
        ball: str = "ball"
        ball_radius: float = M.R_BALL
        origin: Vec = (2.12, -3.84, 0.0)            # bottom-left corner [m]
        across: Vec = (0.0, 1.0, 0.0)               # width direction
        up: Vec = (0.0, 0.0, 1.0)                   # height direction
        width: float = 7.68                         # m
        height: float = 2.56                        # m
        cell: float = 0.12                          # mesh [m]
        linear_density: float = 6.4e-3             # kg per m of cord
        ea: float = 600.0                           # cord EA [N] (ESTIMATE, knotted cord)
        strain0: float = -0.08                      # initial cord strain: < 0 slack, > 0 pretension (ESTIMATE)
        zeta: float = 0.5                           # cord damping ratio against the node mass (ESTIMATE)
        knot_friction: float = 40.0                 # N per cord: the knot slider's limit (ESTIMATE)
        knot_stiffness: float = 10.0                # the slider's pre-slide stiffness, x EA/rest (ESTIMATE)
        pins: tuple[bool, bool, bool, bool] = (True, True, True, True)   # left, right, bottom, top
        ground: bool = True                         # nodes stay at z >= ground_z
        ground_z: float = 0.0
        ground_mu: float = 0.5                      # net on turf (ESTIMATE)
        cord_radius: float = 0.01                   # m, contact radius of a cord (a 3 mm cord, padded)
        contact_k: float = 5.0e4                    # N/m², per metre of cord inside the ball
        contact_zeta: float = 0.5

        def __post_init__(self) -> None:
            if min(self.width, self.height, self.cell, self.linear_density, self.ea, self.contact_k) <= 0.0:
                raise ValueError("width, height, cell, linear_density, ea and contact_k must be > 0")
            if not -0.5 < self.strain0 < 0.5 or min(self.zeta, self.contact_zeta, self.knot_friction,
                                                    self.ground_mu) < 0.0 or self.knot_stiffness <= 0.0:
                raise ValueError("strain0 in (-0.5, 0.5); zeta, contact_zeta, knot_friction and ground_mu >= 0; "
                                 "knot_stiffness > 0")
            if abs(float(np.dot(_unit(self.across), _unit(self.up)))) > 1e-6:
                raise ValueError("across and up must be perpendicular")

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self.balls = ctx.bodies(c.ball)
        self.net = self._net_params()
        rows, cols = self.net.rows, self.net.cols
        self.n_nodes = rows * cols
        u, v = _unit(c.across), _unit(c.up)
        su, sv = c.width / (cols - 1), c.height / (rows - 1)
        grid = np.array([np.array(c.origin) + u * su * j + v * sv * i for i in range(rows) for j in range(cols)])
        self.nodes = self.add_state("nodes", ctx.array(np.tile(grid[None], (W, 1, 1)).astype(np.float32), wp.vec3))
        self.vel = self.add_state("node_vel", ctx.zeros((W, self.n_nodes), wp.vec3))
        self.knots = self.add_state("knots", ctx.zeros((W, self.n_nodes), wp.vec2))
        self.impulse = self.add_state("impulse", ctx.zeros(W, wp.vec3))
        self.dt_prev = self.add_state("dt_prev", ctx.zeros(W))
        self._x2, self._v2 = ctx.zeros((W, self.n_nodes), wp.vec3), ctx.zeros((W, self.n_nodes), wp.vec3)
        self._k2 = ctx.zeros((W, self.n_nodes), wp.vec2)
        self._fc = ctx.zeros((W, self.n_nodes), wp.vec3)
        self._c0, self._vb = ctx.zeros(W, wp.vec3), ctx.zeros(W, wp.vec3)
        g = np.asarray(ctx.model.gravity.numpy(), np.float64).reshape(-1, 3)
        if np.ptp(g, axis=0).max() > 0.0:
            raise PluginError(f"{self.label}: the worlds have different gravity; GoalNet uses one gravity for all")
        self._g = wp.vec3(*g[0])

    def _net_params(self) -> NK.Net:
        c = self.config
        cols, rows = int(round(c.width / c.cell)) + 1, int(round(c.height / c.cell)) + 1
        n = NK.Net()
        n.rows, n.cols = rows, cols
        su, sv = c.width / (cols - 1), c.height / (rows - 1)
        n.origin = wp.vec3(*c.origin)
        n.du, n.dv = wp.vec3(*(_unit(c.across) * su)), wp.vec3(*(_unit(c.up) * sv))
        n.node_mass = c.linear_density * (su + sv)               # one cord of each family per node
        n.rest_u, n.rest_v = su / (1.0 + c.strain0), sv / (1.0 + c.strain0)
        n.ea, n.zeta, n.t_f, n.k_slip = c.ea, c.zeta, c.knot_friction, c.knot_stiffness
        n.pin_left, n.pin_right, n.pin_bottom, n.pin_top = (int(p) for p in c.pins)
        n.ball_r = c.ball_radius + c.cord_radius
        n.k_c = c.contact_k
        n.c_c = 2.0 * c.contact_zeta * math.sqrt(c.contact_k * c.linear_density)
        n.ground_on, n.ground_z, n.ground_mu = int(c.ground), c.ground_z, c.ground_mu
        return n

    def node_rate(self) -> float:
        """Highest node frequency [rad/s]: four cords (each EA/rest, plus the knot slider's pre-slide spring) and the
        ball's push on its two cords' halves."""
        n = self.net
        slider = 1.0 + (n.k_slip if self.config.knot_friction > 0.0 else 0.0)
        k = slider * (2.0 * n.ea / n.rest_u + 2.0 * n.ea / n.rest_v) + n.k_c * (n.rest_u + n.rest_v)
        return math.sqrt(k / n.node_mass)

    def damping_rate(self) -> float:
        """Highest damping rate [1/s] of a node: four cord dashpots and the contact."""
        n = self.net
        per_cord = 2.0 * n.zeta * math.sqrt(0.5 * n.ea / min(n.rest_u, n.rest_v) * n.node_mass)
        return (4.0 * per_cord + n.c_c * (n.rest_u + n.rest_v)) / n.node_mass

    def substeps_for(self, dt: float) -> int:
        """Sub-steps for an exchange of ``dt``: ω·h <= 0.4 and γ·h <= 1 (symplectic Euler is stable to γ·h < 2), an
        even count for the ping-pong buffers."""
        need = math.ceil(max(self.node_rate() * dt / 0.4, self.damping_rate() * dt))
        return max(2, need + need % 2)

    def check_dt(self, dt: float) -> None:
        if self.substeps_for(dt) > 64:
            raise PluginError(f"{self.label}: the net needs {self.substeps_for(dt)} sub-steps per exchange of "
                              f"{dt:.3g} s: lower ea or contact_k, or the exchange step")

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        wp.launch(NK.net_apply, dim=self.ctx.world_count, device=self.ctx.device,
                  inputs=[self.impulse, self.balls, self.dt_prev], outputs=[forces.body_f])

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        W, dev, m = self.ctx.world_count, self.ctx.device, self.ctx.model
        wp.launch(NK.ball_motion, dim=W, device=dev,
                  inputs=[state_in.body_q, state_out.body_q, m.body_com, self.balls, float(dt)],
                  outputs=[self._c0, self._vb, self.impulse, self.dt_prev])
        steps = self.substeps_for(dt)
        h = dt / steps
        bufs = ((self.nodes, self.vel, self.knots, self._x2, self._v2, self._k2),
                (self._x2, self._v2, self._k2, self.nodes, self.vel, self.knots))
        for s in range(steps):
            x, v, kn, xo, vo, ko = bufs[s % 2]
            wp.launch(NK.net_substep, dim=(W, self.n_nodes), device=dev,
                      inputs=[x, v, kn, self.net, self._g, self._c0, self._vb, float(s * h), float(h)],
                      outputs=[xo, vo, ko, self._fc])
            wp.launch(NK.net_reduce, dim=W, device=dev, inputs=[self._fc, float(h)], outputs=[self.impulse])

    def test_scene(self) -> Any:
        from alloy.plugins.sports.laws_net import net_scene  # noqa: PLC0415

        return net_scene()

    def laws(self) -> list:
        from alloy.plugins.sports.laws_net import net_laws  # noqa: PLC0415

        return net_laws()


def _unit(a: Vec) -> np.ndarray:
    v = np.asarray(a, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("a zero direction")
    return v / n


__all__ = ["GoalNet"]
