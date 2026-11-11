# SPDX-License-Identifier: Apache-2.0
"""A stub film for the surface-flow supply's tests: exactly the interface arrays (docs/notes/surface-flow-interface.md
§2) around ``Supply``, in the agreed hook order, with a film that does not flow.

Each exchange it adds the pending ``src_acc`` to ``h`` (keeping it as ``last_acc``) and lets drain cells take
``min(drain_cap·dt, h)`` into ``drained``, so the supply's footprints, drains and ledgers can be checked in isolation
from WP15's film step. Like the film it leaves the per-exchange fields to the supply's clear rule. Also the scenes and
the faucet network those tests share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.fluid import P_ATM
from alloy.plugins.harness.run import run_scene
from alloy.plugins.harness.scene import TestScene, body_xform
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Nozzle, Supply
from alloy.plugins.surface_flow.tables import build_tables

RHO = 998.0
SLAB = (0.2, 0.2, 0.01)                      # half extents; top face at z = 0.02
MAINS = P_ATM + 2.0e5


@wp.kernel
def stub_film_kernel(dt: wp.float64, area: wp.array[wp.float64], h: wp.array2d[wp.float64],
                     src_acc: wp.array2d[wp.float64], last_acc: wp.array2d[wp.float64],
                     drain_of: wp.array[wp.int32], drain_cap: wp.array2d[wp.float64], drained: wp.array2d[wp.float64],
                     supplied: wp.array2d[wp.float64], flooded: wp.array2d[wp.int32], pooled: wp.array2d[wp.float64],
                     pooled_total: wp.array2d[wp.float64], src_mom: wp.array2d[wp.vec3],
                     last_mom: wp.array2d[wp.vec3]):
    w, c = wp.tid()
    last_mom[w, c] = src_mom[w, c]                            # the pending momentum, applied and consumed
    src_mom[w, c] = wp.vec3(0.0, 0.0, 0.0)
    acc = src_acc[w, c]
    h[w, c] = h[w, c] + acc
    supplied[w, c] = supplied[w, c] + acc * area[c]
    last_acc[w, c] = acc
    src_acc[w, c] = wp.float64(0.0)
    if drain_of[c] >= 0:
        take = wp.min(drain_cap[w, c] * dt, h[w, c])
        h[w, c] = h[w, c] - take
        drained[w, c] = drained[w, c] + take
    if flooded[w, c] >= 0:                                    # under a pool's water: all of it joins the pool
        take = h[w, c]
        h[w, c] = wp.float64(0.0)
        pooled[w, c] = pooled[w, c] + take
        pooled_total[w, c] = pooled_total[w, c] + take * area[c]


class StubFilm(ForcePlugin):
    """The interface arrays, a film that only accumulates, and ``Supply`` in the agreed hook order."""

    name = "sf_supply_stub"
    summary = "test stub: surface-flow interface arrays around Supply"
    internal = True

    @dataclass
    class Config:
        charts: tuple[Any, ...] = ()
        nozzles: tuple[Any, ...] = ()
        drains: tuple[Any, ...] = ()
        network: dict[str, Any] = field(default_factory=dict)
        lumps: tuple[tuple[str, float], ...] = ()     # (body label, area m²): WP9's lumped films, as the film registers
        stage: int = 0                                 # stage-drip entries (supply_stage), as a film stage registers
        stage_bodies: tuple[str | None, ...] = ()      # with src_drip_body: each entry's tool body (label)

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        charts = tuple(c.charts)
        self.tables = build_tables(ctx, charts)
        C = self.tables.cell_count
        f64, f32 = wp.float64, wp.float32
        self.h = self.add_state("h", ctx.zeros((W, C), f64))
        self.src_acc = ctx.zeros((W, C), f64)
        self.last_acc = ctx.zeros((W, C), f64)                 # the pending volume the last step applied
        self.cwet = ctx.zeros((W, C), wp.int32)
        # the per-exchange fields are state, as on the film: they persist with the supply's written box
        self.src_mom = self.add_state("src_mom", ctx.zeros((W, C), wp.vec3))
        self.last_mom = ctx.zeros((W, C), wp.vec3)             # the pending momentum the last step applied
        self.impact_p = self.add_state("impact_p", ctx.zeros((W, C), f32))
        self.impact_share = self.add_state("impact_share", ctx.zeros((W, C), f32))
        self.jet_cover = self.add_state("jet_cover", ctx.zeros((W, C), f32))
        self.jet_tau = self.add_state("jet_tau", ctx.zeros((W, C), f32))
        self.pool_jet_tau = self.add_state("pool_jet_tau", ctx.zeros((W, C), f32))   # as WP9 registers it on the film
        self.drained = ctx.zeros((W, C), f64)
        self.drain_cap = ctx.zeros((W, C), f64)
        self.drain_of = ctx.array(np.full(C, -1, np.int32), wp.int32)
        self.supplied = self.add_state("supplied", ctx.zeros((W, C), f64))
        self.flooded = self.add_state("flooded", ctx.full((W, C), -1, wp.int32))
        self.pooled = ctx.zeros((W, C), f64)
        self.pooled_total = self.add_state("pooled_total", ctx.zeros((W, C), f64))
        if c.lumps:
            self._register_lumps(ctx, tuple(c.lumps))
        for name, a in (stage_arrays(ctx, c.stage, bodies=tuple(c.stage_bodies)) if c.stage else {}).items():
            setattr(self, name, self.add_state(name, a))
        self.supply = Supply(c.nozzles, c.drains, c.network)
        self.supply.setup(ctx, self.tables, self)

    def _register_lumps(self, ctx: Any, lumps: tuple[tuple[str, float], ...]) -> None:
        """WP9's lump arrays for the given bodies, one lump each; the per-exchange ones are state."""
        for name, a in lump_arrays(ctx, lumps).items():
            setattr(self, name, a if name in ("lump_of_shape", "lump_area") else self.add_state(name, a))

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        self.supply.apply(state, forces, dt)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        self.dt = dt
        self.supply.before_film(state_in, dt)
        wp.launch(stub_film_kernel, dim=self.h.shape,
                  inputs=[wp.float64(dt), self.tables.area, self.h, self.src_acc, self.last_acc, self.drain_of,
                          self.drain_cap, self.drained, self.supplied, self.flooded, self.pooled, self.pooled_total,
                          self.src_mom, self.last_mom],
                  device=self.ctx.device)
        self.supply.after_film(state_out, dt)

    def test_scene(self) -> TestScene:
        return TestScene(build=slab_scene())


def stage_arrays(ctx: Any, entries: int, n_film: int = 0, bodies: tuple[str | None, ...] = ()) -> dict[str, wp.array]:
    """A film stage's drip arrays (``supply_stage.STAGE_ARRAYS``) for ``entries`` tools, zeroed; with ``bodies`` (a
    label or None per entry) also ``src_drip_body``."""
    W, E = ctx.world_count, entries
    out = {"src_drip_v": ctx.zeros((W, E), wp.float64), "src_drip_p": ctx.zeros((W, E), wp.vec3),
           "src_drip_u": ctx.zeros((W, E), wp.vec3), "src_drip_sp": ctx.zeros((W, E, max(n_film, 1)), wp.float64)}
    if bodies:
        nb = [ctx.bodies(b).numpy() if b is not None else np.full(W, -1) for b in bodies]
        out["src_drip_body"] = ctx.array(np.stack(nb, 1).astype(np.int32), wp.int32)
    return out


def lump_arrays(ctx: Any, lumps: tuple[tuple[str, float], ...], n_film: int = 0) -> dict[str, wp.array]:
    """WP9's lump arrays (``supply_lump.LUMP_ARRAYS``) for the given (body label, area m²), one lump each, as a
    lumped film registers them: every shape of the body belongs to its lump."""
    W, B = ctx.world_count, len(lumps)
    sb = ctx.model.shape_body.numpy()
    of = np.full((W, len(sb)), -1, np.int32)
    for b, (label, _) in enumerate(lumps):
        bodies = ctx.bodies(label).numpy()
        for w in range(W):
            of[w, sb == bodies[w]] = b
    out = {"lump_of_shape": ctx.array(of, wp.int32),
           "lump_area": ctx.array(np.asarray([a for _, a in lumps], np.float64), wp.float64),
           "lump_in": ctx.zeros((W, B), wp.float64), "lump_sp_in": ctx.zeros((W, B, max(n_film, 1)), wp.float64)}
    out.update({name: ctx.zeros((W, B), wp.float32) for name in ("lump_p", "lump_share", "lump_tau")})
    return out


def slab_scene(blocker: bool = False, slab_x: float = 0.0, crate: tuple[float, float] = (1.0, 0.0)):
    """A static slab (the chart, top at z = 0.02), a crate body on the ground, and optionally a static lid at
    z = 0.15 between the faucet and the slab."""
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(slab_x, 0.0, SLAB[2]), hx=SLAB[0], hy=SLAB[1], hz=SLAB[2], label="slab")
        rest = b.add_body(xform=body_xform(crate[0], crate[1], 0.02), label="crate")
        b.add_shape_box(rest, hx=0.02, hy=0.02, hz=0.02)
        if blocker:
            b.add_shape_box(-1, xform=body_xform(slab_x, 0.0, 0.15), hx=0.03, hy=0.03, hz=0.005, label="lid")
    return build


def faucet_network(tank_volume: float = 0.0, pool: bool = False) -> dict[str, Any]:
    """Mains -> valve -> line -> spout (to the air), about 0.1 L/s; a sink tank (0.16 m²) under the slab."""
    net = {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4},
                     {"name": "sink", "medium": "tank", "volume": tank_volume, "area": 0.16}],
           "edges": [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": MAINS, "cd_area": 5e-6,
                      "dp_lam": 1.0},
                     {"name": "spout", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": 1e-4,
                      "dp_lam": 1.0}]}
    if pool:
        net["pools"] = [{"tank": "sink", "floor": (0.0, 0.0, 0.05), "half": (0.2, 0.2)}]
    return net


def run_stub(plugin: StubFilm, periods: int = 20, build: Any = None, worlds: int = 2) -> StubFilm:
    """Run ``plugin`` on CPU (default: the slab scene) and return its set-up instance."""
    scene = TestScene(build=build or slab_scene(), ground=True, worlds=worlds, periods=periods)
    run = run_scene(scene, [plugin], "cpu")
    if run.error is not None:
        raise PluginError(f"stub run failed: {run.error}")
    return run.sim.plugins[plugin.label]


FAUCET = Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.3), exit_area=1e-4)
CHART = PlaneChart("slab", cells=(40, 40), edges=("wall", "open", "wall", "wall"))


def ledger_sides(p: StubFilm, edge: int = 1) -> tuple[np.ndarray, ...]:
    """Per world: (sent + swept + withdrawn, supplied + to_tank + lost + overspray + mist + caught + lumped) and (the
    nozzle edge's network ledger, sent + the water in flight); each pair is equal to float64 rounding."""
    r = p.supply.read()
    supplied = p.supplied.numpy().sum(axis=1)
    in_flight = -r["discharge"][:, edge] * p.dt
    return (r["sent"] + r["swept"] + r["withdrawn"],
            supplied + r["to_tank"] + r["lost"] + r["overspray"] + r["mist"] + r["caught"] + r["lumped"],
            -r["edge_ledger"][:, edge], r["sent"] + in_flight)


__all__ = ["CHART", "FAUCET", "MAINS", "RHO", "SLAB", "StubFilm", "faucet_network", "ledger_sides", "lump_arrays",
           "run_stub", "slab_scene", "stage_arrays"]
