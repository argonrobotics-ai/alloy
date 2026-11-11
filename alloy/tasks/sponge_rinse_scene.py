# SPDX-License-Identifier: Apache-2.0
"""Scene of rinse-sponge: Gym-Anything's pot sink and its Scotch-Brite sponge (spec: [internal note] notes
wp8-results/surface/rinse_spec_more.md §3; gates and score in :mod:`alloy.tasks.sponge_rinse_score`).

- The sink is WP8's station (``rinse_wash_station.WashStation`` with no ware). Three bowls are flooding pools filled to
  10 in: wash 0.75 g/L soap, rinse clean, sanitizer 100 ppm chlorine. The faucet feeds the wash bowl. The ledge deck
  behind the bowls carries a film chart (stainless, 70°: an estimate; their table has no steel entry). Drains, faucet
  and levers start closed.
- The sponge: a Scotch-Brite heavy-duty, 111 x 66 x 17 mm, 8 g (``ware.py:440-455``, ``params.py:192,207-218``). One
  rigid body:
  - a firm core box (99 x 54 mm, GA's 6 mm inset) set ``Q_STOP`` (10.5 mm) above the block's underside;
  - a porous pad (``Applicator(porous=True)``) on the underside, pressing the ledge's chart. Its stiffness is E/H
    (50 kPa over the 12 mm cellulose) over GA's effective contact area A_eff 30 cm², so the pad's force is their foam
    spring's E·A_eff·(ε + 3ε³): a 120 N wring takes ε to 0.48, leaving 52 % (their 41.4 mL).
  - pores (WP9's ``Lumped(pores=Pores(...))``): 0.9 of 111 x 66 x 12 mm, 79.1 mL free. They fill from a bowl at
    k_fill 2/s and swap with it at (0.2 + 5|v|)/s, and a squeeze puts their excess onto the ring of cells round
    the pad.
- Start (``start``): the sponge on the ledge behind the rinse bowl (their (3.2954 ± 0.03, 5.3328), y less the bowls'
  front 4.676 m; yaw 0), pores 75 % full (59.3 mL) of wash at 0.75 g/L x U(0.67, 1.67) per world (world 0: x 1):
  44.5 mg of soap. No soil, no film.
- Not modelled (disclosed):
  - GA's two-body compression slide: here one body, with the foam spring in the pad. A press is pose-controlled
    (WP9's ruling for v1): the hand sets the pad's depth for the force asked and the spring gives the strain;
  - the wring by two hands in the air: here the sponge is pressed on the ledge's left end, the squeezed water going
    to the ring of film cells round the pad (WP9, a682a0146), and set down on its dry right end;
  - the ledge: GA leaves it uncharted, a fixture that keeps what lands on it (``water/system.py:302-312``), so a pad
    there takes nothing back. Here it is charted (the pad's pressure needs a chart) with one drain over it to
    nowhere (``drains``): what lands on it leaves at once, so the porous pad cannot drink earlier squeezings (DC5's
    review). GA's fixtures account keeps that water; no gate reads it;
  - only the ledge compresses the pores (the pad's pressure on a chart): a press on a bowl's floor under water, which
    GA's slide squeezes, does nothing here. The press saturates at K_PAD x Q_STOP = 131 N, where the core lands:
    strain at most 0.50, the pores at least 39.6 mL (GA's slide stops at 433 N, ε 0.875). Both pass wrung_out
    above 120 N;
  - the gates compare raw values, where GA rounds c to 5 places and the fill to 4 (``task.py:595``);
  - the faucet's stream onto the sponge (refills are dunks in the rinse bowl);
  - the pore water's inertia: the body is built at its start water (67.2 g) and the water beyond that weighs on it
    (WP9's ``Lumped(built_water=...)``, a682a0146);
  - per-world pose jitter.
- Contacts: rigid_gap 5 mm (SC9's ``RIGID_GAP``), ``CAPACITY`` per world, overflow bits read every period. MuJoCo
  steps at ``SPONGE_DT`` (1 ms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.rinse_flush_floor import RIGID_GAP
from alloy.tasks.rinse_flush_policy import CONTROL_DT
from alloy.tasks.rinse_sink import BOWL, BOWLS, PotSink, drain_edge
from alloy.tasks.rinse_vessels import xform
from alloy.tasks.rinse_wash_station import SPECIES, WashStation, Y_BACK
from alloy.tasks.sponge_rinse_score import Ledge

SPONGE = "sponge"
CAPACITY = {"nconmax": 64, "njmax": 256}
BLOCK = (0.111, 0.066, 0.017)         # m, full edges
CELLULOSE_H = 0.012                   # m
MODULUS = 5.0e4                       # Pa, wet cellulose (their E_wet)
A_EFF = 30.0e-4                       # m², their foam spring's effective area
Q_STOP = 0.0105                       # m, the compression slide's stop
INSET = 0.006                         # m, the firm core's inset
MASS = 0.008                          # kg, dry
WASH = 0.75                           # kg/m³
WASH_SPREAD = (0.67, 1.67)
START_FILL = 0.75                     # of the free pore volume
BUILT_WATER = START_FILL * 0.9 * BLOCK[0] * BLOCK[1] * CELLULOSE_H   # m³: the body is built at its start water
WET_MASS = MASS + 998.0 * BUILT_WATER  # 67.2 g; the pores' water beyond BUILT_WATER weighs on it (WP9's built_water)
LEDGE_ANGLE = 70.0                    # deg, stainless (estimate)
LEDGE_CELL = 0.010                    # m
SPONGE_DT = 1.0e-3                    # s, MuJoCo's step: the pad's spring on the 67 g body rings at sqrt(k A / m)
                                      # ~ 430 rad/s, and a hand stiffer than it (to command the press's depth) needs it


@dataclass(frozen=True)
class Sponge:
    """The sponge's body, pad and pores (module doc); the body origin is the block's centre."""

    mu: float = 0.5

    def pores(self) -> Any:
        from alloy.plugins.surface_flow.lumped_specs import Pores  # noqa: PLC0415

        return Pores(phi=0.9, height=CELLULOSE_H, area=BLOCK[0] * BLOCK[1], modulus=MODULUS, q_stop=Q_STOP,
                     tool=SPONGE)

    def pad(self, chart: int) -> Any:
        """The underside as a porous pad of area A_eff with the block's aspect, stiffness E/H, on ``chart``."""
        from alloy.plugins.surface_flow.applicators import Applicator  # noqa: PLC0415

        hy = 0.5 * math.sqrt(A_EFF * BLOCK[1] / BLOCK[0])
        return Applicator(body=SPONGE, shape="rect", size=(hy * BLOCK[0] / BLOCK[1], hy),
                          center=(0.0, 0.0, -0.5 * BLOCK[2]), chart=chart, stiffness=MODULUS / CELLULOSE_H,
                          implicit=True, porous=True)

    def lumped(self) -> Any:
        from alloy.plugins.surface_flow.lumped_specs import Lumped  # noqa: PLC0415

        return Lumped(SPONGE, shapes=(f"{SPONGE}_core",), pores=self.pores(), built_water=BUILT_WATER)

    def build(self, b: Any, pos: tuple[float, float, float]) -> int:
        """The free body at ``pos``: the core box, ``Q_STOP`` above the underside, carrying ``WET_MASS``."""
        from alloy._vendor import newton  # noqa: PLC0415

        half = (0.5 * BLOCK[0] - INSET, 0.5 * BLOCK[1] - INSET, 0.5 * (BLOCK[2] - Q_STOP))
        body = b.add_body(xform=xform(pos), label=SPONGE)
        cfg = newton.ModelBuilder.ShapeConfig(density=WET_MASS / (8.0 * half[0] * half[1] * half[2]), mu=self.mu)
        b.add_shape_box(body, xform=xform((0.0, 0.0, -0.5 * BLOCK[2] + Q_STOP + half[2])), hx=half[0], hy=half[1],
                        hz=half[2], cfg=cfg, label=f"{SPONGE}_core")
        return body


@dataclass(frozen=True)
class SpongeSink:
    """WP8's sink with the ledge's chart, and the sponge (station frame: the bowls' front at y = 0)."""

    station: WashStation = field(default_factory=WashStation)
    sponge: Sponge = field(default_factory=Sponge)
    at: tuple[float, float] = (3.2954, 5.3328 - 4.676)

    @property
    def sink(self) -> PotSink:
        return self.station.sink

    @property
    def rest_z(self) -> float:
        """The sponge's origin resting on the ledge (its underside on the deck)."""
        return self.sink.rim_z + 0.5 * BLOCK[2]

    def ledge(self) -> Ledge:
        return Ledge.behind(self.sink, BOWL[1], Y_BACK)

    def charts(self) -> tuple:
        from alloy.plugins.surface_flow import PlaneChart  # noqa: PLC0415

        n = (int(round(3.0 * BOWL[0] / LEDGE_CELL)), int(round((Y_BACK - BOWL[1]) / LEDGE_CELL)))
        return (PlaneChart("ledge", cells=n, edges=("open", "open", "open", "wall"), contact_angle=LEDGE_ANGLE),)

    def drains(self) -> tuple:
        """One drain over the whole ledge, to nowhere: what lands on it leaves (module doc), as GA's uncharted
        fixture keeps it, so the porous pad never takes it back."""
        from alloy.plugins.surface_flow.supply_spec import Drain  # noqa: PLC0415

        return (Drain("ledge_keeps", chart=0, center=(0.0, 0.0, 0.0), radius=2.0 * BOWL[0]),)

    def make_flow(self) -> Any:
        from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415

        return SurfaceFlow(charts=self.charts(), material="water", species=SPECIES, tools=(self.sponge.pad(0),),
                           lumped=(self.sponge.lumped(),), network=self.station.network(),
                           nozzles=self.station.nozzles(), drains=self.drains(),
                           drops={"drop_radius": 2.0e-3, "floor_z": 0.0}, max_depth=0.01)

    def build(self, b: Any) -> None:
        b.rigid_gap = RIGID_GAP
        self.station.build(b)
        self.sponge.build(b, (*self.at, self.rest_z + 1.0e-4))

    def scene(self, init: Any, worlds: int = 1) -> Any:
        from alloy.plugins.harness.scene import TestScene  # noqa: PLC0415

        return TestScene(build=self.build, init=init, worlds=worlds, periods=1, control_dt=CONTROL_DT, dt=SPONGE_DT,
                         solver="mujoco", ground=False, solver_kwargs=dict(CAPACITY))

    def start(self, flow: Any, worlds: int, seed: int = 0) -> np.ndarray:
        """Close the drains and the faucet; fill the pores (module doc). Returns the per-world strength [kg/m³]."""
        rng = np.random.default_rng(seed)
        strength = WASH * np.where(np.arange(worlds) == 0, 1.0, rng.uniform(*WASH_SPREAD, worlds))
        for edge in (*(drain_edge(b) for b in BOWLS), "faucet"):
            flow.set_input(edge, np.zeros(worlds, np.float32))
        water = START_FILL * self.sponge.pores().free
        flow.lumped.set_water(np.full(worlds, water), body=0)
        flow.lumped.set_species(strength * water, "soap", body=0)
        return strength


__all__ = ["A_EFF", "BLOCK", "CAPACITY", "SPONGE", "Sponge", "SpongeSink", "START_FILL", "WASH"]
