# SPDX-License-Identifier: Apache-2.0
"""Scene of rinse-garbage-can: Gym-Anything's can-wash bay, its hose and the scrubbed can (spec: [internal note] notes
wp8-results/surface/rinse_spec_more.md §2; gates and score in :mod:`alloy.tasks.can_wash_score`).

- Bay and hose: SC9's (``rinse_flush_floor.Bay``, ``Hose``). Quarry tile 2.0 x 1.8 m at 10 mm cells, sloping 1/96 to
  the floor drain, with the dunnage shelf's measuring charts. 60 psi mains through the EB-0107 valve: 3.3 gpm at
  24 m/s, nine jets in a 6° cone. There are no food scraps (the can task has none).
- The can: ``Can``, a ``rinse_vessels.Vessel``, Rubbermaid BRUTE 2632. LLDPE, 526 mm across the top tapering to
  448 mm at the base, 692 mm tall, 3 mm wall, 6 mm base, 4.5 kg, μ 0.45 on tile (``params.py:191,197``,
  ``ware.py:323-341``). One revolution chart wraps it at 85°, not their 95° (``can_wash_measure``: the film's limit is
  < 90°): the inner floor and wall, the rim, the outer wall, and the foot down to its outer corner, where the chart
  ends in an open edge (DC5's review). GA sheds film facing down deeper than 1 mm as drops that carry their soap; the
  film here has no drop stage, so an underside chart kept the outside's rinse water (0.95 L) and soap (27 mg).
  Its inside is a cavity pool, as on WP8's wash station: tank ``can_cavity``, spilling over the rim's lowest point
  through ``can_pour`` and the nozzle ``can_lip``. It starts empty; DC5's empty-cavity rule (ebb8bcc5e) starts the
  pool from the floor's film. Debris under the pool's water soaks and erodes under DC5's submerged-jet shear
  (a962d2fc5, WP9's flooded-soil law ddfba5810).
- Start (:meth:`CanWash.start`, their ``layout.py:512-513``, ``system.py:359-375,406-412``):
  - a wet soapy film, 30 µm at 0.75 kg/m³ detergent x U(0.67, 1.67) per world (world 0: x 1);
  - the walls' Langmuir load at that strength, Γm K c / (1 + K c), with K 20 m³/kg and Γm 5e-7 kg/m²;
  - debris on half the inner cells, 0.5e-3 x U(0.2, 1) x 2 kg/m² (20-100 µg/cm²). It soaks from 4 Pa dry to 0.5 Pa
    wet in 2 s, and jets and shear remove it at GA's k_i 2.0 and k_e 0.05. Removed debris rides the film as the
    species ``soil``.
- Differences from GA (disclosed):
  - the underside (outer bottom, 0.15 m², 6 % of their area, 3.2 mg of their start soap) is not charted, so it is out
    of the soap mean's area and of the start soap (51.4 mg here against their 56.5 mg, which also counts foam);
  - the chart has no pole cell: the discs r < ``r_min`` (62 mm full, 78 mm on CROP) carry no cells, which is 1 % of
    the area; GA's chart runs from the axis (``water/surfaces.py:124-137``);
  - 128 sectors at every radius: cells 12.8 mm wide at the rim, where GA's are at most 6.4 mm (round(2πr / cell),
    6 to 256), so a 1 cm gauge on the upper wall reads one coarse cell;
  - every world runs 60 psi mains (GA's seeds scale them by U(40, 80) / 60, ``layout.py:434``), as flush-floor;
  - the gates compare raw values, where GA gates its rounded report (µg 3 places, gauges 4, free water 3, tilt 2:
    ``task.py:484-502``); this matters only at the thresholds;
  - foam (WP15's G7 ruling: no foam model in v1; ``soap_free``'s foam part is not judged);
  - the moulded lip (559 mm);
  - the hose's capsule chain on the floor, and the valve's squeeze actuation (the opening is the input ``valve``);
  - the can's footprint blocking the floor film (the tile chart runs under the can);
  - per-world jitter of the can's start pose;
  - water and soap off the foot's open edge are booked as spilled (the film's ``spilled``) and leave the scene: they
    never reach the tile or the floor drain (GA's drops land), so the floor stays dry at stage 1.
- Contacts: ``rigid_gap`` 5 mm. Newton's 0.1 m default makes every shape pair within 0.2 m a contact, and MuJoCo
  Warp drops what overflows ``nconmax`` silently (WP8, WP15 11-06). Capacities are ``CAPACITY`` per world, and the
  run reads the overflow bits every period (``rinse_hand.contact_overflow``).

``CROP`` is the same task cropped for CPU tests: a 0.8 x 0.8 m tile at 20 mm round the drain and a 30 mm can chart.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.can_wash_measure import CHART_ANGLE_DEG, GAMMA_M
from alloy.tasks.rinse_flush_floor import RIGID_DT, RIGID_GAP, VALVE, Bay, Hose
from alloy.tasks.rinse_flush_policy import CONTROL_DT, parked
from alloy.tasks.rinse_vessels import Vessel, xform
from alloy.tasks.rinse_wash_station import POUR_EXIT, POUR_MAX

SPECIES = ("soap", "soil")
CAN, CAVITY, POUR, LIP = "can", "can_cavity", "can_pour", "can_lip"
CAPACITY = {"nconmax": 256, "njmax": 1024}   # per world; the standing can (CROP) peaks at 96 contacts, 96 rows
WASH = 0.75                           # kg/m³ detergent in the start film (c_wash)
WASH_SPREAD = (0.67, 1.67)            # per-world factor (seed > 0)
FILM_H = 30.0e-6                      # m, the start film
K_LANGMUIR = 20.0                     # m³/kg
K_DESORB = 0.05                       # 1/s
DEBRIS_SHARE = 0.5                    # of the inner cells
DEBRIS_LOAD = 0.5e-3 * 2.0            # kg/m², times U(0.2, 1)


@dataclass(frozen=True)
class Can(Vessel):
    """Rubbermaid BRUTE 2632, 32 gal (module doc): the outer wall tapers linearly from ``r_bot`` at the foot to
    ``r_top`` at the rim over ``height``; inner = outer - ``wall``; 124 L inside."""

    height: float = 0.692
    r_top: float = 0.263
    r_bot: float = 0.224
    wall: float = 0.003
    bottom: float = 0.006
    depth: float = 0.686
    mass: float = 4.5
    contact_angle: float = CHART_ANGLE_DEG
    ds: float = 6.0e-3
    n_theta: int = 128
    ring: int = 24
    mu: float = 0.45

    def radius(self, z: np.ndarray | float) -> np.ndarray | float:
        zz = np.clip(np.asarray(z, np.float64), 0.0, self.depth) + self.bottom
        return self.r_bot + (self.r_top - self.r_bot) * zz / self.height - self.wall

    def profile(self) -> tuple[tuple[float, float], ...]:
        """WP8's profile without the outer bottom: the chart ends at the foot's outer corner (module doc)."""
        return super().profile()[:-1]


def langmuir(conc: np.ndarray | float) -> np.ndarray:
    """The walls' equilibrium detergent load [kg/m²] at film concentration ``conc`` [kg/m³]."""
    kc = K_LANGMUIR * np.asarray(conc, np.float64)
    return GAMMA_M * kc / (1.0 + kc)


@dataclass(frozen=True)
class CanWash:
    """The bay, the hose and the can (bay frame: SC9's, x from their 5.2 m; the can's axis at ``at``)."""

    bay: Bay = field(default_factory=lambda: Bay(debris=(), can=None))   # no scraps, no stand-in can
    hose: Hose = field(default_factory=Hose)
    can: Can = field(default_factory=Can)
    at: tuple[float, float] = (0.90, 1.10)      # their (6.10, 1.10): 0.224 m from the drain's centre
    carry: str = "weight"                       # the pool's water on the can: a force, or "mass" (DC5's G2)
    valve_body: str = VALVE                     # the hose valve's body label (an arm's hand: can_wash_arms)
    table: Any = None                           # a ``can_wash_table.Turntable`` under the can (robot stage), or None

    can_chart: int = 1                          # charts: the floor, the can, then the shelf's (SC9's shelf_grams
                                                # counts every cell past the can's as the shelf's)

    def charts(self) -> tuple:
        floor, *shelf = self.bay.charts()
        return (floor, self.can.chart(CAN), *shelf)

    def network(self) -> dict:
        """SC9's hose, plus the can's cavity tank (empty) and its pour; detergent and soil dissolve in both."""
        from alloy.plugins.surface_flow.supply import Pool  # noqa: PLC0415

        net = self.hose.network()
        rim = float(self.can.radius(self.can.depth))
        cavity = {"name": CAVITY, "medium": "tank", "volume": 0.0, "area": math.pi * rim * rim,
                  "species": (0.0,) * len(SPECIES)}     # no capacity: the profile and spill= bound it (DC5)
        return {"nodes": [*net["nodes"], cavity],
                "edges": [*net["edges"], {"name": POUR, "kind": "source", "a": CAVITY, "flow": -POUR_MAX}],
                "pools": [Pool(CAVITY, body=CAN, cavity=self.can.cavity(), flood=True, spill=POUR,
                               **({"carry": self.carry} if self.carry != "weight" else {}))],
                "species": SPECIES}

    def nozzles(self) -> tuple:
        from alloy.plugins.surface_flow.supply import Nozzle  # noqa: PLC0415

        hose = tuple(dataclasses.replace(n, body=self.valve_body) if n.body == VALVE else n
                     for n in self.hose.nozzles())
        return (*hose, Nozzle(LIP, edge=POUR, rim_of=CAVITY, body=CAN, exit_area=POUR_EXIT))

    def make_flow(self) -> Any:
        """The film: water carrying detergent and soil, the can's walls adsorbing detergent, its debris soil law."""
        from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415
        from alloy.plugins.surface_flow.adsorption import Adsorption  # noqa: PLC0415
        from alloy.plugins.surface_flow.process_specs import Soil  # noqa: PLC0415

        soil = Soil(initial=0.0, charts=(self.can_chart,), sigma_dry=4.0, sigma_wet=0.5, t_soak=2.0, k_e=0.05,
                    k_i=2.0, k_wipe=0.0, species="soil")
        caps = tuple(GAMMA_M if k == self.can_chart else 0.0 for k in range(2 + len(self.bay.levels)))
        return SurfaceFlow(charts=self.charts(), material="water", species=SPECIES, process={"soil": soil},
                           adsorption=Adsorption("soap", K=K_LANGMUIR, k_d=K_DESORB, capacity=caps),
                           nozzles=self.nozzles(), network=self.network(), drains=self.bay.drains(), max_depth=0.01)

    def build(self, b: Any) -> None:
        """The bay, the hose valve docked, and the can standing at ``at`` (a world template)."""
        self.build_static(b, valve=True)

    def build_static(self, b: Any, valve: bool = False) -> None:
        """The bay, the hose valve docked when ``valve`` (else an arm holds it: can_wash_arms), the turntable when
        there is one, and the can standing at ``at``; bodies in stage 1's order (bay, valve, can)."""
        b.rigid_gap = RIGID_GAP
        self.bay.build(b)
        if valve:
            self.hose.add_valve(b, parked(self.bay))
        if self.table is not None:
            self.table.build(b, self.at, self.can)
        self.can.build(b, CAN, xform((*self.at, self.stand + self.can.bottom + 1.0e-4)))

    @property
    def stand(self) -> float:
        """The height the can's foot stands on [m]: the turntable's top, or the tile."""
        return 0.0 if self.table is None else float(self.table.top)

    def scene(self, init: Any, worlds: int = 1) -> Any:
        """The harness scene, stepped by MuJoCo at SC9's ``RIGID_DT`` with ``CAPACITY``."""
        from alloy.plugins.harness.scene import TestScene  # noqa: PLC0415

        return TestScene(build=self.build, init=init, worlds=worlds, periods=1, control_dt=CONTROL_DT, dt=RIGID_DT,
                         solver="mujoco", ground=False, solver_kwargs=dict(CAPACITY))

    def regions(self) -> dict[str, np.ndarray]:
        """``[nx * ny]`` bool per region (inner, rim, outer, foot) of the can chart's cells, by the arc length of
        their profile segment (cells run profile-major, ``n_theta`` around each segment)."""
        from alloy.plugins.surface_flow.charts import revolution_geometry  # noqa: PLC0415

        prof = revolution_geometry(self.can.chart(CAN)).prof
        mid = 0.5 * (prof[:-1, 2] + prof[1:, 2])
        return {k: np.repeat((mid >= a) & (mid < b), self.can.n_theta) for k, (a, b) in self.can.regions().items()}

    def start(self, flow: Any, worlds: int, seed: int = 0) -> dict[str, np.ndarray]:
        """Write the start film, the walls' load and the debris (module doc); the floor's slope. Returns the
        per-world detergent strength [kg/m³] and the debris maps ``[W, nx, ny]``."""
        rng = np.random.default_rng(seed)
        k, (nx, ny) = self.can_chart, flow.tables.shapes[self.can_chart]
        strength = WASH * np.where(np.arange(worlds) == 0, 1.0, rng.uniform(*WASH_SPREAD, worlds))
        inner = self.regions()["inner"]
        debris = np.zeros((worlds, nx * ny))
        for w in range(worlds):
            on = inner & (rng.random(nx * ny) < DEBRIS_SHARE)
            debris[w] = np.where(on, DEBRIS_LOAD * rng.uniform(0.2, 1.0, nx * ny), 0.0)
        h = np.full((worlds, nx, ny), FILM_H)
        flow.set_depth(h, chart=k)
        flow.set_species(h * strength[:, None, None], "soap", chart=k)
        flow.set_adsorbed(np.broadcast_to(langmuir(strength)[:, None, None], h.shape).copy(), "soap", chart=k)
        debris = debris.reshape(worlds, nx, ny)
        flow.process.set_soil(debris, chart=k)
        base = np.zeros(flow.read()["h"].shape, np.float32)
        base[:, flow.tables.cells_of(0)] = self.bay.base().ravel()
        flow.set_input("base", base)
        return {"strength": strength, "debris": debris}


#: The task cropped for CPU tests: 0.8 x 0.8 m of tile at 20 mm round the drain (its frame shifted by (0.5, 0.6) m
#: from the bay's), the shelf and holster where they are, and a 30 mm can chart.
CROP = CanWash(bay=Bay(size=(0.8, 0.8), cell=0.02, drain=(0.5, 0.3), region=((0.0, 0.0), (0.0, 0.0)),
                       holster=(0.58, -0.50, 1.055), shelf=((-1.35, -0.65), (-0.55, -0.05)), debris=(), can=None),
               can=Can(ds=0.03, n_theta=32, ring=16), at=(0.40, 0.50))


__all__ = ["CAN", "CAPACITY", "CAVITY", "CROP", "Can", "CanWash", "FILM_H", "LIP", "POUR", "SPECIES",
           "WASH", "langmuir"]
