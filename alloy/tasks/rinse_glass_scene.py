# SPDX-License-Identifier: Apache-2.0
"""Scene of task rinse_glassware (Gym-Anything's rinse-glassware): six soapy bar glasses on the soiled drainboard, the
pot sink as pools (``rinse_wash_station``), and the glass rack on the clean drainboard.

Sources (``surface_rinsing``; file:line):
- glasses: 4 rocks (H 95 mm, rim r 44.5, foot r 35, wall 3, base 12 mm, 0.32 kg) and 2 mixing glasses (H 150 mm,
  foot r 32, wall 3.5 mm, 0.42 kg): ``params.py:185-186``, ``ware.py:180-192`` (``rinse_vessels.Glass``); start poses
  on the soiled board around its centre x 2.3048: ``layout.py:448-455`` (seed > 0: ±2 cm jitter and a random yaw);
- each glass starts wet: 25 µm of wash water (0.75 g/L) on every cell, and its walls carry the soap adsorbed at that
  strength, Γ = Γ_m K c/(1 + K c) with Γ_m 1e-7 kg/m² (glass), K 20 m³/kg, desorbing at k_d 0.05/s (WP15's
  ``Adsorption``): ``layout.py:453-455``, ``system.py:349``, ``params.py:102, 123-129``; foam patches are not modelled
  (WP15's G7: no foam in v1);
- the rack: Cambro-style 0.500 m square, 5 x 5 compartments at 0.1 m pitch, 100 mm walls, 90 mm dividers, floor top
  0.017 m up, at the clean board's centre, y mid-bowl: ``ware.py:354-381``, ``layout.py:456-458``. Here it is static
  (theirs a 1.6 kg free body) with its floor as slats so a glass mouth down drains through it;
- drainboards 24 in either side of the bowls at the rim's height (their 6 mm pitch toward the bowls is left out).
Coordinates: x is theirs; y has the bowls' front (their 4.676) at 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from alloy._vendor import newton

from alloy.plugins.surface_flow import SurfaceFlow
from alloy.plugins.surface_flow.adsorption import Adsorption
from alloy.tasks.rinse_sink import BOWL
from alloy.tasks.rinse_vessels import Glass, mixing_glass, rocks_glass, xform
from alloy.tasks.rinse_wash_station import SOAP, Item, WashStation

INCH = 0.0254
BOARD = 24 * INCH                     # m, drainboard length along x
START = ((-0.16, 0.13), (0.0, 0.13), (0.16, 0.13), (-0.16, 0.30), (0.02, 0.32), (0.18, 0.33))   # layout.py:448-455
KINDS = ("rocks", "rocks", "rocks", "rocks", "mixing", "mixing")
FILM = 25.0e-6                        # m, the start film (layout.py:453-455)
GAMMA_M, K_ADS, K_DES = 1.0e-7, 20.0, 0.05     # glass: kg/m², m³/kg, 1/s (params.py:102, 123-129)
RACK = dict(half=0.25, pitch=0.10, wall=0.100, divider=0.090, floor=0.017, slat=0.020)


@dataclass(frozen=True)
class GlassStation(WashStation):
    """The sink, ``count`` glasses (theirs six) on the soiled board, the rack on the clean board."""

    count: int = 6
    ds: float = 2.0e-3                # the glasses' chart cells (theirs 2 mm); tests run coarser
    n_theta: int = 32

    @property
    def soiled_x(self) -> tuple[float, float]:
        return self.sink.x0 - BOARD, self.sink.x0

    @property
    def clean_x(self) -> tuple[float, float]:
        x0 = self.sink.x0 + 3 * BOWL[0]
        return x0, x0 + BOARD

    def glass(self, k: int) -> Glass:
        g = rocks_glass() if KINDS[k] == "rocks" else mixing_glass()
        return replace(g, ds=self.ds, n_theta=self.n_theta)

    def items(self) -> tuple[Item, ...]:
        cx = 0.5 * sum(self.soiled_x)
        return tuple(Item(f"glass{k}", g, (cx + dx, dy, self.sink.rim_z + g.bottom + 5.0e-4))
                     for k, ((dx, dy), g) in enumerate(zip(START[:self.count],
                                                           (self.glass(k) for k in range(self.count)), strict=True)))

    def rack_centre(self) -> np.ndarray:
        return np.array([0.5 * sum(self.clean_x), 0.5 * BOWL[1], self.sink.rim_z])

    def cell_centres(self) -> np.ndarray:
        """[25, 2] x, y of the rack's compartments."""
        c, p = self.rack_centre(), RACK["pitch"]
        g = (np.arange(5) - 2) * p
        return np.array([[c[0] + x, c[1] + y] for y in g for x in g])

    def build_boards_and_rack(self, b: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.35)
        z = self.sink.rim_z
        for name, (x0, x1) in (("board_soiled", self.soiled_x), ("board_clean", self.clean_x)):
            b.add_shape_box(-1, xform=xform((0.5 * (x0 + x1), 0.5 * BOWL[1], z - 0.005)), hx=0.5 * (x1 - x0),
                            hy=0.5 * BOWL[1], hz=0.005, cfg=cfg, label=name)
        c, h, t = self.rack_centre(), RACK["half"], 0.003
        for k, y in enumerate(np.arange(-h + 0.5 * RACK["slat"], h, 2 * RACK["slat"])):
            b.add_shape_box(-1, xform=xform(c + [0.0, y, RACK["floor"] - 0.003]), hx=h, hy=0.5 * RACK["slat"], hz=0.003,
                            cfg=cfg, label=f"rack_slat{k}")
        for s in (-1.0, 1.0):
            b.add_shape_box(-1, xform=xform(c + [s * h, 0.0, 0.5 * RACK["wall"]]), hx=t, hy=h, hz=0.5 * RACK["wall"],
                            cfg=cfg, label=f"rack_wall_x{int(s)}")
            b.add_shape_box(-1, xform=xform(c + [0.0, s * h, 0.5 * RACK["wall"]]), hx=h, hy=t, hz=0.5 * RACK["wall"],
                            cfg=cfg, label=f"rack_wall_y{int(s)}")
        for k in range(1, 5):
            off = (k - 2.5) * RACK["pitch"]
            zc = RACK["floor"] + 0.5 * RACK["divider"]
            b.add_shape_box(-1, xform=xform(c + [off, 0.0, zc]), hx=0.5 * t, hy=h, hz=0.5 * RACK["divider"], cfg=cfg,
                            label=f"rack_div_x{k}")
            b.add_shape_box(-1, xform=xform(c + [0.0, off, zc]), hx=h, hy=0.5 * t, hz=0.5 * RACK["divider"], cfg=cfg,
                            label=f"rack_div_y{k}")

    def build(self, b: newton.ModelBuilder) -> None:
        self.build_sink(b)
        self.build_boards_and_rack(b)
        self.build_items(b)

    def adsorption(self) -> Adsorption:
        return Adsorption("soap", K=K_ADS, k_d=K_DES, capacity=GAMMA_M)

    def wet(self, flow: SurfaceFlow) -> None:
        """Every glass's chart: 25 µm of wash water at the wash's soap strength, the walls at Langmuir equilibrium."""
        gamma = GAMMA_M * K_ADS * SOAP / (1.0 + K_ADS * SOAP)
        for k in range(len(self.items())):
            flow.set_depth(FILM, chart=k)
            flow.set_species(SOAP * FILM, "soap", chart=k)
            flow.set_adsorbed(gamma, "soap", chart=k)


__all__ = ["FILM", "GlassStation", "KINDS", "RACK"]
