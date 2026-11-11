# SPDX-License-Identifier: Apache-2.0
"""Measurements of rinse-garbage-can from the simulation: the can's fields of ``can_wash_score.CanReadout``.

The can carries detergent three ways, and soap_free counts all three (WP15's review of the verdict):
- the film on its chart: ``SurfaceFlow.read()["species"]`` [W, C, K] kg/m²;
- adsorbed on its walls (Langmuir, WP15's G6): ``read()["adsorbed"]`` [W, C, K] kg/m²;
- dissolved in its cavity pool: the network tank's species, ``FluidNetwork.species`` [W, nodes, K_net] kg.
The gauges take film + adsorbed per cell (``rinse_plates.gauges``, their ``task.py:74-88``); the soil gauge takes the
adhered soil (``rinse_plates.adhered``). Free water is the cavity pool plus the inner film above what clings
(``can_wash_score.free_film`` on the cells' world normals).

The can's substrate, LLDPE as their ``polymer`` (``params.py:107``, SUBSTRATES): contact angle 95°, adsorption
capacity Γ_m 5e-7 kg/m², film retention h_break 1e-4 m.

Disclosed modelling choice (WP15's ruling): the can's chart takes a contact angle of 85°, not their 95°. SurfaceFlow's
height-field film accepts angles in (0, 90)° only: past 90° it cannot recede into beads anyway, and 85° (tan 11.4)
pins a front on mm cells about as 95° would. None would mean fully wetting, the opposite of LLDPE. The scene stage
checks the water left on the drained can against their report before anything is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.can_wash_score import free_film

THETA_DEG = 95.0                      # LLDPE equilibrium contact angle (theirs)
CHART_ANGLE_DEG = 85.0                # the can chart's contact angle here (module doc: the film's limit is < 90°)
GAMMA_M = 5.0e-7                      # kg/m², adsorption capacity (Adsorption.capacity)
H_BREAK = 1.0e-4                      # m, film retention


@dataclass(frozen=True)
class CanProbe:
    """Where the can's quantities live: its chart cells as integer indices into the film's cells
    (``np.arange(C)[tables.cells_of(chart)]``), the inner region's cells, its cavity tank node, and the detergent's
    index in the film's and the network's species (-1: not carried)."""

    cells: np.ndarray
    inner: np.ndarray
    tank: int
    soap: int
    soap_net: int


def soap_cells(out: dict, probe: CanProbe) -> np.ndarray:
    """``[W, C]`` detergent per area on every film cell [kg/m²]: film + adsorbed (0 without adsorption)."""
    film = out["species"][:, :, probe.soap]
    ads = out.get("adsorbed")
    return film + (ads[:, :, probe.soap] if ads is not None else 0.0)


def soap_kg(out: dict, area: np.ndarray, net_species: np.ndarray | None, probe: CanProbe) -> np.ndarray:
    """``[W]`` detergent on and in the can [kg]: its cells' film + adsorbed, plus its cavity tank's dissolved load."""
    on = (soap_cells(out, probe)[:, probe.cells] * area[probe.cells]).sum(axis=1)
    if net_species is not None and probe.soap_net >= 0:
        on = on + net_species[:, probe.tank, probe.soap_net]
    return on


def free_ml(depth: np.ndarray, normal_z: np.ndarray, area: np.ndarray, pool_m3: np.ndarray,
            probe: CanProbe) -> np.ndarray:
    """``[W]`` free water [mL]: the cavity pool plus the inner film above what clings. ``depth`` [W, C] film depth,
    ``normal_z`` [W, len(inner)] the inner cells' world normal z, ``pool_m3`` [W] the cavity tank's water."""
    i = probe.inner
    h_break = np.full(len(i), H_BREAK)
    film = np.array([free_film(depth[w, i], normal_z[w], area[i], h_break) for w in range(depth.shape[0])])
    return (np.asarray(pool_m3, float) + film) * 1e6


def tilt_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    """``[W]`` angle of the body's z axis from the world's [deg] (quaternions x, y, z, w)."""
    q = np.asarray(quat_xyzw, float).reshape(-1, 4)
    zz = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)          # world z of the body's z axis
    return np.degrees(np.arccos(np.clip(zz, -1.0, 1.0)))


def world_normal_z(normals_body: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """``[W, n]`` world z of body-frame normals ``[n, 3]`` under each world's rotation ``[W, 4]`` (x, y, z, w)."""
    q = np.asarray(quat_xyzw, float).reshape(-1, 4)
    x, y, z, w = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    r20, r21, r22 = 2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)   # third row of R
    n = np.asarray(normals_body, float)
    return r20 * n[None, :, 0] + r21 * n[None, :, 1] + r22 * n[None, :, 2]


__all__ = ["CHART_ANGLE_DEG", "CanProbe", "GAMMA_M", "H_BREAK", "THETA_DEG", "free_ml", "soap_cells", "soap_kg",
           "tilt_deg", "world_normal_z"]
