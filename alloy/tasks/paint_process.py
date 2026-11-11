# SPDX-License-Identifier: Apache-2.0
"""The film's process stage for the painting tasks: each paint's drying, the porous substrates, and the surface-flow
settings that carry a paint's solids (design note §4.8, the plug-in's ``process`` stage).

The solids ride in the film as the species ``solids``, in volume per area [m]. Every source brings them in at the
paint's volume solids φ0 (:func:`source_species`). The stage then publishes per cell, in the film's ``read()``:
- ``dft``: the solids, the dry film;
- ``phi_s``: the solids fraction;
- ``laps``: passes that landed on film past the wet edge (where the old film holds at least 4 µm of solids).

Only solvent leaves, by evaporation or into a porous substrate, so the dry film is what was laid. Mist that dries in
the air lands as solids with no solvent (:func:`lay_solids`).

Sources (drying: ``(phi_m, phi_we, k_m)``):
- latex: random close packing 0.64, wet edge at 0.57 (10x the viscosity), k_m 0.006 m/s (Eastman's 2-4 min wet edge):
  the stage's defaults, as Gym-Anything's latex;
- fluid acrylic: φ0 0.65 is already past random close packing, since the binder coalesces as it dries. The wet edge
  at 0.70 and the end of evaporation at 0.80 are estimates. k_m is the latex's (the same water-borne binder, as
  Gym-Anything has it);
- aerosol enamel: sets at 0.60 (Gym-Anything's ``phi_set``), wet edge 0.45, k_m 0.030 m/s, solvent-borne, touch dry
  in 20 min (Gym-Anything's estimates). Gym-Anything's rate is 3.1e-7 m/s at 23 °C; the stage's water vapour at 20 °C
  gives 2.6e-7 m/s;
- watercolour paper: sorptivity 2.97e-6 m/√s from a Cobb60 of 23 g/m² (Reflex Aquarell data sheet), times 0.3 for
  acrylic (Gym-Anything's ``sorptivity_scale``), capacity 200 µm (0.51 mm at 40 % pores, an estimate);
- PVA-primed drywall: 0.1e-6 m/√s, capacity 5 µm; sealed plywood: 0.05e-6 m/√s, capacity 3 µm (Gym-Anything's
  estimates).
"""

from __future__ import annotations

import numpy as np

from alloy.tasks.paint_materials import film_material, paint

SOLIDS = "solids"
DRYING = {"latex_eggshell": (0.64, 0.57, 0.006),
          "acrylic_fluid": (0.80, 0.70, 0.006),
          "enamel_aerosol": (0.60, 0.45, 0.030)}  # fmt: skip
SUBSTRATES = {"paper": (0.3 * 2.97e-6, 200e-6),           # (sorptivity [m/√s], capacity [m])
              "drywall_primed": (0.1e-6, 5e-6),
              "plywood_sealed": (0.05e-6, 3.0e-6)}  # fmt: skip


def drying(name: str):  # noqa: ANN201 - surface_flow.process_specs.Drying
    """The stage's drying law for paint ``name``."""
    from alloy.plugins.surface_flow.process_specs import Drying  # noqa: PLC0415

    phi_m, phi_we, k_m = DRYING[name]
    return Drying(phi_0=paint(name).volume_solids, phi_m=phi_m, phi_we=phi_we, k_m=k_m, species=SOLIDS)


def film_settings(name: str, substrate: str | None = None, chart: int = 0, library=None) -> dict:  # noqa: ANN001
    """``SurfaceFlow`` keyword arguments for paint ``name``: its film material from the card, the solids species, and
    the process stage (drying, and ``substrate`` under ``chart`` if porous)."""
    from alloy.plugins.surface_flow.process_specs import Process, Substrate  # noqa: PLC0415

    subs = None
    if substrate is not None:
        s, cap = SUBSTRATES[substrate]
        subs = {chart: Substrate(sorptivity=s, capacity=cap)}
    return {"material": film_material(name, library), "species": (SOLIDS,),
            "process": Process(drying=drying(name), substrates=subs)}


def source_species(name: str) -> dict[str, float]:
    """What a source of paint ``name`` carries: its solids at φ0 (volume per volume)."""
    return {SOLIDS: paint(name).volume_solids}


def aged_film(name: str, depth: np.ndarray, age: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paint laid ``depth`` [m] thick ``age`` [s] ago (age < 0: none), dried since at the stage's rate: its depth now
    and its solids [m], for ``set_depth`` and ``set_species``."""
    d = drying(name)
    solids = np.asarray(depth, dtype=np.float64) * d.phi_0
    lost = d.rate * np.clip(np.asarray(age, dtype=np.float64), 0.0, None)
    now = np.maximum(np.asarray(depth, dtype=np.float64) - lost, solids / d.phi_m)
    return np.where(solids > 0.0, now, 0.0), solids


def dry_film(read: dict, depth: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    """The dry film [m] over ``depth``'s cells (chart 0) and where it came from: the stage's ``dft`` ("stage"), or
    without the stage the wet depth x φ0 ("h*phi0", which undercounts once the film dries)."""
    from alloy.tasks.paint_metrics import chart_field  # noqa: PLC0415

    d = chart_field(read, "dft", np.shape(depth))
    return (d, "stage") if d is not None else (np.asarray(depth) * paint(name).volume_solids, "h*phi0")


def lay_solids(flow, solids: np.ndarray, chart: int = 0) -> None:  # noqa: ANN001 - SurfaceFlow
    """Add dry solids [m, (nx, ny) or (W, nx, ny)] to ``chart``: dust that dried in the air, no solvent, no film."""
    flow.set_species(flow.species(SOLIDS, chart) + np.asarray(solids, dtype=np.float64), SOLIDS, chart=chart)


__all__ = ["DRYING", "SOLIDS", "SUBSTRATES", "aged_film", "dry_film", "drying", "film_settings", "lay_solids",
           "source_species"]
