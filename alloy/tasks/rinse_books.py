# SPDX-License-Identifier: Apache-2.0
"""The books a rinse task's ``no_simulation_fault`` closes (their ``ledger_closed``: water and surfactant to 1e-9
relative, ``task.py``), each as its change since the task's start (a start film is set, not supplied):
- ``water``: the film's ledger (film + spilled + evaporated + absorbed + drained + clipped + dripped + pooled - supplied,
  the supply's sweep and the tools' share included), against the larger of the start film and the water supplied;
- ``species``: each film species' ledger (film + walls' adsorbed load + what left - what came), against the larger of
  its start, what came and what left;
- ``supply``: the supply's cross ledger, sent + swept + withdrawn = supplied (pending inflow included, less what the
  drop stage brought: landed drops and lumps' wetting, already booked where they came from) + to_tank + lost +
  overspray + mist + caught + lumped, against what was sent.
The network's tanks keep their own books (FluidNetwork's laws); they are not re-checked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

REL = 1.0e-9


def dropped(flow: Any) -> np.ndarray | float:
    """``[W]`` film the drop stage supplied [m³]: drops that landed on a chart and lumps' water given to the cell
    under them (0 without a drop stage)."""
    drops = getattr(flow, "drops", None)
    if drops is None:
        return 0.0
    d = drops.read()
    return d["landed_volume"] + d["lump_wet_volume"]


@dataclass
class Books:
    """The start of the books; ``residuals`` and ``closed`` read them again."""

    water0: np.ndarray                # [W] the film ledger at the start
    species0: np.ndarray              # [W, K]

    @classmethod
    def start(cls, flow: Any) -> Books:
        r = flow.read()
        sp = r.get("species_ledger", np.zeros((len(r["ledger"]), 0)))
        return cls(water0=np.array(r["ledger"], np.float64), species0=np.array(sp, np.float64))

    def residuals(self, flow: Any) -> dict[str, np.ndarray]:
        """``[W]`` relative residual of each book (module docstring)."""
        r = flow.read()
        water = np.abs(r["ledger"] - self.water0) / np.maximum(np.maximum(np.abs(self.water0),
                                                                          r["supplied_volume"]), 1e-30)
        out = {"water": water}
        if self.species0.shape[1]:
            res = np.abs(r["species_ledger"] - self.species0)
            scale = np.maximum.reduce([np.abs(self.species0), r["species_supplied"].sum(axis=1),
                                       r["species_out"].sum(axis=1), np.full_like(res, 1e-30)])
            out["species"] = (res / scale).max(axis=1)
        if "supply_sent" in r:
            sent = r["supply_sent"] + r["supply_swept"] + r["supply_withdrawn"]
            got = (r["supplied_volume"] + r["pending_volume"] - dropped(flow) + sum(r[f"supply_{k}"] for k in (
                "to_tank", "lost", "overspray", "mist", "caught", "lumped")))
            out["supply"] = np.abs(sent - got) / np.maximum(sent, 1e-30)
        return out

    def closed(self, flow: Any) -> tuple[np.ndarray, float]:
        """``[W]`` every book within ``REL`` and the film finite; and the largest residual."""
        res = self.residuals(flow)
        worst = np.max(np.stack(list(res.values())), axis=0)
        return (worst < REL) & np.isfinite(flow.read()["h"]).all(axis=1), float(worst.max())


__all__ = ["REL", "Books", "dropped"]
