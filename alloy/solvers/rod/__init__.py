# SPDX-License-Identifier: Apache-2.0
"""Alloy's rod direct solve on SolverVBD (SCR-WP2-10, engine item 4; owner WP2, WP10 reviews; SCR-WP9-ROD).

Newton's own rod (`newton.Rod` + `ModelBuilder.add_rod`) on stock SolverVBD fails D8: block Gauss-Seidel leaves it
limp. `SolverVBDRod` solves each rod chain directly; see `solver` for the method and the measured numbers, and
`topology` for what a chain is.

The ROD_VBD route (SCR-WP9-ROD) reaches it through the `vbd` capability. `with_rod_solve(cls)` mixes SolverVBDRod
over the entry's own VBD class (SolverVBDAlloy or SolverVBDMomentum), whose overrides are particle-side only, so both
keep their behaviour. `has_rod_chains(model)` says when an entry needs it.
"""

from __future__ import annotations

import functools

import numpy as np

from .solver import SolverVBDRod, hook_order_ok, rest_relative
from .topology import ROD, RodTopology


def has_rod_chains(model) -> bool:
    """True when `model` has an enabled Newton ROD joint (`ModelBuilder.add_rod`)."""
    if not getattr(model, "joint_count", 0):
        return False
    return bool(np.any((model.joint_type.numpy() == ROD) & model.joint_enabled.numpy().astype(bool)))


@functools.cache
def with_rod_solve(cls: type) -> type:
    """`cls` (a SolverVBD subclass) with SolverVBDRod's chain solve in front of it in the MRO."""
    if issubclass(cls, SolverVBDRod):
        return cls
    name = f"{cls.__name__}Rod"
    mixed = type(name, (SolverVBDRod, cls), {"__module__": __name__, "__qualname__": name,
                                             "__doc__": f"{cls.__name__} with SolverVBDRod's chains."})
    globals()[name] = mixed                 # a module attribute, so instances pickle by name (WP2's review)
    return mixed


__all__ = ["RodTopology", "SolverVBDRod", "has_rod_chains", "hook_order_ok", "rest_relative", "with_rod_solve"]
