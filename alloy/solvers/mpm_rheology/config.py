# SPDX-License-Identifier: Apache-2.0
"""Herschel-Bulkley settings and the per-particle attributes ``rheo:consistency`` / ``rheo:flow_index``."""

from __future__ import annotations

from dataclasses import dataclass

from alloy._vendor import newton
import warp as wp

RHEO = "rheo"


@dataclass(frozen=True)
class RheologyConfig:
    """``gdot_min`` (1/s) sets eta inside plugs (where the yield projection holds the paste rigid anyway) and caps it at
    K gdot_min^(n-1); ``eta_min`` (Pa s) floors eta at very high shear rates. ``smooth``: the radius, in voxels, over
    which the strain-rate tensor is averaged (hat weights, same world) before eta is taken from it; 0 takes each
    particle's own. A particle's APIC velocity gradient carries noise of ~0.5-2 1/s in every component, which a
    shear-thinning eta turns into a fluid plug (docs/notes/viscoplastic-design.md §5). Two voxels give the same film
    speeds at 5 and 2.5 mm voxels; wider averages slow the films (the one-sided average at a base or surface)."""

    gdot_min: float = 1.0e-2
    eta_min: float = 1.0e-3
    smooth: float = 2.0


def register_rheology_attributes(builder: newton.ModelBuilder) -> None:
    """Per-particle K (Pa s^n) and n, default 0 (= no Herschel-Bulkley update for that particle)."""
    for name in ("consistency", "flow_index"):
        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name=name, frequency=newton.Model.AttributeFrequency.PARTICLE,
            assignment=newton.Model.AttributeAssignment.MODEL, dtype=wp.float32, default=0.0, namespace=RHEO))
