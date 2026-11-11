# SPDX-License-Identifier: Apache-2.0
"""Integration profiles: which real parts `alloy.integration.wiring` builds for a scene, and how (SPEC §3.7 presets).

A profile names a precision preset (``alloy.precision.config.preset``) plus the integration choices the preset does
not cover: graph capture, which optional parts to inject, layer capacity and the coupling policy.

| profile | preset | graph | health | gap monitor | notes |
|---|---|---|---|---|---|
| standard | standard | CUDA | full, latch | when precision pairs exist | default |
| precision | precision | CUDA | full every step, raise on INVALID | yes | sub-mm scenes |
| fast | fast | CUDA | cheap every 10 steps | no | throughput runs |
| debug | standard | off | full | yes | eager everywhere, CPU-friendly |
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Literal

from alloy.coupling.types import SchedulerPolicy
from alloy.precision.types import PrecisionConfig
from alloy.scene.steppers import ContactBudget

__all__ = ["IntegrationProfile", "PROFILES", "profile"]

GraphMode = Literal["auto", "on", "off"]
PartMode = Literal["auto", "always", "never"]


@dataclass(frozen=True)
class IntegrationProfile:
    """What to build around a ``BuiltScene`` (see the module table)."""

    name: str
    preset: Literal["fast", "standard", "precision", "reference"] = "standard"
    native_contacts: Literal["mujoco", "newton"] | None = None   # None: the preset's (PrecisionConfig default)
    graph: GraphMode = "auto"           # auto: CUDA graphs on CUDA, eager on CPU; on: CPU graphs too; off: eager
    contact_layer: PartMode = "auto"    # auto: precision pairs, or any scene MuJoCo does not collide on its own
    gap_monitor: PartMode = "auto"      # auto: when the layer has precision pairs
    health: bool = True
    topology: bool = True               # a TopologyController when the scene has bonds
    stock_coupler: bool = False         # True: WP2's stock SolverCoupledProxy instead of WP4's Scheduler
    precision_contacts_per_world: int = 256
    gap_sample_spacing_m: float = 5e-4
    scheduler_policy: SchedulerPolicy | None = None
    contact_budget: ContactBudget | None = None   # per-world caps of the fallback contact buffer (SCR-WP4-6)
    mjw_route_cg: bool = False          # SCR-DC5-4 (ii), WP5: CG on Alloy's ordered MuJoCo route (opt-in: 1.64x on T09)
    mjw_route: bool = False             # WP5, coordinator 2026-11-06 06:23Z: ask for Alloy's ordered MuJoCo route (GPU
    #                                     run-to-run repeat) and build MuJoCo entries with a sparse Jacobian, its path.
    #                                     Opt-in: about 1.3x the MuJoCo step; the default stays off

    def precision_config(self, device: str) -> PrecisionConfig:
        """The preset for ``device``'s GPU tier, with this profile's ``native_contacts``, ``mjw_route`` and
        ``mjw_route_cg``."""
        from alloy.precision.config import gpu_tier, preset  # noqa: PLC0415 - WP5

        cfg = preset(self.preset, gpu_tier(device))
        if self.native_contacts is not None:
            cfg = dataclasses.replace(cfg, native_contacts=self.native_contacts)
        if self.mjw_route_cg:                       # opt-in only: the preset's own value stands otherwise
            cfg = dataclasses.replace(cfg, mjw_route_cg=True)
        if self.mjw_route:                          # opt-in only, as mjw_route_cg
            cfg = dataclasses.replace(cfg, mjw_route=True)
        if self.name == "debug":
            cfg = dataclasses.replace(cfg, checks=dataclasses.replace(cfg.checks, on_invalid="latch"))
        return cfg

    def wants_graph(self, is_cuda: bool) -> tuple[bool, bool]:
        """(use_graph, cpu_graph) for a device."""
        if self.graph == "off":
            return False, False
        if self.graph == "on":
            return True, not is_cuda
        return is_cuda, False


PROFILES: dict[str, IntegrationProfile] = {
    "standard": IntegrationProfile("standard"),
    "precision": IntegrationProfile("precision", preset="precision", gap_monitor="always"),
    "fast": IntegrationProfile("fast", preset="fast", gap_monitor="never"),
    "debug": IntegrationProfile("debug", graph="off", gap_monitor="always"),
}


def profile(name: str | IntegrationProfile = "standard", **overrides) -> IntegrationProfile:
    """A named profile with field overrides, e.g. ``profile("standard", native_contacts="newton", graph="on")``."""
    base = name if isinstance(name, IntegrationProfile) else PROFILES.get(name)
    if base is None:
        raise ValueError(f"unknown integration profile {name!r}; expected one of {sorted(PROFILES)}")
    return dataclasses.replace(base, **overrides) if overrides else base
