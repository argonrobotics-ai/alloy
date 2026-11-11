# SPDX-License-Identifier: Apache-2.0
"""cook_core (heat v2 step 4): roast a beef cube in a fan oven until its core reaches 74 °C, without drying it out.

A 30 mm cube of lean beef (MPM solid, ``phase_materials`` ``beef_lean``: ASHRAE 2014 heat capacities and latent
heat, k 0.48 W/(m·K)) sits on an open rack in a fan oven: every particle exchanges h_eff (T_oven − T) A with the oven
air through its exposed area A (``H_OVEN``, forced convection; ``particle_area``: a face reads one footprint, an edge
about 2, a corner about 3.3). Beef does not melt; its MPM yield stress and stiffness are fixed numerical stand-ins that
hold the cube's shape. Surface water boils at 100 °C as a lumped latent sink (0.718 of the vaporisation heat, beef's
water share): the surface holds at 100 °C while it dries. That is optimistic: a wet surface in an oven sits near the
wet-bulb temperature, below 100 °C, so the model heats the core somewhat faster than a real oven would. A particle that
has boiled off all of its water counts as dried out (burnt crust); it stays in the solver until WP9's
``retire(mask)``.

The 74 °C core target is a task choice: USDA's 165 °F (74 °C) is the safe temperature for poultry and leftovers; for
beef USDA gives 145 °F (63 °C) for whole cuts with a 3 min rest and 160 °F (71 °C) for ground beef.

Heating runs ``TIME_SCALE`` times faster than the motion (design §4; the cube does not move, so this is exact up to
its own sag). Per world (seeded): oven U(170, 200) °C, start U(2, 6) °C (from the fridge). Success per world: the
coldest particle (the core) reaches ``CORE`` within the time limit, before any particle has dried out (the cook takes
the cube out then; the run goes on to the limit to show when the crust would burn).

    python -m alloy.tasks.cook_core --worlds 64 --device cuda:0 --out rows.jsonl
writes one row: success count, time to a 74 °C core (real min), the hottest surface, the most-dried particle's
vapour fraction, and the plug-in's cost per step against the MPM step alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.thermal.particle import ParticleThermal
from alloy.plugins.thermal.phase_materials import material
from alloy.tasks.cook_common import add_block, draws, main_for, mpm_kwargs, run_periods, thermal_of

MATERIAL = "beef_lean"
SPACING = 2.5e-3                     # m (12 particles across)
COUNTS = (12, 12, 12)
VOXEL = 2.0 * SPACING                # m: the router's min(30 mm / 4, 5 mm), 2 particles per voxel per axis
MPM_SOLID = {"mpm:yield_stress": 2.0e4, "mpm:young_modulus": 1.0e6}  # Pa: numerical stand-ins (ESTIMATE)
H_OVEN = 25.0                        # W/(m²·K), fan oven (ESTIMATE: forced convection 15-40)
T_OVEN = (443.15, 473.15)            # K, per-world range
T_START = (275.15, 279.15)           # K
CORE = 347.15                        # K (74 °C)
TIME_SCALE = 600.0
CONTROL_DT, DT = 0.02, 2.5e-3        # s
SECONDS = 4.0                        # of motion: 40 real minutes
PERIODS = round(SECONDS / CONTROL_DT)


def build(b: newton.ModelBuilder) -> None:
    """One world: the cube on the ground (an open rack: the oven air reaches every face), centred at the origin."""
    lo = (-0.5 * COUNTS[0] * SPACING, -0.5 * COUNTS[1] * SPACING, 0.0)
    add_block(b, lo, COUNTS, SPACING, material(MATERIAL), T_START[0], MPM_SOLID)


def plugin() -> ParticleThermal:
    return ParticleThermal(materials=(MATERIAL,), initial=(T_START[0],), ambient=T_OVEN[0], convection=H_OVEN,
                           time_scale=TIME_SCALE)


def scene(worlds: int, periods: int, seed: int, device: str = "cpu") -> TestScene:
    """The task scene; its init draws each world's start and oven (the plug-in's ``ambient`` input) temperatures."""
    def init(sim: Any) -> None:
        therm = thermal_of(sim)
        if therm is not None:
            therm.set_temperature(draws(seed, worlds, *T_START, stream=0)[:, None] * np.ones((1, therm.P)))
            therm.set_input("ambient", draws(seed, worlds, *T_OVEN, stream=1))
    return TestScene(build=build, worlds=worlds, periods=periods, control_dt=CONTROL_DT, dt=DT, solver="mpm",
                     solver_kwargs=mpm_kwargs(VOXEL, device), init=init)


def run(worlds: int, device: str, seed: int = 0, with_plugin: bool = True, periods: int = PERIODS) -> dict:
    """Run the task in ``worlds`` worlds; the row (module doc)."""
    plug = plugin() if with_plugin else None
    sim = build_simulation(scene(worlds, periods, seed, device), [plug] if plug else [], device, worlds)
    if plug is None:
        ms, _ = run_periods(sim, periods)
        return {"task": "cook_core", "plugin": False, "worlds": worlds, "ms_per_step": ms, "success": 0}
    lv = material(MATERIAL).latent_vapor
    mass = sim.model.particle_mass.numpy()[plug._gidx.numpy()].reshape(worlds, plug.P)
    core, dried = np.zeros(worlds), np.zeros(worlds)
    t_core, t_dry, dried_at_core = np.full(worlds, np.nan), np.full(worlds, np.nan), np.full(worlds, np.nan)

    def each(k: int) -> None:
        t = (k + 1) * CONTROL_DT * TIME_SCALE / 60.0
        core[:] = plug.temperature.numpy().min(axis=1)
        dried[:] = np.maximum(dried, (plug.boiled.numpy() / (mass * lv)).max(axis=1))
        now = np.isnan(t_core) & (core >= CORE)
        t_core[now], dried_at_core[now] = t, dried[now]
        t_dry[np.isnan(t_dry) & (dried >= 1.0)] = t

    ms, steps = run_periods(sim, periods, each)
    read = plug.read()
    ok = np.isfinite(t_core) & ~(t_dry <= t_core)
    return {"task": "cook_core", "plugin": True, "worlds": worlds, "device": device, "seed": seed,
            "success": int(ok.sum()), "core_C_min": float(core.min() - 273.15),
            "t_core_real_min_max": float(np.nanmax(t_core)) if np.isfinite(t_core).any() else None,
            "core_reached": int(np.isfinite(t_core).sum()),
            "dried_at_core_max": float(np.nanmax(dried_at_core)) if np.isfinite(dried_at_core).any() else None,
            "t_first_dry_real_min_min": float(np.nanmin(t_dry)) if np.isfinite(t_dry).any() else None,
            "dried_max": float(dried.max()),
            "surface_C_max": float(read["temperature"].max() - 273.15),
            "evaporated_in_solver_max": int(read["evaporated_in_solver"].max()),
            "real_minutes": periods * CONTROL_DT * TIME_SCALE / 60.0, "time_scale": TIME_SCALE,
            "neighbor_overflow_max": int(read["neighbor_overflow"].max()), "ms_per_step": ms, "steps_timed": steps}


def main(argv: list[str] | None = None) -> int:
    return main_for(__doc__, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
