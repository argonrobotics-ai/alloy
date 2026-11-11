# SPDX-License-Identifier: Apache-2.0
"""melt_chocolate (heat v2 step 4): a dark chocolate block melts on a hot plate and spreads.

A 30 x 30 x 10 mm block of dark chocolate (MPM paste; ``phase_materials`` ``dark_chocolate``, values ESTIMATE) rests
on a 120 x 120 x 5 mm plate held at T_plate (a thermostat heater, contact conductance ``H_CONTACT``) in still room air
(convection ``H_AIR`` at 22 °C). ``ParticleThermal`` melts it by the enthalpy method and writes μ, τ_y, friction and E
of each particle's (T, φ) into the MPM arrays, so the melted chocolate slumps and spreads.

Heating runs ``TIME_SCALE`` times faster than the motion (design §4): ``SECONDS`` of simulated motion are
``TIME_SCALE`` x ``SECONDS`` of heating (60 real minutes). The melt spreads that much slower relative to the heat; the
row records the scale.

The material values are the CENTRAL ones of the quoted ranges (``phase_materials``): k 0.3 W/(m·K), c 2000 J/(kg·K),
L 60 J/g. The pass depends on them and on the setup: a 45-55 °C, 60 mm plate passed 14 of 64 worlds in 40 minutes
(GPU smoke; the melt also ran off the plate and set on the floor). With the 120 mm plate at 50-60 °C, the slowest corner
(50 °C plate, 18 °C start) melts 90 % at 16 min and all of it by 25 min, and the fastest (60 °C, 24 °C) 90 % at 10 min;
the melted fraction never falls (L4, 2 worlds, 11-07). At the slow end of the ranges (k 0.2, c 2600 / 2000, L 78
J/g) a 1-D layer on a 45 °C plate needs about 54 min, at the fast end about 13 min (WP13 review of step 4).

Per world (seeded): start temperature U(18, 24) °C, plate U(50, 60) °C. Success per world: melted fraction (mass
mean of φ) >= 0.9 by the end, no particle above its plate's temperature (the maximum principle, 1e-6 K), and the
melted fraction never more than ``DROP`` below its peak (the plate is above the liquidus: melt must not set again).

Plate contact (WP13 re-review R1): MPM holds particles anywhere within a voxel of the plate, so the melt's rim sat
0.7-4.6 spacings up. With a 0.2-spacing gap the touching area fell to 0.66-0.71 of the puddle's base (20-60 min), the
rim stayed solid, and the melted fraction peaked at 0.952 (30 min) and fell to 0.938 (60 min). The gap is now a voxel,
and a particle over a lower one rests on it with the share of its footprint that the lower one does not cover
(``particle_kernels.resting_share``). The touching area over the base stays at 0.86-1.03 through the run. On the same
puddles, a plain 2-spacing band counts 1.96-2.14 of the base, and a yes/no 35° cone up to 1.28 once the melt is thin.
The row reports the touching area per base area and the share of particles with no neighbour inside the heat support
(MPM holds them, but they conduct nothing).

    python -m alloy.tasks.melt_chocolate --worlds 64 --device cuda:0 --out rows.jsonl
writes one row: success count, melted fraction, time to 90 % melted (real s), the largest fall below the peak,
footprint growth, the hottest particle against its plate, the height ratio, the touching area per base area, the
isolated share, and the plug-in's cost per step against the MPM step alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from scipy.spatial import cKDTree

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.thermal.particle import ParticleThermal
from alloy.plugins.thermal.phase_materials import material
from alloy.tasks.cook_common import add_block, draws, main_for, mpm_kwargs, run_periods, thermal_of

MATERIAL = "dark_chocolate"
SPACING = 1.25e-3                   # m, particle lattice (24 x 24 x 8 particles)
COUNTS = (24, 24, 8)
VOXEL = 2.0 * SPACING               # m: the router's min(10 mm / 4, 5 mm), 2 particles per voxel per axis
PLATE_HALF = (0.06, 0.06, 0.0025)   # m: wider than the puddle (on a 60 mm plate the melt ran off and set)
H_CONTACT = 300.0                   # W/(m²·K), block to plate (ESTIMATE: a dry solid contact, 100-1000)
GAP = VOXEL                         # m beyond the radius that still touches: MPM holds particles anywhere within a
#                                     voxel of the plate; a particle over a lower one rests on it (resting_share)
SUPPORT = 2.2 * SPACING             # the plug-in's default neighbourhood: a particle with nothing inside is isolated
DROP = 0.01                         # the melted fraction may not fall more than this below its peak (plate > liquidus)
H_AIR, T_AIR = 10.0, 295.15         # W/(m²·K) still air, K
T_START = (291.15, 297.15)          # K, per-world range
T_PLATE = (323.15, 333.15)          # K, per-world range (50-60 °C)
TIME_SCALE = 600.0
CONTROL_DT, DT = 0.02, 2.5e-3       # s (8 MPM steps per period)
SECONDS = 6.0                       # of motion: 60 real minutes of heating
PERIODS = round(SECONDS / CONTROL_DT)
MELTED = 0.9
HOT_TOL = 1.0e-6                    # K


def build(b: newton.ModelBuilder) -> None:
    """One world: the plate on the ground and the block on the plate, centred at the origin."""
    top = 2.0 * PLATE_HALF[2]
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, PLATE_HALF[2]), wp.quat_identity()), hx=PLATE_HALF[0],
                    hy=PLATE_HALF[1], hz=PLATE_HALF[2], label="plate")
    lo = (-0.5 * COUNTS[0] * SPACING, -0.5 * COUNTS[1] * SPACING, top)
    add_block(b, lo, COUNTS, SPACING, material(MATERIAL), T_START[0])


def plugin() -> ParticleThermal:
    return ParticleThermal(materials=(MATERIAL,), initial=(T_START[0],), heaters=("plate",),
                           heater_modes=("temperature",), heater_values=(T_PLATE[0],), heater_conductance=H_CONTACT,
                           heater_gap=GAP, ambient=T_AIR, convection=H_AIR, time_scale=TIME_SCALE)


def scene(worlds: int, periods: int, seed: int, device: str = "cpu") -> TestScene:
    """The task scene; its init draws each world's start and plate temperatures."""
    def init(sim: Any) -> None:
        therm = thermal_of(sim)
        if therm is None:
            return
        therm.set_temperature(draws(seed, worlds, *T_START, stream=0)[:, None] * np.ones((1, therm.P)))
        therm.set_input("heater_values", draws(seed, worlds, *T_PLATE, stream=1)[:, None])
    return TestScene(build=build, worlds=worlds, periods=periods, control_dt=CONTROL_DT, dt=DT, solver="mpm",
                     solver_kwargs=mpm_kwargs(VOXEL, device), init=init)


def _footprint(q: np.ndarray) -> np.ndarray:
    """[W] 95th percentile of the particles' horizontal distance from the centre [m]."""
    return np.percentile(np.hypot(q[..., 0], q[..., 1]), 95.0, axis=1)


def _height(q: np.ndarray) -> np.ndarray:
    """[W] 95th percentile of the particle centres' height above the plate [m] (one stray particle cannot set it)."""
    return np.percentile(q[..., 2], 95.0, axis=1) - 2.0 * PLATE_HALF[2]


def _base_area(q: np.ndarray) -> np.ndarray:
    """[W] the puddle's base [m²]: 1-spacing cells under the particles within 6 spacings of the plate (WP13 S6)."""
    low = q[..., 2] - 2.0 * PLATE_HALF[2] < 6.0 * SPACING
    return np.asarray([len(np.unique(np.floor(qw[m, :2] / SPACING), axis=0)) for qw, m in zip(q, low)]) * SPACING ** 2


def _isolated(q: np.ndarray) -> np.ndarray:
    """[W] share of particles with no other within the heat support: MPM can hold them, but they conduct nothing."""
    out = []
    for qw in q:
        d, _ = cKDTree(qw).query(qw, k=2)
        out.append(float((d[:, 1] >= SUPPORT).mean()))
    return np.asarray(out)


def run(worlds: int, device: str, seed: int = 0, with_plugin: bool = True, periods: int = PERIODS) -> dict:
    """Run the task in ``worlds`` worlds; the row (module doc)."""
    plug = plugin() if with_plugin else None
    sim = build_simulation(scene(worlds, periods, seed, device), [plug] if plug else [], device, worlds)
    n = sim.model.particle_count // worlds
    q0 = sim.state.particle_q.numpy().reshape(worlds, n, 3).copy()  # numpy() is a live view on the CPU
    if plug is None:
        ms, steps = run_periods(sim, periods)
        return {"task": "melt_chocolate", "plugin": False, "worlds": worlds, "ms_per_step": ms, "success": 0}
    plate = plug.inputs()["heater_values"].numpy()[:, 0].copy()
    melted, hottest = np.zeros(worlds), np.full(worlds, -np.inf)
    peak, drop = np.zeros(worlds), np.zeros(worlds)
    t90, t_all = np.full(worlds, np.nan), np.full(worlds, np.nan)

    def each(k: int) -> None:
        phi, temp = plug.liquid.numpy(), plug.temperature.numpy()
        melted[:] = phi.mean(axis=1)
        peak[:] = np.maximum(peak, melted)
        drop[:] = np.maximum(drop, peak - melted)
        hottest[:] = np.maximum(hottest, temp.max(axis=1))
        t = (k + 1) * CONTROL_DT * TIME_SCALE
        t90[np.isnan(t90) & (melted >= MELTED)] = t
        t_all[np.isnan(t_all) & (phi.min(axis=1) >= 0.99)] = t

    ms, steps = run_periods(sim, periods, each)
    q1 = sim.state.particle_q.numpy().reshape(worlds, n, 3)
    over = hottest - plate
    ok = (melted >= MELTED) & (over <= HOT_TOL) & (drop <= DROP)
    read = plug.read()
    contact = read["heater_area_m2"][:, 0] / _base_area(q1)
    return {"task": "melt_chocolate", "plugin": True, "worlds": worlds, "device": device, "seed": seed,
            "success": int(ok.sum()), "melted_min": float(melted.min()), "melted_mean": float(melted.mean()),
            "t90_real_s_max": float(np.nanmax(t90)) if np.isfinite(t90).any() else None,
            "t90_reached": int(np.isfinite(t90).sum()),
            "t_all_melted_real_s_max": float(np.nanmax(t_all)) if np.isfinite(t_all).any() else None,
            "hottest_over_plate_K_max": float(over.max()), "melted_drop_max": float(drop.max()),
            "contact_share_min": float(contact.min()), "contact_share_max": float(contact.max()),
            "isolated_share_max": float(_isolated(q1).max()),
            "footprint_growth_min": float((_footprint(q1) / _footprint(q0)).min()),
            "height_ratio_max": float((_height(q1) / _height(q0)).max()),
            "height_ratio_min": float((_height(q1) / _height(q0)).min()),
            "real_minutes": periods * CONTROL_DT * TIME_SCALE / 60.0, "time_scale": TIME_SCALE,
            "neighbor_overflow_max": int(read["neighbor_overflow"].max()), "ms_per_step": ms, "steps_timed": steps}


def main(argv: list[str] | None = None) -> int:
    return main_for(__doc__, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
