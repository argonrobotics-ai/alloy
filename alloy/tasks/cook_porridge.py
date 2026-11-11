# SPDX-License-Identifier: Apache-2.0
"""cook_porridge (heat v2): reheat a pot of oat porridge on a hob while a spoon turns it over.

240 mL of oat porridge (SPH liquid, ``phase_materials`` ``porridge_oat``: viscosity ESTIMATE 10 Pa·s at 20 °C falling
to 4 Pa·s at 74 °C, which ``ParticleThermal`` writes into ``sph:viscosity`` each exchange) fills an 80 x 75 mm pot
40 mm deep. The pot floor is a power heater: the hob's power x ``EFFICIENCY`` (0.84) shared by the particles that touch
it. Porridge does not convect (the heat model has no buoyancy: docs/known_gaps/heat-no-buoyancy.md), so all the heat
enters a bottom layer an eighth of the mass, which would boil within about 10 s unless the spoon carries it away.

The spoon is a flat blade (6 x 65 x 20 mm, across the pot) on a scripted fold loop at ``SPEED``: down 8 mm off the -x
wall, along the floor 1.5 mm above it to 6 mm off the +x wall, up that wall out of the porridge; then the same
mirrored. Each floor stroke pushes the bottom layer to a wall and up it while the porridge above sinks behind the
blade, so the pot turns over twice per loop. The blade never comes closer to a wall than a particle's width: at 3 mm
it wedged particles into the floor corners, where they stayed on the hob and boiled dry. It only translates: the SPH
solver advances a non-proxy body with its own velocity, set each control period to the chord of the path.

Heating and motion run at the same time scale (stirring races the hob, so a time scale would not be exact). Water in
the porridge boils at 100 °C as a lumped latent sink; a particle that boils off all of its water has burnt on.

Per world (seeded): start U(4, 20) °C (from the fridge or the counter), hob U(800, 1200) W. The cook turns the hob off
once every portion reaches 74 °C (the four 60 mL portions: the porridge split in two by x, then each half in two by
height); stirring goes on.

**Verdict against Gym-Anything's reheat-porridge** (ingredient-mixing; ``ga_verdict``, row key ``"ga"``). This is a
smaller scene than Gym-Anything's (``SCENE``): 240 mL in an 80 x 75 mm pot against their 152 mm saucepan, another hob
draw and start temperature, and a scripted spoon in place of their arm, thermometer and knob. Of their 5 criteria the
port judges 3 from the simulation state, and success per world is those 3:
- hot_through: every portion at 74 °C or more with the portions within ``SPREAD`` (15 K) of each other, within the
  port's ``SECONDS`` (their limit is none; they read 39 mm cells, the port four 60 mL slabs);
- not_scorched: burnt on at most 2 cm² (particles that boiled off all their water, as footprints on the floor);
- contained: at most 1 % of the porridge outside the pot at the end.
Not judged (None in the verdict, never counted): vessel_in_place (the pot is a static body: it cannot leave the hob)
and appliance_off (the scripted cook switches the hob off itself; the scene has no knob).

    python -m alloy.tasks.cook_porridge --worlds 64 --device cuda:0 --out rows.jsonl
writes one row: success count, the verdict (``"ga"``), time to warm, the portion spread then, the coldest portion at
the end, the most boiled particle, the burnt area, the spill, the turnover (share of the start's bottom layer that
reached the top half), the hob's delivered energy against η·P·t, and the plug-in's cost per step against the SPH step
alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.thermal.particle import ParticleThermal
from alloy.plugins.thermal.phase_materials import material
from alloy.tasks.cook_common import FoldLoop, draws, ga_verdict, main_for, run_periods, thermal_of

MATERIAL = "porridge_oat"
RADIUS = 2.5e-3                      # m, SPH particle radius (spacing 5 mm)
COUNTS = (16, 15, 8)                 # 1,920 particles = 240 mL
INNER = (0.08, 0.075)                # m, pot inside
DEPTH = COUNTS[2] * 2.0 * RADIUS     # m, porridge depth (40 mm)
WALL, POT_H = 0.006, 0.07            # m
EFFICIENCY = 0.84                    # hob to porridge (Gym-Anything's reheat task)
HOB = (800.0, 1200.0)                # W, per-world range
T_START = (277.15, 293.15)           # K
BLADE_HALF = (0.003, 0.0325, 0.010)  # m: 6 mm thick, 65 mm across the 75 mm pot, 20 mm tall
CLEAR = 6.0e-3                       # m, blade to the x wall it sweeps to and rises along (> a particle across)
DESCENT_CLEAR = 8.0e-3               # m, blade to the x wall it comes down along (3 mm wedged particles in)
FLOOR_CLEAR = 1.5e-3                 # m, blade to the floor on the floor stroke
X_END = 0.5 * INNER[0] - BLADE_HALF[0] - CLEAR          # m, blade centre at the end of a floor stroke
X_DOWN = 0.5 * INNER[0] - BLADE_HALF[0] - DESCENT_CLEAR  # m, blade centre on a descent
Z_LOW = FLOOR_CLEAR + BLADE_HALF[2]                     # m, blade centre on the floor stroke
Z_HIGH = DEPTH + 5.0e-3 + BLADE_HALF[2]                 # m, blade centre on the return, clear of the porridge
SPEED = 0.12                         # m/s along the loop
WARM, SPREAD = 347.15, 15.0          # K: every portion >= 74 °C, portions within 15 K
BURNT_CM2, SPILT = 2.0, 0.01         # Gym-Anything's limits: burnt-on area [cm²], share outside the pot
GA_TASK = "ingredient_mixing/reheat-porridge"
SCENE = ("smaller than Gym-Anything's: 240 mL of porridge in an 80 x 75 mm pot 40 mm deep (theirs: a 152 mm "
         "saucepan 89 mm deep); hob U(800, 1200) W (theirs: 580 W at seed 0); start U(4, 20) C (theirs: 5 C); four "
         "60 mL slabs as portions (theirs: 39 mm cells); a kinematic spoon on a scripted loop (theirs: an arm with "
         "tools, a thermometer and a knob)")
NOT_JUDGED = {"vessel_in_place": "the pot is a static body here: it cannot leave the hob or tip",
              "appliance_off": "the scripted cook switches the hob off itself; the scene has no knob"}
CONTROL_DT, DT = 0.02, 1.25e-3       # s
SECONDS = 120.0
PERIODS = round(SECONDS / CONTROL_DT)
CHECK = 25                           # periods between portion checks (0.5 s: portions need a sort)

SPOON = FoldLoop(INNER[0], BLADE_HALF, Z_HIGH, CLEAR, DESCENT_CLEAR, FLOOR_CLEAR, SPEED)
LOOP = SPOON.length                  # m (one loop: 2.5 s, two strokes)


def _wall(b: newton.ModelBuilder, centre: tuple[float, float, float], half: tuple[float, float, float], label: str,
          body: int = -1) -> None:
    from alloy.solvers.sph.config import BOUNDARY_SDF  # noqa: PLC0415 - the SPH package registers on import

    b.add_shape_box(body, xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), hx=half[0], hy=half[1],
                    hz=half[2], custom_attributes={"sph:shape_boundary": BOUNDARY_SDF}, label=label)


def spoon_centre(t: float) -> np.ndarray:
    """The blade's centre at time t [s] on the fold loop (constant speed, period LOOP / SPEED)."""
    return SPOON.centre(t)


def build(b: newton.ModelBuilder) -> None:
    """One world: the pot (floor labelled "pot_floor", the hob), the porridge and the spoon."""
    from alloy.solvers.sph.build import add_fluid_block, register_custom_attributes  # noqa: PLC0415
    from alloy.solvers.sph.config import DFSPHConfig  # noqa: PLC0415

    register_custom_attributes(b)
    hx, hy, t = 0.5 * INNER[0], 0.5 * INNER[1], 0.5 * WALL
    _wall(b, (0.0, 0.0, -t), (hx + WALL, hy + WALL, t), "pot_floor")
    for s in (-1.0, 1.0):
        _wall(b, (s * (hx + t), 0.0, 0.5 * POT_H), (t, hy + WALL, 0.5 * POT_H), f"pot_x{int(s)}")
        _wall(b, (0.0, s * (hy + t), 0.5 * POT_H), (hx, t, 0.5 * POT_H), f"pot_y{int(s)}")
    mat = material(MATERIAL)
    add_fluid_block(b, DFSPHConfig(particle_radius=RADIUS), (-hx, -hy, 0.0), COUNTS, rest_density=mat.density,
                    viscosity=float(mat.viscosity(T_START[0], 1.0)))
    spoon = b.add_body(xform=wp.transform(wp.vec3(*spoon_centre(0.0)), wp.quat_identity()), mass=0.05,
                       inertia=wp.mat33(np.eye(3) * 1.0e-5), label="spoon")
    _wall(b, (0.0, 0.0, 0.0), BLADE_HALF, "spoon_blade", body=spoon)


def plugin() -> ParticleThermal:
    return ParticleThermal(materials=(MATERIAL,), initial=(T_START[0],), heaters=("pot_floor",),
                           heater_modes=("power",), heater_values=(HOB[0],), heater_efficiency=EFFICIENCY)


def scene(worlds: int, periods: int, seed: int) -> TestScene:
    """The task scene; its init draws each world's start temperature and hob power."""
    def init(sim: Any) -> None:
        therm = thermal_of(sim)
        if therm is not None:
            therm.set_temperature(draws(seed, worlds, *T_START, stream=0)[:, None] * np.ones((1, therm.P)))
            therm.set_input("heater_values", draws(seed, worlds, *HOB, stream=1)[:, None])
    return TestScene(build=build, worlds=worlds, periods=periods, control_dt=CONTROL_DT, dt=DT, solver="sph",
                     solver_kwargs={"particle_radius": RADIUS}, ground=False, init=init)


def portions(q: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """[W, 4] mean temperature of the four equal portions: split by x into halves, each half by height."""
    out = np.empty((q.shape[0], 4))
    for w in range(q.shape[0]):
        halves = np.array_split(np.argsort(q[w, :, 0], kind="stable"), 2)
        for h, idx in enumerate(halves):
            for k, part in enumerate(np.array_split(idx[np.argsort(q[w, idx, 2], kind="stable")], 2)):
                out[w, 2 * h + k] = temp[w, part].mean()
    return out


def spilt(q: np.ndarray) -> np.ndarray:
    """[W] share of each world's porridge outside the pot: beyond the inside walls by more than a particle's radius,
    or below the floor (particles of one material: equal mass). Above the pot, over its opening, is not outside."""
    inside = np.all(np.abs(q[..., :2]) <= 0.5 * np.asarray(INNER) + RADIUS, axis=2) & (q[..., 2] > -RADIUS)
    return 1.0 - inside.mean(axis=1)


def verdict(hot: np.ndarray, burnt_cm2: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, dict]:
    """(success per world, the Gym-Anything verdict) from the three judged measures (module doc)."""
    return ga_verdict(GA_TASK, {"hot_through": hot, "not_scorched": burnt_cm2 <= BURNT_CM2, "contained": out <= SPILT,
                                "vessel_in_place": None, "appliance_off": None}, NOT_JUDGED, SCENE,
                      also=f"hot_through within the port's {SECONDS:g} s run (Gym-Anything states no time limit)")


def _stir(sim: Any, spoons: np.ndarray, k: int) -> None:
    """Set the spoons' velocity for period k + 1: the chord of the loop over that period."""
    SPOON.stir(sim, spoons, k, CONTROL_DT)


class _Cook:
    """Per-world bookkeeping of a run: warm time, the spread then, hob on-time, boiling, turnover."""

    def __init__(self, sim: Any, plug: ParticleThermal, spoons: np.ndarray, worlds: int) -> None:
        self.sim, self.plug, self.spoons, self.worlds = sim, plug, spoons, worlds
        self.n = sim.model.particle_count // worlds
        self.mass = sim.model.particle_mass.numpy()[plug._gidx.numpy()].reshape(worlds, plug.P)
        self.lv = material(MATERIAL).latent_vapor
        self.hob = plug.inputs()["heater_values"].numpy()[:, 0].copy()
        self.on_s = np.zeros(worlds)
        self.t_warm, self.spread_warm = np.full(worlds, np.nan), np.full(worlds, np.nan)
        self.burnt = np.zeros(worlds)
        z0 = self.q()[..., 2]
        self.bottom = z0 < 2.0 * RADIUS                       # the start's bottom layer, per world
        self.reached = np.zeros_like(self.bottom)

    def q(self) -> np.ndarray:
        return self.sim.state.particle_q.numpy().reshape(self.worlds, self.n, 3)

    def each(self, k: int) -> None:
        _stir(self.sim, self.spoons, k)
        on = np.isnan(self.t_warm)
        self.on_s += on * CONTROL_DT
        self.burnt[:] = np.maximum(self.burnt, (self.plug.boiled.numpy() / (self.mass * self.lv)).max(axis=1))
        if k % CHECK != CHECK - 1:
            return
        q = self.q()
        self.reached |= q[..., 2] > 0.5 * DEPTH
        part = portions(q, self.plug.temperature.numpy())
        warm = on & (part.min(axis=1) >= WARM)
        if warm.any():                                        # the cook turns those hobs off
            self.t_warm[warm] = (k + 1) * CONTROL_DT
            self.spread_warm[warm] = (part.max(axis=1) - part.min(axis=1))[warm]
            self.plug.set_input("heater_values", np.where(np.isnan(self.t_warm), self.hob, 0.0)[:, None])


def run(worlds: int, device: str, seed: int = 0, with_plugin: bool = True, periods: int = PERIODS) -> dict:
    """Run the task in ``worlds`` worlds; the row (module doc)."""
    plug = plugin() if with_plugin else None
    sim = build_simulation(scene(worlds, periods, seed), [plug] if plug else [], device, worlds)
    spoons = np.array([i for i, lab in enumerate(sim.model.body_label) if lab.split("/")[-1] == "spoon"])
    _stir(sim, spoons, -1)
    if plug is None:
        ms, _ = run_periods(sim, periods, lambda k: _stir(sim, spoons, k))
        return {"task": "cook_porridge", "plugin": False, "worlds": worlds, "ms_per_step": ms, "success": 0}
    cook = _Cook(sim, plug, spoons, worlds)
    ms, steps = run_periods(sim, periods, cook.each)
    part = portions(cook.q(), plug.temperature.numpy())
    spoon_err = np.linalg.norm(sim.state.body_q.numpy()[spoons, :3] - spoon_centre(periods * CONTROL_DT), axis=1)
    read = plug.read()
    burnt_cm2 = ((read["vapor"] >= 1.0) * plug._face.reshape(worlds, plug.P)).sum(axis=1) * 1e4   # boiled dry: burnt on
    out = spilt(cook.q())
    ok, ga = verdict(np.isfinite(cook.t_warm) & (cook.spread_warm <= SPREAD), burnt_cm2, out)
    turnover = (cook.reached & cook.bottom).sum(axis=1) / cook.bottom.sum(axis=1)
    return {"task": "cook_porridge", "plugin": True, "worlds": worlds, "device": device, "seed": seed,
            "success": int(ok.sum()), "ga": ga, "burnt_cm2_max": float(burnt_cm2.max()),
            "spilt_max": float(out.max()), "warm_reached": int(np.isfinite(cook.t_warm).sum()),
            "t_warm_s_max": float(np.nanmax(cook.t_warm)) if np.isfinite(cook.t_warm).any() else None,
            "spread_at_warm_K_max": float(np.nanmax(cook.spread_warm)) if np.isfinite(cook.t_warm).any() else None,
            "coldest_portion_C_min": float(part.min() - 273.15), "portion_spread_K_max_end":
            float((part.max(axis=1) - part.min(axis=1)).max()), "burnt_max": float(cook.burnt.max()),
            "hottest_C_max": float(read["temperature"].max() - 273.15), "turnover_min": float(turnover.min()),
            "evaporated_in_solver_max": int(read["evaporated_in_solver"].max()),
            "heater_idle_J_max": float(read["heater_idle_J"].max()), "seconds": periods * CONTROL_DT,
            "hob_ratio": [float(x) for x in read["heat_in_J"] / (EFFICIENCY * cook.hob * cook.on_s)],
            "spoon_err_mm_max": float(spoon_err.max() * 1e3),
            "neighbor_overflow_max": int(read["neighbor_overflow"].max()), "ms_per_step": ms, "steps_timed": steps}


def main(argv: list[str] | None = None) -> int:
    return main_for(__doc__, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
