# SPDX-License-Identifier: Apache-2.0
"""cook_sauce (heat v2): cook a starch sauce on a hob to the boil while a spoon turns it over (Gym-Anything stir-sauce).

About 600 mL of cold milk with corn starch and sugar stirred in (SPH liquid, ``phase_materials`` ``starch_sauce``):
heat v2's doneness kinetics gelatinise the starch (calibrated so a 5 K/min ramp is 18 % cooked at 60 °C and 95 % at
77 °C), and ``ParticleThermal`` writes the thickening viscosity μ(T, X) into ``sph:viscosity`` each exchange: about
3 mPa·s raw at 20 °C to about 1 Pa·s cooked at 96 °C. It fills a 126 x 120 mm pan 36 mm deep on a hob: the floor is a
power heater, the hob's power x ``EFFICIENCY`` shared by the particles that touch it. ``cook_porridge``'s fold-loop
spoon (``cook_common.FoldLoop``) turns the sauce over, here with a blade 110 mm across.

The hob runs at its top setting: the reference task's own 1000 W custard took 238 s to reach 93 °C (Gym-Anything
ingredient-mixing report §11.6), past its 180 s hob allowance. The cook turns the hob off once the sauce is at the boil
(mass-weighted mean >= 96 °C) and at least 95 % of its starch is cooked, or when the hob has been on for 180 s.

**Verdict against Gym-Anything's stir-sauce** (report §10; ``ga_verdict``, row key ``"ga"``). The scene is not
Gym-Anything's (``SCENE``): their batch, but in a rectangular pan, with a scripted spoon in place of their arm and
knob, and the hob at its top setting. Of their 6 criteria the port judges 4 from the simulation state, and success
per world is those 4 and the port's own extra (the sauce reaches the boil, mean >= 96 °C; theirs sets no minimum):
- cooked: at least 95 % of the starch cooked (mass mean of the doneness) and the sauce's mean reached 88 °C or more;
- not_scorched: burnt on at most 2 cm² (particles that boiled off all their water, as footprints on the floor);
- contained: at most 1 % of the batch outside the pan;
- not_boiled_on: at most 180 s at the boil while the hob heats (the hob's power input read from the plug-in).
Not judged (None in the verdict, never counted): vessel_in_place (the pan is a static body: it cannot leave the hob)
and appliance_off (the scripted cook switches the hob off itself; the scene has no knob).
Score per world: 1 on success, else 0.9 x progress x the share in the pan x the share of the floor not burnt on
x 0.75. That is Gym-Anything's score without its closing credits (+ 0.10 with the pan on the hob, + 0.15 with the hob
off over a cooked sauce), which belong to the two criteria not judged here; progress is the gelatinised share of the
least cooked of three equal-mass layers by height over 0.95 (<= 1).

Per world (seeded): start U(4, 20) °C (from the fridge or the counter), hob U(1700, 1800) W.

    python -m alloy.tasks.cook_sauce --worlds 32 --device cuda:0 --out rows.jsonl
writes one row: success count, the mean score, time to the boil, hob-on time, cooked share, peak mean, burnt area,
spill, the turnover, and the plug-in's cost per step against the SPH step alone.
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

MATERIAL = "starch_sauce"
RADIUS = 3.0e-3                      # m, SPH particle radius (spacing 6 mm)
COUNTS = (21, 20, 6)                 # 2,520 particles = 590 g (Gym-Anything's 500 g milk + 30 g starch + 60 g sugar)
INNER = (0.126, 0.12)                # m, pan inside (21 x 20 particles across)
WALL, POT_H = 0.006, 0.08            # m
DEPTH = COUNTS[2] * 2.0 * RADIUS     # m (36 mm)
EFFICIENCY = 0.84                    # hob to food (as cook_porridge, Gym-Anything's reheat figure)
HOB = (1700.0, 1800.0)               # W, per-world range (the hob's top setting is 1800 W)
T_START = (277.15, 293.15)           # K
BLADE_HALF = (0.003, 0.055, 0.010)   # m: 6 mm thick, 110 mm across the 120 mm pan, 20 mm tall
CLEAR, DESCENT_CLEAR = 7.0e-3, 9.0e-3  # m, blade to the walls it sweeps to / comes down along (> a particle across)
SPEED = 0.20                         # m/s along the loop (1.85 s a loop; 0.12 m/s lost twice the heat to steam)
SPOON = FoldLoop(INNER[0], BLADE_HALF, DEPTH + 5.0e-3 + BLADE_HALF[2], CLEAR, DESCENT_CLEAR, speed=SPEED)
COOKED, PEAK, BOIL = 0.95, 361.15, 369.15       # doneness, K (88 °C), K (96 °C)
HOB_MAX_S = 180.0                    # s, the hob's allowance
BURNT_CM2, SPILT = 2.0, 0.01
GA_TASK = "ingredient_mixing/stir-sauce"
SCENE = ("not Gym-Anything's: their 590 g batch in a rectangular 126 x 120 mm pan 36 mm deep (theirs: a round "
         "saucepan); hob U(1700, 1800) W, its top setting; start U(4, 20) C; a kinematic spoon on a scripted loop "
         "(theirs: an arm with tools and a knob)")
NOT_JUDGED = {"vessel_in_place": "the pan is a static body here: it cannot leave the hob or tip",
              "appliance_off": "the scripted cook switches the hob off itself; the scene has no knob"}
CONTROL_DT, DT = 0.02, 1.25e-3       # s
SECONDS = 180.0
PERIODS = round(SECONDS / CONTROL_DT)
CHECK = 25                           # periods between layer / boil checks (0.5 s)
AFTER_S = 2.0                        # s run on once every world's hob is off


def _wall(b: newton.ModelBuilder, centre: tuple[float, float, float], half: tuple[float, float, float], label: str,
          body: int = -1) -> None:
    from alloy.solvers.sph.config import BOUNDARY_SDF  # noqa: PLC0415 - the SPH package registers on import

    b.add_shape_box(body, xform=wp.transform(wp.vec3(*centre), wp.quat_identity()), hx=half[0], hy=half[1],
                    hz=half[2], custom_attributes={"sph:shape_boundary": BOUNDARY_SDF}, label=label)


def build(b: newton.ModelBuilder) -> None:
    """One world: the pan (floor labelled "pan_floor", the hob), the sauce and the spoon."""
    from alloy.solvers.sph.build import add_fluid_block, register_custom_attributes  # noqa: PLC0415
    from alloy.solvers.sph.config import DFSPHConfig  # noqa: PLC0415

    register_custom_attributes(b)
    hx, hy, t = 0.5 * INNER[0], 0.5 * INNER[1], 0.5 * WALL
    _wall(b, (0.0, 0.0, -t), (hx + WALL, hy + WALL, t), "pan_floor")
    for s in (-1.0, 1.0):
        _wall(b, (s * (hx + t), 0.0, 0.5 * POT_H), (t, hy + WALL, 0.5 * POT_H), f"pan_x{int(s)}")
        _wall(b, (0.0, s * (hy + t), 0.5 * POT_H), (hx, t, 0.5 * POT_H), f"pan_y{int(s)}")
    mat = material(MATERIAL)
    add_fluid_block(b, DFSPHConfig(particle_radius=RADIUS), (-hx, -hy, 0.0), COUNTS, rest_density=mat.density,
                    viscosity=float(mat.viscosity(T_START[0], 1.0, 0.0)))
    spoon = b.add_body(xform=wp.transform(wp.vec3(*SPOON.centre(0.0)), wp.quat_identity()), mass=0.05,
                       inertia=wp.mat33(np.eye(3) * 1.0e-5), label="spoon")
    _wall(b, (0.0, 0.0, 0.0), BLADE_HALF, "spoon_blade", body=spoon)


def plugin() -> ParticleThermal:
    return ParticleThermal(materials=(MATERIAL,), initial=(T_START[0],), heaters=("pan_floor",),
                           heater_modes=("power",), heater_values=(HOB[0],), heater_efficiency=EFFICIENCY)


def scene(worlds: int, periods: int, seed: int) -> TestScene:
    """The task scene; its init draws each world's start temperature and hob power. The SPH entry solves viscosity
    implicitly (weiler2018): from cold milk's 3 mPa·s its 'auto' mode would pick xsph, which ignores per-particle μ."""
    def init(sim: Any) -> None:
        therm = thermal_of(sim)
        if therm is not None:
            therm.set_temperature(draws(seed, worlds, *T_START, stream=0)[:, None] * np.ones((1, therm.P)))
            therm.set_input("heater_values", draws(seed, worlds, *HOB, stream=1)[:, None])
    return TestScene(build=build, worlds=worlds, periods=periods, control_dt=CONTROL_DT, dt=DT, solver="sph",
                     solver_kwargs={"particle_radius": RADIUS, "viscosity": "weiler2018"}, ground=False, init=init)


def layers(q: np.ndarray, value: np.ndarray, parts: int = 3) -> np.ndarray:
    """[W, parts] mean of ``value`` over equal-count layers by height (particles of one material: equal mass)."""
    order = np.argsort(q[..., 2], axis=1, kind="stable")
    return np.stack([np.take_along_axis(value, idx, axis=1).mean(axis=1)
                     for idx in np.array_split(order, parts, axis=1)], axis=1)


CLOSING = 0.75      # Gym-Anything's closing factor without its two credits for criteria not judged here (module doc)


def score(ok: np.ndarray, least: np.ndarray, in_pan: np.ndarray, unburnt: np.ndarray) -> np.ndarray:
    """The port's score: 1 on success, else Gym-Anything's product without the unjudged credits (module doc)."""
    return np.where(ok, 1.0, 0.9 * np.minimum(least / COOKED, 1.0) * in_pan * unburnt * CLOSING)


def hob_off(on: np.ndarray, boiled: np.ndarray, cooked: np.ndarray, on_s: np.ndarray) -> np.ndarray:
    """The worlds whose hob the cook turns off now: on, and at the boil with the starch cooked, or out of hob time."""
    return on & ((boiled & (cooked >= COOKED)) | (on_s >= HOB_MAX_S - 1e-9))


class _Cook:
    """Per-world bookkeeping: hob on-time, peak mean, the boil, burning, spill; turns each hob off when done."""

    def __init__(self, sim: Any, plug: ParticleThermal, spoons: np.ndarray, worlds: int) -> None:
        self.sim, self.plug, self.spoons, self.worlds = sim, plug, spoons, worlds
        self.n = sim.model.particle_count // worlds
        self.mass = sim.model.particle_mass.numpy()[plug._gidx.numpy()].reshape(worlds, plug.P)
        self.hob = plug.inputs()["heater_values"].numpy()[:, 0].copy()
        self.on = np.ones(worlds, bool)
        self.on_s, self.peak = np.zeros(worlds), np.full(worlds, -np.inf)
        self.t_boil, self.t_off = np.full(worlds, np.nan), np.full(worlds, np.nan)
        self.boil_on_s = np.zeros(worlds)                      # s at the boil while the hob heats (not_boiled_on)
        self.bottom = self.q()[..., 2] < 2.0 * RADIUS
        self.reached = np.zeros_like(self.bottom)
        self.t = 0.0

    def q(self) -> np.ndarray:
        return self.sim.state.particle_q.numpy().reshape(self.worlds, self.n, 3)

    def each(self, k: int) -> bool:
        SPOON.stir(self.sim, self.spoons, k, CONTROL_DT)
        self.on_s += self.on * CONTROL_DT
        temp = self.plug.temperature.numpy()
        mean = (temp * self.mass).sum(axis=1) / self.mass.sum(axis=1)
        self.peak = np.maximum(self.peak, mean)
        t = self.t = (k + 1) * CONTROL_DT
        heating = self.plug.inputs()["heater_values"].numpy()[:, 0] > 0.0     # the hob's power input, not a flag
        self.boil_on_s += ((mean >= BOIL) & heating) * CONTROL_DT
        self.t_boil[np.isnan(self.t_boil) & self.on & (mean >= BOIL)] = t
        off = hob_off(self.on, np.isfinite(self.t_boil), self.plug.done.numpy().mean(axis=1), self.on_s)
        if off.any():                                         # the cook turns those hobs to 0
            self.on &= ~off
            self.t_off[off] = t
            self.plug.set_input("heater_values", np.where(self.on, self.hob, 0.0)[:, None])
        if k % CHECK == CHECK - 1:
            self.reached |= self.q()[..., 2] > 0.5 * DEPTH
        return bool(not self.on.any() and t >= np.nanmax(self.t_off) + AFTER_S)


def run(worlds: int, device: str, seed: int = 0, with_plugin: bool = True, periods: int = PERIODS) -> dict:
    """Run the task in ``worlds`` worlds; the row (module doc)."""
    plug = plugin() if with_plugin else None
    sim = build_simulation(scene(worlds, periods, seed), [plug] if plug else [], device, worlds)
    spoons = np.array([i for i, lab in enumerate(sim.model.body_label) if lab.split("/")[-1] == "spoon"])
    SPOON.stir(sim, spoons, -1, CONTROL_DT)
    if plug is None:
        ms, _ = run_periods(sim, periods, lambda k: SPOON.stir(sim, spoons, k, CONTROL_DT))
        return {"task": "cook_sauce", "plugin": False, "worlds": worlds, "ms_per_step": ms, "success": 0}
    cook = _Cook(sim, plug, spoons, worlds)
    ms, steps = run_periods(sim, periods, cook.each)
    return _row(cook, plug, worlds, device, seed, ms, steps)


def _row(cook: _Cook, plug: ParticleThermal, worlds: int, device: str, seed: int, ms: float, steps: int) -> dict:
    q, read = cook.q(), plug.read()
    done, face = read["done"], plug._face.reshape(worlds, plug.P)
    burnt_cm2 = ((read["vapor"] >= 1.0) * face).sum(axis=1) * 1e4     # boiled dry: burnt on, as footprints
    inside = np.all(np.abs(q[..., :2]) <= 0.5 * np.asarray(INNER) + RADIUS, axis=2) & (q[..., 2] > -RADIUS)
    in_pan = inside.mean(axis=1)
    cooked, least = done.mean(axis=1), layers(q, done).min(axis=1)
    boiled = np.isfinite(cook.t_boil)
    judged, ga = ga_verdict(GA_TASK, {
        "cooked": (cooked >= COOKED) & (cook.peak >= PEAK), "not_scorched": burnt_cm2 <= BURNT_CM2,
        "contained": 1.0 - in_pan <= SPILT, "not_boiled_on": cook.boil_on_s <= HOB_MAX_S + 1e-9,
        "vessel_in_place": None, "appliance_off": None}, NOT_JUDGED, SCENE,
        also="the sauce reaches the boil (mean >= 96 C; Gym-Anything sets no minimum)")
    ok = judged & boiled
    unburnt = np.clip(1.0 - burnt_cm2 * 1e-4 / (INNER[0] * INNER[1]), 0.0, 1.0)
    sc = score(ok, least, in_pan, unburnt)
    turnover = (cook.reached & cook.bottom).sum(axis=1) / cook.bottom.sum(axis=1)
    mu = cook.sim.model.sph.viscosity.numpy()[plug._gidx.numpy()]
    return {"task": "cook_sauce", "plugin": True, "worlds": worlds, "device": device, "seed": seed,
            "success": int(ok.sum()), "ga": ga, "score_mean": float(sc.mean()), "score_min": float(sc.min()),
            "boiled": int(boiled.sum()), "boil_on_s_max": float(cook.boil_on_s.max()),
            "t_boil_s_max": float(np.nanmax(cook.t_boil)) if boiled.any() else None,
            "hob_on_s_max": float(cook.on_s.max()), "hob_on_at_end": int(cook.on.sum()),
            "cooked_min": float(cooked.min()), "least_layer_cooked_min": float(least.min()),
            "peak_mean_C_min": float(cook.peak.min() - 273.15), "burnt_cm2_max": float(burnt_cm2.max()),
            "spilt_max": float((1.0 - in_pan).max()), "turnover_min": float(turnover.min()),
            "hottest_C_max": float(read["temperature"].max() - 273.15),
            "boiled_kJ_max": float(read["boiled_J"].max() * 1e-3), "most_boiled": float(read["vapor"].max()),
            "viscosity_Pa_s": [float(mu.min()), float(mu.max())],
            "evaporated_in_solver_max": int(read["evaporated_in_solver"].max()),
            "heater_idle_J_max": float(read["heater_idle_J"].max()), "seconds": cook.t,
            "hob_ratio": [float(x) for x in read["heat_in_J"] / (EFFICIENCY * cook.hob * cook.on_s)],
            "neighbor_overflow_max": int(read["neighbor_overflow"].max()), "ms_per_step": ms, "steps_timed": steps}


def main(argv: list[str] | None = None) -> int:
    return main_for(__doc__, run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
