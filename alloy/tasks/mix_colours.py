# SPDX-License-Identifier: Apache-2.0
"""Mix two colours: red and blue dyed water side by side in a square cup, stirred until each dye is uniform
(species plug-in, alloy/plugins/species; Gym-Anything's mixing criterion, coordinator 2026-11-06).

Scene (public API): a static open box (inner ``cavity`` x ``cavity``), red liquid in its x < 0 half and blue in its
x > 0 half, ``depth`` deep, both water (DFSPH); a paddle stirrer (mix_stir.py) turning at ``rev_per_s``. The dyes
are species carried by the particles at ``conc`` units/m^3 each, with an effective diffusivity standing for mixing
below the particle spacing (plugin.py).

Pass test (Gym-Anything ingredient-mixing): each dye's CV over ``bins`` portions <= ``cv_max`` (0.10), spill <=
``spill_max`` (1 %), and every species conserved exactly. Reported over time: CV per dye, the particle-level CV
(intensity of segregation), spill, conservation. Portions see mixing only down to their own size: stirring draws the
dyes into streaks, and streaks finer than a portion read as mixed while every particle is still pure; hence 4 x 4 x 2
portions (12 mm in the default cup) and the particle CV beside them.

Example::

    from alloy.tasks.mix_colours import MixSpec, run
    out = run(MixSpec(seconds=6.0), device="cpu")
    print(out["success"], out["cv"][-1], out["t_mixed"])

Command line (``--cost``: wall time per control period with and without the plug-in, after a warm-up)::

    python -m alloy.tasks.mix_colours --device cuda:0 --worlds 64 --seconds 4 --out mix.json
    python -m alloy.tasks.mix_colours --device cuda:0 --cost 64,512,1024 --out mix_cost.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass

import numpy as np

from alloy.plugins.species import Species
from alloy.plugins.species.build import build as build_with
from alloy.tasks.mix_stir import HUB_Z, Cavity, particle_cv, portion_cv, spill_fraction, stirrer_mjcf

RGB = {"red": (0.85, 0.10, 0.10), "blue": (0.10, 0.20, 0.85)}


@dataclass(frozen=True)
class MixSpec:
    cavity: float = 0.048        # square cup inner side [m]
    depth: float = 0.024         # liquid depth [m]
    wall: float = 0.004          # [m]
    conc: float = 1000.0         # dye concentration [unit/m^3]
    diffusivity: float = 2.0e-5  # effective dye diffusivity [m^2/s]
    rev_per_s: float = 2.0       # stirrer speed
    paddle: float = 0.016        # blade half-width [m]: a multiple of 4 mm (off the lattice, see build_scene)
    blade: float = 0.0005        # wire radius [m]
    blade_z: float = 0.004       # blade height above the floor [m]: a multiple of 4 mm
    seconds: float = 8.0         # stirring time [s]
    sample: float = 0.25         # [s] between measurements
    bins: tuple = (4, 4, 2)      # portions (x, y, z)
    cv_max: float = 0.10
    spill_max: float = 0.01
    worlds: int = 1              # identical worlds (measures read world 0; conservation every world)


def build_scene(spec: MixSpec):
    """(alloy.Scene, Species plug-in, Cavity) of the task. API liquids start on a 4 mm lattice centred in their box
    (sites at odd multiples of 2 mm from the cup's centre for an even count): a wire paddle whose shaft, posts and
    blades lie on even multiples sits >= 2.8 mm from every site, so no particle starts inside it (one that does is
    thrown out of the cup at the first step)."""
    import alloy  # noqa: PLC0415

    L, w = spec.cavity, spec.wall
    s = alloy.Scene(worlds=spec.worlds, dt=0.01)
    s.add_ground()
    s.add_open_box("cup", inner=(L, L, 2.5 * spec.depth), wall=w, pos=(0.0, 0.0, 0.0))
    z = w + spec.depth / 2.0
    s.add_liquid("red", size=(L / 2, L, spec.depth), pos=(-L / 4, 0.0, z))
    s.add_liquid("blue", size=(L / 2, L, spec.depth), pos=(L / 4, 0.0, z))
    s.add_robot("stirrer", mjcf=stirrer_mjcf(spec.paddle, HUB_Z - spec.blade_z, spec.blade),
                pos=(0.0, 0.0, w + HUB_Z))
    c, big = spec.conc, 1.0
    sp = Species(species=("red", "blue"), diffusivity=(spec.diffusivity, spec.diffusivity),
                 regions=((-big, -big, -big, 0.0, big, big, c, 0.0), (0.0, -big, -big, big, big, big, 0.0, c)))
    return s, sp, Cavity((0.0, 0.0, w), (L, L, 2.5 * spec.depth))


def particle_colours(conc: np.ndarray) -> np.ndarray:
    """[N, 3] RGB of each particle: the dyes' colours weighted by their concentrations ([N, 2], red then blue)."""
    rgb = np.asarray([RGB["red"], RGB["blue"]])
    tot = conc.sum(axis=1, keepdims=True)
    return np.where(tot > 0.0, conc @ rgb / np.maximum(tot, 1e-30), 0.5)


def measure(sim, sp: Species, cavity: Cavity, bins: tuple, world: int = 0) -> dict:
    """CV per dye, spill fraction and the largest conservation error now (one world)."""
    r = sim.plugins.read()[sp.label]
    ids = sp.particles()[world]
    x = sim.engine.state.particle_q.numpy()[ids]
    vol = sp.vol_host[ids]
    amt, on = r["amount"][world].astype(np.float64), r["active"][world]
    return {"cv": portion_cv(x, amt, vol, on, cavity, bins), "particle_cv": particle_cv(x, amt, vol, on, cavity),
            "spill": spill_fraction(x, on, cavity),
            "conservation_error": int(np.abs(r["conservation_error"]).max())}


def run(spec: MixSpec = MixSpec(), device: str | None = None, log=None) -> dict:
    """Stir for ``spec.seconds``; per sample: t, CV per dye, spill, conservation error. ``success`` per the pass
    test at the end; ``t_mixed``: first time every CV <= cv_max (None if never)."""
    scene, sp, cav = build_scene(spec)
    sim = build_with(scene, [sp], device=device)
    rows, theta, t0 = [measure(sim, sp, cav, spec.bins) | {"t": 0.0}], 0.0, time.time()
    for k in range(int(round(spec.seconds / spec.sample))):
        nxt = theta + 2.0 * np.pi * spec.rev_per_s * spec.sample
        sim.run(spec.sample, targets={"spin": (theta, nxt)})
        theta = nxt
        rows.append(measure(sim, sp, cav, spec.bins) | {"t": (k + 1) * spec.sample})
        if log is not None:
            m = rows[-1]
            log(f"t={m['t']:.2f} cv {np.round(m['cv'], 3)} particle cv {np.round(m['particle_cv'], 3)} "
                f"spill {m['spill']:.4f} cons {m['conservation_error']} wall {time.time() - t0:.0f}s")
    cv = np.array([r["cv"] for r in rows])
    mixed = np.nonzero((cv <= spec.cv_max).all(axis=1))[0]
    last = rows[-1]
    ok = bool((last["cv"] <= spec.cv_max).all() and last["spill"] <= spec.spill_max
              and max(r["conservation_error"] for r in rows) == 0)
    return {"t": np.array([r["t"] for r in rows]), "cv": cv,
            "particle_cv": np.array([r["particle_cv"] for r in rows]), "spill": np.array([r["spill"] for r in rows]),
            "conservation_error": max(r["conservation_error"] for r in rows), "success": ok,
            "t_mixed": float(rows[mixed[0]]["t"]) if len(mixed) else None, "health_ok": bool(sim.check()["ok"]),
            "particles": int(sp.particles().shape[1]), "wall_s": time.time() - t0}


def cost(spec: MixSpec, device: str | None = None, periods: int = 50, species: bool = True, warm: int = 3) -> dict:
    """Wall seconds per control period while stirring, with or without the plug-in, after ``warm`` periods (the first
    period of a key runs eagerly and the second captures the CUDA graph: both stay outside the timed window)."""
    import alloy  # noqa: PLC0415

    scene, sp, _ = build_scene(spec)
    sim = build_with(scene, [sp], device=device) if species else alloy.build(scene, device=device)
    rad = 2.0 * np.pi * spec.rev_per_s * scene.dt
    sim.run(warm * scene.dt, targets={"spin": (0.0, rad * warm)})
    sim.state(["stirrer/spin"])
    t0 = time.perf_counter()
    sim.run(periods * scene.dt, targets={"spin": (rad * warm, rad * (warm + periods))})
    sim.state(["stirrer/spin"])
    return {"worlds": spec.worlds, "species": species, "s_per_period": (time.perf_counter() - t0) / periods,
            "particles_per_world": int(sp.particles().shape[1]) if species else None, "health_ok": sim.check()["ok"]}


def _jsonable(x):
    return x.tolist() if hasattr(x, "tolist") else x


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.tasks.mix_colours", description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=MixSpec.seconds)
    ap.add_argument("--rev-per-s", type=float, default=MixSpec.rev_per_s)
    ap.add_argument("--cost", default="", help="comma-separated world counts: time periods instead of the task")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = MixSpec(worlds=a.worlds, seconds=a.seconds, rev_per_s=a.rev_per_s)
    if a.cost:
        rows = [cost(dataclasses.replace(spec, worlds=int(w)), a.device, species=on)     # arms alternate first
                for k, w in enumerate(a.cost.split(",")) for on in ((True, False) if k % 2 else (False, True))]
    else:
        out = run(spec, a.device, log=print)
        rows = {k: _jsonable(v) for k, v in out.items()} | {"spec": dataclasses.asdict(spec)}
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1, default=_jsonable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
