# SPDX-License-Identifier: Apache-2.0
"""Titration colour change: base from a burette into acid with phenolphthalein in a stirred flask until the first faint
pink persists (species plug-in, alloy/plugins/species; Gym-Anything titration, run-in and finish-titration).

Chemistry (species/chem.py): a strong acid (HCl) and a strong base (NaOH) as species carried by the liquid's particles;
each particle's pH follows from its conserved totals and water's autoprotolysis; phenolphthalein (pKa 9.3) is pink at
pH >= 8.2 where it is present. The acid and indicator start in the flask; base enters only with the drops.

Burette. An API pour with a one-particle nozzle above the flask: each drop is one particle (0.064 mL at the API's 4 mm
spacing), one every ``rate``-set interval while the stopcock is open. The stopcock is the SPH emitter's on/off flag
(``Burette``). Delivered base = the species' exact inflow ledger / base concentration; the equivalence volume
V_eq = acid amount / base concentration, from the simulated acid.

Pass test (``verdict``): the flask's pink fraction (volume of its liquid at pH >= 8.2 with indicator) >= ``pink_min``
held ``hold`` s, the delivered base within ``tol_drops`` drops past V_eq, spill <= 1 %, every species conserved exactly.
``run`` is a scripted titrator: run in to ``run_in`` short of V_eq (the rough titre), then drop by drop: ``settle`` s
of stirring after each drop, then more stirring while there is pink and it is not fading (up to ``patience`` s); a
flash that fades means the acid still wins, so the next drop goes in. It stops when the pink holds.

Example::

    from alloy.tasks.titration_colour import TitrationSpec, run
    out = run(TitrationSpec(), device="cpu")
    print(out["verdict"])

Command line (``--hold 30``: Gym-Anything's persistence time)::

    python -m alloy.tasks.titration_colour --device cuda:0 --hold 30 --out titration.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass

import numpy as np

from alloy.plugins.species import Species, chem
from alloy.plugins.species.build import build as build_with
from alloy.plugins.species.kernels import AMT_SCALE_HOST
from alloy.tasks.mix_stir import Cavity, spill_fraction, stir_bar_mjcf

SPECIES = ("acid", "base", "phph")
EPS = 1.0e-6  # [s] hold time sums of float steps


@dataclass(frozen=True)
class TitrationSpec:
    flask: float = 0.032         # square flask inner side [m]
    depth: float = 0.016         # acid depth [m]
    wall: float = 0.004          # [m]
    rim: float = 0.040           # flask inner height [m]
    acid: float = 10.0           # HCl [mol/m^3] (10 mM)
    base: float = 100.0          # NaOH in the burette [mol/m^3] (0.1 M)
    indicator: float = 0.1       # phenolphthalein [mol/m^3]
    diffusivity: float = 1.0e-4  # effective, stirred [m^2/s]
    rate: float = 1.0e-7         # burette flow, stopcock open [m^3/s]: one 0.064 mL drop per 0.64 s
    burette: float = 3.0e-6      # base in the burette [m^3]
    nozzle: tuple = (0.008, 0.008, 0.010)   # burette tip: xy, height above the rim [m]
    rev_per_s: float = 3.0       # stir bar; 0 = no stir bar
    bar: float = 0.010           # stir bar half-length [m]
    run_in: float = 0.4e-6       # stop the run-in this short of V_eq [m^3]
    settle: float = 1.0          # stirring after each drop [s]
    patience: float = 3.0        # longest extra stirring while there is pink, not fading [s]
    hold: float = 3.0           # pink must persist this long [s] (Gym-Anything: 30 s)
    pink_min: float = 0.9        # pink fraction of the flask's liquid
    tol_drops: float = 1.5       # allowed overshoot past V_eq [drops]
    spill_max: float = 0.01
    step: float = 0.05           # control granularity [s]
    max_seconds: float = 60.0


def build_scene(spec: TitrationSpec):
    """(alloy.Scene, Species plug-in, Cavity, nozzle position) of the task."""
    import alloy  # noqa: PLC0415

    L, w = spec.flask, spec.wall
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_open_box("flask", inner=(L, L, spec.rim), wall=w, pos=(0.0, 0.0, 0.0))
    s.add_liquid("acid", size=(L, L, spec.depth), pos=(0.0, 0.0, w + spec.depth / 2.0))
    nozzle = (spec.nozzle[0], spec.nozzle[1], w + spec.rim + spec.nozzle[2])
    s.add_pour("base", spec.burette, pos=nozzle, rate=spec.rate, speed=0.3)
    if spec.rev_per_s > 0.0:
        s.add_robot("stirrer", mjcf=stir_bar_mjcf(spec.bar), pos=(0.0, 0.0, w + 0.004))
    sp = Species(species=SPECIES, diffusivity=(spec.diffusivity,) * 3, inflow=(0.0, spec.base, 0.0),
                 regions=((-L / 2, -L / 2, w - 1e-3, L / 2, L / 2, w + spec.depth + 1e-3, spec.acid, 0.0,
                           spec.indicator),),
                 acid="acid", base="base", indicator="phph")
    return s, sp, Cavity((0.0, 0.0, w), (L, L, spec.rim)), nozzle


class Burette:
    """The pour at ``nozzle``: its SPH emitters (one per world) with their on/off flag as the stopcock
    (``SolverDFSPH.set_emitter_enabled``: a closed emitter emits nothing and its clock pauses)."""

    def __init__(self, sim, nozzle: tuple) -> None:
        solver = next((s for s in sim.engine.solvers if hasattr(s, "set_emitter_enabled")), None)
        if solver is None:
            raise RuntimeError("no SPH solver in the simulation: the burette needs a pour")
        self.solver, self.emitters = solver, solver.emitters
        pos = np.asarray([x[:3] for x in self.emitters.table.xform], float).reshape(-1, 3)
        ids = np.nonzero(np.abs(pos - np.asarray(nozzle)).max(axis=1) < 1e-6)[0]
        world = np.asarray(self.emitters.table.world)[ids]
        if sorted(world.tolist()) != list(range(sim.worlds)):
            raise RuntimeError(f"expected one emitter per world at {nozzle}, found worlds {world.tolist()}")
        self.ids = ids[np.argsort(world)]

    def set(self, open_: bool | np.ndarray) -> None:
        """Open (True) or close the stopcock: one value for every world or one per world."""
        for e, on in zip(self.ids, np.broadcast_to(np.asarray(open_, bool), self.ids.shape), strict=True):
            self.solver.set_emitter_enabled(int(e), bool(on))

    def is_open(self) -> np.ndarray:
        return self.emitters.dev.enabled.numpy()[self.ids] != 0


class Titration:
    """The task's simulation: ``advance`` stirs, ``burette`` opens and closes, ``observe`` reads the flask."""

    def __init__(self, spec: TitrationSpec = TitrationSpec(), device: str | None = None) -> None:
        self.spec = spec
        scene, self.species, self.cavity, nozzle = build_scene(spec)
        self.sim = build_with(scene, [self.species], device=device)
        self.burette = Burette(self.sim, nozzle)
        self.burette.set(False)
        self.theta = 0.0
        r = self.read()
        acid = SPECIES.index("acid")
        self.v_eq = (self.species.total0[:, acid] / AMT_SCALE_HOST) / spec.base     # [W] m^3
        self.drop = float(self.species.vol_host[self.species.particles()[0][~(r["active"][0] != 0)][0]])

    def read(self) -> dict:
        return self.sim.plugins.read()[self.species.label]

    def advance(self, seconds: float) -> None:
        """Run ``seconds`` with the stir bar turning at ``rev_per_s``."""
        if self.spec.rev_per_s <= 0.0:
            self.sim.run(seconds)
            return
        nxt = self.theta + 2.0 * np.pi * self.spec.rev_per_s * seconds
        self.sim.run(seconds, targets={"spin": (self.theta, nxt)})
        self.theta = nxt

    def delivered(self) -> np.ndarray:
        """Base delivered per world [m^3] (exact: the inflow ledger)."""
        return self.read()["ledger_in"][:, SPECIES.index("base")] / AMT_SCALE_HOST / self.spec.base

    def observe(self, world: int = 0) -> dict:
        """t, delivered base [mL], stir bar turns, pink fraction of the flask's liquid, bulk pH (fully mixed), spill,
        conservation error."""
        r = self.read()
        ids = self.species.particles()[world]
        x = self.sim.engine.state.particle_q.numpy()[ids]
        on = (r["active"][world] != 0) & self.cavity.inside(x)
        vol = self.species.vol_host[ids].astype(np.float64)
        pink = on & (r["ph"][world] >= chem.PINK_PH) & (r["conc"][world][:, SPECIES.index("phph")] > 0.0)
        tot = r["amount"][world][on].sum(axis=0).astype(np.float64) / AMT_SCALE_HOST / vol[on].sum() / 1000.0
        stir = float(self.sim.state(["stirrer/spin"], world=world)["stirrer/spin.q"]) if self.spec.rev_per_s else 0.0
        return {"t": float(self.sim.t), "delivered_ml": float(self.delivered()[world] * 1e6), "stir_rev": stir / 6.2832,
                "pink": float(vol[pink].sum() / max(vol[on].sum(), 1e-30)),
                "bulk_ph": float(chem.ph(tot[0], tot[1])), "spill": spill_fraction(x, r["active"][world], self.cavity),
                "conservation_error": int(np.abs(r["conservation_error"]).max())}


def _one_drop(t: Titration, rows: list) -> None:
    """Open the stopcock until one more drop has left the burette, then close it."""
    n0 = t.delivered()[0]
    t.burette.set(True)
    while t.delivered()[0] <= n0 + 0.5 * t.drop and t.sim.t < t.spec.max_seconds:
        t.advance(t.spec.step)
    t.burette.set(False)
    rows.append(t.observe())


def verdict(spec: TitrationSpec, rows: list, held: float, v_eq: float, drop: float) -> dict:
    """The pass test on a finished run (module doc)."""
    last = rows[-1]
    over = (last["delivered_ml"] * 1e-6 - v_eq) / drop
    return {"pink_held_s": held, "over_drops": over, "v_eq_ml": v_eq * 1e6, "v_end_ml": last["delivered_ml"],
            "spill": max(r["spill"] for r in rows), "conservation_error": max(r["conservation_error"] for r in rows),
            "success": bool(held >= spec.hold - EPS and -0.5 <= over <= spec.tol_drops
                            and max(r["spill"] for r in rows) <= spec.spill_max
                            and max(r["conservation_error"] for r in rows) == 0)}


def run(spec: TitrationSpec = TitrationSpec(), device: str | None = None, log=None) -> dict:
    """Scripted titration of world 0 (module doc): rows of ``observe`` and the ``verdict``."""
    t = Titration(spec, device)
    say = log or (lambda _m: None)
    v_eq, rows = float(t.v_eq[0]), [t.observe()]
    say(f"V_eq {v_eq * 1e6:.3f} mL, drop {t.drop * 1e6:.4f} mL")
    if v_eq - spec.run_in > 0.0:
        t.burette.set(True)
        while t.delivered()[0] < v_eq - spec.run_in and t.sim.t < spec.max_seconds:
            t.advance(spec.step)
        t.burette.set(False)
        rows.append(t.observe())
        say(f"run-in done: {rows[-1]}")
    held = _titrate(t, rows, say)
    out = {"rows": rows, "verdict": verdict(spec, rows, held, v_eq, t.drop), "health_ok": bool(t.sim.check()["ok"])}
    say(f"verdict {out['verdict']}")
    return out


def _titrate(t: Titration, rows: list, say) -> float:
    """Drop by drop until the pink holds (module doc); returns the seconds it held."""
    spec = t.spec
    held, waited, last = 0.0, 0.0, 0.0
    while t.sim.t < spec.max_seconds:
        rows.append(t.observe())
        pink = rows[-1]["pink"]
        if pink >= spec.pink_min:
            if held >= spec.hold - EPS:
                break
            held += spec.step
        elif 0.0 < pink >= last and waited < spec.patience - EPS:    # pink, not fading: wait
            held, waited = 0.0, waited + spec.step
        else:
            held, waited = 0.0, 0.0
            _one_drop(t, rows)
            say(f"drop: {rows[-1]}")
            last = rows[-1]["pink"]
            t.advance(spec.settle)
            continue
        last = pink
        t.advance(spec.step)
    return held


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.tasks.titration_colour", description=__doc__.split("\n")[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--hold", type=float, default=TitrationSpec.hold)
    ap.add_argument("--max-seconds", type=float, default=TitrationSpec.max_seconds)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    spec = TitrationSpec(hold=a.hold, max_seconds=max(a.max_seconds, 30.0 + 2.0 * a.hold))
    out = run(spec, a.device, log=print) | {"spec": dataclasses.asdict(spec)}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
