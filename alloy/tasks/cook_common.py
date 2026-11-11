# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the heat-v2 tasks (design note docs/notes/heat-v2-design.md §10, §18, §22-23): MPM particle blocks
with a material's start values, per-world draws, the stirring spoon's fold loop (``FoldLoop``), the timed run loop and
the CLI row.

MPM entries get the router's voxel (``mpm_kwargs``): min(d_feature/4, 5 mm), with 2 particles per voxel per axis as
Alloy's emitters place them. Without it Newton's default voxel is 0.1 m, which put a whole 30 mm block in one cell: it
barely slumped at 1 world and fell through the plate with a dense grid (WP12, 2026-11-06). On the CPU the entry also
asks for Newton's dense grid at any world count: Alloy's factory picks it only from 2 worlds, and the sparse grid at
1 world differed by about 10 % (WP12 is making dense the CPU default).

Every task module exposes ``run(worlds, device, seed, with_plugin=True, ...) -> dict``; ``main_for`` runs it with the
``ParticleThermal`` plug-in and once without (a short run of the same scene), and writes one JSON row with the success
count, the task's numbers and the plug-in's cost against the solver step alone.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.thermal.phase_materials import PhaseMaterial

LABEL = "particle_thermal"
BASELINE_PERIODS = 10  # control periods of the plug-in-free timing run (period 0 compiles and is not timed)


@dataclass(frozen=True)
class FoldLoop:
    """A flat blade's scripted fold loop across a box pot (``cook_porridge`` doc): down ``descent_clear`` off the -x
    wall, along the floor ``floor_clear`` above it to ``clear`` off the +x wall, up that wall to ``z_high``, then the
    same mirrored, at constant ``speed``. The blade never comes closer to a wall than a particle's width."""

    inner_x: float           # m, the pot's inside along x
    blade_half: tuple[float, float, float]  # m
    z_high: float            # m, blade centre on the returns (clear of the food)
    clear: float = 6.0e-3    # m, blade to the wall it sweeps to and rises along
    descent_clear: float = 8.0e-3  # m, blade to the wall it comes down along
    floor_clear: float = 1.5e-3    # m, blade to the floor on a floor stroke
    speed: float = 0.12      # m/s
    corners: np.ndarray = field(init=False, repr=False)
    legs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x_end = 0.5 * self.inner_x - self.blade_half[0] - self.clear
        x_down = 0.5 * self.inner_x - self.blade_half[0] - self.descent_clear
        z_low = self.floor_clear + self.blade_half[2]
        half = [[-x_down, 0.0, self.z_high], [-x_down, 0.0, z_low], [x_end, 0.0, z_low], [x_end, 0.0, self.z_high]]
        corners = np.array(half + [[-x, y, z] for x, y, z in half] + [half[0]])
        object.__setattr__(self, "corners", corners)
        object.__setattr__(self, "legs", np.linalg.norm(np.diff(corners, axis=0), axis=1))

    @property
    def length(self) -> float:
        return float(self.legs.sum())

    def centre(self, t: float) -> np.ndarray:
        """The blade's centre at time t [s]."""
        s = (t * self.speed) % self.length
        for leg, length in enumerate(self.legs):
            if s <= length or leg == len(self.legs) - 1:
                return self.corners[leg] + (self.corners[leg + 1] - self.corners[leg]) * min(s / length, 1.0)
            s -= length
        raise AssertionError("unreachable")

    def stir(self, sim: Any, spoons: np.ndarray, k: int, control_dt: float) -> None:
        """Set the spoons' velocity for period k + 1: the chord of the loop over that period."""
        t = (k + 1) * control_dt
        v = (self.centre(t + control_dt) - self.centre(t)) / control_dt
        qd = sim.state.body_qd.numpy()
        qd[spoons] = [v[0], v[1], v[2], 0.0, 0.0, 0.0]
        sim.state.body_qd.assign(qd)


def mpm_values(mat: PhaseMaterial, t: float) -> dict[str, float]:
    """``mat``'s MPM material values at temperature ``t`` as custom attributes ({} for a material that does not
    flow). The plug-in writes the same values at setup, so the runs with and without it start alike."""
    if not mat.flows:
        return {}
    phi = float(mat.temperature(mat.enthalpy(t))[1])
    out = {"mpm:viscosity": float(mat.viscosity(t, phi)), "mpm:yield_stress": float(mat.yield_stress(phi)),
           "mpm:friction": float(mat.friction(phi))}
    if mat.young_solid > 0.0:
        out["mpm:young_modulus"] = float(mat.young(phi))
    return out


def add_block(b: newton.ModelBuilder, lo: tuple[float, float, float], counts: tuple[int, int, int], spacing: float,
              mat: PhaseMaterial, t: float, attrs: dict[str, float] | None = None) -> None:
    """MPM particles of ``mat`` at temperature ``t`` on a lattice of ``counts`` cells (side ``spacing``) with its
    lower corner at ``lo``; ``attrs`` (MPM custom attributes) for a material that does not flow."""
    newton.solvers.SolverImplicitMPM.register_custom_attributes(b)
    mass, attrs = mat.density * spacing ** 3, mpm_values(mat, t) or attrs or None
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                pos = wp.vec3(lo[0] + spacing * (i + 0.5), lo[1] + spacing * (j + 0.5), lo[2] + spacing * (k + 0.5))
                b.add_particle(pos, wp.vec3(0.0, 0.0, 0.0), mass, 0.5 * spacing, custom_attributes=attrs)


def mpm_kwargs(voxel: float, device: str) -> dict[str, Any]:
    """The MPM entry's kwargs: the router's ``voxel`` [m] and, on the CPU, the dense grid at any world count."""
    kw: dict[str, Any] = {"voxel_size": float(voxel)}
    if not wp.get_device(device).is_cuda:
        kw["grid_type"] = "dense"
    return kw


def thermal_of(sim: Any) -> Any:
    """The simulation's ``ParticleThermal`` plug-in, or None (the plug-in-free timing run)."""
    host = getattr(sim, "plugins", None)
    return host.plugins.get(LABEL) if host is not None else None


def draws(seed: int, worlds: int, lo: float, hi: float, stream: int) -> np.ndarray:
    """``worlds`` uniform draws in [lo, hi], reproducible per (seed, stream)."""
    return np.random.default_rng([seed, stream]).uniform(lo, hi, worlds)


def run_periods(sim: Any, periods: int, each: Callable[[int], bool | None] | None = None) -> tuple[float, int]:
    """Step ``periods`` control periods, calling ``each(k)`` after period k; a ``True`` from it ends the run early.
    Returns (wall ms per exchange, exchanges timed); period 0 compiles kernels and is not timed."""
    wall, steps = 0.0, 0
    for k in range(periods):
        t0 = time.perf_counter()
        sim.step_period()
        wp.synchronize_device(sim.model.device)
        if k >= 1:
            wall += time.perf_counter() - t0
            steps += 2 ** int(sim.clock.j)
        if each is not None and each(k):
            break
    return 1e3 * wall / max(steps, 1), steps


def ga_verdict(task: str, criteria: Mapping[str, Any], not_judged: Mapping[str, str], scene: str,
               also: str = "") -> tuple[np.ndarray, dict]:
    """A port's verdict against Gym-Anything's criteria (audit AB-gym-4, coordinator rule 2026-11-09). ``criteria``
    names every GA criterion of ``task``: a per-world bool array where the port judges it from the simulation state,
    None where it does not (``not_judged`` then gives the reason: a pass by construction or the script's own action
    is not a judgement). Returns (success per world: the judged criteria only, the JSON verdict). The verdict says
    how many of GA's criteria were judged, how the ``scene`` differs from GA's, and what the port asks for ``also``.
    Example: ``ga_verdict("x/y", {"a": np.array([True]), "b": None}, {"b": "static"}, "smaller")[1]["judged"]`` is
    ``"1 of 2"``."""
    judged = [k for k, v in criteria.items() if v is not None]
    if set(not_judged) != set(criteria) - set(judged):
        raise ValueError(f"not_judged {sorted(not_judged)} must name exactly the criteria given as None")
    ok = np.all([np.asarray(criteria[k], bool) for k in judged], axis=0)
    names = ", ".join(judged)
    summary = (f"{len(judged)} of Gym-Anything's {len(criteria)} criteria judged ({names}); not judged: "
               + "; ".join(f"{k} ({why})" for k, why in not_judged.items()) + f". Scene: {scene}"
               + (f". The port also requires: {also}" if also else ""))
    return ok, {"task": task, "judged": f"{len(judged)} of {len(criteria)}", "judged_criteria": judged,
                "criteria": {k: None if v is None else [bool(x) for x in np.asarray(v)] for k, v in criteria.items()},
                "not_judged": dict(not_judged), "scene": scene, "also": also, "summary": summary}


def main_for(doc: str, run: Callable[..., dict], argv: list[str] | None = None) -> int:
    """The task CLI: ``--worlds --device --seed --out``; exit 0 when every world succeeds."""
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="append the JSON row to this file")
    a = ap.parse_args(argv)
    on = run(a.worlds, a.device, a.seed)
    off = run(a.worlds, a.device, a.seed, with_plugin=False, periods=BASELINE_PERIODS)
    row = {**on, "ms_per_step_without": off["ms_per_step"],
           "cost_ratio": on["ms_per_step"] / max(off["ms_per_step"], 1e-9)}
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0 if row["success"] == a.worlds else 1
