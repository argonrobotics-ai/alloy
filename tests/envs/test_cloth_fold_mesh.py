# SPDX-License-Identifier: Apache-2.0
"""ClothFold-v0's towel mesh: two guards beside its 19 cells a side (``TOWEL_CELLS``).

Why 19 cells: on the robot's own fold (1,024 worlds a side on L4s, cap off, 2026-11-14) the 13-cell towel was placed
off the far corner in 23.9 % of worlds and the 19-cell towel in 5.3 % (folded 30 steps after the script 34.3 -> 50.5 %).
That GPU measurement is the evidence; neither guard here is a law of the mesh.

- (A), a design guard (fast, no build): the towel's cell is no larger than its fabric's bending length (B / w)^(1/3),
  B the card's bending rigidity and w the towel's weight per area: 1.57 cm for the shipped towel. A coarser mesh cannot
  form the crease the fold needs. It is a rule of thumb, against the card's model value of B: at half that bending
  (the card's range is 4e-6 to 1e-4 N m) the length is 1.25 cm and 19 cells (1.26 cm) would fail it; and 16 cells
  (1.50 cm) pass it and fail (B).
- (B), a one-rig regression guard (slow, CPU, a few minutes), chosen from a 48-case grid: lift the towel's +x +y
  corner and turn it over the diagonal fold line, as the scripted demo does, and the corner lying on the table (-x -y)
  stays where it is. The towel lies alone on the ground. A 2 cm patch at the held corner (the pinch) moves
  kinematically: its vertices are inactive in the cloth solver and are placed before every substep, the patch keeping
  its orientation as the fingers do. The corner travels a half circle about the fold line, radius its distance from the
  line (the cloth's reach), with the apex at 0.9, 1.0 and 1.1 times that radius (one world each), over 1.2 s (the
  demo's 60 steps of 20 ms), and is then held for 0.2 s. The reading is the lying corner's largest retreat: how far it
  moves back from the point it had been drawn to, toward the held corner. Bound 0.5 cm.
  This rig, on the CPU (2026-11-13), at apex 0.9 / 1.0 / 1.1: 13 cells 0.28 / 1.80 / 3.33 cm, 16 cells 1.14 / 0.44 /
  1.47, 19 cells 0.08 / 0.15 / 0.14, 25 cells 0.07 / 0.09 / 0.08. It separates 13 from 19 cells; other rigs do not
  always. In the grid (one world each), 19 cells' worst was 0.87 cm (a point hold, apex 0.9), and an independent
  check's one-change variants (2026-11-15) kept "19 passes, 13 fails" in 6 of 12: 19 cells read 1.48 cm with a point
  hold (apex 0.9, 1.2 s) and 1.49 / 0.34 / 3.75 cm with a 3 cm patch; 13 cells passed with a point hold at apex 0.8
  and with a 1.8 s turn; 16 cells (0.51) beat 19 (1.48) on the point hold. So a failure here says the towel's turn
  changed, not that a mesh is too coarse; check it against the robot's fold. Not checked: other fabrics, self-contact
  off. The kinematic hold lifts the towel with energy the health monitor does not see, so the worlds read DEGRADED
  (ENERGY_JUMP); none may be INVALID.
- (C), the fold itself (GPU, slow, release tier: about 1.5-2 h on one L4): the scripted demo
  (examples/envs/09_cloth_fold.py, 64 worlds, 35 N press) as shipped. Placed off at most 8 of 64, release misses at
  most 33 of 64. From 16 runs of 64 a side (L4s, main 50cd06a51, 2026-11-14): placed off per run 1-5 at 19 cells and
  9-20 at 13, so the bound separates every run (a 19-cell run passes it with ~99 % odds); release misses 16-27 at 19
  cells (11-22 at 13), so that bound is the 19-cell mean + 3 sd, a guard against the release getting worse. The whole
  run took 85-121 min warm and 109-131 min on a fresh kernel cache (cap 9000 s). This test, once on a fresh L4
  (2026-11-16): placed off 6, release 24, reset 0, folded 53 %; 127 min with no kernel cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy._vendor.newton import ParticleFlags
from alloy.cards.library import CardLibrary, default_root
from alloy.envs.soft_tasks import TOWEL, TOWEL_CELLS, add_towel, cloth_fold_scene

G = 9.81
APEX = (0.9, 1.0, 1.1)       # apex height over the reach, one world each
PATCH = 0.02                 # m, the held corner patch (x and y within this of the corner)
DT = 0.02                    # s, ClothFold's control period
SETTLE_STEPS = 5
TURN_S, HOLD_S = 1.2, 0.2
BOUND = 0.005                # m, the lying corner's largest retreat


def bending_length(cells: int | None = None) -> tuple[float, float]:
    """(cell, bending length) in m for ClothFold's towel as ``cloth_fold_scene`` builds it."""
    item = cloth_fold_scene(1, resolution=cells or TOWEL_CELLS).items["towel"]
    card = CardLibrary(roots=[default_root()]).material(item["material"])
    weight = item["overrides"]["mass"] / (TOWEL[0] * TOWEL[1]) * G
    return TOWEL[0] / (cells or TOWEL_CELLS), (float(card.elastic.bending.value) / weight) ** (1.0 / 3.0)


@pytest.mark.law
def test_towel_cells_are_no_coarser_than_the_fabric_bends():
    import inspect  # noqa: PLC0415

    assert inspect.signature(cloth_fold_scene).parameters["resolution"].default == TOWEL_CELLS
    cell, length = bending_length()
    assert cell <= length, (f"ClothFold's towel cell {100 * cell:.2f} cm is coarser than its fabric's bending length "
                            f"{100 * length:.2f} cm: the fold's crease cannot form (see this file's docstring)")
    coarse, _ = bending_length(13)
    assert coarse > length      # the earlier default: 1.85 cm


def _cloth_solver(sim) -> object:
    """The cloth solver of a towel-alone scene (one solver entry)."""
    solver = getattr(sim.engine.stepper, "solver", None)
    assert solver is not None and solver.model.particle_count > 0, type(sim.engine.stepper).__name__
    return solver


class _Hold:
    """Moves the held patch of every world along its half circle, placed before each cloth substep (CPU)."""

    def __init__(self, sim, solver, start: np.ndarray, a: int, b: int, patch: np.ndarray) -> None:
        first = sim.index.particle_range("towel")[0]
        self.idx = np.stack([sim.index.per_world(first + int(i), "particle") for i in patch], axis=1)   # (W, P)
        a0, b0 = start[:, a], start[:, b]
        self.toward = (a0 - b0) * [1.0, 1.0, 0.0]
        self.toward /= np.linalg.norm(self.toward, axis=1, keepdims=True)
        self.centre = 0.5 * (a0 + b0) * [1.0, 1.0, 0.0]
        self.reach = ((a0 - self.centre) * self.toward).sum(1, keepdims=True)
        self.z0, self.offset = a0[:, 2:3], start[:, patch] - a0[:, None]
        self.apex, self.t = np.array(APEX)[:, None], 0.0
        flags = solver.model.particle_flags.numpy().copy()
        flags[self.idx.ravel()] &= ~int(ParticleFlags.ACTIVE)
        solver.model.particle_flags.assign(flags)
        inner = solver.step

        def step(state_in, state_out, control, contacts, dt, *args, **kwargs):
            self.t += dt
            q, qd = state_in.particle_q.numpy().copy(), state_in.particle_qd.numpy().copy()
            q[self.idx], qd[self.idx] = self.target(self.t), 0.0
            state_in.particle_q.assign(q)
            state_in.particle_qd.assign(qd)
            return inner(state_in, state_out, control, contacts, dt, *args, **kwargs)

        solver.step = step

    def target(self, t: float) -> np.ndarray:
        f = min(t / TURN_S, 1.0)
        corner = (self.centre + self.reach * np.cos(np.pi * f) * self.toward
                  + (self.z0 + self.apex * self.reach * np.sin(np.pi * f)) * [0.0, 0.0, 1.0])
        return corner[:, None] + self.offset


def lying_corner_retreat(cells: int) -> np.ndarray:
    """(worlds,) the lying corner's largest retreat (m) while the held corner is turned over, one world per apex."""
    scene = alloy.Scene(worlds=len(APEX), dt=DT)
    scene.add_ground()
    add_towel(scene, cells)
    sim = alloy.build(scene, device="cpu", seed=0)
    solver = _cloth_solver(sim)
    sim.step(SETTLE_STEPS)
    start = sim.state()["towel.points"].copy()                      # (W, n, 3)
    assert start.shape[1] == (cells + 1) ** 2                       # resolution counts cells a side
    a = int((start[0, :, 0] + start[0, :, 1]).argmax())
    b = int((-start[0, :, 0] - start[0, :, 1]).argmax())
    patch = np.flatnonzero((start[0, :, 0] >= start[0, a, 0] - PATCH) & (start[0, :, 1] >= start[0, a, 1] - PATCH))
    hold = _Hold(sim, solver, start, a, b, patch)
    drawn = []
    for _ in range(int(round((TURN_S + HOLD_S) / DT))):
        sim.step(1)
        drawn.append(((sim.state()["towel.points"][:, b] - start[:, b]) * hold.toward).sum(1))
    drawn = np.array(drawn)                                         # (steps, W): toward the held corner, m
    health = sim.health()
    assert not health["invalid_worlds"], str(health)[:600]
    return (np.maximum.accumulate(drawn, axis=0) - drawn).max(axis=0)


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
def test_turning_the_corner_over_leaves_the_lying_corner_in_place():
    retreat = lying_corner_retreat(TOWEL_CELLS)
    assert retreat.max() <= BOUND, (f"at {TOWEL_CELLS} cells the lying corner retreated "
                                    f"{np.round(100 * retreat, 2).tolist()} cm (apex {APEX}; bound {100 * BOUND} cm)")


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1200)
def test_a_coarse_towel_pushes_the_lying_corner_off():
    """On this rig the 13-cell towel's lying corner is pushed back past the bound, so the guard can fail."""
    retreat = lying_corner_retreat(13)
    assert retreat.max() > BOUND, np.round(100 * retreat, 2).tolist()


FOLD_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "envs" / "09_cloth_fold.py"
FOLD_CAP_S = 9000                # s: 109-131 min measured on a fresh kernel cache
PLACED_OFF_MAX, RELEASE_MAX = 8, 33   # of 64 worlds


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(FOLD_CAP_S + 120)
def test_the_demo_fold_rarely_places_the_corner_off():
    """(C): the shipped demo on 64 worlds of a GPU places the held corner off the far corner in at most 8 and misses at
    release in at most 33 (this file's docstring gives the runs behind both bounds)."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(FOLD_EXAMPLE.parents[2]), os.environ.get("PYTHONPATH", "")]))
    run = subprocess.run([sys.executable, str(FOLD_EXAMPLE), "64", "35"], cwd=FOLD_EXAMPLE.parents[2], env=env,
                         capture_output=True, text=True, timeout=FOLD_CAP_S, check=False)
    assert run.returncode == 0, run.stderr[-3000:]
    out = json.loads([line for line in run.stdout.splitlines() if line.startswith("{")][-1])
    assert out["worlds"] == 64 and out["misses"] is not None, out
    print(json.dumps(out))       # the run's counts, kept in the report (pytest -rP)
    m = out["misses"]
    assert m["placed_off"] <= PLACED_OFF_MAX, (f"placed off in {m['placed_off']} of 64 worlds (bound {PLACED_OFF_MAX}; "
                                               f"19 cells measured 1-5 a run, 13 cells 9-20): {out}")
    assert m["release"] <= RELEASE_MAX, (f"release misses in {m['release']} of 64 worlds (bound {RELEASE_MAX}; "
                                         f"19 cells measured 16-27 a run): {out}")
