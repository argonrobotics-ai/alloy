# SPDX-License-Identifier: Apache-2.0
"""Law: a fabric cantilever droops as its card's hinge chain, and a regression pin on the hinge stiffness (audit
AB-laws-6).

Peirce's cantilever (the textile bending-length test, ASTM D1388): a strip of a library fabric is clamped level at
one end and droops under its own weight. With bending rigidity D [N m] (the card's ``elastic.bending``) and areal
weight w g [N/m2] (the card's density x the sheet thickness), the bending length is c = (D / (w g))^(1/3) and the
rest shape is the heavy elastica D theta'' = -w g (L - s) cos(theta), theta(0) = 0, theta'(L) = 0 (theta the slope
below level, s the arc length from the clamp). At a free length L = 1.5 c the tip hangs 0.3735 L below the clamp;
half the rigidity gives 0.5905 L, twice gives 0.2039 L.

Reference: the elastica is solved here as a two-point boundary value problem (scipy ``solve_bvp``, tolerance 1e-9;
a shooting solve on the root curvature agrees to 4 digits). A mesh of 12 cells cannot follow it exactly: its hinge
across the clamp line spans a whole cell, so a strip with exactly the card's D hangs as if clamped half a cell
further back. The law's reference is therefore the exact rest of the 12-link hinge chain (rigid links, a hinge of
stiffness D / cell at every node including the clamp, the weight lumped at the nodes; WP6's review, the A11
cantilever's precedent): 0.4261 L, 14 % over the elastica; 24 links give 0.3998, 400 give 0.3745. Both references
read only the card's D, its areal weight and g = 9.81.

Rig: ``alloy.Scene.add_cloth`` (the cloth route: the card's D reaches the hinges through the plate hinge map),
cotton_woven at its card thickness 0.6 mm (D 1.98e-5 N m, 300 g/m2, c 18.9 mm), a 30.7 x 7.1 mm sheet along the
mesh's x axis, 13 x 3 cells of 2.36 mm. The two vertex rows of the first cell are held (mass 0 on the engine model);
12 cells hang free. The cards carry no damping and a strip in vacuum never rests, so it hangs in still air: a force
plug-in with the drag -2 w1 m v, critical for the first bending mode w1 = 3.516 sqrt(D / (w L^4)). The drag is zero
at rest, so the rest shape does not depend on it. 14 / w1 = 0.40 s, default cloth accuracy, CPU. Three worlds step
together: the route's hinges, and the same hinges x0.5 and x2. The engine's softer strip is overdamped by that drag
(about 1.3), so the read at 0.40 s (0.5 mm/s) is 0.2 % short of the long run's 0.5706-0.5714 (WP2's review, 1.6 s).

Measured 2026-11-09 (CPU), tip drop over free length; the elastica gives 0.3735:

| hinges | main (b9441c386) | Newton 1.8 tree |
|---|---|---|
| the route's | 0.5705 (+33.9 % on the chain) | 0.5706 |
| x0.5 | 0.7572 | 0.7572 |
| x2 | 0.3534 | 0.3533 |

- The physics law fails: a free strip along a mesh axis bends as a chain of 0.64x its card's D (known gap
  docs/known_gaps/cloth-bending-free-strip-soft.md). The route's plate hinge map puts 0.75 D on axis hinges by
  design (the chain at 0.75 D: 0.5204 L); the rest is not explained. A strict xfail until that gap closes. Its 5 %
  tolerance admits hinges from 0.94 to 1.07 of the card's D.
- The regression pin holds the engine's present droop in a band of +-3 % (CPU repeats agree to 7 digits, the
  Newton 1.8 tree is 0.02 % off; +-3 % of droop is about +-5 % of hinge stiffness) and asserts that a factor of 2
  in the hinge stiffness, either way, leaves the band. The audit's mutant (bending x0.5 in the vendored VBD hinge
  kernel) passed all 213 fast CPU cloth tests but a steel strip's; here it gives 0.7572 and fails the pin.
- The x0.5 and x2 worlds scale ``model.edge_bending_properties[:, 0]`` after the build: the product the kernel
  forms (k = stiffness x rest length), so they are the audit's mutants without a kernel build. The audit's mutant
  itself, compiled once on main, gave the x0.5 world's 0.7572068572 in the route's world, bit for bit.
- One world alone gives 0.5705, cloth_polyester_woven (D 3.43e-6 N m, 158 g/m2) 0.5696 against the same 0.3735.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp
from scipy.integrate import solve_bvp
from scipy.optimize import minimize

import alloy
from alloy.cards.library import CardLibrary
from alloy.plugins import ForcePlugin

G = 9.81                          # m/s2
MATERIAL, THICKNESS = "cotton_woven", 0.6e-3
FREE_LENGTHS = 1.5                # free length over the bending length c
CELLS, WIDTH_CELLS = 12, 3        # free cells along the strip (one more is clamped), cells across
HINGE_SCALES = (1.0, 0.5, 2.0)    # one world each: the route's hinges, then the audit's mutants
SETTLE = 14.0                     # run time x w1: (1 + w1 t) exp(-w1 t) is 1e-5 there for the card's D
REST_SPEED = 1.0e-3               # m/s, fastest vertex of the route's world over the last control step (0.5 mm/s)
CHAIN_TOLERANCE = 0.05            # the physics law, on tip drop / the exact hinge chain's tip drop
ENGINE_DROP, BAND = 0.5705, 0.03  # the regression pin: the engine's tip drop / L on main b9441c386, and its band

pytestmark = pytest.mark.timeout(600)   # 25 s of CPU with warm kernels; a cold VBD kernel build or a loaded host


@wp.kernel
def _drag(qd: wp.array[wp.vec3], mass: wp.array[wp.float32], rate: wp.float32, f: wp.array[wp.vec3]):
    i = wp.tid()
    f[i] = f[i] - rate * mass[i] * qd[i]


class _StillAir(ForcePlugin):
    """Drag -rate m v on every cloth vertex [N]: the air a hanging strip comes to rest in."""

    name = "still_air"
    summary = "mass-proportional drag on particles"

    @dataclass
    class Config:
        rate: float = 50.0        # 1/s

    def apply(self, state, forces, dt: float) -> None:  # noqa: ANN001 - newton.State, alloy.plugins.Forces
        m = self.ctx.model
        wp.launch(_drag, dim=m.particle_count, inputs=[state.particle_qd, m.particle_mass, float(self.config.rate)],
                  outputs=[forces.particle_f], device=m.device)


def elastica_tip_drop(gamma: float) -> float:
    """Tip drop over length of the level-clamped heavy elastica, gamma = w g L^3 / D (in s / L: theta'' =
    -gamma (1 - s) cos(theta), theta(0) = 0, theta'(1) = 0; drop = integral of sin(theta))."""
    s = np.linspace(0.0, 1.0, 201)
    sol = solve_bvp(lambda x, y: np.vstack([y[1], -gamma * (1.0 - x) * np.cos(y[0])]),
                    lambda a, b: np.array([a[0], b[1]]), s, np.zeros((2, s.size)), tol=1e-9, max_nodes=50_000)
    assert sol.success, sol.message
    fine = np.linspace(0.0, 1.0, 4001)
    return float(np.trapezoid(np.sin(sol.sol(fine)[0]), fine))


def chain_tip_drop(gamma: float, links: int) -> float:
    """Tip drop over length at rest of ``links`` rigid links hinged in a row, level-clamped: a hinge of stiffness
    D / link length at every node (the clamp's too), the weight lumped at the nodes (half a link at the tip);
    gamma = w g L^3 / D. The mesh's own reference: what hinges of exactly the card's D would give."""
    dx = 1.0 / links
    weight = np.full(links, gamma * dx)
    weight[-1] *= 0.5

    def energy(a: np.ndarray) -> float:
        bend = np.diff(np.concatenate([[0.0], a]))
        return float(0.5 * np.sum(bend**2) / dx - np.sum(weight * np.cumsum(np.sin(a)) * dx))

    best = minimize(energy, np.zeros(links), method="BFGS", options={"gtol": 1e-10})
    return float(np.sum(np.sin(best.x)) * dx)


@dataclass(frozen=True)
class Droop:
    drops: tuple[float, ...]      # tip drop over the free length, one per HINGE_SCALES world
    speed: float                  # m/s, fastest vertex of the route's world over the last control step
    gamma: float                  # w g L^3 / D at the built free length


def _clamp_and_scale(model, cell: float, sheet_mass: float) -> None:  # noqa: ANN001 - newton.Model
    """Hold the first cell's two vertex rows in every world (mass 0) and scale world k's hinges by HINGE_SCALES[k].
    Before the clamp each world's particles must weigh ``sheet_mass``: the reference's w is the built sheet's."""
    q0 = model.particle_q.numpy()
    held = np.flatnonzero(q0[:, 0] < q0[:, 0].min() + 1.25 * cell)
    mass, inv = model.particle_mass.numpy().copy(), model.particle_inv_mass.numpy().copy()
    built = np.bincount(model.particle_world.numpy(), weights=mass.astype(np.float64))
    assert built == pytest.approx(sheet_mass, rel=1.0e-6, abs=0.0), (built, sheet_mass)
    mass[held], inv[held] = 0.0, 0.0
    model.particle_mass.assign(mass)
    model.particle_inv_mass.assign(inv)
    hinges = model.edge_bending_properties.numpy().copy()
    world = model.particle_world.numpy()[model.edge_indices.numpy()[:, 2]]
    hinges[:, 0] *= np.asarray(HINGE_SCALES)[world]
    model.edge_bending_properties.assign(hinges)


def _hang(sim: alloy.Sim, cell: float, steps: int, weight_over_rigidity: float, sheet_mass: float) -> Droop:
    """Clamp, step ``steps`` control periods and read each world's tip row."""
    start = sim.state(["strip"])["strip.points"].copy()                    # (worlds, vertices, 3)
    xs = np.unique(np.round(start[0, :, 0], 7))
    assert len(xs) == CELLS + 2 and start.shape[1] == (CELLS + 2) * (WIDTH_CELLS + 1)      # the grid asked for
    _clamp_and_scale(sim.engine.model, cell, sheet_mass)
    tip = np.flatnonzero(start[0, :, 0] > xs[-1] - 0.25 * cell)
    held = start[0, :, 0] < xs[1] + 0.25 * cell
    free = float(xs[-1] - xs[1])
    sim.step(steps - 1)
    before = sim.state(["strip"])["strip.points"].copy()
    sim.step(1)
    after = sim.state(["strip"])["strip.points"]
    assert sim.check()["ok"]
    assert np.array_equal(after[:, held], start[:, held])                  # the clamp held in every world
    speed = float(np.linalg.norm(after[0] - before[0], axis=1).max() / sim.dt)
    drops = (start[:, tip, 2].mean(axis=1) - after[:, tip, 2].mean(axis=1)) / free
    return Droop(tuple(float(d) for d in drops), speed, weight_over_rigidity * free**3)


@functools.lru_cache(maxsize=None)
def droop() -> Droop:
    """Hang the strip in len(HINGE_SCALES) worlds (one run, shared by both tests)."""
    mat = CardLibrary().material(MATERIAL)
    D, w = float(mat.elastic.bending.value), float(mat.density.value) * THICKNESS
    length = FREE_LENGTHS * (D / (w * G)) ** (1.0 / 3.0)
    cell = length / CELLS
    lx = (CELLS + 1) * cell
    w1 = 3.516 * math.sqrt(D / (w * length**4))
    scene = alloy.Scene(worlds=len(HINGE_SCALES), dt=0.01)
    scene.add_cloth("strip", size=(lx, WIDTH_CELLS * cell), pos=(0.5 * lx, 0.0, 1.0), thickness=THICKNESS,
                    resolution=(CELLS + 1, WIDTH_CELLS), material=MATERIAL, self_contact=False)
    sim = alloy.build(scene, device="cpu", plugins=[_StillAir(rate=2.0 * w1)])
    try:
        return _hang(sim, cell, int(math.ceil(SETTLE / w1 / sim.dt)), w * G / D, w * lx * WIDTH_CELLS * cell)
    finally:
        sim.close()


def test_the_hinge_chain_reference_tends_to_the_elastica():
    """The two references agree where they must: the 12-link chain is 14 % over the elastica (the mesh's clamp),
    and a fine chain is the elastica."""
    gamma = FREE_LENGTHS**3
    elastica = elastica_tip_drop(gamma)
    assert elastica == pytest.approx(0.3735, abs=2e-4)
    assert chain_tip_drop(gamma, CELLS) == pytest.approx(0.4261, abs=2e-4)
    assert chain_tip_drop(gamma, 400) == pytest.approx(elastica, rel=5e-3)


@pytest.mark.law
@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="known gap docs/known_gaps/cloth-bending-free-strip-soft.md: a free strip along a mesh "
                   "axis bends as 0.64x its card's D (tip drop 0.57 L against the hinge chain's 0.43 L, +34 %)")
def test_a_cotton_cantilever_droops_as_its_cards_hinge_chain():
    """The physics law: at rest the tip hangs within 5 % of the exact hinge chain's drop for the card's D and weight
    on this mesh (hinges of 0.94 to 1.07 of the card's D pass)."""
    got = droop()
    want = chain_tip_drop(got.gamma, CELLS)
    assert got.speed < REST_SPEED, f"not at rest: {got.speed:.4f} m/s"
    assert abs(got.drops[0] / want - 1.0) < CHAIN_TOLERANCE, (
        f"tip drop {got.drops[0]:.4f} L against the chain's {want:.4f} L ({got.drops[0] / want - 1.0:+.1%})")


def test_the_cantilever_droop_is_pinned_and_a_factor_of_two_in_hinge_stiffness_leaves_the_band():
    """A regression pin (audit AB-laws-6: bending x0.5 passed every fast fabric test), not a physics law. The band
    holds the engine's present droop, 0.5705 L, not the chain's 0.4261 L, because of the known gap
    cloth-bending-free-strip-soft; when that gap closes, ENGINE_DROP moves to the chain and the xfail above goes.
    The same strip with its hinges x0.5 (0.7572 L) and x2 (0.3534 L) must leave the band, so the pin shows on every
    run that it sees a factor of 2. Those two worlds are read as snapshots at the same time: the x0.5 world is
    still sinking (away from the band). The route's world must be at rest."""
    got = droop()
    right, soft, stiff = got.drops
    low, high = ENGINE_DROP * (1.0 - BAND), ENGINE_DROP * (1.0 + BAND)
    assert got.speed < REST_SPEED, f"not at rest: {got.speed:.4f} m/s"
    assert low < right < high, f"tip drop {right:.4f} L outside {low:.4f}-{high:.4f} L: the hinge stiffness moved"
    assert soft > high, f"hinges x0.5: tip drop {soft:.4f} L is inside the band (up to {high:.4f} L)"
    assert stiff < low, f"hinges x2: tip drop {stiff:.4f} L is inside the band (from {low:.4f} L)"
    assert not low < chain_tip_drop(got.gamma, CELLS) < high           # the gap is open: the band is not the law
