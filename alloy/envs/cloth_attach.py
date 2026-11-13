# SPDX-License-Identifier: Apache-2.0
"""Attach-on-grasp for cloth: an opt-in shortcut, NOT a friction grasp.

``make("ClothFold-v0", grasp="attach")``. While the gripper's fingers are closed on the towel, the towel vertices
between them are held to the left finger: each by a stiff spring to an anchor fixed in that finger's body frame,
taken where the vertex was when the fingers closed. When the fingers open, the vertices are let go and the towel
is free again. Cloth benchmarks grasp this way (a picked particle follows the picker); a real gripper holds cloth
only by friction, which is ``grasp="friction"``, the default and the measured result (docs/guide/envs.md reports both).

How. The cloth solver's per-colour callback (``alloy.solvers.vbd.base`` ``add_colour_callback``, after-accumulate slot)
adds k (anchor - x) to each held vertex's force and k I to its Hessian, an implicit spring that VBD solves with the
cloth (stable at any k). Which vertices are held is decided on the device before every cloth step from the finger
poses the cloth solver sees: the fingers count as closed below ``CLOSE_GAP`` between their centres (0.09 m open,
0.014 m on the 2-layer ridge, 0.010 m shut on nothing); then every free vertex inside the box between the two
fingers (the fingers' length and width, from one finger's inner face to the other's) is held. They are let go above
``OPEN_GAP``. Nothing runs on the host, so a captured CUDA graph replays it. A reset world lets go of every vertex
it held (the cloth solver's reset, whole or masked, as ``env.reset()`` and an autoreset run it), whatever its
fingers' gap: a gripper that starts less than ``OPEN_GAP`` open would otherwise keep the last episode's holds.

Limits, stated: one-way. The gripper does not feel the held towel's weight or pull (the 30 g towel weighs 0.3 N;
the gantry's servos are kp 3000 N/m). The held vertices stay in the fingers' contacts too, so the fingers still
squeeze them. The callback's force has a nonzero sum over the towel, so it is registered as external: the momentum
solver (``SolverVBDMomentum``) then skips its coarse translation step for the towel's entry in this mode, as for any
external force on cloth. A coupler that steps the cloth twice per exchange from the same input takes the same
decisions both times (they depend on that input only). Which vertices are held, and their anchors, are not snapshot
state: a period the engine re-runs, or a world it restores, keeps the holds of the discarded attempt (they are taken
again from nearly the same inputs).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger(__name__)

HOLD_K = 1.0e3          # N/m per held vertex: 0.1 mm under a 0.1 N share of the towel's weight
CLOSE_GAP = 0.020       # m between the finger centres: closed on cloth below this (on the ridge: 0.014)
OPEN_GAP = 0.030        # m: held vertices are let go above this (hysteresis against a jittering close)
FREE, HELD = 0, 1


@wp.kernel
def update_hold(q: wp.array[wp.vec3], particle_world: wp.array[wp.int32], body_q: wp.array[wp.transform],
                left: wp.array[wp.int32], right: wp.array[wp.int32], half: wp.vec3, close_gap: float,
                open_gap: float, held: wp.array[wp.int32], anchor: wp.array[wp.vec3]):
    """Before a cloth step: hold the free vertices between closed fingers (anchor in the left finger's frame), let go
    of every held vertex of a world whose fingers opened."""
    i = wp.tid()
    w = wp.max(particle_world[i], 0)                       # a one-world model may mark its particles -1
    xl = body_q[left[w]]
    xr = body_q[right[w]]
    pl = wp.transform_get_translation(xl)
    pr = wp.transform_get_translation(xr)
    gap = wp.length(pr - pl)
    if held[i] == HELD:
        if gap > open_gap:
            held[i] = FREE
        return
    if gap > close_gap:
        return
    # the box between the fingers, in the left finger's frame: its y axis points at the right finger
    loc = wp.transform_point(wp.transform_inverse(xl), q[i])
    rel = wp.transform_point(wp.transform_inverse(xl), pr)
    span = wp.abs(rel[1])
    s = wp.where(rel[1] >= 0.0, 1.0, -1.0)
    across = s * loc[1]                                    # 0 at the left finger's centre plane, span at the right's
    if wp.abs(loc[0]) <= half[0] and wp.abs(loc[2]) <= half[2] and across >= 0.0 and across <= span:
        held[i] = HELD
        anchor[i] = loc


@wp.kernel
def hold_accumulate(colour: int, colors: wp.array[wp.int32], q: wp.array[wp.vec3], particle_world: wp.array[wp.int32],
                    body_q: wp.array[wp.transform], left: wp.array[wp.int32], held: wp.array[wp.int32],
                    anchor: wp.array[wp.vec3], k: float, forces: wp.array[wp.vec3], hessians: wp.array[wp.mat33]):
    """After-accumulate slot: the implicit spring k (anchor - x) on every held vertex of ``colour``."""
    i = wp.tid()
    if held[i] != HELD or colors[i] != colour:
        return
    a = wp.transform_point(body_q[left[wp.max(particle_world[i], 0)]], anchor[i])
    forces[i] = forces[i] - k * (q[i] - a)
    hessians[i] = hessians[i] + k * wp.identity(3, float)


@wp.kernel
def release_worlds(particle_world: wp.array[wp.int32], mask: wp.array[wp.bool], held: wp.array[wp.int32]):
    """A reset: let go of every held vertex of the worlds ``mask`` marks (Newton's mask may carry one more entry,
    the global one, which no cloth vertex of a world reads)."""
    i = wp.tid()
    w = wp.max(particle_world[i], 0)                       # a one-world model may mark its particles -1
    if w < mask.shape[0]:
        if mask[w]:
            held[i] = FREE


def _finger_bodies(sim: Any, robot: str, finger: str) -> np.ndarray:
    """(worlds,) engine body index of "<robot>/<finger>" in each world (``sim.index``, the names ``sim.names()``
    lists). Raises UnknownNameError, with the fix, when the scene has no such body."""
    from alloy.api.errors import UnknownNameError  # noqa: PLC0415

    name = f"{robot}/{finger}"
    if name not in sim.index.bodies:
        robots = sorted({b.split("/", 1)[0] for b in sim.index.bodies
                         if b.endswith("/finger_left") or b.endswith("/finger_right")})
        raise UnknownNameError(
            f"cloth attach: no body {name!r} in the scene (attach-on-grasp needs a gripper with bodies "
            f"'<robot>/finger_left' and '<robot>/finger_right', as alloy.envs.robots.gantry_gripper has)",
            "pass robot= the name you gave add_robot for such a gripper (in this scene: "
            f"{', '.join(robots) or 'none'})",
            field="robot")
    return sim.index.per_world(sim.index.body(name), "body").astype(np.int32)


class ClothAttach:
    """Attach-on-grasp on one cloth solver (module docstring). ``install(sim, robot="gripper")`` does the wiring;
    ``read()`` gives the held-vertex count per world."""

    def __init__(self, solver: Any, left: np.ndarray, right: np.ndarray, half: tuple[float, float, float],
                 k: float = HOLD_K, close_gap: float = CLOSE_GAP, open_gap: float = OPEN_GAP) -> None:
        if not hasattr(solver, "add_colour_callback"):
            raise TypeError(f"cloth attach needs Alloy's cloth solver (SolverVBDAlloy), got {type(solver).__name__}")
        if getattr(solver, "_cloth_attach", None) is not None:
            raise ValueError("cloth attach: this cloth solver already has one "
                             "(installing twice would wrap its step twice)")
        solver._cloth_attach = self
        self.solver, m = solver, solver.model
        self.dev = m.particle_q.device
        n = int(m.particle_count)
        self.n, self.k, self.close_gap, self.open_gap = n, float(k), float(close_gap), float(open_gap)
        self.half = wp.vec3(*half)
        self.left = wp.array(np.asarray(left, np.int32), dtype=wp.int32, device=self.dev)
        self.right = wp.array(np.asarray(right, np.int32), dtype=wp.int32, device=self.dev)
        self.world = m.particle_world
        self.held = wp.zeros(n, dtype=wp.int32, device=self.dev)
        self.anchor = wp.zeros(n, dtype=wp.vec3, device=self.dev)
        self._out = None
        inner = solver.step

        def step(state_in, state_out, control, contacts, dt, *args, **kwargs):
            self._out = state_out
            if state_in.body_q is not None and state_in.body_q.shape[0] > 0:
                wp.launch(update_hold, dim=n, inputs=[state_in.particle_q, self.world, state_in.body_q, self.left,
                                                      self.right, self.half, self.close_gap, self.open_gap],
                          outputs=[self.held, self.anchor], device=self.dev)
            return inner(state_in, state_out, control, contacts, dt, *args, **kwargs)

        solver.step = step
        inner_reset = solver.reset

        def reset(state, world_mask=None, flags=None):
            out = inner_reset(state, world_mask=world_mask, flags=flags)
            self.release(world_mask)
            return out

        solver.reset = reset
        solver.add_colour_callback(self._accumulate, external=True, when="after_accumulate")

    def release(self, world_mask: Any = None) -> None:
        """Let go of every held vertex of the worlds in ``world_mask`` (bool or int per world, numpy or Warp; None:
        every world). The cloth solver's reset calls it, so a reset world starts with nothing held."""
        if world_mask is None:
            self.held.zero_()
            return
        mask = world_mask
        if not (isinstance(mask, wp.array) and mask.dtype == wp.bool and mask.device == self.dev):
            host = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
            mask = wp.array(np.asarray(host).reshape(-1) != 0, dtype=wp.bool, device=self.dev)
        wp.launch(release_worlds, dim=self.n, inputs=[self.world, mask], outputs=[self.held], device=self.dev)

    def _accumulate(self, solver: Any, colour: int, state: Any, contacts: Any) -> None:
        # the anchors follow the fingers' pose the contacts use: Newton evaluates the particle iteration's body contacts
        # at state_out.body_q, the step's end pose (solver_vbd.py, body_q_for_particles)
        poses = self._out if self._out is not None else state
        if poses.body_q is None or poses.body_q.shape[0] == 0:
            return
        wp.launch(hold_accumulate, dim=self.n,
                  inputs=[colour, solver.model.particle_colors, state.particle_q, self.world, poses.body_q, self.left,
                          self.held, self.anchor, self.k],
                  outputs=[solver.particle_forces, solver.particle_hessians], device=self.dev)

    def read(self) -> np.ndarray:
        """(worlds,) number of held vertices in each world."""
        held, world = self.held.numpy(), self.world.numpy()
        return np.bincount(world[held == HELD], minlength=int(self.left.shape[0]))


def installed(sim: Any) -> ClothAttach | None:
    """The attach installed on ``sim`` (None without one). Example: ``installed(env.sim).read()``."""
    try:
        return getattr(cloth_solver(sim), "_cloth_attach", None)
    except ValueError:
        return None


def cloth_solver(sim: Any) -> Any:
    """The cloth (VBD) solver of a built simulation's coupled stepper."""
    entries = getattr(getattr(sim.engine, "stepper", None), "_entries", {})
    for entry in entries.values():
        s = getattr(entry, "solver", None)
        if s is not None and hasattr(s, "add_colour_callback") and s.model.particle_count > 0:
            return s
    from alloy.api.errors import SceneError  # noqa: PLC0415  (a ValueError: installed() still returns None)

    raise SceneError("cloth attach: the simulation has no cloth (VBD) entry",
                     "add the cloth with scene.add_cloth(...) before building", field="sim")


def install(sim: Any, robot: str = "gripper", finger_half: tuple[float, float, float] = (0.012, 0.005, 0.03),
            **kw: Any) -> ClothAttach:
    """Attach-on-grasp for the Cartesian gripper ``robot`` (alloy.envs.robots.gantry_gripper: fingers
    "<robot>/finger_left", "<robot>/finger_right", box half sizes ``finger_half``) on the scene's cloth. A scene
    without those bodies raises UnknownNameError, one without a cloth SceneError, each with the fix.
    Example: ``Task(..., on_build=lambda sim: install(sim, "gripper"))``."""
    solver = cloth_solver(sim)
    worlds = int(sim.worlds)
    left = _finger_bodies(sim, robot, "finger_left")          # that robot's fingers, one pair per world
    right = _finger_bodies(sim, robot, "finger_right")
    margin = 0.002                                        # vertices at the fingers' edges, within a contact radius
    att = ClothAttach(solver, left, right, (finger_half[0] + margin, finger_half[1], finger_half[2] + margin), **kw)
    log.info("cloth attach-on-grasp (a shortcut, not a friction grasp) on %s, %d worlds", type(solver).__name__, worlds)
    return att


__all__ = ["CLOSE_GAP", "HOLD_K", "OPEN_GAP", "ClothAttach", "cloth_solver", "install", "installed"]
