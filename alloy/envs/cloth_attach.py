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
``OPEN_GAP``. Nothing runs on the host, so a captured CUDA graph replays it, and a world the env resets lets go by
itself (its fingers start open).

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


def _body_index(model: Any, name: str, worlds: int) -> np.ndarray:
    """(worlds,) global body index of the body named ``name`` (the last part of its label, e.g. "finger_left" in
    "worldbody/.../palm/finger_left") in each world; exactly one per world."""
    labels = [str(x) for x in (getattr(model, "body_label", None) or getattr(model, "body_key"))]
    world = model.body_world.numpy() if getattr(model, "body_world", None) is not None else np.zeros(len(labels), int)
    out = np.full(worlds, -1, np.int32)
    for b, label in enumerate(labels):
        if label.rsplit("/", 1)[-1] != name or not 0 <= world[b] < worlds:
            continue
        if out[world[b]] >= 0:
            raise ValueError(f"cloth attach: two bodies named {name!r} in world {world[b]}")
        out[world[b]] = b
    if (out < 0).any():
        raise ValueError(f"cloth attach: no body {name!r} in worlds {np.flatnonzero(out < 0).tolist()}")
    return out


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
        solver.add_colour_callback(self._accumulate, external=True, when="after_accumulate")

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
    raise ValueError("cloth attach: the simulation has no cloth (VBD) entry")


def install(sim: Any, robot: str = "gripper", finger_half: tuple[float, float, float] = (0.012, 0.005, 0.03),
            **kw: Any) -> ClothAttach:
    """Attach-on-grasp for the Cartesian gripper ``robot`` (alloy.envs.robots.gantry_gripper: fingers
    "<robot>/finger_left", "<robot>/finger_right", box half sizes ``finger_half``) on the scene's cloth.
    Example: ``Task(..., on_build=lambda sim: install(sim, "gripper"))``."""
    solver = cloth_solver(sim)
    model = sim.engine.model
    worlds = int(sim.worlds)
    left = _body_index(model, "finger_left", worlds)          # one gantry gripper per world (ClothFold)
    right = _body_index(model, "finger_right", worlds)
    margin = 0.002                                        # vertices at the fingers' edges, within a contact radius
    att = ClothAttach(solver, left, right, (finger_half[0] + margin, finger_half[1], finger_half[2] + margin), **kw)
    log.info("cloth attach-on-grasp (a shortcut, not a friction grasp) on %s, %d worlds", type(solver).__name__, worlds)
    return att


__all__ = ["CLOSE_GAP", "HOLD_K", "OPEN_GAP", "ClothAttach", "cloth_solver", "install", "installed"]
