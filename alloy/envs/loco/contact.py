# SPDX-License-Identifier: Apache-2.0
"""Foot contact forces from the solver, for ``LocoConfig(contact=...)``.

``FootForces`` asks the rigid solver (MuJoCo Warp, or MuJoCo C on the CPU's default route) to export each physics
step's contacts and their forces, and sums them per foot shape with Newton's ``SensorContact``. ``read()`` gives the
(W, F) world-z component in newtons of the net contact force on each foot (up is positive: on flat ground this is
the normal force; a standing robot's feet add up to its weight). Example::

    from alloy.envs.loco import GO2, go2_velocity
    from alloy.envs.loco.contact import foot_forces
    env = go2_velocity(num_envs=4)
    forces = foot_forces(env.sim, GO2, "go2", env.xp)
    env.step(env.action_space.sample())
    forces.read()                    # (4, 4) N, columns in the order of GO2.feet

What the value is:
- the force of the LAST physics step of the control period (a 20 ms control step has 16 physics steps of 1.25 ms),
  not an average over the period; before the first step, and for a world just reset, it is the last stepped value
  (zeros before any step);
- every contact on the foot's own collision shape counts (floor, obstacles, the robot's other parts); contacts on
  the shank's shapes do not;
- ``read(component="norm")`` gives the magnitude instead, ``read_vectors()`` the (W, F, 3) world-frame vectors.

Cost and exactness: nothing changes until a ``FootForces`` is made (the solver's step is wrapped then, and captured
CUDA graphs are dropped once so the next periods are captured with the export in them). The export only reads the
solver's contacts; the trajectory is the same with and without it (tested bit for bit on the CPU). With the torch
backend the forces are a view of the sensor's device array (no host copy).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError

SPHERE_TOL = 1.0e-4                    # m: a foot shape's radius must match the robot description this closely


def foot_shapes(model: Any, foot_bodies: np.ndarray, radii: Sequence[float]) -> np.ndarray:
    """Shape index (W, F) of each foot: the sphere of the foot's radius on its body (``foot_bodies`` (W, F)).

    Example: ``foot_shapes(sim.engine.model, bodies, [0.022] * 4)[0]``."""
    from alloy._vendor import newton  # noqa: PLC0415

    shape_body = model.shape_body.numpy()
    kind, scale = model.shape_type.numpy(), model.shape_scale.numpy()
    flags = model.shape_flags.numpy()
    collides = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    out = np.zeros(foot_bodies.shape, np.int32)
    for (w, f), body in np.ndenumerate(foot_bodies):
        hit = np.flatnonzero((shape_body == body) & (kind == int(newton.GeoType.SPHERE)) & collides
                             & (np.abs(scale[:, 0] - float(radii[f])) < SPHERE_TOL))
        if hit.size != 1:
            raise EnvError(f"foot {f} (world {w}): {hit.size} colliding spheres of radius {radii[f]} m on its body, "
                           "need exactly one", "give the foot one sphere geom of that radius in the robot file, or "
                           "pass shapes= (W, F) shape indices to FootForces", field="contact")
        out[w, f] = hit[0]
    return out


def other_shapes(model: Any, robot_bodies: np.ndarray, feet: np.ndarray) -> np.ndarray:
    """Shape index (W, K) of every colliding shape on the robot's bodies (``robot_bodies`` (W, B)) that is not a
    foot (``feet`` (W, F)): the parts that should stay off the ground.

    Example: ``other_shapes(sim.engine.model, bodies, feet).shape[1]`` -> 17 for the Go2."""
    from alloy._vendor import newton  # noqa: PLC0415

    shape_body = model.shape_body.numpy()
    collides = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    rows = [np.setdiff1d(np.flatnonzero(np.isin(shape_body, robot_bodies[w]) & collides), feet[w])
            for w in range(robot_bodies.shape[0])]
    if len({len(r) for r in rows}) != 1:
        raise EnvError("the robot has a different number of collision shapes in different worlds",
                       "use the same robot file in every world", field="contact")
    return np.stack(rows).astype(np.int32)


class FootForces:
    """Solver contact force per foot shape. ``shapes``: (W, F) shape indices of the model, one row per world.

    Example: ``FootForces(sim.engine, shapes, env.xp).read()`` -> (W, F) newtons, up positive."""

    def __init__(self, engine: Any, shapes: np.ndarray, xp: Any, others: np.ndarray | None = None) -> None:
        from alloy._vendor import newton  # noqa: PLC0415
        from alloy._vendor.newton.sensors import SensorContact  # noqa: PLC0415

        self.xp, self.feet = xp, int(np.asarray(shapes).shape[-1])
        self.shapes = np.asarray(shapes if others is None else np.concatenate([shapes, others], 1), np.int32)
        if self.shapes.ndim != 2:
            raise EnvError(f"shapes has shape {self.shapes.shape}", "pass (worlds, feet) shape indices",
                           field="contact")
        model, solver = engine.model, self._solver(engine)
        self._check_worlds(model)
        self.sensor = SensorContact(model, sensing_shapes=[int(i) for i in self.shapes.ravel()], measure_total=True,
                                    request_contact_attributes=False)
        if list(self.sensor.sensing_indices) != [int(i) for i in self.shapes.ravel()]:
            raise EnvError("the contact sensor reordered the foot shapes", "report this: FootForces needs rows in "
                           "the order it asked for", field="contact")
        self.contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=model.device)
        self.observables = solver.observables(self.sensor.solver_observable_kinds, contacts=self.contacts)
        self._wrap(solver)
        self._drop_graphs(engine)

    @staticmethod
    def _solver(engine: Any) -> Any:
        """The one rigid solver of the scene, checked for the contact-force export."""
        from alloy._vendor import newton  # noqa: PLC0415

        kind = newton.solvers.SolverBase.ObservableKind.CONTACT_F
        stepper = engine.stepper
        solver = getattr(stepper, "solver", None)
        if solver is None or not hasattr(solver, "get_max_contact_count") or kind not in solver.supported_observables:
            raise EnvError(f"foot contact forces need a scene stepped by one MuJoCo solver; this one is stepped by "
                           f"{type(solver if solver is not None else stepper).__name__}",
                           "use a rigid-only scene (robot and floor), or contact='height'", field="contact")
        if getattr(getattr(stepper, "_alloy_plugin_host", None), "_exported", None) is not None:
            raise EnvError("a plug-in already reads this solver's contacts; FootForces cannot share the export",
                           "use contact='height', or build without plug-ins that need contacts", field="contact")
        return solver

    def _check_worlds(self, model: Any) -> None:
        worlds = model.shape_world.numpy()[self.shapes]
        if not (worlds == np.arange(self.shapes.shape[0])[:, None]).all():
            raise EnvError("foot shapes are not one row per world, in world order",
                           "pass shapes[w, f] = the shape of foot f in world w", field="contact")

    def _wrap(self, solver: Any) -> None:
        """Every step of ``solver`` that brings no contacts of its own exports its contacts and their forces."""
        inner, contacts, observables = solver.step, self.contacts, self.observables

        def step(state_in: Any, state_out: Any, control: Any, step_contacts: Any, dt: float, **kwargs: Any) -> Any:
            if step_contacts is None and "observables" not in kwargs:
                return inner(state_in, state_out, control, contacts, dt, observables=observables, **kwargs)
            return inner(state_in, state_out, control, step_contacts, dt, **kwargs)

        solver.step = step

    @staticmethod
    def _drop_graphs(engine: Any) -> None:
        """Captured CUDA graphs hold the unwrapped step: drop them (the next period is captured again)."""
        from alloy.api._drive import invalidate_graphs  # noqa: PLC0415

        invalidate_graphs(engine)

    def read_vectors(self) -> Any:
        """Net contact force on each foot, world frame, (W, F, 3) N. Example: ``forces.read_vectors()[0, 0]``."""
        return self._all()[:, :self.feet]

    def _all(self) -> Any:
        self.sensor.update(None, self.contacts, observables=self.observables)
        w, n = self.shapes.shape
        return self.xp.view(self.sensor.total_force).reshape(w, n, 3)

    def read_others(self) -> Any:
        """(W, K) newtons: the size of the contact force on each non-foot shape given as ``others`` (K = 0 without).

        Example: ``(forces.read_others() > 1.0).sum(-1)`` -> how many other parts touch something, per world."""
        return self.xp.norm(self._all()[:, self.feet:], axis=-1)

    def read(self, component: str = "z") -> Any:
        """(W, F) newtons: the world-z part of each foot's contact force (``"z"``), or its magnitude (``"norm"``).

        Example: ``forces.read().sum(-1)`` -> each world's weight when the robot stands still."""
        vec = self.read_vectors()
        if component == "z":
            return vec[..., 2]
        if component == "norm":
            return self.xp.norm(vec, axis=-1)
        raise EnvError(f"unknown component {component!r}", "pass 'z' or 'norm'", field="component")


def foot_forces(sim: Any, robot: Any, name: str, xp: Any) -> FootForces:
    """A ``FootForces`` for the feet of ``robot`` (a ``LeggedRobot``) named ``name`` in ``sim``.

    Example: ``LocoConfig(robot=GO2, name="go2", contact=foot_forces(sim, GO2, "go2", xp))``."""
    bodies = np.stack([sim.index.per_world(sim.index.body(f"{name}/{foot.body}"), "body") for foot in robot.feet], 1)
    model = sim.engine.model
    shapes = foot_shapes(model, np.asarray(bodies), [foot.radius for foot in robot.feet])
    parts = [b for n, b in sim.index.bodies.items() if n.startswith(f"{name}/")]
    robot_bodies = np.stack([sim.index.per_world(b, "body") for b in parts], 1)
    return FootForces(sim.engine, shapes, xp, others=other_shapes(model, robot_bodies, shapes))


__all__ = ["FootForces", "foot_forces", "foot_shapes", "other_shapes"]
