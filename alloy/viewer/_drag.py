# SPDX-License-Identifier: Apache-2.0
"""Drag a body with the mouse (``alloy.viewer.launch(sim, drag=True)``): grab a body in the viewer and pull it with a
spring. The pull is held on the body as ``sim.apply_wrench`` (a force at the body's centre of mass and the torque of its
offset), recomputed every frame from the body's pose and velocity; letting go removes it. While nothing is grabbed the
tool calls nothing on the simulation, so the run is bit for bit the one without it.

**How to grab.** Viser: tick "Enable Picking" (on by default), click a body, drag the handle that appears; releasing the
handle lets go. GL window: press and hold the right mouse button on a body and move the mouse; releasing lets go. Other
backends (rerun, usd, null) have no mouse: use :meth:`MouseDrag.grab`.

**The pull.** ``F = k (target - attach) - c v`` at the grabbed point, ``k = m w^2``, ``c = 2 m w`` (critical damping,
``w = 2 pi 2 Hz``): the body hangs ``g / w^2`` = 6.2 cm below a still target. Capped at 50 body weights, and at 200 N on
a robot link. Only bodies Alloy can push: free bodies, robot links, rod segments of the chain route. Static and
kinematic objects, and the particles of cloth, liquid or soft bodies, cannot be grabbed (one info line, the grab is let
go). The tool sets the wrench of the grabbed body in every world (zero in the others), so do not mix it with your own
``apply_wrench`` on that body.

**Frames.** The viewer shifts worlds on a grid; Newton's picking already removes that shift, so grabs and targets
here are in the grabbed world's own simulation coordinates (the line is drawn with the shift added back).

Example::

    with alloy.viewer.launch(sim, backend="null", drag=True) as view:
        drag = alloy.viewer.MouseDrag.of(view)
        drag.grab(view, "block", world=0, target_xyz=(0, 0, 0.8))
        view.run(seconds=1.0, realtime=False)
        drag.release()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.api.errors import AlloyAPIError, SceneError
from alloy.viewer import _drag_force as dforce

log = logging.getLogger("alloy.viewer")

LINE = "/alloy/drag/line"
LINE_COLOR = (0.0, 1.0, 1.0)
HOWTO = {
    "viser": 'Drag: tick "Enable Picking", click a body, drag the handle; release the handle to let go.',
    "gl": "Drag: hold the right mouse button on a body and move the mouse; release to let go.",
}
NO_MOUSE = "Drag: this backend has no mouse; use MouseDrag.grab(view, name, world, target)."


@dataclass
class Grab:
    """What is grabbed: Newton body row ``body`` (Alloy ``name`` in ``world``), the grabbed point ``local`` (body
    frame) and the ``target`` (that world's simulation frame); ``mouse`` says who set it."""

    body: int
    name: str
    world: int
    local: np.ndarray
    target: np.ndarray
    mass: float
    cap: float
    mouse: bool = False


@dataclass(frozen=True)
class _Facts:
    """What a body row is, resolved once: its name and world, mass, force cap (None name: it cannot be pushed)."""

    name: str | None
    world: int = 0
    mass: float = 0.0
    cap: float = 0.0


class MouseDrag:
    """The mouse-drag tool (module docstring). ``MouseDrag.of(view)`` finds the one a ``drag=True`` viewer made."""

    def __init__(self) -> None:
        self._program: Grab | None = None
        self._held: str | None = None           # name of the body that carries our wrench
        self._facts: dict[int, _Facts] = {}
        self._hint = NO_MOUSE
        self._line_on = False

    @classmethod
    def of(cls, view: Any) -> MouseDrag:
        """The drag tool of ``view`` (made with ``launch(sim, drag=True)``)."""
        for tool in view.tools:
            if isinstance(tool, cls):
                return tool
        raise SceneError("this viewer has no drag tool", "launch it with alloy.viewer.launch(sim, drag=True)")

    # --- the tool interface -------------------------------------------------------------------------------------

    def attach(self, view: Any) -> None:
        picking = getattr(view.backend, "picking", None)
        if picking is None:
            log.info("drag: backend %s has no mouse picking; MouseDrag.grab() still works", view.name)
            return
        self._hint = HOWTO.get(view.name, NO_MOUSE)
        if hasattr(view.backend, "picking_enabled"):
            view.backend.picking_enabled = True

    def render(self, ui: Any) -> None:
        ui.text(self._hint)

    def apply(self, view: Any) -> None:
        grab = self._current(view)
        if grab is None:
            self._let_go(view)
            return
        try:
            self._pull(view, grab)
        except AlloyAPIError as exc:
            log.warning("drag: %s; grab dropped", exc)
            self.release()
            self._held = None
            self._drop_mouse(view)

    def draw(self, view: Any) -> None:
        grab = self._current(view)
        if grab is None or (grab.mouse and _draws_own_line(view.backend)):
            if self._line_on:
                view.backend.log_lines(LINE, None, None, None)
                self._line_on = False
            return
        _draw_line(view, grab, self._attach_point(view, grab))
        self._line_on = True

    def detach(self, view: Any) -> None:
        self._program = None
        self._let_go(view)

    # --- programmatic grabs -------------------------------------------------------------------------------------

    def grab(self, view: Any, name: str, world: int, target_xyz: Any, attach_xyz: Any = None) -> bool:
        """Grab body ``name`` of ``world`` and pull ``attach_xyz`` (default: its centre of mass; that world's frame)
        toward ``target_xyz`` from the next frame on: the same force as the mouse's. Call again to move the target.
        False, with one info line, for a body that cannot be pushed. Example: ``drag.grab(view, "block", 0, (0, 0, 1))``."""
        row = _row_of(view.sim, name, world)
        facts = self._facts_of(view, row) if row is not None else _Facts(None)
        if facts.name is None:
            log.info("drag: %r in world %s cannot be grabbed (static, kinematic or not a body)", name, world)
            return False
        pose = _read_pose(view.sim, row)
        point = pose_com_world(pose) if attach_xyz is None else np.asarray(attach_xyz, dtype=np.float64)
        self._program = Grab(row, name, world, dforce.to_body_frame(pose, point),
                             np.asarray(target_xyz, dtype=np.float64), facts.mass, facts.cap)
        return True

    def release(self) -> None:
        """Let go of a programmatic grab (the force is removed at the next frame, or by ``view.step``)."""
        self._program = None

    # --- internals ----------------------------------------------------------------------------------------------

    def _current(self, view: Any) -> Grab | None:
        picking = getattr(view.backend, "picking", None)
        if picking is not None and picking.is_picking():
            return self._from_mouse(view, picking)
        return self._program

    def _from_mouse(self, view: Any, picking: Any) -> Grab | None:
        body = int(picking.pick_body)
        facts = self._facts_of(view, body)
        if facts.name is None:
            picking.release()
            return None
        self._program = None
        local = np.array(picking._pick_state.numpy()[0]["picked_point_local"], dtype=np.float64)
        target = np.array(picking.pick_target, dtype=np.float64)
        return Grab(body, facts.name, facts.world, local, target, facts.mass, facts.cap, mouse=True)

    def _facts_of(self, view: Any, row: int) -> _Facts:
        if row not in self._facts:
            self._facts[row] = _resolve(view.sim, row)
            if self._facts[row].name is None:
                log.info("drag: body row %d cannot be pushed (static, kinematic or not an Alloy body); not grabbed", row)
        return self._facts[row]

    def _pull(self, view: Any, grab: Grab) -> None:
        sim = view.sim
        pose = _read_pose(sim, grab.body)
        wrench = dforce.spring_wrench(pose, grab.local, grab.target, grab.mass, grab.cap)
        if self._held not in (None, grab.name):
            self._clear(sim, self._held)
        force, torque = np.zeros((sim.worlds, 3)), np.zeros((sim.worlds, 3))
        force[grab.world], torque[grab.world] = wrench.force, wrench.torque
        sim.apply_wrench(grab.name, force=force, torque=torque)
        self._held = grab.name
        if grab.mouse:
            _tell_picking(view.backend.picking, wrench.attach)

    def _let_go(self, view: Any) -> None:
        if self._held is not None:
            self._clear(view.sim, self._held)
            self._held = None

    @staticmethod
    def _clear(sim: Any, name: str) -> None:
        zeros = np.zeros((sim.worlds, 3))
        sim.apply_wrench(name, force=zeros, torque=zeros)

    def _drop_mouse(self, view: Any) -> None:
        picking = getattr(view.backend, "picking", None)
        if picking is not None:
            picking.release()

    def _attach_point(self, view: Any, grab: Grab) -> np.ndarray:
        return dforce.spring_wrench(_read_pose(view.sim, grab.body), grab.local, grab.target, grab.mass, grab.cap).attach


# --- reading the simulation ------------------------------------------------------------------------------------


def _row_of(sim: Any, name: str, world: int) -> int | None:
    template = sim.index.bodies.get(name)
    if template is None or not 0 <= int(world) < sim.worlds:
        return None
    return int(sim.index.per_world(template, "body")[int(world)])


def _resolve(sim: Any, row: int) -> _Facts:
    """Name, world, mass and cap of body row ``row``; ``_Facts(None)`` when Alloy cannot push it."""
    from alloy.viewer import _contacts  # noqa: PLC0415

    model, names = sim.engine.model, _contacts._shape_names(sim)
    if row < 0:                                     # the ground or a static shape: no body row
        return _Facts(None)
    world = int(names.body_world[row]) if row < names.body_world.size else -1
    template = _contacts._template(row, names.body_world, names.stride[1])
    name = names.body_name.get(template)
    if world < 0 or name is None or sim.index.bodies.get(name) != template:
        return _Facts(None)
    if _is_kinematic(model, row):
        return _Facts(None)
    mass = float(model.body_mass.numpy()[row])
    return _Facts(name, world, mass, dforce.force_cap(mass, _articulated(model, row)))


def _is_kinematic(model: Any, row: int) -> bool:
    from alloy._vendor.newton import BodyFlags  # noqa: PLC0415

    flags = getattr(model, "body_flags", None)
    return flags is not None and bool(int(flags.numpy()[row]) & int(BodyFlags.KINEMATIC))


def _articulated(model: Any, row: int) -> bool:
    """The body is a link of a chain of more than one body (a robot), so its pull moves more than its own mass."""
    if not int(getattr(model, "articulation_count", 0)):
        return False
    child = model.joint_child.numpy()
    joints = np.nonzero(child == row)[0]
    if not joints.size:
        return False
    art = int(model.joint_articulation.numpy()[joints[0]])
    if art < 0:
        return False
    return int(model.articulation_end.numpy()[art] - model.articulation_start.numpy()[art]) > 1


def _read_pose(sim: Any, row: int) -> dforce.BodyPose:
    """Body ``row``'s pose and velocity from the engine's current state, as host copies (read only)."""
    state, model = sim.engine.state, sim.engine.model
    q = np.array(state.body_q.numpy()[row], dtype=np.float64, copy=True)
    qd = np.array(state.body_qd.numpy()[row], dtype=np.float64, copy=True)
    com = np.array(model.body_com.numpy()[row], dtype=np.float64, copy=True)
    return dforce.BodyPose(q[:3], q[3:7], qd[:3], qd[3:6], com)


def pose_com_world(pose: dforce.BodyPose) -> np.ndarray:
    """The centre of mass of ``pose`` in the world frame."""
    return pose.pos + dforce.rotate(pose.quat, pose.com)


# --- drawing and Newton's picking object -----------------------------------------------------------------------


def _draws_own_line(backend: Any) -> bool:
    """GL and viser draw a cyan line from the pick point to its target themselves (we keep its start current)."""
    return hasattr(backend, "_render_picking_line")


def _tell_picking(picking: Any, attach: np.ndarray) -> None:
    """Keep Newton's picked point on the body (it only moves it when Newton applies its own force), so the line the
    viewer draws starts where the body is."""
    try:
        picking._set_pick_state_field("picked_point_world", np.asarray(attach, dtype=np.float32))
    except (AttributeError, ValueError):
        pass


def _draw_line(view: Any, grab: Grab, attach: np.ndarray) -> None:
    import warp as wp  # noqa: PLC0415

    backend = view.backend
    shift = _world_shift(backend, grab.world)
    device = backend.device
    start = wp.array([wp.vec3(*(attach + shift))], dtype=wp.vec3, device=device)
    end = wp.array([wp.vec3(*(grab.target + shift))], dtype=wp.vec3, device=device)
    color = wp.array([wp.vec3(*LINE_COLOR)], dtype=wp.vec3, device=device)
    backend.log_lines(LINE, start, end, color)


def _world_shift(backend: Any, world: int) -> np.ndarray:
    offsets = getattr(backend, "world_offsets", None)
    if offsets is None or offsets.shape[0] <= world:
        return np.zeros(3)
    return np.array(offsets.numpy()[world], dtype=np.float64)


__all__ = ["Grab", "MouseDrag"]
