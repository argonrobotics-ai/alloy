# SPDX-License-Identifier: Apache-2.0
"""Joint sliders in a viewer's panel: one slider per joint that a position actuator drives, each setting that joint's
target in every world (what ``sim.set_targets({name: value})`` does). Turn them on with
``alloy.viewer.launch(sim, joints=True)``.

A slider starts at the joint's current target (its position in world 0 for a servo whose target the API does not
expose) and runs over the joint's range; a joint without a finite range gets +-pi (hinge) or +-1 m (slide) around
its start. Joints that no actuator drives get no slider, and at most 32 sliders are shown. Nothing is sent to the
simulation until a slider is moved, so a viewer whose sliders are left alone steps the same simulation, bit for bit.
A ``sim.set_targets`` made by the program after the launch does not move the sliders.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from alloy.api.errors import AlloyAPIError

log = logging.getLogger("alloy.viewer")

MAX_SLIDERS = 32
TITLE = "Joint targets (all worlds)"
UNLIMITED = 1e5               # the engine stores "no limit" as +-1e10
HINGE_SPAN, SLIDE_SPAN = math.pi, 1.0
REVOLUTE, PRISMATIC = 1, 0    # joint type codes of the engine's model


@dataclass
class Slider:
    """One joint's slider: the joint's name, the label shown, its range and the value shown."""

    name: str
    label: str
    lo: float
    hi: float
    value: float


class JointSliders:
    """The joint-slider tool (``alloy.viewer._tools`` interface); built by ``launch(joints=True)``."""

    def __init__(self) -> None:
        self.sliders: list[Slider] = []
        self._pending: dict[str, float] = {}

    def attach(self, view: Any) -> None:
        self.sliders = find_sliders(view.sim)
        if not self.sliders:
            log.info("joint sliders: no joint of this simulation is driven by a position actuator")

    def render(self, ui: Any) -> None:
        """Draw the sliders and record the ones the user moved."""
        ui.text(TITLE)
        for s in self.sliders:
            changed, value = ui.slider_float(s.label, s.value, s.lo, s.hi)
            if changed:
                s.value = self._pending[s.name] = float(value)

    def apply(self, view: Any) -> None:
        """Send the recorded targets with one ``set_targets`` call (nothing when no slider was moved)."""
        if not self._pending:
            return
        asked, self._pending = self._pending, {}
        try:
            view.sim.set_targets(asked)
        except AlloyAPIError:
            self._apply_one_by_one(view, asked)

    def draw(self, view: Any) -> None:
        pass

    def detach(self, view: Any) -> None:
        pass

    def _apply_one_by_one(self, view: Any, asked: dict[str, float]) -> None:
        """A joint was refused: find which, warn once and drop its slider; the others still get their targets."""
        for name, value in asked.items():
            try:
                view.sim.set_targets({name: value})
            except AlloyAPIError as exc:
                log.warning("joint slider for %s removed: the simulation refused its target (%s)", name, exc)
                self.sliders = [s for s in self.sliders if s.name != name]


def find_sliders(sim: Any) -> list[Slider]:
    """Sliders for the position-driven joints of ``sim``, qualified names first, at most ``MAX_SLIDERS``."""
    io = sim._io      # ranges, actuator table and target rows have no public reader (names()/state() give names, q)
    names = sorted(sim.names()["joints"], key=lambda n: "/" not in n)       # stable: "robot/joint" before its alias
    seen: set[int] = set()
    found: list[Slider] = []
    for name in names:
        joint = io.index.joints[name]
        if joint in seen:
            continue
        seen.add(joint)
        slider = _slider_for(sim, io, name, joint)
        if slider is not None:
            found.append(slider)
    if len(found) > MAX_SLIDERS:
        log.info("joint sliders: showing the first %d of %d driven joints", MAX_SLIDERS, len(found))
    return found[:MAX_SLIDERS]


def _slider_for(sim: Any, io: Any, name: str, joint: int) -> Slider | None:
    kind = int(sim.engine.model.joint_type.numpy()[joint])
    driven = joint in io.acts.joint_target or joint in io.acts.position
    if kind not in (REVOLUTE, PRISMATIC) or not driven:
        return None
    start = _start_value(sim, io, name, joint)
    dof = int(io.qd_start[joint])
    lo, hi = float(io.limits.lower[dof]), float(io.limits.upper[dof])
    if not (abs(lo) < UNLIMITED and abs(hi) < UNLIMITED and lo < hi):
        span = HINGE_SPAN if kind == REVOLUTE else SLIDE_SPAN
        lo, hi = start - span, start + span
    return Slider(name, name, lo, hi, min(max(start, lo), hi))


def _start_value(sim: Any, io: Any, name: str, joint: int) -> float:
    """The joint's current target when the engine keeps it as a joint target, else its position in world 0."""
    if joint in io.acts.joint_target:
        target = io.engine.control.joint_target_q.numpy()
        return float(target[io.target_start[io.index.per_world(joint, "joint")[0]]])
    return float(sim.state([name], world=0)[f"{name}.q"])
