# SPDX-License-Identifier: Apache-2.0
"""Joint springs that yield in the Scene API (SCR-WP7-15 P2): ``scene.set_joint_yield`` and the
``"<joint>.rest"`` / ``"<joint>.plastic_strain"`` readouts.

``Scene`` records each yielding joint in ``scene.joint_yield`` (plain data, kept by ``to_json`` / ``from_dict``);
``alloy.build`` resolves the names on the built scene and installs ``alloy.plugins.joint_yield.JointYield`` in the
API's own simulation, so ``run``, ``record``, ``state``, ``reset`` and health rollbacks keep working.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from alloy.api.errors import BuildError, SceneError

PLUGIN = "joint_yield"
_FIX = "change the joint's setting: scene.set_joint_yield(name, torque=... or force=..., hardening=...)"


def _number(value: Any, field: str, *, positive: bool) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v) or (v <= 0.0 if positive else v < 0.0):
        need = "> 0" if positive else ">= 0"
        raise SceneError(f"{field} must be a finite number {need}, got {value!r}", _FIX, field=field)
    return v


class JointYieldMixin:
    """``Scene`` methods for joint springs that yield (mixed into ``alloy.Scene``)."""

    joint_yield: dict[str, dict[str, Any]]

    def set_joint_yield(self, joint: str, *, torque: float | None = None, force: float | None = None,
                        hardening: float = 0.0) -> None:
        """Make a joint's spring yield: past ``torque`` (N·m, a hinge) or ``force`` (N, a slide) it takes a
        permanent set, so the joint rests elsewhere once released (a bent clip, an over-twisted torsion spring).
        ``hardening`` k_h (N·m/rad or N/m, default 0: perfectly plastic) raises the yield by k_h per unit of set.
        The joint needs a spring in its MJCF (``stiffness``, ``springref``). Read where the spring rests now as
        ``"<joint>.rest"`` (springref + set, rad or m) and the accumulated |set| as ``"<joint>.plastic_strain"``
        (after a reversed load the two differ). MuJoCo routes only (GPU and CPU), and not yet beside cloth, liquid or
        soft bodies.

        Example: ``scene.set_joint_yield("clip/arm", torque=0.02)``: a 2 N·m/rad clip arm with springref 0, bent to
        0.3 rad and let go, rests at 0.3 - 0.02 / 2 = 0.29 rad."""
        if not isinstance(joint, str) or not joint:
            raise SceneError(f"joint must be a joint name, got {joint!r}", "pass the name from sim.names()['joints'] "
                             "('<robot>/<joint>')", field="joint")
        if (torque is None) == (force is None):
            raise SceneError("set_joint_yield takes one of torque= (a hinge, N·m) or force= (a slide, N)", _FIX,
                             field="torque")
        kind = "torque" if torque is not None else "force"
        limit = _number(torque if torque is not None else force, kind, positive=True)
        self.joint_yield[joint] = {"kind": kind, "limit": limit,
                                   "hardening": _number(hardening, "hardening", positive=False)}


def plugin_for(scene: Any) -> Callable[[Any], Any] | None:
    """A function of the built scene that makes its joint-yield plug-in, or None when no joint yields.

    Example: ``make = plugin_for(scene); plugin = make(built) if make else None``."""
    record = dict(getattr(scene, "joint_yield", None) or {})
    if not record:
        return None

    def make(built: Any) -> Any:
        from alloy.api.names import NameIndex  # noqa: PLC0415
        from alloy.plugins.joint_yield import JointYield  # noqa: PLC0415

        idx = NameIndex.from_built(built)
        jtype = built.model.joint_type.numpy()
        joints = []
        for name, spec in record.items():
            template = idx.joint(name)                        # UnknownNameError lists the valid names
            _check_kind(name, spec["kind"], int(jtype[template]))
            joints.append({"name": name, "limit": float(spec["limit"]), "hardening": float(spec["hardening"]),
                           "rows": [int(r) for r in idx.per_world(template, "joint")]})
        return JointYield(joints=joints, label=PLUGIN)

    return make


def _check_kind(name: str, kind: str, jtype: int) -> None:
    want = {1: "torque", 0: "force"}.get(jtype)                # newton.JointType.REVOLUTE, PRISMATIC
    if want is None:
        raise BuildError(f"joint {name!r} is not a hinge or slide joint: only one-axis joint springs yield", _FIX,
                         field="joint_yield")
    if kind != want:
        what = "hinge" if want == "torque" else "slide"
        raise BuildError(f"joint {name!r} is a {what}: give its yield as {want}=, not {kind}=", _FIX,
                         field="joint_yield")


def readout(engine: Any, index: Any, joints: list[str]) -> dict[str, np.ndarray]:
    """For the yielding joints among ``joints``: ``"<name>.rest"`` (W,), where the spring rests now (springref + set,
    rad or m), and ``"<name>.plastic_strain"`` (W,), the accumulated |set|; ``{}`` when none yields (syncs the device
    once per call)."""
    host = getattr(engine, "plugins", None)
    plugin = getattr(host, "plugins", {}).get(PLUGIN) if host is not None else None
    if plugin is None:
        return {}
    cols = {n: plugin.columns[index.joints[n]] for n in joints if index.joints[n] in plugin.columns}
    if not cols:
        return {}
    ebar, rest = plugin.ebar.numpy(), plugin.s0.numpy() + plugin.q_p.numpy()
    out: dict[str, np.ndarray] = {}
    for n, c in cols.items():
        out[f"{n}.rest"] = rest[:, c].copy()
        out[f"{n}.plastic_strain"] = ebar[:, c].copy()
    return out
