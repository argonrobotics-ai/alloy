# SPDX-License-Identifier: Apache-2.0
"""Test scenes and row readers for the parked-world laws (``alloy.coupling.parked``, gap #10b P2): a fake-solver batch
with a box mass, position and height per world, a Scheduler over it with the clock options a law needs, and one
world's rows of every array a held copy covers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.core.worldops import joint_coord_world, joint_dof_world
from alloy.coupling.types import AlloyEntry, ClockPolicy, PairPolicy, SchedulerPolicy
from alloy.testing.coupling import BOX_HALF
from alloy.testing.core import FakeSolver

STATE = ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q", "joint_qd")
PAD_X, PAD_Z = 1.0, 0.1          # m: each world's destination pad (the contact gate's "near" is beside it)


def fake_mass_model(masses: Sequence[float], xs: Sequence[float], heights: Sequence[float]) -> newton.Model:
    """``fake_drop_model`` with a box mass [kg], x and height [m] per world (CPU)."""
    top = newton.ModelBuilder()
    for m, x, h in zip(masses, xs, heights, strict=True):
        b = newton.ModelBuilder()
        for label, mass, pos in (("box", m, (x, 0.0, h)), ("pad", 0.2, (PAD_X, 0.0, PAD_Z))):
            cfg = b.default_shape_cfg.copy()
            cfg.density = mass / (2.0 * BOX_HALF) ** 3
            body = b.add_body(xform=wp.transform(p=wp.vec3(*pos), q=wp.quat_identity()), label=label)
            b.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)
        top.add_world(b)
    return top.finalize(device="cpu")


def fake_mass_scheduler(model: newton.Model, *, adaptive: bool = False, stiffness: float | None = None,
                        mode: str = "lagged", capture_while: bool | None = None, **clock: Any) -> Any:
    """``fake_scheduler`` on ``fake_mass_model``: with an adaptive destination (built at rung 64), a stated interface
    stiffness [N/m] or ``ClockPolicy`` fields (``contact_gate=True``) when asked."""
    from alloy.coupling.scheduler import Scheduler

    labels, child = list(model.body_label), model.joint_child.numpy()

    def entry(name: str, suffix: str, floor: float | None) -> Any:
        bodies = [i for i, n in enumerate(labels) if n.endswith(suffix)]
        return SolverCoupledProxy.Entry(
            name=name, solver=lambda v: FakeSolver(v, FakeSolver.Config(floor_z=floor)), bodies=bodies,
            joints=[j for j, b in enumerate(child) if b in set(bodies)], substeps=64 if adaptive else 4)

    src, dst = entry("src", "box", None), entry("dst", "pad", 0.0)
    proxy = SolverCoupledProxy.Proxy(source="src", destination="dst", bodies=list(src.bodies), mode=mode,
                                     proxy_relaxation_mode="aitken" if mode == "staggered" else "fixed")
    pair = PairPolicy("src", "dst", mode=mode, proxy=proxy, interface_stiffness=stiffness)
    return Scheduler(model, [AlloyEntry(src, adaptive=False), AlloyEntry(dst, adaptive=adaptive)], [pair],
                     SchedulerPolicy(clock=ClockPolicy(**clock)), capture_while=capture_while)


def world_rows(model: newton.Model, state: Any, sched: Any, world: int) -> dict[str, bytes]:
    """World ``world``'s rows of every State array (``state.<name>``) and of the stepper's ``state_arrays()`` (the
    coupling carry, the watchdog, and the solvers' adapters under ``<entry>.``), as bytes."""
    maps = {"body": model.body_world, "particle": model.particle_world, "joint_q": joint_coord_world(model),
            "joint_qd": joint_dof_world(model)}
    out = {}
    for name in STATE:
        arr = getattr(state, name, None)
        if arr is not None and arr.shape[0]:
            sel = maps[name if name.startswith("joint") else name.split("_")[0]].numpy() == world
            out[f"state.{name}"] = arr.numpy()[sel].tobytes()
    world_maps = sched.state_world_maps()
    for name, arr in sched.state_arrays().items():
        rows = arr.numpy()
        out[name] = (rows[world_maps[name].numpy() == world] if name in world_maps else rows[world]).tobytes()
    return out


__all__ = ["STATE", "fake_mass_model", "fake_mass_scheduler", "world_rows"]
