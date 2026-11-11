# SPDX-License-Identifier: Apache-2.0
"""SCR-WP8-7 M1: a MuJoCo entry's CCD scratch pool sized per world (``alloy:ccd_slots``), and its overflow in health.

Boxes resting 1 mm into a static cylinder collide through MuJoCo Warp's CCD path: cylinder-box is a CONVEX pair, and a
penetrating pair takes one EPA slot per step. A capsule or sphere would not: MuJoCo Warp collides it as a segment or
point with a radius, so a shallow overlap needs no EPA (CPU).
"""

from __future__ import annotations

import dataclasses

import pytest

newton = pytest.importorskip("alloy._vendor.newton")
wp = pytest.importorskip("warp")
pytest.importorskip("alloy._vendor.mujoco_warp")

from alloy._vendor.mujoco_warp import OverflowType  # noqa: E402

from alloy.precision.health import HealthMonitor  # noqa: E402
from alloy.precision.mjw_ccd import CCD_KEY, set_ccd_slots, with_ccd_slots  # noqa: E402
from alloy.precision.types import CheckPolicy, HealthFlags  # noqa: E402
from alloy.scene.builtin_plugins import _mujoco  # noqa: E402

WORLDS, BOXES, NCONMAX = 2, 2, 16


def _boxes_on_cylinder() -> newton.Model:
    b = newton.ModelBuilder()
    b.add_shape_cylinder(-1, radius=0.05, half_height=0.2,
                         xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0),
                                                                                             1.5707963)))
    for i in range(BOXES):
        body = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.1 * i - 0.05, 0.059), wp.quat_identity()), mass=0.05)
        b.add_shape_box(body, hx=0.03, hy=0.01, hz=0.01)
        b.add_articulation([b.add_joint_free(body)])
    s = newton.ModelBuilder()
    s.replicate(b, WORLDS)
    return s.finalize(device="cpu")


def _solver(model, **kw):
    with wp.ScopedDevice("cpu"):
        return _mujoco(model, {"use_mujoco_contacts": True, "nconmax": NCONMAX, **kw})


def test_the_entry_sizes_the_pool_per_world_and_no_kwarg_keeps_the_default() -> None:
    model = _boxes_on_cylinder()
    assert int(_solver(model).mjw_data.naccdmax) == NCONMAX * WORLDS              # MuJoCo Warp's default
    assert int(_solver(model, **{CCD_KEY: 3}).mjw_data.naccdmax) == 3 * WORLDS
    assert int(_solver(model, **{CCD_KEY: 10_000}).mjw_data.naccdmax) == NCONMAX * WORLDS   # capped at naconmax
    with pytest.raises(ValueError, match="slot per world"):
        set_ccd_slots(_solver(model), 0)


def _step_and_check(slots: int):
    model = _boxes_on_cylinder()
    solver = _solver(model, **{CCD_KEY: slots})
    s0, s1, control = model.state(), model.state(), model.control()
    monitor = HealthMonitor(model, CheckPolicy(), solvers=[solver])
    solver.step(s0, s1, control, None, 1.0e-3)
    monitor.update(0, s0, s1, None)
    ccd = (solver.mjw_data.overflow.numpy() & int(OverflowType.CCD)) != 0
    return ccd, monitor.poll()


def test_a_full_pool_overflows_into_contact_overflow_and_a_sized_one_does_not() -> None:
    ccd, report = _step_and_check(slots=BOXES)                  # room for every box in every world
    assert not ccd.any()
    assert not any(int(f) & int(HealthFlags.CONTACT_OVERFLOW) for f in report.flags)
    ccd, report = _step_and_check(slots=1)                         # 2 slots for 4 CCD pairs
    assert ccd.any()
    for w in range(WORLDS):
        assert bool(int(report.flags[w]) & int(HealthFlags.CONTACT_OVERFLOW)) == bool(ccd[w]), (w, ccd)
    assert set(report.invalid_worlds) == {w for w in range(WORLDS) if ccd[w]}


@dataclasses.dataclass(frozen=True)
class _Entry:
    capability: str
    solver_kwargs: dict


def test_with_ccd_slots_marks_only_mujoco_entries_and_keeps_an_entry_s_own_value(monkeypatch) -> None:
    from types import SimpleNamespace

    from alloy.integration import built_copy

    monkeypatch.setattr(built_copy, "replace_built", lambda built, entries: entries)
    entries = [_Entry("mujoco", {"nconmax": 8}), _Entry("mujoco", {CCD_KEY: 5}), _Entry("vbd", {})]
    got = [e.solver_kwargs for e in with_ccd_slots(SimpleNamespace(entries=entries), 32)]
    assert got == [{CCD_KEY: 32, "nconmax": 8}, {CCD_KEY: 5}, {}]
