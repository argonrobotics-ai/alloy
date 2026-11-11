# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD F2: the coupled ModelView memoises its resolved attributes (vendored patch
alloy/_vendor/patches/0004-model-view-memo.patch; coordinator ruling 2026-11-08 15:58Z: bitwise law).

A solver on a view reads its model attributes in every kernel launch. Resolving one (attribute spec, frequency count,
slice) cost about 1/3 of a VBD rod entry's step on the CPU beside a robot (docs/notes/wp9-results-rod.md, F2).
- The memo reuses a value only while the raw object it came from (the override, else the parent's attribute) is the
  same one: a rebound parent attribute, a new or deleted override, and a count change all resolve afresh.
- A cable squeezed by two pads on a MuJoCo wrist (a coupled VBD rod entry, j = 6) steps bit for bit the same with the
  memo and with it switched off (every view's memo replaced by one that never keeps a value).
"""

from __future__ import annotations

import gc

import numpy as np
import pytest
import warp as wp

import alloy
from alloy._vendor import newton
from alloy._vendor.newton._src.solvers.coupled.model_view import ModelView
from alloy.api import _rods

WRIST = """<mujoco><compiler angle="radian"/><worldbody><body name="wrist" pos="0.388 0 0.05">
<joint name="roll" type="hinge" axis="1 0 0" damping="0.02"/><inertial pos="0 0 0" mass="0.3" diaginertia="2e-4 2e-4 2e-4"/>
<body name="left" pos="0 0.013 0"><joint name="lf" type="slide" axis="0 -1 0" range="0 0.02" damping="1"/>
<geom type="box" size="0.01 0.004 0.01" mass="0.2" friction="1.0"/></body>
<body name="right" pos="0 -0.013 0"><joint name="rf" type="slide" axis="0 1 0" range="0 0.02" damping="1"/>
<geom type="box" size="0.01 0.004 0.01" mass="0.2" friction="1.0"/></body></body></worldbody>
<actuator><position joint="roll" kp="20" kv="0.5"/><position joint="lf" kp="300" kv="5"/><position joint="rf" kp="300" kv="5"/></actuator></mujoco>"""


class _NoMemo(dict):
    """A memo that never keeps a value: the view resolves every read as before the patch."""

    def __setitem__(self, key, value) -> None:
        return None


def _view() -> tuple[newton.Model, ModelView]:
    b = newton.ModelBuilder()
    for x in (0.0, 1.0):
        body = b.add_body(xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()), mass=1.0)
        b.add_shape_sphere(body, radius=0.1)
    model = b.finalize(device="cpu")
    return model, ModelView(model, "probe")


def test_the_memo_resolves_afresh_whenever_the_raw_object_changes():
    model, view = _view()
    first = view.body_mass
    assert view.body_mass is first                                      # memoised
    rebound = wp.array([2.0, 3.0], dtype=float, device="cpu")
    model.body_mass = rebound                                           # the parent rebinds the attribute
    assert view.body_mass is rebound
    own = wp.array([5.0, 7.0], dtype=float, device="cpu")
    view.body_mass = own                                                # an override
    assert view.body_mass is own
    del view.body_mass                                                  # back to the parent's
    assert view.body_mass is rebound
    assert object.__getattribute__(view, "_memo")
    view.body_count = 1                                                 # a count change clears it
    assert not object.__getattribute__(view, "_memo")
    assert view.body_mass.shape[0] == 1                                 # sliced to the new count
    view.body_count = 2
    assert view.body_mass is rebound
    for field, value in (("custom_frequency_counts", dict(model.custom_frequency_counts)),
                         ("rigid_contact_max", 16)):
        assert view.body_mass is rebound                                # memoised again
        setattr(view, field, value)                                     # a count source that does not end in _count
        assert not object.__getattribute__(view, "_memo"), field


def test_a_swapped_colour_group_override_reaches_a_prefix_limited_view():
    """AGENTS-5: a view that holds a prefix of the parent's bodies filters the colour groups to its ids and kept one
    filtered result per name and count, so a swapped override (SolverVBDRod, twice per iteration) read as the first
    list. An ADMM entry's view is such a prefix: a clamped rod beside a chain rod raised IndexError in the VBD sweep
    (two colours of bodies against one block-dim row). Each raw list now has its own result."""
    model, view = _view()
    ids = [wp.array([k], dtype=int, device="cpu") for k in (0, 1)]
    model.body_color_groups = [ids[0], ids[1]]
    view.body_count = 1                                                 # the view's bodies: the parent's first one
    both = view.body_color_groups
    assert [g.numpy().tolist() for g in both] == [[0], []]              # filtered to the view's ids
    assert view.body_color_groups is both                               # and kept
    other = [wp.zeros(0, dtype=int, device="cpu")]
    view.body_color_groups = other                                      # the swap
    assert view.body_color_groups is other                              # nothing to filter: itself (was `both`)
    view.body_color_groups = [ids[1], ids[0], ids[1]]
    assert [g.numpy().tolist() for g in view.body_color_groups] == [[], [0], []]
    del view.body_color_groups                                          # back to the parent's list
    assert [g.numpy().tolist() for g in view.body_color_groups] == [[0], []]


def _run(no_memo: bool) -> dict[str, np.ndarray]:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_rod("cable", length=0.4, radius=0.003, pos=(0.0, 0.0, 0.05), segments=20, material="cable_usb_composite")
    sc.add_robot("hand", mjcf=WRIST)
    sim = alloy.build(sc, device="cpu")
    assert sim.routes()["cable"]["route"] == "rod_vbd" and sim.engine.clock.j == 6
    views = [o for o in gc.get_objects() if isinstance(o, ModelView)]
    assert views
    if no_memo:
        for v in views:
            object.__setattr__(v, "_memo", _NoMemo())
    sim.set_targets({"lf": 0.02, "rf": 0.02})
    sim.step(4)
    st = sim.state(["cable", "hand/left", "hand/right"], world=0)
    out = {k: np.array(v, copy=True) for k, v in st.items()}
    assert sim.check()["ok"]
    sim.close()
    return out


@pytest.mark.law
@pytest.mark.timeout(600)
def test_a_rod_beside_a_robot_steps_bit_for_bit_with_and_without_the_memo(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")                         # the CPU route as shipped (MuJoCo C)
    monkeypatch.setattr(_rods, "DEFAULT_ROUTE", "rod_vbd")               # the short cable on the elastic rod
    memo = _run(no_memo=False)
    gc.collect()
    plain = _run(no_memo=True)
    assert memo.keys() == plain.keys()
    for k in memo:
        assert np.array_equal(memo[k], plain[k]), k
