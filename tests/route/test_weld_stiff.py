# SPDX-License-Identifier: Apache-2.0
"""alloy.route.weld_stiff: the row formula against MuJoCo C, and which rows the stiff rule touches (GAP-W)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.route.weld_stiff import (DEFAULT_SOLIMP, DEFAULT_SOLREF, RowParams, impedance, rest_stretch,
                                    row_compliance)

MASS, FORCE, STEP = 2.0, 500.0, 1.25e-3
ROWS = {"default": RowParams(),
        "timeconst": RowParams((0.005, 1.0), (0.99, 0.99, 0.001, 0.5, 2.0)),
        "refsafe": RowParams((0.001, 1.0), DEFAULT_SOLIMP),                    # held at 2 h by MuJoCo
        "direct": RowParams((-230400.0, -960.0), (0.999, 0.999, 0.001, 0.5, 2.0))}


def _mujoco_stretch(p: RowParams) -> float:
    """A free body welded to the world, pulled sideways with gravity off, in plain MuJoCo C."""
    ref, imp = " ".join(map(str, p.solref)), " ".join(map(str, p.solimp))
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><option timestep="{STEP}" gravity="0 0 0"/><worldbody>
<body name="b" pos="0 0 1"><freejoint/><geom type="sphere" size="0.05" mass="{MASS}"/></body></worldbody>
<equality><weld body1="world" body2="b" solref="{ref}" solimp="{imp}"/></equality></mujoco>""")
    data = mujoco.MjData(model)
    data.xfrc_applied[1, 1] = FORCE
    for _ in range(int(2.0 / STEP)):
        mujoco.mj_step(model, data)
    return float(np.linalg.norm(data.xpos[1] - np.array([0.0, 0.0, 1.0])))


@pytest.mark.parametrize("name", list(ROWS))
def test_rest_stretch_is_mujocos(name: str) -> None:
    predicted = rest_stretch(ROWS[name], 1.0 / MASS, FORCE, step_s=STEP)
    assert _mujoco_stretch(ROWS[name]) == pytest.approx(predicted, rel=2e-3, abs=0.0)


def test_impedance_ends_and_midpoint() -> None:
    assert impedance(DEFAULT_SOLIMP, 0.0) == pytest.approx(0.9)
    assert impedance(DEFAULT_SOLIMP, 0.0005) == pytest.approx(0.925)
    assert impedance(DEFAULT_SOLIMP, 0.01) == pytest.approx(0.95)


def test_default_row_is_2_mm_per_kn_on_a_unit_inverse_weight() -> None:
    assert row_compliance(RowParams(DEFAULT_SOLREF, DEFAULT_SOLIMP), 1.0) == pytest.approx(2.0e-5, rel=1e-9)


class _Array:
    """Stand-in for a Warp array: numpy() copies out, assign() copies in."""

    def __init__(self, values) -> None:
        self.values = np.array(values, dtype=np.float64)

    def numpy(self) -> np.ndarray:
        return self.values.copy()

    def assign(self, values) -> None:
        self.values = np.array(values, dtype=np.float64)


class _Model:
    """Four equality rows: default weld, authored weld, default connect, default joint equality (type 2)."""

    def __init__(self) -> None:
        authored = (0.005, 1.0)
        self.body_mass = _Array([5.0, 20.0])
        self.mujoco = type("NS", (), {})()
        ns = self.mujoco
        ns.equality_constraint_count = 4
        ns.equality_constraint_type = _Array([1, 1, 0, 2])
        ns.equality_constraint_body1 = _Array([0, 0, -1, 0])
        ns.equality_constraint_body2 = _Array([1, 1, 1, 1])
        ns.eq_solref = _Array([DEFAULT_SOLREF, authored, DEFAULT_SOLREF, DEFAULT_SOLREF])
        ns.eq_solimp = _Array([DEFAULT_SOLIMP] * 4)


def test_only_default_weld_and_connect_rows_are_stiffened() -> None:
    from alloy.route.weld_stiff import stiffen_default_rows
    from alloy.scene.weld_impedance import WELD_IMP_CAP, WELD_OMEGA_H

    model = _Model()
    done = stiffen_default_rows(model, STEP)
    omega = WELD_OMEGA_H / STEP
    assert done.rows == (0, 2) and done.step_s == STEP
    solref, solimp = model.mujoco.eq_solref.numpy(), model.mujoco.eq_solimp.numpy()
    for e in (0, 2):
        assert solref[e] == pytest.approx((-omega * omega, -2.0 * omega)) and solimp[e, 0] == solimp[e, 1] == WELD_IMP_CAP
    assert tuple(solref[1]) == (0.005, 1.0) and tuple(solref[3]) == DEFAULT_SOLREF      # authored; not a weld
    assert done.rules[0].m_eff == pytest.approx(4.0) and done.rules[1].m_eff == pytest.approx(20.0)
    assert stiffen_default_rows(model, STEP).count == 0                                  # nothing left at defaults


def test_a_model_without_equalities_is_left_alone() -> None:
    from alloy.route.weld_stiff import stiffen_default_rows

    assert stiffen_default_rows(type("M", (), {"mujoco": None})(), STEP).count == 0


def test_the_mode_check_and_the_default() -> None:
    from alloy.route import weld_stiff

    class _Err(Exception):
        def __init__(self, message: str, fix: str, field: str) -> None:
            super().__init__(message)
            self.fix, self.field = fix, field

    other = "stiff" if weld_stiff.DEFAULT_WELDS == "mujoco" else "mujoco"
    for ok in (None, "stiff", "mujoco"):
        weld_stiff.check(ok, None, _Err)
    weld_stiff.check(None, 4, _Err)                                           # workers build with the default
    weld_stiff.check(weld_stiff.DEFAULT_WELDS, 4, _Err)
    with pytest.raises(_Err, match="not a weld setting") as bad:
        weld_stiff.check("rigid", None, _Err)
    assert bad.value.field == "welds" and "stiff" in bad.value.fix
    with pytest.raises(_Err, match="worker processes"):
        weld_stiff.check(other, 4, _Err)
    with weld_stiff.requested("mujoco"):
        assert weld_stiff.apply(type("B", (), {"model": _Model(), "rigid_dt": STEP})()) is None
    with weld_stiff.requested("stiff"):
        assert weld_stiff.apply(type("B", (), {"model": _Model(), "rigid_dt": STEP})()).rows == (0, 2)
    with weld_stiff.requested(None):                                          # the default is its mode
        done = weld_stiff.apply(type("B", (), {"model": _Model(), "rigid_dt": STEP})())
        assert (done is None) == (weld_stiff.DEFAULT_WELDS == "mujoco")
