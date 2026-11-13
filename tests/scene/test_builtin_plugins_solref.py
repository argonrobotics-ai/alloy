# SPDX-License-Identifier: Apache-2.0
"""SCR-SC6-2: the MuJoCo entry rejects a solref-type value Newton read as dampratio 0 (MJCF one-value shorthand, which
MuJoCo reads as (t, 1)), naming the joint/geom; explicit, direct and unauthored values pass (CPU). SAMPLE-CLASSIC C3: a
written dampratio 0 is refused with its own fix (the direct form), not the generic "no solver yet" one."""

from __future__ import annotations

import re
import warnings

import pytest

from alloy._vendor import newton
from alloy.core.errors import RouteError
import alloy
from alloy.scene.builtin_plugins import SOLREF_DAMPRATIO, SOLREF_FIX, _mujoco, check_solref


def _model(joint: str = "", geom: str = "") -> newton.Model:
    xml = f"""<mujoco><worldbody><body pos="0 0 1">
      <joint name="j" type="hinge" limited="true" range="-1 1" {joint}/>
      <geom name="g" type="sphere" size="0.1" mass="1" {geom}/>
    </body></worldbody></mujoco>"""
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # Newton's own warning on the zero dampratio
        b.add_mjcf(xml)
        return b.finalize(device="cpu")


@pytest.mark.parametrize(("joint", "geom", "names"), [
    ('solreflimit="0.005"', "", ["solreflimit on joint", "/j (dof 0) = (0.005, 0)"]),
    ("", 'solref="0.003"', ["solref on geom", "/g = (0.003, 0)"]),
    ('solreflimit="0.005"', 'solref="0.003"', ["solreflimit on joint", "solref on geom"]),
])
def test_shorthand_raises_naming_the_joint_or_geom(joint, geom, names):
    model = _model(joint, geom)
    with pytest.raises(RouteError, match=re.escape(SOLREF_DAMPRATIO)) as err:
        _mujoco(model, {"use_mujoco_contacts": True})
    for name in names:
        assert name in str(err.value)


@pytest.mark.parametrize(("joint", "geom"), [
    ('solreflimit="0.005 1" solreffriction="0.004"', 'solref="0.003 1"'),    # explicit; friction is padded right
    ('solreflimit="-1000 -10"', 'solref="-2000 -20"'),                         # direct stiffness/damping
    ('solreflimit="0 0"', ""),                                                 # authored zero (Newton's mode flag)
    ("", ""),                                                                  # unauthored
])
def test_valid_values_pass(joint, geom):
    model = _model(joint, geom)
    check_solref(model)
    assert _mujoco(model, {"use_mujoco_contacts": True}) is not None


def test_a_written_dampratio_0_gets_the_direct_form_as_its_fix():
    """C3: solref="0.0025 0" (both values written) through the API: the message names the geom and does not call it
    shorthand; the fix is the direct form, not RouteError's generic "no solver yet"."""
    arm = """<mujoco><worldbody><body name="b" pos="0 0 0.2"><freejoint/>
      <geom name="ball" type="sphere" size="0.02" mass="0.1" solref="0.0025 0"/></body></worldbody></mujoco>"""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("bead", mjcf=arm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # Newton's own warning on the zero dampratio
        with pytest.raises(alloy.BuildError) as err:
            alloy.build(scene, device="cpu")
    assert SOLREF_DAMPRATIO in err.value.message and "geom" in err.value.message
    assert "shorthand" not in err.value.message.lower()
    assert err.value.fix == SOLREF_FIX and 'solref="-k -b"' in err.value.fix
    assert "no solver yet" not in err.value.fix
