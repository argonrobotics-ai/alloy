# SPDX-License-Identifier: Apache-2.0
"""MJCF light bodies take the native contact rule on the MuJoCo entry (`alloy.contact.light_body`, coordinator
ruling 2026-11-05 07:19Z): the exact test on host rows, then through the entry factory on MuJoCo Warp and MuJoCo C
(CPU). A loose 12.7 g MJCF block on T12's 2.083 ms row becomes 1.66 ms at T12's step, and so does the file's pair
on it; heavy, hinged, floating-base, coded, static and direct-form rows, a pair Alloy writes between two objects,
material (force-space) rows, and an entry with the rule off keep their rows."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.contact import light_body as LB
from alloy.scene.builtin_plugins import _mujoco
from alloy.scene.native_gains import stable_ke

DT = 0.0008333333333    # T12's step
TAU, DMAX, M_LIGHT, M_HEAVY = 0.0020833333333, 0.99, 0.0127, 5.0
SOLIMP = f"0.95 {DMAX} 0.001 0.5 2"
TAU_SOFT = 0.005        # a file row softer than both caps at DT (F6 R1 test: rk4 cap ~3.3 ms, full ~1.66 ms)
LIGHT_ROW = (f'<geom name="light_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" '
             f'solref="{TAU} 1"')
XML = f"""<mujoco><worldbody>
  <geom name="floor" type="plane" size="1 1 0.1" solref="{TAU} 1" solimp="{SOLIMP}"/>
  <body name="light" pos="0 0 0.0075"><freejoint/>
    <geom name="light_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" solref="{TAU} 1" solimp="{SOLIMP}"/>
  </body>
  <body name="heavy" pos="0.2 0 0.0075"><freejoint/>
    <geom name="heavy_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_HEAVY}" solref="{TAU} 1" solimp="{SOLIMP}"/>
  </body>
  <body name="hinged" pos="0.8 0 0.0075"><joint type="hinge" axis="0 1 0"/>
    <geom name="hinged_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" solref="{TAU} 1" solimp="{SOLIMP}"/>
  </body>
  <body name="base" pos="1.0 0 0.0075"><freejoint/>
    <geom name="base_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" solref="{TAU} 1" solimp="{SOLIMP}"/>
    <body name="arm" pos="0 0 0.03"><joint type="hinge" axis="0 1 0"/>
      <geom name="arm_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" solref="{TAU} 1" solimp="{SOLIMP}"/>
    </body>
  </body>
  <body name="direct" pos="0.4 0 0.0075"><freejoint/>
    <geom name="direct_box" type="box" size="0.0375 0.0125 0.0075" mass="{M_LIGHT}" solref="-230000 -960"
          solimp="{SOLIMP}"/>
  </body>
</worldbody>
<contact><pair geom1="floor" geom2="light_box" solref="{TAU} 1" solimp="{SOLIMP}"/></contact></mujoco>"""


def _tau_native(mass: float, zeta: float = 1.0, rk4: bool = False) -> float:
    ke = min(LB.K_PAIR * (1.0 - DMAX) / mass, stable_ke(zeta, DT, DMAX, "rk4" if rk4 else "implicitfast"))
    return 1.0 / (zeta * math.sqrt(ke))


def test_light_row_rule_on_host_rows():
    rows = np.array([[TAU, 1.0], [TAU, 1.0], [-2.3e5, -960.0], [TAU, 0.5]])
    mass = np.array([M_LIGHT, M_HEAVY, M_LIGHT, M_LIGHT])
    out = LB.light_solref(rows, mass, np.full(4, DMAX), DT, rk4=False)
    assert out[0, 0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-12) and out[0, 0] == pytest.approx(1.66e-3, abs=5e-6)
    assert out[0, 1] == 1.0 and np.array_equal(out[1:3], rows[1:3])        # heavy and direct-form rows kept
    assert _tau_native(M_LIGHT, 0.5) > TAU and np.array_equal(out[3], rows[3])   # zeta 0.5: the cap is softer
    assert np.array_equal(LB.light_solref(rows, mass, np.full(4, DMAX), DT, rk4=True)[0], rows[0])   # rk4 cap: 3.3 ms
    assert LB.light_mass_bound(TAU, 1.0, DMAX) == pytest.approx(0.0434, abs=1e-4)                 # T12: 43 g
    assert LB.K_PAIR == pytest.approx(1.0e6)


def _model(xml: str = XML) -> newton.Model:
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(xml)
    coded = b.add_body(xform=wp.transform((0.6, 0.0, 0.0075), wp.quat_identity()), label="coded")
    s = b.add_shape_box(coded, hx=0.0375, hy=0.0125, hz=0.0075, cfg=newton.ModelBuilder.ShapeConfig(density=300.0))
    light = next(i for i, x in enumerate(b.shape_label) if x.endswith("light_box"))
    b.add_custom_values(**{"mujoco:pair_world": b.current_world, "mujoco:pair_geom1": s, "mujoco:pair_geom2": light,
                           "mujoco:pair_solref": (TAU, 1.0), "mujoco:pair_friction": (1.0, 1.0, 0.005, 1e-4, 1e-4)})
    # ^ a row as Alloy's pair writer adds it
    return b.finalize(device="cpu")


NAMES = ("light_box", "heavy_box", "hinged_box", "base_box", "arm_box", "direct_box", "floor")


def _short(m, g: int) -> str:
    import mujoco

    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
    return next((k for k in NAMES if k in name), "coded")


def _rows(solver, cpu: bool) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Geom rows by name ("coded" for the code-built shape) and pair rows by their other geom ("floor": the file's
    pair, "coded": Alloy's), as the solver steps them."""
    m = solver.mj_model
    if cpu:
        geom, pair = np.array(m.geom_solref), np.array(m.pair_solref)
    else:
        geom, pair = solver.mjw_model.geom_solref.numpy()[0], solver.mjw_model.pair_solref.numpy()[0]
    pairs = {next(x for x in (_short(m, a), _short(m, b)) if x != "light_box"): pair[i]
             for i, (a, b) in enumerate(zip(m.pair_geom1, m.pair_geom2, strict=True))}
    return {_short(m, g): geom[g] for g in range(m.ngeom)}, pairs


def _solver(kw: dict, xml: str = XML):
    model = _model(xml)
    solver = _mujoco(model, {"solver": "newton", "cone": "elliptic", "integrator": "implicitfast", "nconmax": 128,
                             "njmax": 1024, **kw})
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, model.control(), None, DT)
    return solver


def _stepped(kw: dict, xml: str = XML) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    return _rows(_solver(kw, xml), bool(kw.get("use_mujoco_cpu")))


@pytest.mark.parametrize("cpu", [False, True], ids=["mujoco_warp", "mujoco_c"])
def test_entry_stiffens_only_light_mjcf_rows(cpu):
    geom, pair = _stepped({"use_mujoco_cpu": True} if cpu else {})
    assert set(geom) == {*NAMES, "coded"}, set(geom)
    assert geom["light_box"][0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-5) and geom["light_box"][1] == 1.0
    assert geom["heavy_box"][0] == pytest.approx(TAU, rel=1e-6)          # 5 kg: the file's row is stiffer
    assert geom["hinged_box"][0] == pytest.approx(TAU, rel=1e-6)         # a hinged part, not a loose body
    assert geom["base_box"][0] == pytest.approx(TAU, rel=1e-6)           # a floating base with a jointed child
    assert geom["arm_box"][0] == pytest.approx(TAU, rel=1e-6)
    assert geom["floor"][0] == pytest.approx(TAU, rel=1e-6)              # static
    assert geom["direct_box"][0] < 0.0                                   # direct form kept
    assert geom["coded"][0] == pytest.approx(0.02)                        # code-built shape: Newton's fallback kept
    assert pair["floor"][0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-5)   # the file's pair: reduced mass = block's
    assert pair["coded"][0] == pytest.approx(TAU, rel=1e-6)              # Alloy's row between two objects


def test_a_model_without_joints_has_no_loose_bodies():
    """No joint at all (njnt 0: a body welded to the world beside a static geom) is no loose body, not an IndexError
    (WP15's scene: a chart on a fixed link and a static occluder)."""
    import mujoco

    m = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><geom type="plane" size="1 1 0.1"/>
      <body name="chart" pos="0 0 0.1"><geom type="box" size="0.05 0.05 0.005" mass="0.1"/></body>
      </worldbody></mujoco>""")
    assert m.njnt == 0
    assert np.array_equal(LB.loose_bodies(m), np.zeros(m.nbody, dtype=bool))


def test_material_rows_are_not_the_files():
    """A geom whose solref Newton took from a material (FORCE_SPACE) is out of scope; the file's pair stays in."""
    m = _solver({"use_mujoco_cpu": True}).mj_model
    rows = {r.table: r for r in LB.candidate_rows(m, np.zeros(m.ngeom, dtype=bool))}
    assert "geom_solref" not in rows and [_short(m, m.pair_geom1[i]) for i in rows["pair_solref"].idx] == ["floor"]


@pytest.mark.parametrize("cpu", [False, True], ids=["mujoco_warp", "mujoco_c"])
def test_entry_kwarg_turns_the_rule_off(cpu):
    geom, pair = _stepped({LB.LIGHT_KEY: False, **({"use_mujoco_cpu": True} if cpu else {})})
    assert geom["light_box"][0] == pytest.approx(TAU, rel=1e-6) and pair["floor"][0] == pytest.approx(TAU, rel=1e-6)


@pytest.mark.parametrize("cpu", [False, True], ids=["mujoco_warp", "mujoco_c"])
def test_rk4_gains_flag_takes_the_rk4_cap(cpu):
    """F6 R1: an implicitfast entry on rk4's gains (the factories pass ``GAINS_KEY``) gets rows bitwise equal to the
    rk4 entry's; without the flag, the full-cap rows."""
    base = {"use_mujoco_cpu": True} if cpu else {}
    flagged, rk4 = _stepped({**base, LB.GAINS_KEY: True}), _stepped({**base, "integrator": "rk4"})
    plain = _stepped(base)
    for got, want in zip(flagged, rk4, strict=True):
        assert set(got) == set(want)
        for k in got:
            np.testing.assert_array_equal(got[k], want[k])
    assert _tau_native(M_LIGHT, rk4=True) > TAU                          # rk4's cap is softer than the file's row,
    assert flagged[0]["light_box"][0] == pytest.approx(TAU, rel=1e-6)    # which the rule then keeps (it only stiffens)
    assert plain[0]["light_box"][0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-5)        # full cap: stiffened
    assert _tau_native(M_LIGHT) < TAU
    # a 5 ms file row (softer than both caps) tells the caps apart: rk4's (~3.3 ms) under the flag, full (~1.66 ms)
    soft = XML.replace(LIGHT_ROW, LIGHT_ROW.replace(f'solref="{TAU} 1"', f'solref="{TAU_SOFT} 1"'))
    flagged, rk4 = _stepped({**base, LB.GAINS_KEY: True}, soft), _stepped({**base, "integrator": "rk4"}, soft)
    plain = _stepped(base, soft)
    np.testing.assert_array_equal(flagged[0]["light_box"], rk4[0]["light_box"])
    assert flagged[0]["light_box"][0] == pytest.approx(_tau_native(M_LIGHT, rk4=True), rel=1e-5)
    assert _tau_native(M_LIGHT, rk4=True) < TAU_SOFT
    assert plain[0]["light_box"][0] == pytest.approx(_tau_native(M_LIGHT), rel=1e-5)
    assert _tau_native(M_LIGHT) < _tau_native(M_LIGHT, rk4=True)
