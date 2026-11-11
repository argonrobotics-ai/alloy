# SPDX-License-Identifier: Apache-2.0
"""MuJoCo Warp's EPA with PR #1845's first six commits, vendored (``alloy.contact.mjw_epa``; WP7's SB-7 S1s
emb_e8_go2_route 100039:1758 and emb_e2_panda_route 100124:832, EPA_HORIZON at ccd_iterations 35).

The regression case is upstream package 14's pair ([internal note] upstream/14-mjwarp-epa-horizon/repro.py): a 47-vertex
convex mesh and a cylinder about 4e-5 m into each other, at fixed float32 poses.
- Stock MuJoCo Warp 3.16.0 sets EPA_HORIZON there and exports no contact.
- Sizing the horizon (3V - 6) cleared the bit but still gave no contact.
- MuJoCo C and the vendored commits give one contact at -4.21e-5 m.
These tests pin that contact and the install's guards."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

from alloy.contact import mjw_epa

VERTS = [
    -0.0015021562576293945, 0.020433783531188965, 0.048446282744407654, 0.0052721998654305935,
    -0.033696964383125305, -0.037017811089754105, -0.009493966586887836, 0.033869680017232895,
    -0.029716359451413155, -0.04135284945368767, -0.0031548230908811092, -0.01570866070687771,
    0.03771352767944336, 0.0012856075773015618, -0.031613510102033615, 0.004986576270312071,
    -0.028016971424221992, 0.03888874873518944, 0.028070399537682533, -0.003435555612668395,
    0.03879382833838463, -0.022998910397291183, -0.0014609869103878736, 0.04680044576525688,
    0.0023154909722507, 0.04073917493224144, -0.010270276106894016, -0.030038896948099136,
    0.013561966829001904, -0.03350923955440521, 0.0006156846648082137, -0.03757235035300255,
    -0.02325853891670704, 0.03342331573367119, -0.0036944360472261906, -0.036815427243709564,
    0.0011592039372771978, -0.037773843854665756, -0.033158138394355774, 0.0012972088297829032,
    0.04144441708922386, -0.016610966995358467, 0.03407492861151695, 0.0007115650223568082,
    0.010267755948007107, -0.015024322085082531, 0.02918061427772045, -0.03353172168135643,
    0.00047253238153643906, 0.02301742136478424, 0.04654010012745857, 0.0012078852159902453,
    -0.03441990166902542, 0.00872692558914423, 0.02248540334403515, 0.017432697117328644,
    -0.029271401464939117, -0.03634262457489967, 0.0052501121535897255, -0.027193445712327957,
    0.01657942309975624, -0.015522727742791176, 0.04073478654026985, -0.015996582806110382,
    -0.023287087678909302, -0.02937789261341095, -0.03967753052711487, -0.004166594240814447,
    -0.005810529924929142, 0.03711914271116257, 0.00043930765241384506, -0.03431734815239906,
    -0.020392777398228645, 0.000795507337898016, 0.04843014478683472, 0.02606934867799282,
    -0.00543180201202631, 0.04050968959927559, 0.007199244573712349, -0.02553953044116497,
    0.04041726142168045, -0.006622352637350559, 0.03614265099167824, -0.027153663337230682,
    0.0018583477940410376, -0.03714916110038757, -0.034680791199207306, 0.037335433065891266,
    0.0016589998267591, -0.021768638864159584, 0.0025941829662770033, 0.03971537947654724,
    -0.005548776127398014, 0.03492823615670204, -0.0020550433546304703, -0.036354389041662216,
    -0.04140627756714821, -0.0029725355561822653, -0.016563551500439644, 3.4703098208410665e-05,
    0.022296875715255737, 0.047493625432252884, 0.037655044347047806, 0.0011100464034825563,
    -0.0330503024160862, 0.0008421039092354476, -0.03535626456141472, 0.0017919257516041398,
    0.0037044340278953314, -0.03531183302402496, -0.03631862625479698, -0.014235624112188816,
    0.029939401894807816, -0.03331134468317032, 0.020196298137307167, -0.017816687002778053,
    -0.0372319370508194, -0.03423285484313965, 0.00814585667103529, -0.02971658855676651,
    -0.04063747078180313, -0.003922346513718367, -0.0102761909365654, -0.022465316578745842,
    -0.001070411060936749, 0.04739086329936981, -0.010804574936628342, 0.010693807154893875,
    0.048687729984521866, -0.030753009021282196, 0.01272676046937704, -0.033176179975271225,
    0.027748841792345047, -0.003736007958650589, 0.03932749107480049, -0.0007402594783343375,
    0.021291878074407578, 0.048209547996520996, -0.021565401926636696, 0.022432668134570122,
    -0.033724285662174225,
]
# geom_xpos (3) + geom_xmat (9) of the mesh and the cylinder at the colliding step (float32)
POSES = np.array([
    [0.4668733477592468, 0.05498781427741051, 0.15543733537197113, 0.950240969657898, 0.2919934391975403,
     -0.1085459440946579, 0.21020090579986572, -0.8581699132919312, -0.468359112739563, -0.22990863025188446,
     0.4222375154495239, -0.8768453598022461],
    [0.507986307144165, 0.0412236750125885, 0.05806028097867966, 0.7909519076347351, -0.5900591611862183,
     -0.1619430035352707, 0.5663900375366211, 0.6059029698371887, 0.5586450099945068, -0.23151181638240814,
     -0.5335841178894043, 0.8134437203407288],
], dtype=np.float32)
XML = f"""<mujoco>
  <option ccd_iterations="35" ccd_tolerance="9.999999974752427e-07"/>
  <asset><mesh name="hull" inertia="shell" vertex="{' '.join(map(repr, VERTS))}"/></asset>
  <worldbody>
    <geom name="hull" type="mesh" mesh="hull"/>
    <body name="can"><freejoint/><geom name="can" type="cylinder" size="0.029999999329447746 0.05000000074505806"/>
    </body>
  </worldbody>
</mujoco>"""


def test_both_epa_users_run_the_vendored_epa():
    """``import alloy`` puts the PR's GJK / EPA into collision_convex and collision_flex at MuJoCo Warp's import."""
    from alloy._vendor import mujoco_warp  # noqa: F401, PLC0415

    from alloy.contact.vendor import mjw_pr1845_epa as pr  # noqa: PLC0415

    assert mjw_epa.state() == {name: mjw_epa.VENDORED for name in mjw_epa.USERS}
    for name, names in mjw_epa.USERS.items():
        assert all(getattr(sys.modules[name], n) is getattr(pr, n) for n in names), name
    assert mjw_epa.has_pr(pr) and not mjw_epa.has_pr(sys.modules[mjw_epa.GJK])


def test_package14_pair_keeps_mujoco_cs_contact():
    """At package 14's poses, MuJoCo Warp now matches MuJoCo C float64: no overflow bit, one contact, the same depth
    (-4.21e-5 m, float32 vs float64 within 1e-7 m), normal and point."""
    from alloy._vendor import mujoco_warp as mjw  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    mjm = mujoco.MjModel.from_xml_string(XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    mjd.geom_xpos[:], mjd.geom_xmat[:] = POSES[:, :3], POSES[:, 3:]
    mujoco.mj_collision(mjm, mjd)
    assert mjd.ncon == 1
    c = mjd.contact[0]
    with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mjm)
        m.opt.warn_overflow = 0
        d = mjw.put_data(mjm, mujoco.MjData(mjm), nconmax=16, njmax=64)
        d.geom_xpos.assign(POSES[None, :, :3])
        d.geom_xmat.assign(POSES[None, :, 3:].reshape(1, 2, 3, 3))
        mjw.collision(m, d)
    assert int(d.overflow.numpy()[0]) == 0 and int(d.nacon.numpy()[0]) == 1
    assert abs(float(d.contact.dist.numpy()[0]) - c.dist) < 1e-7, (d.contact.dist.numpy()[0], c.dist)
    assert np.dot(d.contact.frame.numpy()[0, 0], c.frame[:3]) > 1.0 - 1e-6
    assert np.linalg.norm(d.contact.pos.numpy()[0] - c.pos) < 2e-5
    assert list(d.contact.geom.numpy()[0]) == list(c.geom)


# DC5's ziptie, world 15 of its 16-start ensemble at step 60: the LC stub's cylinder and strap segment 08's box, just
# touching ([internal note] notes/wp3-epa/defect/repro_gjkdist.py); geom_xpos (3) + geom_xmat (9), float32
TOUCH_POSES = np.array([
    [0.30000001192092896, 0.0033909580670297146, 0.18134914338588715, -2.939486876130104e-09, 0.0, 1.000000238418579,
     -0.03509009629487991, 0.9993844032287598, 0.0, -0.9993844032287598, -0.03509009629487991, 2.939486876130104e-09],
    [0.30000001192092896, 0.00452587753534317, 0.1794755458831787, 0.9999999403953552, -1.3335013093751513e-08,
     9.747369134061046e-09, 1.646411007527604e-08, 0.7572101950645447, -0.6531711220741272, 1.3292384970498006e-09,
     0.6531711220741272, 0.7572101950645447],
], dtype=np.float32)
TOUCH_XML = """<mujoco><option ccd_iterations="35" ccd_tolerance="9.999999974752427e-07"/><worldbody>
  <geom type="cylinder" size="0.001500000013038516 0.019999999552965164" gap="0.1"/>
  <body><freejoint/><geom type="box" size="0.00241499999538064 0.0014776403550058603 0.0006600000197067857" gap="0.1"/>
  </body></worldbody></mujoco>"""


def test_touching_cylinder_box_stays_touching():
    """PR #1845's last commit (54f56ca, not vendored) turns this touching pair (MuJoCo C +1.5e-8 m apart) into two
    contacts at -1.113 mm. The vendored commits keep it touching (+2.3e-7 m; stock +3.7e-9 m)."""
    from alloy._vendor import mujoco_warp as mjw  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    mjm = mujoco.MjModel.from_xml_string(TOUCH_XML)
    mjd = mujoco.MjData(mjm)
    mujoco.mj_forward(mjm, mjd)
    mjd.geom_xpos[:], mjd.geom_xmat[:] = TOUCH_POSES[:, :3], TOUCH_POSES[:, 3:]
    assert 0.0 < mujoco.mj_geomDistance(mjm, mjd, 0, 1, 0.01, np.zeros(6)) < 1e-7
    with wp.ScopedDevice("cpu"):
        m = mjw.put_model(mjm)
        m.opt.warn_overflow = 0
        d = mjw.put_data(mjm, mujoco.MjData(mjm), nconmax=16, njmax=64)
        d.geom_xpos.assign(TOUCH_POSES[None, :, :3])
        d.geom_xmat.assign(TOUCH_POSES[None, :, 3:].reshape(1, 2, 3, 3))
        mjw.collision(m, d)
    dist = d.contact.dist.numpy()[: int(d.nacon.numpy()[0])]
    assert len(dist) >= 1 and np.all(np.abs(dist) < 1e-6), dist


def test_kernels_built_before_the_install_refuse(monkeypatch):
    """A user whose kernels were built with the stock EPA cannot take the PR without mixing the two: it raises."""
    from alloy._vendor.mujoco_warp._src import warp_util  # noqa: PLC0415

    user = sys.modules["alloy._vendor.mujoco_warp._src.collision_flex"]
    monkeypatch.setitem(warp_util._KERNEL_CACHE, (0, user.__name__, "stock_kernel"), None)
    with pytest.raises(RuntimeError, match="import alloy before running MuJoCo Warp"):
        mjw_epa.patch(user)


def test_steps_aside_on_opt_out_off_base_and_once_the_pin_has_the_pr(monkeypatch):
    """``ALLOY_MJW_EPA=stock``, a file other than the BASE_PIN base, or a pinned collision_gjk that already has the
    PR each keep MuJoCo Warp's own EPA, and say why."""
    from alloy.contact.vendor import mjw_pr1845_epa as pr  # noqa: PLC0415

    user = sys.modules["alloy._vendor.mujoco_warp._src.collision_convex"]
    assert mjw_epa.refusal(user) is None
    monkeypatch.setenv(mjw_epa.OPT_OUT, "stock")
    assert mjw_epa.refusal(user) == "ALLOY_MJW_EPA=stock"
    monkeypatch.delenv(mjw_epa.OPT_OUT)
    monkeypatch.setitem(mjw_epa.BASE_SHA256, user.__name__, "0" * 64)
    assert f"is not the {mjw_epa.BASE_PIN[:8]} file" in mjw_epa.refusal(user)
    monkeypatch.setitem(sys.modules, mjw_epa.GJK, pr)
    assert "drop alloy.contact.mjw_epa" in mjw_epa.refusal(user)


@pytest.mark.parametrize("first", ["alloy.contact.vendor.mjw_pr1845_epa", "alloy.contact.vendor.mjw_pr1845_polytope",
                                   "alloy._vendor.mujoco_warp"])
def test_every_import_order_ends_with_both_users_vendored(first):
    """A fresh process that imports a vendored module first (the merge gate's import check; it loads MuJoCo Warp while
    half-built) or MuJoCo Warp before alloy still ends with both EPA users on the vendored EPA, and with the other
    post-import patch on collision_convex (alloy.core.mjw_fastpath) applied too."""
    root = str(Path(__file__).resolve().parents[2])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p))
    code = (f"import {first}\nimport sys\nfrom alloy._vendor import mujoco_warp\nfrom alloy.contact import mjw_epa\n"
            "cc = sys.modules['alloy._vendor.mujoco_warp._src.collision_convex']\n"
            "print(sorted(mjw_epa.state().values()) + [cc.upper_trid_index.__module__])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    # both EPA users vendored, and SCR-WP8-5's post-import fast path still lands on collision_convex beside the EPA
    want = [mjw_epa.VENDORED] * 2 + ["alloy.core.mjw_fastpath"]
    assert r.stdout.strip().splitlines()[-1] == str(want), r.stdout[-2000:]
