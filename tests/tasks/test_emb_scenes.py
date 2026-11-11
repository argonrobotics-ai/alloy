# SPDX-License-Identifier: Apache-2.0
"""P3-2b embodiment scenes (alloy.tasks.emb_*): the vendored MJCF meets WP8's pack contract
(bench/xbench/ours/README.md, "Scene packs for P3-2b embodiments"), and the bully targets run it."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from alloy.tasks.emb_scene import SOURCE_JSON, check_init, scene_path
from alloy.tasks.emb_specs import EMBODIMENTS
from alloy.tasks.mjcf_tree import hull_vertex_count

KEYS = sorted(k for k, e in EMBODIMENTS.items() if scene_path(e).is_file())


def _model(key: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_path(EMBODIMENTS[key])))


def test_the_embodiments_are_vendored() -> None:
    verdict = {"e1_fr3", "e2_panda", "e3_ur10e_2f85", "e4_fr3_bimanual", "e5_fr3_duo", "e6_g1", "e7_h1_2", "e8_go2",
               "e9_leap"}
    stiff = ("e3_ur10e_2f85", "e4_fr3_bimanual", "e5_fr3_duo", "e6_g1", "e7_h1_2", "e8_go2")
    route = ("e1_fr3", "e2_panda", *stiff)                      # coordinator 2026-11-05 21:04Z: the SB-7 classes
    assert verdict | {f"{k}_stiff" for k in stiff} | {f"{k}_route" for k in route} <= set(KEYS)


@pytest.mark.parametrize("key", KEYS)
def test_scene_meets_the_pack_contract(key: str) -> None:
    m = _model(key)
    coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    hulls = [hull_vertex_count(m, int(m.geom_dataid[g])) for g in coll if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    assert max(hulls, default=0) <= 64                                          # Go2 (MJX variant): primitives only
    assert np.all(m.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
    assert np.all(m.actuator_biastype == int(mujoco.mjtBias.mjBIAS_AFFINE))
    assert np.allclose(m.actuator_biasprm[:, 1], -m.actuator_gainprm[:, 0])     # position servo: kp (ctrl - q)
    joints = m.actuator_trnid[:, 0]                                             # B0 legal box: a finite joint range,
    assert np.all(m.jnt_limited[joints] | m.actuator_ctrllimited)               # or the ctrl range (Duo wheels)
    assert np.all(m.actuator_forcelimited) or key == "e9_leap"                  # LEAP ships no torque limit
    assert np.all(m.body_mass[1:][m.body_jntnum[1:] > 0] > 0)
    assert m.opt.jacobian == int(mujoco.mjtJacobian.mjJAC_SPARSE)
    assert m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)       # off, as Alloy's MJCF route (BUL-0013)
    assert 'multiccd="disable"' in scene_path(EMBODIMENTS[key]).read_text()     # MuJoCo C and Newton both read off


@pytest.mark.parametrize("key", KEYS)
def test_starts_at_rest(key: str) -> None:
    init = check_init(scene_path(EMBODIMENTS[key]))
    assert init["ok"] and init["init_deepest_mm"] > -0.5
    m = _model(key)
    k = m.key("init").id
    joints = m.actuator_trnid[:, 0]
    assert np.allclose(m.key_ctrl[k], m.key_qpos[k][m.jnt_qposadr[joints]])  # start command = start pose


@pytest.mark.parametrize("key", KEYS)
def test_no_self_overlap_where_alloy_starts_a_robot(key: str) -> None:
    """alloy.build starts a robot at qpos0, not at keyframe init (release red 2026-11-09: the Panda's hand hull 30 mm
    into link5, the FR3 Duo's link7 43 mm): no robot self-pair overlaps there by more than 0.5 mm either."""
    from alloy.tasks.emb_build import OVERLAP_EXCLUDE_M, _robot_tree

    m = _model(key)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                                       # MjData starts at qpos0
    robot = set(_robot_tree(m))
    deep = [(m.body(int(m.geom_bodyid[c.geom[0]])).name, m.body(int(m.geom_bodyid[c.geom[1]])).name, float(c.dist))
            for c in d.contact[: d.ncon] if c.dist < -OVERLAP_EXCLUDE_M
            and {int(m.geom_bodyid[g]) for g in c.geom} <= robot]
    assert not deep, deep


FRANKA = ("e1_fr3", "e2_panda", "e4_fr3_bimanual", "e5_fr3_duo")


@pytest.mark.timeout(600)
@pytest.mark.parametrize("key", FRANKA)
def test_shipped_franka_scene_starts_with_check_ok(key: str) -> None:
    """AGENTS.md section 5's shipped arms, as a user adds them: two steps on the CPU, then check() is ok (the Panda
    and the Duo failed START_PEN at 30 and 43 mm before link5 | hand / link7 were excluded)."""
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_robot("arm", scene_path(EMBODIMENTS[key]))
    sim = alloy.build(scene, device="cpu")
    sim.step(2)
    report = sim.check()
    assert report["ok"], report["problems"]


@pytest.mark.parametrize("key", KEYS)
def test_source_is_pinned_with_licence(key: str) -> None:
    rec = json.loads((scene_path(EMBODIMENTS[key]).parent / SOURCE_JSON).read_text())
    assert rec["source"]["commit"] == EMBODIMENTS[key].source_ref()[1] and rec["licence"]
    assert rec["hull_split"] <= 2e-4
    for i, _ in enumerate(EMBODIMENTS[key].licences):
        assert (scene_path(EMBODIMENTS[key]).parent / f"LICENSE.{i}").is_file()


@pytest.mark.timeout(600)
def test_targets_build_on_cpu() -> None:
    import alloy.tasks.emb_targets  # noqa: F401 - registers the emb_* targets
    from alloy.bully.targets import get_target

    from alloy.tasks.emb_specs import ROUTE_DT

    assert get_target("emb_e1_fr3_route").make_sim(1, "cpu").dt == pytest.approx(ROUTE_DT, rel=1e-12)
    for key in ("e1_fr3", "e6_g1"):
        m, tg = _model(key), get_target(f"emb_{key}")
        joints = m.actuator_trnid[:, 0]
        assert tg.actions.dim == m.nu and tg.action_kind == "mujoco.ctrl"
        assert np.all(tg.actions.low >= m.jnt_range[joints, 0]) and np.all(tg.actions.high <= m.jnt_range[joints, 1])
        assert np.allclose(tg.actions.initial, m.key_ctrl[m.key("init").id])
        tree = m.body_rootid == m.body_rootid[m.jnt_bodyid[joints[0]]]   # the robot's kinematic tree
        assert len(tg.robot_bodies) == int(np.sum(tree))                    # floating base included (G1 pelvis)


VARIANTS = sorted(k for k in KEYS if EMBODIMENTS[k].contact)


@pytest.mark.parametrize("key", VARIANTS)
def test_contact_variant_changes_only_solref(key: str) -> None:
    """stiff: solref 5 ms; route: solref 2x the route step; both raise a solimp softer than MuJoCo's default to it
    (Go2's feet) and change nothing else. The arms' variants keep the source's solimp where TRAIN-FIX 1b firmed the
    verdict scene's (alloy.tasks.emb_firm)."""
    from alloy.tasks.emb_firm import DEFAULT_SOLIMP_FULL, FIRM_CONTACTS, FIRM_SOLIMP
    from alloy.tasks.emb_specs import DEFAULT_SOLIMP, ROUTE_SOLREF, STIFF_SOLREF

    e = EMBODIMENTS[key]
    var, base = _model(key), _model(e.base)
    assert np.allclose(var.geom_solref, STIFF_SOLREF if e.contact == "stiff" else ROUTE_SOLREF)
    soft = base.geom_solimp[:, 0] < DEFAULT_SOLIMP[0]                           # Go2's feet only
    assert np.allclose(var.geom_solimp[soft], DEFAULT_SOLIMP)
    same = np.all(var.geom_solimp == base.geom_solimp, axis=1) | soft
    if e.base in FIRM_CONTACTS:   # TRAIN-FIX 1b firms the verdict scene only: the variant keeps the source's solimp
        assert (~same).any() and not np.any(var.geom_solmix > 1.0)
        assert np.allclose(var.geom_solimp[~same], DEFAULT_SOLIMP_FULL)
        assert np.allclose(base.geom_solimp[~same], FIRM_SOLIMP)
    else:
        assert same.all()
    for field in ("geom_friction", "geom_margin", "geom_size", "body_mass", "body_inertia",
                  "actuator_gainprm", "actuator_biasprm", "jnt_range", "key_qpos", "key_ctrl", "eq_solref"):
        assert np.array_equal(getattr(var, field), getattr(base, field)), field


@pytest.mark.parametrize("key", sorted(k for k in KEYS if EMBODIMENTS[k].contact == "route"))
def test_route_variant_discloses_its_contacts(key: str) -> None:
    """The card and source.json carry the source solref and the variant's, and say where the variant is valid."""
    from alloy.tasks.emb_scene import ROUTE_VALID, SCENES, card_name, source_solrefs
    from alloy.tasks.emb_specs import ROUTE_DT, ROUTE_SOLREF

    e = EMBODIMENTS[key]
    rec = json.loads((scene_path(e).parent / SOURCE_JSON).read_text())["contact"]
    assert rec["variant_solref"] == list(ROUTE_SOLREF) and rec["route_dt_s"] == ROUTE_DT and rec["valid"] == ROUTE_VALID
    assert rec["source_solref"] == source_solrefs(e)
    assert sum(c["geoms"] for c in rec["source_solref"]) == _model(key).ngeom
    header = (SCENES / f"{card_name(e)}.yaml").read_text().splitlines()[0]
    assert ROUTE_VALID in header and f"source emb_{e.base}: " in header
