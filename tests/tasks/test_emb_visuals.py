# SPDX-License-Identifier: Apache-2.0
"""Arm packs' visual layer (built by alloy.tasks.emb_visuals, loaded by alloy.scene.robot_visuals): every layer resolves
onto the pack's bodies with its source recorded, and attaching it (MJCF tree, Newton builder, clip kit) adds display
geometry only: the same compiled physics and the same MuJoCo C / MuJoCo Warp rollouts as the pack alone."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from alloy.tasks.emb_scene import SOURCE_JSON, scene_path
from alloy.tasks.emb_specs import EMBODIMENTS
from alloy.scene.robot_visuals import (ENV_SWITCH, MANIFEST, add_mjcf, attach_mjcf, layer_dir, scene_with_visuals,
                                       visual_geoms)  # fmt: skip
from alloy.tasks.emb_visuals import LINKS, SOURCES, keys

KEYS = keys()
#: compiled physics the layer must leave alone; collision geoms are compared in order below
MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_subtreemass",
                "jnt_range", "dof_damping", "dof_armature", "actuator_gainprm", "actuator_biasprm", "qpos0", "key_qpos",
                "key_ctrl", "eq_data")  # fmt: skip
GEOM_FIELDS = ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_bodyid", "geom_dataid", "geom_contype",
               "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
               "geom_condim", "geom_priority", "geom_solmix", "geom_rbound", "geom_aabb")  # fmt: skip


def _xml(key: str) -> Path:
    return scene_path(EMBODIMENTS[key])


def _visual(m: mujoco.MjModel) -> np.ndarray:
    return np.array([g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0], dtype=int)


def _rollout(m: mujoco.MjModel, steps: int = 150) -> np.ndarray:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("init").id)
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    out = []
    for i in range(steps):
        d.ctrl[:] = np.clip(d.ctrl + 0.5 * (hi - lo) * np.sin(0.05 * i + np.arange(m.nu)) / steps, lo, hi)
        mujoco.mj_step(m, d)
        out.append(np.concatenate([d.qpos, d.qvel, d.qacc_warmstart]))
    return np.asarray(out)


def _expected_geoms(key: str) -> int:
    base = EMBODIMENTS[key].builder
    if base in LINKS:
        other, prefixes = LINKS[base]
        return len(prefixes) * len(visual_geoms(_xml(other)))
    return len(json.loads((layer_dir(_xml(base)) / MANIFEST).read_text())["geoms"])


@pytest.mark.parametrize("key", KEYS)
def test_layer_resolves_onto_the_pack(key: str) -> None:
    geoms = visual_geoms(_xml(key))
    bodies = {b.get("name") for b in ET.parse(_xml(key)).getroot().iter("body")}
    assert len(geoms) == _expected_geoms(key) > 0
    assert {g.body for g in geoms} <= bodies and all(g.file.is_file() for g in geoms)
    assert len({g.name for g in geoms}) == len(geoms) and all(0 < g.rgba[3] <= 1 for g in geoms)


@pytest.mark.parametrize("key", sorted(SOURCES))
def test_layer_records_its_source(key: str) -> None:
    rec = json.loads((layer_dir(_xml(key)) / MANIFEST).read_text())
    pack = json.loads((_xml(key).parent / SOURCE_JSON).read_text())
    assert (rec["source"]["commit"], rec["source"]["licence"]) == (pack["source"]["commit"], pack["licence"])
    assert set(rec["source"]["models"]) <= set(pack["source"]["files"])
    used = {g["file"] for g in rec["geoms"]}
    assert used == {p.name for p in layer_dir(_xml(key)).glob("*.msh")}       # every mesh file used, none missing
    assert len(rec["source"]["meshes"]) == len(used)


@pytest.mark.parametrize("key", KEYS)
def test_attached_layer_changes_no_physics(key: str, tmp_path: Path) -> None:
    """MuJoCo C: the pack with its layer compiles to the pack's physics (collision geoms in order) and rolls out the
    same, bit for bit."""
    bare = mujoco.MjModel.from_xml_path(str(_xml(key)))
    full = mujoco.MjModel.from_xml_path(str(scene_with_visuals(_xml(key), tmp_path)))
    vis = _visual(full)
    keep = np.setdiff1d(np.arange(full.ngeom), vis)
    assert vis.size == _expected_geoms(key) and keep.size == bare.ngeom and _visual(bare).size == 0
    assert np.all(full.geom_group[vis] == 2)
    for f in MODEL_FIELDS:
        assert np.array_equal(getattr(full, f), getattr(bare, f)), f
    for f in GEOM_FIELDS:                       # mesh ids too: the layer's meshes come after the pack's own
        assert np.array_equal(getattr(full, f)[keep], getattr(bare, f)), f
    assert np.array_equal(_rollout(full), _rollout(bare))


def test_attach_skips_what_is_there() -> None:
    root = ET.parse(_xml("e1_fr3")).getroot()
    n = attach_mjcf(root, _xml("e1_fr3"))
    assert n == len(visual_geoms(_xml("e1_fr3"))) and attach_mjcf(root, _xml("e1_fr3")) == 0


def test_env_switch_skips_the_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ALLOY_ROBOT_VISUALS=0``: scene loads import the pack without its layer (training runs that never render)."""
    from alloy._vendor import newton

    monkeypatch.setenv(ENV_SWITCH, "0")
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    assert add_mjcf(b, _xml("e1_fr3"), ctrl_direct=True) == []
    assert not any("_vis" in str(label) for label in b.shape_label)


def test_clip_arms_take_the_pack_layer() -> None:
    """The clip kit's arm XML carries the pack's layer once (FR3 as before from the vendored copy; Panda now too)."""
    from alloy.clips.arms.mjcf import robot_xml
    from alloy.clips.arms.specs import SPECS

    for key, pack in (("fr3", "e1_fr3"), ("panda", "e2_panda")):
        m = mujoco.MjModel.from_xml_string(robot_xml(SPECS[key], "left_"))
        names = {m.geom(int(g)).name for g in _visual(m)}
        assert names == {"left_" + g.name for g in visual_geoms(_xml(pack))}


@pytest.mark.compile_heavy
@pytest.mark.parametrize("key", ["e1_fr3", "e4_fr3_bimanual"])
def test_newton_attach_adds_display_shapes_only(key: str) -> None:
    """Newton: the layer adds one display shape per visual geom, no mass; SolverMuJoCo builds the same MuJoCo model
    and MuJoCo Warp steps both identically on CPU."""
    from alloy._vendor import newton
    import warp as wp

    def build(layer: bool) -> tuple[newton.Model, list[int]]:
        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        shapes = add_mjcf(b, _xml(key), ctrl_direct=True) if layer else b.add_mjcf(str(_xml(key)), ctrl_direct=True)
        return b.finalize(device="cpu"), shapes or []

    (full, shapes), (bare, _) = build(True), build(False)
    assert len(shapes) == full.shape_count - bare.shape_count == len(visual_geoms(_xml(key)))
    flags, body = full.shape_flags.numpy(), full.shape_body.numpy()
    collide, visible = int(newton.ShapeFlags.COLLIDE_SHAPES), int(newton.ShapeFlags.VISIBLE)
    assert not np.any(flags[shapes] & collide) and np.all(flags[shapes] & visible)
    hulls = np.isin(body, body[shapes]) & (flags & collide).astype(bool)
    assert hulls.any() and not np.any(flags[hulls] & visible)                     # dressed bodies show their meshes
    assert np.array_equal(full.shape_flags.numpy()[: bare.shape_count] | (hulls[: bare.shape_count] * visible),
                          bare.shape_flags.numpy())                                # nothing else changes
    for f in ("body_mass", "body_inertia", "body_com", "joint_q"):
        assert np.array_equal(getattr(full, f).numpy(), getattr(bare, f).numpy()), f

    def steps(model: newton.Model) -> tuple[int, np.ndarray]:
        with wp.ScopedDevice("cpu"):
            solver = newton.solvers.SolverMuJoCo(model, nconmax=256, njmax=1024)   # the emb targets' capacities
            s0, s1, ctl = model.state(), model.state(), model.control()
            for _ in range(25):
                solver.step(s0, s1, ctl, None, 0.002)
                s0, s1 = s1, s0
        return solver.mj_model.ngeom, s0.joint_q.numpy()

    (ng_full, q_full), (ng_bare, q_bare) = steps(full), steps(bare)
    assert ng_full == ng_bare and np.array_equal(q_full, q_bare)
