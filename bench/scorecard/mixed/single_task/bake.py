# SPDX-License-Identifier: Apache-2.0
"""bake.py IN.xml OUT.xml [--key init]: an MJCF whose build pose is its keyframe pose.

``alloy.build`` starts a robot at its file pose (every hinge and slide at 0), and the public API has no call that
sets joint positions. A task that holds a default joint pose therefore needs a file whose zero IS that pose. This
writes one: every body is placed where the keyframe puts it, and each hinge / slide joint is re-zeroed there (its
``range`` and ``springref`` shifted by the keyframe value, ``ref`` set to 0; the position servos' ``ctrlrange``
shifted the same way). The model is the same mechanism in new joint coordinates: q_new = q_old - q_key. The keyframe
is rewritten in the new coordinates (hinge / slide qpos 0, servo ctrl = old ctrl - q_key), so a hold target read from
it holds the same pose. ``meshdir`` becomes absolute, so the result loads from any folder.

Needs flat MJCF as MuJoCo saves it (bodies with ``pos`` + ``quat``, joints with explicit attributes, radians): the
embodiment packs in ``library/robots/emb_*`` are. ``check`` compares the baked file's body poses with the source's
keyframe pose in MuJoCo itself.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HINGE_SLIDE = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def _fmt(v: np.ndarray) -> str:
    return " ".join(repr(float(x)) for x in np.atleast_1d(v))


def _local(m: mujoco.MjModel, d: mujoco.MjData, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Body ``b``'s pose in its parent's frame at ``d``'s configuration."""
    p = int(m.body_parentid[b])
    inv, quat, pos = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_negQuat(inv, d.xquat[p])
    mujoco.mju_mulQuat(quat, inv, d.xquat[b])
    mujoco.mju_rotVecQuat(pos, d.xpos[b] - d.xpos[p], inv)
    return pos, quat / np.linalg.norm(quat)


def _servo(m: mujoco.MjModel, a: int) -> bool:
    """A position servo on a joint: force = kp (ctrl - q) [- kv qd], gear 1."""
    return (int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT) and m.actuator_gear[a, 0] == 1.0
            and m.actuator_gainprm[a, 0] != 0.0 and m.actuator_biasprm[a, 1] == -m.actuator_gainprm[a, 0])


def _bodies(root: ET.Element, m: mujoco.MjModel, d: mujoco.MjData) -> None:
    els = list(root.find("worldbody").iter("body"))
    if len(els) != m.nbody - 1:
        raise ValueError(f"{len(els)} <body> elements for {m.nbody - 1} model bodies: not a flat MJCF")
    for b, el in enumerate(els, start=1):
        if el.get("name") not in (None, m.body(b).name) or set(el.attrib) - {"name", "pos", "quat", "childclass",
                                                                              "gravcomp", "mocap"}:
            raise ValueError(f"body {b} ({el.attrib}): expected MuJoCo's saved form (name, pos, quat)")
        pos, quat = _local(m, d, b)
        el.set("pos", _fmt(pos))
        el.set("quat", _fmt(quat))


def _joints(root: ET.Element, m: mujoco.MjModel, q: np.ndarray) -> dict[int, float]:
    """Re-zero hinge and slide joints; returns {joint id: keyframe value}."""
    els = list(root.find("worldbody").iter("joint"))
    if len(els) != m.njnt:
        raise ValueError(f"{len(els)} <joint> elements for {m.njnt} model joints: not a flat MJCF")
    shift = {}
    for j, el in enumerate(els):
        if int(m.jnt_type[j]) not in HINGE_SLIDE:
            continue
        q0 = float(q[m.jnt_qposadr[j]])
        shift[j] = q0
        el.set("ref", "0.0")
        if m.jnt_limited[j]:
            el.set("range", _fmt(m.jnt_range[j] - q0))
        el.set("springref", _fmt(m.qpos_spring[m.jnt_qposadr[j]] - q0))
    return shift


def _actuators(root: ET.Element, m: mujoco.MjModel, shift: dict[int, float]) -> dict[int, float]:
    """Shift the servos' ctrlrange; returns {actuator id: keyframe value of its joint}."""
    out = {}
    for a, el in enumerate(root.find("actuator")):
        j = int(m.actuator_trnid[a, 0])
        if not _servo(m, a) or j not in shift:
            continue
        out[a] = shift[j]
        if m.actuator_ctrllimited[a]:
            el.set("ctrlrange", _fmt(m.actuator_ctrlrange[a] - shift[j]))
    return out


def key_pose(src: Path, key: str = "init") -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """The keyframe of ``src`` by name: {joint: its qpos entries}, {actuator: ctrl}."""
    m = mujoco.MjModel.from_xml_path(str(src))
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    if k < 0:
        raise ValueError(f"{src}: no keyframe {key!r}")
    ends = [*m.jnt_qposadr[1:], m.nq]
    q = {m.joint(j).name: m.key_qpos[k, m.jnt_qposadr[j]:ends[j]].copy() for j in range(m.njnt)}
    return q, {m.actuator(a).name: float(m.key_ctrl[k, a]) for a in range(m.nu)}


def bake_tree(root: ET.Element, base: Path, pose: dict[str, np.ndarray], ctrl_by_name: dict[str, float],
              key: str = "init") -> str:
    """The baked MJCF text of the tree ``root`` (files relative to ``base``) at ``pose`` ({joint: qpos entries};
    joints not named keep the file pose), with a keyframe ``key`` in the new coordinates (module docstring)."""
    if root.find("compiler").get("angle") != "radian" or root.find(".//frame") is not None:
        raise ValueError("needs radians and no <frame> (MuJoCo's saved form)")
    comp = root.find("compiler")
    comp.set("meshdir", str((base / comp.get("meshdir", ".")).resolve()))
    for kf in root.findall("keyframe"):
        root.remove(kf)
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    q, ends = m.qpos0.copy(), [*m.jnt_qposadr[1:], m.nq]
    for j in range(m.njnt):
        if m.joint(j).name in pose:
            q[m.jnt_qposadr[j]:ends[j]] = pose[m.joint(j).name]
    ctrl = np.array([ctrl_by_name.get(m.actuator(a).name, 0.0) for a in range(m.nu)])
    d = mujoco.MjData(m)
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)
    _bodies(root, m, d)
    shift = _joints(root, m, q)
    for a, q0 in _actuators(root, m, shift).items():
        ctrl[a] -= q0
    for j, q0 in shift.items():
        q[m.jnt_qposadr[j]] -= q0
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name=key, qpos=_fmt(q), ctrl=_fmt(ctrl))
    return ET.tostring(root, encoding="unicode")


def bake(src: Path, key: str = "init") -> str:
    """The baked MJCF text of the file ``src`` at its keyframe ``key``."""
    pose, ctrl = key_pose(src, key)
    return bake_tree(ET.parse(src).getroot(), src.parent, pose, ctrl, key)


def check(src: Path, out: Path, key: str = "init") -> float:
    """Largest body position / quaternion difference between the source at its keyframe and the baked file at its
    build pose, and between the baked file's build pose and its own keyframe (MuJoCo's kinematics)."""
    a, b = mujoco.MjModel.from_xml_path(str(src)), mujoco.MjModel.from_xml_path(str(out))
    da, db, dk = mujoco.MjData(a), mujoco.MjData(b), mujoco.MjData(b)
    da.qpos[:] = a.key_qpos[mujoco.mj_name2id(a, mujoco.mjtObj.mjOBJ_KEY, key)]
    dk.qpos[:] = b.key_qpos[0]
    for mm, dd in ((a, da), (b, db), (b, dk)):
        mujoco.mj_kinematics(mm, dd)
    flip = np.minimum(np.abs(da.xquat - db.xquat).max(axis=1), np.abs(da.xquat + db.xquat).max(axis=1))
    return float(max(np.abs(da.xpos - db.xpos).max(), flip.max(), np.abs(db.xpos - dk.xpos).max()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--key", default="init")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(bake(a.src, a.key))
    print(f"baked {a.src} -> {a.out}: largest pose difference {check(a.src, a.out, a.key):.2e}")


if __name__ == "__main__":
    main()
