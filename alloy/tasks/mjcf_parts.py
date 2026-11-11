# SPDX-License-Identifier: Apache-2.0
"""Non-tree MJCF of a compiled ``MjModel``: compiler, options and flags, actuators, equalities, contact pairs and
excludes. Companion of :mod:`alloy.tasks.mjcf_tree`; every value is the compiled one at ``repr`` precision."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from alloy.core.errors import TraceError
from alloy.tasks.mjcf_tree import Names, flag, fmt, fmt_int, ranged

log = logging.getLogger("alloy.tasks")

_UNSUPPORTED_COUNTS = ("ntendon", "nflex", "nhfield", "nplugin", "nskin")
_FLOAT_OPTIONS = (
    "timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance", "sleep_tolerance",
    "gravity", "wind", "magnetic", "density", "viscosity", "o_margin", "o_solref", "o_solimp", "o_friction",
)  # fmt: skip
_INT_OPTIONS = ("iterations", "ls_iterations", "noslip_iterations", "ccd_iterations", "sdf_iterations", "sdf_initpoints")
_ENUM_OPTIONS = {
    "integrator": ("Euler", "RK4", "implicit", "implicitfast", "discrete"),
    "cone": ("pyramidal", "elliptic"), "jacobian": ("dense", "sparse", "auto"), "solver": ("PGS", "CG", "Newton"),
}  # fmt: skip
_DYN = ("none", "integrator", "filter", "filterexact", "muscle", "dcmotor", "pid", "user")
_GAIN = ("fixed", "affine", "muscle", "dcmotor", "so3", "pid", "user")
_BIAS = ("none", "affine", "muscle", "dcmotor", "so3", "user")
_TRN_ATTR = {0: "joint", 1: "jointinparent", 2: "tendon", 4: "site", 5: "body"}


def check_supported(m: mujoco.MjModel) -> None:
    """Raise :class:`TraceError` for model parts the converter does not write."""
    present = {k: int(getattr(m, k)) for k in _UNSUPPORTED_COUNTS if int(getattr(m, k)) > 0}
    if present:
        raise TraceError(f"model uses parts the converter does not handle: {present}")
    if np.any(m.geom_type == int(mujoco.mjtGeom.mjGEOM_SDF)):
        raise TraceError("model has SDF geoms; the converter does not handle them")


def write_compiler(root: ET.Element) -> None:
    """Explicit-value compiler settings: radians, explicit inertials, nothing fused, aligned or bounded."""
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", inertiafromgeom="false",
                  balanceinertia="false", boundmass="0", boundinertia="0", fusestatic="false",
                  discardvisual="false", alignfree="false")  # fmt: skip


def _flag_attrs(bits: int, enum: type, prefix: str, value: str) -> dict[str, str]:
    out = {}
    for name, member in enum.__members__.items():
        if name.startswith(prefix) and int(member) & bits and bin(int(member)).count("1") == 1:
            out[name[len(prefix) :].lower()] = value
    return out


def write_option(root: ET.Element, m: mujoco.MjModel) -> None:
    """``<option>`` with every float, int and enum field, plus ``<flag>`` for the disable/enable bits."""
    o = m.opt
    attrs = {name: fmt(getattr(o, name)) for name in _FLOAT_OPTIONS}
    attrs |= {name: fmt_int(getattr(o, name)) for name in _INT_OPTIONS}
    attrs |= {name: table[int(getattr(o, name))] for name, table in _ENUM_OPTIONS.items()}
    if int(o.disableactuator):
        attrs["actuatorgroupdisable"] = " ".join(str(g) for g in range(31) if int(o.disableactuator) >> g & 1)
    opt = ET.SubElement(root, "option", attrs)
    flags = _flag_attrs(int(o.disableflags), mujoco.mjtDisableBit, "mjDSBL_", "disable")
    flags |= _flag_attrs(int(o.enableflags), mujoco.mjtEnableBit, "mjENBL_", "enable")
    if flags:
        ET.SubElement(opt, "flag", flags)


def _actuator_target(m: mujoco.MjModel, k: int, names: Names) -> dict[str, str]:
    trn = int(m.actuator_trntype[k])
    lookup = {0: names.joint, 1: names.joint, 2: names.tendon, 4: names.site, 5: names.body}
    if trn not in _TRN_ATTR:
        raise TraceError(f"actuator {names.actuator[k]}: transmission type {trn} is not converted")
    attrs = {_TRN_ATTR[trn]: lookup[trn][int(m.actuator_trnid[k, 0])]}
    if trn == 4 and int(m.actuator_trnid[k, 1]) >= 0:
        attrs["refsite"] = names.site[int(m.actuator_trnid[k, 1])]
    return attrs


def _check_actuator_extras(m: mujoco.MjModel, k: int, names: Names) -> None:
    extras = {
        "damping": m.actuator_damping[k], "dampingpoly": m.actuator_dampingpoly[k],
        "armature": m.actuator_armature[k], "delay": m.actuator_delay[k],
    }  # fmt: skip
    nonzero = {key: val for key, val in extras.items() if np.any(np.asarray(val) != 0)}
    if nonzero:
        raise TraceError(f"actuator {names.actuator[k]}: unconverted parameters {nonzero}")


def write_actuators(root: ET.Element, m: mujoco.MjModel, names: Names) -> None:
    """Each actuator as ``<general>`` with the compiled gain, bias and dynamics parameters."""
    act = ET.SubElement(root, "actuator")
    for k in range(m.nu):
        _check_actuator_extras(m, k, names)
        attrs = {"name": names.actuator[k]} | _actuator_target(m, k, names)
        attrs |= {
            "gear": fmt(m.actuator_gear[k]),
            "dyntype": _DYN[int(m.actuator_dyntype[k])], "gaintype": _GAIN[int(m.actuator_gaintype[k])],
            "biastype": _BIAS[int(m.actuator_biastype[k])], "dynprm": fmt(m.actuator_dynprm[k]),
            "gainprm": fmt(m.actuator_gainprm[k]), "biasprm": fmt(m.actuator_biasprm[k]),
            "ctrllimited": flag(m.actuator_ctrllimited[k]), "forcelimited": flag(m.actuator_forcelimited[k]),
            "actlimited": flag(m.actuator_actlimited[k]), "actearly": flag(m.actuator_actearly[k]),
            "group": fmt_int(m.actuator_group[k]),
        }  # fmt: skip
        attrs |= ranged("ctrlrange", m.actuator_ctrllimited[k], m.actuator_ctrlrange[k])
        attrs |= ranged("forcerange", m.actuator_forcelimited[k], m.actuator_forcerange[k])
        attrs |= ranged("actrange", m.actuator_actlimited[k], m.actuator_actrange[k])
        ET.SubElement(act, "general", attrs)


def _equality_attrs(m: mujoco.MjModel, k: int, names: Names) -> tuple[str, dict[str, str]]:
    eq_type, data = int(m.eq_type[k]), m.eq_data[k]
    o1, o2 = int(m.eq_obj1id[k]), int(m.eq_obj2id[k])
    if eq_type == int(mujoco.mjtEq.mjEQ_JOINT):
        attrs = {"joint1": names.joint[o1], "polycoef": fmt(data[:5])}
        if o2 >= 0:
            attrs["joint2"] = names.joint[o2]
        return "joint", attrs
    if eq_type not in (int(mujoco.mjtEq.mjEQ_CONNECT), int(mujoco.mjtEq.mjEQ_WELD)):
        raise TraceError(f"equality {names.equality[k]}: type {eq_type} is not converted")
    is_site = int(m.eq_objtype[k]) == int(mujoco.mjtObj.mjOBJ_SITE)
    lookup, key = (names.site, "site") if is_site else (names.body, "body")
    attrs = {f"{key}1": lookup[o1], f"{key}2": lookup[o2] if o2 >= 0 else "world"}
    if eq_type == int(mujoco.mjtEq.mjEQ_CONNECT):
        return "connect", attrs | ({} if is_site else {"anchor": fmt(data[:3])})
    weld = {} if is_site else {"anchor": fmt(data[:3]), "relpose": fmt(data[3:10])}
    return "weld", attrs | weld | {"torquescale": fmt(data[10])}


def write_equalities(root: ET.Element, m: mujoco.MjModel, names: Names) -> None:
    """Each equality with its compiled data (anchors, relative pose or polynomial)."""
    if m.neq == 0:
        return
    eq = ET.SubElement(root, "equality")
    for k in range(m.neq):
        tag, attrs = _equality_attrs(m, k, names)
        attrs = {"name": names.equality[k]} | attrs
        attrs |= {"solref": fmt(m.eq_solref[k]), "solimp": fmt(m.eq_solimp[k]), "active": flag(m.eq_active0[k])}
        ET.SubElement(eq, tag, attrs)


def write_contacts(root: ET.Element, m: mujoco.MjModel, names: Names) -> None:
    """Explicit contact pairs and body-pair excludes."""
    if m.npair == 0 and m.nexclude == 0:
        return
    contact = ET.SubElement(root, "contact")
    for k in range(m.npair):
        ET.SubElement(contact, "pair", geom1=names.geom[int(m.pair_geom1[k])], geom2=names.geom[int(m.pair_geom2[k])],
                      condim=fmt_int(m.pair_dim[k]), friction=fmt(m.pair_friction[k]), solref=fmt(m.pair_solref[k]),
                      solreffriction=fmt(m.pair_solreffriction[k]), solimp=fmt(m.pair_solimp[k]),
                      margin=fmt(m.pair_margin[k]), gap=fmt(m.pair_gap[k]))  # fmt: skip
    for k in range(m.nexclude):
        sig = int(m.exclude_signature[k])  # (body1 << 16) + body2
        ET.SubElement(contact, "exclude", body1=names.body[sig >> 16], body2=names.body[sig & 0xFFFF])


def write_keyframes(root: ET.Element, m: mujoco.MjModel) -> None:
    """``<keyframe>``: every key's time, qpos, qvel and ctrl (act / mocap when the model has them)."""
    if m.nkey == 0:
        return
    keyframe = ET.SubElement(root, "keyframe")
    for k in range(m.nkey):
        attrs = {"name": m.key(k).name} if m.key(k).name else {}
        attrs |= {"time": fmt(m.key_time[k]), "qpos": fmt(m.key_qpos[k]), "qvel": fmt(m.key_qvel[k])}
        if m.nu:
            attrs["ctrl"] = fmt(m.key_ctrl[k])
        if m.na:
            attrs["act"] = fmt(m.key_act[k])
        if m.nmocap:
            attrs |= {"mpos": fmt(m.key_mpos[k]), "mquat": fmt(m.key_mquat[k])}
        ET.SubElement(keyframe, "key", attrs)
