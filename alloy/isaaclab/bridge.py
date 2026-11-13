# SPDX-License-Identifier: Apache-2.0
"""The model Isaac Lab built (its own ``newton``) as a model of Alloy's bundled Newton, sharing every array.

Isaac Lab finalizes a ``newton.Model`` with the Newton it installs (1.6.1 for Isaac Lab 3.0.0). Alloy steps with its
bundled copy (``alloy._vendor.newton``), whose ``Model`` has more attributes and a few renamed ones. Both run on the
one installed Warp, so a Warp array from either is usable by both: :func:`bridge_model` makes a bundled-Newton
``Model`` whose arrays ARE the host model's arrays (no copy, so a change Isaac Lab makes to its model, such as a
randomized mass, is what Alloy sees) and fills what the bundled Newton has and the host lacks with that attribute's
declared default.

Nothing here imports Isaac Lab; the host model is read by attribute name only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton as ven

log = logging.getLogger("alloy.isaaclab")

# host attribute name -> bundled name, where a newer Newton renamed or made private an attribute
RENAMES = {
    "rigid_contact_max": "_rigid_contact_max",
    "mesh_edge_centers": "_mesh_edge_centers",
    "mesh_edge_halves": "_mesh_edge_halves",
    "mesh_edge_indices": "_mesh_edge_indices",
    "shape_collision_filter_pairs": "_shape_collision_filter_pairs",
    "shape_contact_pair_count": "_shape_contact_pair_count",
    "shape_contact_pairs": "_shape_contact_pairs",
    "shape_edge_range": "_shape_edge_range",
    "joint_twist_lower": "_deprecated_joint_twist_lower",
    "joint_twist_upper": "_deprecated_joint_twist_upper",
}
# host custom attribute (namespace "mujoco") -> bundled core attribute, where a newer Newton moved it out of the namespace
CUSTOM_TO_CORE = {"dof_passive_stiffness": "joint_stiffness", "dof_springref": "joint_rest_q"}
NAMESPACES = ("mujoco",)


@dataclass
class BridgeReport:
    """What :func:`bridge_model` shared, renamed, filled with defaults and left out."""

    shared: int = 0
    renamed: dict[str, str] = field(default_factory=dict)
    defaulted: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)


def _enum_by_name(value: Any, enum_cls: type) -> Any:
    """A host IntEnum member as the bundled enum's member of the same name (values may be renumbered)."""
    name = getattr(value, "name", None)
    return enum_cls[name] if name is not None and name in enum_cls.__members__ else value


def _frequency_count(m: ven.Model, frequency: Any) -> int:
    if isinstance(frequency, str):
        return int(m.custom_frequency_counts.get(frequency, 0))
    return m._attribute_frequency_count(frequency)


def _copy_core(host: Any, m: ven.Model, report: BridgeReport) -> None:
    mine = vars(m)
    for name, value in vars(host).items():
        if name in NAMESPACES or name == "attribute_frequency":
            continue
        target = RENAMES.get(name, name)
        if target not in mine:
            target = f"_{name}" if f"_{name}" in mine else None
        if target is None:
            report.dropped.append(name)
            continue
        if target != name:
            report.renamed[name] = target
        mine[target] = value
        report.shared += 1


def _copy_frequencies(host: Any, m: ven.Model) -> None:
    for name, freq in getattr(host, "attribute_frequency", {}).items():
        m.attribute_frequency[name] = freq if isinstance(freq, str) else _enum_by_name(freq, ven.Model.AttributeFrequency)


def _default_array(attr: Any, count: int, device: Any) -> wp.array | list:
    """The declared default for every row (a list for string attributes, as the bundled builder makes them)."""
    return attr._build_default_array(count, device=device)


def _copy_namespace(host: Any, m: ven.Model, ns: str, report: BridgeReport) -> None:
    src = getattr(host, ns, None)
    if src is None:
        return
    dst = getattr(m, ns, None) or ven.Model.AttributeNamespace(ns)
    setattr(m, ns, dst)
    for name, value in vars(src).items():
        core = CUSTOM_TO_CORE.get(name) if ns == "mujoco" else None
        if core is not None:  # moved out of the namespace in the bundled Newton (the old name is a deprecated alias)
            setattr(m, core, value)
            report.renamed[f"{ns}:{name}"] = core
            continue
        setattr(dst, name, value)
    declared = ven.ModelBuilder()
    ven.solvers.SolverMuJoCo.register_custom_attributes(declared)
    for key, attr in declared.custom_attributes.items():
        if attr.namespace != ns or attr.name in vars(dst) or attr.name in CUSTOM_TO_CORE:
            continue
        freq = attr.frequency if isinstance(attr.frequency, str) else _enum_by_name(attr.frequency, ven.Model.AttributeFrequency)
        setattr(dst, attr.name, _default_array(attr, _frequency_count(m, freq), m.device))
        m.attribute_frequency.setdefault(key, freq)
        report.defaulted.append(key)


def _fill_core_defaults(m: ven.Model, report: BridgeReport) -> None:
    """Per-entity arrays the bundled Newton reads that the host model does not have."""
    zeros = {"joint_stiffness": ("joint_dof_count", wp.float32), "joint_rest_q": ("joint_coord_count", wp.float32)}
    for name, (count_attr, dtype) in zeros.items():
        if getattr(m, name, None) is None:
            setattr(m, name, wp.zeros(int(getattr(m, count_attr)), dtype=dtype, device=m.device))
            report.defaulted.append(name)


def _mimic_per_joint(host: Any, m: ven.Model, report: BridgeReport) -> None:
    """Newton 1.6 keeps mimic joints as a constraint table (joint0 follows coef0 + coef1 * joint1); Newton 1.8 keeps
    them per joint (``joint_mimic_joint``: the leader or -1, ``joint_mimic_coeffs``: (coef0, coef1))."""
    if m.joint_mimic_joint is not None:
        return
    leader = np.full(int(m.joint_count), -1, dtype=np.int32)
    coeffs = np.tile(np.array([0.0, 1.0], dtype=np.float32), (int(m.joint_count), 1))
    count = int(getattr(host, "constraint_mimic_count", 0) or 0)
    if count:
        follower = host.constraint_mimic_joint0.numpy()
        lead = host.constraint_mimic_joint1.numpy()
        on = host.constraint_mimic_enabled.numpy().astype(bool)
        c0, c1 = host.constraint_mimic_coef0.numpy(), host.constraint_mimic_coef1.numpy()
        leader[follower[on]] = lead[on]
        coeffs[follower[on]] = np.stack([c0[on], c1[on]], axis=1)
    m.joint_mimic_joint = wp.array(leader, dtype=wp.int32, device=m.device)
    m.joint_mimic_coeffs = wp.array(coeffs, dtype=wp.vec2, device=m.device)
    report.renamed["constraint_mimic_*"] = f"joint_mimic_joint/joint_mimic_coeffs ({count} mimic joints)"


def bridge_model(host: Any) -> tuple[ven.Model, BridgeReport]:
    """A bundled-Newton ``Model`` over the host model's arrays, and a report of what was shared and filled."""
    m = ven.Model(host.device)
    report = BridgeReport()
    _copy_core(host, m, report)
    _copy_frequencies(host, m)
    for ns in NAMESPACES:
        _copy_namespace(host, m, ns, report)
    _fill_core_defaults(m, report)
    _mimic_per_joint(host, m, report)
    log.info("bridged the host model: %d attributes shared, %d renamed, %d filled with defaults, %d left out (%s)",
             report.shared, len(report.renamed), len(report.defaulted), len(report.dropped),
             ", ".join(report.dropped) or "none")
    return m, report


def resync_arrays(host: Any, m: ven.Model) -> int:
    """Point the bridged model at any host array Isaac Lab replaced (rather than wrote into) since the bridge;
    returns how many were re-pointed. Cheap: compares object identities only."""
    mine, changed = vars(m), 0
    for name, value in vars(host).items():
        target = RENAMES.get(name, name)
        if target not in mine:
            target = f"_{name}" if f"_{name}" in mine else None
        if target is not None and isinstance(value, wp.array) and mine[target] is not value:
            mine[target] = value
            changed += 1
    for ns in NAMESPACES:
        src, dst = getattr(host, ns, None), getattr(m, ns, None)
        if src is None or dst is None:
            continue
        for name, value in vars(src).items():
            owner, target = (m, CUSTOM_TO_CORE[name]) if ns == "mujoco" and name in CUSTOM_TO_CORE else (dst, name)
            if isinstance(value, wp.array) and vars(owner).get(target) is not value:
                setattr(owner, target, value)
                changed += 1
    return changed


def joint_type_ok(host: Any) -> None:
    """Raise when the host model has something the bundled Newton reads differently: a joint type it has no number
    for (none known today), or joint position targets in Newton 1.6's old per-DOF layout (the bundled Newton reads
    them per coordinate, Newton 1.6's default and what Isaac Lab writes)."""
    if getattr(host, "use_coord_layout_targets", True) is False:
        raise NotImplementedError("this model keeps joint position targets per DOF (newton.use_coord_layout_targets = "
                                  "False); Alloy reads them per coordinate. Leave newton.use_coord_layout_targets at "
                                  "its default (True) before Isaac Lab builds the scene, or use physics=newton_mjwarp")
    if not getattr(host, "joint_count", 0):
        return
    kinds = set(np.unique(host.joint_type.numpy()).tolist())
    unknown = kinds - {int(v) for v in ven.JointType}
    if unknown:
        raise NotImplementedError(f"joint types {sorted(unknown)} of this Newton have no match in Alloy's Newton "
                                  f"{ven.__version__}; use physics=newton_mjwarp for this task")
