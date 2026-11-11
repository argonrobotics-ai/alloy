# SPDX-License-Identifier: Apache-2.0
"""MJCF back end: a ``LatchSpec`` as an MJCF piece for an object card (``shape: {mjcf: {path, body}}``).

The piece holds one root body (``spec.name``; a free joint unless ``spec.static``) with the members nested below
it. Each member joint carries ``stiffness``/``springref`` (Newton 1.8 imports them as ``Model.joint_stiffness`` /
``Model.joint_rest_q``), ``damping``, ``range = [0, travel]`` and a raw direct ``solreflimit`` (stiff stop,
CON-47). Every pair of bodies in the assembly gets a ``<contact><exclude>`` (MuJoCo's parent filter does not apply
under a welded root, CON-47); a kept contact (``LatchSpec.contacts``, a detent) is a ``<pair>``, which the exclude
does not filter. Couplings become ``<equality><joint polycoef>``. Bodies carry explicit
``<inertial>`` (regularised, CON-45); geoms carry density 0 so the importer does not add mass on top.

A static spec without members is written as plain world geoms (no body), taken as-is by the STATIC route.

``gap`` (optional) writes a ``<default><geom gap>``: Newton imports it as each shape's contact gap. Without it the
imported shapes keep Newton's default rigid gap of 0.1 m (the assembler's ``_restyle`` does not set it), and MuJoCo
generates inactive contacts for every geom pair within 10 cm (DC1's task scene: 964 contacts at rest, 238
constraint rows; SCR-DC1-4). Pass ``alloy.scene.emitters.common.RIGID_GAP`` (5 mm, SPEC §4.2).

The card's material is applied to every imported shape by the assembler (``alloy/scene/emitters/rigid.py``
``_restyle``); touched faces get their stiff contact from the card's ``precision_features`` crops.
"""

from __future__ import annotations

from itertools import combinations
from xml.sax.saxutils import escape

import numpy as np

from alloy.latch.mass import MassProps, member_armature, member_props, root_props, stop_solimp, stop_solref
from alloy.latch.spec import ROOT, Box, ContactPair, LatchSpec

FMT = "{:.9g}"
LIMIT_MARGIN = 1.0e-5   # rad or m: MuJoCo activates a limit only inside its margin, so a member preloaded on its
                        # stop at q = 0 started free and overshot ~5 deg (DC4); the rest shifts in by the margin


def _v(xs) -> str:
    return " ".join(FMT.format(float(x)) for x in xs)


def body_name(spec: LatchSpec, part: str) -> str:
    """MJCF/Newton body name of a part (the root is the spec name)."""
    return spec.name if part == ROOT else f"{spec.name}_{part}"


def joint_name(spec: LatchSpec, member: str) -> str:
    return f"{spec.name}_{member}_joint"


def _inertial(p: MassProps) -> str:
    I = p.inertia
    full = (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])
    return f'<inertial pos="{_v(p.com)}" mass="{FMT.format(p.mass)}" fullinertia="{_v(full)}"/>'


def _geom(b: Box) -> str:
    x, y, z, w = b.quat
    off = ' contype="0" conaffinity="0"' if (b.role == "visual" or not b.collide) else ""
    return (f'<geom name="{escape(b.name)}" type="box" size="{_v(b.half)}" pos="{_v(b.pos)}" '
            f'quat="{_v((w, x, y, z))}" density="0"{off}/>')


def _joint(spec: LatchSpec, name: str, imp: dict | None) -> str:
    """``imp``: member -> (armature, solimplimit or None, limit margin); ``None`` writes the metric model (none of
    them, see ``stop_inv_weights``)."""
    m = spec.member(name)
    kind = "hinge" if m.kind == "hinge" else "slide"
    k, d = stop_solref(spec)
    arm, si, mg = imp[name] if imp is not None else (0.0, None, 0.0)
    extra = f' solimplimit="{_v(si)}"' if si is not None else ""
    marg = f' margin="{FMT.format(mg)}"' if mg else ""
    return (f'<joint name="{joint_name(spec, name)}" type="{kind}" axis="{_v(m.axis)}" limited="true" '
            f'range="0 {FMT.format(m.travel)}" stiffness="{FMT.format(m.spring.rate)}" '
            f'springref="{FMT.format(m.spring.springref)}" damping="{FMT.format(m.spring.damping)}" '
            f'armature="{FMT.format(arm)}"{marg} solreflimit="{_v((k, d))}"{extra}/>')


def stop_inv_weights(spec: LatchSpec) -> dict[str, float]:
    """Member -> MuJoCo ``dof_invweight0`` of its joint in the spec alone, compiled without armature (root free
    unless ``spec.static``). An upper bound on the scene's value: MuJoCo 3.16 ignores armature in it under a welded
    root and counts it under a free one, and a mounted root only lowers it."""
    import mujoco                      # noqa: PLC0415 (only for members with a stop_stiffness)

    mm = mujoco.MjModel.from_xml_string(_text(spec, "metric", None, None))
    out = {}
    for m in spec.members:
        j = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, joint_name(spec, m.name))
        out[m.name] = float(mm.dof_invweight0[mm.jnt_dofadr[j]])
    return out


def joint_impedance(spec: LatchSpec, margin: float = LIMIT_MARGIN) -> dict[str, tuple]:
    """Member -> (armature, flat solimplimit or None, limit margin): ``Member.regularise``,
    ``Member.stop_stiffness`` and the margin that keeps the limit active on the stop."""
    stiff = [m for m in spec.members if m.stop_stiffness is not None]
    inv = stop_inv_weights(spec) if stiff else {}
    k, _ = stop_solref(spec)
    return {m.name: (member_armature(m, spec.design_dt),
                     stop_solimp(m, k, inv[m.name]) if m.name in inv else None, margin) for m in spec.members}


PAIR_FRICTION_REST = (0.005, 1.0e-4, 1.0e-4)        # torsional, rolling x2 (MuJoCo's pair defaults)


def pair_params(spec: LatchSpec, c: ContactPair) -> dict:
    """The explicit pair of a kept contact (``LatchSpec.contacts``), shared by the MJCF and the builder."""
    return {"condim": c.condim, "friction": (c.mu, c.mu, *PAIR_FRICTION_REST), "solref": spec.contact_solref(c),
            "solimp": tuple(c.solimp), "margin": 0.0, "gap": 0.0}


def _pair(spec: LatchSpec, c: ContactPair) -> str:
    p = pair_params(spec, c)
    return (f'<pair geom1="{escape(c.box1)}" geom2="{escape(c.box2)}" condim="{p["condim"]}" '
            f'friction="{_v(p["friction"])}" solref="{_v(p["solref"])}" solimp="{_v(p["solimp"])}" margin="0" '
            f'gap="0"/>')


def _body(spec: LatchSpec, part: str, indent: int, imp: dict | None) -> list[str]:
    pad = " " * indent
    if part == ROOT:
        props, shapes = root_props(spec), spec.root_shapes
        head = f'{pad}<body name="{body_name(spec, ROOT)}" pos="0 0 0">'
        lines = [head] + ([f"{pad}  <freejoint name=\"{spec.name}_free\"/>"] if not spec.static else [])
    else:
        m = spec.member(part)
        props, shapes = member_props(m, spec.design_dt), m.shapes
        lines = [f'{pad}<body name="{body_name(spec, part)}" pos="{_v(m.anchor)}">', f"{pad}  {_joint(spec, part, imp)}"]
    lines.append(f"{pad}  {_inertial(props)}")
    lines += [f"{pad}  {_geom(b)}" for b in shapes]
    for child in (c.name for c in spec.members if c.parent == part):
        lines += _body(spec, child, indent + 2, imp)
    lines.append(f"{pad}</body>")
    return lines


def latch_mjcf(spec: LatchSpec, comment: str = "", gap: float | None = None, margin: float = LIMIT_MARGIN) -> str:
    """The MJCF text of ``spec`` (deterministic: same spec, same bytes); ``gap`` (m) as the geoms' contact gap,
    ``margin`` the member joints' limit margin (0 writes none)."""
    return _text(spec, comment, gap, joint_impedance(spec, margin))


def _text(spec: LatchSpec, comment: str, gap: float | None, imp: dict | None) -> str:
    parts = [ROOT] + [m.name for m in spec.members]
    note = comment or f"alloy.latch spec {spec.name} (generated; edit the spec, not this file)"
    out = [f"<!-- {escape(note)} -->", f'<mujoco model="{spec.name}">', '  <compiler angle="radian"/>']
    if gap is not None:
        out += ["  <default>", f'    <geom gap="{FMT.format(gap)}"/>', "  </default>"]
    out.append("  <worldbody>")
    if spec.static and not spec.members:      # plain world geoms: the assembler's STATIC route takes them as-is
        out += [f"    {_geom(b)}" for b in spec.root_shapes]
    else:
        out += _body(spec, ROOT, 4, imp)
    out.append("  </worldbody>")
    if len(parts) > 1:
        out.append("  <contact>")
        out += [f'    <exclude body1="{body_name(spec, a)}" body2="{body_name(spec, b)}"/>'
                for a, b in combinations(parts, 2)]
        out += [f"    {_pair(spec, c)}" for c in spec.contacts]
        out.append("  </contact>")
    if spec.couplings:
        out.append("  <equality>")
        for c in spec.couplings:
            coef = tuple(c.polycoef) + (0.0,) * (5 - len(c.polycoef))
            out.append(f'    <joint joint1="{joint_name(spec, c.follower)}" joint2="{joint_name(spec, c.leader)}" '
                       f'polycoef="{_v(coef[:5])}"/>')
        out.append("  </equality>")
    out.append("</mujoco>")
    return "\n".join(out) + "\n"


def mjcf_masses(spec: LatchSpec) -> dict[str, float]:
    """Body name -> mass written to the MJCF (regularised members included), for card checks."""
    out = {body_name(spec, ROOT): float(root_props(spec).mass)}
    for m in spec.members:
        out[body_name(spec, m.name)] = float(member_props(m, spec.design_dt).mass)
    return out


def total_mass(spec: LatchSpec) -> float:
    return float(np.sum(list(mjcf_masses(spec).values())))
