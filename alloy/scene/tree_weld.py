# SPDX-License-Identifier: Apache-2.0
"""Chain-end welds as fixed tree joints (SCR-WP2-9, coordinator ruling at the end of [internal note]).

An unbreakable scene weld (no break_force / break_strain) between a CHAIN object's root link (link 0) and a body is
not written as a MuJoCo weld equality. The chain is emitted with its root link fixed (`ChainOptions.clamp`), and that
fixed joint is moved under the other body at the placed relative pose. The chain then hangs off the body in the
kinematic tree, so the weld has no equality row and its separation is zero by construction. The near-hard weld row of
SCR-DC3-3 (solimp 0.9999) did not condition in float32 on 0.08-0.65 g links: T09's fast profile had 19/64 invalid
worlds and its standard profile ran at half speed. A static other end (body -1) leaves the root fixed to the world.

Rules:
- at most one end per chain takes the tree path: the first qualifying weld in scene order. A second weld on the same
  chain (a hose welded at both ends) stays an equality row, which closes the loop (`alloy.scene.weld_impedance`);
- the weld must land on link 0. It names ":0", or the nearest link to its anchor is link 0, found from a dry emission
  of the chain. Otherwise the weld stays an equality row and the route report gives the reason;
- Newton keeps each articulation's joints contiguous, so a tree chain is emitted right after its parent object
  (`emission_order`), and the chain's articulation is folded into the parent's;
- the chain's entry must be the MuJoCo entry, as for any weld;
- link 0's child joints (the chain's joint 1) carry no armature when the parent is a body (SCR-DC3-14, coordinator
  ruling 2026-11-05 20:01Z in [internal note]). MuJoCo adds armature to the mass matrix diagonal only, so a
  joint-1 armature acts against the parent's own rotation: DC3's 1e-5 made 88 % of the LC plug's pitch inertia
  (8.2e-6 kg m² vs 9.7e-7 from the cord's masses; 1.0e-6 without it), and the root-torque peaks under DC3's unlatch
  drag were that armature term. `attach` zeroes it at emission and labels the fixed joint (`TREE_LABEL`), and a
  MuJoCo entry zeroes it again right before its solver is built (`rigid_armature_off`), after scene-level armature
  overrides (`alloy.integration.throughput.chain_armature`). The other chain joints keep theirs.
- the parent's bodies keep the contact weight (``body_invweight0``) they have on the equality path
  (`alloy.scene.tree_invweight`, SCR-DC3-14 part 2): one physics for both weld paths;
- welded bodies do not collide (SCR-DC3-14 part 3, coordinator ruling 2026-11-06 03:07Z). The parent body and link 0
  get shape filter pairs here (`filter_welded`), and an equality-row weld on a chain gets them in
  `alloy.scene.attachments`, before the explicit MuJoCo pair rows are built (`alloy.scene.mujoco_pairs` skips
  filtered pairs). DC3's LC cord root sat 0.1 mm inside its own boot: a 0.25 N preload on the equality row, inert on
  the tree (zero relative Jacobian).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import Attachment
from alloy.core.errors import AssemblyError
from alloy.route.types import Route
from alloy.scene.template import TemplateObject, pose_transform

log = logging.getLogger("alloy.scene")
TREE_NUMBER = "tree_root"        # decision number on a chain emitted with its root fixed for a tree weld
TREE_LABEL = "alloy_tree_weld"   # label suffix of a tree weld's fixed joint under a body (rigid_armature_off)


@dataclass(frozen=True)
class TreeWeld:
    att: Attachment
    chain: str                   # the chain object's name
    chain_ref: str               # the attachment's chain endpoint ("chain" or "chain:0")
    parent_ref: str              # the other endpoint ("object[:link]")

    @property
    def parent(self) -> str:
        return self.parent_ref.partition(":")[0]


def _obj(ref: str) -> str:
    return ref.partition(":")[0]


def candidates(objs: Sequence[TemplateObject], atts: Sequence[Attachment]) -> dict[str, TreeWeld]:
    """Per chain, the first unbreakable weld (scene order) between its root and a non-chain object."""
    by = {t.name: t for t in objs}
    out: dict[str, TreeWeld] = {}
    for att in atts:
        if att.kind != "weld" or att.break_force is not None or att.break_strain is not None:
            continue
        refs = [r for r in (att.a, att.b) if _obj(r) in by]
        chains = [r for r in refs if by[_obj(r)].decision.route == Route.CHAIN]
        if len(refs) != 2 or len(chains) != 1 or _obj(att.a) == _obj(att.b):
            continue
        cref = chains[0]
        chain = by[_obj(cref)]
        if chain.name in out or cref.partition(":")[2] not in ("", "0") or chain.decision.entry != "mujoco":
            continue
        out[chain.name] = TreeWeld(att, chain.name, cref, att.b if cref == att.a else att.a)
    return out


def emission_order(objs: list[TemplateObject], tree: dict[str, TreeWeld]) -> list[TemplateObject]:
    """Scene order, with each tree chain moved right after its parent (and after earlier chains of that parent)."""
    if not tree:
        return list(objs)
    moved = {c for c, tw in tree.items() if tw.parent in {t.name for t in objs}}
    out = [t for t in objs if t.name not in moved]
    by = {t.name: t for t in objs}
    for c in [t.name for t in objs if t.name in moved]:              # scene order among chains
        parent = tree[c].parent
        i = next(k for k, t in enumerate(out) if t.name == parent)
        while i + 1 < len(out) and out[i + 1].name in moved and tree[out[i + 1].name].parent == parent:
            i += 1
        out.insert(i + 1, by[c])
    return out


def _predicted_links(t: TemplateObject, cap, make_sub) -> np.ndarray:
    """World COM of each chain link, from a dry emission of the chain at its placement."""
    sub = make_sub()
    cap.emitter.emit(sub, t.obj, t.decision)
    x = pose_transform(t.placement.pose)
    return np.asarray([wp.transform_point(x, wp.vec3(*sub.body_q[i][:3])) for i in range(sub.body_count)], np.float64)


def _target(b: newton.ModelBuilder, objs: dict[str, TemplateObject], tw: TreeWeld) -> np.ndarray | None:
    """The point the weld's chain end is chosen by (`attachments.endpoint`): the anchor in the parent's frame when the
    parent is `a`, else the chain's own placement origin."""
    from alloy.scene import attachments as A

    if tw.chain_ref == tw.att.a:
        return None
    a = A.endpoint(b, objs, tw.att.a, None)
    return A._anchor_world(a, tw.att)


def root_link(b: newton.ModelBuilder, objs: dict[str, TemplateObject], tw: TreeWeld, cap, make_sub) -> int:
    """Index of the chain link this weld lands on (0 = the root)."""
    if tw.chain_ref.endswith(":0"):
        return 0
    t = objs[tw.chain]
    links = _predicted_links(t, cap, make_sub)
    target = _target(b, objs, tw)
    if target is None:
        target = np.asarray(pose_transform(t.placement.pose).p, np.float64)
    return int(np.argmin(np.linalg.norm(links - target, axis=1)))


def mark_root(t: TemplateObject) -> None:
    """The chain emitter fixes link 0 (ChainOptions.clamp) when its decision carries TREE_NUMBER."""
    t.decision = replace(t.decision, numbers={**t.decision.numbers, TREE_NUMBER: 1.0})


def parent_body(b: newton.ModelBuilder, objs: dict[str, TemplateObject], tw: TreeWeld) -> int:
    from alloy.scene import attachments as A

    return A.endpoint(b, objs, tw.parent_ref, None).body


def parent_shapes(objs: dict[str, TemplateObject], tw: TreeWeld) -> Sequence[int]:
    """The parent object's own shapes: for a static parent (body −1) the welded-pair filter's other side."""
    return objs[tw.parent].ranges.shape


def attach(b: newton.ModelBuilder, chain: TemplateObject, body: int, parent_shapes: Sequence[int] = ()) -> int:
    """Move the chain's fixed root joint under `body` (world frame kept) and fold the articulations; returns it.
    `parent_shapes`: a static parent's own shapes (body −1), filtered against link 0 as a body parent's are."""
    j = chain.ranges.joint.start
    if b.joint_parent[j] != -1 or b.joint_type[j] != newton.JointType.FIXED:
        raise AssemblyError(f"{chain.name}: tree weld expects a fixed root joint at joint {j}")
    if body < 0:
        filter_welded(b, parent_shapes, b.body_shapes.get(b.joint_child[j], ()))
        return j
    x_world = wp.transform(*b.joint_X_p[j])
    b.joint_parent[j] = body
    b.joint_X_p[j] = wp.transform_multiply(wp.transform_inverse(wp.transform(*b.body_q[body])), x_world)
    b.joint_label[j] = f"{chain.name}/{TREE_LABEL}"
    link0 = b.joint_child[j]
    for k in (k for k, p in enumerate(b.joint_parent) if p == link0):
        for d in range(b.joint_qd_start[k], b.joint_qd_start[k] + sum(b.joint_dof_dim[k])):
            b.joint_armature[d] = 0.0
    filter_welded(b, b.body_shapes.get(body, ()), b.body_shapes.get(link0, ()))
    _merge(b, body, j)
    return j


def filter_welded(b: newton.ModelBuilder, shapes_a: Sequence[int], shapes_b: Sequence[int]) -> int:
    """No contact between two welded bodies (module docstring): a shape filter pair for every shape pair of the two.
    Call it before `alloy.scene.mujoco_pairs.add_native_pairs`, which skips filtered pairs; a row made earlier
    bypasses any later filter. Returns the pairs added."""
    pairs = [(int(sa), int(sb)) for sa in shapes_a for sb in shapes_b if int(sa) != int(sb)]
    for sa, sb in pairs:
        b.add_shape_collision_filter_pair(sa, sb)
    if pairs:
        log.info("weld: %d shape pairs of the welded bodies filtered (SCR-DC3-14)", len(pairs))
    return len(pairs)


def rigid_armature_off(model) -> int:
    """Zero the armature of every joint whose parent is a tree weld's link 0 (module docstring), in place on `model`
    (a Model or a ModelView of it), all worlds. Returns the dofs that had armature."""
    labels = [str(x) for x in (getattr(model, "joint_label", None) or ())]
    roots = [k for k, lab in enumerate(labels) if lab.endswith("/" + TREE_LABEL)]
    if not roots:
        return 0
    child, parent, qd = (a.numpy() for a in (model.joint_child, model.joint_parent, model.joint_qd_start))
    links = {int(child[k]) for k in roots}
    dofs = [d for k in range(len(parent)) if int(parent[k]) in links for d in range(int(qd[k]), int(qd[k + 1]))]
    arm = model.joint_armature.numpy()
    hit = int(np.count_nonzero(arm[dofs]))
    if hit:
        arm[dofs] = 0.0
        model.joint_armature.assign(arm)
        log.info("tree weld: armature off on %d link-0 child dofs (%d tree welds, SCR-DC3-14)", hit, len(roots))
    return hit


def adjacent(b: newton.ModelBuilder, body: int) -> bool:
    """True when `body` is in the builder's last articulation (the chain emitted next becomes adjacent)."""
    if body < 0:
        return True
    js = [k for k, c in enumerate(b.joint_child) if c == body]
    return bool(js) and b.joint_articulation[js[0]] == len(b.articulation_start) - 1


def _merge(b: newton.ModelBuilder, body: int, j: int) -> None:
    """Fold the chain's articulation (the last) into the parent's (the one before it)."""
    ka = b.joint_articulation[next(k for k, c in enumerate(b.joint_child) if c == body)]
    kc = b.joint_articulation[j]
    if kc != ka + 1 or b.articulation_end[ka] != b.articulation_start[kc]:
        raise AssemblyError(f"tree weld: articulations {ka} and {kc} are not adjacent")
    b.articulation_end[ka] = b.articulation_end[kc]
    for k, art in enumerate(b.joint_articulation):
        if art == kc:
            b.joint_articulation[k] = ka
        elif art > kc:
            b.joint_articulation[k] = art - 1
    for name in ("articulation_start", "articulation_end", "articulation_label", "articulation_world"):
        del getattr(b, name)[kc]
    starts = getattr(b, "articulation_world_start", None)
    if starts:
        starts[:] = [s - 1 if s > kc else s for s in starts]
    _remap_attributes(b, ka, kc)


def _remap_attributes(b: newton.ModelBuilder, ka: int, kc: int) -> None:
    """Custom attributes that reference articulations, or are stored per articulation, follow the fold."""
    art = newton.Model.AttributeFrequency.ARTICULATION
    new = lambda v: v if v is None or int(v) < kc else (ka if int(v) == kc else int(v) - 1)   # noqa: E731
    for attr in b.custom_attributes.values():
        ref = newton.Model._normalize_attribute_reference(attr.references) if attr.references else None
        vals = attr.values
        if vals is None:
            continue
        if ref == art:
            attr.values = [new(v) for v in vals] if isinstance(vals, list) else {k: new(v) for k, v in vals.items()}
            vals = attr.values
        if attr.frequency == art:
            if isinstance(vals, list):
                if len(vals) > kc:
                    del vals[kc]
            else:
                attr.values = {(k if k < kc else k - 1): v for k, v in vals.items() if k != kc}


def report_line(tw: TreeWeld, body: int) -> str:
    where = "the world" if body < 0 else f"{tw.parent_ref} (body {body})"
    return (f"mujoco weld {tw.att.a}–{tw.att.b}: tree joint, {tw.chain} link 0 fixed to {where}; no equality row, "
            "separation 0 (SCR-WP2-9)")


__all__ = ["TREE_LABEL", "TREE_NUMBER", "TreeWeld", "adjacent", "attach", "candidates", "emission_order",
           "filter_welded", "mark_root", "parent_body", "parent_shapes", "report_line", "rigid_armature_off",
           "root_link"]
