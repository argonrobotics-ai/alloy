# SPDX-License-Identifier: Apache-2.0
"""``IOModel`` to Alloy cards: one object card (and its material card) per object, robot references, one scene.

Card choice per group: one primitive or one mesh collision shape becomes a ``Primitive`` or ``MeshSource`` card;
several shapes on one body, or jointed bodies without drives, become an ``MJCFSource`` card over a generated MJCF;
groups with driven joints become ``RobotRef``s. Values read from a file are tagged ``source: model`` with a note
naming the file (an author's value, not a measurement); values the file does not have stay ``default``, so the
physics check lists them.

Example::

    from alloy.io.importer import read_file
    from alloy.io.to_cards import write_cards
    res = write_cards(read_file("mug.urdf"), "my_library", source="mug.urdf")
    print(res.objects, res.scene)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from alloy.cards import types as ct
from alloy.core.quantity import Q
from alloy.io.from_model import clean_name
from alloy.io.geometry import compose, shape_volume, write_obj
from alloy.io.groups import Group, split_groups
from alloy.io.model import IOBody, IOMaterial, IOModel, IOShape, Pose, Unmapped
from alloy.io.usd_text import num

log = logging.getLogger("alloy.io")
PRIMITIVES = ("box", "sphere", "capsule", "cylinder", "cone", "ellipsoid", "plane")
ROUND = ("sphere", "capsule", "cylinder", "ellipsoid")   # shapes a rolling lever applies to
DENSITY_RANGE = (10.0, 25_000.0)          # kg/m^3, the card validator's range
MUJOCO_LEVERS = (0.005, 0.0001)           # MuJoCo's default torsional/rolling friction: not an author value


@dataclass
class ImportResult:
    """Cards written into a library folder, the scene that places them, and what did not map."""

    name: str
    library: str
    objects: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    robots: list[dict] = field(default_factory=list)
    scene: str | None = None
    files: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    unmapped: list[Unmapped] = field(default_factory=list)

    def to_json(self) -> dict:
        """Plain data for agents and the command line."""
        return {"name": self.name, "library": self.library, "objects": self.objects, "materials": self.materials,
                "robots": self.robots, "scene": self.scene, "files": self.files, "issues": self.issues,
                "unmapped": [u.as_dict() for u in self.unmapped]}


def _q(value: float, src: str, what: str) -> Q:
    return Q(float(value), "model", note=f"imported from {src}: {what}; the file author's value, not a measurement "
                                         "(replace it with a sourced value)")


def tidy(values) -> tuple[float, ...]:
    """Values rounded to their shortest float32 text (the engine's precision): 0.30000001192 -> 0.3."""
    return tuple(float(num(v)) for v in values)


def card_name(text: str) -> str:
    """Lower-case card name: ``card_name("My Mug.01") == "my_mug_01"``."""
    return clean_name(text, "object").lower()


@dataclass
class _Ctx:
    io: IOModel
    out: Path
    src: str                    # source file name for notes
    src_path: Path              # the source file
    fmt: str                    # "mjcf" | "urdf" | "usd"
    prefix: str
    res: ImportResult


def _density(ctx: _Ctx, g: Group, name: str) -> Q:
    """Uniform density reproducing the group's mass from its collision volume (default 1000 if unusable)."""
    mass = sum(b.mass for b in g.bodies)
    vol = sum(shape_volume(s) for b in g.bodies for s in b.shapes if s.collide)
    if not g.bodies or mass <= 0 or vol <= 0:
        return Q(1000.0, "default")
    rho = mass / vol
    if not DENSITY_RANGE[0] <= rho <= DENSITY_RANGE[1]:
        ctx.io.note("cards", name, f"file mass {mass:.4g} kg over collision volume {vol:.4g} m^3 is {rho:.4g} kg/m^3, "
                                   f"outside {DENSITY_RANGE}: check the collision mesh is closed; density left default")
        return Q(1000.0, "default")
    return _q(rho, ctx.src, f"mass {mass:.6g} kg / collision volume {vol:.6g} m^3")


def _surface(ctx: _Ctx, m: IOMaterial | None, rolls: bool = True) -> ct.Surface:
    """Card surface from a file material; ``rolls`` False drops a rolling lever (only round shapes roll)."""
    if m is None:
        return ct.Surface(Q(0.5, "default"), Q(0.5, "default"), Q(0.0, "default"))
    mu_d = _q(m.mu_dynamic, ctx.src, "friction")
    mu_s = _q(m.mu_static, ctx.src, "static friction") if m.mu_static != m.mu_dynamic else \
        Q(m.mu_dynamic, "model", note=f"imported from {ctx.src}: the file has one friction; static set equal")
    e = _q(m.restitution, ctx.src, "restitution") if m.restitution > 0 else Q(0.0, "default")
    levers = (m.torsional_lever, m.rolling_lever)
    keep = [lv if lv > 0 and abs(lv - d) > 1e-9 else None for lv, d in zip(levers, MUJOCO_LEVERS)]
    if not rolls:
        keep[1] = None
    return ct.Surface(mu_s, mu_d, e,
                      torsional_lever=_q(keep[0], ctx.src, "torsional friction lever") if keep[0] else None,
                      rolling_lever=_q(keep[1], ctx.src, "rolling friction lever") if keep[1] else None,
                      contact_stiffness=_q(m.contact_stiffness, ctx.src, "contact stiffness")
                      if m.contact_stiffness else None)


def _write_card(ctx: _Ctx, card, kind: str) -> Path:
    from alloy.cards.loader import dump_card  # noqa: PLC0415

    path = ctx.out / kind / f"{card.name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_card(card, header=f"Imported by alloy.io from {ctx.src}."))
    ctx.res.files.append(str(path))
    return path


def _material(ctx: _Ctx, g: Group, name: str, shapes: list[IOShape]) -> str:
    mats = [ctx.io.materials.get(s.material) for s in shapes if s.collide]
    mats = [m for m in mats if m is not None]
    if len({m.name for m in mats}) > 1:
        ctx.io.note("cards", name, f"{len({m.name for m in mats})} surface materials on one object: the first is used")
    rolls = bool(shapes) and all(s.kind in ROUND for s in shapes)
    card = ct.MaterialCard(name=f"{name}_material", density=_density(ctx, g, name),
                           surface=_surface(ctx, mats[0] if mats else None, rolls))
    _write_card(ctx, card, "materials")
    ctx.res.materials.append(card.name)
    return card.name


def _asset(ctx: _Ctx, filename: str) -> tuple[Path, str]:
    rel = f"assets/{ctx.prefix}/{filename}"
    return ctx.out / rel, rel


def _single_shape(ctx: _Ctx, s: IOShape, name: str) -> ct.ShapeSource:
    if s.kind in PRIMITIVES:
        return ct.Primitive(s.kind, tidy(s.size))
    path, rel = _asset(ctx, f"{name}.obj")
    digest = write_obj(s.mesh, path)
    ctx.res.files.append(str(path))
    return ct.MeshSource(rel, digest, collision="convex" if s.convex else "mesh")


def _mjcf_shape(ctx: _Ctx, g: Group, name: str, density: float | None, robot: bool) -> tuple[Path, str]:
    from alloy.io.mjcf_write import write_group_mjcf  # noqa: PLC0415

    path, rel = _asset(ctx, f"{name}.xml")
    write_group_mjcf(ctx.io, g, path, robot=robot, density=density)
    ctx.res.files.append(str(path))
    return path, rel


def _motion(ctx: _Ctx, g: Group) -> str:
    if g.root is None:
        return "static"
    if g.root.motion == "kinematic":
        return "kinematic"
    return "static" if g.fixed_base and not g.articulated else "dynamic"


def object_card(ctx: _Ctx, g: Group) -> tuple[ct.ObjectCard, Pose]:
    """The object card of a non-robot group and its placement pose."""
    name = card_name(f"{ctx.prefix}_{g.name}" if not g.name.startswith(ctx.prefix) else g.name)
    body: IOBody | None = g.root
    shapes = [s for s in (body.shapes if body is not None else g.world_shapes) if s.collide]
    if len(shapes) < len(body.shapes if body is not None else g.world_shapes):
        ctx.io.note("cards", name, "visual-only shapes dropped: the card renders its collision shape")
    mat = _material(ctx, g, name, shapes)
    if not g.articulated and len(g.bodies) <= 1 and len(shapes) == 1:
        base = body.pose if body is not None else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        shape, pose = _single_shape(ctx, shapes[0], name), compose(base, shapes[0].pose)
    else:
        rho = _density(ctx, g, name).value
        _, rel = _mjcf_shape(ctx, g, name, rho, robot=False)
        shape, pose = ct.MJCFSource(rel), g.base_pose()
    if body is not None and len(g.bodies) == 1 and any(abs(c) > 1e-6 for c in body.com) and len(shapes) == 1:
        ctx.io.note("cards", name, "the file's centre of mass is off the shape centre; the card uses uniform density")
    return ct.ObjectCard(name=name, shape=shape, material=mat, motion=_motion(ctx, g)), pose


def robot_ref(ctx: _Ctx, g: Group, whole_file: Path | None) -> ct.RobotRef:
    """A robot reference: the original file when it holds only this robot, else a generated MJCF."""
    name = card_name(f"{ctx.prefix}_{g.name}" if not g.name.startswith(ctx.prefix) else g.name)
    if whole_file is not None:
        return ct.RobotRef(str(whole_file.resolve()), ctx.fmt, tidy(g.base_pose()) if ctx.fmt == "urdf"
                           else (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), name)
    _, rel = _mjcf_shape(ctx, g, f"{name}_robot", None, robot=True)
    ctx.io.note("cards", name, "robot taken out of a larger file into a generated MJCF: tendons, sensors, joint "
                               "couplings, non-position actuators, contact excludes and collision masks are not "
                               "kept, and it starts at joint positions 0. Run sim.check(): a start_overlap between "
                               "two robot links means the source excluded that pair")
    return ct.RobotRef(str((ctx.out / rel).resolve()), "mjcf", tidy(g.base_pose()), name)


def _with_display_layer(src: Path, tmp: Path) -> Path:
    """The source with its display-mesh layer written in (a robot pack's ``visuals/`` folder,
    :mod:`alloy.scene.robot_visuals`), so the cut-out robot renders as the pack does; the source itself otherwise."""
    from alloy.scene import robot_visuals  # noqa: PLC0415

    if not robot_visuals.has_visuals(src):
        return src
    try:
        return robot_visuals.scene_with_visuals(src, cache=tmp)
    except (KeyError, ValueError, OSError) as exc:
        log.warning("%s: display-mesh layer not carried (%s)", src, exc)
        return src


def pruned_robots(ctx: _Ctx, robots: list[Group]) -> tuple[ct.RobotRef, bool] | None:
    """The robots of an MJCF scene as one robot file pruned from the source (:mod:`alloy.io.mjcf_prune`): excludes,
    couplings, tendons, actuators and keyframes stay as written. Returns the reference and whether the world's own
    geoms stayed in that file; None (with a note) when the file cannot be pruned."""
    import tempfile  # noqa: PLC0415
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    from alloy.io.mjcf_prune import PruneError, prune_mjcf  # noqa: PLC0415

    name = card_name(f"{ctx.prefix}_{robots[0].name}" if len(robots) == 1 else f"{ctx.prefix}_robots")
    path, _ = _asset(ctx, f"{name}.xml")
    try:
        tops = [el.get("name") for el in ET.parse(ctx.src_path).getroot().find("worldbody") if el.tag == "body"]
        by_clean = {clean_name(n, ""): n for n in tops if n}
        keep = {by_clean[g.root.name] for g in robots}
        with tempfile.TemporaryDirectory() as tmp:
            rep = prune_mjcf(_with_display_layer(ctx.src_path, Path(tmp)), keep, path)
    except (PruneError, KeyError, AttributeError, ET.ParseError) as exc:
        ctx.io.note("cards", name, f"the robot could not be cut from the source file ({exc}); a generated file is "
                                   "used instead")
        return None
    ctx.res.files.append(str(path))
    for kind in ("equality", "actuator"):           # the reader's "dropped" rows: these stayed in the robot file
        if not rep.removed.get(kind):
            ctx.io.unmapped[:] = [u for u in ctx.io.unmapped if not u.item.startswith(f"{kind} ")]
    for note in rep.notes:
        ctx.io.note("cards", name, note)
    return ct.RobotRef(str(path.resolve()), "mjcf", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), name), rep.kept_world


def is_robot(g: Group) -> bool:
    """A group with driven joints (or marked robot by the reader)."""
    return g.kind == "robot" or any(j.actuated or j.stiffness > 0 for j in g.joints)


def write_cards(io: IOModel, out_dir: str | Path, *, source: str | Path, fmt: str,
                prefix: str | None = None) -> ImportResult:
    """Write cards for every group of ``io`` into the library folder ``out_dir`` and a scene card placing them."""
    out = Path(out_dir)
    src = Path(source)
    pre = card_name(prefix or src.stem)
    res = ImportResult(pre, str(out))
    ctx = _Ctx(io, out, src.name, src, fmt, pre, res)
    groups = split_groups(io)
    placements, robots = [], []
    robot_groups = [g for g in groups if is_robot(g)]
    cut = pruned_robots(ctx, robot_groups) if fmt == "mjcf" and robot_groups and len(groups) > 1 else None
    if cut is not None:
        robots.append(cut[0])
    for g in groups:
        if is_robot(g):
            if cut is None:
                robots.append(robot_ref(ctx, g, src if len(groups) == 1 else None))
            continue
        if cut is not None and cut[1] and g.root is None:
            continue                                    # the world's own geoms stayed in the robot file
        card, pose = object_card(ctx, g)
        _write_card(ctx, card, "objects")
        res.objects.append(card.name)
        placements.append(ct.Placement(card.name, tidy(pose)))
    res.robots = [{"name": r.name, "path": r.path, "kind": r.kind, "base_pose": list(r.base_pose)} for r in robots]
    res.scene = str(_write_scene(ctx, ct.SceneCard(robots=tuple(robots), objects=tuple(placements), worlds=1)))
    res.unmapped = list(io.unmapped)
    return res


def _write_scene(ctx: _Ctx, scene: ct.SceneCard) -> Path:
    from alloy.cards.loader import dump_card  # noqa: PLC0415

    path = ctx.out / "scenes" / f"{ctx.prefix}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_card(scene, header=f"Imported by alloy.io from {ctx.src}."))
    ctx.res.files.append(str(path))
    return path
