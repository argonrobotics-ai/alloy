# SPDX-License-Identifier: Apache-2.0
"""``alloy.Scene``: describe a scene in Python (objects, robots, cloth, liquid, materials), then ``alloy.build`` it.

A Scene is plain data: ``to_dict()`` / ``Scene.from_dict()`` round-trip through JSON, and ``save(dir)`` writes the
engine's YAML cards. Nothing heavy is imported until ``alloy.build``. Units are SI (metres, kilograms, seconds), z is
up, gravity is (0, 0, -9.81). Sizes are full edge lengths (not half extents) unless an argument says ``half_``.

Example::

    import alloy
    scene = alloy.Scene(dt=0.01)                      # dt: the control period (sim.step() advances one of them)
    scene.add_ground()
    scene.add_box("table", size=(1.0, 1.0, 0.75), pos=(0, 0, 0.375), static=True)
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0, 0, 1.0), mass=1.0, friction=0.8)
    print(scene.summary())
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alloy.api import _args as A, _contact_relax, catalog
from alloy.api._cloth_route import add_cloth_route
from alloy.api.errors import SceneError
from alloy.api import _heat, _liquid, _liquid_accuracy, _quasi_static, _random, _rigid_step, _robots, _rods, _worlds
from alloy.api._items import _box_volume, _cloth_vertices, _mesh_spec, _overrides
from alloy.api._heat import HeatMixin
from alloy.api._joint_yield import JointYieldMixin
from alloy.scene import cloth_accuracy as CA
from alloy.api.fixtures import FixtureMixin

RIGID_MATERIAL = "abs"
GROUND_MATERIAL = "plywood_marine"
CLOTH_MATERIAL = "cotton_woven"
LIQUID_MATERIAL = "water"
SPH_SPACING = 4.0e-3       # m, liquid_accuracy "default"'s spacing: add_pour sizes pours in it (the build re-sizes)
COLLISIONS = ("convex", "decompose", "sdf", "mesh")
RANDOM_FIELDS = _random.FIELDS     # short names accepted by Scene.randomize -> card paths


class Scene(FixtureMixin, HeatMixin, JointYieldMixin, _quasi_static.QuasiStaticMixin, _rigid_step.RigidStepMixin,
            _contact_relax.ContactRelaxMixin):
    """A scene under construction. ``worlds``: parallel copies; ``dt``: control period in seconds; ``roots``: your
    own card folders (``materials/``, ``objects/``, ...: e.g. cards made by ``alloy.create`` or ``alloy.io``),
    searched before the shipped and imported libraries; ``liquid_accuracy``: "default" (liquid particles 4 mm apart)
    or "super" (2 mm, about 16x the cost; a build warns when the default is too coarse for an opening or a stream);
    ``cloth_accuracy``: "default", "super" (nearer convergence, 1.4x the cloth cost) or "fast" (the cloth solver's own
    10 iterations, 0.4x the default's cloth cost, CPU 0.25x; a drape stretches 12 %); soft bodies use it too. Example:
    ``s = alloy.Scene(worlds=64); s.add_ground(); s.add_sphere("ball", 0.05, pos=(0, 0, 0.3))``."""

    def __init__(self, *, worlds: int = 1, dt: float = 0.01, roots: Sequence[str | Path] = (),
                 liquid_accuracy: str = "default", cloth_accuracy: str = "default") -> None:
        if not isinstance(worlds, int) or worlds < 1:
            raise SceneError(f"worlds must be an int >= 1, got {worlds!r}", "pass worlds=1 (or more)", field="worlds")
        self.worlds, self.dt = worlds, A.positive(dt, "dt", "s")
        self.liquid_accuracy = liquid_accuracy           # SPH spacing 4 mm, or 2 mm "super" (checked: the property)
        self.cloth_accuracy = cloth_accuracy             # the VBD entry's solve (alloy.scene.cloth_accuracy)
        self.items: dict[str, dict[str, Any]] = {}   # name -> one placed object / robot (plain data)
        self.materials: dict[str, dict[str, Any]] = {}   # custom material cards (data)
        self.pairs: list[dict[str, Any]] = []         # pair friction overrides between objects
        self.roots: list[str] = [str(Path(r).expanduser()) for r in roots]   # your card folders, searched first
        self.extra: dict[str, Any] = {}               # scene-card fields kept from alloy.load (attachments, emitters)
        self.thermal: dict[str, Any] = _heat.empty()  # heat: start temperatures, heaters, settings (Scene.set_heat)
        self.joint_yield: dict[str, dict[str, Any]] = {}  # joint springs that yield (Scene.set_joint_yield)
        self.robot_random: list[dict[str, Any]] = []      # robot / gravity draws (Scene.randomize, _robot_random)
        self.pushes: list[dict[str, Any]] = []            # timed pushes (add_push, _robot_random_push)

    @property
    def liquid_accuracy(self) -> str:
        """"default" (liquid particles 4 mm apart) or "super" (2 mm, about 16x the cost); set it before building."""
        return self._liquid_accuracy

    @liquid_accuracy.setter
    def liquid_accuracy(self, value: str) -> None:
        self._liquid_accuracy = _liquid_accuracy.checked(value)     # a bad value fails here, not at build

    cloth_accuracy = CA.scene_property(SceneError)   # "default" / "super" / "fast" (alloy.scene.cloth_accuracy)

    # --- rigid primitives -----------------------------------------------------------------------------------

    def add_ground(self, name: str = "ground", *, z: float = 0.0, material: str = GROUND_MATERIAL,
                   friction: float | None = None) -> str:
        """An infinite static plane at height ``z``. Example: ``scene.add_ground()``."""
        shape = {"primitive": {"kind": "plane", "size": [50.0, 50.0]}}
        pos = [0.0, 0.0, float(z)]
        return self._add(name, "ground", shape, pos, None, material=material, static=True, friction=friction)

    def add_box(self, name: str, size: Sequence[float] | None = None, *, half_extents: Sequence[float] | None = None,
                pos: Any = (0.0, 0.0, 0.0), **kw: Any) -> str:
        """A box. ``size``: full edge lengths (x, y, z) in m; or ``half_extents``. Keywords: see ``add_sphere``.

        Example: ``scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0, 0, 1), mass=1.0)``."""
        if (size is None) == (half_extents is None):
            raise SceneError("add_box needs exactly one of size / half_extents",
                             "pass size=(x, y, z) full edge lengths in metres", field="size")
        half = [0.5 * s for s in A.sizes(size, 3, "size")] if size is not None else A.sizes(half_extents, 3,
                                                                                            "half_extents")
        return self._add(name, "box", {"primitive": {"kind": "box", "size": half}}, pos, _box_volume(half), **kw)

    def add_sphere(self, name: str, radius: float, *, pos: Any = (0.0, 0.0, 0.0), quat_xyzw: Any = None,
                   quat_wxyz: Any = None, rpy: Any = None, material: str = RIGID_MATERIAL, mass: float | None = None,
                   density: float | None = None, friction: float | None = None, restitution: float | None = None,
                   static: bool = False, rigid: bool | None = None, kinematic: bool = False) -> str:
        """A sphere. Shared keywords of every ``add_*``:

        - ``pos``: (x, y, z) in m, or (worlds, 3) for one start per world;
        - orientation: one of ``quat_xyzw``, ``quat_wxyz`` or ``rpy`` (roll, pitch, yaw in radians);
        - ``material``: a name from ``alloy.materials()`` (or ``add_material``); ``mass`` (kg) or ``density``
          (kg/m^3), ``friction`` (Coulomb mu) and ``restitution`` override it for this object only;
        - ``static=True``: fixed in place (tables, walls, bins);
        - ``kinematic=True``: moved only by ``sim.set_pose`` / ``sim.drive`` (a tilting tray, a moving wall); other
          objects, liquids and cloth feel it, gravity and contacts do not move it;
        - ``rigid=True``: always a rigid body. Without it a soft material (rubber, foam) can make the object a soft
          body (``sim.routes()`` says which), which is far slower; ``rigid=False`` asks for a soft body.

        Example: ``scene.add_sphere("ball", 0.05, pos=(0, 0, 0.5), material="steel_carbon")``."""
        r = A.positive(radius, "radius", "m")
        return self._add(name, "sphere", {"primitive": {"kind": "sphere", "size": [r]}}, pos, 4.0 / 3.0 * math.pi * r ** 3,
                         quat_xyzw=quat_xyzw, quat_wxyz=quat_wxyz, rpy=rpy, material=material, mass=mass,
                         density=density, friction=friction, restitution=restitution, static=static, rigid=rigid,
                         kinematic=kinematic)

    def add_cylinder(self, name: str, radius: float, height: float, *, pos: Any = (0.0, 0.0, 0.0), **kw: Any) -> str:
        """A solid cylinder along its local z axis; ``height`` is the full height. Keywords: see ``add_sphere``.

        Example: ``scene.add_cylinder("can", 0.033, 0.12, pos=(0, 0, 0.06), mass=0.35)``."""
        r, h = A.positive(radius, "radius", "m"), A.positive(height, "height", "m")
        shape = {"primitive": {"kind": "cylinder", "size": [r, 0.5 * h]}}
        return self._add(name, "cylinder", shape, pos, math.pi * r * r * h, **kw)

    def add_capsule(self, name: str, radius: float, height: float, *, pos: Any = (0.0, 0.0, 0.0), **kw: Any) -> str:
        """A capsule along local z; ``height`` is the straight part between the two caps. See ``add_sphere``.

        Example: ``scene.add_capsule("rod", 0.01, 0.2, pos=(0, 0, 0.3))``."""
        r, h = A.positive(radius, "radius", "m"), A.positive(height, "height", "m")
        shape = {"primitive": {"kind": "capsule", "size": [r, 0.5 * h]}}
        return self._add(name, "capsule", shape, pos, math.pi * r * r * h + 4.0 / 3.0 * math.pi * r ** 3, **kw)

    def add_mesh(self, name: str, path: str | Path | None = None, *, vertices: Any = None, faces: Any = None,
                 scale: float | Sequence[float] = 1.0, collision: str = "convex", pos: Any = (0.0, 0.0, 0.0),
                 **kw: Any) -> str:
        """A rigid triangle mesh from an OBJ/STL file or from ``vertices`` (V, 3) and ``faces`` (F, 3) arrays.

        ``collision``: "convex" (hull; fast), "decompose" (convex parts), "sdf" (exact, for cups, bins, holes) or
        "mesh"; with "sdf" or "mesh" ``alloy.build`` uses Newton's contact pipeline (MuJoCo's own collision would
        fill the hollow). "decompose" splits the mesh into at most 32 convex parts with CoACD at build, once per mesh
        (cached under ``$ALLOY_CACHE_DIR/hulls``), or uses a ``<stem>.hulls.obj`` shipped beside the mesh file; it
        needs ``pip install "alloy-engine[import]"`` (CoACD, trimesh) and a closed mesh. For cups, bins and holes prefer ``add_open_box`` / ``add_open_cylinder`` /
        ``add_block_with_hole`` (static boxes: exact and fast). Faces wind counter-clockwise seen from outside.

        Example: ``scene.add_mesh("cup", vertices=v, faces=f, collision="sdf", static=True)``."""
        A.one_of(collision, COLLISIONS, "collision")
        mesh = _mesh_spec(path, vertices, faces)
        s = [float(scale)] * 3 if isinstance(scale, (int, float)) else A.sizes(scale, 3, "scale")
        shape = {"mesh": {"path": "", "sha256": "", "scale": s, "collision": collision}}
        return self._add(name, "mesh", shape, pos, None, mesh=mesh, **kw)

    # --- deformables, liquids, robots, library objects -----------------------------------------------------------

    def add_cloth(self, name: str, size: Sequence[float] = (1.0, 1.0), *, pos: Any = (0.0, 0.0, 0.5),
                  resolution: int | Sequence[int] = 21, thickness: float = 2e-3, material: str = CLOTH_MATERIAL,
                  mass: float | None = None, friction: float | None = None, quat_xyzw: Any = None,
                  quat_wxyz: Any = None, rpy: Any = None, self_contact: bool = True) -> str:
        """A cloth sheet, flat in its local x-y plane, centred at ``pos``. ``size``: (x, y) in m; ``resolution``:
        vertices along the longer side (or (nx, ny)); ``mass``: total kg (else from the material and thickness).
        ``thickness``: the contact thickness (particles of radius thickness / 2; a fabric bends as its card says at any
        thickness); the health monitor flags penetration past half of it, so thinner sheets need a smaller ``dt``.
        ``self_contact`` (default True): the sheet collides with itself, so a fold or a pile stays in layers
        ``thickness`` apart instead of passing through itself. With two or more cloth or soft objects in a scene they
        always collide with each other and with themselves. Turning it off makes a lone sheet cheaper to step (cost:
        ``docs/guide/deformables.md``) when it never touches itself (a flag, a sheet that hangs or lies flat).

        Example: ``scene.add_cloth("towel", size=(0.5, 0.5), pos=(0, 0, 0.6), resolution=25, mass=0.1)``."""
        lx, ly = A.sizes(size, 2, "size")
        t = A.positive(thickness, "thickness", "m")
        n = _cloth_vertices(resolution, lx, ly)
        shape = {"primitive": {"kind": "box", "size": [0.5 * lx, 0.5 * ly, 0.5 * t]}}
        thick = A.user_q(t, "sheet thickness set through alloy.Scene.add_cloth")
        if self_contact not in (True, False):
            raise SceneError(f"self_contact must be True or False, got {self_contact!r}",
                             "pass self_contact=True (a sheet that folds) or False", field="self_contact")
        feats = {"thickness": thick, "min_wall": thick, **({"self_contact": True} if self_contact else {})}
        extra = {"features": feats, "budget": {"particles": n}, **add_cloth_route(material, self.materials, self.roots)}
        return self._add(name, "cloth", shape, pos, lx * ly * t, quat_xyzw=quat_xyzw, quat_wxyz=quat_wxyz, rpy=rpy,
                         material=material, mass=mass, friction=friction, card_extra=extra)

    def add_rod(self, name: str, points: Any = None, *, radius: float, length: float | None = None,
                pos: Any = (0.0, 0.0, 0.0), direction: Sequence[float] = (1.0, 0.0, 0.0),
                material: str = _rods.ROD_MATERIAL, segments: int | None = None, clamp: str | None = None,
                mass: float | None = None, friction: float | None = None, route: str | None = None) -> str:
        """An elastic rod (a cable, wire or cord) of round section ``radius`` (m) along ``points`` (a polyline,
        relative to ``pos``), or ``length`` along ``direction`` from ``pos``. It bends, twists and stretches as its
        material says (E, nu), rolls on surfaces and can be gripped. ``segments``: rigid segments along it (>= 3;
        default: enough that it sags under its own weight within 5 % of the continuum). ``clamp``: "start", "end" or
        "both" holds that end where it is built; a clamped rod steps on the elastic-rod solver (``rod_vbd``) unless
        ``route="chain"`` (with clamp="start": far faster beside a robot; docs/guide/rods.md). Read its centre line
        as ``"<name>.points"``, its twist per joint as ``"<name>.twist"``; push or twist it with ``sim.apply_wrench``.

        Example: ``scene.add_rod("cable", length=0.5, radius=0.002, pos=(0, 0, 0.6), clamp="start")``."""
        if _worlds.per_world(pos):
            _rods.refuse_per_world(name)
        shape, extra, volume = _rods.spec(points, radius, length, direction, segments, clamp, route)
        self._add(name, "rod", shape, pos, volume, material=material, mass=mass, friction=friction, card_extra=extra)
        self.items[name]["rod"] = {"clamp": clamp, **({"route": route} if route else {})}
        return name

    def add_liquid(self, name: str, size: Sequence[float], *, pos: Any = (0.0, 0.0, 0.5),
                   material: str = LIQUID_MATERIAL, shape: str = "box", temperature: float | None = None,
                   heat_material: str | None = None) -> str:
        """Liquid (SPH particles) at rest, filling a box of full size (x, y, z) m centred at ``pos``, or with
        ``shape="cylinder"`` an upright cylinder of diameter x (= y) and height z: liquid already in a container, or
        a puddle. Match the container: in a round cup use ``shape="cylinder"``; a box in a round cup is not at rest
        (its corners stick out), collapses and jets over the rim. To fill a container use ``add_pour``: a block
        released in the air splashes several percent out when it lands (flat-impact pressure jets, real physics).

        The engine picks the particle size and fills the shape with whole particles (``sim.liquid_volume(name)``).
        ``temperature``: start temperature in K; the liquid then takes heat (``Scene.set_temperature``).
        Example: ``scene.add_liquid("water", size=(0.07, 0.07, 0.04), pos=(0, 0, 0.024), shape="cylinder")`` (in a
        round cup of inner radius 0.035 m at the origin)."""
        sx, sy, sz = A.sizes(size, 3, "size")
        _liquid.refuse_solid(material, self.materials, self.roots)
        kind = A.one_of(shape, ("box", "cylinder"), "shape")
        if kind == "box":
            half = [0.5 * sx, 0.5 * sy, 0.5 * sz]
            prim, volume = {"kind": "box", "size": half}, _box_volume(half)
        else:
            if abs(sx - sy) > 1e-9 * max(sx, sy):
                raise SceneError(f"a cylinder of liquid needs size x == y (its diameter), got {sx} and {sy}",
                                 f"pass size=({sx}, {sx}, {sz}) or shape=\"box\"", field="size")
            prim, volume = {"kind": "cylinder", "size": [0.5 * sx, 0.5 * sz]}, math.pi * 0.25 * sx * sx * sz
        name = self._add(name, "liquid", {"primitive": prim}, pos, volume, material=material)
        return self._heated(name, temperature, heat_material)

    def add_pour(self, name: str, volume: float, pos: Any, *, rate: float = 2.0e-4, speed: float = 0.5,
                 material: str = LIQUID_MATERIAL, temperature: float | None = None,
                 heat_material: str | None = None) -> str:
        """Pour ``volume`` m^3 of liquid from a nozzle at ``pos`` straight down at ``rate`` m^3/s, leaving at about
        ``speed`` m/s (a square nozzle of whole particles; the exit speed is set so the rate is exact). This is how to
        fill a container: a block of liquid dropped from above splashes several percent over the rim (flat-impact
        pressure jets, real physics), a stream does not. The liquid waits at the nozzle as whole particles, so it
        pours ``volume`` within half a particle (0.03 ml at the default liquid accuracy; ``sim.liquid_volume(name)``
        reads it). ``sim.state()["<name>.active"]`` marks the poured particles.
        The energy a pour brings in (and takes out when particles retire) is booked as external work, so the health
        monitor's energy check stays on. ``temperature``: the liquid's start temperature in K (it then takes heat).

        Example: ``scene.add_pour("water", 200e-6, pos=(0, 0, 0.15))`` (200 ml over 1 s into a cup below)."""
        v = A.positive(volume, "volume", "m^3")
        q, c = A.positive(rate, "rate", "m^3/s"), A.positive(speed, "speed", "m/s")
        if _worlds.per_world(pos):
            _worlds.refuse_static(name, "a pour (one nozzle)", "give one pos=(x, y, z); move the container per world")
        p0 = A.vector(pos, 3, "pos")
        s = SPH_SPACING
        k = max(1, round((q / c) ** 0.5 / s))               # nozzle: k x k particles per layer
        if k * s > 0.2:
            raise SceneError(f"a {q} m^3/s pour at {c} m/s needs a {k * s:.2f} m nozzle", "lower rate or raise speed",
                             field="rate")
        from alloy.api._pour import reservoir  # noqa: PLC0415

        (a, b, m), count = reservoir(v, s)                   # whole particles: `volume` within half a particle
        size = ((a + 0.5) * s, (b + 0.5) * s, (m + 0.5) * s)
        name = self.add_liquid(name, size=size, pos=p0, material=material)
        self.items[name]["pour"] = {"pos": p0, "rate": q, "volume": count * s ** 3, "asked": v, "count": count,
                                    "velocity": [0.0, 0.0, -q / (k * s) ** 2],
                                    "half_extents": [(k + 0.5) * s / 2, (k + 0.5) * s / 2, s / 2]}
        return self._heated(name, temperature, heat_material)

    def add_robot(self, name: str, path: str | Path | None = None, *, mjcf: str | None = None,
                  pos: Any = (0.0, 0.0, 0.0), quat_xyzw: Any = None, quat_wxyz: Any = None, rpy: Any = None,
                  contact_gap: float | None = None, sysid: Any = None) -> str:
        """A robot from an MJCF / URDF / USD file, or an MJCF string (``mjcf=``). Its base is welded to the world
        unless the file gives it a joint. Bodies and joints are named "<name>/<body or joint>" in the sim; MJCF
        ``<position>`` actuators follow ``sim.set_targets``. ``pos`` (worlds, 3) starts a free-base robot (MJCF
        ``<freejoint/>`` on its root body) at one position per world. ``contact_gap`` (m): how far apart its shapes
        start making contacts; default 5 mm as every scene object (an MJCF geom's own ``gap`` is kept); 0.1 gives
        the importer's old default, which fills the contact buffer in packed scenes. ``sysid``: a file from
        ``python -m alloy.sysid fit``, its dict or the fit's result (fitted joint parameters; docs/guide/sysid.md).

        Example: ``scene.add_robot("arm", mjcf=open("arm.xml").read(), pos=(0, 0, 0))``."""
        A.check_name(name, list(self.items))
        kind, text, src = _robots.source(name, path, mjcf)
        p0, per_world = A.positions(pos, self.worlds)
        fitted = _robots.sysid_item(name, sysid)             # {} without sysid=; a bad file raises before the add
        self.items[name] = {**fitted, "kind": "robot", "robot_kind": kind, "path": src, "mjcf": text,
                            "per_world_pos": per_world, "pose": p0 + A.orientation(quat_xyzw, quat_wxyz, rpy),
                            "randomize": [], "contact_gap": _robots.contact_gap(contact_gap)}
        return name

    def add_object(self, name: str, card: str, *, pos: Any = (0.0, 0.0, 0.0), quat_xyzw: Any = None,
                   quat_wxyz: Any = None, rpy: Any = None, static: bool | None = None, kinematic: bool = False,
                   rigid: bool | None = None, material: str | None = None, mass: float | None = None,
                   density: float | None = None, friction: float | None = None, restitution: float | None = None,
                   temperature: float | None = None, heat_material: str | None = None, anchor: str = "auto") -> str:
        """An object from the card library by card name (``alloy.objects()`` lists them; on a fresh install, with
        the ``import`` extra, also the Hub's object library, fetched at build or by ``alloy fetch``).
        The shared keywords of ``add_sphere`` (``static``, ``kinematic``, ``rigid``, ``material``, ``mass`` or
        ``density``, ``friction``, ``restitution``) change this object only; without them it is as its card says.
        ``temperature`` (K) heats a particle object (MPM or SPH card) as ``Scene.set_temperature`` does. ``anchor``: the
        point of the object that goes to ``pos``: "center" (its bounding box's centre), "base" (its bottom's centre:
        ``pos`` z is what it stands on), "origin" (its card's frame), or "auto" (default): "base" for scanned objects
        from an imported library (YCB, Google Scanned Objects, RoboCasa, ...; their frames sit centimetres off the
        object), "origin" for Alloy's own library objects, whose frames are set on purpose (a cup's base), so ``pos`` is
        where an object stands either way. What you set is what you read: ``"<name>.pos"`` in ``sim.state()`` and
        trajectories is that anchor point (it turns with the object), ``sim.set_pose(name, pos=...)`` puts it there, and
        ``"<name>.body_pos"`` is the card frame's origin (for objects placed by an anchor off their origin).

        Example: ``scene.add_object("peg", "peg_d12", pos=(0, 0, 0.04))``."""
        from alloy.api._card_objects import card_item  # noqa: PLC0415

        shared = {"static": static, "kinematic": kinematic, "rigid": rigid, "material": material, "mass": mass,
                  "density": density, "friction": friction, "restitution": restitution}
        self.items[name] = card_item(self, name, card, pos, (quat_xyzw, quat_wxyz, rpy), anchor, shared)
        return self._heated(name, temperature, heat_material)

    # --- materials, friction, randomization ---------------------------------------------------------------------

    def add_material(self, name: str, *, density: float, friction: float = 0.5, restitution: float = 0.3,
                     like: str | None = None) -> str:
        """A new rigid material (or a copy of ``like`` with these values). Example::

            scene.add_material("rubber_pad", density=1100, friction=1.2, restitution=0.2)"""
        if not A.NAME_RE.match(name) or name in self.materials or catalog.has_card("materials", name, self.roots):
            raise SceneError(f"material name {name!r} is invalid or taken", "pick a new name such as 'my_rubber'",
                             field="name")
        if like is not None:
            self._check_material(like)
        self.materials[name] = {"like": like, "density": A.positive(density, "density", "kg/m^3"),
                                "friction": A.positive(friction, "friction", "-"),
                                "restitution": float(min(max(restitution, 0.0), 1.0))}
        return name

    def set_friction(self, a: str, b: str, mu: float) -> None:
        """Coulomb friction between two objects (overrides their materials' combined value).

        Example: ``scene.set_friction("finger_pads", "cube", 1.2)``."""
        for n in (a, b):
            self._item(n)
        self.pairs.append({"a": a, "b": b, "mu": A.positive(mu, "mu", "-")})

    def set_inertia_floor(self, on: bool = True) -> None:
        """Turn the inertia floor on (the default) or off. The floor gives light slender free bodies (a 16 g marker
        pen) at least m·(9 mm)² of rotational inertia at the default rigid step, so a rim contact cannot spin them
        up until a world blows up; they then spin and roll slower about their thin axis (the build warns, naming
        them; ``sim.routes()`` says so). Off: exact rolling, which needs a smaller rigid step for such parts.

        Example: ``scene.set_inertia_floor(False)``."""
        if not isinstance(on, bool):
            raise SceneError(f"set_inertia_floor takes True or False, got {on!r}", "pass True or False", field="on")
        if on:
            self.extra.pop("inertia_floor", None)          # unset is on: the scene card and JSON stay as before
        else:
            self.extra["inertia_floor"] = False

    def randomize(self, name: str, field: str, lo: float, hi: float, when: str = "build") -> None:
        """Draw ``field`` per world uniformly in [lo, hi] (seeded by ``alloy.build(seed=)``). Fields: x, y, z (start
        position, m), density, friction (static; the object's dynamic friction is capped at ``lo``), youngs, or a
        card path such as "material.surface.mu_dynamic". For a free-base robot: x, y, z of its base. Robots, their
        links and joints, and "gravity" take the fields of docs/guide/randomize_robots.md (mass_scale, friction,
        kp_scale, ...); ``when="reset"`` draws those again at every ``sim.reset()``.

        Example: ``scene.randomize("cube", "x", -0.005, 0.005)``."""
        _random.add_row(self, name, field, lo, hi, when)

    def add_push(self, name: str, force: Any, every: Any = (4.0, 8.0), duration: float = 0.1) -> None:
        """Push body ``name`` at random times, per world: after a wait drawn in ``every`` = (lo, hi) seconds, a force
        drawn in +-``force`` (N, per world axis) is held for ``duration`` s (docs/guide/randomize_robots.md).

        Example: ``scene.add_push("go2/base", force=(80, 80, 0), every=(4.0, 8.0))``."""
        from alloy.api import _robot_random_push  # noqa: PLC0415

        _robot_random_push.add(self, name, force, every, duration)

    # --- load and build -------------------------------------------------------------------------------------------

    @classmethod
    def load(cls, source: str | Path, *, worlds: int | None = None) -> Scene:
        """An editable Scene from a library scene name, scene card, Scene JSON or robot file (see ``alloy.load``).

        Example: ``scene = alloy.Scene.load("demo_peg_d12"); scene.add_sphere("ball", 0.01, pos=(0, 0, 0.1))``."""
        from alloy.api._load import load_scene  # noqa: PLC0415

        return load_scene(source, worlds=worlds)

    def build(self, worlds: int | None = None, device: str | None = None, seed: int = 0, *,
              profile: str = "standard", plugins: Sequence[Any] = ()) -> Any:
        """Same as ``alloy.build(self, ...)``: returns an ``alloy.Sim``. Example: ``sim = scene.build(worlds=64)``."""
        from alloy.api._build import build  # noqa: PLC0415

        return build(self, worlds=worlds, device=device, seed=seed, profile=profile, plugins=plugins)

    # --- inspection and data ------------------------------------------------------------------------------------

    def names(self) -> list[str]:
        """Every object and robot name, in insertion order. Example: ``["ground", "cube"]``."""
        return list(self.items)

    def summary(self) -> dict[str, Any]:
        """A short JSON-ready description. Example: ``{"worlds": 1, "dt": 0.01, "objects": {"cube": "box"}}``."""
        return {"worlds": self.worlds, "dt": self.dt,
                "objects": {n: it["kind"] for n, it in self.items.items()},
                "materials": sorted(self.materials), "pairs": len(self.pairs)}

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready data. Example: ``alloy.Scene.from_dict(json.loads(json.dumps(scene.to_dict())))``."""
        return {"schema": "alloy.api.scene/1", "worlds": self.worlds, "dt": self.dt, "items": copy.deepcopy(self.items),
                "materials": copy.deepcopy(self.materials), "pairs": copy.deepcopy(self.pairs),
                "roots": list(self.roots), "extra": copy.deepcopy(self.extra), "thermal": copy.deepcopy(self.thermal),
                **_liquid_accuracy.saved(self.liquid_accuracy), **_rigid_step.saved(self.rigid_step),
                **({"joint_yield": copy.deepcopy(self.joint_yield)} if self.joint_yield else {}),
                **({"robot_random": copy.deepcopy(self.robot_random)} if self.robot_random else {}),
                **({"pushes": copy.deepcopy(self.pushes)} if self.pushes else {}), **_contact_relax.saved(self),
                **({} if self.cloth_accuracy == "default" else {"cloth_accuracy": self.cloth_accuracy})}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        """A Scene from ``to_dict()`` data. Example: ``alloy.Scene.from_dict(scene.to_dict())``."""
        if not isinstance(data, dict) or data.get("schema") != "alloy.api.scene/1":
            raise SceneError("not Scene data (schema 'alloy.api.scene/1' missing)", "pass the output of "
                             "Scene.to_dict()", field="schema")
        scene = cls(worlds=int(data["worlds"]), dt=float(data["dt"]),
                    liquid_accuracy=data.get("liquid_accuracy", "default"),
                    cloth_accuracy=data.get("cloth_accuracy", "default"))
        scene.items, scene.materials = copy.deepcopy(data["items"]), copy.deepcopy(data["materials"])
        scene.pairs, scene.roots = copy.deepcopy(data["pairs"]), list(data.get("roots", []))
        scene.extra = copy.deepcopy(data.get("extra", {}))
        scene.thermal = copy.deepcopy(data.get("thermal") or _heat.empty())
        scene.rigid_step = data.get("rigid_step", _rigid_step.DEFAULT)      # never set: profile="fast" may apply
        scene.joint_yield = copy.deepcopy(data.get("joint_yield") or {})
        scene.robot_random = copy.deepcopy(data.get("robot_random") or [])
        scene.pushes = copy.deepcopy(data.get("pushes") or [])
        return _contact_relax.load(scene, data)                 # the relaxed names back; returns the scene

    def to_json(self) -> str:
        """``to_dict()`` as a JSON string. Example: ``Path("scene.json").write_text(scene.to_json())``."""
        return json.dumps(self.to_dict(), indent=1)

    def save(self, directory: str | Path) -> Path:
        """Write the engine's YAML cards (materials/, objects/, scenes/scene.yaml) to ``directory``, with the cards
        and meshes it takes from its own card roots; returns the scene file; it warns for what cards cannot keep
        (per-world starts, heat, ...). Example: ``scene.save("my_scene")`` then
        ``alloy.load("my_scene/scenes/scene.yaml")``."""
        from alloy.api import _save  # noqa: PLC0415 - imports the card layer
        return _save.save(self, directory)

    # --- internals ----------------------------------------------------------------------------------------------

    def _item(self, name: str) -> dict[str, Any]:
        if name not in self.items:
            raise SceneError(f"no object named {name!r}", f"use one of {self.names()}", field="name")
        return self.items[name]

    def _heated(self, name: str, temperature: float | None, heat_material: str | None) -> str:
        if temperature is None and heat_material is None:
            return name
        try:
            if temperature is None:
                raise SceneError("heat_material= without temperature=", "pass temperature= (K) as well, e.g. "
                                 "temperature=293.15", field="temperature")
            self.set_temperature(name, temperature, heat_material=heat_material)
        except SceneError:
            del self.items[name]
            raise
        return name

    def _check_material(self, material: str) -> None:
        if material in self.materials or catalog.has_card("materials", material, self.roots):
            return
        raise SceneError(f"unknown material {material!r}",
                         f"use one of {catalog.near('materials', material, self.roots)} (alloy.materials() lists "
                         "all) or define it with scene.add_material", field="material")

    def _add(self, name: str, kind: str, shape: dict[str, Any], pos: Any, volume: float | None, *,
             quat_xyzw: Any = None, quat_wxyz: Any = None, rpy: Any = None, material: str = RIGID_MATERIAL,
             mass: float | None = None, density: float | None = None, friction: float | None = None,
             restitution: float | None = None, static: bool = False, rigid: bool | None = None,
             kinematic: bool = False, mesh: dict | None = None, card_extra: dict | None = None) -> str:
        A.check_name(name, list(self.items))
        card_extra = dict(card_extra or {})
        if kinematic:
            if static:
                raise SceneError("an object cannot be both static and kinematic", "pass one of static=True (never "
                                 "moves) or kinematic=True (moved by sim.drive / set_pose)", field="kinematic")
            card_extra["motion"] = "kinematic"
        if rigid is not None and not static:
            route = "rigid" if rigid else "soft_vbd"
            card_extra["route"] = {"route": route, "reason": f"set through alloy.Scene (rigid={rigid})"}
        self._check_material(material)
        if static and _worlds.per_world(pos):
            _worlds.refuse_static(name)
        p0, per_world = A.positions(pos, self.worlds)
        item = {"kind": kind, "shape": shape, "material": material, "static": bool(static),
                "pose": p0 + A.orientation(quat_xyzw, quat_wxyz, rpy), "per_world_pos": per_world,
                "volume": volume, "mesh": mesh, "card_extra": card_extra or {}, "randomize": [],
                "overrides": _overrides(mass, density, friction, restitution, static)}
        self.items[name] = item
        return name

