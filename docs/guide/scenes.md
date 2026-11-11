# Scenes: shapes, materials, friction

A scene is plain data until you build it. `alloy.Scene(worlds=1, dt=0.01)` starts one; `dt` is the control period
(one `sim.step()`), and the engine takes as many physics substeps inside it as each solver needs.

## Shapes

| Method | Size arguments | Notes |
|---|---|---|
| `add_ground(name="ground", z=0.0)` | | infinite static plane |
| `add_box(name, size=(x, y, z))` | full edge lengths (or `half_extents=`) | |
| `add_sphere(name, radius)` | | |
| `add_cylinder(name, radius, height)` | full height, along local z | |
| `add_capsule(name, radius, height)` | straight part between the caps, along local z | |
| `add_mesh(name, path, scale=1.0, collision="convex")` | OBJ / STL file, or `vertices=` (V, 3) + `faces=` (F, 3) | see below |

Shared keywords of every `add_*` (except the ground):
- `pos=(x, y, z)` in metres, or a `(worlds, 3)` array for one start per world, for anything that moves: free and
  kinematic objects, kinematic fixtures (each part follows), cloth, liquids and free-base robots. A static object,
  static fixture or `add_pour` nozzle refuses an array when added (`SceneError`); `kinematic=True` holds still
  unless driven and can start per world;
- orientation: one of `quat_xyzw=`, `quat_wxyz=` or `rpy=(roll, pitch, yaw)` in radians;
- `material=` a name from `alloy.materials()` (default `abs`, a rigid plastic);
- `mass=` (kg) or `density=` (kg/m³), `friction=` (Coulomb μ) and `restitution=` override the material for this
  object only;
- `static=True` fixes the object in place (tables, walls, bins). Fixtures (`add_open_box`, `add_open_cylinder`,
  `add_block_with_hole`) are static already and accept it; `kinematic=True` makes one movable, and `static=False`
  raises (a fixture is never a free body);
- `rigid=True` / `rigid=False` overrides the solver choice the material implies (see Materials below).

Light slender parts (a marker pen, a pencil, a utensil) get a little extra inertia about their thin axis at build, so
a contact cannot spin them up until a crowded scene blows up. They roll and spin slower about that axis (the build
warns and names them; `docs/known_gaps/rigid-slender-spin-floor.md`). `scene.set_inertia_floor(False)` keeps the
true inertia, for exact rolling at a smaller `dt`.

Every `add_*` returns the name; names must be unique and are how you read state later.

Per-world random values: `scene.randomize(name, field, lo, hi)` with field `x`, `y`, `z` (start position), `density`,
`friction`, `youngs`, or the card path `material.surface.mu_dynamic`. `mass` and `restitution` are not fields:
randomize `density` (mass = density x volume), and set restitution for all worlds with `restitution=` on the object. A
wrong field raises `SceneError` listing these. More in `worlds.md`.

## Containers and fixtures

Static containers and holed parts are built from boxes, so every face collides exactly:
- `add_open_box(name, inner=(x, y, depth), wall=0.004, pos=...)`: a bin or square cup; `pos` is the centre of its
  bottom face, the cavity floor is `wall` higher;
- `add_open_cylinder(name, inner_radius, depth, wall=0.004, pos=...)`: a round cup (wall slabs around a floor);
- `add_block_with_hole(name, size=(x, y, z), hole=(x, y), depth, pos=...)`: a block with a blind rectangular hole. It
  makes the whole scene a precise-fit scene (`precise=True`, the default): rigid bodies step at 0.25 ms or finer
  (`sim.dt` is unchanged; one `step()` runs several physics steps) and plastics, wood and metals get machined-part
  contact stiffness, so a dropped peg sits about 0.3 mm deep (0.27 mm measured; a test bounds it at 0.5 mm) instead of
  about 1 mm. That costs 5x more rigid steps at `dt=0.01`; `precise=False` keeps the default step.

Each returns its part names (`"<name>_floor"`, `"<name>_wall0"`, ...), which `sim.names()["static"]` lists too.
A kinematic fixture (`kinematic=True`) is one kinematic body carrying every part: `"<name>"` (and `"<name>_floor"`)
is its state, the floor's frame, and `sim.drive(name)` / `sim.set_pose(name, ...)` move it. Its other parts have no
state of their own (a kinematic cup is 6 dofs, not 25 bodies' 150).

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("bin", inner=(0.12, 0.12, 0.06), wall=0.005)   # cavity floor at z = 0.005
scene.add_sphere("marble", 0.01, pos=(0.02, 0.0, 0.15), material="glass_soda_lime")
sim = alloy.build(scene)
sim.run(1.0, record=[])
print(sim.state(world=0)["marble.pos"])                        # inside the bin, on its floor: z close to 0.015
```

## Kinematic objects: moved by you, not by forces

`kinematic=True` on any shape or fixture (`add_box`, `add_open_box`, ...) makes it follow commands only: it pushes
everything it touches but nothing pushes it. Use it for a tilting tray, a moving wall, a conveyor pusher or a pouring
cup (liquids see the moving walls every physics step).
- `sim.drive(name, velocity=, angular_velocity=, pivot=, duration=)`: constant linear (m/s) and angular (rad/s,
  world axes) velocity about `pivot` (a world point; default the object's origin or the fixture's centre), for
  `duration` seconds, then hold. Drives on several objects run at once.
- `sim.set_pose(name, pos=, quat_wxyz=, world=)`: teleport a kinematic or free body now, with zero velocity.
- `sim.stop_drives()`: every kinematic object holds where it is.
- `sim.settle(seconds)`: run with no commands, then make the settled state the start that `sim.reset()` returns to.
- Tilting a container to pour: choose the rotation's sign so the far side rises (the right-hand rule about the axis),
  and put `pivot` on the lip you pour over.
- A kinematic object driven into the ground or a static object cannot be stopped by it: the overlap grows until the
  health monitor freezes that world (PEN_FAIL). Stop the drive (`duration=`) before it reaches a surface.

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("pusher", size=(0.02, 0.2, 0.05), pos=(-0.2, 0, 0.026), kinematic=True)   # moved only by drive/set_pose
scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.0, 0, 0.026), mass=0.2)
sim = alloy.build(scene)
sim.settle(0.3)                                          # let it come to rest; reset() returns to this state
sim.drive("pusher", velocity=(0.2, 0, 0), duration=1.5)  # 20 cm/s along +x for 1.5 s, then hold
traj = sim.run(2.0, record=["block", "pusher"])
print("block x %.3f -> %.3f" % (traj["block.pos"][0, 0, 0], traj["block.pos"][-1, 0, 0]),
      "pusher x %.3f" % traj["pusher.pos"][-1, 0, 0])
sim.stop_drives()
sim.set_pose("pusher", pos=(-0.3, 0, 0.026))             # teleport it back (zero velocity)
sim.step(10)
print("pusher back at x %.3f" % sim.state(["pusher"], world=0)["pusher.pos"][0], "ok:", sim.check()["ok"])
```

## Meshes

`add_mesh(name, path | vertices=, faces=, collision=...)` takes an OBJ or STL (binary or ASCII) file or `(V, 3)` /
`(F, 3)` arrays; faces wind counter-clockwise seen from outside, and the mesh must be watertight. `collision=`:
- `"convex"` (default): the convex hull. Fast; right for solid, roughly convex parts.
- `"decompose"`: convex pieces of a concave part, split once at build and cached (needs CoACD and trimesh, the
  `import` extra: `install.md`; without them the build raises `FeatureUnavailableError` naming the command).
- `"sdf"`: a signed-distance field of the true shape, for concave parts; contact is approximate (about a millimetre),
  so prefer the box-built fixtures above for containers and holes that must be exact. A static sdf mesh cup holds
  liquid: on the CPU a 6 cm cup kept all of a block of `add_liquid` at rest for 1 s and 98 % of a 40 ml `add_pour`,
  with contacts within about 1 mm.
- `"mesh"`: the triangles as they are.
- Bodies on `"sdf"` or `"mesh"` contacts: a scene with such a shape makes its contacts in Newton's contact pipeline
  instead of MuJoCo's own, and its free bodies step on implicitfast at the fixed rigid step, never rk4. A body that
  moves on `sdf` or `mesh` contacts settles within 0.25 mm of its rest height but keeps a vibration of up to 0.3 mm at
  2-25 mm/s (it is never fully at rest), for bodies of about 40 g to 35 kg; lighter or heavier ones make the world
  INVALID (`sim.check()`), so use `convex` or `decompose` for parts that move unless you need the exact concave shape.
  A light body can bounce for up to 2 s with health OK before its world goes INVALID, so read `sim.check()["ok"]` at
  the end of the run. The mass table is in `docs/known_gaps/rigid-sdf-mesh-contact-limits.md`.

`alloy.api.meshes` makes watertight `(vertices, faces)` for a box, an open box, an open cylinder and a block with a
hole, when you need them as meshes (for example to scale or export them).

A soft body from a mesh: give `add_mesh` a soft material (`material="silicone_40a"`, a rubber or a foam; or pass
`rigid=False`). The build fills the closed mesh with tetrahedra and `sim.routes()` says `soft_vbd`. It needs the
`mesh` extra (`pytetwild`, `install.md`). Soft bodies step much slower than rigid ones: on the CPU (an AMD EPYC 7B13)
a 6 cm cube from a mesh took 11 s for 0.1 s simulated, and the same cube from `add_box` 3.7 s. On one L4, a box
resting on a soft pad ran 0.28x to 0.30x real time over 64 worlds (0.018x to 0.021x for one, two runs); the pad's
squeeze stayed within 4.2 % of the neo-Hookean law (median 1.9 %), and none of 256 randomized drops glitched.
`rigid=True` keeps a soft-material mesh rigid.

## Materials and friction

- `alloy.materials()` lists every library material with its kind (rigid / elastic / fluid / granular), density,
  friction and restitution. Each library value records its source (measured, catalogue, fitted, or a model
  estimate: about half are estimates) with a note and an uncertainty range, in `library/materials/`.
- Contact uses one friction coefficient per pair of materials. There is no separate static friction: a body starts to
  slide at the same coefficient it slides with. For a pair with a pair card it is the card's `mu_dynamic` (aluminium
  on steel: 0.47, not its static 0.61), or the card's `mu_static` when the card says `single_mu: static` (one shipped
  card, aluminium on aluminium: 1.05, so it also slides at 1.05, not at the kinetic 0.40). Without a pair card it is
  the harmonic mean of the two materials' `mu_dynamic`. `friction=` and `scene.set_friction` set that one number. A
  coefficient above 1 tips a cube before it slides, so check such a pair with a flat block.
- Restitution reads higher than set. On MuJoCo contacts (rigid bodies and robots), a restitution of 0 still rebounds
  at about 0.1-0.15 of the impact speed (0.11-0.15 at 1-5 m/s on a 0.3125 ms rigid step; less at the default 1.25 ms
  step: 0.08 at 2 m/s), and every restitution reads a little higher than set (0.5 gives about 0.55): the contact lets
  go when its force reaches zero, not at zero overlap. Measured with a 5 mm ball on a pane, on the CPU. There is no
  setting for a dead stop yet; a fix is planned for 0.1.1.
- `scene.add_material("my_rubber", density=1100, friction=1.2, restitution=0.2)` makes a new rigid material;
  `like="nitrile_70a"` copies another one's other properties.
- `scene.set_friction("finger", "cube", 1.1)` sets the friction of one pair of objects, overriding their materials.
- Contact stiffness comes from the materials; change materials or `dt`, not solver internals.
- Impacts go deeper than the contact stiffness says. The contact stiffness of a material card (1e6 N/m for a default
  pair) holds for a resting load. On an impact, MuJoCo's soft contact acts about 10-20x softer, so a heavy body
  hitting a wall at walking speed goes 3-4x deeper than that stiffness gives (a 6.8 kg ball at 1.7 m/s: 14-15 mm,
  measured on one L4 at 3.33 / 1.67 / 0.42 ms steps, so a smaller step does not change it). Past 1 cm health reads
  PEN_WARN, and a harder hit can reach the depth at which it freezes the world (PEN_FAIL: [health.md](health.md)). A
  fix is planned for 0.1.1.
- The material also decides the solver. Stiff materials (metals, hard plastics, wood, glass) stay rigid; soft ones
  (rubbers, foams, silicone) can make the object a soft body (`sim.routes()[name]["route"] == "soft_vbd"`), which
  deforms and steps much slower, especially on the CPU; the build logs a warning when that happens. Pass
  `rigid=True` to keep such an object rigid with that material (or `rigid=False` to ask for a soft body).

## Library objects and scenes

- `scene.add_object(name, "<object card>", pos=...)` places an object from the card library: the 118 shipped objects
  plus the ones fetched from the object library on the Hugging Face Hub (about 5,400 cards: YCB, Google Scanned
  Objects, Objaverse, Lightwheel kitchen objects as packaged by RoboCasa, Gym-Anything object parts; most of their
  masses are estimates, [fetch.md](fetch.md)), each bringing its own shape, material and, for precision parts, its
  precision contact settings. `from alloy.api.catalog import objects; objects()` lists the ones on disk,
  `alloy fetch --list` the library's; `alloy.materials(imports=True)` adds the imported objects' own materials.
  Fetching needs the `import` extra; with it, a build fetches a library object it lacks ([fetch.md](fetch.md)).
- `pos` is where the object stands; `anchor=` picks the point of the object that goes there. `"auto"` (the default)
  is `"base"` for objects from imported libraries (YCB, Google Scanned Objects, RoboCasa, Objaverse, the Hub), whose
  file origins sit wherever the scan put them, and `"origin"` for objects in `library/` and your own cards, whose
  frames are set on purpose (a cup's base). The modes:
  - `"base"`: the bottom centre of the object's box, so `pos` z is the top of what it stands on;
  - `"center"`: the centre of its box;
  - `"origin"`: the card's own frame.

  Per-world `pos` rows and `scene.randomize(name, "x" | "y" | "z", ...)` follow the anchor too. Scenes saved before
  anchors existed keep their poses.
- What you set is what you read. `"<name>.pos"` (in the state, `record=` and trajectories) is the point `anchor=` put
  at `pos`, and it turns with the object. `sim.set_pose(name, pos=)` puts that point there, and a turn alone turns
  about it. `"<name>.body_pos"` is a scan's raw body frame, which can be centimetres away: `ycb_025_mug` placed at
  `pos=(0, 0, 0.4)` on a table top at 0.4 m reads `.pos` `(0, 0, 0.4)` and `.body_pos` `(0.009, -0.017, 0.401)`.
  Primitives, Alloy's own objects, your own cards and `anchor="origin"` read their body origin as `.pos`, as before.
- Search by words ("a mug", "three wooden cubes") with the scene builder, `alloy.compose` (`compose.md`).
- `alloy.scenes()` lists library scenes; `alloy.load(name)` builds one, `alloy.Scene.load(name)` gives an editable
  copy you can add to.

## Saving and loading

- `scene.to_json()` / `alloy.Scene.from_dict(data)`: the scene as JSON data (keeps everything you added).
- `scene.save(directory)` writes the engine's YAML cards (`materials/`, `objects/`, `scenes/scene.yaml`) and
  returns the scene file; `alloy.load(that_file)` builds it again. See `cards.md` for the card format.
- `alloy.load(path)` also takes `.json` Scene data, a `.yaml` scene card, and robot files (`.xml` MJCF, `.urdf`,
  `.usd`), which it places on a ground plane.
