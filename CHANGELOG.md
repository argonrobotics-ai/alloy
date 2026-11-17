# Changelog

Changes to Alloy's public API: `alloy.api`, re-exported at the top level (`alloy.Scene`, `alloy.build`, ...).
Versions follow [semantic versioning](https://semver.org): from 0.1.0, a minor version may add to the public API and
a patch version only fixes it; anything that would break code written against it waits for the next major version
(or, before 1.0, the next minor version) and is announced here first as a deprecation. Modules outside `alloy.api`
are engine internals with no stability promise.

## [0.1.7] - 2026-11-17

### Added
- `alloy.Scene(liquid_tension=True)` (also `scene.liquid_tension`, kept in Scene JSON): each liquid feels its
  material's surface tension (water 0.072 N/m), with a free contact line on every wall. Off by default. A thin stream
  then holds together: in a pour from a tilted 50 mm beaker into a 15 mm vial at 2 mm particles, the vial caught
  52-57 % of what left the beaker, against 2-5 % without, at about the same cost. The tilted beaker pours a half to
  a quarter as much in the same time (less at finer settings), and at 1.5 mm a few droplets fly off the pour (a 50 mm
  receiver caught 94 %): `docs/known_gaps/liquid-tension-lip-drain.md`. A free oscillating blob of water grows its
  oscillation (`docs/known_gaps/liquid-tension-free-drops.md`).
- `alloy.Scene(liquid_accuracy="ultra")`: liquid particles 1.5 mm apart, 8 across a 12 mm opening (a 15 mm vial's
  neck is 10), at about 50x the default's work. On one L4 a pour ran 3.0x super's wall time for one world and 7.0x
  (61x default) for 16 worlds. `alloy.estimate` shows all three settings, and the coarse-opening warning names the
  cheapest setting that resolves each item.

### Changed
- Full-quality renders with a look (`sim.render(quality="full", look=...)`) apply its `render.white_balance` as
  Blender 4.5 does, `"auto"` unless the look says otherwise: the camera renders a small grey ball at the subject under
  the look's lights and world and adapts that light's colour to white, so under a 2800 K light a grey ball reads grey,
  not orange. `{"kelvin": K, "tint": t}` adapts a fixed white (tint 10 when left out) and `"off"` gives the image as
  before. Looks whose light is far from white change most: of three shipped looks, the kitchen's image mean moved 22.5
  levels, the workshop's 3.3 and the studio's 1.0. A render without a look is unchanged. The measurement adds 1-3 s to
  a look's first scene build on the CPU.

### Fixed
- Water at rest in a narrow round cup (`add_open_cylinder`) no longer starts INVALID. Cups of 12-24 mm at 4 mm and
  8 and 16 mm at 2 mm went INVALID within 0.04 s; every cup measured now stays healthy and keeps its water (12-32 mm
  at 4 mm, 8-24 mm at 2 and 1.5 mm), and pours into 12-24 mm cups at 4 mm keep 100 % (was 88.6-97.5 %). Three causes:
  walls built from many slabs were counted twice where neighbouring slabs meet, a crowded corner could lose the floor
  from the walls a particle sees, and a cylinder of liquid could start with up to 27 % more particles than its area
  holds. A cylinder of liquid now fills whole rings of particles, no more than its area holds, the outer ring at least
  0.375 particle spacings inside its side (a fuller or closer ring threw the top layer up to 40 mm, over the rim of a
  16 mm cup). The column then adds layers on top until it holds the volume asked to within half a layer, so it starts
  taller than asked in a narrow cup (40 mm asked in a 12 mm cup at 2 mm: 48 mm); a 40 mm cylinder holds 26.1 ml at
  4 mm and 24.0 ml at 2 mm, was 25.6 and 25.3 ml, asked 25.1 ml. The coarse-opening warning drops its 24 mm floor: it
  fires under 8 particles across at every setting.
- Liquid holds in walls thinner than its particles: a particle that moves more than half the thinnest wall in one
  step is tested along its whole path, so it cannot step across the wall. Measured with 1 mm glass at every setting:
  still water for 5 s in a 1 mm tub and a pour into a 15 mm vial with 1 mm walls lost no particle through a wall.
  (Static and kinematic walls down to 1 mm, such as `add_open_cylinder(..., wall=0.001)` or a static box, now build
  without `card_extra`; beside a free body or a robot the build warns once per wall with the speed that can pass it,
  its thickness / (2 x 1.25 ms): 0.4 m/s at 1 mm. Free parts still need 1.25 mm, or
  `card_extra={"features": {"min_wall": ...}}`.)
- Liquid no longer crosses the wall of a narrow round cup at the inner corner between two wall slabs. Near the floor,
  a particle there could keep two farther slabs among its four wall contacts and leave out the nearer one, step into
  it and come out through the cup's outer face (once in a probe pour into a 15 mm cup at 2 mm particles). The nearer
  slab is now kept.
- `alloy.estimate` on the CPU counts the parts each free body comes to rest on, from its start pose, and charges
  those beyond its first in a scene of free bodies on the default route (MuJoCo C), where rk4 tests each against its
  support in each of its four stages: about 3x as much on a table, a bin's floor or another body as on the ground.
  Three YCB scans dropped into a bin (32 convex parts each, resting on 15-17 of them) stepped 2.0-2.6x the estimate,
  which said about 0.9x real time where the run gave 0.38-0.48x; they now step 1.04-1.14x it, and every body measured
  that settles (scans, a body of up to 32 cubes and decomposed bowls, on the ground and on a box table) reads within
  1.3x. A body that keeps rocking on many parts (a decomposed bowl upside down on its rim) steps 1.6-2.4x it (was
  1.8-3.8x). Bodies resting on one part each (a pile of separate boxes), robot scenes and MuJoCo Warp read as before.
- `alloy.estimate(..., measure=True)` times the first second of the run (all of a shorter run) and charges the rest
  at the rate of its last quarter. Before, it timed the first 0.2 s, before dropped objects land and while a pour
  is still starting, so it read three YCB scans dropped into a bin about 2x fast (it said 0.78-0.87x real time where
  the run gave 0.38-0.48x on the CPU) and a pour 3-6x fast. A pour still flowing after the timed second reads fast:
  a 2.5 s pour into a cup, in a 3 s run, read 24.7 s against the run's 52.3 s (the estimate without `measure` said
  67.8 s).
- On a machine with no NVIDIA driver, Warp's line "Warp CUDA warning: Could not find or load the NVIDIA CUDA driver.
  GPU execution will not be available." no longer prints, once per Python process, at the first `alloy.build`,
  `alloy.envs.make` or CPU worker process: Alloy runs on the CPU there and says so where it matters (the build's
  CPU note for cloth, liquid, sand and soft bodies). `ALLOY_VERBOSE=1` keeps Warp's own lines, that one included.
- A soft mesh without the `mesh` extra (a YCB apple of vinyl_soft) names `rigid=True` on `add_object` / `add_mesh` in
  the error itself, so `alloy.check` gives it too (it gave only a generic fix; `alloy.build`'s fix already had it).
- `alloy.envs.make` prints none of Warp's start-up banner or "Module ... load" lines, as `alloy.build` does:
  `PickPlace-v0` and `ClothFold-v0` printed the 6-line banner and 11-13 module lines, because their scene variant
  started Warp before the build quieted it. `ALLOY_VERBOSE=1` still shows them.
- A camera-only marker on a robot link (a geom with `contype` and `conaffinity` 0) no longer hides that link's
  collision geoms in camera images. Newton's MJCF import hides every collider once a file has one such geom, so a hand
  carrying a 4 mm marker and a jug of 5 box geoms drew only the marker, and the jug's water floated in the image. The
  camera now also draws, on moving bodies with a display shape, the hidden colliders of MJCF geom groups 0-2 that
  stick out of the body's display box by more than 25 % of its size; parts welded to the world keep the import's
  choice.
- A `VectorEnv` warns at build when a `jitter_joint` moves a body that a liquid or cloth starts in or on and `carry=`
  does not name it. A wrist holding a jug turned it at each reset and left its water in mid-air: 14 % spilled before
  the first action, with no message. The warning names the fix: `carry=[...]` on a slide joint; for a hinge, a slide
  jitter or none. `Pour-v0` and `ClothFold-v0` build without it.
- Both array backends of `alloy.envs` offer the same basic math (`sqrt`, `abs`, `exp`, `log`, `sin`, `cos`, `arcsin`,
  `arccos`, `minimum`, `maximum`, `sum`, `mean`), so a `checks.custom` function written with `s.xp.sqrt(...)` runs
  with `backend="numpy"` too; it raised `AttributeError: 'NumpyBackend' object has no attribute 'sqrt'`.
  `docs/guide/envs.md` lists what `s.xp` offers.
- `alloy.envs.mjcf.without_bodies` also trims a geom directly under `<worldbody>`: `without_bodies(path,
  ["obstacle"])` on the H1-2 file drops its obstacle box (a geom there; the G1 and Go2 files wrap theirs in a body)
  instead of raising `EnvError`. Trims of bodies are unchanged.
- Shipped arms and the recorded task scenes no longer print MuJoCo Warp's MJ_MINMU friction warnings. Behaviour
  unchanged.
- In one process, repeated `alloy.build`, `sim.render(quality="full")` and `sim.close()` no longer grow GPU
  memory: each scene's bevel mesh (its static triangles and their BVH) was kept for good, about 47 MiB per frame on
  an L4 (4,674 to 13,698 MiB over 182 frames). It is now freed with the scene.
- Sand, soil or paste in a container built of parts narrower than the material itself (the 4 mm wall slabs of a
  default `add_open_cylinder`) gets a finer MPM voxel and many more particles: a 12 mm cube of sand in a vial of 10 mm
  inner radius became 13,824 particles instead of the 512 its own size gives, with nothing said at build. The build
  now warns once when a container part multiplies the count 4x or more, naming the part and both counts, with the fix
  (a thicker `wall=`, fewer `segments=`, or a container of fewer, larger parts); `sim.routes()` and
  `alloy.estimate(...)["routes"]` list both counts.
- `alloy fetch` and builds that fetch a Hub object no longer log the Hub's note about unauthenticated requests
  (twice per `alloy fetch`, once per cold `alloy.envs.make` of an environment with Hub objects): the library is
  public. A download the Hub rate-limits now says to set `HF_TOKEN`.
- A cut `alloy fetch --list` (over 200 objects) lists names from every category in turn and says how to list one in
  full: `--library ycb_gso` listed 200 `gso_*` names and no `ycb_*`.
- The RoboCasa object importer (`python -m alloy.importers.robocasa`) no longer leaves `MUJOCO_GL` and
  `PYOPENGL_PLATFORM` set in the process after its thumbnail render, which made a later `import mujoco` in a
  subprocess fail where OSMesa does not work.

### Repository
- The quick CPU tests that pull requests run take about 35-45 minutes with a warm kernel cache, up to about 1 h 40 on
  a cold first run (0.1.6 said about 80 minutes and 2 h 15, measured before its fix for tests that share a file's
  one-time setup). On GitHub's runner they took 34 minutes warm on 0.1.6's push and 43 on 0.1.5's last run.

## [0.1.6] - 2026-11-16

### Added
- `american_walnut_veneer` (Poly Haven, CC0) joins the public wood texture pool, so the `wood_walnut_oiled` preset's
  textures come with `python -m alloy.looks --fetch-textures --family wood`, and variants can swap it for another wood.

### Changed
- A full-quality render of a look with no `show_hdri` now draws the look's `background_color` behind the scene,
  which is dark by default, as Blender does, not the studio gradient.
- `ClothFold-v0`'s towel has 19 cells a side by default (was 13), finer than the length its fabric bends over: the
  half being turned over no longer pushes the half lying on the table off its place (placed off in 5 % of worlds, was
  24 %), and the scripted demo folds 55 % of 1,024 worlds on L4s (38 % at 13 cells). More corners spring off as the
  fingers open: 34 % of worlds, was 26 %. A step costs about 1.5x on an L4 (a 64-world fold about 1 h 40 min, was an
  hour); `make("ClothFold-v0", resolution=13)` gives the earlier towel.

### Fixed
- A scene of only static and kinematic parts builds and steps: posed parts drawn frame by frame with
  `sim.set_pose`, a tray moved with `sim.drive`. A driven part takes the same path as it does beside a free body.
  Before, `alloy.build` raised `BuildError` ("scene has no dynamic entry to step") unless something in the scene
  could move freely, so such a scene needed a free ball out of view. A scene of static parts alone builds too.
  `alloy.check` runs such a scene for its seconds and reports that run's findings; before, it stopped at t=0 with the
  same error.
- A part too small for MuJoCo's inertia limit at its default density (a 0.37 mm thread segment: smallest moment
  about 2e-15 kg·m², where MuJoCo takes 1e-14) builds. A kinematic one builds with no warning, in a scene of posed
  parts or beside moving bodies: its inertia is raised to Newton's floor and its mass is kept. A free one steps on
  the larger inertia Newton gives it, which makes it turn far too slowly, and the build warns once, naming it. Before,
  MuJoCo refused the build with a bare `ValueError` ("inertia must have positive eigenvalues"), so such a part needed
  a raised density.
- Full-quality renders with a look (`sim.render(quality="full", look=...)`) now honour the look's
  `world.rotation_deg` (the HDRI turns about +z, as in Blender), `show_hdri`, `background_color` and
  `background_from_hdri`, and a world key they cannot draw raises `CameraError` naming the keys they read. Before,
  they read only the HDRI and its strength.
- A look rule on an item also reaches its `<item>/visual` shape, the mesh drawn in place of its collision hulls,
  where a decomposed or large static mesh drew default white: in full-quality renders it takes the item's material,
  and in the Blender renderer the item's material, hide and bevel rules. A rule that matches only a hidden hull warns
  once, naming the item.

### Repository
- The quick CPU tests that pull requests run take about 80 minutes with a warm kernel cache, and up to about 2 h 15 on
  a cold first run, which warms the cache for about 50 minutes first (0.1.5 said under 90 minutes). The job's time
  limit is now 180 minutes, so a cold run finishes instead of being cancelled.
- When the quick CPU tests leave out the test that pays for a file's one-time setup, they now leave out the other
  tests that share that setup too, so no kept test pays for it: on 0.1.5's GitHub run one timed out that way. The
  quick tests keep about 87 % of the CPU suite (0.1.5: about 89 %).

## [0.1.5] - 2026-11-15

### Fixed
- Liquid coupled with rigid bodies or robots steps at its own stable step again, as in 0.1.2. Since 0.1.3 the extra
  margin meant for liquid beside cloth (a pour splashed over a container's rim without it) applied to every coupled
  liquid, so a robot scene with 16,575 liquid particles ran its liquid at about twice the substeps, 1.9x slower in
  all (to 3 simulated seconds on one L4: 234 s in 0.1.2, 445 s in 0.1.3, 238 s now). Liquid beside cloth or a soft
  body keeps the margin.
- `add_robot(name, mjcf=<text>)` reads the files the text names as MuJoCo reads an XML string: from the working
  directory when `add_robot` runs (meshes, height fields and skins under the compiler's `meshdir`, textures under its
  `texturedir`, `<include>` files), and keeps them as absolute paths, so `scene.save()` and `to_json()` load from
  any folder. Before, a mesh named relative to your folder was looked for in the build's own card folder and failed
  with trimesh's bare "string is not a file". A file that is not there now raises `SceneError` naming the robot, the
  element and the path it resolved to. Without the `import` extra, a robot whose file or text names OBJ or STL meshes
  now raises `FeatureUnavailableError` with `pip install "alloy-engine[import]"`, not a `BuildError` without the
  install command; the shipped robots need no trimesh.
- A mesh object of a soft material (YCB toy fruit and balls are `vinyl_soft`, the kitchen sponge is foam) still
  becomes a soft body by default, and without the `mesh` extra its build raises `FeatureUnavailableError` as before.
  Its fix now names `rigid=True` on the `add_object` / `add_mesh` call first, which keeps the object rigid with its
  material and builds on a base install, and `pip install "alloy-engine[mesh]"` second, for a soft body. Before,
  the fix named only the install line, though an object for a robot to pick is rigid in most scenes.
- The first `alloy.build()` of a process is freed after `del sim`, like every later build. Before, the bundled
  Newton kept the import error of its optional USD schema package with its traceback, which led back to the frame
  that first imported Newton (the first build's) and kept that sim and its device arrays for the life of the
  process. The error is now kept without the traceback (vendored patch 0018); USD reading is unchanged.
- `camera_scene(sim, lighting=Lighting(...))` now lights every later render of that sim: a varied scene's looks,
  `quality="full"`, a camera on another device, and a scene rebuilt after a cut or tear. Before, those went back to
  the studio light without a word, and each new lighting kept one more copy of the scene on the device. A sim built
  with `processes=` now raises `CameraError` naming `processes=None`, not an error about `sim.index`.
- The learned denoiser no longer leaves PyTorch holding GPU memory after a render (10 GB at 1,024 images on an L4),
  so the next `sim.step` can use it.
- `sim.check()` no longer runs out of device memory where the run itself fits. Its contact probe starts at least at
  the simulation's own contacts per world and grows from there when it loses contacts; before, it was rebuilt at
  Newton's automatic size (about 12 GiB for a 1,024-world kitchen of 134 shapes beside an arm, which then ran an L4
  out of memory). When its buffers do not fit the free device memory, it collides the worlds in chunks.
- `sim.check()` collides MJCF meshes as their convex hulls, as MuJoCo does, so the depths it reports match the
  simulation's own contacts. Before, it collided them as triangle meshes, and a RoboCasa kitchen pool at 1,024
  worlds overflowed its probe (contacts dropped, then a rebuild); now neither happens (11.4 GB peak on an L4).
- A part thinner than 1.25 mm from `alloy.Scene` (a 0.2 mm tab, a 0.61 mm wire, a 0.3 mm film) still fails the
  build, and the error's fix now names the form that keeps it: `rigid=True, card_extra={"features": {"min_wall":
  {...}}}` on its `add_box`, `add_cylinder`, `add_capsule` or `add_mesh` call, with the measured value filled in.
  Such a part builds and rests; landing or hit faster than its thickness over two rigid steps (0.08 m/s for 0.2 mm),
  it can go more than half its thickness deep, and `sim.check()` then fails that world. `add_sphere`, `add_rod` and
  the fixtures take no `card_extra`, and the fix says so. Before, the fix named only a thicker part, a mesh's error
  pointed to a file that does not ship, and `card_extra` without `rigid=True` made such a part a cloth sheet or a
  chain.
- A look's texture that the local cache holds at no size now raises `CameraError` naming the public fetch command,
  `python -m alloy.looks --fetch-textures --family <family>`, not an internal class. A look that names its texture
  with no size renders from a cache that holds only the 1k images, with one notice.
- The render backends `"wavefront-warp"` and `"optix"` now raise `CameraError` before rendering when the render
  device is not a CUDA GPU, naming the default `backend="warp"`, instead of a bare `RuntimeError`.

### Documentation
- The camera guides show how to set the sun and sky (`camera_scene(sim, lighting=Lighting(...))`), how to place a
  camera with `alloy.camera.pose.look_at` (an up vector as an optional third item) and how to give each world its
  own camera pose.
- The scenes guide shows how to build a rigid part thinner than 1.25 mm, and how fast it can be hit.

### Repository
- Pull requests on GitHub run the quick CPU tests, about 89 % of the CPU suite, in under 90 minutes. The long and
  wall-clock tests they leave out are listed in `.github/ci-quick-deselect.txt`; the whole CPU and GPU suites run
  before each release (`CONTRIBUTING.md`).
- The quick CPU tests pass on a clone of the public repository: the known-limit pages keep their `owner:` line,
  tests that read files the repository does not hold skip there, and the importer tests no longer leave a MuJoCo
  rendering backend set for the tests after them.

## [0.1.4] - 2026-11-14

### Changed
- Pillow (10.1 or newer) is now a base dependency, so textured looks render on a plain `pip install alloy-engine`.
  Before, it came only with the `import` extra, and on a base install the first `sim.render` frame of a textured
  look raised `CameraError` ("Pillow is not installed").
- Bundled engines: Newton v1.8.0 (`425db81f`, 2026-11-11) with the same 12 commits and 17 patches of ours, in place
  of release-1.8 `db915e1c` (1.8.0rc1). Upstream's changes between the two touch only docstrings, warning classes
  (`NewtonWarning` is now at the package root), ray casting and the camera sensor, none of which Alloy steps with:
  every scene steps as before. `alloy/_vendor/VENDORED.md` lists each commit and patch.

### Fixed
- `add_cloth`'s `resolution` is documented as what it builds: cells along the longer side (20 on a square sheet: 20 x 20
  cells, 21 x 21 = 441 vertices), not vertices; the API's own count said 400. The sheets built are unchanged. A grid
  the build cannot keep now warns, naming the grid it makes: an `(nx, ny)` whose shape differs from the sheet's
  (cells are as near square as whole counts allow: `(10, 6)` on a square sheet gives 8 x 8), a sheet too narrow for
  its short side, or more than 1,600 cells (`resolution=60` on a square sheet gives 40 x 40).
- `add_robot(..., pos=(worlds, 3))` (and per-world `scene.randomize` coordinates) now moves the whole robot file in
  each world, as the docs say: its free base, its own floor and the parts it welds to the world. Before, only the base
  moved, and the file's floor and welded parts stayed at the first world's row in every world: with a Go2 scene file
  and rows 10 cm apart in height, the lower world started with its legs 100 mm inside its floor (a start overlap
  that fails `sim.check()`) and the higher one dropped. On the CPU's default route a floor raised for one world also held no contacts (MuJoCo's
  compile-time frame flag and bounding tree kept the first world's floor), now refreshed for each world that differs.
  A file with a welded part (the Go2 scene's obstacle) no longer raises "no free base" for per-world starts or
  `sim.set_pose`.
- Objects resting beside a robot or a moving fixture no longer jitter. On the CPU, a scene with a kinematic object
  steps on MuJoCo Warp, and there a stack of light boxes kept bobbing a few mm/s for good: an 18-level tower of alder
  blocks stayed at 3.9 mm/s with its friction nearly used up, so pushing one block dragged the whole tower (32 mm at
  the top). Boxes and capsules in scenes with a robot or a kinematic object now take a contact damping ratio of at
  least 1.0 (0.7 before), and such a tower rests at 0.001 mm/s. On a GPU, such a tower bobbed at 3.3 mm/s too and now
  rests at 0.03 mm/s. Cost: those boxes rebound a little less (a 5 cm aluminium box dropped 10 cm: e 0.22, was 0.26).
  Scenes of only free bodies keep 0.7 and are unchanged.
- A `collision="convex"` mesh now collides with a hull that stays within 0.1 mm (or 1 % of its smallest size) of the
  mesh's own hull, using up to 2,048 hull vertices, instead of a fixed 64. With 64, a brass top's rounded tip became
  one point off its axis, and the top gained spin and energy where tip friction should slow it; now it slows. A mesh
  whose hull is already that close at 64 vertices keeps 64 (over 50,000 vertices: MuJoCo's hull at that cap, in
  place of Newton's); every shipped object card, `collision="decompose"` parts and robot meshes are unchanged. A
  mesh that cannot get that close warns once, naming the gap and the fix. Cost (64 worlds of 12 scanned objects as
  convex meshes): on one L4, 15 % fewer steps per second for a typical set and 39 % for the most detailed; on the
  CPU's default route, up to 24 % (8 worlds). To keep the earlier fast hull, pass
  `scene.add_mesh(..., hull_vertices=64)` (or `shape.mesh.hull_vertices: 64` in the mesh's card; an int from 4 to
  2,048, on a convex mesh only); 64 steps bit for bit as before.
- A mesh with `collision="sdf"` or `"mesh"` beside liquid or sand no longer reserves soft-contact room for every
  liquid and sand particle against every shape (their solvers read no contacts). Before, 16 worlds of water, a cloth
  and one SDF bowl among 550 small static boxes asked for 129 M contact slots and ran out of memory on a 24 GB L4 in
  their first steps; they now ask for 8.6 M and step in 7.1 GB. Cloth and soft-body contacts are unchanged.
- On the CPU's default route, a body welded to the world at a different pose in each world now stands at its own
  world's pose, also after its joint frame is changed or the worlds are reset. Before, it stood at world 0's pose in
  every world, and the other bodies of a world met it there. This needed a model built at the engine layer (one
  Newton builder per world); scenes built with `alloy.Scene` weld a body at one pose in every world, and scenes with
  kinematic objects step on the other CPU route, so they were not affected.
- On the CPU's default route (MuJoCo C), `sim.reset()` after `sim.settle()` now repeats the run that followed the
  settle bit for bit. Before, the reset restarted each world from the settled state rounded to float32 rather than
  the float64 state the settle ended in, and a two-joint arm drifted by 3e-8 rad within 20 steps.
- On the CPU's default route, a run that goes back to a saved state now continues bit for bit as the unbroken run
  did: a world re-run after a rollback, the worlds kept beside it, and a checkpoint resumed in a new simulation.
  Before, the route went on from that state rounded to float32 rather than the float64 state MuJoCo held at that point,
  and the run slowly left the unbroken one.
- A ball's restitution now matches its setting in scenes of free rigid bodies only: within 0.05 of it with up to 8
  free bodies (and at any count on the CPU's default route), within about 0.08, varying with impact speed, with more
  than 8 on MuJoCo Warp. A restitution of 0 now stops it within 0.07 of the impact speed; before, it rebounded at
  0.12-0.15, and 0.5 read about 0.55. A heavy ball hitting a wall there goes about half as deep (6.8 kg at 1.7 m/s:
  7.7 mm at the automatic step, against 14.3 mm). Balls of about 0.1 kg and under keep the earlier rebound. Boxes,
  capsules, cylinders and meshes, a ball on one of them, and every body in a scene with a robot, a kinematic object or
  a gripped chain keep the earlier contacts.

## [0.1.3] - 2026-11-13

### Added
- `alloy.isaaclab` (optional; needs Isaac Lab 3.0 in the same environment): Alloy as a physics solver for Isaac
  Lab's Newton backend. `physics=alloy` on Isaac Lab's `isaaclab` command (tasks that offer `physics=newton_mjwarp`)
  or `AlloySolverCfg` / `alloy_physics(...)` in Python runs the task unchanged on Alloy's rigid route at the task's
  own physics step, with per-world health checks; `rigid_step="alloy"` opts into 1/600 s steps and
  `noslip_iterations` into MuJoCo's noslip for grasps. Trained on one L4 GPU: Isaac-Cartpole-Direct,
  Isaac-Velocity-Flat-AnymalD, Isaac-Velocity-Flat-H1, Isaac-Reach-Franka; 18 rigid tasks step and reset. Isaac Lab `main`: four tasks
  step on the CPU. `docs/guide/isaaclab.md`.
- `alloy.envs.jitter_joint(..., carry=[...])` also carries particle objects: a liquid or a cloth named there moves
  with the jittered slide at each reset, as a carried free object does. Before, only free objects moved, so a cup on a
  jittered slide started beside its own water (with a 3 cm jitter, 20-41 % of the water was outside the cup after 10
  steps; carried, none). `docs/guide/envs.md`.
- `alloy.build(scene, clock_worlds=n)` (opt-in; 0, the default, is off): in a scene with several solvers, the up to n
  worlds that need the smallest exchange step move into a second simulation and step there at their own step, and the
  rest share a larger one. 64 worlds of a robot beside a towel with randomized pad masses on one L4 ran 1.12-1.24x
  faster with `clock_worlds=4` (three runs); a 4-world CPU rig ran 0.88x (slower). Not with `rerun_worlds` or
  `processes=` (`BuildError`); with liquid, MPM, the contact gate, the CPU's default MuJoCo route or user plug-ins the
  build warns and steps as before. `docs/guide/worlds.md`, Build options for speed.
- `scene.relax_contact_depth(name, depth=None)`: health judges the contacts of a robot, one of its bodies or an object
  by their depth past the contact's own MJCF `solimp` width (where that is wider than MuJoCo's 1 mm) or past `depth`,
  for contacts a file makes soft on purpose: Menagerie ANYmal C's feet rest 17 mm deep and now read OK standing.
  Reported depths stay true; `scene.save()` warns that its cards do not keep the setting (Scene JSON does).

- `alloy.il`: `EvalResult.invalid` marks and `GenerateResult.invalid` counts the episodes whose world went INVALID
  (the engine froze it) at any step. They are read from `info["invalid"]` at each step, because the env's reset
  clears the health record; such an episode is never a success and `keep="success"` does not save it.
  `docs/guide/imitation.md`.

- `alloy.il`: `HDF5Writer(resume=True)` adds to an earlier file and `next_obs_images=False` leaves the camera frames
  out of `next_obs`; `copy_hdf5(path, writer)` writes a file's episodes to another writer (a LeRobot dataset from
  a long HDF5 recording); `LeRobotWriter(resume=True)` appends to a closed dataset and `env_state=[...]` moves
  named state columns to `observation.environment_state`; `Recorder(manual=True)` leaves saving to
  `finish` (teleoperation). `docs/guide/imitation.md`.

### Changed
- `alloy.envs.placement.rest_on_floor` rests the object in every world by that world's own shift and returns the
  shifts per object as an array of shape (W,) (before: one float). It measured world 0 only and moved every world by
  that amount, so an object started at another height in another world began buried in the floor or floating above
  it. Worlds with identical starts get the same shift as before.
- `alloy.envs` delta and hand-motion actions: `ActionSpec(mode="delta")` and `EEActionSpec` start each step from the
  last target pulled back toward the joint's measured position (within `window`: 5 x `delta` for joint deltas; 0.03
  rad for an arm under `EEActionSpec`, set for the shipped Franka Panda's gains, the arm's lead shrinking as one
  vector so the hand keeps its direction), so a joint held by a table, a part or a stop no longer runs its target on:
  after a 3 s press on the table, an up command lifts the Panda's hand 1 cm in 4 steps (12-23 before). Costs: the
  servos now set the top speed, so an open-loop 10 cm hand command over 5 steps moves 4.1 cm and a slow servo moves
  slower (the custom-task example's pusher: 91 steps, 47 before). Trajectories change even where outcomes do not:
  the window binds in 86 of the GripperLift demonstrator's 100 steps, and its outcome is unchanged. With the
  shipped arms' firm contacts it does not change whether a world freezes. `window=float("inf")` restores the old
  behaviour.
  `docs/guide/envs.md`, `docs/guide/rl.md`.
- A cloth sheet from `add_cloth` whose material card has `elastic.cloth` (a woven card such as `cotton_woven_aniso`)
  now routes to `cloth_vbd` with the orthotropic membrane by default, alone or beside other moving things. Before, it
  went to `cloth_style3d` when alone, which barely drapes a KES-stiff fabric, and was refused beside other moving
  things. The membrane costs 1.9-2.5x the Style3D route on grid sheets, and the sheet drapes differently (vertices up
  to 0.13 m from where Style3D put them). `add_cloth`'s default `cotton_woven` is isotropic and unchanged: isotropic
  sheets, and woven mesh sheets without a route override, step bit for bit as before. A woven mesh sheet still needs
  the object card's override `route: {route: cloth_vbd, ...}`; on a mesh without warp / weft edges each triangle now
  carries one yarn, so the membrane no longer locks there. `docs/known_gaps/cloth-woven-drape-isotropic.md`,
  Workaround.
- A free-base robot's per-world `pos` array, and a robot's `x`, `y` or `z` drawn by `scene.randomize`, move the whole
  robot file as a `pos` tuple does: the root body ends at the row plus where the file places it. Before, they put the
  root body itself at the row, so a root body the file places above its origin (a quadruped's base) started in the
  ground (START_PEN, PEN_FAIL), and the same rows as a tuple started higher.
- `add_pour` now pours its `volume` within half a particle (0.03 ml at the default liquid accuracy, 0.004 ml at
  `"super"`), and `sim.liquid_volume` reads it. Before, the liquid waited at the nozzle as a whole block of particles,
  up to 11 % off the asked volume under 50 ml (20 ml poured 18.8 ml), and `"super"` rounded it twice. Every pour's
  particle count changes (`Pour-v0`'s 100 ml: 1,584 particles before, 1,562 now), so pour scenes do not repeat 0.1.0's
  results bit for bit.
- The shipped LEAP hand (`emb_e9_leap`) has MuJoCo Playground's joint and servo values (v0.2.0, Apache-2.0, cited in
  its `source.json`): joint damping 0.2 N m s/rad (was 0.03), armature 0.00149376 kg m² (was 0), frictionloss
  0.02 N m (was 0.001), and each servo's torque capped at ±0.2196 N m (was unlimited). As shipped, random finger
  targets flung a 7 cm cube off the palm: on 256 worlds every cube dropped and 59 % ended 0.5 m away (peak speed
  2.2 m/s median). Now, with every world starting alike, none drops, the fastest moves 0.18 m/s (0.20 m/s on 1,024
  L4 worlds) and no world goes INVALID. Under a rotation env's randomized reset (cube turned up to 180°) 15-19 % still
  drop, against 93-100 % before: the turned cube can start inside the fingers. The hand moves more slowly and presses
  more softly, so policies trained on 0.1.0's hand do not carry over. Over 4 s of noisy hold targets on 2,048 GPU
  worlds, about 0.7 % of worlds list a passing UNCONVERGED (0.3 % with 0.1.0's values; `sim.check()` stays ok): in
  one substep the solver's line search reaches the file's `ls_iterations` (50) while the solve itself converges;
  with `ls_iterations="100"` in the file's `<option>`, 1 world in 4,096 does.
- The shipped arms (`emb_e1_fr3`, `emb_e2_panda`, `emb_e3_ur10e_2f85`) have firm contacts: every robot geom's solref
  is 0.0025 s, 2x the 1.25 ms robot step (MuJoCo's default 0.02 s before; the 2F-85's pads 0.004 s), with solmix 1000,
  so the robot's contact governs a pair with a table or an object (on a user's own MJCF table at MuJoCo's default the
  hand presses 0.2 mm deep, not 2.1), and solimp 0.95 0.99 0.001 (was 0.9 0.95); the files' own table, floor,
  box and cylinder take the same solref and solimp. An arm pressing its hand on its table (a scripted
  press) froze 9 of 256 worlds INVALID with contacts 19 mm deep (FR3: 37 worlds, 23 mm); now none does, the
  press holds 0.85 mm deep once the hand has landed (FR3 0.86 mm, UR10e 0.48 mm) and random actions stay under
  1.2 mm, on the CPU and on a GPU. The hand still overshoots 1-5 mm for a step when it lands at 1.3 m/s. A 10 g part
  picked up with the Panda lifts as before, and a box on the table rests 0.001 mm deep (was 0.034) and slides with
  Coulomb friction. The `_route` and `_stiff` variants keep the source's contacts.

### Fixed
- `sim.drive` turns a kinematic fixture placed per world (`pos=(worlds, 3)`) about each world's own centre. Before,
  every world turned about world 0's, so the fixtures of the other worlds swung far off their places (1.5 m for the
  fourth of four worlds 0.35 m apart) with health ok. `velocity`, `angular_velocity` and `pivot` each take one
  (x, y, z) for every world or a (worlds, 3) array; another shape, a NaN or a negative `duration` raises
  `SimulationError` naming the argument (a (worlds, 3) velocity raised a bare `TypeError`). The same in
  `processes=N`.
- Liquid in a kinematic container moved by `sim.drive` stays simulated wherever the drive takes it. Before, liquid
  carried past the region the build set around the scene was retired and frozen in mid-air (`retired_particles`),
  with health ok: a closed box driven 0.75 m at 5 m/s left all its water behind.
- `sim.set_pose` on a kinematic body in a cloth, soft-body or cable (`rod_vbd`) scene starts it at rest. Before, the
  solver read the teleport as a move in one step: a 0.25 m teleport froze the world (`SPEED_FAIL`), and a 19 mm one
  gave the body 15 m/s and dragged the cloth under it, with health ok.
- Joint targets and raw controls that are not finite, not numbers or the wrong length raise `SimulationError`
  naming the joint (also from the command line). Before, a NaN passed the range check, was reported as held at the
  limit and reached the actuator, with health ok; a wrong-length list raised `UnknownNameError`. `sim.run` checks its
  targets before it steps, also when `seconds` is under half a control period (it steps whole periods, so none).
- On MuJoCo Warp (any GPU), one articulation with more than 6 joints in a chain (a 7-joint arm, the Franka Panda, a
  cable on the chain route) beside many free objects no longer makes the whole scene solve its contacts the slow
  way. 64 worlds with 200 boxes on one L4, env-steps per second before and after: the shipped Panda 44.7 and 1,669, a
  7-joint arm 108 and 2,021, a 1-joint arm with a 6-link cable 48.6 and 1,177 (25.8 and 740 with a free cable). A
  1-joint arm alone with the boxes runs about 2,100. Results match the old path within float32 rounding.
  `ALLOY_MJW_BLOCK_CHUNK=0` keeps the old path.
- `alloy.envs`: a rod's centre line `"<rod>.points"` is in the task state, for observations
  (`Task(obs=["cable.points"])`) and checks (`checks.fraction_inside("cable", ...)`), as `sim.state()` reads it.
  Before, both raised `EnvError`, whose fix named `"<rod>.pos"` (the first segment only).
- A body with several MJCF joints (a two-axis wrist, a planar base) loads as one joint of one coordinate per MJCF
  joint (`"<robot>/wx_wy"`, named in the file's order). Each of its actuators now drives its own coordinate, and
  `sim.set_targets` takes one value per coordinate (`[0.3, -0.4]`, or `(worlds, n)`), each to its own servo; a single
  number raises `SimulationError` naming the count. Before, every actuator of such a body drove its first
  coordinate (told to tilt about y, the wrist tilted about x), and only one `<general>` servo of the body was
  reachable. A fixed joint has no coordinate: a target on it raises `SimulationError`. Before, it moved the next
  joint. A `set_targets` call that raises now writes none of its targets.
- Robot part names keep the name in the robot's file whole, also when it holds "/" (MuJoCo Menagerie's two-arm
  files name parts "left/waist", "right/waist"): `"<robot>/left/waist"`. Before, the name was cut at its last "/",
  so both arms' parts shared one name and one arm could not be reached. A tendon or site actuator now belongs to its
  robot (`"<robot>/<actuator>"`), and an actuator name that two robots share works only with its robot's prefix
  (the bare name raises `UnknownNameError` naming both); before, the bare name drove the first robot alone.
- Joint targets and raw controls follow MuJoCo's meaning for every servo: an `<intvelocity>` takes
  `kind="velocity"` (a position target read as a velocity ran it to its stop); a target of the kind a joint's servo
  does not take raises `SimulationError` naming the kind to use (it did nothing); `sim.set_controls` on a geared
  actuator is MuJoCo's ctrl (the joint moves ctrl / gear, not ctrl), while `sim.set_targets` stays a joint value.
- A mesh with NaN or inf vertices (`add_mesh` or an object card), and a moving mesh whose faces are wound both ways
  (so it encloses no volume and the body would have no mass), raise an API error with a fix. Before, they raised a
  bare `LinAlgError` or `ZeroDivisionError`. Static and kinematic meshes and cloth sheets build as before.
- `.usdz` robot files load like `.usd` ones (`add_robot`, `alloy.load`). `alloy.load` and `Scene.load` of a mesh file,
  or of a file they cannot read, raise `SceneError` naming what to use instead (`scene.add_mesh`, `alloy.create`, or
  the formats they read). Before, they raised a bare `UnicodeDecodeError`.
- `sim.check()` names a robot's parts in a start overlap by the name given to `add_robot`
  (`counter_arm/floor and floor_dog/RL_calf`). Before, it used the robot file's model name, which two robots from one
  file share.
- `scene.add_object` takes the shared keywords of the `add_*` calls: `static`, `kinematic`, `rigid`, `material`,
  `mass`, `density`, `friction` and `restitution` (a card made of several materials refuses the value keywords with
  a fix). Before, they raised a bare `TypeError`, although a build error's fix said to pass `rigid=True`.
  `scene.set_friction` and a low `friction` range in `scene.randomize` work on library objects and on the objects
  of a scene from `Scene.load`, as on primitives. Before, they raised or failed the build.
- `alloy.objects()` lists the objects on disk, like `alloy.materials()` and `alloy.scenes()`. Before, it raised
  `AttributeError` in a fresh process.
- `scene.save()` copies what its cards take from the scene's own card roots (`Scene(roots=[...])`, `alloy.create`
  and imported cards: object and material cards, pair cards, looks and mesh files) into the saved folder, so it
  loads on its own. Before, it failed to load ("no material ...", "file not found"). `save()` also warns for what
  cards cannot keep, each with `scene.to_json()` as the way to keep it: heat (temperatures, heaters, `set_heat`),
  `cloth_accuracy`, `liquid_accuracy`, `rigid_step`, `set_joint_yield`, robot randomization, `add_push` and object
  anchors. Before, these were dropped silently. `export_scene` of an `alloy.Scene` lists them in `unmapped`.
- An object with its own `mass=`, `density=`, `friction=` or `restitution=` keeps its material's look (in
  `alloy.vary`, the full-quality camera and saved cards). Before, it lost it.
- `scene.save()`: an `mjcf=` robot's file is referenced relative to the saved folder, so the folder still loads
  after it is moved or copied (it held an absolute path), and a missing file's error names the file instead of a
  material card. `save()` now warns for what its cards cannot keep: per-world starts (`pos=` rows, randomized robot
  bases, cloth and liquid starts), a kinematic fixture as one body, and a precise fit's control period, each with
  `scene.to_json()` as the way to keep them. Before, a loaded scene silently started every world at world 0's start.
- `scene.randomize("<fixture>", "x" | "y" | "z", lo, hi)` on a kinematic fixture moves the whole fixture (its `z` is
  the bottom face, as in `pos`). Before, it raised with a fix naming a wall part, which then failed the build.
- `sim.reset()` after `sim.settle()` in a scene with `add_pour` goes on from the settled pour. Before, the reset
  started the pour's emitter over while the settled liquid stayed poured: its first layer pulled settled particles
  back to the nozzle, the pour stalled and the poured mass was counted twice. A pour never settled still starts over.
- `sim.reset()` puts joint targets and controls back to the start (as before the first command, or as at
  `sim.settle`). Before, they stayed, and a robot snapped back toward its last targets after the reset.
- Contact-force plug-ins (`Tactile`, `GelSensor`, `ModalSound`, `Thermal`) work in a scene that also holds liquid,
  cloth, a soft body or sand, and read the same forces there as in a rigid-only scene. Before,
  `alloy.build(scene, plugins=[...])` raised a `BuildError` that named "method" and gave an unrelated fix. A scene
  whose MuJoCo solver steps on Newton's contacts (SDF or mesh collision shapes) now raises a `BuildError` that says
  so, where its readings were 0; and a plug-in's own fix reaches `BuildError.fix`.
- `examples/export/rerun_live.py` runs with `--worlds 1` and logs the last frame of each picked world. Before,
  `--worlds 1` raised `SceneError`, and on a GPU the last captured frame never reached Rerun.
- `alloy.il` with observation groups (`Task(obs={...})`): `LeRobotWriter` and `replay` raised `KeyError: 'state'`.
  Every group is now recorded with its own column labels (`DatasetSpec.obs_key_names`), the first group is LeRobot's
  `observation.state`, and `replay` compares every group.
- `teleop_loop` saved attempts nobody approved: with `keep="all"`, an episode that ran into the task's `max_steps`
  was written without Enter. While the loop runs, an episode the env ends is now kept only when it passed the
  success check; `finish(save=True)` honours a `keep` function; `saved` and `discarded` count episodes per world.
- `alloy.il.generate` on an env with `autoreset="disabled"` kept stepping ended worlds and counted them at every
  step; it now resets them. `evaluate(Recorder)` saved the extra episodes of quick worlds; it now saves the counted
  ones only. Absolute-mode actions are recorded as the env clipped them. A camera-only task and a `/` in a camera
  name no longer crash `LeRobotWriter`. `Recorder`, `evaluate`, `generate` and `replay` raise `EnvError` with a
  fix for an env that is not a `VectorEnv` (before: `AttributeError`). `replay` refuses an env it cannot restore
  (a rod on the rod solver, an actuator model) instead of reporting a wrong zero error, and restarts observation
  layers, so `ObsHistory` replays exactly on the CPU. An episode's `obs` and `next_obs` share one array.
- `alloy.il.replay` with `Scene.randomize`: episodes were replayed in world 0, 1, ... whatever world recorded them,
  so with per-world draws they ran with another world's values and did not match (2 of 6 matched in one case).
  Each episode
  now replays in the world that recorded it, its `start` keeps the robot and gravity draws, and `replay` raises
  `EnvError` when the world holds other values (always for values drawn at every reset). `ReplayResult` gains
  `invalid` and `recorded_invalid`; a world whose episode is over is no longer driven by that episode's last
  action. `SpaceMouseTeleop` takes `link=` and `keys=` (discard and quit from the terminal's keys).
- `VectorEnv` in `"absolute"` action mode: `obs["action"]` and `env.state["action"]` now show the action as the env
  clipped it to the `ActionSpec`'s `low` / `high` (they showed the raw one while the joints got the clipped one). A
  policy that reads its own last action sees the change. `alloy.envs.realism.obs_names(env)` works on an env
  without observation groups (it raised `EnvError`).
- A pour beside cloth, in a scene with no free rigid body, no longer splashes a fifth of itself over a container's
  rim with health OK (20 % of a 6 ml and 24 % of a 20 ml pour from 12 cm, all over the rim). Liquid beside other
  solvers now takes a quarter of its stability step while it moves fast, and its largest step at rest. A pour
  beside cloth costs 1.4x as much on an L4, and a cloth sheet landing on water up to 2.7x while it stirs the water;
  a cup of water at rest beside cloth takes the same liquid steps.
  `docs/known_gaps/liquid-pour-over-the-rim.md`.
- `sim.drive` moves its object for exactly its time: the first physics step after the call already moves at the
  drive's velocity, and a drive with a `duration` ends at rest on its last step. Before, the velocity came one step
  late and stayed one step past the end, so a press that ended by its duration pushed the material one step longer
  (3.2 % more holding force in one measured press on a paste).
- `alloy.envs`: a world the engine froze (INVALID) earns nothing: on its last step its reward (except
  `invalid_penalty`), reward terms, success and last observation read 0, and the rsl_rl, SB3 and skrl wrappers
  never bootstrap its value; `info["episode"]["invalid"]` and `env.invalid_episodes` count frozen episodes, the
  latter since the env was built. `docs/guide/envs.md`, Ready-made environments.
- The rsl_rl wrapper no longer resets every env with seed 0; the env's own seed stands.
- `ObsNoise` works on a flat observation list (it raised `EnvError`). `docs/guide/rl.md`.
- With `scene.randomize` on an object (friction, density, start `x` / `y` / `z`, ...), resetting some worlds (one
  world's episode ending in a `VectorEnv`) now resets only those worlds. Before, every world's objects went back to
  their start while their episodes went on, with no error or warning. Such a reset also no longer writes the drawn
  values again, so it costs about what a reset without the rows does and a GPU run keeps its captured step graphs:
  on one L4, 1,024 worlds, a reset of 32 of them and the step after it took 1.0 ms + 7.7 ms, against 27-37 ms + 26 ms
  before (on an AMD EPYC 7B13 CPU a density row's reset took 2.2 s before; now 13 ms, as without the row). Rows
  drawn at every reset (`when="reset"`) still cost more: `docs/guide/randomize_robots.md`.

## [0.1.2] - 2026-11-12

### Added
- `python -m alloy` runs the command line and prints its help when given no arguments.
- A build that finds no NVIDIA GPU (no `device` given) and holds cloth, liquid, sand or paste, or soft bodies logs one
  line per process: such scenes run many times slower than real time on the CPU, `alloy.estimate` predicts the wall
  time, and `device="cpu"` keeps the line quiet. Rigid and robot scenes and rods get no line; it never stops a build.

### Fixed
- `alloy.__version__` reads the installed package's version and never raises.
- `alloy.compose`: a scene that leaves out an object it was asked for fails the call (`ok` false), and when the Hub
  library is out of reach the fix names `pip install "alloy-engine[import]"`.
- `python -m alloy.looks --fetch-textures` without the `import` extra prints one error naming
  `pip install "alloy-engine[import]"`, not a traceback.
- Camera kernels compile without the compiler's integer-constant warnings on the first render; images are unchanged.

### Documentation
- New answer pages: a FAQ, how-to pages and comparison pages.
- The docs take the corrections of a review of the website; the "What ships" page says when its lists were read
  from the package instead of naming a version.
- The README and the PyPI description say Python 3.11 or 3.12, with a `uv` line for systems whose Python is older,
  and give the range of worlds a robot arm runs at as the docs measure it, in place of "hundreds".
- Internal team and work-item names are gone from the known-limit pages, other prose and data notes (exported JSON,
  card and MJCF notes); the release scan now fails on them.

### Packaging
- PyPI keywords and classifiers; the repository has `CITATION.cff` (how to cite Alloy) and `context7.json`.
- `llms.txt` ships in the wheel and the sdist, beside `AGENTS.md`.

## [0.1.1] - 2026-11-11

### Fixed
- Camera environments with `vary=True` on a fresh install (texture images not downloaded): a varied look is drawn in
  a mid-tone of its texture family (a brown floor, honey wood, mid-grey plastic), with one warning naming
  `python -m alloy.looks --fetch-textures`. Before, such a look fell back to its white placeholder colour, so the
  floor drew pure white. A colour map the fast camera cannot read draws the same tone. `docs/guide/envs.md`.
- The fast camera behind Gym camera observations reads a look's base colour as sRGB and converts it to linear, as the
  full-quality camera does. Before, it used the colour as linear, so varied looks were drawn too bright. A policy
  trained on 0.1.0's camera images sees different colours: retrain it.

### Changed
- `PickCube-v0`'s rsl_rl recipe (`train_rsl_rl.RECIPES`) takes an entropy term of 0.003 (it had none), and the
  documented training command runs 1,000 iterations (it ran 500). Results vary between runs on a GPU, so watch the
  success rate the trainer prints. `docs/guide/rl.md`, Arm pick.

### Documentation
- Every page's "Limits" section is now "Keep in mind", and the limits page is the "Keep in mind" page.
- A new page, "What ships", lists the ready-made environments with their observations, the robots with their source
  licences, and the object libraries with their counts, licences and how to get each.
- The README and the docs home page link the Apache-2.0 licence and the GitHub repository, where issues,
  contributions and private security reports go.

### Packaging
- `pip install alloy-engine` is the only install path: there is no `install.sh`, and the GitHub Release carries
  release notes only.
- The package metadata links the GitHub repository and its issue tracker, as in 0.1.0.post1.

## [0.1.0.post1] - 2026-11-11

Package metadata only: the PyPI page links the GitHub repository and its issues. The code is 0.1.0's.

## [0.1.0] - 2026-11-11

### Fixed
- Gym camera observations (`camera=True`, `Task(cameras=...)`) show the scene as `sim.render()` and `alloy.Camera`
  draw it: library looks, scanned objects' photo textures and a grey ground. Before, they drew the engine's palette
  colours (the ground and the first object the same blue) and left out every look, so `alloy.vary` textures never
  reached a policy. A scene variant's looks (`vary=True`) are drawn by the fast camera as each look's colour and
  colour map, where `Camera.render(sim)` uses the full-material tracer; on one L4 at 512 worlds that costs nothing
  measurable (`PickPlace-v0`: 15,100 64 px images/s, 14,400 before). A policy trained on the earlier images sees new
  colours: retrain it. `docs/guide/envs.md`.
- Since 0.1.0rc2: `sim.stop_drives()`, `sim.set_pose()` and `sim.reset()` bring a kinematic object to rest in both of
  the engine's states. Before, a stopped or reset tool kept its old speed or pose every second exchange, and MPM
  material under it (sand, paste, dough) kept being pushed.
- Since 0.1.0rc2: two `add_pour` calls in one world each pour their own volume. Before, they split their particles
  evenly: a 20 ml and a 60 ml pour delivered 38 ml each.
- Since 0.1.0rc2: MPM (sand, paste) at many GPU worlds no longer lets grains into or through moving and free bodies.
  Warp's packed sparse-grid cell lookup lost float32 precision as a world's offset grew, so higher-numbered worlds
  behaved worse (a blade let sand through in 57 of 64 worlds); Alloy now picks the cell in integers. Validated up to 64
  worlds on one L4; one world is unchanged bit for bit.

### Added
- Scanned library objects (YCB, Google Scanned Objects, Objaverse and the other fetched scans) render with their own
  photo texture in both cameras: the quick look (`sim.render()`, `alloy.Camera`) and `quality="full"` / looks. The
  camera reads each scan's `visual.glb` (mesh, UVs, base-colour image); a look that names the object still wins.
  Without Pillow (the `import` extra) an object keeps its flat colour. Scans that simulate as soft bodies show it
  too (fixed at the build pose; thin streaks can show along the texture's seams). Cost: 2-3 % on a scene of scans on
  one L4, none on scenes without them. `docs/guide/camera.md`.
- Opt-in building blocks for RL environments (`docs/guide/rl.md`): hand-motion actions for arms by differential
  inverse kinematics (`alloy.envs.diffik.EEActionSpec`, `panda_reach_ee`), per-world observation noise, observation
  delay, observation history and action delay (`Task(layers=[...])`, `alloy.envs.realism`), and PD / DC-motor
  actuator models computed at every physics step on motorized robots (`alloy.envs.actuators`) (first versions,
  tested on the CPU; limits in the guide).
- `alloy.viewer` (opt-in: `import alloy.viewer`; each backend needs its own optional package): `launch(sim,
  backend="viser" | "gl" | "rerun" | "usd" | "null")` shows a running sim with pause, single steps, world picking
  and contact points and forces of rigid bodies and robots; `alloy.viewer.contacts(sim, world=0)` returns those
  contacts as numpy arrays and `alloy.viewer.force_on(sim, name)` the net contact force on one object;
  `python -m alloy.viewer <scene>`. The viewer only reads the sim, unless you opt in (both off by default; tested
  on the CPU: sliders on a one-hinge arm, dragging on free boxes; dragging robot links not tested):
  `launch(..., joints=True)`
  (`--joints`) adds a slider per position-driven joint that calls `sim.set_targets`, `launch(..., drag=True)`
  (`--drag`) lets the mouse pull a body with a spring through `sim.apply_wrench` (`alloy.viewer.MouseDrag` grabs from
  a script). `docs/guide/viewer.md`.
- Fixes from a user-side audit of the release candidate. `sim.set_targets` (and `run(targets=)`) holds a position
  target inside the joint's range, each world on its own: a target past a stop rested up to 0.32 rad past it, health
  OK. It raises for a robot with no actuators, naming the MJCF line to add (the call returned and the arm fell).
  `sim.set_controls` on an MJCF `<position>` or `<velocity>` actuator moves its joint (it was ignored). `sim.reset()`
  returns to the `sim.settle()` state in a scene with `scene.randomize` too (it went back to the build start). A robot
  joint named like its body keeps its `"<joint>.q"`. `scene.randomize` refuses, when it is called, a card number the
  engine cannot draw per world (`material.surface.restitution`, `material.elastic.poisson`: both failed the build)
  and lists the fields it can. A malformed robot MJCF and `add_liquid` of a solid material raise `SceneError` (a bare
  XML ParseError; one rigid box named like the liquid). A soft body from a mesh without the `mesh` extra raises
  `FeatureUnavailableError` with the install command. `Scene.save` to a relative folder writes a robot path that
  loads. At the world-count limit `alloy.build` raises `BuildError` (field `worlds`) naming what ran out (Warp's
  32-bit array size, or GPU memory) and at how many worlds, instead of a raw Warp error. A step graph that runs out
  of GPU memory steps down without losing the GPU (it failed later with "CUDA error 101: invalid device ordinal").
- Health, from the same audit. Under `profile="precision"`, `sim.step` and `sim.run` raise `alloy.SimulationError`
  in the control period a world becomes INVALID, naming the world, once per world until `sim.reset()` (they
  returned, and only `sim.check()` raised, with an engine error that was not a `SimulationError`); `sim.check()` and
  `sim.report()` read afterwards. Each world row of `sim.health()` has `seen_flags`: every flag raised since the
  build or the last reset, also one that has gone, and `sim.check()` lists those as warnings without changing `ok`.
  A steady energy gain of more than 10 % of the scene's energy scale with nothing adding it is ENERGY_JUMP
  (`energy_gain` in the row; `sim.check()` is not ok): a free mesh body that pumped itself upward read OK.
  PEN_WARN and PEN_FAIL read the contacts of rigid bodies and robots on the MuJoCo routes too (they read none
  there); `peak_penetration_m` keeps the deepest overlap since the build or the last reset. JOINT_FAIL covers a weld
  or connect more than 1 mm or 1 mrad apart, and is measured under `profile="fast"` too. TUNNEL (INVALID) is a point
  inside a moving body (a free body or a robot link) that passed through a static box, plane, sphere, capsule,
  cylinder or box-like convex mesh. A `profile="precision"` build warns once when its runs cannot repeat bit for bit
  on the device (`docs/guide/health.md`).
- `sim.render(cam, output="depth" | "segmentation" | "body_id" | "world_id")` or a tuple such as `("rgb", "depth")`:
  depth in metres and per-pixel body ids from one ray per pixel (no sampling noise); colour images are
  unchanged and a depth-only render traces no colour. `alloy.camera.aov.segment_labels(sim)` names the ids. Fast
  camera only. `docs/guide/camera.md`.
- `sim.render(cam, output="instance_id" | "semantic" | "flow")`, alone or in a tuple: one id per drawn shape
  (`alloy.camera.aov.instance_labels`), classes you name (`set_semantics`), and optical flow in pixels since the
  camera's previous flow render, from the poses of rigid bodies (robot links and kinematic objects included), static
  shapes and the camera (NaN on cloth, soft bodies, films and liquid). Fast camera only.
  `docs/guide/camera.md`.
- `scene.randomize` for robots and gravity, off by default: per-world `mass_scale`, `mass_add`, `com_x/y/z`, `friction`,
  `kp_scale`, `kd_scale`, `armature(_scale)` and gravity `x/y/z`, drawn at build or again at each reset
  (`when="reset"`); `sim.randomized()` reads the draws (`docs/guide/randomize_robots.md`).
- Reset-time events, off by default: `start_x/_y/_z` and `start_yaw` rows move a free-base robot's start at every
  reset, and `scene.add_push(body, force, every, duration)` pushes a body at random times per world (`sim.pushes()`
  reads the force held now). Checked on the CPU only (`docs/guide/randomize_robots.md`).
- `alloy.envs.wrappers`: adapters from `VectorEnv` to rsl_rl, Stable-Baselines3 and skrl (each an optional install),
  one-command PPO training (`python -m alloy.envs.wrappers.train GripperLift-v0 --lib rsl_rl`), and observation
  groups, `Task(obs={"policy": [...], "critic": [...]})`, for a critic that sees more than the policy
  (`docs/guide/rl.md`).
- `alloy.envs.loco`: a reference legged environment, the Go2 on flat ground following velocity commands, with
  pushes, gravity / accelerometer / solver foot-force observations and `python -m alloy.envs.loco.train` (rsl_rl
  PPO; a walk learned in the two runs we finished; `docs/guide/locomotion.md`).
- `alloy.envs.loco.go2_rough` (opt-in): the same Go2 velocity task on a heightfield of terrain tiles (rough
  ground, waves, blocks) in 5 levels, with a per-world level curriculum (`docs/guide/locomotion.md`, Rough terrain).
- `alloy.envs.make("ClothFold-v0", grasp="attach")` (opt-in): the cloth vertices between the closed fingers are held
  to them by a spring, a shortcut rather than a friction grasp; the default `"friction"` is unchanged
  (`docs/guide/envs.md`).
- Opt-in, `ALLOY_PROXY_TENSOR=1`: a robot or rigid body's proxy in cloth, soft bodies, rods and film moves with the
  body's own 3x3 inverse mass instead of one scalar, so a body that moves easily along one axis and hard along
  another pushes the cloth as it should. CPU only so far (`docs/known_gaps/coupling-proxy-isotropic-mass.md`).
- Engine layers, opt-in: `FilmConfig(limiter="stress")` holds a film past its stretch cap with a force inside the
  solve: a 15 µm HDPE strip with 50 g hung on it stays 1.03x its cap, where the default projection let it run to
  24-34x (`docs/known_gaps/film-projection-force.md`).
- `alloy.build(..., contact_capacity=n)`: MuJoCo's contact buffer, n contacts per world on average (MuJoCo Warp's
  pool of n x worlds is shared by the worlds; on the CPU's MuJoCo C route, the contacts health reads). Unless
  `constraint_rows` is given it also sizes the constraint rows per world from the model: each contact's rows x 2 n
  plus the scene's joint-limit, equality and friction-loss rows. Default None sizes it from the scene as before (8
  per collision shape, at least 64, 6 rows per contact). Worlds that overflow are flagged CONTACT_OVERFLOW and
  frozen as before, and the console now says so once, naming the first flagged world and the n to pass, instead of
  MuJoCo Warp's warning on every step. One process only for now.
- `alloy.build(..., constraint_rows=m)` (opt-in): MuJoCo's constraint row buffer, rows per world. The default gives
  a robot of hundreds of collision shapes thousands of rows where it uses tens, and MuJoCo Warp runs its solver
  over the whole buffer. Sized to the scene, on one L4 at 2,048 worlds a Panda stepped 1.55x faster
  (`contact_capacity=64, constraint_rows=384`), a LEAP hand 2.2x and a G1 (512 worlds) 1.7x, with the same
  contacts, rows and solver iterations. With too few rows the worlds are ROW_OVERFLOW and frozen. The CPU's MuJoCo
  C route sizes its own rows.
- A scene with a robot can step its rigid bodies coarser than the fixed 1.25 ms (opt-in): `scene.rigid_step =
  seconds` pins a step, and an explicit `scene.rigid_step = "auto"` takes the largest step the robot's caps allow
  (its servos, joint damping, contact time constants, link speed against the contact gap, loose bodies in its
  file). The shipped Go2 gets 2.5 ms, half the physics steps. A pinned step is taken as asked, and the build warns
  once, naming each cap it breaks and the cost. A robot scene that never sets `rigid_step` steps bit for bit as
  before (`docs/known_gaps/robot-fixed-step.md`).
- `alloy.estimate` is fitted to measured runs. Wall time: 55 rows (a robot carrying water, bins, an arm, cloth, a
  pour; 1 to 512 worlds; CPU and one L4), within 1.3x on each; it was up to 18x optimistic. GPU `memory_bytes`: 20
  rows on one L4 (1 to 16,384 worlds), within 1.2x on each, with the 0.4 GB a GPU run holds before its first world;
  it read 4-6x low. The keys are unchanged, and a scene unlike the rows can be further off (`docs/guide/worlds.md`).
- Cloth and soft bodies take their velocity from each step's displacement, with the part float32 positions round
  away kept beside them: a sheet falls at g at every substep and height. Before, a coupled cloth at z 2 m fell at
  0, g or 2 g depending on the substep. Default cloth and soft-body results change bits on the CPU and the GPU.
- A cable on the chain route, twisted by a gripper and held, keeps its twist on MuJoCo Warp (any GPU): all of it
  after 5 s on one L4 with pads that slide on the wrist (87 % before), 99.8 % with pads welded to it (52 % before),
  at 1.00-1.04x the cost. On the CPU's MuJoCo C route it creeps as before.
- `sim.reset()` puts the coupled exchange clock back too, so in a scene with several solvers the run after a reset
  repeats the first one bit for bit on the CPU. The adapted clock stayed before (water ended 3.4 mm apart).
- Cutting: by default a blade cuts only seams aligned with it (`docs/guide/cutting.md`). Over 2,048 random stroke
  cuts of a block on one L4, clean episodes rose from 1,431 to 1,959 and failed ones fell from 615 to 86.
- MJCF weld / connect equalities that keep MuJoCo's default solref / solimp take the stiff rule of scene
  attachments by default, sized for the rigid step (30 kN on a 25 kg pair stretches the weld 0.015 mm instead of
  70 mm; 1,024 and 4,096 worlds, no cost measured); rows that set their own are kept, and
  `alloy.build(scene, welds="mujoco")` keeps every row as written (`docs/known_gaps/weld-stretch-under-load.md`).
- Cloth and soft-body self-contact rows are sized to the scene by default: each row grows between control periods
  before it fills, and a period that dropped pairs re-runs. They held 32 / 64 pairs per vertex / edge, and a pile
  of folded sheets dropped pairs. A row stops at 256 pairs; past it pairs drop and health flags
  SELF_CONTACT_OVERFLOW as before. Opt-in, `ALLOY_SELF_CONTACT_CROSS=report` or `invalid` measures whether dropped
  pairs let surfaces pass through each other (`docs/known_gaps/vbd-self-contact-buffer-overflow.md`).
- A kinematic fixture (`add_open_box`, `add_open_cylinder`, `add_block_with_hole` with `kinematic=True`) is one
  kinematic body: `"<name>"` is its state (the floor's frame; `"<name>_floor"` still works), and `sim.drive(name)`
  moves it. A kinematic cup beside a 2-dof arm is 8 dofs, was 152 (25 bodies). Its wall parts have no state of their
  own (reading one names the fixture). Static fixtures are unchanged.
- Cloth that keeps a crease: a cloth material with KES bending hysteresis (`elastic.bending_hysteresis`, KES 2HB in N)
  holds a fold by fibre friction, M = B K + HB sign(dK/dt), so a released fold keeps the curvature HB / B instead of
  springing back flat. The library card `cotton_single_jersey` (KES-sourced B, 2HB, MIU) has it; other cloth cards
  are unchanged. `docs/guide/deformables.md`.
- `Scene.add_rod(name, points | length=, radius=, direction=, material="cable_usb_composite", segments=,
  clamp="start" | "end" | "both")`: an elastic rod (cable, wire, cord) that bends, twists and stretches as its
  material says, rolls on surfaces and can be gripped and twisted by a robot. A clamped rod steps on the elastic-rod
  solver (`sim.routes()` says `rod_vbd`); state keys `"<rod>.points"` (centre line) and `"<rod>.twist"` (rad per
  joint); the CLI recipe kind `"rod"`. `docs/guide/rods.md`.
- A robot's grip on an elastic rod (`rod_vbd`) holds once the gripper stops: the rod's contacts with the robot keep
  their friction anchor from step to step (sticky contact matching), so a twisted cable no longer runs on or
  untwists (was +1.9 % after 0.7 s, and a hard squeeze untwisted it). `docs/guide/rods.md`, Robots.
- Liquid beside a moving body steps about 1.3x faster: the coupled clock takes the liquid's own stable step (its
  CFL step already carries its margin) instead of half of it, so the liquid runs half as many substeps (2 instead
  of 4 in a tilting jug beside a free ball). The liquid laws hold as before.
- `Scene.add_rod(..., clamp="start", route="chain")`: a clamped rod stays on the chain, its first segment welded
  where it is built, instead of the elastic-rod solver. Beside a robot that is about 60x faster on the CPU (a
  gripped, twisted cable: 1.3 s per simulated second, was 83), and the grip holds as on `rod_vbd` on the CPU; on a
  GPU a held chain stays with its grip too (it crept back about 2 % per second until a stick anchor on the gripped
  links' friction rows; `docs/known_gaps/rod-chain-grip-slips.md`). `route="rod_vbd"` asks for the elastic rod.
  `docs/guide/rods.md`, Robots.
- An elastic rod beside a robot steps 1.7x faster on the CPU: a solver entry of rods only runs 4 VBD iterations
  instead of 10 (the grip holds as before). The cable squeezed by two pads takes 95 s per simulated second, was
  161. `docs/known_gaps/rod-vbd-beside-robot-cost.md`.
- `Sim.apply_wrench(name, force=, torque=, at=)`: hold a force and a torque on a body (a rod's `"start"` or
  `"end"` segment) until changed; `sim.reset()` removes every wrench. Health counts its work as energy from outside,
  so a held push raises no ENERGY_JUMP.
- Wires that keep a bend: a rod whose material has `plastic.model: j2` (`copper_c110` and other metal cards)
  routes to `rod_vbd`, yields per joint through a 220-fibre round section (von Mises, linear hardening) and keeps
  its set; `"<rod>.plastic_strain"` gives each joint's largest equivalent plastic strain. `docs/guide/rods.md`.
- A rod that needs more than 40 segments steps on the elastic-rod solver (`rod_vbd`) instead of failing to build;
  short elastic rods stay on the chain route. `alloy.estimate(...)["routes"]` names each object's route and the
  reasons, as `Sim.routes()` does.
- `alloy.build` picks the rigid step per scene: a rigid-only scene of free bodies steps its rigid bodies at the
  largest stable step (up to the control period), with contacts no softer than before. Up to 8 free bodies step on
  rk4; more step on implicitfast with rk4's gains and the friction pull capped, on MuJoCo Warp (any GPU, or
  `ALLOY_CPU_ROUTE=warp`; the CPU's default MuJoCo C route keeps rk4). Stacks of 27 cubes (1 m, 10 kg/m³, resting;
  3,072 worlds, one L4) step at 5 ms and ran 766x real time, none falling (2.5 ms on the CPU's MuJoCo C route).
  Newton's SolverMuJoCo at the same pinned commit with one 10 ms step ran them at about 1,800x (five runs) with
  softer contacts (rest error 1.00 mm against 0.19 mm): the default is about 2.4x slower there
  (`docs/known_gaps/rigid-free-body-solve-cost.md`). `Scene.rigid_step`: "auto" (default), "fixed" (1/600 s snapped
  down to the dt grid, the step used before: 1.25 ms at dt 0.01), a step in seconds, or "fast" (also `profile="fast"`
  when `rigid_step` was never set), an opt-in fast mode: contacts may soften to a rest sag of 1 % of a body's thinnest
  member, and free bodies step on implicitfast on MuJoCo Warp (rk4 on the CPU's default MuJoCo C route), so 1 m cubes
  step at 10 ms and the same stacks ran 1,973x to 2,135x in three runs, about the same as the plain solver. Its cost:
  a 30 cm cube sliding at 5 m/s rocks under 1° and lifts 1 mm with the friction pull capped (MuJoCo Warp), 6° and
  8 cm on the CPU's MuJoCo C route (2° and 1 cm at 2 m/s); the build warns. `sim.routes()` and
  `alloy.estimate(...)["rigid_step"]` report the step and what bound it. Robots, kinematic objects, deformables,
  precision fits, plug-ins and the precision / debug profiles keep their step bit for bit.
- `alloy.Scene(cloth_accuracy="default" | "super" | "fast")`: cloth solves close to convergence by default, with 40
  accelerated solver iterations per step (adaptive start, Chebyshev) instead of the cloth solver's own 10: a draped
  sheet stretches 0.36 % instead of 12 % and hangs at drape coefficient 0.728 instead of 0.19 (converged 0.74). It
  costs time: a 64-world drape on one GPU ran 2.7x slower than at 10 iterations, and on the CPU each cloth step runs
  4x the iterations. "super" runs 60 iterations, about 1.4x the default's cloth cost, for 0.34 % and 0.732. "fast"
  runs the cloth solver's own 10, about 0.4x the default's cloth cost (0.25x on the CPU), for runs where the cloth's
  shape does not matter: far from converged (12 %, 0.19), and under CLOTH_STRETCH's warning, so health reads OK. Soft
  bodies in the same scene step with the same setting; a scene without cloth is unchanged. Kept in Scene JSON;
  `alloy.estimate` shows every setting under `"cloth"` and charges cloth per solver iteration (`["vbd"]`: its
  particles per world, iterations and self-contact).
- Health flag CLOTH_STRETCH: a cloth stretched more than 50 % past its rest shape makes its world DEGRADED for the
  rest of the run (`sim.check()` stays ok); past 500 % the sheet blew up, so the world is INVALID and `sim.check()`
  names the cloth with a fix. `sim.health()` rows carry `cloth_stretch` (`percent`, `cloth`, `level`), the run's
  peak. Before, a sheet that blew up but stayed finite and slow read OK.
- `Scene.set_joint_yield(joint, torque= | force=, hardening=0)`: a hinge's or slide's spring (MJCF `stiffness`,
  `springref`) takes a permanent set past a yield torque (N m) or force (N), with optional linear hardening;
  `"<joint>.rest"` (where the spring rests now) and `"<joint>.plastic_strain"` (accumulated yield) in `sim.state()`
  and `run(record=)`; kept in Scene JSON. MuJoCo routes only; a scene with cloth, liquid or a soft body refuses it.
- `Scene.add_cloth` with a paper or plastic film material (`paper_80gsm`, `paper_kraft`, `paper_board`, `film_hdpe`,
  `film_ldpe`, `film_lldpe`) builds a film sheet (`cloth_film`); it failed to build before (the sheet was taken for a
  paste). A paste material (dough, mayonnaise, ...) raises `SceneError` at `add_cloth`, naming `add_box` as the fix.
- Metal sheets that dent: `Scene.add_cloth(..., material=<a metal with a yield law>)` (`aluminium_6061_t6`,
  `steel_tinplate`, `stainless_304`, ...) steps on the film route (`cloth_film`), whose hinges bend as the metal's
  section (pure bending within 0.3 % of the closed form) and keep a permanent bend past its yield;
  `"<sheet>.plastic_strain"` (worlds, vertices), the largest plastic strain in the sheet at each vertex, in
  `sim.state()` and `run(record=)`. Other cloth and metal objects keep their routes.
- Coupled scenes (solvers that exchange forces, such as cloth beside a robot): a world's first escalation (its
  energy or residual check tripped) now re-runs only that period, one step finer, and the exchange clock every world
  shares is unchanged afterwards. Before, one world's trip slowed every world for the rest of the run. A repeat trip
  within the watchdog's window, or a period that stays escalated, still takes the lasting remedy.
  `sim.health()["refined_periods"]` counts each world's re-run periods since the build (None without coupling; joined
  over `processes=` workers). Engine layers: `alloy.coupling.types.WatchdogPolicy(per_period_ladder=False)` restores
  the old behaviour.
- Coupling on the GPU (MuJoCo Warp route): a robot body that cannot translate at its centre of mass, such as a link
  on a hinge through its centre of mass, now couples to cloth, liquid and soft bodies with its own mass, as on the
  CPU route. Before, MuJoCo Warp's inverse-weight fallback gave it a proxy hundreds of times too light (a 0.3 kg
  wrist read as 1.6 g).
- `alloy.Scene(liquid_accuracy="default" | "super")`: liquid particles 4 mm apart by default, 2 mm with "super"
  (about 16x the cost); kept in Scene JSON, not in saved cards; `alloy.estimate` shows both settings; the CLI and MCP
  take `--liquid-accuracy`. A build warns once, naming the fix, when a container's opening (under 32 mm at default,
  24 mm at super) or a pour stream (under 2 particles) is too narrow for the setting. The router no longer refines
  the particle size silently for a narrow container card's opening.
- `alloy.Scene(worlds=, dt=, roots=[...])`: build a scene in Python; `roots` are your own card folders, such as
  cards made by `alloy.create` or `alloy.io`. Rigid boxes, spheres, cylinders, capsules and meshes (`add_mesh`, from a
  file or from vertex/face arrays). Cloth (`add_cloth`), liquid (`add_liquid`), robots from MJCF / URDF / USD or an
  MJCF string (`add_robot`), and library objects (`add_object`). Per-object mass, density, friction and restitution,
  per-world start positions, per-world randomization seeded by `alloy.build(seed=)` (randomizing friction caps the
  dynamic coefficient at the lowest static draw), and pair friction (`set_friction`). `rigid=True` keeps an object
  rigid: a soft material can otherwise make it a soft body, which is much slower. The build warns when that
  happens. Scenes round-trip
  through JSON (`to_dict` / `from_dict`) and save to engine cards (`save`).
- `alloy.load(source)` loads and builds in one call, returning an `alloy.Sim`. `source` is a library scene, a
  scene card, a Scene JSON file or a robot file (MJCF / URDF / USD). The robot stands on a ground plane, or on
  its own floor when the MJCF has one, as MuJoCo Menagerie `scene.xml` files do.
  `alloy.Scene.load(source)` returns the same scene as an editable `Scene` instead.
- `alloy.build(scene, worlds, device, seed, profile=)` and `Scene.build(...)` return an `alloy.Sim`. It runs on the
  GPU when there is one, else on the CPU.
- `alloy.Sim`:
  - `step(n)`, `run(seconds, record=[...], targets=...)` (returns an `alloy.Trajectory`; a `(start, end)` target
    ramps over the run), and `state()` by name;
  - `set_targets()` for joint position or velocity targets. It drives MJCF `<position>` / `<velocity>` actuators,
    and `<general>` servos with an affine bias, as MuJoCo Menagerie robots use. A joint with no actuator raises
    an error that says what to add;
  - `set_controls()` writes raw MuJoCo `ctrl` per actuator name (`sim.names()["actuators"]`);
  - `reset()`, `faces()`, `particle_volume()`, `routes()`, `health()`, `check()` and `render()` (renders through
    `alloy.camera` when it is installed).
- `alloy.estimate(scene, worlds, seconds)`: predicts memory, step time, wall time and first-run compile time.
  `measure=True` times a short real run instead.
- `alloy.Camera` (from `alloy.camera`) is re-exported, and `sim.render()` returns an image of a world.
- `alloy.materials()` and `alloy.scenes()` list what the card library holds. The library includes every imported
  library (`<alloy-assets>/imports/<source>/`: YCB, Google Scanned Objects, RoboCasa, Gym-Anything, ...), so
  `Scene.add_object(name, card)` and `alloy.load` find all of their objects. An imported object's meshes are fetched
  from `$ALLOY_ASSET_STORE` the first time a scene that uses it is built.
- `Scene.add_liquid(..., shape="cylinder")`: liquid at rest in a round container, an upright cylinder of diameter
  `size[0]` (= `size[1]`) and height `size[2]`. A box of liquid in a round cup is not at rest (its corners stick out
  of the cup), collapses and jets over the rim; a cylinder stays put.
- Precise fits by default (fresh-agent round 2, T3): `add_block_with_hole(..., precise=True)` makes its scene a
  precise-fit scene: rigid bodies step at 0.25 ms or finer (`sim.dt` is unchanged; the engine runs several
  periods per `step()`) and plastics, wood and metals get a machined part's contact stiffness, 2e7 N/m. At the
  default 1-1.25 ms step the contact stiffness is capped by the step, and a 10 mm peg landing on the hole floor at
  0.8 m/s sank 1.01 mm (both T3 misses: 1.013 mm against a 1 mm limit); precise, 0.27 mm (CPU). `precise=False`
  keeps the default step (4x fewer rigid steps). `alloy.estimate` counts the extra steps.
- Scene-card pours: `alloy.load` / `Scene.load` turn a card's `emitters` into `Scene.add_pour`
  streams named `<material>_pour_<k>` (the engine does not run card emitters, so a card pour built with no liquid).
  An emitter must be a finite downward stream (velocity `[0, 0, -v]`, `stop_s`); card sinks are dropped (the
  escape box retires liquid that leaves the scene).
- Liquids beside rigid bodies: `"<liquid>.active"` and `retired_particles` read the liquid solver's
  live flags when the coupler steps it on its own model view (a scene with a free rigid body read 0 poured);
  a pour in a scene with cloth and rigid bodies (the liquid then follows one way) builds. A kinematic container
  holds liquid beside free rigid bodies, still or tilted with `sim.drive`: a tilt pour lands about the same share in
  the cup as with every other object static, and a driven object stays on schedule when the engine re-runs a control
  period. With cloth and liquid only (no rigid solver) kinematic objects are carried too: before, the liquid fell
  through a kinematic container and the cloth through a kinematic tray.
- `add_mesh(..., collision="decompose")` builds (it always failed: the engine wants a convex-parts file beside the
  mesh, and the API never wrote one). The build splits the mesh into at most 32 convex parts with CoACD, the
  importers' settings, once per mesh (cached in `$ALLOY_CACHE_DIR/hulls`, default `~/.cache/alloy`), or copies a
  `<stem>.hulls.obj` shipped beside the mesh file. CoACD and trimesh come with `pip install "alloy-engine[import]"`;
  without them the build raises `FeatureUnavailableError` naming that command and the alternatives (`"convex"`,
  `"sdf"`). A ball dropped into an L-shaped block's inner corner now falls past it (the hull filled the corner).
- Heat (on the heat v2 plug-in, `alloy.plugins.thermal`): `add_liquid(..., temperature=K)`,
  `add_pour(..., temperature=)`, `add_object(..., temperature=)` or `Scene.set_temperature(name, K,
  heat_material=)` start a liquid or particle object at a temperature and let it take heat;
  `Scene.add_heater(name, size, pos=, temperature=K | power=W)` and `Scene.set_heater("pan_floor", ...)` make a
  static box a heater; `Scene.set_heat(time_scale=, ambient=, convection=, ...)` sets the rest. `alloy.build` runs
  the plug-in inside the `Sim`, so `run`, `record`, `state` and `reset` keep working: `"<name>.temperature"` (K per particle) is in `sim.state()` and recordings,
  `sim.temperature(name)`, `sim.heat(name)` (per world mean / min / max K, melted, done, evaporated) and
  `sim.set_heater(name, temperature= | power=)` while running. Kelvin throughout; a value under 150 K is refused
  as a likely °C. The guide's pan of water gives the same numbers as the plug-in's own path (mean 307.548 K after 90 s).
  `examples/api/07_heat_water.py`.
- `Scene.set_heat(buoyancy=True)`: the thermal buoyancy force on water (NIST's water density table at 1 atm). At
  the default liquid accuracy a hot region starts to rise within 10 % of the textbook rate for its shape, but the
  flow then stalls, so it does not give convection yet. It is off by default (heat without it is unchanged, bit for
  bit), and only water has the data: `docs/known_gaps/heat-no-buoyancy.md`.
- `alloy.build(scene, plugins=[...])`: your own `alloy.plugins` plug-ins (or functions of the built scene that return
  one) run inside the `Sim`; `sim.plugins` reads them.
- Robots collide like every scene object: their shapes start making contacts 5 mm apart (an MJCF geom's own `gap`
  is kept). They had the importer's 0.1 m, so each robot shape made contacts with everything within 0.2 m and packed
  robot scenes overflowed MuJoCo Warp's contact buffer (a carton among six boxes: 53 rigid contacts, now 28).
  `add_robot(..., contact_gap=0.1)` keeps the old value; a scene card's robot entry takes `contact_gap` too.
- Free-base robots per world: a robot whose root body has a free joint (MJCF `<freejoint/>`, such as a carton with
  flaps) takes `add_robot(pos=(worlds, 3))`, `Scene.randomize(robot, "x" | "y" | "z", lo, hi)` and, while running,
  `sim.set_pose(robot, pos=, quat_wxyz=)`. Its links follow the base with their joint angles kept, and `reset`
  returns to the drawn starts. A robot welded to the world raises an error naming the fix. `sim.set_pose` also takes
  one orientation per world, and a teleport no longer reads as an ENERGY_JUMP that froze the world (the health
  check's energy baseline restarts at the new pose). `examples/api/08_robot_per_world.py`.
- `alloy.vary(scene, n, pos=, yaw_deg=, textures=, seed=)`: `n` edited copies of a scene for visual and placement
  diversity, the same for the same seed. Free objects' starts move a little (touching ones together, never into an
  overlap), and each look takes another texture of its family plus a small tint; `out=` writes the variants and a
  manifest, `contact_sheet=True` a picture of them all.
- Objects from the object library on the Hugging Face Hub work like library objects on a fresh install:
  `Scene.add_object` finds an object fetched with `alloy fetch <name>` (it said "no object card": the API's catalog
  did not read the fetch cache), accepts an object that is only in the Hub's index, and the build fetches it.
  `alloy.catalog.objects()` lists fetched objects, and an unknown name's fix suggests close names from the Hub.
- `sim.drive` of an object replaces its earlier drive. Drives used to pile up: one per control period made every step
  slower (4 s of a whisk task: 1,600 drives, 3,200 kernel launches per step). The motion is the same, since the
  newest drive always won.
- Per-world starts for everything that moves: `pos=` as a `(worlds, 3)` array (and `Scene.randomize(name, "x" | "y"
  | "z")`) now work for kinematic objects and kinematic fixtures (each part follows), cloth and liquids (their
  particles shift per world), besides free bodies and free-base robots. Static things (static shapes and fixtures,
  static library objects, a pour's nozzle) are refused when added, with the fix `kinematic=True`. Before, they
  failed at build with three different errors (a raw AttributeError among them).
- `add_mesh` reads STL files (binary or ASCII; vertices merged): the build used to refuse the `.stl` that `add_mesh`
  had accepted.
- `add_cloth(..., self_contact=True)`: the sheet collides with itself, so a fold rests as two layers one thickness
  apart. It is on by default; `False` steps a sheet that never folds faster (cost in `docs/guide/deformables.md`).
  Scenes saved earlier (`to_dict` / JSON) keep it off.
- A scene whose only dynamics are plug-ins builds and steps: a surface film on a static plate failed with "scene has
  no dynamic entry to step" and needed a dummy free body. It now flows exactly as it did with the dummy.
- An engine error that carries its own fix (an unsupported solver pair names its nearest supported alternative) now
  gives that fix in `err.fix` instead of the generic line for its kind.
- Fixtures (`add_open_box`, `add_open_cylinder`, `add_block_with_hole`) accept `static=True`, as every `add_*` does
  (they raised TypeError; a fixture is static unless `kinematic=True`).
- `Scene.randomize` checks the field when it is named: a wrong one (`"mass"`) is a `SceneError` that lists the valid
  fields and says what to use instead (`density`), not a `CardError` at build.
- `alloy.build(scene, device="cpu", processes=N | "auto")` splits the worlds over worker processes (Warp's CPU
  kernels use one core each) and returns an `alloy.Sim`-like object with the same calls; `sim.close()` stops the
  workers. A world steps bit for bit as it would in one process while every process keeps the same clock; adaptive
  steps are chosen per process, and the sim warns when the clocks part. 64 worlds on 16 cores, 16 processes against
  1: liquid 14.4x, cloth 12.9x, paste 20x, rigid 6.5x faster. On the CPU's MuJoCo C route, each world's contact
  softness now follows its own randomized density (it took the first world's), so split runs stay bit for bit with
  randomized density too.
- `add_liquid(..., material="mayonnaise")` (any paste) no longer crashes `alloy.build`: the build's liquid-volume
  report skipped nothing and asked a paste (MPM) for its SPH particle size.
- `Scene.add_object(..., anchor=)`: a library object's own geometry goes to `pos`, not its file origin. Scanned
  objects (YCB, Google Scanned Objects, RoboCasa, the Hub) have origins 19-85 mm off the mesh, so objects placed by
  origin landed inside walls and tables. Default `"auto"`: an object from an imported library stands on `pos` (its
  bounding box's bottom centre), Alloy's own and your own cards keep their authored frame (a cup's base), so existing
  scenes are unchanged; `"center"`, `"base"` and `"origin"` choose explicitly. Per-world positions and randomized
  x / y / z ranges follow the anchor. Scenes saved before keep their poses. What you set is what you read: for an
  object placed by its anchor, `"<name>.pos"` (`sim.state()`, `run(record=)`, trajectories) is that anchor point in
  the world, turning with the object, `sim.set_pose(name, pos=)` puts the anchor there (and a turn alone keeps it
  where it is), `sim.drive`'s default pivot is it, and `"<name>.body_pos"` is the body frame's origin. Primitives,
  Alloy's own and your own cards and `anchor="origin"` read as before.
- `alloy.estimate`'s memory for MPM particles (sand, pastes) uses the measured model the build refuses on: about 6 kB
  per particle, 32 kB beside a moving body. It took 400 B (right for liquid and cloth), 15-80x low for MPM on a GPU.
  `counts` has `mpm_particles`.
- `traj.meta["mpm_grid"]`: the grid each MPM solver used in the run (sparse or dense, its cell budget, and whether a
  dense grid fell back to sparse); `[]` without MPM.
- Static `add_open_box`, `add_open_cylinder` and `add_block_with_hole` walls reach down through the floor, so dry sand
  (MPM) no longer leaks out through the seam at a wall's foot. Kinematic fixtures keep walls that stand on the floor.
- API warnings (frozen worlds, soft-body routes, liquid volume) print to stderr as `alloy: ...` when no log handler
  shows them; Alloy's package logger has a NullHandler, which silenced them in most scenes.
- `Scene.add_pour(name, volume, pos, rate=, speed=)` streams liquid from a nozzle (an SPH emitter): the way to fill
  a container. `add_liquid` is for liquid that starts at rest, such as liquid already in a container. A block of
  liquid dropped from above splashes several percent over the rim, because a flat liquid face hitting a flat floor
  makes real pressure jets; the same 200 ml poured stays 100 % in the cup (L4). The state and trajectories have
  `"<liquid>.active"` per particle: False until a pour releases it, and False again once it is retired (it left the
  scene and is frozen where it left). `sim.health()` and `sim.check()` report `retired_particles` per world, and
  check() warns for each world that has any. The energy a pour brings in, minus what retired particles take out, is
  booked as external work after every step (the SPH solver's `export_flux_energy`), so the health monitor's energy
  check stays on in scenes with a pour.
- `examples/api/06_tilt_pour.py`: pouring by tilting a held (kinematic) jug about its lip. The stream lands
  about v·sqrt(2 dz / g) beyond the lip, so the cup goes a few cm past it: 95 % of the water lands in a cup 4 cm
  past the lip, about a third in the same cup under the lip (CPU).
- Kinematic objects: `kinematic=True` on `add_box` / `add_sphere` / ... and on the fixtures. `sim.drive(name,
  velocity=, angular_velocity=, pivot=, duration=)` moves one with a constant twist about a pivot, and a fixture's
  parts move together; for example, tilt a cup about its lip to pour. Its pose and velocity are written every physics
  step, so contacts and liquids feel the moving walls. `sim.set_pose` teleports a free or kinematic body and removes
  its drive, which would write over the pose (a one-world teleport under a running drive raises instead),
  `sim.stop_drives()` stops every drive, and `sim.settle(seconds)` runs undisturbed and makes the result the state
  `reset()` returns to.
- Static fixtures built from boxes: `Scene.add_open_box` (a bin or square cup), `Scene.add_open_cylinder` (a round
  cup) and `Scene.add_block_with_hole` (a socket). MuJoCo collides each box exactly, so a 10 mm peg drops into a
  10.2 mm hole. `alloy.api.meshes` gives the same shapes as watertight meshes for `add_mesh`.
- `alloy.build(..., contacts="auto")` uses Newton's contact pipeline when a mesh has collision "sdf" or "mesh".
  MuJoCo would collide such a mesh as its convex hull.
- Error standard: every public error (`alloy.SceneError`, `BuildError`, `SimulationError`, `UnknownNameError`,
  `FeatureUnavailableError`) carries `message`, `fix` and `field`, and `to_dict()` gives it as JSON. Engine errors
  raised while building or stepping are wrapped with a fix, and the engine error stays as the cause. Each health
  flag comes with what it means and what to change.
- Lazy imports: `import alloy` loads no NumPy, Warp or Newton. Engine set-up that needs Newton runs when Newton is
  first imported.
- The project description (AGENTS.md, llms.txt, the README) names the solver each material steps in: MuJoCo Warp for
  rigid bodies and robots (MuJoCo C on the CPU), Newton's VBD for cloth and soft bodies, Newton's MPM for sand and
  pastes, Alloy's DFSPH for liquids. Sub-millimetre contact is stated for declared precision fits (pegs, threads),
  and the world counts by kind of scene: thousands for free rigid bodies, hundreds for a robot arm, tens to hundreds
  with cloth or liquid, with the measured rows in `docs/guide/worlds.md`.
- `docs/guide/worlds.md`, "Robot scenes: what they cost": the shipped robots' rates on one L4 at 1.25 ms, physics
  only (for example a Panda with a cube 1,168x real time over 2,048 worlds, a G1 88x over 512), and Alloy against
  plain MuJoCo Warp on 17 robot tasks: 0.30x its rate (geometric mean; 0.11-0.76), mostly because Alloy takes 4
  solver steps per 5 ms where the tasks' preset takes 2, with what is not like for like. Two opt-ins buy speed:
  `contact_capacity=64` 1.79x with no measured change in quality, plus `scene.rigid_step = 0.0025` 3.33x with
  deeper contacts and, in pick-and-place, glitches in 0.47 % of episodes (none at 1.25 ms).
- Bundled engines: Newton release-1.8 (`db915e1c`, 2026-11-09; version 1.8.0rc1, with the Steel solver's 1.8
  backports) plus 12 commits and 14 patches of ours, and MuJoCo Warp v3.17.0 (`f34c6dd5`, files unchanged), on
  MuJoCo 3.17.0 and Warp 1.18.0. Before: Newton `ac7f7830` plus 13 commits, MuJoCo Warp `86e2fae0`, MuJoCo 3.16.0.
  `alloy/_vendor/VENDORED.md` lists each commit and patch with its reason and its test, and what moved between the
  pins; `alloy._vendor.VERSIONS` names the versions. What you see: every library scene's `check()` verdict is the one
  it had. On the CPU, cloth and poured liquid keep their state bit for bit, and robot scenes move at the solver's
  convergence level (0 to 0.015 mm: MuJoCo 3.17 scales its stop test by the current inertia). A mesh hull that
  rested 41.6 mm deep on MuJoCo Warp 3.16 rests at 0.05 mm. A seated peg pushed sideways leans 2.3-5.5 % past its
  geometric limit (2.5-4.5 % before), with wall penetration unchanged at 13-20 µm. On a GPU the cloth drape is bit
  for bit the previous pin's (VBD fast math stays off by default). Newton's Steel solver comes in the bundled copy
  (`from alloy._vendor.newton.solvers import SolverSteel`), but no Alloy route uses it: robots and rigid bodies step
  on MuJoCo as before (`sim.routes()`), and stepped directly it lets a robot arm's contacts through far more often
  than the copy before (`docs/known_gaps/rigid-steel-pile-backend.md`).
- Alloy and a `newton` / `mujoco_warp` you installed yourself now run in one process, imported in either order:
  Alloy's bundled copies compile their Warp kernels under names of their own (`alloy_vendor__` in Warp errors, the
  profiler and the kernel cache), so neither copy runs the other's kernels. Objects from the two still do not mix.
- `alloy.export`: frames of up to 8 picked worlds for a live viewer, as numpy arrays: body poses, cloth and soft-body
  vertices, a fixed subset of liquid and MPM particles, rod centre lines, and shapes and faces when the topology
  changes. `FrameExporter(sim, worlds)`: `capture()` between steps never waits, `poll()` from any thread makes no GPU
  call, and stepping with it on equals it off, bit for bit. `frame_export(sim, worlds)` is the one-shot form
  (`docs/guide/frame-export.md`).
- `alloy.il` (opt-in, outside `alloy.api`): `Recorder` records a `VectorEnv`'s episodes to HDF5 in the robomimic
  layout (`HDF5Writer`, needs h5py) or a LeRobotDataset v3.0 (`LeRobotWriter`, needs lerobot); `replay`, `evaluate`,
  and keyboard / SpaceMouse teleoperation of action columns. It trains no policy; joint-space teleoperation only, no
  live view, no cloth or liquid replay. `docs/guide/imitation.md`.
- `alloy.il.generate(env, policy, writers, episodes)` saves scripted demonstrations from every world of a
  `VectorEnv`; `PhaseScript` writes a demonstrator as phases, `demonstrator(env)` gives the shipped one (GripperLift,
  GripperLiftCamera, PickPlace). No trajectory augmentation or motion planning. `docs/guide/imitation.md`.
- `alloy.sysid` (opt-in, outside `alloy.api`): `python -m alloy.sysid fit --robot <name or file> --data <recording>
  --out my_robot.sysid.json` fits a robot's servo stiffness, kd + damping, dry friction and action delay per joint to
  a recording (CSV, HDF5, LeRobot dataset) by replaying it in parallel worlds; `excite`, `report`, `demo`;
  `scene.add_robot(..., sysid=)` applies a file; without `sysid=` nothing is imported and a scene steps as before.
  Its accuracy numbers are on synthetic recordings made with the same model (hidden parameters), not on a real robot;
  link masses, backlash and contact are not fitted. `docs/guide/sysid.md`.
