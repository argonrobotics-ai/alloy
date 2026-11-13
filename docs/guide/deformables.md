# Cloth and liquid

Cloth and liquid are objects like any other: add them to the same `Scene` as rigid parts and robots, and Alloy
routes each to its solver (cloth to a VBD cloth solver, liquid to an SPH solver) and couples them with the rest.
`sim.routes()` shows where every object went and why.

## Cloth

`scene.add_cloth(name, size=(x, y), pos=..., resolution=21, mass=None, thickness=0.002, material="cotton_woven",
self_contact=True)`
- a flat sheet in its local x-y plane, centred at `pos`; orient it with `rpy=` or a quaternion keyword;
- `resolution`: vertices along the longer side (or `(nx, ny)`); finer sheets fold more smoothly and cost more;
- `mass`: total kg; without it the mass comes from the material and thickness;
- `thickness` is the contact thickness (particles of radius `thickness / 2`); sheets thinner than the 2 mm default
  need a smaller `Scene(dt=...)`. A fabric bends as its material card says (`elastic.bending`: the KES bending
  rigidity where measured, else derived from the card's own fit) at any thickness, so a thicker sheet that a
  gripper can hold still folds like cloth. Its mass does grow with the thickness (density x thickness): pass `mass`
  for the fabric's real weight.
- `material`: a fabric (`alloy.materials()` lists the cards), or a sheet that is not a fabric:
  - paper and plastic film (`paper_80gsm`, `paper_board`, `paper_kraft`, `film_ldpe`, `film_hdpe`, `film_lldpe`)
    build as a film sheet (`sim.routes()`: `cloth_film`);
  - a metal with a yield law makes a sheet that dents (Metal sheets that dent, below);
  - a paste (mayonnaise, ketchup, ...) is refused at `add_cloth`, naming `add_box` for a solid shape of it;
  - a stiff plastic or board (`pet`, `nylon_pa6`, `pp_homopolymer`, `cardboard_corrugated`: a bending length over 5 cm)
    stays on the cloth route but sags several times too deep (Accuracy, below). Give a part that should barely bend
    a rigid box instead, `add_box(..., rigid=True)` (without `rigid=True` a thin box of these goes to the cloth route
    too), at least 1.25 mm thick and with the sheet's mass (`mass=`)
    ([cloth-stiff-sheet-sag.md](../known_gaps/cloth-stiff-sheet-sag.md));
  - a fabric card with warp and weft stretch (`elastic.cloth`, such as `cotton_woven_aniso`) on an `add_cloth`
    sheet routes to `cloth_vbd` with an orthotropic membrane (warp and weft stretch, shear and per-direction
    bending from the card; warp along the sheet's x). A sheet from a mesh with such a card routes to
    `cloth_style3d` when nothing else in the scene moves, where a stiff fabric barely drapes
    ([cloth-style3d-stretch-lock.md](../known_gaps/cloth-style3d-stretch-lock.md)); the object card's route
    override `route: {route: cloth_vbd, reason: ...}` puts it on the membrane too. KES-built twill and wool drape
    within 0.05 of a Cusick tester on the grid and on an irregular mesh (0.035 on the grid; up to 0.046 on one
    irregular mesh, in a second run of it); a heavy 2.4 mm plain weave hangs 0.16-0.19 low. Against the old
    Style3D route such a sheet drapes differently (up to 0.13 m on a sheet alone) and steps 2-2.5x slower. DC is
    converged at
    the shipped iterations and the yarns stretch under 0.6 %, but on the lightest fabric the local shear is not: its
    99th-percentile shear angle is 9.5° at 60 iterations and 2.2° at 160, and health does not flag shear. Without the
    override a mesh sheet with such a card beside other moving things is refused at build
    ([cloth-woven-drape-isotropic.md](../known_gaps/cloth-woven-drape-isotropic.md)).

  Very thin films need care at sharp edges. A 0.05 mm LDPE film draped over a box's edge went INVALID (a contact
  0.5 mm deep, more than half its thickness), where 0.1 mm was fine. A smaller `Scene(dt=...)` does not help, and
  the library's film objects do the same.
- There is no cloth from a mesh: `add_cloth` makes flat sheets only. To cover a shape (a tablecloth, a cover on a
  part), make a sheet a little larger than the shape, start it a few mm above it, and let it drape
  (`sim.settle(seconds)`); a rigid `add_mesh` body under it keeps its shape and no cloth vertex ends up inside it.
- `self_contact` (on by default): the sheet collides with itself, so a fold or a pile stays in layers one `thickness`
  apart. Without it a folded sheet passes through itself. It costs time per control step (measured at `dt=0.01`):
  - on an L4 GPU, 1.44x on a self-contacting drape at the default 40 iterations, and 2.1x with
    `cloth_accuracy="fast"` (most of its cost is per step, not per iteration). At 10 iterations it was 1.3-1.8x for a
    sheet that never touches itself (484-1,681 vertices, 1 or 64 worlds), and 2.2x (1 world) to 4.4x (64 worlds)
    while a 671-vertex strip piles onto itself;
  - on the CPU, about 1.6x and 3.2x.
  Pass `self_contact=False` for a sheet that never folds (a flag, a sheet that hangs or lies flat) to save that; a
  drape folds at its skirt and corners, so keep it on there. Beside a robot on the CPU (see Cost below) the cloth's
  share of each step grows by the same factor. With two or more cloth or soft objects in one scene, contact between
  them, and each one's self contact, is always on.
  Sheets that fall onto each other can still pass through each other: 12 cotton sheets of 16 x 16 vertices, 5 cm
  apart, released together over the ground (CPU) had crossed at 27 vertex columns by 1.36 s, with gaps between
  neighbours down to -10.5 mm, and 20 sheets crossed at 70-108 columns. No contact was dropped for lack of room and
  health did not read INVALID, so nothing flags it: compare neighbouring sheets' heights yourself. Sheets resting in
  a stack keep their order (gaps 1.98-2.0 mm, none crossed). No workaround is verified.
  Keep mesh edges at least one `thickness` long (`size / resolution >= thickness`): with shorter edges the sheet
  starts in contact with itself and pushes itself apart (a disc with 1.1 mm edges at 2 mm stretched 89 % in 0.5 s).
  The build warns, naming the sheet.
- Bending: a sheet that bends freely along one of its mesh axes bends about 35 % too soft, as if its rigidity were
  0.64 of its card's `elastic.bending`: a strip clamped level at one end hangs 0.57 of its free length at the tip
  where the card's rigidity on that mesh gives 0.43. So an edge that hangs square to the grid droops further than its
  fabric would. Plates supported on their edges are not affected (0.99 to 1.02 of the textbook sag)
  ([cloth-bending-free-strip-soft.md](../known_gaps/cloth-bending-free-strip-soft.md)).
- Crease memory: a cloth material with KES bending hysteresis (`elastic.bending_hysteresis`, KES 2HB; the library's
  `cotton_single_jersey` has it) holds a fold by fibre friction, as KES-FB2 measures it: its bending moment is
  B K + HB sign(dK/dt), so a released fold keeps the curvature HB / B (0.61 /cm for the jersey) instead of springing
  back flat. Other cloth cards bend elastically, as before.
- A soft fold stays folded when it is let go: a 4 mm terry sheet folded in half and released is still folded 1 s
  later (0.76 of its flat width: half, plus the fold's rounded loop). A sharp fold can spring part-way open. The
  cloth's bends are elastic and keep no crease, unlike real fabric: T-shirt sleeves (plain-weave card) folded onto
  the body end 52-67 % on it after release. Score a sharp fold while it is held, or keep holding until a later step
  presses it flat (`docs/known_gaps/cloth-sleeve-fold-springback.md`).
- Cost: on the CPU cloth runs on one core (a 20 x 20 sheet over a ball, 2 worlds: 52 s per 0.3 s simulated, 33 s
  with `self_contact=False`). The default's 40 iterations cost about 4x VBD's own 10 there (13.6 s and 8.4 s), which
  `cloth_accuracy="fast"` runs (Accuracy, below). Since 2026-11-09 a CPU step skips the self-contact work of its
  iterations while no sheet is in self-contact (bitwise the same result): a drape over a sphere steps 1.29x faster.
  Beside a robot the cost follows how often cloth and robot exchange forces:
  - the exchange step is the largest `dt / 2^j` no longer than 0.5·sqrt(m / k), where m is the mass of the lightest
    robot body that could touch the cloth and k the contact stiffness (about 2e6 N/m), and never over 2 ms. Every
    body that could touch counts, touching or not;
  - light finger pads set a fine clock: 0.05 kg pads mean 128 exchanges per 10 ms control step, 1 kg pads 32, and
    wall time grows about linearly with the count (a trial on the CPU, self contact on, at VBD's own 10
    iterations, as "fast" runs: 12 s and 2.5 s per 10 ms step, and a 2.2 s fold took 30-65 min per world). The
    default's 40 iterations cost 3.4-3.9x that: a 1 kg arm link beside a sheet went from 2.5 s to 9.8 s per step;
  - so give the pads and links near the cloth realistic, not tiny, masses. On one L4 GPU, 64 worlds of a 12 s fold
    took 47 min (self contact off, at 10 iterations as in "fast"; a drape runs 2.6x slower at the default 40).
    Plan cloth + robot work for a GPU and time a short run first.
  - a world that health freezes (INVALID) still sets that clock. One that froze on a start overlap (a sheet started
    inside a wall, a soft body flush on its support) can then make every world's step take minutes. Set
    `ALLOY_PARK_LATCHED=1` in the environment before the build when you run many worlds or long runs of such a scene
    and do not reset frozen worlds at once: a frozen world then leaves the clock until it is reset. 1,024 worlds of a
    sheet beside a free ball, one frozen at its start, stepped at 2.1 s per control step instead of over 37 min on
    one L4. Nothing changes until a world freezes; after that the other worlds step on their own clock, so their
    results differ from a run without it
    ([coupling-frozen-world-sets-clock.md](../known_gaps/coupling-frozen-world-sets-clock.md)).

- Accuracy: `alloy.Scene(cloth_accuracy="default" | "super" | "fast")` (or `scene.cloth_accuracy`), the words of
  `liquid_accuracy` plus "fast". The cloth solver (Newton's `SolverVBD`) runs a fixed number of iterations per
  step; at that solver's default of 10 a sheet that hangs or drapes is far from converged (it stretches 13 % and
  hangs too low). The default and "super" start each step from the last step's motion (VBD's adaptive initial
  guess) and accelerate the iterations (Chebyshev, rho 0.9): the default runs 40 of them, "super" 60. Particles
  touching the cloth itself (a fold's plies) are not accelerated, since without that a folded stack kept bouncing
  (`docs/known_gaps/cloth-fold-unsettled-default.md`; the plies still creep). The drape figures below predate
  that; the drape laws still pass with it. Measured on the rc1-1110 release candidate, on a Cusick-style drape (a
  30 cm disc over an 18 cm stand, 64 worlds on one L4), by the drape coefficient DC, the draped ring's shadow as a
  share of the flat ring (1 stays flat, 0 hangs straight down):
  - default: DC 0.741 (0.716 to 0.761 over the frames of the measuring window) at 0.43-0.52 % strain (1.64 simulated
    seconds per wall second over the 64 worlds). For scale, MuJoCo Warp's flex cloth with settings we tuned ran the
    same scene at 0.79 simulated seconds per wall second on 4.0.0, so the default is 2.08x as fast there, and 1.39
    on 3.17 (one run each, one L4). The engines settle at different drapes (DC 0.74 here, 0.87 in MuJoCo 4.0.0) and
    no measured drape exists for this sheet, so this compares cost, not accuracy;
  - "super": DC 0.730 at 0.30 % strain, at about 1.4x the cloth cost (1.14 simulated seconds per wall second; about
    1.5x on the CPU).
  - "fast": VBD's own 10 iterations, without the adaptive start or Chebyshev: DC 0.19 at 13 % strain, at about
    0.4x the default's cloth cost (4.37 simulated seconds per wall second; about 0.25x on the CPU). Use it only where
    the cloth's shape is not what you score (a first look, a smoke run, a robot test that only needs cloth present).
    `sim.check()` and `sim.health()` keep reporting, but 13 % is under CLOTH_STRETCH's 50 % warning, so a fast drape
    reads OK: its stretch is not flagged. On a GPU the fast drape did not repeat bit for bit: the 64 identical
    drapes ended up to 57 mm apart, and a second run differed in all 64 worlds.

  The drape moved from the earlier figures (default DC 0.728 at 0.36 %, "super" 0.732 at 0.34 %) with one change, a
  float-precision fix for very slow motions: the version just before it reproduces the earlier rows to the bit, and
  the fix alone gives these. Why it raises the strain of the default and lowers that of "super" is not traced. The
  default sizes a sheet's self-contact rows to the scene, which costs about 9 % on a converged drape;
  `ALLOY_SELF_CONTACT_ROWS=fixed` in the environment keeps the fixed rows, for speed where a sheet never folds (the
  same shape, bit for bit).

  It applies to every cloth and soft body of the scene, since they share one solver. `alloy.estimate(scene, ...)
  ["cloth"]` shows each setting's time and accuracy. The default is already close to converged on a drape; use
  "super" as the check when a hanging or draped sheet's shape is what you score (a drape, a sag, a fold). Converged
  is not yet measured: woven fabrics built from KES data drape away from a Cusick tester's measured DC by up to 0.28
  at any setting ([cloth-woven-drape-isotropic.md](../known_gaps/cloth-woven-drape-isotropic.md)).
  A stiff sheet (bending length over 5 cm: PET, nylon, polypropylene, cardboard) starts each step from the plain
  inertial guess instead, since the adaptive start pumped energy into it (the build says so in `sim.routes()`). It
  then comes to rest, but at 40 iterations it sags far too deep (`docs/known_gaps/cloth-stiff-sheet-sag.md`); a
  fabric in the same scene starts that way too, since they share one solver.

  Repeat runs on a GPU: cloth runs without Warp's ordered mode at every setting (the build logs determinism NONE and
  `sim.health()` names DET_DOWNGRADED). Where a cloth point takes two or more contacts at once (the sheet touching
  itself in a fold, a pile or a crumpled sheet, or one point touching two objects), the GPU adds them in no fixed
  order, and the last-bit difference can grow into different folds: identical worlds and repeat runs can differ at
  every setting. The default and "super" drapes above repeated bit for bit only because their converged sheet never
  touches itself; the fast drape's unconverged sheet keeps 4-22 edge pairs in contact. On the CPU every setting
  repeats bit for bit, and identical worlds match, also once a sheet touches itself: each world adds its
  self-contact forces in the same order (not checked for a cloth that touches itself in a robot's grip).

State: `"<name>.points"`, shape `(worlds, vertices, 3)`; `sim.faces(name)` gives the triangles `(F, 3)` as indices
into those points. Save both for a mesh you can look at: `traj.save("cloth.npz", faces=sim.faces(name))`.
Do not create a sheet inside or touching another object; start it a little above.

### Metal sheets that dent

`add_cloth(..., thickness=t, material=...)` with a metal that has a yield law (`plastic.model: j2` on its card:
`aluminium_6061_t6`, `steel_tinplate`, `steel_carbon`, `steel_zinc_plated`, `stainless_303`, `stainless_304`,
`brass_c360`, `copper_c110`) makes a sheet that springs back below its yield and keeps a dent past it. `sim.routes()`
says `cloth_film`, however small or thick the sheet. Unlike a fabric, `thickness` here is the sheet's real thickness
and sets how stiffly it bends (E t³ / 12).

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
for name, x in (("left", -0.02), ("right", 0.02)):              # round supports 4 cm apart, tops at z = 0.05
    scene.add_cylinder(name, 0.005, 0.07, pos=(x, 0, 0.045), rpy=(1.5708, 0, 0), static=True)
scene.add_cloth("sheet", size=(0.06, 0.06), pos=(0, 0, 0.0503), resolution=13, thickness=0.00025,
                material="aluminium_6061_t6")                 # 0.25 mm aluminium: thickness sets its bending
scene.add_sphere("finger", 0.005, pos=(0, 0, 0.056), kinematic=True)
sim = alloy.build(scene)
print(sim.routes()["sheet"]["route"])                         # cloth_film: a metal sheet that keeps a dent
sim.step(20)
rest = sim.state(world=0)["sheet.points"][:, 2].min()
sim.drive("finger", velocity=(0, 0, -0.05), duration=0.3)     # press 15 mm at mid-span
sim.step(30)
sim.drive("finger", velocity=(0, 0, 0.02), duration=0.6)      # lift away slowly
sim.step(80)
st = sim.state(world=0)
print(st["sheet.plastic_strain"].max())                       # about 0.02: it yielded
print(rest - st["sheet.points"][:, 2].min())                  # about 0.003 m: the dent it kept
```

- `"<sheet>.plastic_strain"`: shape `(worlds, vertices)`, dimensionless, in `sim.state()` and `record=`. At each
  vertex, the largest equivalent plastic strain of the hinges along it (each hinge's section has 21 layers through the
  thickness). It is 0 until a hinge yields and only grows; `sim.reset()` sets it back to 0. A plain cloth has no such
  key. A metal sheet's film runs at least 20 cloth iterations (about 1.6x a film step's cost on the CPU); the example
  above takes about 1 min there.
- At the card's real density the cloth solver still lets a stiff sheet sag far more than a real one: its iterations do
  not reach the statics. The 0.25 mm sheet above sags about 2 mm on its supports (about 2 µm in theory) and droops
  about 5 mm hung over a 20 mm bar. That sag alone can yield a sheet: steel, stainless and copper read a plastic
  strain of 0.001-0.007 before any press, and every metal more on a finer mesh (aluminium and brass read 0 on 5 mm
  cells). Read `plastic_strain` before the press and count only what the press adds.
- Each hinge yields on its own (no plate plasticity). Across the mesh's diagonals the hinge network bends up to
  1.5x too stiff below yield and carries up to 1.9x the moment past it (along its rows both are right), and a
  dent's kept depth depends on the mesh: a can wall pressed 8 mm kept
  0.68-1.07 mm on three meshes. Bend along the mesh rows where you can, and compare two meshes before you trust a
  depth.
- Rest and press it on rounded parts. Released fast (0.1 m/s) or slowly, the dented sheet stays on the example's round
  supports and settles. On two boxes' sharp edges it stays on too, but creeps sideways (about 1 mm in 1.5 s, 2.4 mm by
  3 s, still moving), and health reads CLOTH_STRETCH there (DEGRADED: the sheet 67 % past its rest size over the
  edges).
- More: [film-metal-sheets.md](../known_gaps/film-metal-sheets.md) (the sag and the hinge network, with numbers)
  and [plasticity-metal.md](../known_gaps/plasticity-metal.md) (no Bauschinger effect).

A solid block of a yielding material (a `plastic.model: j2` card with no hardening modulus, such as `brass_c360`)
can step on MPM by choice: give its object card the route override `paste_mpm`
(`scene.add_box(..., card_extra={"route": {"route": "paste_mpm", "reason": "j2 solid on MPM"}})`). Squeezed
evenly it yields at the card's yield stress, and a tool that is moved into it leaves a dent that stays. A soft
card's punch force converges at the route's settings. A metal's does not (brass reads under 1 % of its punch force,
and the build warns) unless the block rests under the tool with quasi-static mass scaling:
`scene.set_quasi_static_mass_scale("block", tool_speed=0.002)` multiplies its density by s with its weight kept true
(`scale="auto"`: the largest s whose inertial share stays within 1 % at that tool speed), and brass pressed at 2 mm/s
then reads 0.99 of its converged force. `tool_speed` is your statement of how fast anything presses on the block; the
build's limits on s (auto within 1 %, refused over 5 %) are on the safe side, and a tool faster than declared reads
the force within about 9 % at 3x and 10x (medians +1 to +3 %). Quasi-static only: the build refuses a block that does
not rest on the ground or a static box, and a world reads DEGRADED with QUASI_STATIC while the block as a whole
moves faster than half `tool_speed` (pushed along, the push read up to 4.4x the force) or a body touching it faster
than 12x (at 100x the force read 38 % low where the tool first hit).
A block that
loses its support falls s times slowly (g / s), and health flags it QUASI_STATIC while it is pushed and as it falls:
use the setting only for a block that stays on its support. The punch laws
(mean pressure by the measured contact radius) are not claimed yet:
[plasticity-mpm-solid-punch-force.md](../known_gaps/plasticity-mpm-solid-punch-force.md). Without the override
such a solid is a soft body, which can yield at rest when its yield strain is small
([plasticity-soft-tets-low-yield.md](../known_gaps/plasticity-soft-tets-low-yield.md)).

## Liquid

Pour to fill a container: `scene.add_pour(name, volume, pos=..., rate=2e-4, speed=0.5, material="water")` streams
`volume` m³ straight down from a nozzle at `pos`, at `rate` m³/s. The particles wait at the nozzle until released;
`"<name>.active"` in the state marks the poured ones. A block of liquid released in the air splashes several percent
over the rim when it lands (a flat liquid face hitting a flat floor makes real pressure jets). A stream splashes much
less: 3 % of a 6 ml pour from 12 cm into an 8 cm box, and health stays OK
([liquid-pour-over-the-rim.md](../known_gaps/liquid-pour-over-the-rim.md)).

Pouring from a held container instead: make the container kinematic (`add_open_box(..., kinematic=True)`) and tilt
it about its lip with `sim.drive(name, angular_velocity=..., pivot=lip, duration=...)` (`scenes.md`, kinematic
objects). The stream leaves the lip at about `v = sqrt(g h)` (h: liquid height above the lip) and falls `dz` to the
target in `sqrt(2 dz / g)`, so it lands about `v sqrt(2 dz / g)` beyond the lip: put the target's centre that far
past the lip, not under it, and tilt slowly. `examples/api/06_tilt_pour.py` does this end to end.

Beside free bodies too: a jug tilted this way with a free ball beside it lands about the same share in the cup as with
the ball static (94 %). On the CPU that scene steps 10-12x slower than with every other object static (10.5x and 12.3x
on two machines), so make objects that never move `static=True`. A scene with cloth and liquid but no free rigid
object or robot handles kinematic objects as well: a jug holds its water through a tilt and a cloth rides a moving
tray as each does in a scene of its own (water level and cloth height within 1 mm), about 2x slower on the CPU than
with those objects static. Beside free rigid bodies or a robot, cloth and liquid build when those weigh at least
twice the cloth and soft bodies in all; lighter ones raise `BuildError` with the mass to use
(`docs/known_gaps/liquid-cloth-light-rigid.md`). A robot that pours can also carry the container as a body of its hand
in the robot's MJCF (`robots.md`, Grasping): in a trial that poured 96-99 % into a second container in all 64 worlds,
beside free bodies.

```python
import numpy as np
import alloy

scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_open_cylinder("cup", inner_radius=0.03, depth=0.06, wall=0.004)    # about 170 ml, floor top at z = 0.004
scene.add_pour("water", 30e-6, pos=(0.0, 0.0, 0.12), rate=1e-4)              # 30 ml from 12 cm up, over 0.3 s
sim = alloy.build(scene)
sim.run(1.0, record=[])
s = sim.state(["water"], world=0)
p, active = s["water.points"], s["water.active"].astype(bool)
in_cup = active & (np.hypot(p[:, 0], p[:, 1]) < 0.03) & (p[:, 2] < 0.064)
print("poured ml:", round(active.sum() * sim.particle_volume("water") * 1e6, 1),
      "in the cup:", f"{in_cup.sum() / max(active.sum(), 1):.1%}", "ok:", sim.check()["ok"])
```

Liquid that starts at rest, already in a container or as a puddle: `scene.add_liquid(name, size=(x, y, z),
pos=..., material="water")` fills a box with SPH particles (the engine picks the particle size). Match the
container: in a round cup use `shape="cylinder"` with `size=(diameter, diameter, height)`; a box of liquid in a
round cup is not at rest (its corners stick out), collapses and jets over the rim. Contain it in a static container: `scene.add_open_box(...)` or `scene.add_open_cylinder(...)`
(see `scenes.md`).
- Cost: on the CPU liquid runs on one core at about 40,000 particle-steps per second on an AMD EPYC 7B13 and
  25,000-27,000 on a 2.8 GHz Xeon cloud VM (GCP n2-standard-16), so time is linear in the particle count (one world at
  rest: 512 particles 10-15 s per simulated second, 13,800 5-7.5 min). A free body floating in the liquid costs 10-21x
  more: the liquid is stepped again with it on every coupling pass (a floating body 21x on the Xeon; a pour beside a
  free ball 10.5x and 12.3x as long as with the ball static, on the two machines; a 12,000-particle floating cup: 77
  min per simulated second on the CPU, 4 min on one L4). For many worlds, split them over processes
  (`alloy.build(scene, device="cpu", processes="auto")`: 64 worlds of water in a box ran 14x faster on 16 cores,
  `install.md`) or use a GPU (one L4 ran 64 worlds of a 6 s pour in 3.4 min).
- Cost with cloth: cloth and liquid in one scene are cheap while they do not touch and very slow once they do. On one
  L4, one world, a cloth lying beside a cup of water (the cup static or kinematic) steps at about 11 wall seconds per
  simulated second. A 10 cm cotton sheet dropped onto 0.8 l of water steps at 3-4 s per 10 ms control step while it
  falls and at about 50 s per control step from first touch (about 80 minutes per simulated second), almost all of it
  in the cloth solver, and no setting helps: plan such a scene in tenths of a second, or use a film-route sheet
  (paper, metal foil), which ran 1.6 simulated seconds on water in 5-11 minutes on an L4. Those are the numbers of
  0.1.0 and 0.1.1. After 0.1.1 liquid that moves fast takes smaller steps, so a sheet that stirs the water costs
  more. Measured on one
  L4 in a rebuilt scene (a 10 cm sheet at resolution 10 dropped 2 cm onto 0.8 l of water in a static 13 x 13 cm box),
  a cotton sheet's steps after first touch went from 33 s to 88 s each, and a paper sheet (`paper_80gsm`) cost 2.3x
  over its first 0.3 s. A cup of water at rest beside cloth costs the same. The first run of a
  cloth + liquid scene also builds kernels for several minutes.
- Pouring onto cloth: a stream poured onto a fabric sheet with self-contact on (the default) makes the world INVALID,
  mostly within 0.1 s of landing (0.39 s on a full block top; CPU rows); give a sheet that takes a pour
  `self_contact=False` (it stayed OK in the one scene measured) or use a film sheet (`film_ldpe`). Cloth takes up no
  water (`docs/known_gaps/liquid-on-cloth.md`).
- Accuracy: one setting for all the scene's liquid, `alloy.Scene(liquid_accuracy=...)` (or `scene.liquid_accuracy`;
  CLI `--liquid-accuracy`). `alloy.estimate(scene, worlds, seconds)["liquid"]` gives both settings' particles, time,
  memory and accuracy before you run. `sim.routes()` names the setting and spacing, and `traj.meta["liquid_accuracy"]`
  records it.

  | setting | particles | cost | measured accuracy |
  |---|---|---|---|
  | `"default"` | 4 mm apart | 1x | hydrostatic wall force 1.1 %, floating draft 4.9 % (5 particles deep), dam front within Ritter's bound, pours keep 99.7 % in cups from 32 mm |
  | `"super"` | 2 mm apart | about 16x the work (8x the particles, 2x the steps). Measured: 11-20x on CPU pours; on one L4 3.5-3.8x for one world and 36-49x at 64-256 worlds (two measurements) | wall force 0.2 %, draft 4.2 %, dam front within Ritter's bound, pours keep 97.6-99.8 % in cups from 24 mm |

  - When the setting is too coarse, the build says so once, in one line naming each item, its width in particles and
    the fix. It checks:
    - the opening of an `add_open_box` (its narrower inner side), `add_open_cylinder` (inner diameter) or
      `add_block_with_hole` (the hole), and a container card's `passage_width`: under 8 particles across (32 mm at
      default) or under 24 mm at either setting. From 24 to 32 mm the fix is `liquid_accuracy="super"`; under 24 mm,
      a wider container;
    - an `add_pour` stream under 2 particles wide (8 mm at default, 4 mm at super; the default pour is 20 mm wide).
      The fix is super, or a higher `rate` or lower `speed`, which widens the stream.
    - An `add_mesh` container is not checked: give its opening the same widths yourself.
  - Containers narrower than 24 mm are unreliable at both settings: liquid at rest in them can start INVALID
    (`docs/known_gaps/liquid-narrow-containers.md`).
  - Saved cards (`scene.save()`) don't keep the setting; Scene JSON (`scene.to_json()`) does. On a scene loaded from
    cards, set `scene.liquid_accuracy = "super"` again.
- `sim.liquid_volume(name)`: the simulated volume in m³. Whole particles fill the box, so it can be a few percent
  below the box volume; enlarge the box for an exact amount. A pour (`add_pour`) holds its volume within half a
  particle.
- `sim.particle_volume(name)`: the volume one particle stands for. Liquid in a region = particles inside it times
  that volume.
- State: `"<name>.points"`, shape `(worlds, particles, 3)` (and `"<name>.active"` for a pour).
- `sim.health()["retired_particles"]`: liquid particles per world that left the scene (fell off the world).
- Other fluid materials: `[m["name"] for m in alloy.materials() if m["kind"] == "fluid"]`.

```python
import numpy as np
import alloy

scene = alloy.Scene(dt=0.005)
scene.add_ground()
scene.add_open_box("tank", inner=(0.08, 0.08, 0.08), wall=0.005)  # cavity 8 cm a side, floor top at z = 0.005
scene.add_liquid("water", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.03))
sim = alloy.build(scene)
print("simulated volume (ml):", round(sim.liquid_volume("water") * 1e6, 1))
sim.run(0.5, record=[])
p = sim.state(["water"], world=0)["water.points"]
inside = (np.abs(p[:, 0]) < 0.04) & (np.abs(p[:, 1]) < 0.04) & (p[:, 2] > 0.0)
print("particles in the tank:", int(inside.sum()), "of", len(p))
```

## Heat

Liquids and particle objects can carry a temperature: heaters, melting, boiling, cooking. See [heat.md](heat.md)
and AGENTS.md section 7.
