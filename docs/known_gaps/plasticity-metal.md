# Plasticity: J2 covers soft volumes, joint springs, metal sheets, wires and coil springs; bulk metal parts stay rigid

Status: open, being closed in phases by the J2 metal-sheet change. Build estimate below.

## Now covered

- **One J2 model** (`alloy/solvers/plastic/j2.py`): von Mises, linear isotropic hardening. Card fields
  `plastic.model: j2`, `plastic.yield_stress` (σ_y0), `plastic.hardening_modulus` (H). A j2 card keeps its solid
  route; a yield_stress alone still makes a paste. Metal cards carry sourced values (db81a79f0).
- **Soft volumes that yield** (VBD tets, `alloy/solvers/plastic/tets.py`): a j2 card on the soft_vbd route.
  - MuJoCo 4.0's column row: onset bracket (76, 77] mm against 76.45 analytic (RERUN.md's
    `alloy.scene.analytic_plastic:column_case`).
  - A stretched bar's plastic strain is within 0.3 % of the 1-D law.
  - Off is bitwise the stock solver. The hook costs about 1.05x a VBD step (CPU, 8,640 tets), 0.99x on an L4.
  - At a small yield strain (σ_y / E under 10 %) a resting soft body can yield spuriously, and the build warns:
    [plasticity-soft-tets-low-yield.md](plasticity-soft-tets-low-yield.md).
  - A bent tet bar keeps a bend (`analytic_plastic.bend_case`), but only to 0.75-1.26 of the section's closed-form
    residual at 4-8 elements through the depth: linear tets bend about 11 % stiff, and VBD settles bending slowly.
    Precise bending is the sheet and rod sections' job.
  - L4 smoke: 64 column worlds yield in height order, with the onset bracket (75.87, 76.51] mm.
- **Joint springs that yield** (P2, `scene.set_joint_yield`, `alloy/plugins/joint_yield`): a hinge's torsion spring
  or a slide's spring (MJCF `stiffness`, `springref`) takes a permanent set past a yield torque or force, with
  optional linear hardening. The 1-D J2 return moves MuJoCo's own per-world spring reference (MuJoCo Warp's
  `qpos_spring` rows, each C-route world's model), so no force is added.
  - J1, a torsion spring cycled past yield with hardening (forward set, reverse yield, an elastic poke): rests within
    2.2e-8 rad of the closed form. J2, a preloaded slide overloaded and released: set q_max - F_y / k within 6e-9 m.
    Both CPU routes (MuJoCo C, MuJoCo Warp).
  - Off is bitwise: no yield set, no plug-in; a yield never reached gives the stock trajectory bit for bit.
  - The hook costs about 40-60 µs per physics step on a loaded 4-core host (2-3 % of a step), on both CPU routes.
- **Metal sheets that dent** (P3, `alloy/solvers/film/sheet_j2.py`): `scene.add_cloth(..., material=<a j2 card>)` routes
  `cloth_film`, whose hinges carry the plane-strain J2 section and keep a dent; `"<sheet>.plastic_strain"` reads it. S1
  pure bending within 0.27 % of the closed form (Gardiner within 0.33 %); S2 a strip dented by a cylinder follows the
  statics and its kept curvature hinge by hinge; S3 a can wall keeps a fingertip dent (measured rows). Design and
  numbers: the sheet-plasticity development notes (not shipped). Its limits (VBD sags a stiff sheet at real density; the
  hinge network bends unlike a plate): [film-metal-sheets.md](film-metal-sheets.md).
- **Wires and coil springs that yield** (`rod_vbd`, `alloy/solvers/rod/plastic.py` on the shared J2 fibre section): a
  rod whose card has `plastic.model: j2` (e.g. `copper_c110`) keeps a bend or twist past yield, and
  `"<rod>.plastic_strain"` reads it. Laws (`tests/solvers/plastic/test_rod_laws.py`, `tests/solvers/rod/
  test_rod_plastic.py`; solver laws, 4 VBD iterations; CPU, copper, perfectly plastic):
  - a 0.2 m cantilever wire at a root moment of 1.5 M_y deflects within 0.7 % of the closed form on the rod's own
    joints and keeps a tip set within 2.8 % of it;
  - a coil spring (coil radius 10 mm, 3 turns, 32 segments per turn) compressed at a wire torque of 1.25 T_y
    compresses within 0.9 % of wire torsion theory and loses 0.58 mm of free length against the theory's 0.57 mm
    (+2 %). At 16 segments per turn it keeps 8 % less: each joint twists about its segment's axis, half a corner off
    the wire's tangent, so give a coil at least 32 segments per turn;
  - below yield both spring back fully.
  A yielding rod always steps on `rod_vbd`, which is slow beside a robot. Near the fully plastic moment it settles
  slowly and less accurately: [rod-plastic-near-full-yield.md](rod-plastic-near-full-yield.md).

## Limits of the J2 model

- **No kinematic hardening:** no Bauschinger effect. A reversed load yields at -σ_y(ε̄p), the isotropic surface, so
  cyclic bending under-predicts reverse yielding.
- **No metal bulk forming on VBD tets:** at E 200 GPa a VBD tet volume needs tiny steps. The volume laws use soft
  solids (E 1e5 Pa, as MuJoCo's column), and a forged or pressed metal block is not claimed.
- **Joint yield (P2) covers one-axis MJCF joints on the MuJoCo routes.** Hinge and slide joints with a linear
  spring only. Not yet beside cloth, liquid or soft bodies: the build refuses a coupled stepper, naming this file.
  The return runs once per physics step, after it, on the step's coordinate, as MuJoCo's own spring is explicit.
  Tendon springs, rods and cables stay elastic. MuJoCo 4.0's own MJCF spring-yield attribute is not read: Newton's
  importer drops it, so give the yield with `scene.set_joint_yield`.
- **The soft_vbd default start under-converges a tall soft solid:** Newton's inertial start settles a 70 mm column
  (E 1e5) 1.6-2x past its static sag at 20-100 iterations, and 1.04x with Chebyshev 0.9. VBD's adaptive start
  (`alloy_initial_guess="adaptive"`) settles it to 0.999; the column twin uses it. A plastic soft body on the default
  start would yield early.

## What is wrong

What Alloy's plasticity covers today:
- **Soil and grains** (implicit MPM, `grain_mpm`): Mohr-Coulomb through `plastic.friction_angle` + `plastic.cohesion`.
  Cards: soil_sand_dry, soil_clay_loam, soil_loam_moist, soil_hardpan, sand_silica_dry, flour_wheat, rice_grain.
  Laws: a vertical cut stands below the critical height, column collapse vs Lube 2004, cone index vs Luke 1992
  (the soil design note, not shipped).
- **Pastes and dough** (implicit MPM, `paste_mpm`): yield stress + Bingham / Herschel-Bulkley flow. Cards:
  dough_wheat, batter_bread_wheat, peanut_butter, mayonnaise, sauce_ketchup, yogurt_plain, the paints.
- **Paper and plastic film** (`cloth_film`): an elastic-perfectly-plastic bending hinge (crease memory: past a yield
  angle the rest angle follows; `alloy/solvers/film/hinge.py`, the dead-fold test), plus an in-plane strain cap at
  yield / E. No hardening.

What it does not cover:
- **Bulk metal bending and forming.** A thick metal part (a bar, a bracket, a rack) keeps the rigid route and never
  takes a permanent bend; J2 runs on soft volumes, joint springs and thin sheets only (Now covered). Before
  the J2 metal-sheet change none of the 11 metal cards had plastic fields; they now carry sourced J2 values.
- **Springs that yield**, apart from joint springs and j2 rods (above). MuJoCo tendon springs, rod and cable cards
  without `plastic` fields, and the clips' sprung members are linear elastic at any load. An overloaded spring of
  that kind returns to its rest length.

What a user sees if they try:
- A metal bar or sheet bent by a robot or a press does not deform (`rigid` route). Nothing warns.
- A metal card given `plastic.yield_stress` (say 250 MPa) is routed `paste_mpm` with the reason "paste: yield
  stress 2.5e+08 Pa". It becomes MPM particles: a pressure-independent yield (Newton's implicit MPM with friction 0)
  at the router's voxel, min(d/4, 5 mm). A 200 x 20 x 20 mm bar gets 4 voxels across, 5,120 particles. A
  200 x 50 x 2 mm sheet fails to build: "CapacityError: sheet: 1280000 MPM particles > budget 200000". Nothing about
  that MPM solid is validated as a metal: stiffness, yield, hardening, spring-back.
- A wire or coil spring whose card has no `plastic` fields springs back fully past its yield load.

## Number

- Before the J2 metal-sheet change: law tests for metal plasticity 0, metal cards with plastic fields 0 of 11 (now:
  Now covered).
- The probe above: CPU, main a6b698923, `alloy.build` only, no step. `steel_carbon` plus `plastic.yield_stress`
  250 MPa:
  - the 200 x 20 x 20 mm bar routes paste_mpm, 5 mm voxel, 5,120 particles;
  - the 200 x 50 x 2 mm sheet: CapacityError, 1.28 M particles;
  - plain steel_carbon, the same 2 mm sheet: rigid.
- Scale of what is missing:
  - steel yields at about 0.11 % strain (S235: 235 MPa / 210 GPa); the film route's paper caps at 1.4 %;
  - a 1 mm steel sheet's elastic curvature limit is 2 sigma_y / (E t), about 2.2 1/m (a 0.45 m radius);
  - before P3 no route resolved it: MPM needs sub-mm voxels, and the film hinge had no hardening and no metal card
    (P3's J2 sheets now resolve sheet bending; a bulk part still has no route).

## Why

The plastic models were built for what the task list asked for: food, soil, paper and film. Metal parts entered as
rigid bodies (connectors, tools, racks), where the contact matters and their deformation does not.

The pieces that exist do not combine into metal bending:
- MPM's yield is volumetric and needs many voxels across a thickness;
- the film hinge is the right shape for sheet bending (moment-curvature at hinges), but it is perfectly plastic and
  tuned to paper's yield angles;
- rods yield (j2 cards on `rod_vbd`), but a rod is a line, not a bulk part.

## Workaround

- For a permanent bend in a task, model the bent part in its bent shape (a second card), and switch it at the step
  the bend happens.
- A clip or spring on a hinge or slide joint: `scene.set_joint_yield(joint, torque= or force=)` gives it a permanent
  set. A rod or tendon that must not exceed yield: read its force (`sim.state`, or the rod's internal moment) and
  fail the task when it passes the material's yield load.
- Do not put `plastic.yield_stress` on a metal card: it turns the part into an unvalidated MPM paste.

## Planned fix

None scheduled. Build estimate, in the order it would pay off:
1. **Springs and wires that yield: done** (J2 rod joints on `rod_vbd`, 2026-11-08; the laws above: a cantilever's
   tip set, and a coil spring's free-length loss against wire torsion theory). Its limit is near full yield:
   [rod-plastic-near-full-yield.md](rod-plastic-near-full-yield.md).
2. **Sheet-metal bending: done** (P3, the J2 metal-sheet change: J2 sheets on the film route, laws S1-S3). Its own
   limits: [film-metal-sheets.md](film-metal-sheets.md).
3. **Bulk metal parts, 1-3 weeks.** Either:
   - validate Newton's implicit MPM von Mises (friction 0, yield_stress, hardening) at metal stiffness, about
     1 week; limited by resolution, needs ~8 voxels across a section;
   - or add J2 return mapping to VBD tet soft bodies, 2-3 weeks with a tensile-curve law and a bending law.
   Only if a task needs bulk forming.
