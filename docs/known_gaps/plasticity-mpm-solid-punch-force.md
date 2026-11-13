# A yielding solid on MPM: a punch's force converges for soft solids, and for metals only with mass scaling

Status: open. The route is opt-in (a `plastic.model: j2` solid whose object card overrides its route to `paste_mpm`).
Uniform compression yields at the card's yield stress there. Since 2026-11-10 the route's entry solves each step to
tolerance 1e-5 with up to 1,000 sweeps and runs on the dense grid on any device; a punch's force then converges on
soft solids, but it is not verified against the indentation laws. **Metals are supported on this route only with
quasi-static mass scaling** (`scene.set_quasi_static_mass_scale`, opt-in, for a block that rests under a tool): without
it their stress solve does not converge, so a tool's force reads a few % of the truth, and the build warns.

## What

- **Stiff solids read a tool's force low.** Each step Newton's implicit MPM solves the stress with projected
  Gauss-Seidel and drops what it has not converged when it stops (`carry_residual_strain="none"`). The sweeps a step
  needs grow with the square of the wave speed: in the rows below they follow roughly 0.2 (c dt / h)^2, with c =
  sqrt(E / density), dt the MPM step and h the particle spacing (half the voxel). That is a rough fit to the rows that
  converge (E 3-300 MPa, one voxel and step; the coefficient runs 0.05-0.2 there), not a derived law, and nothing
  stiffer converged, so the numbers for metals extrapolate it by 3-4 decades. At 3 mm voxels and the 1.25 ms step that
  is about 390 sweeps for E 3 MPa at 1,076 kg/m3, 3,900 for 30 MPa, and an extrapolated 1.6 million for brass (97 GPa,
  8,500 kg/m3). A card over the entry's 1,000 gets a build warning naming the number. All three shipped j2 cards the
  route takes (`brass_c360`, `steel_tinplate`, `steel_zinc_plated`) are metals, so all three warn.
- **No punch law is claimed.** On a resolved rig, converged at 1.5 mm voxels, the mean pressure by the measured
  contact radius reads 2.4 of the yield stress (the usual 2.8-3.0 is for shallow dents; the value at this rig's
  a / R is being cited), and the elastic force does not follow Hertz at 3 or 1.5 mm voxels. The force starts about
  0.2 voxel before the surfaces meet.
- A tool that is moved (`kinematic=True`, `sim.drive`) leaves a dent that follows it and stays, on soft and stiff
  cards alike. A tool that presses with a given force, or lands on the solid, sinks too deep where the force reads low.

## Number

CPU dense grid unless named; a kinematic ball pressed at 2 mm/s into a block on a rough floor; E 3 MPa, Poisson 0.3
unless named. Rows: development notes, not shipped (rig `rig3.py`, batches b1-b4, g1-g4).

Converged share of the force at 1.0 mm (5 mm ball pressed to 1.2 mm, the force read as it passes 1.0 mm; 30 x 30 x 15
mm block, 3 mm voxels, 4,000 particles), by sweeps per step; the yield stress is E / 312 (brass's ratio) except the
first row (50 kPa):

| E (density 1,076 kg/m3 unless named) | sweeps needed (estimate) | 250 | 1,000 | 4,000 | 16,000 |
|---|---|---|---|---|---|
| 3 MPa, yield 50 kPa | 390 | 0.84 | 0.99 (the route since 2026-11-10) | 1.00 | |
| 30 MPa | 3,900 | 0.63 | 0.94 | 1.02 | |
| 300 MPa | 39,000 | 0.07 | 0.28 | 0.77 | 0.98 |
| 3 GPa | 390,000 (extrapolated) | | 0.03 | 0.11 | 0.43 |
| brass, 97 GPa, 8,500 kg/m3 | 1.6 million (extrapolated) | 0.002 | 0.007 | 0.03 | 0.11 |

- Converged reference: E 3 MPa at 4,000 sweeps, 3.90 N; the stiffer rows scale it with the yield stress
  (p_m / yield 3.02 at E 3 MPa, yield 9.6 kPa, which converged in about 100 sweeps).
- A shorter MPM step also converges (0.99-1.01 at 0.31-0.08 ms), at 2-7x the wall time of 1,000 sweeps.
  Pressing at half the speed changes nothing (within 2 %). Newton's
  `carry_residual_strain="unconverged"`, a Krylov pass before Gauss-Seidel, the batched or reordered sweeps, and
  other warm starts give no gain; without a warm start the force is 1 % of the converged one.
- The route's change, one L4, the same rig, identical worlds: 3.31 N before (tolerance 1e-4, 250 sweeps; 1, 64 and
  1,024 worlds), 3.86 N after (64, 256 and 1,024 worlds; both at 1.0 mm), the same in every world (spread under 3e-7);
  1.4-1.5x the wall time (1,024 worlds: 2,666 s against 3,808 s for 1.2 s simulated). Under CUDA graphs the entry had
  the sparse grid, where the force rose from 2 mm above the surface and read 2.56 N at 1.0 mm; the dense grid reads
  3.864 N on the L4 and on the CPU alike. Running without graphs cost nothing for this one-world scene (52 s eager on
  the dense grid against 90-100 s with graphs on the sparse grid, 1.2 s simulated, single runs); graphs against eager
  at many worlds is not measured.

Resolved rig: 25 mm ball, frictionless, block 72 x 72 x 36 mm on node planes, pressed at 2 mm/s; soft j2 card (E
3 MPa, yield 10 kPa, E* a / (Y R) about 115-160 at a / R 0.35-0.49). Readings, not laws:

| | 3 mm voxels (55,000 particles) | 1.5 mm voxels (442,368; 4,000 sweeps, converged) |
|---|---|---|
| mean pressure / yield at 1.5-2.5 mm, by the measured contact radius (grid nodes; top-layer particles) | 2.11-2.24 (particles unresolved) | 2.43-2.46; 2.36-2.44 |
| the same by the nominal radius sqrt(2 R depth) | 2.85-3.39 | 2.75-2.89 |
| from the slope (least-squares line of force on depth, 1.0-2.5 mm; p_m = slope / (2 pi R)) | 2.04 | 2.55 |
| where the force starts, against the surfaces meeting | 0.66 mm early (0.22 voxel) | 0.28 mm early (0.19 voxel) |
| elastic force against depth, 1-2.5 mm (Hertz: d^1.5) | about d^0.52 | about d^0.84 |

At 1.5 mm voxels the two measured radii agree within 2 % and exceed sqrt(2 R depth) by 8-10 % (pile-up). Brass
x16,000 reads 2.2-2.4 by the measured radius at 1.5 mm (1,000 sweeps), within 2-5 % of the soft card. The elastic
force at 2 mm reads 0.57 of the 3 mm row's: it is not converged with the voxel, and an earlier "0.97 of Hertz" at
3 mm was two curves crossing, not a match. A 0.75 mm row and the cited mean pressure at this a / R decide what is
claimed.

## Why

- Gauss-Seidel moves stress one cell per sweep and damps a smooth error slowly, and an implicit step of a stiff solid
  leaves little inertia to steady it: the sweeps needed go as (c dt / h)^2. What a step does not converge it drops.
- Newton's rheology solve adds a fixed 1e-6, in SI units, to every local Delassus block. A soft card's own block is
  about 6e-4, so it does not matter there; a metal under quasi-static mass scaling has blocks of about 1e-7 to 1e-9
  (x1,000 to x64,000; read at x16,000: 1.3e-9 to 5.5e-9), where it makes every Gauss-Seidel step about 12-770x too
  short: mass-scaled brass stalled at 0.35-0.48 of its force at 1,000 sweeps and 0.73-0.83 at 4,000. Since vendored
  patch 0015 the route's entry runs at 1e-12 (a soft card's punch force moves by 7e-5 on the small rig above); unscaled
  brass still reads 0.007, so this alone does not make metals work.
- Not established (a hypothesis): the slope and the force disagree because the grid's contact surface is not the
  material's. The force starts about 0.2 voxel before the surfaces meet at 3 and at 1.5 mm voxels, and halving the
  voxel moves the slope reading from 2.04 to 2.55, toward the measured-radius reading. (An earlier reading at 3 mm,
  "particles stand 0.9 mm off the ball", came from the coarse top-layer sampling; at 1.5 mm the particle radius
  exceeds the nominal one.)

## Workaround

- Soft solids (E / density under about 7,000 m2/s2 at 3 mm voxels: E under about 8 MPa at 1,000 kg/m3) converge with
  the route's defaults. For a force read, keep the block's faces on node planes (sizes that are whole multiples of the
  voxel) and expect the contact to start a third of a voxel early.
- Metals (a block that rests on the ground or a static box under a tool):
  `scene.set_quasi_static_mass_scale("block", tool_speed=0.002)` multiplies its density by s with its weight kept true.
  `scale="auto"` takes the largest s whose inertial share v sqrt(E rho s) / yield stays within 1 % at that tool speed
  (brass at 2 mm/s: x2,913); a given s over 1 % warns and over 5 % is refused, and so is a block that does not rest on
  a support. Brass with "auto" pressed at 2 mm/s reads 0.991 of its converged force at 1 mm on the small rig (0.007
  unscaled); on 1,024 worlds on one L4 (one run), 0.989 with "auto" against 0.007 unscaled, every world the same
  force and none flagged. Quasi-static only: the block is s times heavier for anything that moves it (a drop, a
  throw, a fast hit).
  A block that loses its support falls s times slowly (g / s: brass x29 pushed off a table's edge fell 32 mm in the
  0.46 s after it began to fall, where g / s gives 36 mm and g 1.0 m), and health flags it QUASI_STATIC while it is
  pushed and all the way down: use the setting only for a block that stays on its support.
  The declared `tool_speed` is your statement of how fast anything presses on the block. At build, "auto" stays
  within 1 %, a given s warns over 1 % and is refused over 5 %. At run time a world reads DEGRADED with QUASI_STATIC
  while the block as a whole moves faster than half `tool_speed` (its mass-weighted mean velocity: a correct press
  reads at most 0.14x), or a body touching it moves faster than 12x `tool_speed`. Not flagged, a tool up to 12x the
  declared speed reads the force within about 9 % (medians +1 to +3 %; the speed table below). 12x is a conservative
  choice: 30x measured within 10 % too, and 100x is the first speed clearly off (up to 38 % low where it first hits);
  a flag you can avoid by declaring the true speed costs little, and the 9 % scatter of the declared-speed press itself
  on 1.5 mm voxels leaves no room to set the bound tighter. The inertial-share formula gives 7 and 23 % at 3x and 10x,
  against those medians: the build bounds are conservative too.
  A block pushed along is flagged. Carried at its declared speed, the push read 4.4x the unscaled block's force at
  x2,913 ("auto" for 2 mm/s) and +5.0 % at x29 ("auto" for 20 mm/s), and the block slid on 0.1-0.6 mm after the
  pusher stopped (unscaled: under 0.01 mm). Those are bounds, not exact errors: unscaled brass is itself no clean
  reference (its stress solve does not converge, so it acts soft).
  Without the setting, move the tool and read the dent's shape, not the force. Metal sheets and cans dent on another
  route (`add_cloth(..., material="aluminium_6061_t6")`, `docs/guide/deformables.md`).

Speed table: the force of a tool faster than declared against the declared-speed press at the same depth (25 mm ball
rig, brass x16,000, one L4; each period of the faster tool against the median of the declared press's 5 periods
within 0.02 mm of its depth; development notes, not shipped). Part of each spread is noise: one
period of the declared-speed press scatters up to 9 % from its neighbours on 1.5 mm voxels (2 % on 3 mm). A CPU
repeat of the 10x row read +6.1 % at 2 mm.

| tool (flagged?) | voxel | 0.2-0.5 mm deep (first contact) | 0.5-1 mm | 1-2 mm |
|---|---|---|---|---|
| 3x (no) | 3 mm | +0.8 to +1.5 % | -0.6 to +2.1 % | -0.4 to +4.4 % |
| 3x (no) | 1.5 mm | -1.6 to +1.4 % | -3.8 to +6.4 % | -1.8 to +7.8 % |
| 10x (no) | 3 mm | +1.8 to +2.9 % | -0.8 to +4.6 % | -0.2 to +7.3 % |
| 10x (no) | 1.5 mm | -8.7 to -2.7 % | +0.5 to +6.9 % | +0.1 to +9.2 % |
| 30x (yes) | 3 mm | +1.8 % (0.4 mm) | -6.2 to +5.6 % | -3.8 to +7.0 % |
| 30x (yes) | 1.5 mm | -9.6 % (0.4 mm) | -4.2 to 0.0 % | +0.1 to +3.8 % |
| 100x (yes) | 3 mm | +4.9 to +7.2 % | -11.5 to +11.7 % | -2.5 to +6.3 % |
| 100x (yes) | 1.5 mm | -38 % (0.2 mm), -20 % (0.4 mm) | -7.5 to +0.2 % | -6.9 to +4.7 % |

## Planned fix

After 0.1.1 (dents slice 2b): the quasi-static mass scaling above is done (on the resolved rig, brass x16,000
follows the soft solid of the same E / yield within 2.4 % at 0.5-2.5 mm; development
notes, not shipped). Next (2026-11-11): a 0.75 mm row and a Richardson
extrapolation of the mean pressure and the elastic exponent over 3, 1.5 and 0.75 mm, against the mean pressure the
literature gives at this a / R (not 2.6-3.2 by default). If they agree, the default route for small yield strains
follows with that law; if not, the gap stays as stated here.
