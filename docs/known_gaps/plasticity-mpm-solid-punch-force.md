# A yielding solid on MPM: a punch's force converges only for soft solids; metals are too stiff for the solve

Status: open. The route is opt-in (a `plastic.model: j2` solid whose object card overrides its route to `paste_mpm`).
Uniform compression yields at the card's yield stress there. Since 2026-11-10 the route's entry solves each step to
tolerance 1e-5 with up to 1,000 sweeps and runs on the dense grid on any device; a punch's force then converges on
soft solids, but it is not verified against the indentation laws. **Metals are not supported on this route yet**:
their stress solve does not converge, so a tool's force reads a few % of the truth. The build warns for a card too
stiff for the solve.

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
- **The punch laws are not established, even converged.** On a resolved rig the force starts about a third of a
  voxel before the surfaces meet, and the curve's slope reads a lower mean pressure than the force at a given depth.
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

Resolved rig: 25 mm ball, block 72 x 72 x 36 mm on node planes, 3 mm voxels (55,000 particles; contact radius 10-11 mm,
3.5-3.7 voxels at 2 mm), converged (1,000 against 4,000 sweeps within 1 %, and density x8 within 6 %):

| | at 2.0-2.5 mm depth | from the slope (least-squares line of force on depth, 1.0-2.5 mm; p_m = slope / (2 pi R)) | textbook |
|---|---|---|---|
| plastic, yield 10 kPa (E* a / (Y R) 130): mean pressure / yield, contact radius sqrt(2 R depth) | 2.83-3.00 | 2.05-2.14 (2.3 over other ranges) | 2.8-3.0 (Tabor) |
| elastic: force at 2 mm over Hertz times the bonded-layer factor (1.20) | 0.97 | | 1 |
| where the force starts, against the surfaces meeting | | 0.75-0.96 mm early (0.3 voxel) | 0 |

The elastic force at 0.5 mm reads 32 N against Hertz's 8 N: the curve rises too early and too flat. At 6 mm voxels the
plastic and elastic rows read 1.2-2x higher and start 2.5-5 mm early.

## Why

- Gauss-Seidel moves stress one cell per sweep and damps a smooth error slowly, and an implicit step of a stiff solid
  leaves little inertia to steady it: the sweeps needed go as (c dt / h)^2. What a step does not converge it drops.
- The grid's contact surface is not the material's: the collider meets the grid about a third of a voxel before the
  particles' face, so the depth a punch reads is offset and the slope and the force disagree.

## Workaround

- Soft solids (E / density under about 7,000 m2/s2 at 3 mm voxels: E under about 8 MPa at 1,000 kg/m3) converge with
  the route's defaults. For a force read, keep the block's faces on node planes (sizes that are whole multiples of the
  voxel) and expect the contact to start a third of a voxel early.
- Metals: move the tool and read the dent's shape, not the force. Metal sheets and cans dent on another route
  (`add_cloth(..., material="aluminium_6061_t6")`, `docs/guide/deformables.md`).

## Planned fix

After the release: quasi-static mass scaling for stiff cards (density times s, chosen so the
solve converges in its sweeps, with the weight kept true and the kinetic energy checked against the tool's work), not
measured yet; the punch laws (mean pressure 2.6-3.2 of the yield stress fully plastic, Hertz within 15 %) on a rig
that reads the contact radius, and the depth under a given force; then the default route for small yield strains.
