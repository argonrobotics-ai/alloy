# Known gap (reference limit): the seam knife-force reference holds only until the first cut

## What
- The knife-force calibration on a soft block (TOP-2) uses a pre-split seam twin as its reference.
- After the first node is cut, the twin's force is a contact artefact, not cutting resistance.
- So the calibration describes only the force before the first cut and the cut timing. It does not describe the
  force while the blade cuts through.

## Number
- After the first cut, the seam twin reads 0.65-0.95 N down to 3.5 mm. Each stuck twin carries 0.12-0.19 N of it.
- With face-aware blade contacts, the same twin reads about 0 N.
- Before the cut, virtual-node copies match the twin within 2.1 % pointwise:
  - peak 0.613 vs 0.615 N;
  - mean 0.421 vs 0.422 N;
  - cut at step 205 vs 206.

## Why
- The reference runs on stock contacts.
- Its cut twins stay stuck under the edge and compress the column below.
- The + twin is pushed onto the wrong face and rides up.

## Workaround
- Cut models are judged on the pre-cut segment and on the cut timing.
- The production cut targets use face-aware blade contacts, so they do not carry the plateau. Seams use
  `BladeContact`; copies use `VNodeTopology`.
- The copies' Simulation entry splits with the `lead` rule, so its knife meets no cutting resistance at all.

## Tried: a face-aware seam reference (2026-11-08)
Setup: the TOP-2 rig on agar (softness 1, cut_spring_ke 1000) on CPU, at the law's dt (5e-4 s) and at dt/2. The knife
force is read two ways:
- the twin's readout: normal penalty force only;
- VBD's whole contact law at the solved positions: normal, damping and Coulomb friction, with the knife's descent
  velocity set.

Results:
- **Friction is part of the force today.**
  - Stock seam twin: the whole contact law reads 0.674 N pre-cut peak against the normal-only 0.615 N (+10 %).
  - Copies under the same readout: 0.678 N, so pre-cut agreement is unchanged.
  - The seam twin's `drive` used to set no knife velocity, so its friction never saw the descent; the copies twin's
    did. Both now set it (`twin._drive_knives`). The seam twin's normal-only numbers do not move: peak 0.615 N before
    the cut, the first cut at step 206, 0.956 N after it.
- **The seam twin under `BladeContact` (C2's ramp) is not a reference:**
  - the knife's depth at the first cut is 3.0 / 2.94 mm (dt / dt/2), against 1.16 / 1.10 mm with stock contacts;
  - before that cut the knife carries 2.1 N, all face friction (the normal-only readout reads 0 N);
  - particles reach 4.0 / 7.7 m/s (worse at dt/2).

  Twins activated by bond damage ahead of the edge go onto the faces while still welded. They clamp the blade, and
  the edge stops loading their bond.
- **Copies under face contacts (work rule) are not converged:**
  - past the first cut the force peaks at 9.0 / 7.8 N and dips to -3.5 N;
  - particles reach 5.4 / 6.0 m/s;
  - 18 / 15 nodes are cut by 4.3 mm.

Measured with a development probe script (not shipped).

## Planned fix
- The face-aware reference needs face records that do not clamp the blade before the bond breaks.
- The readout and both twins' knife velocities should match VBD's contact law.
- Or a measured cut-through force (for example agar cutting data) for the `work` split rule. There is none in the
  repository: DiSECt's force CSVs are LS-DYNA output.
