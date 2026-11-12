# Cloth resting under its own weight away from the world origin grips harder than its friction says

## What is wrong
- A VBD sheet that rests on a support under its own weight should slide when the slope passes its friction angle.
- On a table top at 0.75 m it holds, or creeps at a fraction of the right speed. This holds with
  `make_vbd(..., coarse_contact=True)` too, which slides it correctly at z = 0.
- The friction load of each light vertex comes out many times its weight, so friction is effectively several times
  mu for a sheet loaded only by its weight.
- Pressed contacts are not affected: a pad dragging the sheet across a table at 0.4 m or 0.75 m follows Coulomb as
  it does at z = 0.

## Number
CPU, `make_vbd` (log barrier), 4 ms, 10 iterations. A 7 x 7-vertex sheet on 8.5 mm cells: 20 mg vertices, r 1 mm.
mu_eff 0.2, a 35 deg slope made by tilting gravity, 0.2 s. The block value is 80.4 mm. Measured on
wp11-coarse-contact on 2026-11-08, with development probe scripts (not shipped).

| table top | contact law | stock VBD | coarse_contact |
|---|---|---|---|
| z = 0 | log barrier | 0.17 mm | 81.9 mm |
| z = 0 | linear penalty | 0.06 mm | 80.9 mm |
| z = 0.75 m | log barrier | 0.16 mm | 10.5 mm |
| z = 0.75 m | linear penalty | 0.04 mm | 0.035 mm |

- **Stock holds at every height.** That is a separate gap: the slope test's stock xfail in
  `tests/solvers/vbd/test_coarse_contact.py`. That law (mu 0.5, 0.4 s, block 128.7 mm) gives 128.0 mm with
  coarse_contact at z = 0 and 0.6 mm at 0.75 m (its strict xfail).
- **A plane tilted at the origin.** Its normal coordinate mixes y and z of about 3 cm. Barrier with coarse_contact:
  42.9 mm against 80.4 at mu 0.2, and 2.2 mm against 32.2 at mu 0.5.
- **Per-vertex trace** (penalty, tilted plane, 49 vertices). The normal converges to 1.9 nm by iteration 3. Then 8-14
  vertices per iteration land 40, 80 or 160 nm deep. Their contact force is up to 250x the vertex weight, and the
  sheet's summed load is 15-30x its weight.
- **The drag law at 0.4 m and 0.75 m** (C7 pad, coarse_contact): 0.973 of the pad's travel, the same as at z = 0.
- Not yet measured on library scenes.

## Why
- **Positions are float32 world coordinates.** A coordinate can only move in steps of its float32 spacing: 2 nm at
  3 cm, 30 nm at 0.4 m, 60 nm at 0.75 m.
- **The penalty's resting depth is smaller than that.** It is about weight / ke = 2e-5 x 9.81 / 1.3e5 = 1.5 nm, so a
  resting vertex lands one spacing to either side of it:
  - out of contact, the next solve pushes it tens of nm deep;
  - in contact, its force is ke x depth (8 mN = 40x the weight at 60 nm).
- **Friction follows that force.** It is mu x the current normal force, so the sheet sticks.
- **The barrier rests deeper**, but on the tilted plane it shows the same quantized depths.
- **Pressed contacts are unaffected:** a pad pushing far more than one spacing x ke resolves fine.

## Workaround
- **When a result depends on cloth sliding under its own weight:** put the support near the world origin (the table
  top at z = 0, the sheet within a few cm of it) and use `coarse_contact=True` (opt-in).
- **Otherwise,** treat friction between a resting sheet and a support well away from the origin as an upper bound.

## Planned fix
- A follow-up, not started. Candidates:
  - take each vertex's friction load from its normal momentum balance (resolution m / dt^2 x spacing, under 1e-3 of the
    weight) instead of the contact law's force;
  - give VBD positions a per-object origin.
- After step 3's regression list.
