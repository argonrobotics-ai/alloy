---
owner: Alloy maintainers
---

# A cloth particle that starts a step inside a rigid contact is thrown out at its depth / dt

## What is wrong
- A cloth particle that starts a step inside a rigid object's contact distance leaves it at about the depth the step
  pushed it out by, divided by the step. How stiff the contact is does not matter.
- Small steps make it large: robot fingers coupled to cloth step the cloth at 19.5-39 us, where 0.5 mm of depth is
  18.7 m/s.
- `ClothFold-v0`'s friction pinch is where it shows. As the closing fingers squeeze the gathered towel corner, a
  corner particle is thrown at 14-33 m/s and SPEED_FAIL marks the world INVALID (frozen), in about 40 % of the worlds
  of the pinch-and-lift repro on the 13-cell towel (before 0.1.6). Since 0.1.6 the `ClothFold-v0` towel has 19
  cells a side; the rows below marked 13-cell were not measured again.
- The health report names the fastest gripper body with that speed, not the particle. In 19 of 23 such resets every
  gripper body was under 3 m/s.

## Number
One 60 mg particle (1 mm radius), started inside a static box, no gravity. CPU, `make_vbd`'s defaults (log-barrier
contact), contact stiffness 2e6 mixed with the box's 2.5e3. Main 816eca76e, 2026-11-10; a development
probe, not shipped.

| Start depth | Step | Leaves at | Depth / step |
|---|---|---|---|
| 0.12 mm | 20 us | 3.4 m/s (5.2 with Newton's linear penalty) | 6.0 m/s |
| 0.17 mm | 39 us | 3.4 m/s | 4.4 m/s |
| 0.17 mm | 20 us, 40 iterations | 5.3 m/s | 8.5 m/s |
| 0.5 mm | 20 us | 18.7 m/s | 25 m/s |
| 0.8 mm | 20 us | 31.6 m/s | 40 m/s |

- The speed equals the depth the step resolved / dt to 0.1 %. Some depth is left at the step's end (0.13 mm of the
  0.5 mm) and is resolved in the next step.
- ClothFold, the pinch-and-lift repro, 13-cell towel (before 0.1.6) (GPU, 128 worlds, 2026-11-10; development
  notes, not shipped): 55 worlds reset, at 20.2-37.5 m/s (median 25.1).
  - In 45 of the 62 reset dumps the probe's fastest towel particle matches the reported speed to 0.1 m/s.
  - In world 12, a corner particle went from 0.6 to 13.9 m/s in one 39 us exchange while the finger's records went
    from 0 to 0.12 mm penetration at 47 N. The particle left along the finger's normal; the finger moved at 0.2 m/s.

## Why
- VBD solves a step implicitly and reads each particle's velocity off its displacement, v = (x - x_start) / dt. The
  depth a particle starts inside is undone within the step, and that displacement becomes speed.
- Contact damping acts only while a particle approaches.
- The only bound on the speed is self contact's motion truncation, 0.85 mm per detection: 44 m/s at 39 us, 87 m/s at
  19.5 us.
- Not traced: how a towel particle gets 0.5 mm or more inside a finger at a step's start. It is either the particle's
  own move or a newly listed edge or face record (full-surface contacts are on in ClothFold). The finger's record
  plane was not stale: the finger moved 2-4 um per exchange.

## Workaround
- For a sheet you mesh yourself, the fix for the turn resets below is the mesh: keep the cell (`size / resolution`)
  at or under the fabric's bending length (B / w)^(1/3), B the material's `elastic.bending` and w the sheet's weight
  per area, about 1.6 cm for `cotton_woven` at a towel's weight. A coarser mesh cannot buckle where it folds. It
  stores the compression in its triangles and can fling one particle past 20 m/s (a turn reset). On ClothFold's
  towel (bending length 1.57 cm) 19 cells (1.26 cm) cut turn resets from 7.3 % to 2.1 % of worlds against 13 cells
  (1.85 cm). Leave a margin: 16 cells (1.50 cm) still let its fold drag the towel (`docs/guide/deformables.md`).
- The exit-speed cap. Opt-in, and it stays opt-in (ruling 2026-11-14): `body_particle_exit_speed=<m/s>` in the
  cloth solver's settings (`True` for 1 m/s), or `ALLOY_EXIT_SPEED=<m/s>` in the environment
  (`alloy/solvers/vbd/exit_speed.py`).
  - What it trades: it removes energy and momentum from the capped particles only (Limits below).
  - When it helps: coarse cloth meshes squeezed between fingers. On ClothFold's 13-cell towel (before
    0.1.6) it nearly halved the frozen worlds. On a 19-cell towel it gained about half a point, measured across two trees. The finer towel
    already removes most of the cause.
- How it works (split impulse): after each step, a particle whose record started the step inside its contact
  distance keeps its solved position. The outward normal speed that undoing the start depth gave it, past the cap, is
  taken off.
  - A corner of an edge or face record loses its share of that speed.
  - It never takes more than the particle's own outward speed, so a particle resting on a pad that pulls away is not
    pulled after it.
  - A particle in several records is cut by the one that cuts most, so one started deeper in one finger's record and
    thrown out of the other finger's is capped too.
  - Tangential velocity is untouched, and so is a particle that was already leaving.
  - An impact still stops in one step.
- On the probe's particle 0.5 mm deep it leaves at 1.0 m/s instead of 18.7 m/s; on a cloth face pressed 0.3 mm into a
  sphere, at 1.04 m/s instead of over 5 m/s (`tests/solvers/vbd/test_exit_speed.py`).
- Limits:
  - It removes energy and momentum from the particle only. A body that VBD integrates keeps the push it got in the
    step; the coupler's record-momentum feedback gives a robot body back only what the particles took.
  - A particle squeezed sideways out of a wedge between two records leaves along the wedge, not along either normal,
    and that speed is not capped.
- Measured on ClothFold, 13-cell towel (before 0.1.6) unless a row says 19 (one L4 per run, main b5121ce81 with
  `ALLOY_EXIT_SPEED=1.0`; 2026-11-11):
  - The pinch-and-lift repro, 128 worlds: no world reset, against 31 with the cap off.
  - The task, 1,024 worlds a side (16 runs), against a control of 1,088 a side with the cap off (main 1db240364; the
    trees differ by the cap and by changes ClothFold does not use). The shipped demo's frozen worlds fell from 10.8 %
    to 6.1 %, the slow-release script's from 8.9 % to 6.9 %. Resets fell from 214 (in 2 x 1,088 worlds) to 133 (in
    2 x 1,024).
  - Its fold count did not rise measurably with the demo (37.6 % against 35.9 %, 30 steps after the script's end;
    at 960 worlds a side +1.6 points, t 0.8). It rose with the slow release (44.5 % against 39.0 %).
  - The resets left come while the fingers squeeze and carry the towel in the turn. At 960 worlds a side: in the turn
    4.7 % of the worlds against 5.9 % with the cap off; at and after the release 1.7 % against 3.9 %.
  - On a 19-cell towel (1.26 cm cells, against the 13 shipped before 0.1.6, 1.85 cm) the cap does less: 5.0 % frozen
    with it (b5121ce81) and 5.5 % without it (main 50cd06a51, 1,024 worlds a side each; the two trees are 484 commits
    apart, so read this as about half a point). With the cap off the finer towel alone took the frozen worlds from
    11.9 % to 5.5 % (same tree and VMs).
- The turn's resets are mostly not a contact exit. A particle probe on the demo (13-cell towel (before 0.1.6), cap on)
  caught 25 turn resets at the 20 m/s limit:
  - The fastest particle speeds up over about 30 force exchanges (0.3-0.6 ms), not in one step. In about half of
    the cases no finger is near it.
  - Its triangles are squeezed in their own plane: 20-43 % edge compression in 19 of the 25, against a median of
    1-2 % on the crease particles of healthy worlds at the same time. That compression falls as the particle
    speeds up.
  - A folded crease alone is not the marker: every healthy world has crease hinges as flat at that time.
  - Cause: the membrane resists compression at full stiffness, and a coarse towel cannot buckle at the crease, so
    the compression is stored and then released into one particle. Bending holds about 1e-4 J per folded hinge,
    against 0.04 J for a 0.2 g particle at 20 m/s.
  - The energy balance (a second probe round, 2026-11-14) closes. 31 turn resets in 4 x 128 worlds, same settings,
    with the whole towel recorded at every exchange, measured from 85 exchanges before the 20 m/s mark (about 6 m/s)
    to the mark:
    - The membrane energy of the particle's six triangles fell by as much as the kinetic energy of the particle and
      its neighbours rose: ratio 0.8-1.3 in 21 of 31.
    - With no finger touching the particle (16 cases) the median ratio is 1.02, and 14 of the 16 close. Those six
      triangles held about 0.05 J and gave up about 0.025 J.
    - Bending changed by about 1e-5 J.
    - In 7, mostly a particle held in 4-8 finger records, the membrane gave up more than the particle gained, and
      the fingers took the rest.
    - In 3 the membrane did not account for the gain. One of them, after the release, jumped from 7.6 to 26.5 m/s
      in one exchange beside the floor, with no record started inside its contact distance, so the cap did not see
      it. Not traced.
  - A finer towel buckles instead. At 19 cells with the cap off, turn resets fell from 75 to 21 per 1,024 worlds
    (after the release, from 32 to 20). With the cap on they fell from 46 to 15, while resets after the release rose
    from 9 to 18.

## Planned fix
- Not a default cap: ruled 2026-11-14 that it stays opt-in (Workaround above).
- The turn's resets: closed with the mesh rule (Workaround; ruling 2026-11-14). A limit on the membrane's
  compressive stiffness in Newton's membrane kernel is an idea, not scheduled.
- Then trace how the particle starts a step that deep, which the cap does not fix.
