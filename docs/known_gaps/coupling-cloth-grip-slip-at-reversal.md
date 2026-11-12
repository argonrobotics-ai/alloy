# Cloth gripped by friction on a moving rigid body can slip on it right after the body reverses

## What is wrong
- Observed with sewing plies without bending on a reversing feed dog, at the default 20 VBD iterations. In the drive
  strokes right after the dog reverses, the plies move 0.35-0.65 of the dog's travel and slip on its teeth.
- The friction the contacts record is 0.15-0.19 of mu N while it happens. The dog carries 99 % of the 40 N clamp, the
  foot never lifts, and no fold reaches the teeth. A converged clamp would not slip there.
- Whether a given reversal slips at 20 iterations is close to a threshold: a one-period change in the settle time turns
  it on or off, with either body feedback.
- With the body feedback that hands the bodies the momentum the cloth took (gaps-now #10a, record momentum,
  cacec9764), the slip goes away at 100 iterations. With the previous feedback (the records' force at VBD's last
  iterate), it is still there at 100.
- With the plies' measured (KES) bending, no slip was seen on this machine, at 20 iterations with either feedback or at
  100 with the previous one. That is one machine, not a cure.

## Number
CPU, Warp route, main 201bdc94a (2026-11-09). The machine is the plain-seam rig (`alloy.tasks.sew_seam`): two 0.61 mm
muslin plies on a 10 x 11 mm feed dog (pair mu 1.0) under a 20 x 12 mm foot at 40 N, one stitch per 0.2 s cycle,
reversed for the backtacks. Measured with three development probe scripts on this rig (not shipped).
- Feed of the cycles right after the first reversal, in stitch lengths, for plies without bending
  (`SeamSpec(bending=None)`). "Momentum" is record momentum's feedback (main since cacec9764), "records" the previous
  one. The settle is in 1/60 s periods (12 is the default):

  | VBD iterations | settle | momentum | records |
  | --- | --- | --- | --- |
  | 20 | 11 | 0.58, 0.59 | 0.98 (no slip) |
  | 20 | 12 | 0.97 (no slip) | 0.51, 0.51 |
  | 20 | 13 | 0.86, 0.86 | 0.51, 0.51; 0.21 at the end reversal |
  | 50 | 12 | 0.61, 0.65 | 0.45, 0.30 |
  | 100 | 12 | 0.99-1.00 (no slip) | 0.92, 0.88, 0.90 |
  | 100 | 13 | 0.95-0.99 (no slip) | 0.59, 0.85, 0.48 |

  Every other cycle feeds 0.95-1.03. The seam check asks 0.7-1.15 in every cycle.
- With the KES bending (the default plies), records feedback, the cycles through the reversal feed 0.97-1.01 at both 20
  and 100 iterations. Over all 22 cycles at 20 iterations, the momentum feedback feeds 0.98-1.00 and the records
  feedback 0.87-1.01.
- In the slipping cycle (20 iterations, records feedback, settle 12, drive periods 3-7):
  - the dog moves 0.38-0.54 mm per period and the plies 0.18-0.28 mm, both plies alike (they slip on the teeth, not on
    each other); the next forward stroke loses 0.76 mm in one period;
  - N_dog 52 -> 42 N, N_bed 0.3-0.6 N;
  - friction use |F_t| / (mu N_dog) is 0.15-0.19 as a period mean and up to 0.55 in one exchange.
  At 50 iterations it is the same with either feedback: the plies move at a third to half the dog's speed through
  whole strokes, at a friction use of 0.12-0.18.
- The friction law: VBD's IPC friction reaches a use r at a slip speed of x * friction_epsilon, with (2 - x) x = r.
  The rig's friction_epsilon is 1e-3 m/s (`sew_ply_pinch.vbd_kwargs`), so r 0.15-0.55 allows 0.08-0.33 mm/s. The
  measured slip is 8-20 mm/s, and 45 mm/s in the worst period: 25-250 times more.
- The momentum books through the drive stroke: the records' force and the plies' momentum change differ by 7-11 N at 20
  iterations, 3-5 N at 50 and about 2 N at 100, mostly along the feed. The records under-read the dog's push (about
  3 N against the foot's measured 6.5 N sliding drag) while the stack moves steadily with it.

## Why
- Not the friction law: its smoothing allows 25-250 times less slip at the recorded forces.
- Not a load split: the bed carries under 1 % of the clamp in the slipping strokes.
- Not a fold under the foot: the sole stays at 1.10-1.23 mm, and the highest ply beside the foot reaches 2.9-3.2 mm.
- An unconverged coupled solve: the slip comes in the first strokes after the dog reverses, and it is gone at 100
  iterations when the bodies get the momentum the cloth took. With the records' force fed back, the bodies take
  impulses the cloth never did (the books above), and the slip stays at 100 iterations.
- The working guess for why the reversal and the missing bending matter: the solve's tangential iterate on the dog's
  contacts starts from the previous direction's state, and a cloth with no bending stiffness gives the iterate
  nothing that couples the particles across the teeth. Not yet traced.

## Workaround
- Keep a cloth's measured bending (the material's KES values, or the route's defaults). Do not zero it to simplify a
  scene that grips the cloth by friction.
- In a task that grips cloth through a reversal, read the cloth's own motion against the gripper's per cycle (feed,
  carry), not only at the end. If it slips at the default 20 iterations, re-run at 100 (`iterations=`): with main's
  feedback the sewing rig then fed every cycle through the reversal.

## Planned fix
- Trace the solve at the reversal: the per-iteration tangential residual and friction use of the dog's contacts in the
  first stroke after a flip, at 20 and 100 iterations. Then decide whether the cloth entry needs more iterations
  where a reversing body grips it (a contact-driven iteration count) or a better start for the tangential iterate.
  Not scheduled.
- Other cases this may also affect: robot pinches that carry cloth through a load reversal (ClothFold;
  not measured), and the 3-ply pinch (press, then lift). Its carry was 67 % when it was written. On main it is
  25-54 % at 30.5 / 30.0 / 29.5 N (52 % at the test's 30 N, 25 % at 30.5 N, under the 40 % minimum). With the records
  feedback the stack is not lifted at all: -18 to -19 % at all three loads. Readers: sewing, cloth fold.
