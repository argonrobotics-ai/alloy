---
owner: Alloy maintainers
---

# Known gap: the tearable gift bag does not finish the gift-bag demo

## What
- `CandleSpec(tearable=True)` (`python -m alloy.tasks.gift_bag_candle --tearable`) runs the gift-bag demo with the bag
  that has tear lines, `gift_bag_kraft_tear`. The verdict then judges all 11 criteria, `bag_not_torn` included.
- The demo does not finish with that bag. The run reports UNCONVERGED from the time the carton is lowered into the
  bag. As the fingers lift out the world goes INVALID, and it stays frozen there: the bag tipped over with
  part of a seam torn. The verdict reads `bag_not_torn` False and the run fails. `tearable=True` logs a warning naming
  this page.
- The same demo with the default bag (`gift_bag_kraft`, no tear lines) succeeds, judged 10 of 11 (`bag_not_torn` is
  reported as not judged, with the reason).
- What holds (`tests/tasks/test_gift_bag_tear.py`):
  - the tearable bag stands on its mat for 3 s with no seam torn;
  - a pulled rim tears it, and the torn length it reads matches the broken seams.

## Number
One world, seed 0, scene dt 10 ms, one NVIDIA L4:

| bag | wall time | result |
|---|---|---|
| `gift_bag_kraft` (10 VBD iterations) | 719 s | success, `sim.check()` ok; it notes UNCONVERGED earlier in the run, while every phase ends OK |
| `gift_bag_kraft_tear` (40, its card's `film.vbd_iterations`) | 5,757 s (about 8x) | UNCONVERGED from the lowering (4.2-5.1 s); INVALID at 7.31 s as the fingers lift out (a precision pair past its clearance: carton, bag, fingers); frozen with the bag tilted 53 deg and 25.7 mm of seam torn |

- Phases with no failure take 6.3x the default bag's wall time: 4x the VBD iterations, plus the seam kernels.

The tearable bag's own laws, on the same L4:

| law | result |
|---|---|
| standing 3 s on its mat | tilt 0.6 deg, rim 3.4 mm under its start, 0 mm torn, health OK; at the film route's own 10 iterations it tips 83 deg |
| bottom held, a 4 cm stretch of the front rim lifted 0.1 m/s | 12 mm torn at 20 mm of lift, 512 mm at 40 mm (41 seams); the broken seams' rest geometry gives 571 mm |

## Why
Not fully diagnosed. A tear line is a pair of duplicated vertices joined by a stiff zero-length spring (a seam). With
no contact, the bag converges only from 40 VBD iterations; at 10 it folds over within 0.5 s. Under the carton's and
the fingers' contacts the run stays UNCONVERGED at 40; which solve does not converge there is not known yet.

## Workaround
- Keep `tearable` off for the full demo (the default). `bag_not_torn` is then not judged, and the verdict says why.
- To judge a tear in a scene of your own, use `gift_bag_kraft_tear`. Read `sim.check()` and the UNCONVERGED flag in
  `sim.health()` before trusting it.

## Planned fix
Being looked into, with no release named: how the seams solve under contact, so that the tearable bag finishes the
demo at an iteration count it can afford.
