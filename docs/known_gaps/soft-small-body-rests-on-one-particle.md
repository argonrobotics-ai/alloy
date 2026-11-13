# A small body resting on a coarse soft solid can sink into one surface particle past health's limit

## What is wrong
- A soft solid (`soft_vbd`) touches rigid bodies through one particle per mesh vertex. The particle radius r is 1/5
  of the tet mesh's shortest cell edge. A 15 x 15 x 5 cm pad has 12.5 mm cells and r = 2.5 mm, so its surface
  particles stand 7.5 mm apart. A body smaller than a few cells can come to rest on a single particle.
- Health's limit for a soft contact is r (half of 2r, the thinner part), and a deeper contact held for 0.1 s fails
  the world (PEN_FAIL). In sample-sweep's drop, a steel ball came to rest 3.36 mm into one particle, and the world
  froze 0.1 s after it settled.
- Every landing touches 3.4-3.7 mm for a moment (DEGRADED, PEN_WARN). Whether the body then rests that deep depends
  on where and how it settles, so the same drop at another `dt` or start height can pass.
- A soft solid placed flush on its support starts one radius inside it. Alone on the ground, such a pad is thrown
  3.6 cm up within 0.09 s, and nothing is flagged.

## Number
One L4, main bd9e6b412 (rc6 steps the same code). Sample-sweep's ab_drop4 `panel sil` scene without its render: a
static worktop (top at z 0), a 15 x 15 x 5 cm `silicone_40a` pad (845 particles, r 2.5 mm) and a 27 mm
`steel_carbon` ball (80 g) dropped 15 cm onto the pad's top. Scripts: development probes, not shipped.

| pad | `dt` | result |
|---|---|---|
| flush on the worktop (pos z 0.025) | 0.01 | INVALID at 0.536 s: the ball rests 3.36 mm into one particle |
| flush | 0.005 | check ok; the ball rests 0.2-0.4 mm deep |
| lifted one radius (pos z 0.0275) | 0.01 | check ok; 1.2 mm for a moment at a landing, then 0.2-0.3 mm |
| lifted one radius | 0.005 | check ok; 3.7 mm and 1.5 mm for moments at landings |

- In the frozen run the ball sits on the pad's top-centre particle. The next particles are 3.7-4.3 mm from touching
  it, so that one particle carries the ball's 0.79 N at 3.36 mm. Rolling over the same particle in the lifted run,
  the ball was 0.2 mm deep.
- The ball bounces several times before it settles (`soft-solid-bounce.md`), so the runs settle in different places.
- The flush pad alone, at `dt` 0.005: its bottom is at 5 mm after one step, 3.6 cm at 0.09 s, then it falls back.

## Why
- Contact with a soft solid is vertex-only (`alloy/scene/emitters/deformable.py`, `soft_particle_radius`): the
  surface between vertices does not push back. A body larger than the gaps cannot pass through, but it can balance
  on one vertex.
- Health measures a soft contact against r, the particle's own radius.
- Not traced: why one particle holds the ball 3.36 mm deep in the frozen run, against 0.2 mm when the ball rolls over
  it.
- A VBD solid rests one radius above what it touches, so a solid placed flush starts that far inside its support,
  and the first steps push it out.

## Workaround
- Place a soft solid one particle radius above its support (its pos z + r), not flush.
- Under a small body, use a finer tet mesh (the object card's `budget.tets` sets the cell), so several particles
  carry the body.
- At `dt=0.005` the same drop passed. Always read `sim.check()["ok"]` after a body settles on a soft solid.

## Planned fix
- Contact on a soft solid's surface faces, not only its vertices (as cloth's full-surface contacts do), or a
  surface-particle radius sized to the gaps.
- A start-overlap check for soft and cloth vertices (START_PEN today judges rigid contacts only), so a flush
  placement is reported at build.
