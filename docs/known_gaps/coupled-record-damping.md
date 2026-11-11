# Known gap: contact between a soft or cloth particle and a MuJoCo body takes its gains from the exchange clock

## What is wrong
- A particle solver (VBD: soft solids, cloth, film) feeds its contact force back to a MuJoCo-owned body through the
  lagged proxy. Newton's feedback is the contact records' force at VBD's last iterate
  (`coupling_harvest_proxy_wrenches`), which an unconverged solve does not give the particles.
- Fixed where a world is closed (2026-11-08, `alloy.coupling.record_momentum`, gaps-now #10a slice A): the feedback
  gets the particles' own momentum change over the step (less gravity and their external forces), shared over the
  contacts they touch by force; a lone body gets back exactly what they took. A world stays on Newton's feedback when
  something outside the records moves its particles: a pinned, inactive or proxy particle (cloth hung from a pin,
  cut fragments, liquid proxies in the cloth's view), or a dynamic body VBD itself integrates.
- The records' damping is still one-sided (it acts only while the particle approaches the body) and sized from the
  clock (below), not from the pair's restitution.
- Every record touching the body carries the soft material's K (2e6 N/m by default), and the exchange clock bounds by
  one record. Under a wide body that adds up to far more than the lagged exchange holds.
- So Alloy does not take these gains from the material (`alloy.contact.particle_gains`, 2026-11-08). The n records
  VBD lists against a card body of mass m share one spring and one damper, held to the exchange clock's own bounds
  for that body at the exchange interval dt (bound 0.5 = the clock's safety x omega_dt_max):
  - per record, ke = min(VBD's mix, 0.5² m / (dt² n)) and kd = 0.5 m / (dt n);
  - so the body meets a contact of damping ratio 0.5 (restitution about 0.16 for that contact alone) whose stiffness
    grows as dt shrinks.
- The card's K only caps ke; the card's restitution sets nothing. Records on static shapes keep Newton's soft mix.
  Robot links are not covered: their records still read the importer's values (not measured).
- A particle view whose shape ids are not the Model's keeps VBD's mix, with a warning. Newton's pin (d15a836c) keeps
  them global under compaction, so this does not happen today.

## Number
CPU, development probe scripts (not shipped). C3 box_on_pad: a 0.515 kg platen on a 40 x 40 x 10 mm pad of 900
particles, 7-9 mg each, analytic stretch 0.9029. At dt 1/6400 s it lists 1,457 records on 226 pad particles: 3.6e3
N/m and 1.13 N·s/m each.
- At rest the result is right: placed at its contact distance, the platen reads stretch 0.9035-0.9040 at 0.74-0.8 s.
  Its first compression peaks at 0.888 near 22 ms (a sudden load; undamped it would reach 0.806).
- A 1 mm drop rings down slowly: stretch 0.905-0.911 at 0.44-0.5 s (box ±3 µm), as with the first version's fixed
  1e4 N/m and 1 N·s/m per record.
- A 0.5 kg steel ball dropped 2 mm on an 11 x 11 cloth (3 cm vertex spacing, few records) rests 0.05 mm into its
  contact distance by 0.1 s; a 2 kg ball 0.10 mm by 0.15 s. With the fixed per-record gains the 0.5 kg ball hovered
  0.65-0.97 mm off at 0.15-0.3 s and the 2 kg ball sank 0.2-0.9 mm.
- Momentum, mid-air (CPU, 2026-11-08, development probe scripts, not shipped; law
  tests/coupling/test_record_momentum.py, which compares against a second world where the body falls free above the
  particles):
  - Newton's feedback: C3's platen at 0.1 m/s onto its free pad gains the pair +2.4 mN·s (4.5 % of its 51.5 mN·s,
    1.3x what the pad took); in the first contact exchange the platen was fed 0.587 mN·s for the pad's 0.030. A 2 kg
    ball at 0.31 m/s into an 11 x 11 cloth: +27.5 mN·s (4.4 % of 626; the cloth took 3.1).
  - With the fix: +0.004 mN·s (platen) and 0.23 mN·s (ball, mostly the cloth's own fast fall below).
  - The law's scenes (Warp CPU route, worst excess against what the particles took): Newton's feedback 214 % (ball)
    and 543 % (platen); with the fix 7.8 % and 0.7 %. On that route the 0.5 ms exchange can make and break a contact
    inside one step (no record force at its end); the correction then goes by record count.
- On main a coupled cloth in free fall fell 2 % faster than g (+0.20 m/s², 2026-11-08). It is not this feedback:
  VBD's float32 velocity update v = (x - x_prev) / dt quantizes each velocity change to ulp(z) / dt on fine steps
  (WP4's triage write-up, not shipped). It biases a momentum check by about 10 mN per 50 g at that
  height and step; the law compares against a second world on the same route.
- A 2 kg ball dropped 5 mm on that cloth lying on the ground lands at 0.34-0.36 m/s and rebounds at 0.20-0.25 m/s
  (e 0.58-0.69; with the fix about 0.23 m/s, e 0.68, resting 0.08 mm into contact). The cloth's own contact with the
  ground sets that bounce.
- At the material's 2e6 N/m per record, the Warp route's first rebound injected energy (ENERGY_JUMP latch at
  18 ms). With MuJoCo's native gains read as forces (before 2026-11-08) the pad did not load (stretch 0.998).
- Landings are therefore softer than the material's restitution says.
- Under its own weight a body sinks about 4 g dt² into the contact: 1 µm at dt 1/6400 s, 11 µm at 1/1920 s. A light
  body pushed by a large force F sinks 4 F dt² / m.

## Why
- VBD is not converged at its 10 iterations. Its penetration-free truncation also limits how far a particle may move
  per iteration. So the relative velocity in the damping term at the last iterate is not the one the particle
  ended with.
- The reaction is evaluated, not accumulated: it is not the impulse integrated over the iterations.
- The clock's interface bound reads one record's stiffness and damping from the Model, not the records' sum.

## Workaround
- For a resting load (a box on a pad, a cloth under a weight) none is needed.
- For a landing whose bounce matters, keep the lighter body rigid-on-rigid. The card's restitution is not used: the
  records against the body have damping ratio 0.5, and a cloth lying on a support bounces the body back at e 0.6-0.7
  through its own ground contact.
- For sub-millimetre fits, use rigid precision pairs instead.

## Planned fix
- Done: the momentum-consistent feedback (slice A above).
- Not planned as a default: record damping sized from the pair's restitution, and an exchange clock from the summed
  stiffness of a body's records (gaps-now #10a slice B, measured and dropped 2026-11-08). A body landing on a cloth
  lying on the ground rebounds at e 0.76 / 0.62 / 0.41 with the records' damping x0.2 / x1 / x3, against cotton's
  model restitution 0.0 (no measured reference); on a soft pad the bounce (e 0.89) is the pad's bulk, whatever the
  records do (known gap soft-solid-bounce). The summed-stiffness clock would take C3's ~265 touching records at the
  material's 2e6 N/m to dt_x ~16 us (today 156 us), about 10x the exchanges.
- Robot links get the momentum-consistent feedback above; their records still read the importer's gains.
