# Film: the strain projection carries no force

Status: open; being fixed now (2026-11-08). The projection's reaction on pinned vertices is read
out (below); it does not reach bodies yet. An opt-in limiter that carries force, `FilmConfig(limiter="stress")`, is
in (below, "The stress limiter"); the default is still the projection.

## What

SolverFilm keeps a film near-inextensible with a strain-limit projection. Its VBD membrane runs at 1e-4 of the
material's stiffness. After the VBD solve, the projection moves vertices directly, by inverse mass, and keeps no
multipliers. The tension it enforces is never a force, so no rigid body or robot feels it. That covers:
- a tape or label bond holding a film edge;
- pads pinching a film;
- a wrist force sensor above a carried bag.

Only VBD's own terms reach bodies: contact, friction, bonds and the soft membrane.

## Number

- **Bond tractions.** In a 90° peel of a 25 x 100 mm PET 80 µm strip (G = 380 J/m², 50 mm/s), bonds acting through
  the projection read 42.9 N against Kendall's 8.7 N, with a large sideways component.
- **The same peel with the in-plane stiffness inside VBD** (full membrane, gravity off): the grip reads 9.66 N
  against 9.62 N. That route fails under gravity, because Gauss-Seidel does not converge on a stiff sheet: paper sags
  725x.
- **The reaction, measured.** `SolverFilm.projection_reaction` [N, per particle, the last step] holds what the
  projection puts on each pinned vertex: per constraint, minus the momentum it gives its free vertices, shared by its
  pinned ones (`strain.vertex_reaction`; positions unchanged). On a hung strip it equals minus the projection's
  momentum change of the free film over dt within 0.5 % every step (tests/solvers/film/test_film_projection_reaction.py).
  A constraint's share is split equally among its pinned vertices, so only sums over pins are forces; a torque needs
  the split by moment balance (slice 2). With `FilmConfig.mass_ratio_cap` > 0, free-free pairs and unpinned
  triangles (the capped centroid) do not conserve momentum among free vertices, so then the free film's momentum
  change is not all from the pins (from review).
- **The limiter does not converge on a loaded strip.** A 40 x 80 mm HDPE 15 um strip hung from its top row, with
  2-50 g on its bottom row, stands 2-22x past its strain cap (p99) at 16-64 Jacobi sweeps and rings (0.3-1 m/s). So
  its reaction reads the dynamics of an unconverged stretch, not the weight (50 g: 0.47 N vs 0.49 N at 0.6 s). Unloaded,
  a strip's own weight (0.43 mN) stays under what the 1e-4 membrane holds at the cap (2.6 mN), and the projection
  never acts.
- **Pinches.** The film grip terms cancel the projection's pull on a pinched vertex, with no reaction on the pad. So
  the pads carry a strap's weight only through VBD's contact and friction on the pinched vertices. This follows from
  the construction; it has not been measured as a force.

## The stress limiter (opt-in)

`FilmConfig(limiter="stress")` (engine layers; `alloy/solvers/film/stress.py`) drops the projection. Past its cap a
triangle is an elastic membrane of the film's real stiffness, E h / (1 - nu²), inside VBD's own vertex solve, so the
tension is a force and a heavy load's own solve sees the film's stiffness. Below the cap the film is VBD's soft
membrane, as before. `SolverFilm.limiter_force` [N, per particle, the last step] is the limit's force on each
particle, pinned ones too (each pin its own force, so moments are right).

The same strip (40 x 80 mm HDPE 15 um, pinned top row), CPU, 10 VBD iterations
(`tests/solvers/film/test_film_limiter_laws.py`):

| load on the bottom row | stretch past the cap (x cap) | pins' force / weight | load's vertical speed |
|---|---|---|---|
| 2 g | 1.001 (projection 11.5) | 0.742 (0.638) | 0.009 mm/s (projection 10) |
| 50 g | 1.030 (24.4-33.6) | 0.989 (1.008) | 0.016 mm/s (83) |
| 500 g | 1.286 (16.6-18.8) | 0.998 (0.914) | 0.16 mm/s (38) |

The projection does not converge, so its numbers move from machine to machine (two CPUs shown); the stress
limiter's repeat within 1 %.

- The stretch past the cap is elastic (50 g: 0.07 % of the length; 3 % of the 3.7 % cap at the pinned row, which
  cannot narrow). The limit has no Poisson coupling, so in a one-way pull it is 27 % stiffer than the film: the
  strip stretches 0.79-0.82 of m g / (E h b) past the cap.
- **At loads near the film's own weight `limiter_force` under-reads the pins' reaction by up to a quarter**: the
  2 g strip's missing 5 mN is carried by VBD's soft membrane below the cap, which `limiter_force` leaves out (the
  law's floor for it, 0.67 of the weight, is set from that membrane's pull at the cap). From 50 g it reads 0.99 or more.
- An off-centre load: the pins' moment matches the free film's weight-less-inertia moment within 0.9 % over 0.4 s
  (0.2 % and 0.9 % on two machines; 10 % rms per step: VBD at 10 iterations). A 50 g pendulum: period within 0.03 %,
  7 % of the swing energy lost per period (the implicit step's own damping), none gained. A load taken off: the pins
  read the film's own weight within 0.1 % after 10 steps (the projection's load recoils at 270-370 mm/s).
- **No yield; an overload is not flagged.** Past the cap the film stays elastic at any load. 100 g to 1 kg hold
  steady; 5 kg (49 N, 2.2x the strip's 22 N yield force) is still stretching at 36 mm/s after 2 s, 3.8x past the cap,
  with no health flag (CLOTH_STRETCH warns only past 50 %).
- The loaded particles still move against each other across the pull at 2-20 mm/s: with the cross direction slack
  this model has next to no shear stiffness (a real film has).
- The model is soft below its cap (a real film is elastic from zero strain): a load let go at the film's rest length
  first falls the slack (cap x length, 3 mm here) and is caught there; the fall's energy is lost.
- float32: a particle's `limiter_force` is quantised at about E h eps32 |x| (1.8 mN 1 m from the origin for this
  film). Sums hold: the 2 g strip's pins read 0.742 of its weight at the origin, 0.738 at 1 m, 0.736 at 4 m.

Not covered yet: tear seams and compress caps (the build refuses them with this limiter), the contact skin and grip
limits with it, forces on bodies through contact, bonds and pinches (the next slice), GPU numbers.

## Why

The projection is position-based. Strain caps and long-range limits are Jacobi sweeps of corrections, with no
multipliers λ. Bonds added inside the projection (an XPBD bond after each sweep) cannot be consistent, because the
caps keep no λ to balance them.

## Workaround

- **Bonds** (`alloy/plugins/bonds`). Bonded vertices are pinned for the projection. The peel force comes from
  Kendall's energy balance, T = G(θ)·b/(1 - cos θ), and is applied to the grip body with the reaction on the
  substrate. The free arm's internal forces never reach a body.
- **Pinches** (`FilmConfig.projection_friction`, `static_friction_hessian`; the C16 film-grip development notes, not
  shipped). A pinch holds within mu·N, so a carried bag stays in the jaws, and a force-limited servo still sees the
  right squeeze. Read the carried weight from the scene's masses, not from the jaws.
- **A task that needs the robot-side force of a stiff film:** check it on plain VBD with the full membrane and
  gravity off.

## Planned fix

Steps:
- Done: the reaction on pinned vertices is read out; the opt-in stress limiter carries a hung load (above).
- Multipliers on the projection were tried and dropped: a multiplier updated by local sweeps under-estimates by the
  load-to-vertex mass ratio (15,000 here), as the projection's corrections do, and carried into VBD as a force it was
  unstable on 0.36 mg vertices.
- Next: forces on bodies with the stress limiter (a 90° peel against Kendall, a pinched strap's weight on the pads),
  then tear seams and the grip and skin limits with it.
- Then the default: every film, Bonds and paper-pack law with `limiter="stress"`, and a decision.
