# A cloth particle that starts a step inside a rigid contact is thrown out at its depth / dt

## What is wrong
- A cloth particle that starts a step inside a rigid object's contact distance leaves it at about the depth the step
  pushed it out by, divided by the step. How stiff the contact is does not matter.
- Small steps make it large: robot fingers coupled to cloth step the cloth at 19.5-39 us, where 0.5 mm of depth is
  18.7 m/s.
- `ClothFold-v0`'s friction pinch is where it shows. As the closing fingers squeeze the gathered towel corner, a
  corner particle is thrown at 14-33 m/s and SPEED_FAIL marks the world INVALID (frozen), in about 40 % of the worlds
  of the pinch-and-lift repro.
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
- ClothFold, the pinch-and-lift repro (GPU, 128 worlds, 2026-11-10; development notes, not shipped): 55 worlds reset,
  at 20.2-37.5 m/s (median 25.1).
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
- Opt-in: `body_particle_exit_speed=<m/s>` in the cloth solver's settings (`True` for 1 m/s), or
  `ALLOY_EXIT_SPEED=<m/s>` in the environment (`alloy/solvers/vbd/exit_speed.py`).
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
  - Not yet measured on ClothFold; the 128-world pinch row with it is to run.

## Planned fix
- The ClothFold rows with the cap (INVALID worlds, fold count). Then the cap as a default, after its laws
  and an engine audit.
- Then trace how the particle starts a step that deep, which the cap does not fix.
