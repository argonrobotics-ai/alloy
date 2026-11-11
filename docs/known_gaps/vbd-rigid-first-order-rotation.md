# Stock SolverVBD rigid bodies: an applied torque wraps and a spin slows down

## What is wrong
Newton's `SolverVBD` builds each rigid body's inertial target with one first-order rotation step,
normalize(q + dt/2 omega q), where omega already includes dt I^-1 tau from any torque in `body_f`. A step that asks
for an angle x then turns 2 atan(x / 2). That gives two errors.
- **An applied torque reaches the body short, and can wrap.** x = dt² tau / I. A light, thin body (a cable segment)
  has a tiny axial inertia, so x is large. A clamped 12-segment rod twisted 0.51 and 0.21 of T L / (G J) at
  0.01 / 0.03 N m; the A11.torsion rod twisted 0.018 of it.
- **A spinning body slows down.** Each step loses (omega dt)² / 12 of the spin, so a body that rolls feels a drag
  that grows with speed and is first order in dt.

## Number
CPU, 1.25 ms, measured with two development scripts, a single-body roll and a roll check (not shipped). One rigid
capsule (r 5 mm, 0.3 m, 1200 kg/m³) and a 12-segment rod roll down an incline, mu 0.8, 0.4 s from rest:

| body, slope, dt | acceleration / its rolling value | energy lost |
|---|---|---|
| capsule, 10°, 1.25 ms | 0.956 | 4.8 % |
| capsule, 10°, 0.625 ms | 0.983 | 2.5 % |
| capsule, 20°, 1.25 ms | 0.830 | 15 % |
| rod on the first-order target, 10°, 1.25 ms | 0.948 | 5.1 % |

- The (omega dt)² / 12 loss predicts 4.1 % at 10° and 16 % at 20°.
- The loss does not move with contact damping, stiffness (ke 250-25,000), friction_epsilon or hard contacts.

## Why
The target's rotation is a first-order (Cayley-like) step, not the exponential map. A torque in the target is
also scaled by dt² / I, which no rotation step can take for large values: even an exact exp map wraps past pi.

## Workaround
- **Rods (ROD_VBD):** `alloy.solvers.rod.SolverVBDRod` fixes both for its chain bodies.
  - Torques go into the chain's Gauss-Newton step as a potential: twist exact to 1e-7.
  - The target rotates by exp(omega dt) (`exact_rotation_target`): rolling 0.998 / 0.991 of its value at
    10 / 20 deg.
- **Other bodies on VBD:** use a smaller `dt` (the spin loss is first order in it), and keep applied torques with
  dt² tau / I well under 1. Alloy routes free rigid bodies to MuJoCo, which does not have either error.

## Planned fix
None for stock SolverVBD bodies. The same exp-map target belongs in Newton's `forward_step_rigid_bodies`; that is
an upstream change, not scheduled.
