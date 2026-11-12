# Rigid clutter: first-order spin in free flight on MuJoCo Warp, rk4's cost on the CPU's MuJoCo C route

## What is wrong
Which integrator a robot-free scene of free bodies steps on depends on the route (`alloy.scene.integrator_route`):
- MuJoCo Warp (any GPU, and the CPU warp route `ALLOY_CPU_ROUTE=warp`): more than 8 free bodies, none raised by the
  inertia floor, step on implicitfast with rk4's contact gains and a cap on the friction rows' pull (F6 R1, back on
  2026-11-09 with the friction cap). A body tumbling in the air there turns first order: about 0.6 % of its angular
  momentum over a 2 s toss at 10 rad/s; A4's torque-free tumble drifts 2.3e-3 at 1.25 ms against rk4's 1.1e-4 (bound
  1e-3).
- MuJoCo C (the CPU default): the cap has no hook there yet, so the same scene keeps rk4, which collides once per
  stage: about four collision passes per step, 3-6x slower per step than implicitfast would be.
- Up to 8 free bodies, or one the inertia floor raises, step on rk4 on both routes.
`sim.routes()` names the integrator and the reason on every object.

## Number
CPU, the default route, the rocking law's cases (tests/bench/test_rocking_law.py; pitch peak to peak of a cube
pushed at its COM by 1.5 mu m g, mu 0.5, beside 8 resting cubes; per-route result rows, not shipped):
- 30 cm cube over 2 s (to 4.9 m/s), last third: MuJoCo C (rk4) 3.86 deg, lift 17.5 mm; MuJoCo Warp (R1 + cap)
  0.48 deg, lift 0.8 mm. Without the cap R1 rocked 36.4 deg (MuJoCo C), which is why R1 was off until the cap.
- 5 cm cube over 2 s: rk4 flips it (the sliding hop); R1 + cap 0.14 deg.
- The 27-cube stacks benchmark: rk4 at 2.5 ms and R1 + cap at 5 ms both settle to 0.22 mm, 0 falls.
- On the CPU, MuJoCo C's rk4 still steps these scenes faster than MuJoCo Warp's implicitfast (0.23 against 0.87 s
  per simulated second for the pushed cube beside 8; 1.33 against 1.69 for two stacked worlds).

## Why
implicitfast integrates a body's rotation to first order: the gyroscopic term w x (I w) drifts unless the inertia is
isotropic. rk4 integrates contact four times per step. Uncapped, implicitfast let a sliding body rock and grow
(MuJoCo's soft friction pull, -b v_slip, coupled into the normal rows by the elliptic cone); the cap saturates that
pull, and it is a MuJoCo Warp kernel.

## Workaround
- Accurate spin in the air in a large scene: `profile="precision"` keeps rk4, or keep the scene at 8 free bodies.
- Robot scenes are not affected: they step on implicitfast on either route.

## Planned fix
The same cap on the MuJoCo C route (the friction cap's rule after `mj_step1` in `alloy/route/_mjc/step.c`; on Alloy's
numpy tick it matched MuJoCo Warp to 3 digits) would let the C route take R1 too.
