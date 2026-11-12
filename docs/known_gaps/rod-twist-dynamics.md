# A thin rod's twist vibration rings too slowly or not at all (CHAIN route)

## What is wrong
- A flexible rod object (a card the router reads as a rod, such as a cable laid along a curve) goes to the CHAIN
  route: MuJoCo D6 joints with a twist spring k = G J / l per link. Under a steady end torque it twists by exactly
  T L / (G J). That is law A11.torsion, error 1.1e-6.
- How the twist moves on the way there is not physical for thin rods. A real rod released from a twist rings at its
  first torsional mode. Alloy's chain, at the production step, either decays smoothly (card damping) or rings at a
  few steps per period and dies out within tens of milliseconds (near-undamped). At a step small enough to resolve
  the mode, it rings 2.6x too slowly.
- Bending dynamics are not affected. Their modes are far slower and have their own laws (A11.cantilever, the rod
  self-test, the hanging-chain swing that A10's armature check moved by 0.28 % at most).

## Number
CPU, MuJoCo C, measured with development scripts (a twist-release script and a floor-pair script; not shipped). The rod
is the law's: 30 links, 0.5 m, 2 mm radius, G J = E I / 1.3. It is twisted 0.5 rad, held, then released. The continuum's
first torsional mode is 3,051 rad/s (486 Hz, period 2.06 ms).

| step | armature on each twist dof | damping | released twist |
|---|---|---|---|
| 1.25 ms (production) | 1.29e-6 kg·m² (stability term, 2,478x the link's I) | 0.05 | decays, e-fold 3.6 ms (continuum: 4.5) |
| 1.25 ms (production) | same | 0.0005 | rings a few steps per period, gone in 20-50 ms |
| 25 µs | 1e-6 kg·m² (the floor, 1,923x) | 0.0005 | period 5.32 ms (2.58x) |
| 25 µs | none (probe only) | 0.0005 | period 2.061 ms (+0.07 %) |

Damping is the bend-mode damping ratio the chain's stiffness-proportional damping is set from; at 0.05 the
continuum's twist mode is damping-limited too.

**On the ROD_VBD route** (a development twist-release script, not shipped). This is Newton's
Cosserat rod on `SolverVBDRod`, the same rod from the ROD_VBD emitter, with no armature:

| step | damping | released twist |
|---|---|---|
| 1.25 ms (production) | 0.05 | decays, no ringing, e-fold 5.1 ms (continuum: 4.5) |
| 1.25 ms (production) | 0.0005 | 0.042 rad after one step: backward Euler damps the mode out within ~2 steps |
| 25 µs | 0.0005 | period 2.080 ms (+1.0 %) |

So the chain itself carries torsion correctly; the armature is what slows it. An estimate, not measured beyond this
rod: the first mode's period grows by about sqrt(1 + a (pi / 2N)² / I), with a the armature, I = m r² / 2 one link's
axial inertia and N the link count (2.5 here at 25 µs, measured 2.58).
- The stability term makes this sqrt(1 + (omega1 h)² / 2) for any radius: 2.9x for this rod at the production step,
  where the mode lands at about 4.4 steps per period (it can never exceed omega h = sqrt(2)) and implicitfast damps it
  out.
- The floor's share shrinks as r⁴: for a 1 cm rod of the same length and links it is about 0.4 %.

## Why
- The torsional wave speed sqrt(G / rho) is about that of sound in the material, so a rod short enough to handle has
  its first twist mode at hundreds of Hz. Its period is 4 L / sqrt(G / rho), independent of the radius: at 10 steps
  per period the 1.25 ms step resolves only rods longer than about 3 m of this material.
- The CHAIN emitter gives every rotational dof armature max(2 k h² / 4 - I, 1e-6 kg·m²). The first term keeps
  implicitfast stable at the step h; the floor keeps MuJoCo Warp's float32 Newton solver finite on contact-loaded
  light links (A10's rope went non-finite in 16 of 16 worlds at 1e-9). On a thin rod's twist axis both are far more
  than the link's own inertia, and armature acts on the joints' relative rates, which the twist mode drives.

## Workaround
- For a steady or slowly applied twist (a cable routed with a set twist, unwinding paced by damping) none is
  needed: statics are exact, and the damped release decays in 3.6 ms against the continuum's 4.5 ms.
- A thick rod's twist vibration needs a step that resolves it (omega1 h well under 1, e.g. 20 steps per period);
  the floor then moves it little (the estimate above). Check sqrt(1 + a (pi / 2N)² / I) for your rod first.
- No setting removes CHAIN's floor, so on CHAIN a smaller `dt` alone gives the 2.6x-slow mode above.
- A thin rod's torsional vibration (a torsional pendulum, a twist wave) needs the ROD_VBD solver (no armature) at a
  step that resolves the mode: the period came out +1.0 % at 25 µs. ROD_VBD is routable from R2 of the rod change.

## Planned fix
None planned. The floor could drop where float32 does not apply (the CPU route runs MuJoCo C in float64), or on the
twist dofs if the float32 failure does not need them there (A10 found armature on one dof group alone not enough).
Either needs A10's 16-world check rerun. Revisit if a task needs torsional vibration; the law to add is the release
period against omega1 within a few percent at a resolving step.
