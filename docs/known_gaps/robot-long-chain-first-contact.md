# A long chain of light bodies in a robot file can go non-finite at its first contact on MuJoCo Warp

## What is wrong
A robot file that is a long chain of light bodies can go INVALID the moment it first touches something. Examples are a
ball chain, a necklace, or a rope of rigid links: 10 or more nested bodies, each a small sphere on a ball joint.
`sim.health()` shows NONFINITE_Q, NONFINITE_QD and FALLBACK_USED. This happens wherever MuJoCo Warp steps the robot:
on a GPU, and on the CPU with `ALLOY_CPU_ROUTE=warp`. The CPU's default route (MuJoCo C, float64) steps the same chain
without trouble. An INVALID world freezes at its last good state, so read `sim.check()["ok"]` before you trust a result.

Cables made with `scene.add_rod` (no clamp) are also MuJoCo chains, but they are not affected: see Number.

## Number
The rig: steel beads 8 mm across and 10 mm apart. The first bead has a free joint, and each neighbour pair a ball joint.
The chain is added with `scene.add_robot(name, mjcf=...)` beside a static `add_box` bench, at `dt=1/300` and
`contact_capacity=4096`. Measured 2026-11-11 on a CPU host and one L4.

| chain | where it starts | MuJoCo Warp (GPU, or CPU with `ALLOY_CPU_ROUTE=warp`) | MuJoCo C (CPU default) |
|---|---|---|---|
| 1, 2 or 5 beads | dropped 1 cm onto the bench, or draped over its edge | OK | - |
| 10 beads | in the air, no contact | OK | - |
| 10, 20 or 30 beads | dropped 1 cm onto the bench | INVALID at the landing (t 0.047 s) | OK (30 beads) |
| 30 beads | draped over the bench edge | INVALID at t 0.008 s (L4) | - |
| 60 or 120 beads | dropped or draped | INVALID at the first contact (L4) | - |

- On the 30-bead chain, dropped or draped, none of these changes the result: beads 100x heavier, 16 mm beads, joint damping 1e-3, armature
  1e-8 to 1e-6 on every joint, `condim="1"`, a single friction value, `dt=1e-3`, two hinges in place of each ball
  joint, a 0.5 mm or 5 mm `contact_gap`.
- `add_rod` with no clamp, 30 and 40 segments, was run with a 0.6 m cable of radius 2 mm (the default material) and a
  rod of radius 4 mm with 10 mm segments. Each was dropped flat 1 cm onto the same box or draped over its edge, at
  `dt` 10 ms and 1/300 s: 16 of 16 runs OK and finite on the CPU with `ALLOY_CPU_ROUTE=warp`, the drapes with PEN_WARN
  only.

## Why
MuJoCo's default solver, Newton, factorizes the matrix H = M + JᵀDJ in float32 in MuJoCo Warp. On a long tree of
light bodies that all touch down at once, this factorization breaks down, and the step returns non-finite values.
Stepping the same robot file with MuJoCo Warp alone, with no Alloy solver code in the step, gives a NaN at the same landing
step. That holds at the current MuJoCo Warp (3.17) and at the previous pin (3.16). With `solver="CG"`, which does not
factorize, MuJoCo Warp stays finite; so does MuJoCo C in float64. Why `add_rod` chains of the same length stay finite
is not yet known.

## Workaround
- Put `<option solver="CG"/>` in the robot's MJCF. The 30-bead chain then lands and rests on the bench on the CPU
  with `ALLOY_CPU_ROUTE=warp`: the world stays OK. `sim.health()` notes UNCONVERGED, the CG solver's residual, and
  PEN_WARN.
- On the CPU, keep the default route (MuJoCo C).
- For a cable, cord or rope, use `scene.add_rod` rather than a chain of bodies in a robot file.

## Planned fix
In 0.1.1, Alloy moves the MuJoCo entry of a long light chain to CG by itself. It does this already for an `add_rod`
chain with a weld or joint attached. Alternatively, Alloy re-steps a world with CG when the Newton solver's pivot guard
fires. The breakdown itself is in MuJoCo Warp's Newton solver.
