# A long chain of light bodies in a robot file goes non-finite at its first contact on MuJoCo Warp's Newton solver

## What is wrong
A robot file that is a long chain of light bodies can go INVALID the moment it first touches something. Examples are a
ball chain, a necklace, or a rope of rigid links: from 6 nested bodies, each a small sphere on a ball joint, at
Alloy's default impratio (Why).
`sim.health()` shows NONFINITE_Q, NONFINITE_QD and FALLBACK_USED. This happens wherever MuJoCo Warp steps the robot:
on a GPU, and on the CPU with `ALLOY_CPU_ROUTE=warp`. The CPU's default route (MuJoCo C, float64) steps the same chain
without trouble. An INVALID world freezes at its last good state, so read `sim.check()["ok"]` before you trust a result.

Cables made with `scene.add_rod` (no clamp) are also MuJoCo chains, but they are not affected: see Why.

Alloy now handles such a chain itself. At build it screens every MJCF and URDF robot file, and when the file is a long
chain of light bodies (the rule in Why) it steps the scene's MuJoCo entry on the CG solver, which stays finite. The build
warns, and the route report says so in a line that starts "mujoco solver: CG for robot". What is left:
- A robot file that sets its own `<option solver=...>` keeps it: a long chain held on Newton that way still goes INVALID.
- The rule's bound was measured on chains of steel beads. A chain of another shape that falls just above the bound has
  no measurement behind it.
- On CG, `sim.health()` notes UNCONVERGED (the CG residual at its iteration cap), and every robot and rigid body in the
  scene steps on CG with the chain, not only the chain.
- USD robot files are not screened.

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
- With Alloy's rule (CG): the 10-bead chain dropped 1 cm lands and stays OK on the CPU with `ALLOY_CPU_ROUTE=warp`;
  held on Newton by its file, it goes INVALID at the landing. The 30-bead chain dropped 1 cm stays OK on one L4 over
  0.2 s (PEN_WARN only) (`tests/scene/test_robot_chain_cg.py`).
- `add_rod` with no clamp, 30 and 40 segments, was run with a 0.6 m cable of radius 2 mm (the default material) and a
  rod of radius 4 mm with 10 mm segments. Each was dropped flat 1 cm onto the same box or draped over its edge, at
  `dt` 10 ms and 1/300 s: 16 of 16 runs OK and finite on the CPU with `ALLOY_CPU_ROUTE=warp`, the drapes with PEN_WARN
  only.

## Why
MuJoCo's default solver, Newton, factorizes the matrix H = M + JᵀDJ in float32 in MuJoCo Warp. On a long tree of
light bodies that all touch down at once, this factorization breaks down, and the step returns non-finite values.
Stepping the same robot file with MuJoCo Warp alone, with no Alloy solver code in the step, gives a NaN at the same landing
step. That holds at the current MuJoCo Warp (3.17) and at the previous pin (3.16). With `solver="CG"`, which does not
factorize, MuJoCo Warp stays finite; so does MuJoCo C in float64.

A margin measures it: eps32·|S·JᵀDJ·S| / λ_min(S·M·S), where S = diag(M)^-1/2 and JᵀDJ sums the constraint rows
(`alloy.scene.newton_rule`, MuJoCo C, the worst control period of a 1 cm drop). The chain makes λ_min small: its joints'
inertias are nearly dependent, whatever the bead mass. The contact rows' curvature grows with the impratio, which Alloy
sets to 300 for a robot whose file sets none.

| chain | λ_min(S·M·S) | impratio | margin | MuJoCo Warp |
|---|---|---|---|---|
| 5 beads | 1.22e-3 | 300 | 11 | OK |
| 6, 7, 8 beads | 6.1e-4, 3.5e-4, 2.2e-4 | 300 | | INVALID |
| 10 beads | 9.9e-5 | 300 | 296 | INVALID |
| 10 beads | 9.9e-5 | 1 (in the file) | 1.0 | OK |
| 30 beads | 2.6e-6 | 300 | 36,400 | INVALID |
| 30 beads | 2.6e-6 | 1 (in the file) | 122 | INVALID |
| 30 beads, armature 1e-6 | 2.4e-4 | 1 (in the file) | 1.6 | - |
| `add_rod`, 30 segments, 4 mm rod and 0.6 m cable | 2.0e-3, 9.9e-4 | 1 | 0.60, 0.94 | OK |
| shipped robots (FR3, Panda, UR10e, G1, H1, Go2, LEAP) | 0.012 (G1) to 0.25 | 1 to 100 | up to 1.95 (LEAP) | OK |

So an `add_rod` chain stays finite for two reasons together: Alloy's CHAIN armature floor (1e-6 kg·m² on every rotational
dof) raises λ_min about 100x, and a chain on Newton runs impratio 1, not 300. Armature alone did not save the bead chain
at impratio 300 (Number).

Alloy's rule screens the robot file at its own pose (`alloy.scene.robot_chain`, about 2-45 ms per file) and picks CG
when λ_min(S·M·S) < 1e-3 · max(impratio, 10) / 300: below 1e-3 at the route's impratio 300, so 6 beads and more go to
CG and 5 stay on Newton; at impratio 1 in the file, below 3.3e-5, so 10 beads keep Newton and 30 go to CG. Every shipped
robot is at least 12 times above the bound, and so is every robot file in the repository's library, tests and examples.

## Workaround
- Leave the solver unset in a chain's robot file (Alloy picks CG), or put `<option solver="CG"/>` in it yourself. The
  30-bead chain then lands and rests on the bench on the CPU with `ALLOY_CPU_ROUTE=warp`: the world stays OK.
  `sim.health()` notes UNCONVERGED, the CG solver's residual, and PEN_WARN.
- On the CPU, keep the default route (MuJoCo C).
- For a cable, cord or rope, use `scene.add_rod` rather than a chain of bodies in a robot file.

## Planned fix
Done after 0.1.1 for robot files that leave the solver unset (above). Not planned: re-stepping a world on CG when the
Newton solver's pivot guard fires. The breakdown itself is in MuJoCo Warp's Newton solver.
