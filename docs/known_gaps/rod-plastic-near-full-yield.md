# A wire loaded close to its fully plastic moment settles slowly and less accurately

## What is wrong
A yielding wire (a `rod_vbd` rod whose card has `plastic.model: j2`) updates each joint's rest shape once per solver
step, from the converged pose (explicit in time). Under a held moment close to the section's fully plastic moment
M_p (or torque T_p) the wire takes many steps to settle, because the section is nearly flat there: a small moment
step gives a large curvature step. The settled curvature is then less accurate: the fibre section's few-tenths-
of-a-percent error in M becomes several percent in κ.

## Number
CPU, SolverVBDRod, a clamped copper-like wire (0.2 m, R 1 mm, 20 segments; E 117 GPa, σ_y0 70 MPa, perfectly
plastic), gravity off, end moment ramped over 0.2 s, held, released (`tests/solvers/rod/test_rod_plastic.py` rig):

| end moment | loaded κ vs the closed form | residual κ after release |
|---|---|---|
| 1.2 M_y (0.71 M_p) | -0.13 % | -4.8 % of itself, 0.13 % of the loaded κ |
| 1.5 M_y (0.88 M_p) | +0.20 % | +0.84 % |
| 1.65 M_y (0.97 M_p), 0.4 s hold | -17 % (not settled) | -27 % |
| 1.65 M_y, 1.6 s hold | +1.9 % | +3.2 % |

Torsion (T_p = 4 T_y / 3): 1.15 T_y, loaded -0.38 %; 1.25 T_y (0.94 T_p), loaded +0.74 %. The residual twist is
within 0.4 % and 0.7 % of the loaded twist.

## Why
- The rest update is a fixed-point iteration with the elastic stiffness as its step. It contracts by
  (1 - k_t / k), with k_t the section's tangent stiffness, which goes to 0 at M_p for a perfectly plastic wire.
- The 220-fibre section is within 0.35 % of the closed-form M(κ) and T(φ') up to 0.98 M_p
  (`alloy/solvers/plastic/j2.py`). Near M_p, dM/dκ is small, so that error in M is several percent in κ.

## Workaround
- Load a wire to at most about 0.9 M_p when the bent shape matters, or hold the load longer. At 0.97 M_p it took
  1.6 s to settle.
- Card hardening (H > 0, e.g. `copper_c110`'s 603 MPa) keeps k_t above 0 and helps.
- Read `"<rod>.plastic_strain"` to see where and how far a wire yielded.

## Planned fix
An implicit rest update, with the section's tangent in the rod solve's joint Hessian, so the step converges like
the elastic one. Not scheduled. Reversed loading has no Bauschinger effect (the model:
[plasticity-metal.md](plasticity-metal.md)).
