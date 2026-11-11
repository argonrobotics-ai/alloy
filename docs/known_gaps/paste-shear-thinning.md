# Shear-thinning pastes thin a little too little (MPM paste route)

**What is wrong.** A material with Herschel-Bulkley `fluid.consistency` K and `fluid.flow_index` n (yogurt, batters,
purees, peanut butter, tomato concentrate) thins with shear rate a little less strongly than its card says. Slow flow
under low stress runs close to the card's speed; flow at higher stress runs up to ~15 % slow. The yield stress and a
constant (Bingham) viscosity are right.

**The number.** Law L5: 20 mm films of K 10 Pa·s^n, n 0.4 on 3 and 4.5 deg slopes, 600 mm long, on the dense grid
(the film laws' grid). Speed / analytic at the window thickness; the slope exponent should be 1/n = 2.5:

| Voxel | 3 deg | 4.5 deg | Fitted 1/n |
|---|---|---|---|
| 5 mm | 0.993 | 0.864 | 2.16 |
| 2.5 mm | 0.989 | 0.878 | 2.21 |
| Before the fix (per-particle strain rate, 5 mm) | 1.76 | 1.41 | 1.95 |
| 5 mm, CPU, re-measured on main 4522df750 (2026-11-08) | 1.001 | 0.867 | 2.15 |
| 5 mm, CPU, a linear least-squares fit instead of the mean (tried, not landed) | 1.24 | 1.09 | 2.18 |

The test is `tests/solvers/mpm_rheology/test_herschel_bulkley.py::test_l5_power_law_index_from_two_slopes_on_gpu`
(1/n within 15 %).

**Why.** eta = K gdot^(n-1) is taken per particle from the strain rate. Each particle's velocity gradient carries
about 0.5-2 1/s of noise in every component. For n < 1 that noise collapsed eta where the true shear rate is small:
in the plug-like upper half of a film, eta read 11-16 Pa·s instead of 34-158, and the film ran 1.5-2.5x too fast. The
strain-rate tensor is now averaged over each particle's neighbours within two voxels before eta is taken from it, which
cancels the noise. Near a base or a free surface the average is one-sided. A hat-weighted linear least-squares fit
removes that bias (on a synthetic linear shear profile, at the base: the fit 1.04, the mean 3.32, true 1.00), but on
L5 it sped both films up (1.24, 1.09) and left the exponent at 2.18: the one-sided bias is not what holds the exponent
low, and the cause is not yet identified. The numbers are for 20 mm
(four-voxel) films. In layers thinner than about four voxels (a 5-10 mm smear at a 5 mm voxel) the two-voxel average
spans the whole thickness and is one-sided everywhere, so that bias applies throughout the layer.

**Workaround.** For spreading, slumping, holding a shape or stopping on a slope, the yield stress dominates and the
results hold (laws L1-L3). For flow speed, expect the card's speed within about 15 %, slow at the higher stresses.
Check against the card's flow curve at the shear rate you care about. A Bingham card (yield stress and a viscosity
fitted at your shear rate, no K and n) is exact at that rate. On the default sparse grid, flows along the ground also
run slow ([mpm-viscous-film](mpm-viscous-film.md)).

**The fix and when.** After the 1109 release: find what holds the exponent low first (the linear fit at boundaries is
ruled out, above). The candidate left is the strain rate from the grid nodes inside Newton's flow rule. Probes: a
development script for L5 on CPU (`FIT=1` swaps the fit in) and a fit-check script (not shipped).
