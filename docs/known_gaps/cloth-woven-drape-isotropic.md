# A woven fabric card from KES data drapes wrong on the VBD cloth route

## What is wrong
- A cloth card built from a woven fabric's published KES-F data (areal mass, thickness, bending rigidity B) drapes
  differently from the same fabric on a Cusick drape tester, even when the solve is converged.
- With the shipped KES cards' convention (`cloth_cotton_plain`, `cloth_polyester_woven`: youngs back-calculated from
  B at the KES thickness, no `elastic.cloth`, so the router sends it to CLOTH_VBD), a light polyester twill drapes too
  stiff (+0.08 to +0.12) and a heavy polyester/viscose plain weave hangs far too low (-0.28). A wool plain weave is
  within 0.05 on four of five meshes and misses on the fifth.
- `cloth_accuracy="super"` does not fix it. The DC is converged in iterations, so the gap is in the model, not the
  solve: membrane locking, the card mapping and the 2 mm contact thickness (Why, below).
- The orthotropic membrane (Workaround) brings the twill and the wool within 0.035 on a warp / weft grid mesh, where
  an `add_cloth` sheet gets it by default, and within 0.05 on a mesh that does not follow warp and weft (each triangle
  carries one yarn; a mesh sheet still needs the route override). The heavy plain weave still hangs 0.16 too low.

## Number
D03M (`bench/scenarios/fabric_drape*.py`, pack `bench/scenarios/packs/fabric_drape_petrak2021.json`; its L4
result rows are not shipped) and the R6 law on the same data
(`alloy.scene.twins.drape_route`, dataset `bench/rigs/R6.cloth_drape/petrak2021-v1`). The data is Petrak, Mahnić
Naglić, Rogale and Geršak 2021 (Materials 14:6259): 30 cm specimens clamped between 18 cm discs, KES-FB inputs, each
fabric's measured DC the mean of 5 specimens. Alloy: the production route, one L4, 8 worlds per fabric, 8 mm mesh,
scored over the last 1 s of 4.5 s.

| fabric | measured DC | super (60 it), mesh seed 0 | 240 it | R6 law, super, mesh seed 7 |
|---|---|---|---|---|
| F1 polyester twill 3/1, 135 g/m², 0.41 mm | 0.222 | 0.302 (+0.080) | 0.315 (+0.093) | 0.342 (+0.120) |
| F4 wool plain, 162 g/m², 0.44 mm | 0.319 | 0.306 (-0.013) | 0.319 (-0.000) | 0.381 (+0.062) |
| F9 polyester/viscose plain, 367 g/m², 2.42 mm | 0.647 | 0.365 (-0.282) | 0.368 (-0.279) | 0.365 (-0.282) |

The mesh moves these by up to about 0.08. Super on a square-grid disc:
- 8 mm: F1 0.315, F4 0.367, F9 0.406;
- 5 mm: F1 0.300, F4 0.340, F9 0.429.
F1 and F9 miss on every mesh. F4's pass is not robust: on the R6 mesher's seed 7 the fold count, and with it DC,
changes (2-5x the world spread of seed 0).

The opt-in orthotropic membrane (KES EM warp / weft stretch, KES G shear, KES B per hinge direction; super, 8 worlds,
2026-11-08):

| fabric | warp / weft grid disc, 8 mm, 60 it | same, 160 it | same, contact thickness T0 | R6 mesh (jittered hex), seed 7, T0, before the yarn split |
|---|---|---|---|---|
| F1 | 0.254 (+0.032) | 0.257 (+0.035) | 0.252 (+0.030) | 0.500 (+0.278) |
| F4 | 0.314 (-0.005) | 0.313 (-0.006) | 0.294 (-0.025) | 0.395 (+0.076) |
| F9 | 0.487 (-0.160) | 0.494 (-0.153) | 0.487 (-0.160) | 0.499 (-0.148) |

The same isotropic card convention on that grid disc: 0.315 / 0.367 / 0.406. With one yarn per triangle (2026-11-10)
the jittered-hex disc drapes F1 0.24-0.26 and F4 0.27-0.30 (two mesh seeds, 16 worlds each; Workaround).
DC is converged in iterations there. Measured at the end of the drape (8 worlds), the yarns stretch little: the
largest yarn strain |w| - 1 is 0.6 % at 60 iterations (99th percentile 0.08 %) and 0.03 % at 160. The largest edge
strain, 8 % at 60 iterations and 4.3 % at 160, is the grid's cell diagonals as the sheet shears. On F1, the lightest,
the local shear is not converged at 60 iterations: its 99th-percentile shear angle is 9.5°, against 2.2° at 160 (F4
and F9 about 2° at both). `sim.health()` does not flag shear, so raise the iterations before reading a woven sheet's
local shear.

The same fabrics with other isotropic membranes (super, 60 iterations; bending and mass unchanged):

| fabric | E h from KES shear G (E h = 2 (1 + nu) G) | E h from KES tensile (EM secant) |
|---|---|---|
| F1 | 0.217 (-0.005) | 0.730 (+0.508), 3 nodes |
| F4 | 0.251 (-0.068) | 0.724 (+0.405), 4 nodes |
| F9 | 0.440 (-0.207) | 0.590 (-0.057) |

## Why
Measured (2026-11-08, L4, super, R6 seed-7 mesh, 8 worlds): the drape on CLOTH_VBD is set by the membrane's
in-plane stiffness through locking, more than by the bending.
- Under the drape's own loads a fabric stretches about 0.02 %, so a converged sheet would sit at its inextensible
  limit and its DC would hardly move with E h. On the 8 mm mesh it moves a lot. With the bending fixed, scaling the
  card's E h by 0.3 / 1 / 3 / 10 gives F1 0.225 / 0.347 / 0.564 / 0.714 (F4 0.269 / 0.370 / 0.508 / 0.720). At
  E h x10 a 6 mm mesh drapes lower than the 8 mm one (F1 0.587 against 0.714).
- That is membrane locking: linear triangles cannot fold into a drape's doubly curved shape without stretching, so the
  stiffer the membrane, the stiffer the drape, and the mesh changes the answer. Scaling the bending by 0.72 moves DC
  by only 0.04-0.05.
- The card mapping decides where on that curve a fabric sits. The convention above (E from B at the KES thickness)
  makes the membrane's shear stiffness 7x KES G for F1, and its stretch stiffness 17-42x below KES EM (F1). A woven fabric
  is stiff along its yarns and soft in shear between them, which no single isotropic E h matches.
- The sheet's contact thickness matters too. The rig floors it at 2 mm (5x F1's 0.405 mm), and VBD's self-contact
  then keeps the folds from closing: 9 folds instead of 12. With only the thickness changed to T0 (E h kept, 16
  worlds), F1 drapes 0.280 +- 0.006 instead of 0.355 +- 0.014, and F4 0.298 +- 0.003 instead of 0.365 +- 0.011.
- The clamp is not the cause (a pinned rim and an unclamped specimen drape within the world spread), nor is the
  iteration budget (60 and 240 iterations agree), nor settling (9 s drapes as 4.5 s).
- The orthotropic membrane does not lock on a grid whose edges run along warp and weft. Each triangle's two yarn
  conditions are then its two grid-edge lengths, about two per vertex against three degrees of freedom, so the sheet
  can shear into a drape like a real woven fabric (a Chebyshev net). On an irregular mesh each triangle adds two
  independent conditions, about four per vertex, and the stiff yarns lock harder than the isotropic membrane does.
- F9 (2.42 mm, 367 g/m²) misses with every membrane. Its inputs match the paper's table. The paper's FAST bending
  (a cantilever, low curvature) is 2.3-2.8x its KES B for every fabric. As a diagnostic (not the pre-registered
  mapping), FAST bending on the orthotropic membrane drapes F9 within 0.02 (0.630) but F1 and F4 too stiff (+0.105,
  +0.079). So neither bending source fits all three: F9 bends stiffer at drape curvatures than its KES B, F1 and F4 do
  not (bending hysteresis, which Petrak does not report, is one candidate).

## Workaround
- For the drape shape of a light, limp woven fabric, set the card's youngs from KES shear: E = 2 (1 + nu) G / h, with h
  the sheet thickness you pass to `add_cloth`. Here it was within 0.07 on F1 and F4.
- For a heavy fabric whose hang matters, use the tensile stiffness (E = F / (EM h) from KES-FB1). It was within 0.06 on
  F9.
- Check one Cusick drape against a measured DC for your fabric before trusting a drape shape.
- For a woven fabric with KES data, use the orthotropic membrane:
  - give the material card `elastic.cloth` (stretch_warp and stretch_weft = KES load / EM strain in N/m, shear = KES G
    in N/m per radian, bend_warp and bend_weft = KES B in N m);
  - an `add_cloth` sheet (a warp / weft grid, warp along x) then routes to `cloth_vbd` with the membrane by default,
    alone or beside other moving things; `sim.routes()` says so;
  - a sheet from a mesh needs the object card's route override `route: {route: cloth_vbd, reason: ...}`. Without it
    the card routes to CLOTH_STYLE3D when nothing else moves and is refused at build beside other moving things (the
    error names the override). On a mesh without warp / weft edges each triangle carries one yarn, so it does not
    lock: on the R6 jittered-hex disc F1 drapes 0.24-0.26 and F4 0.27-0.30 (measured 0.222 and 0.319; two mesh
    seeds, 16 worlds each), as on the grid (0.26, 0.31). Before that split the same disc gave 0.49 and 0.43. The
    split keeps 1 % of each triangle's other yarn (none at all leaves free modes and the sheet blows up), and the
    drape depends on that number: F1 0.334 / 0.289 / 0.263 / 0.251 / 0.245 at 10 / 3 / 1 / 0.5 / 0.2 % kept.
  Youngs randomization of such a card is refused. `add_cloth`'s default card `cotton_woven` is isotropic (youngs and
  poisson, no `elastic.cloth`) and is not changed by any of this; `cotton_woven_aniso` is the woven one. The
  membrane route costs 1.9-2.5x the Style3D route it replaced on grid sheets (a 0.4 m sheet of 625 vertices on a
  box, 64 worlds, L4: 2,160 against 4,151 world-steps per second; a sheet alone, 1 s on the CPU: 121 s against
  49 s), and the sheet drapes differently: on that sheet alone vertices end up to 0.13 m (mean 0.04 m) from where
  Style3D put them, with a peak edge strain of 8.3 % against 5.0 % (a benchmark audit, 2026-11-10).
- CLOTH_STYLE3D is no alternative for these fabrics today. In one world, clamped or not, the Style3D sheet stays
  nearly flat (DC 0.98-0.99): its solve locks when the KES stretch stiffness outweighs the inertia per step
  (`cloth-style3d-stretch-lock.md`).

## Planned fix
The law is R6 / D03M: each fabric's DC within 0.05 of its measured value at
convergence. Next:
- F9: its bending at drape curvatures (FAST fits F9 but not F1 / F4; a KES 2HB for it would test the hysteretic
  hinge, `elastic.bending_hysteresis`). A free strip also bends about 35 % too soft on the cloth route (0.64 D;
  the cantilever law). With the cards' bending times 1 / 0.6 on the grid, F9 drapes 0.579
  (-0.068), F1 0.285 (+0.063) and F4 0.360 (+0.041). So that softness is part of F9's miss, and F1 passes with
  today's bending but would leave the 0.05 bar once the hinge map is fixed: the two get re-scored together then.
- Mesh sheets by default: woven cards on a mesh still need the route override (done for grid sheets).
- R6 at the fabrics' real thickness (the 2 mm floor).
