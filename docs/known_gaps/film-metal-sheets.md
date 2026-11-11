# Metal sheets on the film: VBD sags a stiff sheet at real density, and the hinge network bends unlike a plate

Status: open. J2 metal sheets (SCR-WP7-15 P3) are on main; their laws run mass-scaled, which hides both limits. A J2
film runs 20 VBD iterations from the inertial start. That stops the sag from yielding for a 6061 or brass sheet on 5 mm
cells; steel, stainless and copper sheets, and any metal on finer cells, still yield under their own weight before
anything presses them. `add_cloth` with a j2 material always makes a film sheet. Design, laws and the full numbers are in the sheet-plasticity
development notes (not shipped).

## What is wrong

- **A stiff sheet sags, and the sag could yield.** `scene.add_cloth(..., thickness=t,
  material="aluminium_6061_t6")` (any `plastic.model: j2` card) routes `cloth_film`. At the card's real density
  VBD's Gauss-Seidel does not reach a metal sheet's statics in a step's iterations, so a sheet draped over a bar or
  laid on supports droops far below its static shape: a metal sheet in a scene looks floppy.
  - At VBD's 10 iterations that sag bent the sheet past yield. The sheet then kept a dent nothing pressed, and
    `plastic_strain` read it.
  - At 20, the J2 film's default, it still does for most sheets: a 6 cm, 0.25 mm sheet at rest on two supports
    reads ε̄p 0 in 6061 and brass on 5 mm cells, 0.001-0.007 in steel, stainless and copper, and 0.009-0.016 in
    6061 and steel on 3 mm cells (a finer mesh sags more: 2 mm, then 5-9 mm). `sim.check()` stays ok. The sag is
    1,000 x or more the static one.
- **A dented sheet on box edges creeps.** Released from a press, it stays on round or box supports. On round
  supports it settles within 1 mm; on the sharp edges of boxes it keeps sliding sideways, about 1 mm/s.
- **VBD's adaptive start throws a dented sheet.** With `alloy_initial_guess="adaptive"` (the J2 default in
  b650fd2fd only), a dented metal sheet released from a press walked sideways and climbed off its supports, while
  `sim.check()` stayed ok. Stiff CLOTH_VBD sheets did the same until WP2's 24178a806, which starts them inertial.
- **The hinge network is not the plate.** Each hinge carries the plane-strain J2 section on its own curvature. Bent
  along the mesh grid that is exact, but:
  - under a concentrated load (a cylinder pressing a strip) the network is more compliant than the plate;
  - bent at an angle to the grid it is stiffer below yield (up to 1.5 × at 45°), and past yield it carries more
    moment than the plate (up to 1.94 ×);
  - a dent's kept depth therefore depends on the mesh.

## Number

- **Sag** (CPU, main a08d7007e): a 1 mm steel strip on two supports 100 mm apart, dt 2e-5, 50 ms. The static sag is
  5.4 µm, and the first swing reaches 10.8 µm. Measured sag by solver setting:
  - default start, 20 iterations: 1,170 µm;
  - `alloy_initial_guess="adaptive"`: 42 µm;
  - adaptive plus `alloy_chebyshev_rho` 0.9: 100 µm at 20 iterations, 49 µm at 60.

  At dt 1 ms the real-density strip fell off its supports in every variant.
- **Sag that yields, through the API** (CPU, dt 0.01, a 6 cm, 0.25 mm 6061 sheet, res 13, 1.5 s):

  | case | inertial, 10 iterations (before P3's follow-ups) | adaptive, 10 (b650fd2fd) | inertial, 20 (the J2 default now) |
  |---|---|---|---|
  | hung over a 20 mm bar | droops 9 mm, ε̄p 9.6e-4 | rings 1.5-5 mm, ε̄p 0 | 5.1 mm, ε̄p 0 |
  | on two 10 mm round supports 4 cm apart (static sag ~2 µm) | 2.7 mm, ε̄p 1.8e-3 | 0.27-0.36 mm, ε̄p 0 | 1.9 mm, ε̄p 0 |

  OSS-17 measured the 10-iteration inertial start independently: 10-15 mm and 0.0041 over a bar; 0.0003-0.0012
  after a 0.14-0.3 mm drop onto round supports. At 40 iterations the bar droop is 3.1 mm. With
  `alloy_chebyshev_rho` 0.9 on the inertial start at 20 (main 4522df750): bar 3.2 mm, round supports 1.5 mm, ε̄p 0,
  every press-and-release case below stays on its supports, 1.14 x the step cost. Still 100 x or more the statics,
  so the default stays as it is.
- **Yield at rest, by metal and mesh** (CPU, dt 0.01, main 270ead86d; the 6 cm, 0.25 mm sheet laid on two 10 mm round
  supports 4 cm apart, nothing pressing; found by the pre-release audit, then re-measured).
  Sag and largest ε̄p at 0.2 s, and at 1.0 s where they moved:

  | material | res 13 (5 mm cells) | res 21 (3 mm cells) |
  |---|---|---|
  | aluminium_6061_t6 | 2.2 mm, 0 | 5.4 mm, 0.0085 (7.7 mm at 1 s) |
  | steel_carbon | 2.2 mm, 0.0025 | 7.6 mm, 0.0152 (9.0 mm, 0.0159 at 1 s) |
  | stainless_304 | 2.1 mm, 0.0015 | |
  | copper_c110 | 3.5 mm, 0.0072 | |

  The audit's rows (a 0.2 mm lower start): brass 0, steel 0.0014, stainless 0.0007, copper 0.0055 at res 13; 6061 on
  2 mm cells (res 31) 0.110, sag 10.3 mm; a 0.5 mm 6061 sheet at res 13 0.0014. Scene dt 0.005 gave identical
  digits, an L4 the same to 4 digits. The static sag is about 2 µm and its strain 1e-6.
- **Press and release** (OSS-17's rig, CPU, dt 0.01; `test_a_dented_sheet_released_fast_stays_on_its_supports`): the
  same sheet on two 10 mm round supports 4 cm apart (or 2 cm boxes 5 cm apart), pressed 15 mm by a kinematic 5 mm
  sphere, released at 0.1 m/s (fast) or 0.02 m/s (slow), watched to t 3 s:
  - inertial, 10: stays on round supports; on boxes it went INVALID (PEN_FAIL, contact 0.6 mm deep; scene dt 0.005
    gave identical digits);
  - adaptive, 10: walks sideways 0.1 s after the release ends, climbs to 74-80 mm (supports' tops at 50 mm), on the
    ground by t 2.9; on boxes it rises 12 mm and falls off. Slow: stays on, but creeps upward (52.5 -> 56.1 mm in
    1.4 s). At 40 iterations the same (80 mm); with Chebyshev 0.9 at 40 it climbs slower (61 mm at t 3). Nothing
    flags it: `sim.check()` stays ok, health reads DEGRADED (UNCONVERGED) as during any press, and the film ledger's
    largest one-step energy gain does not move (6.9 µJ, from the press): the ~0.6 mJ climb comes over ~70 steps;
  - inertial, 20: stays on both, fast or slow. Round: settles, 0.7 mm sideways. Boxes: 2.3 mm sideways in 2.2 s and
    still moving. A flat sheet at rest stays still with every start.
- **Network** (CPU; the strip and can rows mass-scaled ×10⁴ / ×10³, quasi-static):
  - elastic 3-point bending of a 1 mm steel strip deflects 1.20 / 1.27 / 1.33 × the plane-strain beam for 5 / 10 /
    20 mm strips (a narrow real plate: 1.10 ×);
  - pure bending at 30° / 45° to the grid (`sheet_laws.s1_off_axis`): below yield its stiffness is 1.375 / 1.5 ×
    the plate's; past yield (8 κ_y) it carries 1.80 / 1.94 × the plate's moment and keeps 0.92 of the plate's
    curvature. Both are pinned by `test_s1_off_axis_measures_the_hinge_networks_anisotropy`;
  - a 0.2 mm tinplate can wall pressed 8 mm by a fingertip keeps 1.07 / 0.96 / 0.68 mm on 32 x 8 / 48 x 14 /
    64 x 20 walls. With the stretch cap at 5 % instead of σ_y/E it keeps 0.14 mm (48 x 14).

## Why

- **Sag:** a metal sheet's bending stiffness D/dx⁴ dwarfs its mass per step (ρ t / dt²), and VBD's per-vertex
  Gauss-Seidel moves a long-wavelength bend a little per iteration. The inertial start puts a full step of gravity
  into every vertex's guess, and too few iterations leave much of it in the result. More iterations take it out.
- **Adaptive start:** it keeps gravity in a vertex's guess only where the vertex accelerated downward on the last
  step (Chen et al. 2024). That removes most of the sag (the 1 mm strip: 1,170 -> 42 µm). But a stiff sheet far
  from converged follows its guess, so gravity then acts one step late. A restoring force with a lag feeds energy in,
  and a dented sheet resting on two supports rocks itself off them.
- **Network**, a film model property (SC8's film; the J2 section only uses it):
  - The membrane runs at 1e-4 of the material for VBD, and the strain cap carries the stretch, so in-plane shear and
    twist are nearly free below the cap and rigid past it.
  - On the alternating right-triangle grid the axis hinges read κxx and κyy and the diagonals the twist κxy. With
    ke = D/l_h on all of them the energy is κxx² + κyy² + 4 κxy², where a plate's cylinders take 2 κxy².
  - Under a concentrated load the rows zig-zag cell by cell, a mesh-scale mode that only the diagonal hinges resist.
  - Each hinge yields on its own curvature: there is no plate plasticity (no biaxial interaction).
  - Halving the diagonals' section (tried as P3b, not landed) makes elastic cylinders exact at every angle
    (1.000), but makes the zig-zag cheaper: 3-point compliance 1.51-1.61 ×, and the strip's set no longer matches
    its hinges. Past yield it still carries 1.38 / 1.44 ×, since independent hinges sum to 1.5 M_p at 45°.

## Workaround

- Shape and dent laws: run mass-scaled with gravity off and a slow press, as S2 / S3 do (`sheet_dent.py`,
  `sheet_can.py`). Light hinge damping while the sheet yields, and heavy damping once nothing yields any more.
- A J2 film runs 20 iterations. More cut the sag further (40: 3.1 mm over the bar, 2x the cost); in engine-layer
  rigs pass `iterations=` to SolverFilm.
- Keep `alloy_initial_guess="adaptive"` off a J2 sheet that moves under gravity. The mass-scaled laws run with gravity
  off, where it changes nothing.
- Give a pressed metal sheet round supports (`add_cylinder`), not sharp box edges.
- Read `plastic_strain` on the sheet at rest, before the press: anything above 0 there is the sag, not the task.
  Count only what the press adds, and keep the mesh coarse (5 mm cells or larger): a finer one sags and yields more.
- Bend along the mesh grid where the bend line is known (a fold, a cylinder press): the network is exact there.
- Read a dent's depth as mesh-dependent: compare meshes before trusting a number.

## Planned fix

After the 1109 release. More iterations and Chebyshev buy factors of 1.3-1.6; the sag needs a solve that moves the
whole sheet's bend at once (a Newton or multigrid step on the bending, in SC8's film or VBD; WP2's
cloth-stiff-sheet-sag has the same need).
- J2 films run 20 iterations from the inertial start. An adaptive start without the lag (or one used only near
  convergence) is the VBD accelerators' owners' call (WP2, WP11). CLOTH_VBD keeps it for fabrics, which the drape
  needs, and since 24178a806 starts sheets with a bending length over 50 mm (PET, nylon, PP, cardboard) inertial;
  their sag is [cloth-stiff-sheet-sag.md](cloth-stiff-sheet-sag.md).
- A plate-like network needs hourglass control, or another bending discretization, in SC8's film. That is a film
  design change, not a J2 one.
