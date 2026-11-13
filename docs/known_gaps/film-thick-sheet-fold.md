# Known gap: a paper or film sheet much thicker than its card is launched when folded

## What
- A paper or plastic-film sheet (route `cloth_film`) given about 3x or more of its card's thickness goes INVALID when
  it is folded. `add_cloth(..., material="paper_80gsm", thickness=0.001)` is a 1 mm sheet of 80 gsm paper, 10x
  the card's 0.10 mm. A kinematic paddle folds half of it over: the sheet is launched (SPEED_FAIL, 40-72 m/s) as the
  paddle nears vertical, or goes PEN_FAIL. film_ldpe at 1 mm does the same.
- At the card's own thickness the same fold works: paper_80gsm, film_hdpe and film_ldpe at 0.1 mm fold to 178 deg,
  are pressed, and stay shut after the paddle goes back (`sim.check()` ok).
- `add_cloth`'s thickness defaults to 2 mm, the cloth default: give a paper or film sheet its card's thickness.
- `cotton_woven` (route `cloth`) folds cleanly at 1 and 2 mm.

## Number
The fold: a 15 x 15 cm sheet (31 x 31 cells, 4.84 mm), half on a static plinth, half on a 4 mm kinematic paddle,
`sim.drive(..., angular_velocity=(0, 3.2, 0), pivot=<the midline>)` to 178 deg, scene dt 5 ms. CPU, main f9340c512,
1 world; paper_80gsm unless named:

| thickness | result |
|---|---|
| 0.1 mm (the card's) | folds to 178 deg, pressed 0.3 s, paddle back: the flap stays shut (1.1 deg open), `sim.check()` ok; film_hdpe and film_ldpe the same (1.0 deg) |
| 0.3 mm | INVALID at 0.93 s (paddle at 170 deg): CLOTH_STRETCH, 2.6 mm into the plinth |
| 1 mm | INVALID at 0.82 s (paddle at 94 deg): SPEED_FAIL, the sheet at 41.6 m/s |
| 1 mm, 40 VBD iterations (10 by default) | INVALID earlier, at 0.71 s (132 deg), 47 m/s |

- The jump is in one 1.25 ms film step, in VBD's solve. The fastest particle goes from 0.89 to 54.9 m/s, and 376 of
  1,024 particles pass 5 m/s. The strain projection after it lowers the peak (41.6 m/s), it does not cause it.
- From 71 deg on, VBD already moves free-edge particles at up to 4x the paddle's own speed there.
- With the projection off, the contact skin limit on, or the film's friction terms off, it still fails (0.78-0.86 s).
- With the opt-in `FilmConfig(limiter="stress")` it is not flagged, but the sheet thrashes (3-13 m/s every step, the
  fold angle jumping between 12 and 175 deg).
- It does not need the paddle to move. Driven to 60 or 70 deg at 3.2 rad/s and held still, it fails at 0.70 and
  0.745 s. Driven at 0.5 rad/s, holds at 30-75 deg pass and 85 deg fails.
- A 12 mm paddle and plinth do not help: paper 72 m/s, film_ldpe 48.5 m/s, both at about 0.84 s.
- Earlier trees: PyPI 0.1.0rc2 and rc6-1110 gave the same result, and an earlier main read PEN_FAIL 4-19 mm at
  0.54-1.39 s. Neither 1-2 mm, dt 5 or 2.5 ms, nor fold rates of 1.5-3.2 rad/s helped.

## Why
Not fully diagnosed. VBD's own solve diverges in a single film step, and the divergence grows with thickness, not
with fewer iterations. The likely mechanism: a sheet's bending stiffness grows as the cube of its thickness, while
the film route keeps VBD's membrane at 1e-4 of the material's stiffness (the strain limit holds the stretch instead).
So at 1 mm, a 1000x stiffer bend acts against a soft membrane in each vertex's local solve, and a sheet bent near the
fold, carrying its own weight sideways, gets steps that do not settle. The cloth route keeps its full membrane and
does not do this.

## Workaround
- Give a paper or film sheet its card's thickness (`paper_80gsm` 0.1 mm; the card's note says its caliper):
  `add_cloth(..., material="paper_80gsm", thickness=1e-4)`.
- For a thick, foldable sheet, use a cloth material, or build the folded part from creased cards (the library's
  paper bags and cartons fold on their crease lines).
- Always read `sim.check()`: the failure is flagged (SPEED_FAIL or PEN_FAIL), and the world is frozen, never silently
  wrong.

## Planned fix
Planned, with no release named:
- A build warning when a film sheet's thickness is well past its card's (the 0.3 mm threshold above is 3x).
- Then the solve itself: the membrane share that keeps VBD's local solves stable for a bending-stiff sheet, or the
  stress limiter, whose membrane is the film's own stiffness past the cap (it did not blow up here, but it did not
  settle either).
- docs/guide/deformables.md should say to pass the card's thickness for paper and film (that page is its owner's
  edit).
