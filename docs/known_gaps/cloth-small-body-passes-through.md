# A rigid body smaller than a cloth cell can pass through the sheet

## What is wrong
- A rigid ball whose radius is under about 0.7 of the sheet's cell size can go through a cloth or film sheet
  (`add_cloth`, routes `cloth_vbd` and `cloth_film`) without touching it. The sheet stays where it was.
- It happens with a kinematic ball pushed down onto the sheet. A free ball fell through in the audit; see Number.
- Nothing flags it: `sim.health()` reads OK (or DEGRADED with UNCONVERGED on the aluminium sheet) and
  `sim.check()["ok"]` is True.
- It depends on where the ball meets the grid. A ball right over a vertex pressed the sheet at every radius tried
  (down to 0.2 of the cell). A ball over the middle of a cell passed through below about 0.65 of the cell.
- The build warns when a rigid box, sphere, cylinder or capsule is under 2 cells across (a ball with R / cell under
  1.0), naming the sheet, the bodies and the cell size that holds the smallest. It warns by size, wherever the
  body is, also when it never comes near the sheet. Bodies from a mesh and robot links are not checked.

## Number
The scene: two static cylinders (radius 5 mm, length 0.12 m, axes along y, at x = -0.02 and +0.02 m, tops at
z = 0.05 m); a 6 cm x 6 cm sheet resting on them; a kinematic ball 0.5 mm above the sheet. Run 0.3 s, then
`sim.drive("tip", velocity=(0, 0, -0.05), duration=0.3)` and run 0.4 s: the ball goes down 15 mm. CPU, one world,
`dt` 0.01 s, thickness 0.25 mm. `resolution` counts cells, not vertices: 6, 13 and 24 give cells of 10.00, 4.62 and
2.50 mm (the median rest edge of the built grid). A ball that ends with its bottom within 2 mm of the sheet's centre
"pressed" it; one that ends more than 3 mm below the centre, with the sheet where it was, "passed through".

Ball over the middle of a cell (centre z in mm, before to after):

| sheet | cell (mm) | R (mm) | R / cell | sheet centre z | ball bottom z | verdict |
|---|---|---|---|---|---|---|
| `cotton_woven` | 4.62 | 2 | 0.43 | 46.4 to 46.7 | 36.0 | passed through |
| `cotton_woven` | 4.62 | 3 | 0.65 | 46.4 to 46.7 | 36.0 | passed through |
| `cotton_woven` | 4.62 | 3.5 | 0.76 | 46.4 to 37.9 | 36.0 | pressed |
| `cotton_woven` | 4.62 | 4 | 0.87 | 46.4 to 37.4 | 36.0 | pressed |
| `cotton_woven` | 4.62 | 5 | 1.08 | 46.4 to 37.0 | 36.0 | pressed |
| `aluminium_6061_t6` (`cloth_film`) | 4.62 | 3 | 0.65 | 48.7 to 48.8 | 36.0 | passed through |
| `aluminium_6061_t6` (`cloth_film`) | 4.62 | 4 | 0.87 | 48.7 to 37.9 | 36.0 | pressed |
| `cotton_woven` | 10.00 | 3 | 0.30 | 48.5 to 48.4 | 36.0 | passed through |
| `cotton_woven` | 10.00 | 4 | 0.40 | 48.5 to 48.4 | 36.0 | passed through |
| `cotton_woven` | 10.00 | 5 | 0.50 | 48.5 to 48.4 | 36.0 | passed through |
| `cotton_woven` | 10.00 | 6 | 0.60 | 48.5 to 47.7 | 36.1 | passed through |
| `cotton_woven` | 10.00 | 7 | 0.70 | 48.5 to 40.7 | 36.0 | partly (centre 4.7 mm above the ball) |
| `cotton_woven` | 10.00 | 8 | 0.80 | 48.5 to 37.8 | 36.1 | pressed |
| `cotton_woven` | 2.50 | 2 | 0.80 | 42.3 to 36.7 | 36.0 | pressed |
| `cotton_woven` | 2.50 | 3 | 1.20 | 42.3 to 36.3 | 36.0 | pressed |

Ball over a vertex: every run pressed the sheet (sheet centre within 1 mm of the ball's bottom).

| sheet | cell (mm) | R (mm) | R / cell |
|---|---|---|---|
| `cotton_woven` | 10.00 | 2, 3, 4, 5, 8 | 0.20, 0.30, 0.40, 0.50, 0.80 |
| `cotton_woven` | 4.62 | 2, 3, 4 | 0.43, 0.65, 0.87 |
| `cotton_woven` | 2.50 | 2, 3 | 0.80, 1.20 |

- **Mid-cell threshold:** the smallest R / cell that pressed is 0.76 at 4.62 mm cells (0.87 for the aluminium film)
  and 0.80 at 10 mm cells (0.70 was partial). The largest that passed through is 0.65 at 4.62 mm and 0.60 at 10 mm.
  At 2.50 mm cells the smallest ball tried (0.80) pressed, so no pass-through is measured there.
- **On-vertex threshold:** none found. The smallest ratio tried, 0.20, pressed.
- Every run here has `sim.check()["ok"]` True. Health was OK, except UNCONVERGED (DEGRADED) on
  both aluminium runs.
- The sheet started 0.3 mm above the cylinders, and the ball 0.5 mm above the sheet.
- **Not measured:** the default 2 mm thickness (in this layout the 2 mm cotton sheet fell off the supports to the
  ground before the drive, so that case is void); aluminium over a vertex; a free ball (below).
- **Free ball:** a free 2 g ball (radius 3 mm) on the 4.62 mm sheet did not finish its 1.0 s in 15 minutes on the CPU
  and was stopped, so it is not measured here. The audit saw a free 2 g ball of that size end on the ground, with the
  sheet unmoved (aluminium sheet, ball over the middle of a cell).

## Why
- Cloth and film touch rigid shapes at their vertices only. Each vertex is a small sphere of half the sheet's
  thickness.
- There is no contact between a rigid shape and the inside of a cloth triangle. A body that fits between the vertices
  meets nothing.
- The largest ball that fits between the four vertices of a square cell has a radius of cell / sqrt(2) less half
  the sheet's thickness, 0.71 of the cell for a thin sheet. That is the measured edge: 0.65 passed, 0.76 pressed.
- A ball over a vertex meets that vertex and presses the sheet, at every size tried.

## Workaround
- A finer `resolution`, so that the cells are smaller than the body. The build warning's rule: cells at most the
  ball's radius, half the body's smallest extent. In the rows above R / cell of 0.76 to 0.87 already pressed over a
  cell middle, the worst case; only two cell sizes above 2.5 mm were scanned, so the rule keeps a margin.
- The cost grows with the particle count: `resolution` r gives (r + 1)^2 vertices. Check the step time with
  `alloy.estimate` before a long run.
- A larger tool tip, if the task allows it.
- Do not trust a small body on cloth with no check: read the body's z and the sheet's z after a short run.

## Planned fix
- Contact between a rigid shape and a cloth triangle, so that a body between the vertices still meets the sheet.
  Not scheduled.
