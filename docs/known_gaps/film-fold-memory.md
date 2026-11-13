# Known gap: paper and film keep only a flat fold made on one mesh line

## What
- The film route's fold memory lives on mesh edges. Each edge has a yield angle from its material card
  (`film.dead_fold_yield`: paper_80gsm 36 deg, film_hdpe 73 deg); bent past it, the edge's rest angle follows. The angle
  is per edge, whatever the cell size.
- A bend that spreads over several rows of edges keeps each edge under its yield angle, so the sheet springs back flat.
  A paper or HDPE sheet bent 90 deg over a plate edge does not keep the fold. A real sheet pressed over a sharp edge
  keeps part of it.
- A sheet folded flat about one mesh line keeps its fold, and library objects with creases (bags, cartons) fold on
  their crease lines.
- A 180 deg fold pressed against a plate goes INVALID (PEN_FAIL; film-thin-sheet-box-edge.md).

## Number
CPU, 1 world, a 40 x 30 mm sheet of 0.1 mm at 2.35 mm cells, half under a static plate, the other half folded up by a
kinematic paddle, held 0.3 s and let back (the audit's rig; re-measured on main 692add686, after the drive-timing
change 7690f7a84: every number here as before but the 180 deg row's):
- **90 deg over the plate edge.** The bend sits on three rows of edges: 46 / 25 / 9 deg for paper_80gsm, 38 / 29 / 10
  deg for film_hdpe. Paper: one row is 11 deg past its 36 deg and keeps that (rest angle 11.5 deg); the flap is back
  at 0.02 deg. HDPE: no edge reaches 73 deg; the flap is back at 0.01 deg. paper_kraft, film_ldpe and film_lldpe
  return to 0.00-0.08 deg as well (the audit's runs, CPU and NVIDIA L4).
- **Plate edge on a vertex row.** The bend becomes a loop over about nine rows, 31 deg at most; nothing yields.
- **170 deg over a 1.25 mm plate.** Edges reach 31-36 deg; nothing yields; the flap returns to 0.2 deg.
- **180 deg against the plate.** INVALID at 1.24 s for paper (1.26 s for HDPE), PEN_FAIL first read at 1.0 mm, 2.8 mm
  deep at its deepest (edges had yielded by then: rest angles to 66 deg for paper, 36 deg for HDPE).
- **135 deg arc** (the audit's run): paper_80gsm keeps 175 deg, film_hdpe opens to 0.4 deg (0.6 deg 1.5 s later).
- **The engine's own law** (`alloy.solvers.film.laws.fold_release`: a strip folded flat about one mesh line, let go,
  no gravity): HDPE 15 um opens to 79 deg against its 73 deg, paper_80gsm to 53.5 deg against its 36 deg. The test
  allows 15 deg and runs HDPE and LLDPE only, so paper's 17 deg miss was never tested.

## Why
The yield angle comes from a flat-fold measurement (the angle a pressed fold opens to) and is applied to each mesh
edge as an angle. A real sheet yields by curvature, so one sharp bend and the same bend spread over three cells are
the same event; here they are three edges, each under its limit. The result depends on where the fold falls on the
mesh and on the cell size. Paper's 17 deg in the flat-fold law is not diagnosed.

## Workaround
- Put a mesh line on the fold and fold the sheet flat about it, or use a library object with creases on its fold
  lines (the paper bags and cartons).
- Do not expect a bend over an edge or around a tool to leave a crease.
- Read `sim.check()` after a tight fold against a body.

## Planned fix
After the release: a yield by curvature (an edge's yield angle from the bend it shares with its neighbours, so a
fold yields the same on any mesh), and the flat-fold law run on the paper cards.
