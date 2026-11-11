# Known gap: a very thin film draped over a sharp box edge goes INVALID (the fix is opt-in for this release)

## What
- A 0.05 mm film (route `cloth_film`) draped over a static box's edge pushes particles into the box until health's
  depth limit fires (PEN_FAIL), and the world goes INVALID. The library's `film_ldpe_sheet` (0.3 m, 50 um LDPE) did
  it, as does `add_cloth(..., thickness=5e-5, material="film_ldpe")`. Found by OSS-17 (docs/guide/deformables.md
  warns about it).
- The fix is in the film plug-in, `FilmConfig.contact_skin_entry` (`alloy/solvers/film/skin.py`), and is off by
  default in this release: with it the same scenes stay valid with no penetration flag
  (`tests/solvers/film/test_film_skin.py`), but one Bonds law (a label rolled onto a tube at 1.25 mm cells) reads no
  roller load with it on. It goes on by default once that law re-passes.
- The fix covers static shapes only. A film on a dynamic or kinematic body (a free box, a tray moved
  with `sim.drive`, a robot's pads) keeps VBD's overshoot below, since those contact planes move with the body.

## Number
- Off, the release default (CPU, 1 world: a static 0.1 m cube, `film_ldpe_sheet` 2 mm over its top, scene dt 0.01): PEN_FAIL at
  t 0.11 s, 0.52 mm (limit 0.5 mm = half the particle's 1 mm contact skin). A smaller scene dt did not help: the
  film steps at its own 1.25 ms.
- On: 0.3 s with no penetration flag, deepest contact 0.05 mm; the world is DEGRADED by the strain projection's
  UNCONVERGED flag only (film-projection-force.md). The engine rig at the API's gains (0.15 m sheet, 7 mm edges):
  0.22 mm deep without the limit; with it 0.001 mm after VBD and 0.05 mm after the projection (0.1 and 0.2 s).
- NVIDIA L4, 8 worlds, 0.3 s: the library sheet and a 50 um `add_cloth` over the cube, 0.001 mm deep and
  `sim.check()` ok (before: INVALID, PEN_FAIL 0.54 mm and 0.45 mm).
- Cost: a drape step +4 % on the CPU, +10 % on an L4 without graph capture (the film suite's wall: within its run
  to run spread).

## Why
Two stages each pushed contact vertices into the box (measured per film step, the contact records' own planes):
- **VBD's step from outside the skin.** Newton's VBD gives a particle-shape contact no Hessian while the particle is
  outside its skin. A film vertex just outside took its Newton step with the membrane's and inertia's small Hessian
  alone, overshot into the shape, and the contact's full stiffness (ke 2e6 N/m on a 2-9 mg particle) threw it back in
  the next iteration. The step ended wherever the last iteration left it: 0.23 mm into the skin at 10 iterations,
  0.16 mm at 200, the depth non-monotone in the count.
- **The strain projection.** It moves vertices after VBD and ignored contacts that carry no force yet, so it pulled
  the taut sheet over the edge: VBD left the library sheet 0.001 mm deep (with the fix above), the projection took it
  to 0.52 mm.

The fix limits both, for film particles and static shapes: in each VBD iteration a vertex outside the skin may enter
it by at most 0.2 % of its contact radius (1 um), so the next iteration has the contact in its Hessian; inside the
skin it is VBD's, so the limit holds no load. After the projection a vertex VBD left lightly touching (within 2 um of
the skin) ends at most 10 % of the skin deep (0.05 mm); a loaded vertex is left as before. The excess goes back out
along the record's normal, at the particle's deepest record only (two coplanar boxes would push it twice).
- The guard's 10 %: at 1 um it narrowed a roller's contact patch on a tube by 6 % (the Bonds C-1 law), since a
  lightly pressed patch edge is the projection's to set.
- The entry: 5 % (25 um) left ~50 N on a vertex at the box's gains, and the friction on it tore the library sheet at
  the cube's corner (55 % stretch).
- A first version held vertices it had placed in the skin there, so they could not take load through contact depth.
  Eight Bonds laws read zero wet-out pressure under a roller; fixed before merge.

## Workaround
- At static shapes, engine layers: `FilmConfig(contact_skin_entry=0.002)` for the SolverFilm entry. Through
  `alloy.build`: a film of 0.1 mm or more at sharp static edges, or a rounded edge (a capsule or cylinder along it, or
  a mesh with a bevel).
- Check `sim.check()`: an INVALID world is flagged and frozen, never silently wrong.
- On a dynamic or kinematic body: a film of 0.1 mm or more at its sharp edges, or a rounded edge (a capsule or
  cylinder along it, or a mesh with a bevel); grip.py keeps the projection from deepening only the contacts that
  already carry force.

## Planned fix
SC8: on by default after the release, once the Bonds C-1 tube row at 1.25 mm cells re-passes with it (the other
Bonds, film, dent and SOFT-1 laws re-pass already). Then dynamic and kinematic bodies, the same two limits against planes that move with the body (Newton's rigid-soft
DAT does that for its own truncation); not yet scheduled. docs/guide/deformables.md's warning should now name moving
bodies only (that page is its owner's edit).
