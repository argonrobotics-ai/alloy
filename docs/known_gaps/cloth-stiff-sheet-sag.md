# A stiff sheet on the VBD cloth route sags far too deep

## What is wrong
- A sheet of a stiff material on the cloth route (`add_cloth` with PET, nylon, polypropylene, corrugated cardboard, or
  a metal without a yield law: bending length over 5 cm) is far from converged at the default 40 VBD iterations. It
  sags several times too deep between supports, and a thin one can slide off round supports into the gap.
- Since 2026-11-08 such a sheet starts each step from Newton's inertial guess (`alloy.scene.stiff_sheet`). The
  fabrics' adaptive start gained energy on it: it climbed and never came to rest. It now loses energy and comes to
  rest, but it still sags too far.
- `sim.check()` and `sim.health()` read OK either way.

## Number
Sheets dropped 5 mm onto two 1 cm round supports 10 cm apart, a 20 x 10 cm sheet, CPU, 3 s:

| sheet | plate theory sag | inertial start (shipped), 40 it | adaptive start (before), 40 it |
|---|---|---|---|
| corrugated cardboard, 3 mm | under 0.1 mm | rests, centroid 2.6 mm low, lowest point 12 mm below the support tops | 0.45 m/s at 3 s, walks |
| PET, 0.5 mm | about 0.3 mm | sinks into the gap (centroid 17 mm low), still sliding | climbs 28 mm above the supports, never rests |

At 400 iterations the PET sheet from the inertial guess nearly rests (0.013 m/s) with its centroid still 8.5 mm low.
From the adaptive start it still moves at 0.37 m/s and walks 6 mm, so the pump is not only an effect of too few
iterations. A dent-and-release rig (80 x 30 mm, round supports 40 mm apart) gives the same picture: about 18 mm
mid-span sag for 0.5 mm polycarbonate and PET at every 40-iteration setting.

## Why
VBD runs a fixed number of iterations per step. A stiff sheet's bending converges slowly (its stiffness is large
against the inertia term m/dt²), so after 40 iterations the step result stays close to where the iterations started.
From the inertial guess that is a step with all of gravity and too little of the bending, so the sheet sags. The
fabrics' adaptive start keeps only the share of gravity each particle realised over its last step: dropped on the
up-swing, kept on the down-swing, which on an unconverged stiff sheet feeds energy in. The library's fabrics
(bending length 7-20 mm) converge at 40 iterations and need the adaptive start for their drapes.

## Workaround
- A stiff part that only needs to bend a little: make it a rigid box with `rigid=True` (without it a box of these
  materials goes to the cloth route too), at least 1.25 mm thick (the thinnest box the build accepts, at any `dt`) and
  with the sheet's own mass. The PET sheet above:
  `scene.add_box("sheet", size=(0.2, 0.1, 0.00125), pos=(0, 0, 0.055625), material="pet", mass=0.0135, rigid=True)`.
  On the rig above (CPU) it comes to rest on the support tops, centred, with `sim.check()` ok, also when dropped from
  10 cm; it does not bend at all. A thinner box raises `SceneError` naming that limit: two thin rigid parts can pass
  through each other (a 0.5 mm plate dropped 5 mm onto a 0.5 mm shelf fell through it).
- A metal sheet that dents: its yield law sends it to the film route (`plastic.model: j2`), which handles stiff sheets.
- If the sheet's shape matters, raise the iteration count. The sag falls with iterations; check one case against
  plate theory first.

## Planned fix
None before launch. A stiff sheet needs a solve that converges its bending: more iterations scaled to the sheet's
stiffness, or the film route for stiff sheets. A velocity-gated adaptive start (a development branch) took
out the climb on its rig, but undid the drape's convergence (L5 drape: 4.5 % strain against 0.36 %), so it did not
ship.
