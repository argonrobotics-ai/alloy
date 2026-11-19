---
owner: Alloy maintainers
---

# Some pairs of materials do not build in one scene

## What is wrong
One scene can hold robots, rigid bodies, cloth, liquids, sand, pastes, soft bodies and cables, but not every pair of
them. `alloy.build` refuses the pairs below with a `BuildError` (`UnsupportedPairError`) that names the two objects,
their routes and the nearest setup that does build. A pair is refused whether or not the two objects ever touch.

| Pair | What would happen | Nearest setup that builds (as the error names it) |
|---|---|---|
| Sand or paste with fabric cloth | the two solvers cannot exchange contacts yet | make the grains or paste rigid (`rigid=True`); or a paper or foil sheet, which runs beside sand or paste, slowly (8.3 s per 10 ms control step on the CPU) |
| Sand or paste with a soft body | the same | `rigid=True` on one of them; there is no other way |
| Sand or paste with a clamped cable (an elastic rod) | the same | make the grains or paste rigid; or keep the cable on the chain route (an unclamped rod of at most 40 segments with an elastic material, or `clamp="start", route="chain"`) |
| A soft body with a paper, film or foil sheet | the same | `rigid=True` on the soft body; or a woven material such as `cotton_woven` for the sheet, which does not keep a fold as paper does |
| Fabric cloth with a paper, film or foil sheet | the two kinds of sheet do not touch, so the cloth would fall through | a woven material for both sheets; or the two sheets in separate scenes |
| A liquid with a second liquid on the paste solver (`liquid_mpm`) | the second liquid would be a solid wall to the first, so they would never mix | both liquids on the default liquid route |
| A clamped cable beside a paper, film or foil sheet, in some scenes | the two solvers cannot exchange contacts yet | the cable on the chain route; or a woven material for the sheet |

Two more refusals are about where a liquid starts, not about the pair: a liquid that starts inside a paste or powder
(start it beside the material or pour it on top: [liquid beside paste](sph-mpm-mixing.md)), and a liquid that starts
inside a free cloth or soft container (make the container static, or pour the liquid in with `add_pour`).

"Sand or paste" here is every material on the MPM solver: grains, pastes, and liquids routed there. "Fabric cloth"
is a sheet on the cloth solver (`sim.routes()`: `cloth_vbd`); paper, plastic film and metal foil sheets are on the
film route (`cloth_film`).

## Number
There is no size to measure: the build stops before the first step. Three of the refusals were checked on the CPU
on 2026-11-18 (a 0.1.8 pre-release build), each a `BuildError` whose text starts `UnsupportedPairError`:

- a cotton sheet and a block of `sand_silica_dry`;
- a cotton sheet and a `paper_80gsm` sheet;
- a block of `sand_silica_dry` and a cable with `clamp="start"`.

## Why
Each kind of material steps in its own solver, and two solvers share a scene by exchanging contact forces. The cloth
and soft-body solver cannot yet take contacts from the sand and paste solver, or from the film route, over a whole
surface: before the refusal was added, such a scene built and then stopped at its first step with a solver error and
no fix. Fabric cloth and a film sheet built and ran, but did not touch, and the cloth fell through the paper.

## Workaround
Do what the error names (the table's last column); `e.fix` carries it. Where only one object matters as a
deformable, make the other rigid with `rigid=True`. Pairs that do share a scene: a robot or rigid bodies with any one
of these materials; fabric cloth with a liquid; fabric cloth with a soft body; a liquid with sand or paste (the
paste is a wall to the liquid); a cable on the chain route with sand or paste.

## Planned fix
Not scheduled. Each pair needs the two solvers to exchange contacts over a surface, which is engine work per pair.
Until then the refusal stays, so a scene that builds does not stop at its first step for this reason.
