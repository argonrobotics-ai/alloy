# Liquid beside a paste or powder: the paste is a solid wall to the liquid (no wetting, no percolation)

Status: open, by design for now. A liquid (SPH) and a paste or powder (MPM) in one scene step together since
the liquid-paste coupling change, but they meet only at the paste's surface.

## What

Since the liquid-paste coupling change an SPH liquid and an MPM paste or powder step in one scene; before, the build
refused them. The paste's particles are a moving wall of the liquid, and the liquid's pressure pushes back on them, so
a paste block floats, sinks and gets pushed around as it should. A whisk or spoon stirs both.

What is not modelled is everything that happens inside the paste or powder:

- liquid does not soak into a powder or seep through its pores;
- a powder does not wet, dissolve, hydrate into a batter or form lumps;
- nothing sticks at the contact line (no contact angle between the liquid and the paste);
- the liquid slips along the paste's surface, as it does along every wall in a routed scene: SPH wall viscosity
  (no-slip) is off by default, so a paste blob in a viscous liquid feels its pressure but not its skin friction.

A flour heap in milk is therefore a heap of dry grains that the milk flows around. It is also squeezed by the full
liquid pressure, where a real wet powder carries only its buoyant weight (Terzaghi's effective stress). Mixing
spreads the two materials through each other, at the size of a few particles; it does not blend them into one.

Three cases are refused at build, with the fix named:
- a second liquid on the MPM route (`liquid_mpm`) beside an SPH liquid;
- a liquid that starts inside the paste or powder;
- a cloth or soft body in the same scene (as before).

## Number

| check | result | how |
|---|---|---|
| Archimedes, 3 cm cube, rho 600 | submerged 0.606-0.611 against 0.602 | one L4, 4 mm liquid spacing, 2 s |
| Archimedes, 2.4 cm slab, rho 600 | draft 1.07-1.08 x rho_s / rho_f H (under 2 particles) | CPU law; L4, graphs |
| fluid force on a floating block | 1.003 x its weight (cube); 1.02-1.03 x (slab, bobbing) | same runs |
| paste bed under poured water | carries the column, rho_f g V, within 5 % | CPU law |
| momentum through the interface | exact to float32; a slab's drop within 0.65 % of its weight | CPU laws |
| a world beside another | bitwise the same whatever the other world's paste does | CPU law; one L4 |
| liquid inside the paste | 0 particles | every law run |
| sphere settling, 0.9 Pa s, no-slip | paste 3 mm/s, rigid 3.5 mm/s, Stokes 14.7 mm/s | L4, 2 mm spacing |
| passes per exchange | 2.48 mean (4.12 before the L2 gate), cap 8; 8 beside a robot | CPU, 1.6 cm cube |
| cost, 4000 liquid + 1728 paste particles | about 1.2 s per 10 ms period on an L4, 30 s on the CPU | 1 world |
| cost, slab scene, 64 worlds | 2.7 s per 10 ms period on an L4 | 64 worlds |

- A floating cube between 0.21 and 0.79 of the liquid's density tips onto an edge, as a real one does. Use a flat
  slab, or a cube outside that range, to see it float upright.
- The interface is resolved at the liquid's spacing: 4 mm by default, 2 mm with `liquid_accuracy="super"`.
- The settling sphere is slow because of the liquid's own viscous drag at 2 mm spacing: a rigid sphere through the
  same liquid is as slow. The paste's particle wall drags like a rigid wall.
- A robot arm beside the paste and the liquid steps (the paste follows the arm one exchange behind, the liquid and the
  paste converge together), but every exchange takes the full 8 passes, about 3x the coupling cost without the arm.
- Heat between the liquid and the paste (both are particles to the heat model) is not checked in a coupled scene.

## Why

- The liquid sees the paste as a boundary built from its particles (`alloy.solvers.sph.particle_walls`): at each
  liquid particle, the paste's local surface and velocity. It has no model of liquid between the grains.
- No-slip needs the SPH wall viscosity (``boundary_viscosity``), which the router does not set for any wall; the
  particle wall reads it like any wall when it is set.
- The two solvers run separately and meet every exchange: the paste moves, then the liquid, then the liquid's force
  goes back to the paste. Wetting and percolation need the two to share a grid (a two-phase mixture), and hydration
  needs mass to move from one solver to the other. Neither exists yet.

## Workaround

- For something floating, sinking or being pushed through a liquid (a dumpling, dough in water, a paste blob), use it
  as is. Give the paste a few liquid particles across: about 1.5 cm or more at default accuracy.
- For a batter or a dissolved powder, use one material for the result. A batter card (`batter_bread_wheat`,
  `batter_cake_quinoa`) mixed with a spoon behaves like the batter, not like flour plus milk.
- Pour the liquid onto the paste or powder (`Scene.add_pour`) instead of starting it inside.

## Planned fix

- Hydration (after the coupling's laws pass): a powder particle takes up the liquid it touches and
  turns into batter, with mass moving exactly from the liquid to the powder. Lumps follow from wetting only the
  exposed surface. Sized at about 4-5 days.
- Percolation and hindered settling need the shared grid (a two-phase mixture in MPM). Not planned before launch.
