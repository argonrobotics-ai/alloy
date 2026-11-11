# A chain rod resting on an elastic rod sinks into it

## What is wrong
A rod on the chain route laid across a rod on the elastic-rod route (`rod_vbd`, for example a clamped wire) does not
rest on top of it. It sinks into the wire by up to the wire's radius. With a thin wire the overlap is deeper than
half the wire's thickness, so the health check flags PEN_FAIL, the world becomes INVALID and is frozen. The flag is
right: the motion is wrong.

## Number
CPU, a chain rod of radius 4 mm laid across a clamped copper wire:
- wire radius 1.5 mm: the rod's crest sits 1.5 mm low (the wire's whole radius), at `dt` 0.01 and 0.005 alike;
  PEN_WARN and PEN_FAIL latch at 0.285 s and `sim.check()["ok"]` is False;
- wire radius 3 mm: the crest sits 1.9 mm low, the deepest contact is 2.5 mm (84 % of the radius);
  `sim.check()["ok"]` stays True with a warning.

Two rods that never touch are not affected: each moves as it does alone
(`tests/api/test_rods_mixed.py`: 0.000 um for the clamped rod, 0.014 mm for the chain at rest).

## Why
Not pinned yet. The two rods step in different solvers (the chain in the rigid solver, the elastic rod in its own),
and their contact is carried by the strong coupler between them, not by either solver's own contact. A smaller `dt`
does not change the overlap, so it is not a step-size error.

## Workaround
- Put both rods on one route. Two chain rods touch as rigid bodies do: `add_rod(..., clamp="start", route="chain")`
  keeps a clamped rod on the chain. Two elastic rods share one solver.
- Keep a chain rod and an elastic rod apart, or accept the overlap with a thick elastic rod (3 mm radius: 1.9 mm),
  and read `sim.check()`.

## Planned fix
The contact between a chain rod and an elastic rod at the stiffness of the rods' own contacts, with a resting law
(crest height within a stated share of the thinner radius). After the release tag.
