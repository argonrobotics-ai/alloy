# A peg that rolls on its precision partner slowly gains speed

## What is wrong
A precision pair (a peg and the block it fits, `precision_partners` on their cards) is made for the fit: the peg in
its bore. A peg that misses the bore, tips over and rolls across the top of the block speeds up a little by itself
instead of slowing down. The world's health stays OK: the energy check flags a gain of 10 % of the scene's energy
scale, and this one stays under it over a few seconds.

The shipped `demo_peg_d12` no longer shows it: since 2026-11-10 every world of the demo starts inside the fit's
clearance and the peg seats. Any scene where a part rolls or slides for long on its precision partner can.

## Number
`demo_peg_d12` with its former start spread (the peg up to 0.5 mm off the axis, against 0.1 mm of radial clearance),
profile `precision`, one L4, in an energy sweep of 53 scenes:
- worlds where the peg missed the bore and rolled on the block: the peg's speed rose about 10 % in 1.7 s, and the
  world's energy rose by 1.1 to 5.8 % of the scene's energy scale;
- worlds where the peg seated in the bore: 0 %;
- the other scenes of the sweep that were measured: 0 %.

It has not been measured for other parts (a bolt rolling on its nut, a plug on its receptacle) or on the CPU.

## Why
Not found yet. The contacts of a precision pair come from the precision layer, built from the two features' exact
shapes, and the pair's plain contact is switched off. A contact point that travels round a rolling peg is the case
those rows were not written for: the fit's laws (clearance, lean, insertion) all hold the peg nearly still in its
bore.

## Workaround
- Start a precision fit inside its clearance, so the part seats (the demo does; a peg dropped 0.15 mm or more off the
  axis of a bore without a chamfer lands on the rim).
- For a part that is meant to roll or slide on the other, do not use the precision pair: place a copy of the card
  without `precision_partners`, and the two collide as plain shapes (a bored block then collides as its convex
  hull: its bore is closed).
- In a long run, compare the part's speed over time; do not rely on `sim.check()` alone for this.

## Planned fix
After the release candidate: reproduce it with a peg laid on the block, find the row that does the work, and add the
law that a rolling peg's energy never rises.
