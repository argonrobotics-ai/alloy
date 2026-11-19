---
owner: Alloy maintainers
---
# With surface tension on, less liquid leaves a tilted container

## What
- With `alloy.Scene(liquid_tension=True)`, a container tilted past its pouring angle lets less liquid over its lip
  in the same time than without tension: a half to a quarter as much, less at finer settings (about half at the
  default 4 mm particles, a third at 2 mm, under a quarter at 1.5 mm).
- The stream that does leave holds together, so a narrow receiver catches more of it.
- Liquid at rest, `add_pour` streams and containers that do not tilt are not affected.
- Fixed in 0.1.8: at `liquid_accuracy="ultra"` (1.5 mm particles) a few droplets used to fly off a tension pour and land
  outside a receiver that caught all of the pour without tension. Small pieces of liquid (droplets, a stream one or
  two particles wide) now feel a force between pairs of particles instead of the curvature force, and the receiver
  catches all of the pour.

## Number
A 50 mm beaker holding about 60 ml of water is tilted about its lip from 0 to 64 deg over 3.5 s, held for 0.5 s and
tilted back to 45 deg. Simulated on one L4. Liquid that left over the lip in 5.5 s, tension off / on:
- 4 mm particles: 6.3 / 3.2 ml (2.0x less with tension);
- 2 mm: 4.5 / 1.6 ml (2.8x less);
- 1.5 mm: 5.0 / 1.15 ml (4.3x less).

With tension the pour starts 0.7-1.2 deg later. Held at 64 deg for 4 s instead of 0.5 s: 10.6 / 9.2 / 9.9 ml left
without tension at 4 / 2 / 1.5 mm, and 6.8 / 5.3 / 4.3 ml with it. With tension the flow was still 0.7-0.8 ml/s at
the end of the hold.

The same pour into a 50 mm receiver 77 mm below the lip, the receiver moved by up to 0.5 mm between five runs: with
tension it caught 100 % of what left the beaker in every run at 2 mm and at 1.5 mm (and in one run at 1.5 mm with
1 mm glass walls). Without tension it caught 99.9-100 %.

## Why
- The liquid does not pin at the lip: with tension it keeps flowing through the 4 s hold, where a pinned edge
  would stop it.
- The layer over the lip is thinner than one particle spacing: 0.7 mm with tension and 1.4 mm without, at 2 mm
  particles. A flow that thin is under-resolved, and two things slow it.
  - The velocity smoothing that runs with tension. On its own, without the surface force, it takes about a quarter
    off what leaves in the 4 s hold at 2 mm.
  - The surface force.
- Finer particles do not close the gap: in the 4 s hold, tension lets out 1.6x less at 4 mm, 1.7x at 2 mm and 2.3x at
  1.5 mm.
- How much a real beaker lets out under the same tilt has not been measured.

## Workaround
- Leave tension off (the default) where it matters how fast liquid leaves a tilted container.
- Turn tension on where a thin stream must hold together, such as a pour into a narrow receiver.
- With tension on, tilt further or hold the tilt longer for the same amount.

## Planned fix
For the lip drain, nothing is scheduled. Surface tension stays off by default until all three hold: a tilted pour
with it lets out within 20 % of the amount without it (or this gap is closed), no detached particle gains energy from
it, and a 50 mm receiver catches at least 95 % at every setting.
