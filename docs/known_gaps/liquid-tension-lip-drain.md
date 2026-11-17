---
owner: Alloy maintainers
---
# With surface tension on, less liquid leaves a tilted container, and at ultra a few droplets fly off

## What
- With `alloy.Scene(liquid_tension=True)`, a container tilted past its pouring angle lets less liquid over its lip
  in the same time than without tension: a half to a quarter as much, less at finer settings (about half at the
  default 4 mm particles, a third at 2 mm, under a quarter at 1.5 mm).
- The stream that does leave holds together, so a narrow receiver catches more of it.
- At `liquid_accuracy="ultra"` (1.5 mm particles) a few droplets fly off a tension pour and land outside a receiver
  that catches all of the pour without tension.
- Liquid at rest, `add_pour` streams and containers that do not tilt are not affected.

## Number
A 50 mm beaker holding about 60 ml of water is tilted about its lip from 0 to 64 deg over 3.5 s, held for 0.5 s and
tilted back to 45 deg. Simulated on one L4. Liquid that left over the lip in 5.5 s, tension off / on:
- 4 mm particles: 6.5 / 3.3 ml (1.9x less with tension);
- 2 mm: 4.4 / 1.6 ml (2.8x less);
- 1.5 mm: 5.0 / 1.1 ml (4.5x less).

With tension the pour starts 0.3-0.5 deg later. Held at 64 deg for 4 s, at 2 mm, with a 58.7 ml fill: 8.6 ml left
without tension and 5.2 ml with it. With tension the flow was still 0.8 ml/s at the end of the hold.

Droplets at 1.5 mm, the same pour into a 50 mm receiver 77 mm below the lip:
- With tension the receiver caught 94.3 % of what left the beaker, 93.6 % with 1 mm glass walls. Without tension
  it caught 99.9 %.
- 19 of about 330 particles poured missed. Nothing was still in the air at the end, and none clung to the beaker.
- 12 of the 19 landed 37-118 mm from the receiver's axis (its outer wall is at 27 mm), 1 on its rim, and 6 were
  flung 370-410 mm, out of the scene.
- At 2 mm and 4 mm with tension the receiver caught 96.5 % and 100 %.

## Why
- The liquid does not pin at the lip: with tension it keeps flowing through the 4 s hold, where a pinned edge
  would stop it.
- The layer over the lip is thinner than one particle spacing: 0.6 mm with tension and 1.3 mm without, at 2 mm
  particles. A flow that thin is under-resolved, and two things slow it.
  - The velocity smoothing that runs with tension. On its own it takes the 4 s hold from 8.6 ml down to 6.6 ml.
  - The surface force.
- Finer particles do not close the gap. In the hold, with tension, 5.9 / 5.2 / 3.4 ml left at 4 / 2 / 1.5 mm;
  without it, 8.5 / 8.6 / 8.3 ml.
- How much a real beaker lets out under the same tilt has not been measured.
- Why droplets fly off at 1.5 mm is not known yet. A droplet flung 0.4 m from a 77 mm fall has gained energy, so it
  is treated as a fault in the model and is being investigated.

## Workaround
- Leave tension off (the default) where it matters how fast liquid leaves a tilted container.
- Turn tension on where a thin stream must hold together, such as a pour into a narrow receiver.
- With tension on, tilt further or hold the tilt longer for the same amount.
- With tension at `liquid_accuracy="ultra"`, expect a few percent of a pour to land outside the receiver. Where
  every drop counts, leave tension off (99.9-100 % caught) or use `"super"` (96.5 % with tension).

## Planned fix
The flying droplets are being investigated. For the lip drain, nothing is scheduled. Surface tension stays off by
default until all three hold: a tilted pour with it lets out within 20 % of the amount without it (or this gap is
closed), no detached particle gains energy from it, and a 50 mm receiver catches at least 95 % at every setting.
