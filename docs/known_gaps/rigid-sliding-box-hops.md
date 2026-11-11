# A box sliding at metres per second hops and rocks

## What is wrong
A box on MuJoCo's soft contact that slides fast lifts off its support and rocks, more the faster it slides. Its
slide keeps the Coulomb rate. A cube pushed along the floor to about 2.5 m/s rises a few millimetres and pitches by
1-2 degrees peak to peak (a small cube by up to 7 degrees), where a rigid one would stay flat. A low, wide box
slides flat. Slow slides (the light-box law `tests/contact/test_light_sliding_tip.py`, up to 1.6 m/s) stay within
0.1 mm.

## Number
Cubes (1,000 kg/m3) on the ground (mu 0.5), settled, then pushed at their centre by 1.5 mu m g, so they reach about
2.5 m/s in 1 s. CPU, main 132d4450f (free bodies on rk4), the same to the last digit as release-1109; measured with
WP3's SPEED-S1 development probe (2026-11-09, not shipped):

| run | rise by 1 s (2.5 m/s) | pitch, peak to peak | slide / 0.5 mu g |
|---|---|---|---|
| 0.3 m cube, rigid step "auto" (5 ms here) | 3.4 mm | 1.2 deg | 0.998 |
| 0.3 m cube, step pinned at 1.25 / 10 ms | 9.0 / 1.9 mm | 1.4 / 1.5 deg | 0.995 / 1.000 |
| 0.1 m cube, "auto" | 4.1 mm | 2.4 deg | 0.998 |
| 0.05 m cube, "auto" | 2.2 mm | 7.4 deg | 0.999 |
| 0.3 m cube, MuJoCo C alone, stock solref (0.02, 1), elliptic, 5 ms | 16.6 mm | 1-2 deg | - |
| 0.3 m cube, MuJoCo C alone, stock, pyramidal, 5 ms | 94 mm at 4 m/s | 20 deg | - |

A 40 x 40 x 10 cm box pushed the same way slides flat: 0.24 deg peak to peak, 0.25 mm.

## Why
MuJoCo's soft friction rows ask for a reference acceleration of about -b v_slip, with b = 2 / (dmax tau). It grows
with the sliding speed, and the friction cone couples it into the normal rows, so a sliding face is pushed off its
support step by step ([contact-light-sliding-tip.md](contact-light-sliding-tip.md) has the same mechanism at a short
time constant). A box also feels the friction's moment about its centre, mu N h / 2: the leading edge digs in and
the box rocks on its corners, and a small one rocks more. Raw MuJoCo C does it at its stock gains, on both cones and
at 1.25 and 5 ms; stiffer, less damped contacts are worse (solref 0.01, 0.7 tumbled it).

## Workaround
- Slide fast things on wide bases, or slowly: the 0.3 m cube rose 0.2 mm by 1 m/s.
- A finer rigid step does not help: at 1.25 ms the 0.3 m cube rose 9 mm, against 3.4 mm at "auto"'s 5 ms.
- Read the slide's position and speed, which keep the Coulomb rate, not its height.

## Planned fix
None scheduled. Candidate: a longer friction time constant (solreffriction). It holds a light sliding box flat
([contact-light-sliding-tip.md](contact-light-sliding-tip.md)), but stiction creep grows with it, so a law for fast
slides comes first.
