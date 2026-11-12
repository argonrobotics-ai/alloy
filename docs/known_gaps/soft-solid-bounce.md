# Known gap: a body dropped on a soft solid bounces back far more than the materials' restitution says

## What is wrong
- A rigid body landing on a soft solid (a VBD tet mesh: pads, foams, silicone) bounces back almost as fast as it
  arrived. The soft solid's bulk returns the impact energy, and the card's restitution sets nothing on this path.
- The contact records between the body and the soft solid do not change this: scaling their damping 0.2x to 3x
  leaves the bounce the same. The dissipation that should set it is missing in the bulk (material damping), not at
  the interface.
- A body landing on a cloth lying on a support is different. There the records' damping does set the bounce (with
  the cloth's own contact with the support), and no measured reference exists to size it from: cotton's card
  restitution 0.0 is a model value.

## Number
CPU, Warp route, main cacec9764 (2026-11-08); measured with a development bounce probe script (not shipped). Each
landing body's record damping is scaled x1 (as built), x0.2 and x3.
- C3's 0.515 kg wood_beech platen dropped 1 mm onto its 40 x 40 x 10 mm tissue_fat_silicone pad on the ground: lands at
  0.154 m/s and rebounds at 0.137 m/s, e 0.89 (x1), 0.88 (x0.2), 0.88 (x3). The pair's card restitution (the lower
  of the two materials) is 0.2.
- A 2 kg steel_carbon ball dropped 5 mm onto an 11 x 11 cotton_woven cloth (0.05 kg) on the ground: lands at
  0.342 m/s, e 0.62 (x1), 0.76 (x0.2), 0.41 (x3). Cotton's card restitution is 0.0 (model), steel's 0.6.
- Resting loads are right: the same platen at rest reads stretch 0.9035-0.9040 against the analytic 0.9029 (known gap
  coupled-record-damping).

## Why
- The soft solid's elasticity has almost no material damping. An impact's strain energy comes back as rebound, and
  the records' damping only acts over the thin contact layer.
- A cloth on a support is one layer thick. Most of the dissipation is in the records against the body and in the
  cloth's own contact with the support, so the records' damping moves the bounce.

## Workaround
- For a resting load (a box on a pad, a weight on a cloth), none is needed: place the body on the soft solid, or let it
  settle (`sim.settle`), instead of dropping it.
- For a landing whose bounce matters, keep the landing pair rigid-on-rigid, where the card's restitution sets the
  bounce through MuJoCo.

## Planned fix
- Soft solids: a material damping term for VBD tet solids, sized from the card (a loss factor, or the
  restitution of a drop test). Its law: C3's 1 mm drop rebounds within 0.1 of a measured e. Not scheduled.
- Cloth landings: record damping sized from the pair only once a measured drop reference exists (a body on
  cotton over a hard support). The restitution-sized record damping of gaps-now #10a (slice B) was measured and
  dropped on 2026-11-08: the numbers above have no sound target to size it to.
