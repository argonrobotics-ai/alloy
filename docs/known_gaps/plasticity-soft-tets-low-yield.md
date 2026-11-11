# A soft body that yields can yield spuriously at rest when its yield strain is small (under about 10 %)

Status: open, accepted for launch (coordinator ruling 2026-11-09). The build warns. A punch dent in a metal-like
solid is not supported on this route. Since 2026-11-10 such a solid can run on MPM instead, opt-in (Workaround),
where it yields at its yield stress but reads a punch's force low
([plasticity-mpm-solid-punch-force.md](plasticity-mpm-solid-punch-force.md)).

## What is wrong

A soft body made of a `plastic.model: j2` card (the `soft_vbd` route, VBD tets) should yield only where its stress
passes σ_y. When σ_y / E is small, a body resting on the ground under gravity also yields where the stress is far
below σ_y. It then slumps and creeps.

The cause is VBD's unconverged solve. At the start, and every step at rest, the solver leaves the lower layers more
compressed than the static solution. The J2 return reads that strain as real and keeps it as plastic flow. With
gravity off, nothing yields.

It applies to every solid (tet) body of a j2 card whose σ_y / E is under about 10 %, resting or pressed: metal-like
solids (σ_y / E about 0.1-0.3 %) and most plastics. Sheets and cans of metal cards (`add_cloth`, the `cloth_film`
route) are a different route and are not affected.

## Number

CPU, blocks with a 30 x 30 mm footprint (E 3 MPa, ν 0.3, 1,000 kg/m³, perfectly plastic) resting on the ground,
default mesh (about 3,000 tets), default VBD iterations (10). "Top" is the top face's height above the ground, which
includes the 0.66-0.75 mm contact gap under the bottom face; the elastic 30 mm block's top sits at 30.62-30.75 mm.

30 mm tall, 0.4 s; its self-weight stress is about 300 Pa (strain 0.01 %):

| σ_y / E | σ_y | tets yielded | largest plastic strain | top after 0.4 s |
|---|---|---|---|---|
| 0.33 % | 10 kPa | 100 % | 23 % and growing | 28.07 mm, still sinking |
| 1 % | 30 kPa | 78 % | 7.8 %, still growing | not measured |
| 3.3 % | 100 kPa | 28 % (the bottom layers) | 3 %, then stops | 30.54 mm, steady |
| 10 % | 300 kPa | none | 0 | not measured (no plastic strain) |

The 1 %, 3.3 % and 10 % rows were measured on 2026-11-08 on the mesh before that day's fix (4,374 tets).

At σ_y / E 0.33 %, changing where each step's solve starts, or adding iterations, shrinks the spurious strain but does
not remove it. 60 mm tall, 0.5 s; true self-weight strain 0.02 %:

| solve (probe settings, not public options) | tets yielded | largest plastic strain | wall time per simulated s |
|---|---|---|---|
| adaptive start (a resting body is a fixed point) | 86 % | 9.7 %, growing | 37 s |
| adaptive start, plasticity off for the first 0.2 s | 69 % | 1.7 %, growing | 36 s |
| adaptive start, 40 iterations, plasticity off for the first 0.1 s | 58 % | 0.62 %, then stops | 142 s |
| the same with Chebyshev acceleration (ρ 0.8) | 27 % | 0.55 %, then stops | 143 s |

The 30 mm block with the adaptive start and 0.2 s without plasticity does not yield; the 60 mm block does, so no
start rule carries. The spurious strain grows with the load (about ρ g h / E, times what the solve leaves unconverged:
25-150x here).

## Why

- J2 is exact only on a converged elastic state: it turns any strain above yield into permanent strain.
- VBD solves a soft body by per-vertex sweeps. A tall body under gravity needs many sweeps before the ground's
  support reaches its top, so the lower layers are over-compressed meanwhile, and a tall body's slow mode keeps
  moving at rest.
- With a small yield strain, that excess is enough to yield: even at 4x the iterations it is 25x the true strain.

## Workaround

- Metal dents: use a sheet or a can of a metal card (`add_cloth(..., material="aluminium_6061_t6", thickness=...)`,
  `docs/guide/deformables.md`, Metal sheets). Those dent, spring back below yield and keep a dent past it.
- A soft solid that yields: keep its yield strain well above its weight strain. For the 30 mm block, 10 % or more
  (σ_y ≥ 0.1 E). The build warns below 10 %, naming this page; a taller or heavier body may need more.
- A solid with a small yield strain, opt-in: give its object card the route override `paste_mpm`
  (`scene.add_box(..., card_extra={"route": {"route": "paste_mpm", "reason": "j2 solid on MPM"}})`). It
  then steps on implicit MPM as a von Mises solid and does not yield at rest; a block squeezed between plates
  yields within 1 % of σ_y. Cards with a hardening modulus are refused (of the shipped metals `brass_c360`,
  `steel_zinc_plated` and `steel_tinplate` have none), and a punch's force reads low there
  (plasticity-mpm-solid-punch-force.md).
- Read the plastic strain on the engine (`TetPlastic.ebar`) to check that a resting body has not yielded. The API has
  no soft-body plastic-strain key yet.

## Planned fix

After launch (ruling 2026-11-09): solid j2 cards with σ_y / E under 10 % route to implicit MPM by themselves, once
the punch laws pass there (the open part: [plasticity-mpm-solid-punch-force.md](plasticity-mpm-solid-punch-force.md)).
The route is opt-in until then, and the VBD tet route keeps the warning.
