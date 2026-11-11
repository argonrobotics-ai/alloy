# A body on `collision="sdf"` or `"mesh"` contacts never fully rests, and under 40 g or from 40 kg it fails

## What is wrong
A scene with an `add_mesh(..., collision="sdf")` or `"mesh"` shape is collided by Newton's contact pipeline instead of
MuJoCo's own (`alloy.build`'s `contacts="auto"`; `contacts="newton"` asks for it by hand). A rigid body in such a
scene behaves in one of three ways, set by its mass:
- **Under about 40 g it is thrown off.** The world goes INVALID (ENERGY_JUMP), but not always at once: the body can
  first bounce up to 64 mm above the height it was dropped from, for up to 2 s, with health still OK. Do not trust a
  light body on these contacts before `sim.check()["ok"]` has been read at the end of the run.
- **From about 40 g to 35 kg it settles but never fully rests.** It ends within 0.25 mm of its rest height with
  health OK, and keeps vibrating by 0.01 to 0.3 mm. `sim.state()` shows a speed of 2 to 25 mm/s that never reaches
  zero. With `collision="convex"` the same body is at rest to the last digit.
- **From about 40 kg it sinks in.** The world goes INVALID (PEN_FAIL) at the first contact, about 9 mm deep.

A static `sdf` or `mesh` part that only liquid touches is not affected (a mesh cup holds its liquid).

Until 2026-11-09 these bodies also gained energy at each bounce and never came to rest, with health OK: free bodies
stepped on rk4, whose four stages saw the pipeline's once-per-step contacts frozen. That is fixed (law:
`tests/contact/test_newton_contacts_rest.py`); this page is what is left.

## Number
A cube of 12 triangles, `collision="sdf"`, dropped 1 cm onto the ground; `dt=0.01` (rigid step 1.25 ms), 1 world,
CPU (MuJoCo Warp route), 5 s, main 105c33f6c with the fixed rigid step these scenes keep. Edges from 2 cm to 1 m at
densities from 50 to 8,000 kg/m3 (53 cases): the behaviour follows the mass, not the size. "Vibration" is the height's
range and the largest speed over the last second. "At rest" would be a speed under 1 mm/s: no case reaches it.

| mass (edge) | end height - rest | vibration | health |
|---|---|---|---|
| 0.4 g to 8 g (2 to 5 cm) | - | - | INVALID (ENERGY_JUMP) within 0.1 s |
| 13.5 g (3 cm) | rose 23 mm above its drop | - | INVALID after 0.3 s |
| 20 g (10 cm) / 25 g (5 cm) | rose 26 / 6 mm above its drop | - | INVALID after 0.7 s |
| 27 g / 30 g (3 cm) | rose 19 / 11 mm above its drop | - | INVALID after 2.0 / 1.9 s |
| 35 g (10 cm) | rose 64 mm above its drop | - | INVALID after 1.4 s |
| 40 g (3 cm, 10 cm) | 0.00 mm | 0.010 mm, 8 mm/s | OK |
| 50 g (2 cm / 10 cm) | 0.00 / -0.01 mm | 0.003 mm, 4 mm/s / 0.057 mm, 25 mm/s | OK |
| 62 g to 73 g (2 to 5 cm) | 0.00 mm | 0.03-0.05 mm, 15-19 mm/s | OK |
| 0.125 kg to 0.5 kg (3 to 20 cm) | -0.03 to 0.00 mm | 0.000-0.024 mm, 2-7 mm/s | OK |
| 1 kg / 2.7 kg (10 cm) | -0.01 / -0.04 mm | 0.018 / 0.042 mm, 7 mm/s | OK |
| 5 kg (10 cm) / 8 kg (20 cm) | -0.11 / 0.00 mm | 0.12 / 0.10 mm, 15 / 8 mm/s | OK |
| 20 kg (20 cm) / 25 kg (50 cm) | -0.07 / -0.09 mm | 0.18 / 0.20 mm, 8 mm/s | OK (PEN_WARN at first contact) |
| 30 kg / 35 kg (20, 50 cm) | -0.23 / -0.21 mm | 0.25 / 0.27 mm, 9 mm/s | OK (PEN_WARN at first contact) |
| 40 kg (20 cm) to 8,000 kg (1 m) | about 9 mm deep | - | INVALID (PEN_FAIL) within 0.1 s |

- `collision="mesh"` gives the same rows as `"sdf"` (checked at 27 g, 0.5 kg, 5 kg, 20 kg and 125 kg).
- A 0.5 kg, 10 cm cube on a static box: 149.993 of 150 mm, 0.014 mm, 5 mm/s. A 4 cm, 0.1 kg `add_box` on a static sdf
  or mesh cube: 119.998 of 120 mm, 0.027 mm, 18 mm/s. A 4 cm, 50 g ball on a static sdf slab: 120.003 of 120 mm,
  0.011 mm, 9 mm/s. One L4 gives the same heights within 0.02 mm, and vibrations of 0.015-0.048 mm at 5-15 mm/s.
- Finely meshed flat faces do rest: a 0.5 kg cube of 768 triangles (sdf or mesh) ends at 50.000 mm with no vibration,
  on the CPU and on an L4. A 10 cm sphere of 1,280 triangles ends 0.14 mm low and moves 0.11 mm at 38 mm/s.
- The heavy limit is for a 1 cm drop (0.44 m/s at contact). A higher drop lowers it: a 1 m, 100 kg cube dropped 5 cm
  goes INVALID too, where the same cube as a convex hull rests 0.43 mm deep with health OK.
- A smaller `dt` does not rescue a light body: a 5 g, 2 cm cube also fails at `dt=0.002` and `dt=0.004`.
- No row that stays valid rises above its drop height.

## Why
On this pipeline a contact row takes the pair table's stiffness and damping (2 MN/m per material, damping from the
pair's restitution) instead of the gains MuJoCo's own contacts get, which are sized for the body's mass and the step.
- The damping is light (a damping ratio of 0.11 for the default pair). At a 1.25 ms step gravity alone changes the
  speed by 12 mm/s per step, so the body leaves the contact for one step in about nine and is caught again: a
  steady cycle instead of rest.
- The stiffness is one number for every mass. On a body under about 40 g it is too stiff for the step, and the
  contact goes unstable; how soon the energy check catches it depends on how the first bounces fall. On a body of
  40 kg it is too soft (the solver's soft contact acts on many times the body's mass), so the body sinks a centimetre
  before it stops.
- These scenes keep the fixed rigid step. The coarser steps a rigid-only scene may otherwise take are sized for
  MuJoCo's own gains: at the 5 and 10 ms the planner would pick, a 5 kg cube vibrated by 1.0 mm at 87 mm/s and a
  20 kg one by 2.5 mm at 116 mm/s with PEN_WARN.

With these rows on the geoms' own gains (a probe, not shipped) the 0.5 kg cube on the ground and on the box, the box
on the static cube and the ball on the slab all rest exactly (no vibration), and the 5 g cube rests too. The 100 kg
cube still sinks 1 cm at its first contact: rows on this pipeline start only once the surfaces overlap.

## Workaround
- Use `collision="convex"` for a solid, roughly convex part and `"decompose"` for a concave one that moves: both
  collide on MuJoCo's own contacts and rest exactly, at any mass.
- Keep `"sdf"` and `"mesh"` for static concave parts (cups, bins, holes) and for moving bodies of about 40 g to 35 kg
  that need the exact concave shape, and read `sim.check()["ok"]` after the run: outside that range the world reads
  INVALID, for light bodies only after up to 2 s.
- To tell whether such a body has settled, compare its height over time (it stays within 0.3 mm), not its speed
  against zero.

## Planned fix
After the release candidate: contact rows from Newton's pipeline take the geoms' own gains on MuJoCo, as MuJoCo's own
contacts do (the pair table keeps the friction). The probe above is that change; it is what widens the mass range and
ends the vibration. The first-contact depth of heavy bodies needs rows that start before the surfaces meet, which is
not scheduled.
