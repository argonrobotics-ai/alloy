# Rigid and robot overlap flags: what PEN_WARN and PEN_FAIL do not catch yet

## What is wrong
Since the fix of audit finding ERR-1, `sim.health()` flags overlap between rigid bodies and robot parts. The limit
of a contact is 1 cm, or half the thinner part along the contact normal when that is less. PEN_WARN (DEGRADED)
while a contact is deeper than 1 mm or past its limit; PEN_FAIL (INVALID, the world frozen, `profile="precision"`
raising `alloy.SimulationError` from `sim.step` / `sim.run`) when a contact has stayed past its limit for 0.1 s of
simulated time, or past twice the limit (2 cm, or through the thinner part) at the fourth health check in a row; for
a body larger than 20 cm that hard depth is 10 % of the smaller moving body's smallest size (a static shape sets
none). Cloth on rigid bodies follows the 0.1 s rule only: a box dropped 3 cm onto a 2 mm sheet pushes cloth
particles 2.7 mm into it for a few steps without passing the sheet. A fast body's one-step dip into a thin part is
left to FAST_CONTACT and TUNNEL. Each world also reports `peak_penetration_m`, its deepest contact since the build
or the last reset. Before the fix no rigid or robot contact raised either flag. Seven limits remain:
- The flags follow the depth at the latest check. PEN_WARN clears when the bodies come apart, so a run that ends at
  rest can read OK after a dip; `peak_penetration_m` and `sim.check()` ("deepest contact ... mm") keep it.
- A landing or a bounce that passes 1 cm for less than 0.1 s reads PEN_WARN, not PEN_FAIL, by rule: a soft contact
  compresses that far in a hard impact, and further for a large body. Between 1 cm and the hard depth a world
  therefore steps on for up to 0.1 s before it freezes (up to 2 cm deep for small bodies, 10 cm for a 1 m crate),
  and past the hard depth for three checks (3.75 ms at the default 1.25 ms step).
- The 0.1 s clock is one per world, not one per contact: different contacts dipping past their limits one after
  another keep it running, so a long stream of landings on thin parts can reach PEN_FAIL with no single contact
  resting deep. No shipped scene does (bins and marbles read PEN_WARN while they land and OK at the end).
- On a thin part PEN_WARN also fires under 1 mm: past half the part's thickness (a 1.6 mm connector part at
  0.88 mm).
- A frozen world shows its last good state, so `max_penetration_m` beside PEN_FAIL can read less than the depth
  that failed. That depth is `peak_penetration_m`, and it is in the world's `problem` text.
- A robot file whose contacts are soft on purpose rests deeper than 1 cm and reads INVALID while it stands still.
  MuJoCo Menagerie's ANYmal C is one: its feet (`solimp` 0.015) rest 17.2 mm into the floor, and plain MuJoCo C
  reads the same depth. So the physics matches MuJoCo, and it is this health rule that freezes the world. To see
  it: `sim = alloy.load("<menagerie>/anybotics_anymal_c/scene.xml", worlds=2, device="cpu")`, `sim.step(100)`,
  then `sim.check()["ok"]` is False, with PEN_FAIL and START_PEN.
- A legged robot that falls and stays down can read INVALID. Three biped files run with no policy (an H1, a simple
  humanoid and a Cassie) fell in every world, and their links then lay 2.0 to 3.1 cm into the floor, past the 1 cm
  limit: after 4 s, 976, 419 and 14 of 1,024 worlds were frozen (one L4). They lay as deep before the fix, when
  nothing flagged it. Robots that stood in the same runs (a G1, a Go2, an ANYmal D) were not flagged. So a
  rollout in which a biped falls can end with frozen worlds; no workaround is measured.

## Number
CPU, 1 world, `dt=0.01` (physics step 1.25 ms), MuJoCo C and MuJoCo Warp routes alike
(`tests/precision/test_health_pen_mujoco_law.py`; the same on one L4 for the box cases):
- A 0.3 m box on a 50 g, 5 cm cube. 5 kg: 0.35 mm, OK. 80 kg: 2.9 mm, PEN_WARN. 200 kg: rests 7.0 mm deep,
  PEN_WARN; it passes 1 cm for a few milliseconds as it lands. 500 kg: passes 2 cm at t = 0.105 s, PEN_FAIL,
  frozen. 2,500 kg on a 2.5 g cube: PEN_FAIL.
- Health's depth equals the depth worked out from the bodies' heights within 0.01 mm.
- A position-servoed arm link (kp 20,000) pressed onto a table passes 1 cm at t = 0.355 s and is frozen at
  t = 0.455 s, 18.7 mm deep. Before the fix it went 104.5 mm into the table with health OK.
- 5 cm balls dropped 0.3, 0.6 and 0.9 m with restitution 0.6 dip 5.8, 8.2 and 10.1 mm at the bounce: PEN_WARN
  while deep, OK at rest, never INVALID. A 10 cm, 5 kg cube dropped 0.4 m: the same.
- A 1 m cube at 10 kg/m3 dropped 5 cm / 0.5 m / 2 m / 5 m / 10 m dips 5.9 / 19 / 39 / 63 / 87 mm, each for under
  50 ms past 1 cm, and rests 0.05 mm deep: PEN_WARN while deep, never INVALID (87 mm is under its 10 cm).
- Thin parts (`tests/precision/test_health_pen_measured.py`): a 5 mm plate pressed 2 / 3 / 6 mm into the ground
  reads PEN_WARN / PEN_FAIL after 0.1 s / PEN_FAIL at the fourth check; a 2 mm sheet at 0.5 / 1.5 / 2.5 mm reads
  no flag / PEN_FAIL after 0.1 s / PEN_FAIL at the fourth check.

## Why
Contact on the MuJoCo route is soft: depth grows with load and with impact speed. The checks read MuJoCo's own
contact distances each physics step and keep the latest reading and the deepest one, with one clock per world for
how long a contact has been past its limit. Which flags were raised earlier in a run is not kept yet.

## Workaround
- Read `sim.health()["worlds"][w]["peak_penetration_m"]` after a run, or `sim.check()`, which also names the pair
  of objects.
- A scene that overlaps by design at a thin part (a blade edge on a strap) marks those shapes at the engine layer:
  `sim.engine.health.relax_thin_part_rule(shape_ids)` holds them to the 1 cm rule alone.
- Two parts that press into each other by design (a sprung connector finger on its card) mark the pair instead:
  `relax_designed_overlap(sim.engine, [(("finger", "card"), "0.85 mm preload")])` from `alloy.precision.health_pen`
  lifts the thin-part limit between those two only; against every other shape it still applies.
- Lower `dt`, lighter loads or stiffer contact materials reduce the depth itself.
- For the ANYmal C above (measured standing, CPU): in a copy of `anymal_c.xml` delete the feet's
  `solimp="0.015 1 0.03"` (MuJoCo's default contact stiffness then applies) and raise the base from z 0.62 to 0.64
  (at 0.62 the legs start 10.5 mm into the floor). It then stands OK for 5 s, 1 mm deep at rest. Either change alone
  is not enough. This changes the foot compliance the file's authors chose, so a policy trained on the soft feet may
  behave differently; walking and the GPU are not measured. The two `relax_...` calls above do not help here: both
  still hold a contact to the 1 cm limit.

## Planned fix
A "seen since reset" flag list beside the current flags, so a PEN_WARN raised during a run is visible at its end
(in work under audit finding ERR-11). A public option to relax the depth limit for a robot file with soft
contacts comes in 0.1.1; today there is none. Not scheduled: a shorter dwell for stiff contacts.
