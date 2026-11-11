# Known gap: light slender rigid parts spin and roll slower about their thin axis (the inertia floor)

## What is wrong
- A light slender free part (a marker pen, a pencil, a thin rod or a utensil) has a tiny inertia about its long
  axis. A contact at its rim then spins it to hundreds of rad/s, 0.4-0.7 rad per 1.25 ms rigid step. The contacts,
  linearised over that much rotation, inject energy, and a crowded scene can blow up.
- So Alloy raises such parts' rotational inertia at build (`alloy.contact.inertia_floor`, 2026-11-08). A free
  MuJoCo body is raised when:
  - its largest / smallest principal inertia is at least 4;
  - its smallest is below m·r0²;
  - it is not in a precision pair.
- It gets armature on its 3 rotational dofs, up to m·r0². r0 = 9 mm at the default 1.25 ms rigid step (flat below,
  in proportion above).
- Such a part therefore spins and rolls slower about its thin axis than the real part, and tumbles slightly slower.
  Every other body is unchanged, bit for bit.
- Not covered (they keep their inertia; if one spins up, e.g. a coin struck at its rim, use a rigid step below
  1.25 ms, `Scene(dt)` <= 1 ms, and read `sim.health()`):
  - compact parts, however light, whose largest / smallest principal inertia is under 4: a thin disc or coin (2), a
    short cylinder, a marble, a cube;
  - parts in a precision pair (the precision ladder sizes its own pairs);
  - robot links and parts of multi-body objects.

## Number
- YCB's large marker (15.8 g, 121 mm, axial radius of gyration 6.2 mm):
  - its axial inertia goes from 6.1e-7 to 1.28e-6 kg m² (x2.09);
  - rolling down a slope, it accelerates 0.76x the true value, (1 + 0.40) / (1 + 0.83) at its 9.9 mm radius;
  - its tumbling inertia goes up 3 %.
- Any raised part rolling on a round face of radius R accelerates (1 + r_g²/R²) / (1 + r0²/R²) times the true
  value, r_g being its own smallest radius of gyration. A solid 9 mm-radius cylinder: 0.75x (the law in
  `tests/contact/test_inertia_floor.py`, within 3 %).
- Without the floor, WP13's clutter pile (100 objects dropped into a bin, 8 worlds, CPU) had 3 of 8 worlds INVALID.
  A marker reached 240 m/s, +705 J against the pile's 70 J.
- With it the pile is clean at 0.625, 1.25 and 1.5625 ms: 0 INVALID and nothing faster than the 3.3 m/s free fall
  + 1 m/s (measured with development scripts, not shipped).
- The law (`tests/contact/test_inertia_floor_pile.py`, ruling 2026-11-08 13:36Z) runs seeds 0, 1 and 2. It requires:
  - no world's energy above its start + 1 %;
  - no body above 2 x the free fall (6.6 m/s);
  - no INVALID world and no SPIN.
  - Fastest bodies are 3.26 / 4.47 / 3.76 m/s. Seed 1's is a hand-off: a 2.4 m/s box strikes a marker, and the
    bodies around it lose energy.
- At 1.25 ms, 8 mm gives 0 INVALID but still flings a body to 9.7-17 m/s; 7 mm leaves 3 of 8 INVALID.

## Why
- The rotation per step, not the hulls: one convex hull or merged hulls still fail 3-4 of 8. RK4 vs implicit and
  MuJoCo's multi-CCD change nothing essential.
- A smaller step alone is not clean: at 0.625 ms a 6.5 mm floor still leaves 1 of 8 INVALID.
- The integrators take the contact geometry at the start of a step, which a part spinning 0.5 rad per step leaves
  far behind.

## Workaround
- Exact rolling or spinning of such a part (a rolling pencil, a spun pen) needs the floor off and a smaller rigid
  step: `scene.set_inertia_floor(False)` (`inertia_floor: false` on a scene card). Check `sim.health()`.
- The build warning names every raised part and its increase; `sim.routes()[name]["reasons"]` says so too.

## Planned fix
- Contacts that follow the rotation within a step (substepping only the fast-spinning body, or a rotation-aware
  contact frame). Then the floor can go.
- A spin health flag (DEGRADED above 0.5 rad per step, naming the body and the fix) is coming from WP5 (SCR-WP5-14).
- Estimate: several days, research grade. Not scheduled.
