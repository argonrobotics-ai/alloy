# Heated liquid does not convect: `buoyancy=True` gives the first push only

**What.** By default, particle heat changes a liquid's temperature but not how it moves. Liquid heated from below
stays in layers instead of turning over, when in reality it rises in natural convection and mixes the pot.
`scene.set_heat(buoyancy=True)` adds the thermal buoyancy force for water, and a hot region starts to rise at the
right rate, but the flow stalls after about a millimetre: convection does not develop, and a heated pan stays
layered with it on as with it off. It is off by default, and only water has the density data it needs; the build
says so for other heated liquids (chocolate, porridge).

**Number.** Off (the default): water 14 mm deep on a floor held at 80 °C for 90 s, starting at 20 °C, averages
53.6, 30.0, 22.7 and 22.5 °C in its four 4 mm layers from the bottom up. At this depth and temperature difference the
Rayleigh number is about 1e7, so real water would be convecting and well mixed. On, with 80 °C water in 20 °C water
(CPU laws, `tests/plugins/test_thermal_buoyancy.py`):
- at the default liquid accuracy, the start of rise matches the Boussinesq estimate for a body of fluid starting
  from rest, g Δρ/ρ / (1 + C_a), over the first 30 ms: a sphere of radius 16 mm (C_a 1/2) 0.94 of it, and a
  horizontal cylinder of radius 12 mm (C_a 1) 0.91 of it;
- at `liquid_accuracy="super"` (8 particles across the radius instead of 3) the same cylinder starts slower, 0.78 of
  the estimate;
- afterwards the rise falls behind the free estimate, to 0.65 of it at 150 ms, at both accuracies and for both
  shapes. The cause is not pinned. It does not shrink with finer particles, so it is not the region's size in
  particles. Under test: the liquid's velocity smoothing, applied every step, so more often at the finer step;
- the flow then stalls. Hot water (80 °C) beside cold (20 °C) in a 60 x 60 mm cup, 32 mm deep, side by side: real
  water would trade places in under a second (front speed about 4 cm/s). Here the hot half's mean height is 0.7 mm
  above the cold half's at 0.25 s, 1.1 mm at 1 s and 1.2 mm at 2 s, then still; a full exchange would end at 16 mm.
  At `liquid_accuracy="super"` it reads 1.3 mm at 0.25 s. A pan of water 16 mm deep on an 80 °C floor has the same
  four layer temperatures after 30 s with buoyancy on as off (38.4, 22.6, 22.4, 20.3 °C from the bottom up);
- the cause of the stall is not pinned. Not the cause: the pressure solve's tolerance (10x and 100x tighter: 1.3 and
  1.2 mm at 2 s) and wall friction (off: 1.0 mm). The liquid's velocity smoothing is part of it: with it off (not a
  public setting) the same cup reads 3.3 mm at 2 s and is still creeping, 10x slower than real;
- the laws check the force at the start of rise only;
- the buoyancy force's work is booked in the health energy check as external work. It matches the work summed
  independently over every particle to 1e-5 (the test's bound is 1e-3).

**Why.** The liquid solver keeps one rest density per material, so heat alone moves nothing. The buoyancy force is the
Boussinesq approximation: each liquid particle gets F = m (ρ(T) − ρ̄) / ρ̄ g. ρ(T) for water is NIST's table at 1 atm
(NIST Chemistry WebBook, 0-95 °C, with the 4 °C maximum; `alloy.plugins.thermal.phase_materials.WATER_DENSITY`). ρ̄ is
the mean over all the water in the world, so the forces sum to zero (to the sums' 2^-40 fixed point, about 1e-8 of the
water's weight), and heating all the water evenly moves nothing. It is off by default until a default change is ruled
on. Other materials have no ρ(T) table yet. With `time_scale` above 1, heat runs that many times faster than motion
while buoyant flow does not, so convection is that much weaker relative to conduction than in reality (the Rayleigh
number is divided by `time_scale`); the build warns once when both are set.

**Workaround.** There is none that gives convection: `buoyancy=True` does not mix a heated liquid. Stir it (a
kinematic paddle), or read only results that conduction controls (a solid or paste that melts, a food block's core),
and compare the heat that went in (`heat_in_J`) with the mean temperature. With `buoyancy=True`, two separate bodies
of water in one world (a hot cup and a cold cup) share one ρ̄, so each feels a small force against its container,
about Δρ/ρ of its weight (1.5 % at 60 K).

**Planned fix.** First the stall: a law for a lock exchange's front speed, and the liquid solver's damping of slow
flows under it. Then buoyancy on by default for water, and ρ(T) tables for the other heat materials, from sourced
data. No date yet.
