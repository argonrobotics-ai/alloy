# Boiled-off liquid particles stay in the solver and can block a heater

**What.** When a liquid particle boils off, the heat plug-in stops exchanging heat with it, but the particle stays in
the liquid solver with its mass and position. A boiled-off layer on a heater holds the remaining liquid off the heater,
so heating stops.

**Number.** Water at 99 °C on a 177 °C plate: heating stops once the bottom layer, 25 % of the mass, has boiled off.
The rest stays at 99.05 °C.

**Why.** The liquid solver has no public call yet to remove particles during a run.

**Workaround.** Read `evaporated_in_solver` from the plug-in. It counts the boiled-off particles that are still in
the solver, and a run that boils must check it is 0. Keep boiling scenes short of boil-off, or heat with
`power_density` (volumetric, needs no contact).

**Planned fix.** The plug-in will retire boiled-off particles once the liquid solver has a retire call; it is in
progress.
