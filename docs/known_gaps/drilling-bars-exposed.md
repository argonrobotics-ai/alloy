# A bit that lands on an exposed bar never cuts it

**What is wrong.** A bit that meets a bar before its hole has a stage (a bar whose top is at the concrete's face) is
held there and reported jammed with its thrust, but cuts nothing into the bar, however hard it is pushed. A bar under
any cover is cut: the bit opens its hole in the concrete, meets the bar inside it, cuts into the bar at the steel
estimate when pressed past the chisel threshold, and drills on once the hole's bottom has passed the bar
(the rebar section of the drill-holes design note, not shipped).

**The number.** A #4 bar whose top is the face, a 12.7 mm bit at 1.8 J x 85 /s fed 0.5 mm onto it: held, thrust
over 100 N, `removed` 0 and `bar_cut` 0 (`test_a_jam_on_an_exposed_bar_is_reported`).

**Why.** The cut into a bar moves a stage's bottom. A hole is opened by its first cut, and the bar stops that cut, so
there is no stage whose bottom could move.

**Workaround.** Give the bar a little cover (0.1 mm is enough to open the hole first).

**The fix and when.** Open the hole's first stage, at zero depth, when the point lands on a bar, so the jam's cut has
a bottom to move. Small; not in plan (no task drills an exposed bar).
