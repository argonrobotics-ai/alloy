# A weld bead's start, its tack starts and a crater mid-joint are not right yet

**What.** `GridThermal` (the weld heat plug-in: a heat grid with melting, a moving arc and deposited weld metal) lays
a bead whose steady part and, with the pool flow on, end crater at a joint's end are sound, but three places are
wrong. Where an arc starts on cold steel, the first wire piles up on steel that has not melted (a cold lap, the root
not fused for a millimetre or two). Where a pass runs over the start of an earlier tack weld, the bead laps the plate
by two cells where one is allowed, the root stays unfused for 1.5 mm, and the leg on the upright member is 0.4 to 0.9
mm short. Where a pass stops mid-joint (half-fillet), the crater's bead is one cell short of size (6.0 mm where 6.35
mm is asked).

**Number.** Gym-Anything's four welding tasks, run with its own program on its own grid, the pool flow on
(`arc_pressure=weld_ga.ARC_PRESSURE`; the tasks as shipped run without it and score 0.774, 0.774, 0.754 and 0.775),
and judged by a port of its evaluator (`alloy.tasks.weld_ga_tasks`, one L4 each): fillet-2f 0.806, set-up-and-weld
0.806 (the same weld, not run separately), half-fillet 0.761, tack-and-weld 0.808 of 1; none succeeds. Gym-Anything's
own weld model scores 0.894 on fillet-2f by its own evaluator (it leaves a 5.5 mm cold lap at its own start), so 0.894
is the mark, not 1. On fillet-2f the steady bead is 6.64 mm (6.35 mm is asked; Gym-Anything's is 6.71 mm), and throat,
convexity, the five fusion sections and the crater pass. The failures: the root is not fused on 1.25 to 2.75 mm and
9.25 to 10.75 mm (it must be fused from 5 mm on); the lap is 0.75 mm at the strike and 0.5 mm over the tack starts
(0.5 and 0.25 mm are allowed); the size is under 6.35 mm on 18 % of the weld, at the two tacks (least 5.5 mm). The
steady bead's penetration into the plate is 0.93 mm (mean over 30 to 110 mm) where Gym-Anything's is 1.5 mm.

**Why.** The pool's metal is shared along the liquid run at one pressure and dented by the arc
(`alloy/plugins/thermal/grid_flow.py`, opt-in: `arc_pressure`), but each slice's surface is one circular cap with no
weight. Gym-Anything's liquid lies on the plate under gravity, its first droplets stack as a dome until the steel
under them melts, and a bulge can roll onto cold steel; without these the wire must land where the arc was 0.5 s
before, which leaves the first 0.5 s of wire at the strike point. The tack starts follow from the tacks themselves:
they fuse less of the upright member than Gym-Anything's.

**Workaround.** Judge a bead more than 15 mm from a start or a tack's start: there size, throat, convexity, fusion and
the crater follow the heat input. The pool flow is verified at 0.25 mm cells at the joint (a 30 mm fillet at 0.5 mm
cells also ran without a starved section; it warns on cells coarser than 0.6 mm); on a coarser grid leave
`arc_pressure` at its default (off).

**Planned fix.** Three more slices of the weld model (`docs/notes/heat-v3-design.md`, Next slices): a pool shape
with gravity, cold starts as Gym-Anything stacks them (then the wire lands under the arc), and the tacks' fusion.
