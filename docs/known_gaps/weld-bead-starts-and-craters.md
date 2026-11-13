# A weld bead's start, its tack starts and a crater mid-joint are not right yet

**What.** `GridThermal` (the weld heat plug-in: a heat grid with melting, a moving arc and deposited weld metal) lays
a bead whose steady part and, with the pool flow on (as in the welding tasks), end crater at a joint's end are sound,
but three places are wrong. Where an arc starts on cold steel, the first wire piles up on steel that has not melted (a
cold lap, the root not fused for a millimetre or two). Where a pass runs over the start of an earlier tack weld, the
bead laps the plate by two cells where one is allowed, the root stays unfused for 1.5 mm, and the leg on the upright
member is 0.4 to 0.9 mm short. Where a pass stops mid-joint (half-fillet), the crater's bead is one cell short of size
(6.0 mm where 6.35 mm is asked).

**Number.** Gym-Anything's four welding tasks, run with its own program on its own grid and judged by a port of its
evaluator (`alloy.tasks.weld_ga_tasks`, one L4 each). The tasks run the cap model with the pool flow
(`weld_ga.ARC_PRESSURE`, Gym-Anything's arc force; GridThermal's own default leaves the flow off), and score fillet-2f
0.806, set-up-and-weld 0.806, half-fillet 0.761 and tack-and-weld 0.808 of 1 (without the flow: 0.774, 0.774, 0.753
and 0.775; half-fillet's difference is within one section's credit); none succeeds, with the flow or without.
Gym-Anything's own weld model scores 0.894 on fillet-2f by its own evaluator (it leaves a 5.5 mm cold lap at its own
start), so 0.894 is the mark, not 1. On fillet-2f the steady bead is 6.64 mm (6.35 mm is asked; Gym-Anything's is 6.71
mm), and throat, convexity, the five fusion sections and the crater pass. The failures: the root is not fused on 1.25
to 2.75 mm and 9.25 to 10.75 mm (it must be fused from 5 mm on); the lap is 0.75 mm at the strike and 0.5 mm over the
tack starts (0.5 and 0.25 mm are allowed); the size is under 6.35 mm on 18 % of the weld, at the two tacks (least 5.5
mm). The steady bead's penetration into the plate is 0.93 mm (mean over 30 to 110 mm) where Gym-Anything's is 1.5 mm.

**Why.** The pool's metal is shared along the liquid run at one pressure and dented by the arc
(`alloy/plugins/thermal/grid_flow.py`, on in the welding tasks, off by default: `arc_pressure`), but each slice's
surface is one circular cap with no weight. Gym-Anything's liquid lies on the plate under gravity, its first droplets
stack as a dome until the steel under them melts, and a bulge can roll onto cold steel; without these the wire must
land where the arc was 0.5 s before, which leaves the first 0.5 s of wire at the strike point. The tack starts follow
from the tacks themselves: they fuse less of the upright member than Gym-Anything's.

**Workaround.** Judge a bead more than 15 mm from a start or a tack's start: there size, throat, convexity, fusion and
the crater follow the heat input. The pool flow is verified at 0.25 mm cells at the joint (a 30 mm fillet at 0.5 mm
cells also ran without a starved section; it warns on cells coarser than 0.6 mm); on a coarser grid leave
`arc_pressure` at its default (off).

**Planned fix.** None in this weld model: measured, not solvable with it (heat-v3-design, Next slices). Two
slices aimed at the starts, each run with the wire landing under the arc on fillet-2f: the pool's shape with the
liquid's weight (opt-in `bead_model="surface"`) scores 0.789, and Gym-Anything's cold starts (the droplets
stacked as a dome on cold steel, measured, not shipped) 0.794 with the cap model and 0.798 with the surface,
against 0.784 for the cap model alone and the shipped 0.806 (its landing delay of 0.5 s is our estimate; 0.828 needs
a delay tuned to the score). With the dome the tack starts hold Gym-Anything's metal (13-17 mm2 a slice against its
13; 26-29 with the delay) and, in the cap model, the root under the first tack's start fuses, but the arc start's
first 5.5 mm, the size and the overlap at the starts still fail, so no task succeeds.
