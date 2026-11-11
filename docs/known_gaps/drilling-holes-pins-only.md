# Drilled holes are seen by listed pins only

**What is wrong.** The drill plug-in records each hole it cuts (`Drill.holes()`), and the parts listed as the
drill's `pins` (bolts, dowels, anchors) pass through those holes. Every other body still meets the undrilled
shapes. A bolt that is not listed rests on the face over a hole as if the hole were not there. So does a pencil
dropped into a drilled hole, or a second drill bit's housing. A pin listed for the drill also meets only the drilled
parts through the plug-in: the whole pin (shank and head) must have no solver collision with them, or the head is
seated twice (by the solver and by the plug-in).

**The number.** Not a magnitude: a body that is not listed sees no hole at all. The listed pins pass the holes they
fit, with their heads seated F / k (0.75 mm at 150 N) into the face. They are stopped at any push, resting where the
end meets the narrowing to within 1 µm of the geometry plus F / k: by a hole narrower than the shank or a pilot, a
stepped hole's shoulder, a blind hole's point cone, an undrilled face, or two parts' holes that no single axis fits
(offset by more than both clearances; a chamfered end guides an offset pin in only within them). A head narrower
than a counterbore rests on its floor. A pin pushed hard can still tip over: its seated head resists tilt only
as a disk on four rim springs, k r_h^2 / 2 per axis (4.2 N m/rad for an M8 head at 2e5 N/m), so a push F applied a
lever L above the head tips it once F L exceeds that plus whatever holds the pin's axis. In the laws' scene (a 20 N
m/rad hold, 25 mm up) an aligned square bolt passes cleanly at 2000 N; a chamfered one guided in 0.7 or 0.95 mm off
is clean at 1000 N and lost at 2000 N; one pre-tilted 1 deg passes at 1000 N but ends 13 deg tilted. A chamfered
bolt wedged on a second part's misaligned mouth holds upright only to about 150 N (2.7 deg at 150 N, 16 deg at 300
N; it tips over at 500 N): the walls of the first part's hole resist its tilt by two springs a plate thickness
apart (k s^2 / 2, 3.6 N m/rad for 6 mm at 2e5 N/m) plus the pin's hold, while the chamfer pushes the tip sideways
with about the whole push. Hold the axis firmly (a gripper) to push harder. A chamfered end resting on a mouth it
does not fit can gain energy when nothing damps it: its landed spring holds the deeper of the landed and today's
rest and turns between a 45 deg and an axial push, which is not one potential, so a bounce with sideways motion
pumps (0.13 to 0.40 J from a 0.01 J nudge at 150 N, undamped). A chamfered pin therefore needs damping of at least
2e-4 s (the default), where the same nudges settle; 1e-4 s still gains. The laws are `pin_through` (square end) and
`pin_chamfer` (chamfered end), both at 150 N, in `tests/plugins/test_drilling.py`.

**Why.** Contact geometry is fixed when a scene is built: MuJoCo and Newton cannot cut a hole in a box shape during a
run. The drill owns the contact between its pins and the parts it drilled, as it owns the bit's contact.

**Workaround.** List the body as one of the drill's pins: `Drill(pins=(Pin("bolt", tip=..., radius=..., head=...),))`,
with the pin's collision with the drilled parts turned off (`ShapeConfig(has_shape_collision=False)`, or a
collision filter). Its shank is then a segment with a radius. A body that is not a straight shank, such as a cable
or a bit of debris, has no workaround yet. A listed wedge anchor (`Pin(anchor=...)`) reads its clamp load T / (K d)
but does not apply it: the pin springs hold the clamped parts. It has no pull-out capacity either: the clip holds
by the pin spring (pulled at 5 kN it slides about 25 mm and stays engaged), and the concrete's strength is unused.

**The fix and when.** Holes as solver geometry: a per-world SDF of a part with its holes, which other bodies would
meet. This is an engine change on the precision contact route, not in plan.
