# A lightly pressed sliding contact tips when its MuJoCo gains bypass the route

**What.** A box pressed lightly onto a face while it slides (a magnet eraser on a board) tips over at a 1.25 ms rigid
step, but only in scenes that write MuJoCo contact gains themselves (Newton shape `ke` / `kd` on MuJoCo-owned bodies,
as the plug-in test harness's scenes do). On the default route (the public API, library scenes) the same contact
slides flat: the law `tests/contact/test_light_sliding_tip.py` checks a 48 g box down a 44 deg slope at mu 0.5 (the
eraser's press / weight ratio) and a 20 g cardboard box on a 50 g dynamic plate.

**Number.** Harness gains (Newton reads kd 2e3 as tau = 2 / kd = 1 ms; MuJoCo's refsafe raises it to 2 dt; impratio 1):
the box tips 14 deg in 0.3 s (peak 23 deg) and its face lifts 11 mm off the board; with a magnet's 1/h^4 pull, 61 deg.
Default route: rotation 0.04-0.11 deg and lift 0.01-0.08 mm (2-48 g; ABS, steel, glass, PMMA, cardboard, paper board,
foam; 44-60 deg slopes, up to 1.6 m/s).

**Why.** MuJoCo's soft friction rows ask for a reference acceleration of -b·v_slip, b = 2 / (dmax·tau): at tau = 2 dt
and 1 m/s that is ~800 m/s^2, and the cone lifts the sliding box off its face each step until it rocks over. The
route's gains keep tau at ~8-12 dt for such pairs (a body's material gains mix with the face's, MuJoCo solmix) with
impratio 300. A 25 ms friction time constant alone (solreffriction) also holds it at the harness's normal gains.

**Workaround.** Build contact gains with the route's `alloy.scene.native_gains.apply_native_gains` (or keep tau = 2 / kd
at 8 dt or more), or run at dt <= 2.5e-4 s. Do not lengthen the friction time constant everywhere: stiction creep
grows with it (a box held at 98 % of its friction limit crept 3 mm/s at 2.5 ms, 34 mm/s at 12.5 ms).

**Planned fix.** The plug-in harness taking the route's gains and impratio (`alloy/plugins/harness`; its laws' rows
would move, so their owners re-baseline): about half a day plus the re-baselines.
