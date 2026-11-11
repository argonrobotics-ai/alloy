# A weld or connect in a robot's MJCF is a stiff spring, not a rigid joint: limits of the stiff rule

**What is wrong.** A `<weld>` or `<connect>` equality in a robot's MJCF that sets no `solref` / `solimp` was
MuJoCo's soft spring: under load the two bodies came apart 2.3 mm per kN on a 9 kg pair. Since the default change
(`alloy.build(scene, welds=...)` left out is "stiff"), such rows take the stiff rule of scene attachments and hold
3-30 kN to hundredths of a millimetre. What is left: a weld is still a spring whose force stiffness falls with the
lighter body, it trails by a step-size error in motion, and `welds="mujoco"` (or a row written by hand) keeps
whatever the file says.

**The number.** Rig: a free 20 kg body welded to a 5 kg body on a slide, pulled sideways (the audit's ERR-11 rig;
`tests/precision/test_weld_stiff_law.py`), 1.25 ms rigid step, CPU, MuJoCo C and MuJoCo Warp alike:

| pull | `welds="mujoco"` (MuJoCo's row; the default before the change) | default (`welds="stiff"`) |
|---|---|---|
| 3 kN | 7.0 mm, JOINT_FAIL | 0.0015 mm, OK |
| 10 kN | 23.3 mm, JOINT_FAIL | 0.0051 mm, OK |
| 30 kN | 70.0 mm, JOINT_FAIL | 0.0152 mm, OK |

Both columns are the row's own compliance to within 2 % (the law): a row carrying a force F rests at
`r = F W (1 - d) / (d² k)`, where W is the two bodies' inverse weights summed (0.117 per kg on the rig), d the
impedance (`solimp`) and k the stiffness from `solref`. MuJoCo's defaults (0.02 s, 0.9-0.95) give 2.0e-5 s² x W per
newton; the stiff rule gives 4.3e-9 s² x W at the 1.25 ms step, about 4,600 times less. The stiff rule's stretch grows
with the square of the rigid step (`sim.routes()` names the step).

In motion a weld also trails by a step-size error that stiffness does not remove. Over one step MuJoCo holds the
row's direction fixed, so a body carried around a curve at acceleration a keeps a small row velocity, and the row's
damping holds it off by `B h a / (2 K d)`: about `a h τ` for a time constant τ, and `(5/3) a h²` under the stiff rule
(2.6 µm per m/s² at 1.25 ms). Measured on a 1 kg tool welded 0.4 m out on a link spinning at 2 / 4 / 8 rad/s (the
same law file): default 0.38 / 0.39 / 0.44 mm off (mostly the sag under its own weight), stiff 0.004 / 0.017 /
0.066 mm. A position servo whose target is ramped by `sim.run(targets=...)` gets a new target once per control
period. A stiff servo then kicks its link every period, and the weld follows those kicks with a larger error: 4.4
mrad and 1.8 mm on that rig at 4 rad/s, the same in plain MuJoCo with the same controls. A shorter control period
(`alloy.Scene(dt=...)`) makes the kicks smaller. On ordinary arms: the shipped
Franka Panda carrying a 0.5 kg tool welded to its hand peaks at 0.01 mm with stiff welds through slow, fast and
pick ramps, against 0.42-1.49 mm with the default rows, which already read JOINT_FAIL holding still (2.6 mm sag). A
light two-link arm on a very stiff servo (kp 20,000) does read JOINT_FAIL with stiff welds on fast ramps at
dt = 0.01 (2.0-2.2 mm; default 19.6-43.1 mm); at dt = 0.002 it peaks at 0.55 mm.

The stiff rule is stiff in acceleration, so in force a weld is as stiff as its most mobile body lets it be: about
`ω² / ((1 - d) W)`, which is 0.36 / (h² (1 - d) W), or 2.3e8 / W N/m at 1.25 ms with W in 1/kg. The 9 kg pair
above holds 2e9 N/m. A 1 kg body welded to a 1 g carrier (W = 334 per kg) holds 6.9e5 N/m: 0.02 mm under 10 N at
1.25 ms, 0.33 mm at a 5 ms step. A 1 t body hung from that 1 g carrier is outside what a weld row can do: it sags
about 0.1 m and swings, against 5-7 m with MuJoCo's default row. For such a pair, make the heavy body a child of the
carrier (Workaround). The stiff rule costs nothing measurable: 1,024 and 4,096 worlds on one L4 step equally fast (within
0.5 %, three alternating runs), and the solver takes the same single iteration per step.

Where the weld point is matters. In MuJoCo a weld's `anchor` is a point of its second body, so
`<weld body1="b"/>` (second body: the world) anchors at the world origin, and a load on b far from that point is
held by the weld's rotation rows: a 20 kg body 0.6 m from the origin moves 0.58 mm under 3 kN with stiff rows (the
same in plain MuJoCo; MuJoCo's default row goes unstable there), against 0.0007 mm with `anchor` at the body
(`<weld body1="b" anchor="0.1 0 0.6"/>`) or written `<weld body1="world" body2="b"/>`.

A weld's rotation rows follow the same rule and are just as limited by the bodies' rotational inverse weights, so a
moment shows as an angle first: five 1 kg boxes welded in a 1 m cantilever bend 0.4 mrad under their own weight and
1.7 mrad with 50 N at the tip (JOINT_FAIL from about 25 N), while their weld points move 0.001 mm; released, they are
back to 0.0004 mm. MuJoCo's own rows let the same cantilever fall 0.78 m.

How health reads it: JOINT_FAIL (DEGRADED) above 1 mm or 1 mrad for an MJCF weld, which health judges both as a
loop joint at that flat floor and as a row at its own limit. On a ramped arm the angle can trip first: a tool welded
0.3 m out on a kp 2,000 link ramped 1.5 rad in 0.5 s peaks at 0.29 mm and about 1 mrad with stiff rows. The stiff rows read far below both. JOINT_FAIL counts here
are from a tree with the flag's hold between checks (since e21f29f28).

**Why.** It is MuJoCo's soft-constraint model working as specified, at defaults chosen for a 2 ms step and light
loads; nothing is wrong in the step or the solver (plain MuJoCo C gives the same numbers). Alloy's stiff rule for
attachments (`alloy/scene/weld_impedance.py`: stiffness set from the rigid step, impedance 0.999) was never applied
to equalities that come in through a robot file.

**Workaround.**
- Nothing to do for the common case: leaving `welds` out gives the stiff rule. Rows that write their own `solref` or
  `solimp` stay as written (a row written out as exactly `solref="0.02 1" solimp="0.9 0.95 0.001 0.5 2"` counts as
  a default); `welds="mujoco"` keeps every row as written. `alloy.load(path)`, the command line and the Gym
  environments build with the default.
- For a heavy body on a light carrier, or wherever even microns matter: do without the equality. Make the second
  body a child of the first in the MJCF, with no joint between them; it then moves exactly with it
  (`docs/guide/robots.md`, Grasping).
- Anchor a weld to the world at the body: `<weld body1="world" body2="b"/>` or `anchor=` at the body's position.

**Planned fix.** None for the rows: the remaining limits are those of a soft-constraint step (smaller steps, or a
tree joint instead of an equality, remove them). A public call that attaches or welds an object while a simulation
runs is separate and not built ([grasp-no-attach-or-weld.md](grasp-no-attach-or-weld.md)).
