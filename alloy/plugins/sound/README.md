# Sound: `ModalSound` (engine item 10, part 5)

This package gives audio per world and per microphone. It uses modal synthesis, driven by the engine's own contacts.
The plug-in has no feedback into the dynamics. The plan is `[internal note]` section 5, and
the handoff is `wp13-engine10-sound-handoff.md`. WP13 reviewed the design changes on 2026-11-06.

```python
from alloy.plugins.sound import ModalSound
s = ModalSound(bodies=("bar",), presets=("free_beam",), dims=((0.3, 0.02, 0.005),), materials=("steel",),
               mics=((0.0, 0.0, 0.5),))
sim = alloy.plugins.simulation(built, [s])      # MuJoCo route (needs_contacts): MuJoCo Warp, or C in one world
for _ in range(50):
    sim.step_period()
sim.plugins["modal_sound"].write_wav("bar.wav")
```

## Model

**Modes** (`modes.py`, `presets.py`). Each sounding body has a table of modes in its body frame:
- frequency, damping ζ = tan δ / 2, and modal mass;
- a gain vector for each mode at each sample point;
- a radiation weight.

The tables come from:
- presets: a free-free beam (one or two bending planes), a cantilever, or a simply supported plate;
- DC6's RJ45 tab and housing (`dc6_tab`, `dc6_housing`);
- `fem_box`: a free box solved by P2 tet FEM;
- a `modes.npz` file (`ModeSet.save`/`load`).

**FEM modes** (`fem.py`).
- Element: P2 (10-node) tetrahedra by default, P1 as an option.
- Exact element matrices from the barycentric integrals ∫ L^α dV = 6V·α! / (|α| + 3)!.
- Assembly with scipy sparse; solve with shift-invert `eigsh` in two passes. The first pass estimates the lowest
  elastic eigenvalue, the second shifts to half of it: at σ = -1, next to the 6 rigid modes, half the pairs came
  back unconverged. Any pair with a residual > 1e-6 raises.
- Gains: sqrt(m)·φ at the boundary vertices.
- CLI: `python -m alloy.plugins.sound.fem part.obj --material steel --modes 16 --edge 0.004 --out part.npz`
  tetrahedralises an OBJ with pytetwild.
- Checks (`tests/plugins/test_sound_fem.py`):
  - on a free 300 x 5 x 5 mm steel bar, P2 is within 1 % of Euler-Bernoulli for modes 1-4 (-0.1 ... -0.9 %: shear
    and rotary inertia, which Euler-Bernoulli omits);
  - it moves 0.03 % on mode 1 when the mesh doubles;
  - the axial mode matches rod theory to 0.2 %;
  - a clamped end gives the cantilever within 1 %;
  - P1 on the same mesh reads +59 % (locking), so do not use P1 for thin parts.

Modes at or above 0.45·fs are dropped.

**Excitation.** There are two paths per step (`kernels.py`).
- **Impact.** It fires when a body pair first touches at |v_n| > `rest_speed`.
  - Each mode gets a velocity kick g_k·J·H(ω_k τ).
  - The impulse is J = (1 + e)·m_eff·v_n. The approach speed v_n is taken before contact, and m_eff includes
    rotation.
  - τ is the Hertz contact time, 2.87 (m*²/(R* E*² v))^(1/5) (`hertz.py`). H is the half-sine pulse spectrum.
  - The kick does not depend on the sim step or on how many steps MuJoCo's soft contact takes.
  - The pair's later forces stay out of the sustained path until it separates or stays slow for `settle` steps, so
    nothing is counted twice.
- **Sustained.** Settled contacts add their force g_k·F, interpolated linearly over the step. A resting load gives
  only a static deflection.

**Synthesis.**
- Each mode is an exact one-sample damped oscillator update at `sample_rate`.
- Radiation is a monopole proxy, p = ρ_air·S·a/(4π·r), to each microphone. It has an optional r/c delay.
- Sums over contacts use int64 fixed point, and each output element has one writer. The result is deterministic and
  safe to capture in a graph.

**Readings.** `audio [W, M, N]` is a ring buffer of `duration`. Also read `samples`, `kicks` and `sustained` (the
cumulative modal impulse per path), `audio_of(world, mic)` and `write_wav(path, world, mic)`.

## Laws (`laws.py`; a steel bar 300 x 20 x 5 mm struck by 5 mm steel balls)

The results below are from `python -m alloy.plugins check modal_sound` on CPU, 2026-11-06.

| Law | Check | Bound | Result |
|---|---|---|---|
| S-1 frequencies | FFT peaks against Euler-Bernoulli free-free (tabulated β_k L), modes 1-5 | 1 % | -0.00 % each |
| S-1 FEM | the same strike, `fem_box` preset (P2): peaks against the tet mesh's own eigenvalues, every excited mode (7) | 1 % | -0.000 % |
| S-2 decay | envelope decay rate against ζω, tan δ 0.01, modes 1-3 | 5 % | -0.09, -0.12, +0.14 % |
| S-3 linearity | two strikes against the sum of the singles (fixed bar) | 1e-4 of peak | 9.9e-7 |
| S-4 distance | microphones at 0.5 and 1 m: peak ratio 1/2, and a lag of r/c | 1 %, 1.5 samples | 0.5000, 65 vs 64.3 |
| S-5 contact duration | centroid at 0.3, 0.8 and 2 m/s (E 2 GPa striker); log slope against a NumPy Hertz synthesis | 20 % | 0.172 vs 0.171 |
| S-6 step independence | kicks at dt/4 against the dt-free prediction | 5 % | 1.3e-7 |
| impulse_once | kicks against the prediction; sustained share (MuJoCo's soft impact not counted twice) | 1e-4; 1 % | 1.3e-7; 0 |
| null | no contact: every sample | exactly 0 | 0 |
| rest | 5 mm drop onto a bar resting on the ground (tan δ 0.05): last 0.1 s RMS / peak | 1e-3 | 2.1e-8 |
| harness | crash, hook and run determinism, identical and isolated worlds, graph replay | | all pass |
| speed | CPU, 3 worlds | 2x | 1.05x |

`tests/plugins/test_sound.py` adds two tests. In the first, the same strike at dt and dt/2 gives mode peaks within
5 %. In the second, one strike gives kicks equal to the prediction and a zero sustained path. Host math is tested in
`test_sound_modes.py`: the beam roots against the tables, the oscillator against `expm`, the pulse spectrum against
a numerical Fourier transform, and the analysis on synthetic signals.

GPU smoke (testing directive 2026-11-06: <= 64 worlds, one dt). Setup: an L4 (spot, us-central1-b), Newton
4f3b41ef, 64 worlds of the strike scene at 1.25 ms, CUDA-graph periods, 1 object x 13 modes (`free_beam_2`):
- step time 0.214 ms without the plug-in and 0.251 ms with it (+17 %, including the contact + CONTACT_F export);
- kicks match the dt-free prediction to 5.8e-8;
- kicks and audio are bit-identical across the 64 worlds;
- the sustained path is 0;
- the audio peak equals the CPU run's (2.4877 Pa).

Scripts: `[internal note]` and `sound_cost.py`. Runs at 512 and 1,024 worlds wait for the
reconciliation pass.

## Limits

- **m_eff ignores joints.** It uses the two bodies' free-body inertia. For a jointed or fixed body this
  underestimates m_eff, and so J.
- **Restitution e is a sound-side parameter per object.** MuJoCo's own bounce comes from solref.
- **A pair that already touches makes no new impact.** A hit on a resting contact enters as sustained force, which
  is band-limited by the step.
- **Tangential (friction) forces count only through the gain vectors.** There is no separate scrape or roughness
  model.
- **The monopole proxy is not absolute SPL.** Compare spectra and decay.
- **Audio inherits the contact model.**
  - WP13, 2026-11-06: Newton's default kd 100 at a stiff ke of 1e5 gives MuJoCo a damping ratio of ~0.15. Contacts
    under a steady load then chatter at ~100 Hz, ±1 mm, and ModalSound hears that as repeated impacts or a buzz.
  - Give sounding contacts a damping ratio near 1. The law scenes use ke 2,500 and kd 100.
- **The oscillator state is float32.**
  - Decay rates are within 0.16 % for tan δ ≥ 1e-3 down to 20 Hz.
  - They are about 5 % off at 50 Hz with tan δ 1e-4 (WP13).
  - Setup warns when 2ζω/fs < 1e-5.
- **One step size per run.** The audio clock counts steps.
- **Shear and rotary inertia are ignored** in the beam and plate presets. The FEM path includes them.
- **v2 idea (WP13): take J as the episode's integrated normal impulse at its end.** That gives the right J for jointed
  bodies, at a few ms of onset delay.
