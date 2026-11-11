# Heat on particles: warming, melting, boiling and cooking

Every particle of a liquid (SPH) or particle object (MPM) can carry a temperature. Particles conduct heat to each
other, take it from heaters and lose it to the air. A material's enthalpy curve sets its melting, and melting changes
how the material flows. Water boils off at 100 °C, and food cooks as its doneness rises. Heat adds no forces unless
you turn on buoyancy (section 8), so it works beside any solver. Heat in rigid bodies (a pan, a plate) is the
`Thermal` plug-in (`alloy.plugins.thermal`); a "body" heater couples the two. Known limits: section 8 and
`docs/known_gaps/heat-*.md`.

## 1. Quick start: water in a pan with an 80 °C floor

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0, 0, 0.02))      # floor top at z = 0.025
scene.add_liquid("water", size=(0.07, 0.07, 0.02), pos=(0, 0, 0.0355), temperature=293.15)   # 20 °C: takes heat
scene.set_heater("pan_floor", temperature=353.15)                 # the pan's floor held at 80 °C (or power= in W)
scene.set_heat(time_scale=60.0)                                   # heat runs 60x faster than motion
sim = alloy.build(scene)
traj = sim.run(1.5, record=["water"], every=25)                   # 1.5 s of motion = 90 s of heating
print(sim.heat("water"))         # per world: mean_K, min_K, max_K, melted, done, evaporated, particles
print(traj["water.temperature"].shape)                            # (frames, worlds, particles), K
print(sim.plugins.read()["particle_thermal"]["heater_area_m2"])  # [W, heaters]: touching area; check it is not 0
print(sim.check()["ok"])
```

- Temperatures are in kelvin (20 °C = 293.15 K); a value under 150 K raises with the °C conversion in its fix.
  `temperature=` also works on `add_pour`, and `scene.set_temperature(name, K)` heats a liquid or particle object
  card added earlier.
- A heater is a static box: an open box's floor (`add_open_box` names its parts `<name>_floor` and `<name>_wall0`
  to `<name>_wall3`), a box from `add_box(..., static=True)` passed to `set_heater`, or
  `scene.add_heater(name, size, pos=, temperature=)`.
- `scene.set_heat(...)` takes `time_scale`, `ambient`, `convection` and the other `ParticleThermal.Config` fields
  (section 7). `sim.set_heater("pan_floor", temperature=373.15)` changes a heater while running.
- Under the API, heat is the `ParticleThermal` plug-in, installed in `sim` (`sim.plugins["particle_thermal"]`), so
  `run`, `record`, `state` and `reset` work as usual. For a scene assembled with the engine layers, or for the
  body heaters of section 3, build the plug-in yourself: `therm = for_scene(built, objects=, heaters=, **config)`
  from `alloy.plugins.thermal.particle_scene` (objects `{name: (material, K)}`, heaters `{box: K | (mode, value)}`),
  then step `alloy.plugins.simulation(built, [therm])` with `step_period()`. Your own plug-ins can also run inside the
  API's `sim`: `alloy.build(scene, plugins=[...])`.

## 2. Materials

A phase material holds the thermal values, each with its source: density, heat capacity and conductivity (solid and
liquid), the melting range, latent heats, and optionally the boiling point, a doneness rate and a flow law. A heated
object uses its own material's heat data when it has some (`water`); else pass `heat_material=` to `add_liquid`,
`add_pour` or `set_temperature`:

| Material | Melts (°C) | Boils (°C) | Flow changes with T | Use |
|---|---|---|---|---|
| `water` | 0 | 100 | no | heating, boiling (SPH) |
| `dark_chocolate` | 30.2 - 36.2 | - | yes: viscosity, yield, friction, stiffness | melting pastes (MPM) |
| `beef_lean` | -10 - -1.7 | 100 | no | cooking to a core temperature (MPM) |
| `porridge_oat` | -10 - -1 | 100 | yes | reheating a thick liquid (SPH) |

`phase_materials.material(name)` shows the values and their sources. Names starting with `law_` are test materials.
A start temperature must be below the boiling point. The particles' own density (the solver's) must match the
material's: the plug-in warns when they differ by more than 1 % (SPH's rest density, or for MPM m/(2r)³, a particle's
mass over the cube of its diameter).

## 3. Heaters

| Mode | How to set it | What it does |
|---|---|---|
| temperature | `scene.set_heater(box, temperature=353.15)` | holds the box at that K; each touching particle gains h·A·(T_heater − T) |
| power | `scene.set_heater(box, power=800.0)` | puts that many W (× `heater_efficiency`) into the touching particles, shared by area |
| body | plug-in only: `heaters={box: ("body", 0.0)}` in `for_scene` | the box's body is a `Thermal` body: particles and body trade exactly opposite heat |

- A particle touches a heater when its centre is within its radius plus `heater_gap` of the box. A particle over a
  lower one rests on that one: it touches with the share of its footprint that no particle at least a quarter
  spacing nearer the heater covers. Only the part of a footprint over the heater's face counts, so a particle past a
  plate's edge takes no heat from it.
- The default gap, a quarter spacing, is right for liquids. SPH rests its bottom layer about 0.1 mm beyond a radius
  from a floor. With no gap, the quick start's touching area fell from 28 % of the floor to 4 % within 1.5 s; with
  the default it stays at 1.05-1.10 floors, because SPH packs its bottom layer about 10 % tighter than one particle
  per footprint.
- For melting solids and pastes (MPM), set `heater_gap` to the MPM voxel. MPM holds particles anywhere within a
  voxel of a collider: with a 0.2-spacing gap, the rim of a melting chocolate puddle never touched its plate and set
  again.
- h is `heater_conductance` (default 1000 W/(m²·K)) in series with half a particle of the material:
  h_eff = 1/(1/h + s/2k). A is the particle's footprint s², with s = (m/ρ)^(1/3) its spacing.
- `sim.plugins.read()["particle_thermal"]["heater_area_m2"]` is the touching area per heater; `heater_idle_J` is
  power a power heater put nowhere because nothing touched it.
- Change a heater while running: `sim.set_heater(box, temperature=...)` (or `power=`), one value for every world or
  a `(worlds,)` array. The mode is fixed at build.
- Body heaters (plug-in form): list `Thermal` before `ParticleThermal` in the plug-ins, then call
  `therm.link_thermal(<plug-in simulation>.plugins)`; `time_scale` must be 1.

## 4. Air

`scene.set_heat(convection=..., ambient=...)`: `convection` (W/(m²·K), default 0: off) cools or warms exposed
particles towards `ambient` (K). The exposed area is
measured per particle: one footprint on a flat face, about 2 on an edge, 3 on a corner, 6 for a lone particle, none
inside. An oven is hot air: `ambient=473.15, convection=25`. Per world: the plug-in's `ambient` input `[worlds]`
(`sim.plugins["particle_thermal"].set_input("ambient", values)`).

## 5. Time scale and step limits

`scene.set_heat(time_scale=k)` runs heat k times faster than motion: with `dt=0.01` and `k=60`, each step moves
particles for 0.01 s and heats them for 0.6 s. Use it when heating takes minutes but the motion it causes is slow (a
melt that slumps, a roast that sits still). Building checks the explicit step: conduction needs
dt·k ≤ 0.1·ρc·(support/2)²/k_thermal, and a heater or air needs dt·k·h·A/(m·c) < 1. The error says which one to change.

## 6. Melting, boiling, doneness

- **Melting** follows the enthalpy method: the temperature holds in the melting range while the latent heat goes in.
  `liquid` (φ, 0 to 1) is each particle's melted share.
- **Melting changes the flow** (`feedback="auto"`, the default) for materials with a flow law. On MPM the plug-in
  writes viscosity, yield stress, friction and stiffness per particle; on SPH it writes viscosity, and the SPH entry
  needs `viscosity="weiler2018"` (the plug-in warns when it would be ignored). These values replace the object card's.
  An SPH "solid" is only a thick liquid, so it creeps: solid melts belong on MPM.
- **Boiling**: at the boiling point, more heat evaporates mass (`vapor` per particle, `evaporated_kg` per world).
  An evaporated particle takes no more heat, but it stays in the liquid solver. Next to a heater such a layer holds
  the rest of the liquid off it, so heating stalls. A scene that boils off must check that `evaporated_in_solver`
  is 0.
- **Doneness** (`done`, 0 to 1) rises at a temperature-dependent rate for materials that have one (food).

## 7. Reading results and checking a run

`sim.plugins.read()["particle_thermal"]` (or `therm.read()` on the plug-in) gives arrays with the world first:

| Key | Shape | Meaning |
|---|---|---|
| `temperature`, `liquid`, `done`, `vapor` | [W, P] | per particle: K, melted share, doneness, evaporated share |
| `exposure` | [W, P] | m² of air contact, the last exchange (0 when convection is off) |
| `heater_area_m2`, `heater_idle_J` | [W, heaters] | touching area; heat a power heater could not place |
| `stored_J`, `heat_in_J`, `heat_out_J`, `boiled_J`, `evaporated_kg` | [W] | the energy ledger |
| `neighbor_overflow`, `evaporated_in_solver` | [W] | both must be 0 for a trustworthy run |

`sim.heat(name)` and `sim.temperature(name)` (also `"<name>.temperature"` in `sim.state()` and recordings) pick one
object's particles; spares not yet poured and evaporated particles are left out of `sim.heat`.

Checks for any run:
- the ledger closes: `stored_J` minus its start, plus `boiled_J`, equals `heat_in_J − heat_out_J`, to round-off;
- `neighbor_overflow` is 0 (else raise `neighbor_capacity`; an overflowing row loses heat one way);
- `heater_area_m2` is not 0 for a heater that should touch;
- with temperature heaters, no particle is hotter than the hottest heater or the air. A power heater has no
  temperature: it heats until something takes the heat away.

Main `Config` keywords (pass them to `scene.set_heat`): `heater_conductance`, `heater_efficiency`, `heater_gap` (m,
default a quarter spacing), `ambient`, `convection`, `time_scale`, `power_density` (W/kg per object,
microwave-like), `support` (m, default 2.2 spacings), `neighbor_capacity` (64), `feedback` ("auto" or "off"),
`buoyancy` (False; True: the buoyancy force on water, which does not yet give convection, section 8).

## 8. Limits

- Heat moves by conduction and by the particles' own motion. By default hot liquid does not rise, so water heated
  from below stays layered: after the quick start's 90 s its four layers average 53.6, 30.0, 22.7 and 22.5 °C from
  the bottom up. `scene.set_heat(buoyancy=True)` adds the thermal buoyancy force for water (NIST's density table):
  a hot region starts to rise within 10 % of the textbook rate, but the flow stalls after about a millimetre, so it
  does not convect or mix a heated pan (hot beside cold water in a cup: 1.2 mm of rise in 2 s, where real water
  trades places in under a second). To mix, stir (`docs/known_gaps/heat-no-buoyancy.md`).
- Boiled-off particles stay in the liquid solver and can hold the rest off a heater
  (`docs/known_gaps/heat-boiled-particles-stay.md`).
- Heat into or out of rigid bodies needs a body heater (plug-in form) and the `Thermal` plug-in; no radiation.
- Heaters are boxes. Every world has the same particles; the particle spacing sets the resolution (a few particles
  across a thin layer resolve its temperature poorly).
- Melting chocolate and cooking a beef block are MPM particle blocks built with the engine layers:
  `alloy/tasks/melt_chocolate.py` and `alloy/tasks/cook_core.py` show how.

## 9. Tasks

```bash
python -m alloy.tasks.melt_chocolate --worlds 4 --device cuda:0 --out rows.jsonl   # 3 x 3 x 1 cm chocolate, 50-60 °C plate
python -m alloy.tasks.cook_core --worlds 4 --device cuda:0 --out rows.jsonl        # 3 cm beef cube, 170-200 °C oven, core 74 °C
```

Each row reports success per world and the cost against the same scene without the plug-in.
