# Cards and engine layers

Every `alloy.Scene` becomes YAML cards before it is built (`scene.save(dir)` writes them). You can write cards
yourself when you need a field the API does not expose, such as precision features, or to keep a library of your
own objects.

## Card kinds

Each card is a YAML file with `schema: alloy.<kind>/1` and a `name` equal to its file name.

| Kind | Directory | Holds |
|---|---|---|
| `material` | `materials/` | density, `surface` (mu_static, mu_dynamic, restitution; contact uses one friction coefficient per pair, from `mu_dynamic`: `scenes.md`, Materials and friction), `elastic`, `fluid` properties |
| `pair` | `pairs/` | friction for one pair of materials |
| `object` | `objects/` | `shape` (primitive or mesh) + `material` + `motion: static` or `dynamic`; precision features |
| `scene` | `scenes/` | placed `objects` with `pose: [x, y, z, qx, qy, qz, qw]`, `robots`, `worlds`, `control_dt`, `randomize` |

- Quantities take units: `10 mm`, `0.5 MPa`, `1040 kg/m^3`. Primitive sizes in cards are **half** extents
  (boxes), `[radius]` (spheres) and `[radius, half height]` (cylinders, capsules); the Python API takes full sizes.
- Quantities you write by hand need a source and a note (a citation or how you got the number); the loader rejects
  a bare value with "needs a note (citation or derivation)":
  ```yaml
  density: {value: 1180, source: measured, note: "weighed 23.6 g, displaced 20.0 ml"}
  ```
  Sources: `measured`, `catalog`, `fit`, `model` (and `default`, for library defaults only).
- Library materials (`library/materials/`) give every value a source, a note and an `unc: [lo, hi]` range. About
  half of the values are `model` (the card author's estimate, explained in the note) and about 6 % `measured`; the
  rest are `catalog` or `fit`. Refer to materials by name, and read the card before relying on a value.

## Precision features

On both objects of a tight fit:
```yaml
precision_partners:
  - other_object:its_feature
precision_features:
  - name: shaft
    crop: [[-0.007, -0.007, -0.0255], [0.007, 0.007, -0.01]]     # box (object frame, m) around the mating region
    clearance: {value: 0.0002, source: catalog, note: "diametral clearance from the drawing"}
```
Static receivers (holes, sockets) usually use `shape: {mesh: {..., collision: sdf}}`. Build with
`profile="precision"`. Working pair: `library/objects/peg_d12.yaml` and `library/objects/block_bore_d12_2.yaml`,
placed by `library/scenes/demo_peg_d12.yaml`.

## Using your own card directory

`alloy.load("my_cards/scenes/my_scene.yaml")` builds a scene card; objects and materials are looked up in that card
directory first, then in the shipped `library/`. In Python, `alloy.Scene.load(...)` gives an editable copy, and
`alloy.Scene(roots=["my_cards"])` lets `scene.add_object(name, card)` place your own object cards.

## Engine layers under the API

`alloy.build` runs three engine steps; reach for them only when the API cannot express something.
1. `alloy.cards.library.CardLibrary(roots=[my_dir, ...])` reads cards (first root wins per name).
2. `alloy.scene.assembler.SceneAssembler(lib).assemble(lib.scene(name), device, world_count=N)` routes each object
   to a solver and returns a `BuiltScene` (`built.model` is a Newton `Model`, `built.report` the routing decisions).
3. `alloy.integration.wiring.build_simulation(built, profile, seed=...)` wires the solvers, coupling, health and
   CUDA graphs into the stepping simulation (`step_period()`, `state`, `control`, `report()`).

`sim.engine` and `sim.built` on an API `Sim` are these objects. Runnable: `examples/basics/09_engine_layers.py`.
