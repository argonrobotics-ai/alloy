# Scene builder: a sentence in, a checked scene out

`alloy.compose.compose(sentence, variations=N)` picks objects from the card library, places them with support,
clearance and clutter rules, checks the scene after every placement and once more with a short simulation, fixes what
the checks flag, and returns the scene plus N variations as scene cards.

```python
import alloy
import alloy.compose

result = alloy.compose.compose("two wooden cubes and a mug on a table", variations=1, device="cpu")
paths = result.save("composed")                       # composed/scenes/<name>.yaml per scene, compose.json
print(result.ok, [str(p) for p in paths][:2])
sim = alloy.load(str(paths[0]))
sim.run(0.5, record=[])
print(sim.names()["bodies"], sim.check()["ok"])
```

- `alloy.compose.parse(sentence)` shows how a sentence is read (objects, counts, relations such as "in" or "on",
  clutter) before building anything.
- Command line: `python -m alloy.compose "a mug and two cubes on a table" --variations 2 --out composed` (JSON).
- Liquids, grains, cloth and cables: a sentence can also fill a container, pile grains, drape a cloth or plug in a
  cable ("a mug with 200 ml of water and a towel draped over a jar"); see "Liquids, grains, cloth and cables" in
  `alloy/compose/README.md` for what a sentence can say, a tilting pour and the limits.
- Fruit, sponges and other objects with a soft material are placed as rigid bodies (a card `<name>_rigid` beside the
  scenes, same shape and material), since a soft body steps far slower (three soft apples: about 90 s per 20 ms
  control step on a CPU, about 0.1 s rigid). Ask for "a soft apple" to get the soft body; `result.scenes[i].log` says
  which way each one went. In the scene the object is named `<name>_rigid` (`sim.names()`), and
  `alloy.compose.load_scene()` fetches the object it was copied from on a machine without it.
- A composed pour runs when you build the scene (`alloy.build(result.scene(i))` or
  `alloy.load(alloy.compose.load_scene(path))`); `alloy.check` on the saved card does not run the pour.
- More: `examples/compose/` (a robot in the scene, searching the library, a JSON spec instead of a sentence, soft
  scenes and their build speed).
