# alloy create and alloy audit

`alloy create` turns your own object into a checked object card that Alloy can simulate. The object can be a
mesh file, one photo, or a text description. `alloy audit` stress-tests a whole scene and reports what breaks.

## alloy create

```bash
python -m alloy.create --mesh cup.stl --material plastic --out my_objects          # a mesh (units inferred)
python -m alloy.create --photo bowl.jpg --name yellow_bowl --out my_objects        # one photo
python -m alloy.create --text "a red ceramic coffee mug with a handle" --out my_objects
python -m alloy.create --mesh arm_link.urdf --out my_objects                       # MJCF / URDF / USD, joints kept
```

```python
import alloy.create

r = alloy.create.create(mesh="cup.stl", material="plastic", out="my_objects")
r.ok            # True: the card is valid and passed the drop test
r.estimates     # ['size', 'material', 'mass']: values that are guesses
r.verify        # what a person should check, most important first
print(r.to_json())
```

What it does, in order:

1. **Mesh.** It uses your mesh, or makes one from the photo or text (see *The generator*). A generated mesh is
   turned onto the face it would rest on, so `up` is ignored for photos and text.
2. **Category.** It works out what the object is (`mug`, `bowl`, ...) from `category=`, the text or the name.
   The category gives the typical size, mass and material.
3. **Real size.** The size comes from `size=` (longest side in metres) or `units=` (`m`, `cm`, `mm`, `in`, `ft`).
   Without either, the units are inferred from the category. A generated mesh gets the category's typical size,
   so a photo or text of an object whose category is unknown needs `size=` or `category=` (the error says so).
   Anything below 3 mm or above 3 m is refused, and the error says which argument fixes it.
4. **Physics mesh.** It builds a watertight collision mesh, convex hulls (CoACD), a visual GLB and a thumbnail.
5. **Material and mass.**
   - Material: `material=` (a word or a library material name), else a material word in the text, else the
     category's material, else ABS plastic. Words map to library materials: `plastic` -> `abs`, `wood` ->
     `wood_beech`, `steel`/`metal`/`iron` -> `steel_carbon`, `aluminium` -> `aluminium_6061_t6`, `glass` ->
     `glass_soda_lime`, `ceramic`/`porcelain`/`stoneware` -> `stoneware_glazed`, `paper` -> `paper_80gsm`,
     `cardboard` -> `cardboard_corrugated`, `brass` -> `brass_c360`, `copper` -> `copper_c110`. `create` makes
     rigid objects, so a word never picks a soft material (rubber, foam, fabric): a "rubber duck" gets the
     category's or a plastic material, and the result says so. Naming a soft library material (`silicone_40a`)
     tries a soft body; if the engine cannot drop-test it, the result has a `drop_test` error with the fix.
     `material=` is read in any case and may be several words: `Steel` is `steel`, `stainless steel` picks a
     stainless card, `Stainless 304` is `stainless_304`. A `material=` that is neither a library material nor a
     material word (`titanium`) is an error listing the words, and so is a liquid or a granular library material
     (`water`, sand): create makes a solid and would keep only its density. A word in the text is only a hint.
   - Mass: `mass=` (kg) with `mass_source=` (for example "kitchen scale" or a datasheet). Without it, the
     category's typical mass with its range is used, unless you named a material the category does not have.
     Failing that, the material's density times the mesh volume. That fills the closed mesh, so a hollow object
     (a bottle, a closed box) weighs less than this estimate: weigh it and pass `mass=`.
6. **Cards and drop test.** It writes `objects/<name>.yaml` and `materials/<name>_eff.yaml` to `out`, then drops
   the object onto a plane with `alloy.check`.

A missing citation never stops it: values without one are marked `ESTIMATE` with a range.

| Argument | Meaning |
|---|---|
| `mesh` / `photo` / `text` | exactly one input: OBJ, STL, GLB/GLTF, PLY, OFF, MJCF, URDF or USD; a JPEG/PNG; a description |
| `name` | card name (snake_case); default: the file name, or the first words of the text |
| `size`, `units`, `up` | longest side in metres; mesh units; which mesh axis is up (`z`, `y`, `-y`, `x`) |
| `mass`, `mass_source` | kg, and where that number comes from |
| `material`, `category` | a material word or library name; what the object is (`mug`, `bowl`, ...) |
| `out` | the folder for the cards (default `my_objects`); names already used by another library are refused |
| `device` | where the drop test runs: `cpu` or `cuda:0` (default: the GPU if there is one) |
| `check=False` / `--no-check` | skip the drop test |
| `--request FILE` | all of the above as one JSON object (`-` reads stdin), e.g. `{"mesh": "cup.stl", "units": "mm"}` |

### Model files (MJCF, URDF, USD)

A model file becomes one object card per body tree, joints kept:

- A file with one object names its card `name` (default: the file's name), so `Scene.add_object("drawer",
  "my_drawer")` works after `--name my_drawer`. A file with several objects names them `<name>_<body>`; the result's
  `objects` lists them.
- Joints without an actuator stay as passive joints. A URDF has no actuators, so every jointed URDF becomes an
  articulated object: with a moving joint its base link is fixed in place, with only fixed joints it is one free
  object. A file with actuators is a robot and is refused: use `Scene.add_robot`.
- Masses, materials, units (metres) and axes are the file's own. `material`, `mass`, `size`, `units` and `up` cannot
  be applied to a model file and raise an error that says so; set them in the file. `values` reports each object's
  mass as the sum of the file's body masses (tag `MODEL`).

### The result (JSON, `schema: alloy.create/1`)

```json
{"schema": "alloy.create/1", "ok": true, "name": "yellow_bowl", "card": "my_objects/objects/yellow_bowl.yaml",
 "estimates": ["size", "material", "mass", "shape"],
 "verify": ["size: measure its longest side (typical ceramic_bowl 12-25 cm); pass size=",
            "material: check that a ceramic_bowl of stoneware_glazed is right",
            "mass: weigh it (typical ceramic_bowl 350-750 g)",
            "shape: look at the thumbnail: the hidden back is guessed and openings may come out closed"],
 "values": [{"field": "size", "value": 0.16, "unit": "m", "tag": "ESTIMATE", "lo": 0.12, "hi": 0.25,
             "source": "typical ceramic_bowl longest side (...)"},
            {"field": "mass", "value": 0.5, "unit": "kg", "tag": "ESTIMATE", "lo": 0.35, "hi": 0.75,
             "source": "typical ceramic_bowl: 3 restaurant bowl listings ...", "evidence": ["https://..."]}],
 "check": {"ok": true, "errors": 0, "warnings": 0, "findings": []},
 "generator": {"backend": "remote", "models": {"image_to_3d": {"name": "TripoSR", "licence": "MIT"}}},
 "files": {"object": "...", "material": "...", "physics.obj": "...", "physics.hulls.obj": "...",
           "visual.glb": "...", "thumb.png": "..."}}
```

Every value carries one of these tags:

- `USER`: you gave it.
- `CITED`: you gave it with a source.
- `LIBRARY`: from a library material card.
- `MODEL`: read from your MJCF, URDF or USD file.
- `ESTIMATE`: a typical value with a range. Check it.

Only the JSON goes to stdout; logs and the engine's start-up lines go to stderr. The command exits 0 when the
card is made and passes, 1 when the drop test fails, and 2 for a bad request (with `error`, `message`, `fix` and
`field` as JSON). A card that fails the drop test is still written so you can look at it; running again with
the same name replaces it.

If the drop test says the object slides or tipped over, the mesh file is probably lying on its side: pass `up=`
(the axis that is up in your file: `z`, `y`, `-y` or `x`). The result's `verify` list says so too.

### Using the new object

Pass the folder wherever Alloy looks for cards:

```bash
# scene.json pins your objects by name:
#   {"items": [{"query": "yellow_bowl", "object": "yellow_bowl"}, {"query": "cube", "words": ["wooden"]}]}
python -m alloy.compose build --spec scene.json --root my_objects --out my_scenes --device cpu
python -m alloy.check my_scenes/scenes/scene.yaml --library my_objects --library my_scenes
```

In Python, use `alloy.compose.compose(spec, roots=["my_objects"])`. The spec can also set `"support"` (`table`,
`floor` or an object name), `"clutter"` (`tidy`, `normal`, `cluttered`) and `"arrangement"`; see
`python -m alloy.compose --help` and `examples/compose/`.

### The generator (photo and text)

| Step | Model | Licence |
|---|---|---|
| text to image | Kandinsky 2.2 | Apache-2.0 |
| background removal | rembg (U^2-Net) | MIT, weights Apache-2.0 |
| image to 3D | TripoSR | MIT (code and weights) |
| marching cubes | scikit-image | BSD-3-Clause |

All four allow commercial use. FLUX.1 [schnell] (Apache-2.0) can replace Kandinsky: set `ALLOY_GEN3D_T2I=flux` and
log in to Hugging Face with an account that has accepted its terms.

### Run the generator on your own GPU

You need a Linux machine with an NVIDIA GPU of at least 16 GB (for example an L4, A10, RTX 4090 or larger), its
driver, Python 3.10 or newer, git, and about 30 GB of disk (10 GB of packages, 17 GB of models). The GPU machine does
not need Alloy's physics engine: the server imports only `alloy.create` from an Alloy checkout, NumPy and the models'
packages.

```bash
# on the GPU machine, from an Alloy checkout: a venv, the packages, TripoSR, and a serve script (about 3 minutes)
bash examples/create/generator_setup.sh ~/alloy-generator
ALLOY_GEN3D_TOKEN=<a secret you choose> ~/alloy-generator/serve.sh      # port 8765; HOST=127.0.0.1 keeps it local

# on the machine that runs alloy create
export ALLOY_GEN3D_URL=http://<gpu-machine>:8765 ALLOY_GEN3D_TOKEN=<the same secret>
python -m alloy.create --text "a wooden cutting board" --out my_objects
```

- **Network.** Keep port 8765 on a private network, or reach it through an SSH tunnel
  (`ssh -L 8765:localhost:8765 <gpu-machine>`, then `ALLOY_GEN3D_URL=http://localhost:8765`). With
  `ALLOY_GEN3D_TOKEN` set, every request must carry it. Without it, anyone who can reach the port can use the GPU.
- **First request.** The first request downloads the models (about 10 GB, into `~/alloy-generator/hf`; set
  `HF_HOME` to reuse a cache) and loads them (1-2 minutes). After that, generating from text takes about 30 s and
  from a photo about 5 s. A whole `create` with the CPU drop test takes about one to two minutes. The server
  handles one request at a time and keeps both models in GPU memory (about 14 GB).
- **One machine.** On the GPU machine itself, with Alloy installed in the same environment, `alloy create` runs the
  generator in-process: set `ALLOY_TRIPOSR_DIR=~/alloy-generator/TripoSR` and leave `ALLOY_GEN3D_URL` unset.
- **Check it.** `curl http://<gpu-machine>:8765/health` returns `{"ok": true, "models": {...}, "token_required": ...,
  "busy": ..., "loaded": [...]}`: the models with their licences, whether requests need the token, whether a
  request is running, and which models are already in GPU memory (none right after a start).

When a photo or text request fails, the error names the cause and the fix:

| Error says | What to do |
|---|---|
| needs a token and none is set / refused: the token is not the server's | set `ALLOY_GEN3D_TOKEN` to the server's token |
| nothing is listening | start the server (`serve.sh` or `python -m alloy.create.serve`); check the port in the URL |
| answered HTTP 400 | the request was not a generation request (the message says which field); `alloy create` builds it for you |
| answered without a mesh | the URL is another service: point `ALLOY_GEN3D_URL` at `python -m alloy.create.serve` |
| cannot reach the generator | check the URL and network, and that the GPU machine is running |
| dropped the connection and no longer answers | the machine went away (spot GPU machines can be preempted): restart it, wait for `/health`, retry |
| no answer within 900 s | the server is busy or still loading the models: check `/health` and its log, retry |

A connection dropped mid-request is retried once when the server still answers `/health`.

### Good photos and descriptions

- Show one object, whole, on a plain background, seen from the side and a little above.
- Pass `size=` when you know the size. The image has no scale. A text that names a known object ("a mug",
  "a cutting board") gets that object's typical size and mass; for anything else pass `size=` or `category=`.
- Look at `meshes/<name>/thumb.png`. The back of the object is guessed, and openings (a mug's mouth) may come out
  closed. If `verify` says the mesh "is not a closed solid", the generator lost part of the object (often one that
  blends into the background): the card still simulates as the closed shape built from it, but check the
  thumbnail and retry with another photo or description.
- Objects with a round bottom (a ball, a rubber duck) may keep rolling, and the drop test then fails with "slides"
  or "still moving". That is the shape, not the card: describe or photograph an object that stands on a flat base,
  or pass `check=False` if rolling is what you want.

Without a GPU or `ALLOY_GEN3D_URL`, photo and text requests fail with an error whose `fix` says what to set.

## alloy audit

```bash
python -m alloy.audit my_scenes/scenes/scene.yaml --library my_objects --library my_scenes --rounds 3 --episodes 16
```

```python
import alloy.audit

report = alloy.audit.audit("my_scenes/scenes/scene.yaml", rounds=3, episodes=16, library=["my_objects"])
report.ok, report.counts()          # (True, {'error': 0, 'warn': 1, 'info': 0})
print(report.to_json())
```

It runs two kinds of test:

- `alloy.check` over `rounds` seeds. Each seed randomizes the worlds differently.
- The bully agent (`alloy.bully`) over `episodes` random legal episodes of `--seconds` each (default 2 s). An
  episode drives the scene's actuators with random commands inside their ranges and rate limits; a scene without
  actuators gets random pushes on its free objects (at most 0.2 x each object's weight per axis, so a push never
  lifts anything). A finding is an episode that reached S1 (impossible: non-finite state, deep penetration, energy
  created, an impossible speed; reported as `error`), S2 (degraded; `warn`) or S3 (far from the origin; `info`).

On CPU, 16 episodes of a three-object scene take about 5 minutes; a GPU runs them in parallel. A finding at the
first tick means the start pose is already wrong: fix the `alloy.check` overlap findings first.

The findings are merged and sorted by severity (`error`, `warn`, `info`). Each finding has:

- `kind`: the check, or the bully class.
- `message` and `fix`.
- `objects`: the objects or bodies involved.
- `rate`: the share of rounds or episodes in which it appeared.
- `first_time_s`.
- `replay`: the `alloy.check` command with its seed, or the bully episode handle.

`ok` is true when there are no errors. The command exits 0 (ok), 1 (errors found) or 2 (bad arguments). The JSON:

```json
{"schema": "alloy.audit/1", "scene": "scene", "ok": true, "counts": {"error": 0, "warn": 1, "info": 0},
 "findings": [{"source": "check", "severity": "warn", "kind": "rest_drift", "message": "...", "fix": "...",
               "objects": ["green_mug"], "rate": 0.33, "seen": 1, "first_time_s": null,
               "replay": {"seed": 1, "command": "python -m alloy.check ... --seed 1 --library my_objects --json"}}],
 "runs": {"rounds": 3, "episodes": 16, "bully": true,
          "bully_summary": {"counts": {"clean": 16, "S1": 0, "S2": 0, "S3": 0}, "by_cause": {}, "s1_upper95": 0.19,
                            "channel": "push: force +-0.2 x weight per axis on 3 free bodies"}},
 "notes": [], "timings_s": {"check_rounds": 44.6, "bully": 318.9, "total": 363.4}}
```

A bully finding's `replay` holds a `handle` and, when the scene uses card folders, `library` (the folders):
`alloy.bully.replay_finding(replay["handle"], library=alloy.audit.library(replay.get("library", [])))` runs that
one episode alone.

A scene file's own folder (`my_scenes` for `my_scenes/scenes/scene.yaml`) is always searched, as in `alloy.check`,
so its generated cards and robot files resolve without `--library`; a check replay command names the file by its
full path. A scene card that does not load raises `alloy.SceneError` with the line to edit.

Through `alloy audit` and the MCP tool, a scene given as a recipe, Scene JSON, a `.json` file or a robot file is
first written as cards to a folder that stays, so the replays work after the call: `--cards <folder>`, else a folder
per scene under `~/.cache/alloy/audit` (`$ALLOY_CACHE/audit`). The reply names it in `cards`.
`s1_upper95` is the 95 % upper bound on the share of episodes that would reach S1 (3/N when none did).

In Python, `report.summary()` (and `result.summary()` for create) gives the same as a few lines of text.

Pass `--no-bully` to run the check rounds only (an older Alloy without the bully agent does that and says so in
`notes`).

## Examples

- `examples/create/mesh_to_card.py`: a mesh in millimetres becomes a card. CPU only.
- `examples/create/photo_to_card.py`: a photo becomes a card. Needs a generator.
- `examples/create/text_to_card.py`: a description becomes a card. Needs a generator.
- `examples/create/create_compose_audit.py`: create a card, compose a scene with it, then audit the scene.
- `examples/create/generator_setup.sh`: set up and serve the generator on your own GPU machine.
