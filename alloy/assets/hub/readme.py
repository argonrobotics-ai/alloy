# SPDX-License-Identifier: Apache-2.0
"""The dataset card (``README.md``) of the library dataset, filled with the per-library counts.

Example: ``readme({"ycb_gso": {"objects": 1111, "bytes": 2e9, ...}}, ["CC-BY-4.0"])`` returns the Markdown.
"""
from __future__ import annotations

from string import Template
from typing import Any

from alloy.assets.hub.source import DEFAULT_REPO

KITCHEN_URL = "https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
#: library -> (source, for the table; what its cards are, for the list under the title)
ABOUT = {
    "ycb_gso": ("YCB Object and Model Set and Google Scanned Objects",
                "household objects scanned for the YCB Object and Model Set and Google Scanned Objects: watertight "
                "physics mesh, convex hulls, textured glTF. The mass is a measured or catalogue value where one was "
                "found, else an estimate."),
    "robocasa": ("Lightwheel kitchen assets (NVIDIA PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF, as "
                 "packaged by RoboCasa)",
                 f"kitchen objects and fixtures made by Lightwheel, from NVIDIA's [PhysicalAI-Robotics-Manipulation-"
                 f"Objects-Kitchen-MJCF]({KITCHEN_URL}) dataset as packaged by RoboCasa (its `objects_lightwheel` and "
                 "`fixtures_lightwheel` packs): MJCF models with their meshes and textures. The collision hulls, the "
                 "masses (all estimated) and the materials are Alloy's, and some cabinets are assembled by Alloy from "
                 "the packs' door panels and handles. RoboCasa's kitchens and tasks are not here, nor its other object "
                 "packs."),
    "gymanything": ("Gym-Anything for Robotics",
                    "object parts taken from the source code of 28 simulated scenes of [Gym-Anything for Robotics]"
                    "(https://github.com/gym-anything/robotics): one card per rigid part, so an articulated object is "
                    "several cards, with the colours and materials those scenes render with. These are objects "
                    "only: no Gym-Anything task, robot or score is here."),
    "objaverse": ("Objaverse 1.0 (objects whose uploader declared CC BY 4.0 or CC0)",
                  "objects from Objaverse 1.0 in categories a robot handles (food, office, kitchen, tools, containers, "
                  "packaging, lab), limited to those whose uploader declared CC BY 4.0 or CC0. Every mass is an "
                  "estimate from the object's category."),
    "alloy": ("Alloy's own models", "props authored for Alloy's own scenes."),
}

CITATIONS = """```bibtex
@misc{alloy_library,
  title  = {Alloy object library},
  author = {{Alloy contributors}},
  year   = {2026},
  howpublished = {Hugging Face dataset},
}
@article{calli2015ycb,
  title   = {Benchmarking in Manipulation Research: Using the {Yale-CMU-Berkeley} Object and Model Set},
  author  = {Calli, Berk and Walsman, Aaron and Singh, Arjun and Srinivasa, Siddhartha and Abbeel, Pieter and
             Dollar, Aaron M.},
  journal = {IEEE Robotics \\& Automation Magazine},
  year    = {2015},
}
@inproceedings{downs2022gso,
  title     = {Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items},
  author    = {Downs, Laura and Francis, Anthony and Koenig, Nate and Kinman, Brandon and Hickman, Ryan and
               Reymann, Krista and McHugh, Thomas B. and Vanhoucke, Vincent},
  booktitle = {ICRA},
  year      = {2022},
}
@inproceedings{nasiriany2024robocasa,
  title     = {RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author    = {Nasiriany, Soroush and Maddukuri, Abhiram and Zhang, Lance and Parikh, Adeet and Lo, Aaron and
               Joshi, Abhishek and Mandlekar, Ajay and Zhu, Yuke},
  booktitle = {RSS},
  year      = {2024},
}
@misc{nvidia_kitchen_mjcf,
  title  = {{PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF}},
  author = {{NVIDIA}},
  howpublished = {Hugging Face dataset,
    \\url{https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF}},
}
@inproceedings{deitke2023objaverse,
  title     = {Objaverse: A Universe of Annotated 3D Objects},
  author    = {Deitke, Matt and Schwenk, Dustin and Salvador, Jordi and Weihs, Luca and Michel, Oscar and
               VanderBilt, Eli and Schmidt, Ludwig and Ehsani, Kiana and Kembhavi, Aniruddha and Farhadi, Ali},
  booktitle = {CVPR},
  year      = {2023},
}
```"""


def _table(libs: dict[str, dict[str, Any]]) -> str:
    """One row per library: its source, cards, estimated masses, flagged cards, size and declared licences."""
    lines = ["| Library | Source | Cards | Mass estimated | Flagged | Size | Licences, as the source declares |",
             "|---|---|---:|---:|---:|---:|---|"]
    for lib, s in sorted(libs.items()):
        src = ABOUT.get(lib, (lib, ""))[0]
        lic = ", ".join(f"{k} ({v:,})" for k, v in sorted(s["licences"].items()))
        lines.append(f"| `{lib}` | {src} | {s['objects']:,} | {s.get('mass_estimate', 0):,} | {s.get('flagged', 0):,} "
                     f"| {s['bytes'] / 1e9:.2f} GB | {lic} |")
    n, b = _total(libs, "objects"), _total(libs, "bytes")
    lines.append(f"| **all** | | **{n:,}** | **{_total(libs, 'mass_estimate'):,}** | **{_total(libs, 'flagged'):,}** "
                 f"| **{b / 1e9:.2f} GB** | |")
    return "\n".join(lines)


def _total(libs: dict[str, dict[str, Any]], key: str) -> int:
    return sum(s.get(key, 0) for s in libs.values())


# SPDX id -> Hugging Face metadata licence id
HF_LICENSE = {"CC-BY-4.0": "cc-by-4.0", "CC0-1.0": "cc0-1.0", "MIT": "mit", "Apache-2.0": "apache-2.0",
              "ODC-By-1.0": "odc-by"}


def _front(libs: dict[str, dict[str, Any]]) -> str:
    n = sum(s["objects"] for s in libs.values())
    size = "1K<n<10K" if n < 10_000 else "10K<n<100K"
    ids = sorted({HF_LICENSE.get(k, "other") for s in libs.values() for k in s["licences"]})
    return f"""---
pretty_name: Alloy object library
license: [{", ".join(ids)}]
tags: [robotics, simulation, physics, 3d, meshes, manipulation]
size_categories: [{size}]
configs:
- config_name: index
  data_files: index.jsonl
---"""


TEMPLATE = Template("""$front

# Alloy object library

$n object cards for the [Alloy](https://alloy.argonrobotics.ai) robot simulator, from these sources:

$about

Each object is a card (YAML: shape plus material values: density, friction, restitution, stiffness, each with where
it comes from and an uncertainty range) together with its geometry (a physics mesh with a convex-hull collision
sidecar and a visual glTF, or an MJCF model with its meshes and textures) and, where the library has one, a rendered
thumbnail and a look.

Most physical values are estimates, not measurements. $mass_est of the $n masses are estimates (from the object's
category or material and its mesh volume). Friction, restitution and stiffness are values for the object's material,
not measured on the object. The importers flagged $flagged of the cards (an implausible mass or size, or a failed
physics check); the reason is in `flags`. See [Values: where they come from](#values-where-they-come-from).

The dataset is public: anyone can download it, with no access request, account or token. Each object's card records
the licence its source declares ($licence_counts); check it, and the credit line in `ATTRIBUTION.md`, before
commercial use (see [Licences and attribution](#licences-and-attribution)).

$table

## Use it with Alloy

```bash
pip install huggingface_hub          # once; Alloy downloads with it (no account or token needed)
alloy fetch --list --library ycb_gso                  # what is there (nothing downloaded)
alloy fetch ycb_025_mug                               # one object
alloy fetch --library robocasa --category blenders    # a category
alloy fetch --library objaverse                       # a whole library
```

Objects go to `~/.cache/alloy/imports/<library>/` (or `$$ALLOY_CACHE/imports/<library>/`); files already there are
not downloaded again. A Hugging Face login (`hf auth login` or `HF_TOKEN`) is optional: the Hub gives logged-in
downloads a higher rate limit, which helps when fetching whole libraries. Alloy finds fetched objects by card name:

```python
import alloy

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_object("mug", "ycb_025_mug", pos=(0, 0, 0.1))
sim = alloy.build(scene)
```

`alloy compose "a mug and a bowl on a table"` picks from every fetched object. `ALLOY_ASSETS_REPO=<owner>/<name>`
points Alloy at another copy of this dataset (default `$repo`); `ALLOY_ASSETS_REVISION` pins a revision.

## Layout

```
index.jsonl                      one row per object
objects/<library>/<name>/        card.yaml, material.yaml, meshes, physics.hulls.obj, textures, thumb.png
libraries/<library>/             files shared by a library's objects (manifest.json, shared materials, looks)
libraries/textures/              CC0 texture sets (Poly Haven, ambientCG) for scene variants; credits in its README
denoiser/                        weights of Alloy's learned image denoiser (Apache-2.0); see its README
LICENSES/  ATTRIBUTION.md        licence texts and every object's credit line
```

An `index.jsonl` row: `name`, `library`, `category`, `title`, `licence`, `attribution`, `source` (upstream URL),
`showcase_ok`, `mass_basis`, `estimate`, `sources`, `flags` (only when flagged), `size` (bytes), `sha256` (the object's tree digest) and `files`
(each file's name in the object folder, its `path` under the library root, `sha256` and `bytes`).

Without Alloy:

```python
from huggingface_hub import snapshot_download
snapshot_download("$repo", repo_type="dataset", allow_patterns=["objects/ycb_gso/ycb_025_mug/*"])
```

## Values: where they come from

Every card value records its source, with a note. In `index.jsonl`:

- `mass_basis` says where an object's mass comes from, in the importer's own words: for example `measured`,
  `sourced` or `datasheet` (a measurement, a catalogue or a datasheet), `authored` or `derived` (set or computed by
  the object's author), or `estimate`. $mass_est of the $n masses are estimates (the table above counts them per
  library); such a card's note usually starts with `ESTIMATE`.
- `sources` gives the kind of each material value: `measured`, `catalog` (a cited datasheet or handbook value for the
  material), `fit` (a density fitted to a measured or catalogue mass), or `model` (a stated model, a typical value or
  a mid-range pick). Friction, restitution and stiffness are values for the object's material (almost all `catalog`
  or `model`), not measurements of the object.
- `estimate: true` marks the $est of $n objects whose mass or at least one material value is an estimate.
- `flags` gives the reason on a card its importer flagged ($flagged of $n).

Read the card's note before you rely on a value.

## Licences and attribution

Each object keeps the licence its source declares ($licence_counts): the `licence` field of `index.jsonl`, with
the licence texts in `LICENSES/` ($licenses). The licences are recorded as declared; for Objaverse that is what each
uploader declared. Check the licence and the credit line of the objects you use before commercial use.
CC BY 4.0 objects need credit when you share them or images of them: `ATTRIBUTION.md` lists every object's credit
line (also the `attribution` field of the index). Alloy changed the objects (repaired and simplified
meshes, convex hulls, re-exported visuals, estimated physical values); the upstream files are linked in `source`.

**Brands.** Scans of branded products (most of Google Scanned Objects, and some objects of the other libraries) show
third-party trademarks: packaging, logos, product names. A Creative Commons licence covers the scan, not the
trademark. `showcase_ok: false` marks the $branded objects whose name or label shows a brand (or may); keep them out
of marketing images, docs and launch material. The mark is about showing a brand; it does not change an object's
licence.

## Citation

If you use this library, cite Alloy and the upstream datasets you use:

$citations
""")


#: SPDX id -> the name a reader knows
LICENCE_NAME = {"CC-BY-4.0": "CC BY 4.0", "CC0-1.0": "CC0 1.0", "MIT": "MIT", "Apache-2.0": "Apache 2.0"}


def licence_counts(libs: dict[str, dict[str, Any]]) -> str:
    """Objects per licence over every library, most first. Example: ``"CC BY 4.0: 3,902 objects, CC0 1.0: 1,128"``."""
    total: dict[str, int] = {}
    for s in libs.values():
        for k, v in s["licences"].items():
            total[k] = total.get(k, 0) + v
    items = sorted(total.items(), key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{LICENCE_NAME.get(k, k)}: {v:,}" + (" objects" if i == 0 else "")
                     for i, (k, v) in enumerate(items))


def readme(libs: dict[str, dict[str, Any]], licenses: list[str]) -> str:
    """The dataset card for the per-library summary ``libs`` (``alloy.assets.hub.card.summary``) and the licence
    files written."""
    n = _total(libs, "objects")
    about = "\n".join(f"- `{lib}` ({s['objects']:,}): {ABOUT.get(lib, (lib, ''))[1]}"
                      for lib, s in sorted(libs.items()))
    return TEMPLATE.substitute(front=_front(libs), n=f"{n:,}", table=_table(libs), about=about, repo=DEFAULT_REPO,
                               est=f"{_total(libs, 'estimate'):,}", mass_est=f"{_total(libs, 'mass_estimate'):,}",
                               flagged=f"{_total(libs, 'flagged'):,}", licenses=", ".join(licenses),
                               branded=f"{n - _total(libs, 'showcase_ok'):,}", citations=CITATIONS,
                               licence_counts=licence_counts(libs))
