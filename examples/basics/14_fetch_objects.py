# SPDX-License-Identifier: Apache-2.0
"""Fetch objects from Alloy's object library on the Hugging Face Hub, then drop them on a table.

Run: python examples/basics/14_fetch_objects.py [object names ...]   (default: ycb_025_mug; CPU or GPU)
Needs ``pip install huggingface_hub`` and the network (the library is public: no login).
Files land in ``~/.cache/alloy/imports/<library>/``
(``$ALLOY_CACHE`` moves it) and download once; scenes, compose and the API find them there with no other setting
(on a fresh install the build also fetches a library object it does not have yet).
The command line does the same with JSON in and out: ``alloy fetch ycb_025_mug``.
"""

import json
import sys

import alloy
from alloy.assets.hub.fetch import fetch, list_objects

names = sys.argv[1:] or ["ycb_025_mug"]

print(list_objects(library="ycb_gso")["categories"])     # what there is, nothing downloaded: {"ycb_gso/gso": ...}
reply = fetch(names)                                     # a second run downloads 0 files and keeps them all
print(json.dumps({key: reply[key] for key in ("objects", "downloaded", "kept", "roots")}))

scene = alloy.Scene(dt=0.01)
scene.add_ground()
scene.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)                      # top at z = 0.4
for i, name in enumerate(names):
    scene.add_object(name, name, pos=(0.15 * i - 0.075 * (len(names) - 1), 0, 0.45))           # card name = object
sim = alloy.build(scene)
sim.run(1.0, record=[])
print({name: sim.state(world=0)[f"{name}.pos"].round(3) for name in names}, "ok:", sim.check()["ok"])
