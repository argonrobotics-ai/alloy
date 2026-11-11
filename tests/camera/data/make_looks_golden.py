# SPDX-License-Identifier: Apache-2.0
"""Regenerate looks_golden.json from a real Alloy Blender export (needs the C3 pack.json and look.json).

    python make_looks_golden.py <pack.json> <look.json>

Writes the look specs, the generated Warp source and meta of every pack material, and a hash of its node tree.
No file paths from the export are kept.
"""
import hashlib
import json
import sys
from pathlib import Path

from alloy.camera.pbr.codegen import gen_material


def tree_hash(tree: dict) -> str:
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()


def main(pack_path: str, look_path: str) -> None:
    pack = json.loads(Path(pack_path).read_text())
    look = json.loads(Path(look_path).read_text())
    out = {}
    for name, mat in pack["materials"].items():
        src, meta = gen_material(name, mat, pack["textures"])
        for im in meta["images"]:  # keep the key and colour space; drop the pack-relative file name
            im["file"] = im["key"]
        out[name] = {"spec": look["materials"][name], "source": src, "meta": meta, "tree_sha256": tree_hash(mat["tree"])}
    textures = {k: {"colorspace": v["colorspace"]} for k, v in pack["textures"].items() if v["colorspace"] != "Linear Rec.709"}
    dest = Path(__file__).with_name("looks_golden.json")
    dest.write_text(json.dumps({"materials": out, "textures": textures}, indent=0, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
