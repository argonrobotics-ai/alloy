# SPDX-License-Identifier: Apache-2.0
"""Offline conversion of recorded MuJoCo scenes (``model.mjb``) into vendored, self-contained MJCF.

SPEC §7 WP13: Newton's MJCF importer cannot read ``.mjb``, and ``mj_saveLastXML`` refuses a model loaded from a
binary ("No XML model loaded", MuJoCo 3.16.0), so the compiled model is written back as explicit MJCF at full
precision (:mod:`alloy.tasks.mjcf_tree`, :mod:`alloy.tasks.mjcf_parts`). Meshes go to
``meshes/*.msh`` and every file is listed in ``SHA256SUMS.json``. CPU MuJoCo is used here offline only (README
ground rule); nothing in this module runs inside an Alloy simulation loop.

CLI: ``python -m alloy.tasks.mjb_convert <recorded trace dir> <robots root> <scene name>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from alloy.core.errors import TraceError
from alloy.tasks.msh import write_manifest
from alloy.tasks.mjcf_parts import (
    check_supported,
    write_actuators,
    write_compiler,
    write_contacts,
    write_equalities,
    write_keyframes,
    write_option,
)
from alloy.tasks.mjcf_tree import model_names, write_assets, write_tree
from alloy.tasks.split_scene import REPORT_NAME, SceneSplit, split_scene_hulls, write_report

log = logging.getLogger("alloy.tasks")

SCENE_XML = "scene.xml"
MESH_DIR = "meshes"
INPUT_KINDS = ("ctrl", "qfrc", "xfrc", "afr", "eq")


@dataclass(frozen=True)
class RecordedTrace:
    """One recorded episode window from ``[internal reference]/<name>``.

    Recording layout (verified by CPU replay, 2026-11-01): row ``i`` of every input is written before ``mj_step``
    ``i``; ``ref_qpos[k]`` is ``qpos`` after ``(k + 1) * ref_every`` steps; ``state0`` is ``mjSTATE_INTEGRATION``.
    Inputs are stored as a first row (``*_first``) plus the columns that vary (``*_cols``, ``*_var``).
    """

    name: str
    root: Path
    model: mujoco.MjModel
    inputs: dict[str, np.ndarray]           # kind -> [steps, width] dense, float64
    state0: np.ndarray                      # mjSTATE_INTEGRATION
    ref_qpos: np.ndarray                    # [samples, nq]
    meta: dict = field(default_factory=dict)
    ref_every: int = 50

    @property
    def steps(self) -> int:
        return int(self.inputs["ctrl"].shape[0])

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)


def _dense_input(z: np.lib.npyio.NpzFile, kind: str) -> np.ndarray:
    first = np.asarray(z[f"{kind}_first"], dtype=np.float64)
    cols = np.asarray(z[f"{kind}_cols"], dtype=np.int64)
    var = np.asarray(z[f"{kind}_var"], dtype=np.float64)
    dense = np.repeat(first[None, :], var.shape[0], axis=0)
    if cols.size:
        dense[:, cols] = var
    return dense


def load_recorded(root: Path) -> RecordedTrace:
    """Load a recorded trace directory (``model.mjb``, ``inputs.npz``, ``state0.npy``, ``ref_qpos.npy``)."""
    if not (root / "model.mjb").is_file():
        raise TraceError(f"{root}: no model.mjb")
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    with np.load(root / "inputs.npz") as z:
        inputs = {kind: _dense_input(z, kind) for kind in INPUT_KINDS}
    state0 = np.load(root / "state0.npy")
    want = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    if state0.shape != (want,):
        raise TraceError(f"{root}: state0 has {state0.shape}, mjSTATE_INTEGRATION needs ({want},)")
    meta_path = root / "trace.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    ref = np.load(root / "ref_qpos.npy")
    return RecordedTrace(root.name, root, model, inputs, state0, ref, meta)


def decompile(m: mujoco.MjModel, mesh_store: Path, model_name: str = "scene",
              keyframes: bool = False) -> ET.ElementTree:  # fmt: skip
    """MJCF tree equal to ``m`` for a file one level below ``mesh_store``'s parent (meshes are ``../meshes/*``);
    stores the meshes as a side effect. ``keyframes`` also writes ``m``'s keyframes (the recorded scenes keep their
    start state in the trace instead)."""
    check_supported(m)
    names = model_names(m)
    root = ET.Element("mujoco", model=model_name)
    write_compiler(root)
    write_option(root, m)
    write_assets(root, m, names, mesh_store, f"../{MESH_DIR}")
    write_tree(root, m, names)
    write_actuators(root, m, names)
    write_equalities(root, m, names)
    write_contacts(root, m, names)
    if keyframes:
        write_keyframes(root, m)
    tree = ET.ElementTree(root)
    ET.indent(tree, space=" ")
    return tree


def write_scene(m: mujoco.MjModel, robots_root: Path, scene: str, split: SceneSplit | None = None,
                keyframes: bool = False) -> Path:  # fmt: skip
    """Decompile ``m`` into ``robots_root/<scene>/scene.xml`` with meshes in ``robots_root/meshes`` and refresh
    ``robots_root/SHA256SUMS.json``; returns the XML path. ``split`` replaces the large collision hulls of the matching
    bodies by error-bounded convex pieces (:mod:`alloy.tasks.split_scene`, report in ``hull_split.json``);
    ``keyframes`` writes ``m``'s keyframes too."""
    out_dir = robots_root / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = decompile(m, robots_root / MESH_DIR, scene, keyframes)
    (out_dir / REPORT_NAME).unlink(missing_ok=True)
    if split is not None:
        write_report(out_dir, split, split_scene_hulls(tree.getroot(), robots_root / MESH_DIR, split))
    xml_path = out_dir / SCENE_XML
    tree.write(xml_path, encoding="unicode", xml_declaration=False)
    write_manifest(robots_root)
    log.info("wrote %s (%d bodies, %d geoms, %d meshes)", xml_path, m.nbody, m.ngeom, m.nmesh)
    return xml_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("robots_root", type=Path)
    parser.add_argument("scene")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rec = load_recorded(args.trace_dir)
    write_scene(rec.model, args.robots_root, args.scene)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
