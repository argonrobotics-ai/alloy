# SPDX-License-Identifier: Apache-2.0
"""Command line for the task-scene pipeline.

``python -m alloy.tasks.cli convert <recorded dir> <scene> <trace name> [--seconds S] [--attribute S]``
    offline (CPU MuJoCo): mjb -> ``library/robots/<scene>/scene.xml`` (+ shared meshes, manifest), recorded window
    -> ``bench/traces/<trace name>/``, with the source files' sha256, the CPU replay check of the converted model and
    the cause attribution over the first ``--attribute`` seconds written into the trace metadata.
``python -m alloy.tasks.cli replay <trace name> [--seconds S] [--multiccd 0|1] [--nconmax N] [--njmax N]
[--nativeccd 0|1] [--noslip N]``
    Newton ``SolverMuJoCo`` replay of a saved trace on its vendored scene; prints the object-pose deviation as JSON.
    ``--nconmax`` / ``--njmax`` size MuJoCo-Warp's contact and constraint buffers (the 1000-body piles need more
    than the defaults; an overflow drops contacts and is reported as ``overflow`` != 0). Box-box runs through
    MuJoCo-Warp's primitive collider, as in MuJoCo C (``ReplayConfig.nativeccd``); ``--nativeccd 1`` keeps the scene's
    flag (Warp GJK/EPA box-box). ``--noslip`` overrides the recorded noslip iterations.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from alloy.core.ids import sha256_hex
from alloy.tasks.hull_split import SplitConfig
from alloy.tasks.msh import sha256_file
from alloy.tasks.split_scene import REPORT_NAME, SceneSplit
from alloy.cards.library import default_root

log = logging.getLogger("alloy.tasks")

REPO = Path(__file__).resolve().parents[2]
ROBOTS = default_root() / "robots"
TRACES = REPO / "bench" / "traces"
SOURCE_FILES = ("model.mjb", "inputs.npz", "state0.npy", "ref_qpos.npy", "trace.json")


def scene_xml(scene: str) -> Path:
    return ROBOTS / scene / "scene.xml"


def _field_diffs(rec, xml: Path, split: bool) -> list:  # noqa: ANN001 - RecordedTrace (lazy import)
    """Converter check: field diffs of the converted model against the mjb. A split scene changes the geoms on
    purpose, so the check runs on the same conversion without the split, in a scratch directory."""
    import tempfile

    from alloy.tasks.mjb_convert import write_scene
    from alloy.tasks.mjb_verify import compare_models

    with tempfile.TemporaryDirectory() as tmp:
        plain = write_scene(rec.model, Path(tmp), xml.parent.name) if split else xml
        converted = mujoco.MjModel.from_xml_path(str(plain))
        return [d for d in compare_models(rec.model, converted) if d.max_abs > 1e-12 * max(1.0, d.ref_scale)]


def _offline_checks(rec, xml: Path, seconds: float, split: bool) -> dict:  # noqa: ANN001 - RecordedTrace
    from alloy.tasks.attribution import attribute

    diffs = _field_diffs(rec, xml, split)
    converted = mujoco.MjModel.from_xml_path(str(xml))
    steps = int(round(seconds / rec.dt))
    variants = attribute(converted, rec, steps)
    return {"field_diffs": [asdict(d) for d in diffs], "attribution_seconds": seconds,
            "attribution": {v.variant: {"max_point_err_m": v.max_point_err, "first_over_1mm_s": v.first_over_1mm}
                            for v in variants}}  # fmt: skip


def _with_reference(rec, seconds: float | None):  # noqa: ANN001, ANN202 - RecordedTrace (lazy import)
    """Recordings without ``ref_qpos`` (the r5 pile drops) get one from an offline CPU MuJoCo replay of the
    recorded mjb itself, which is how the recorder produced ``ref_qpos`` for the robot traces."""
    import copy
    import dataclasses

    from alloy.tasks.mjb_verify import replay_cpu

    if rec.ref_qpos.shape[0]:
        return rec, "recorded ref_qpos"
    steps = rec.steps if seconds is None else min(rec.steps, int(round(seconds / rec.dt)))
    ref = replay_cpu(copy.copy(rec.model), rec, steps - steps % rec.ref_every)
    note = f"offline CPU MuJoCo {mujoco.__version__} replay of the recorded model.mjb from state0 (no qpos recorded)"
    return dataclasses.replace(rec, ref_qpos=ref), note


def convert(src: Path, scene: str, name: str, seconds: float | None, attribute_s: float,
            split: SceneSplit | None = None) -> Path:  # fmt: skip
    from alloy.tasks.mjb_convert import load_recorded, write_scene
    from alloy.tasks.trace import TaskTrace, build_trace, save_trace

    rec, ref_note = _with_reference(load_recorded(src), seconds)
    xml = write_scene(rec.model, ROBOTS, scene, split)
    tt = build_trace(rec, sha256_hex(xml.read_bytes()), seconds)
    meta = tt.meta | {"scene": f"library/robots/{scene}/scene.xml", "reference": ref_note,
                      "source_sha256": {f: sha256_file(src / f) for f in SOURCE_FILES if (src / f).is_file()},
                      "offline": _offline_checks(rec, xml, attribute_s, split is not None)}  # fmt: skip
    if split is not None:
        meta["hull_split"] = json.loads((xml.parent / REPORT_NAME).read_text())
    out = TRACES / name
    save_trace(out, TaskTrace(tt.trace, tt.ref, meta))
    return out


def replay(name: str, overrides: dict) -> dict:
    """``overrides``: ``ReplayConfig`` fields (seconds, multiccd, nconmax, njmax) given on the command line."""
    from alloy.tasks.replay import ReplayConfig, replay_scene
    from alloy.tasks.trace import load_trace

    tt = load_trace(TRACES / name)
    r = replay_scene(REPO / tt.meta["scene"], tt, ReplayConfig(**overrides))
    return {"trace": name, "seconds": r.steps * tt.trace.control_dt, "steps": r.steps, "wall_s": r.wall_s,
            "max_point_err_m": r.max_point_err, "max_pos_err_m": float(r.pos_err.max(initial=0.0)),
            "max_ang_err_rad": float(r.ang_err.max(initial=0.0)), "first_over_1mm_s": r.first_exceed(1e-3),
            "first_over_0p1mm_s": r.first_exceed(1e-4), "overflow": r.overflow, "settings": r.settings,
            "per_object_max_point_err_m": dict(zip(r.objects, np.round(r.point_err.max(axis=0), 9).tolist()))}  # fmt: skip


def _replay_overrides(args: argparse.Namespace) -> dict:
    flags = {"multiccd": None if args.multiccd is None else bool(args.multiccd)}
    if args.nativeccd is not None:
        flags["nativeccd"] = None if args.nativeccd else False  # 1: as the scene says
    out = {"seconds": args.seconds} | flags
    return out | {k: v for k in ("nconmax", "njmax", "noslip", "device") if (v := getattr(args, k)) is not None}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("src", type=Path)
    c.add_argument("scene")
    c.add_argument("name")
    c.add_argument("--seconds", type=float, default=None)
    c.add_argument("--attribute", type=float, default=2.0)
    c.add_argument("--split-bodies", default=None, help="regex: split these bodies' large collision hulls")
    c.add_argument("--split-tol-mm", type=float, default=0.2)
    c.add_argument("--split-max-verts", type=int, default=64)
    r = sub.add_parser("replay")
    r.add_argument("name")
    r.add_argument("--seconds", type=float, default=2.0)
    r.add_argument("--multiccd", type=int, choices=(0, 1), default=None)
    r.add_argument("--nconmax", type=int, default=None)
    r.add_argument("--njmax", type=int, default=None)
    r.add_argument("--nativeccd", type=int, choices=(0, 1), default=None, help="1: scene flag (Warp GJK/EPA box-box)")
    r.add_argument("--noslip", type=int, default=None)
    r.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import warp as wp

    wp.config.quiet = True
    if args.cmd == "convert":
        split = None if args.split_bodies is None else SceneSplit(
            args.split_bodies, SplitConfig(max_verts=args.split_max_verts, tol_m=args.split_tol_mm * 1e-3))
        print(convert(args.src, args.scene, args.name, args.seconds, args.attribute, split))
    else:
        print(json.dumps(replay(args.name, _replay_overrides(args)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
