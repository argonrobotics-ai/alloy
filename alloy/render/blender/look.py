# SPDX-License-Identifier: Apache-2.0
"""Look file (docs/RENDER.md section 3): parsing, validation, frame lists and material assignment. No bpy."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import presets

RENDER_DEFAULTS: dict = {
    "resolution": [1080, 1920], "samples": 512, "noise_threshold": 0.01, "denoiser": "OIDN", "view": "AgX",
    "look": "AgX - Medium High Contrast", "exposure": 0.0, "motion_blur": {"shutter_s": 1.0 / 60.0},
    "bounces": {"total": 12, "transmission": 16, "volume": 2}, "caustics": {"reflective": False, "refractive": True},
    "film_transparent": False, "white_balance": "auto",
}
WORLD_DEFAULTS: dict = {"hdri": None, "strength": 1.0, "rotation_deg": 0.0, "show_hdri": False,
                        "background_color": [0.02, 0.02, 0.02], "background_from_hdri": False}
LIGHT_TYPES = ("area", "spot", "point", "sun")


class LookError(Exception):
    """The look file is invalid; the message names the field and the bad value."""


def _vec(value, n: int, where: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise LookError(f"{where} is {value!r}, expected {n} numbers")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise LookError(f"{where} is {value!r}, expected {n} numbers") from exc


def _regex(pattern: str, where: str) -> re.Pattern:
    try:
        return re.compile(pattern)
    except (re.error, TypeError) as exc:
        raise LookError(f"{where}: bad regex {pattern!r} ({exc})") from exc


def _check_fade(world: dict) -> None:
    fade = world.get("fade")
    if fade is None:
        return
    if not isinstance(fade, dict) or set(fade) - {"material", "start_m", "depth_m"} or "material" not in fade:
        raise LookError(f"world.fade is {fade!r}, expected {{material, start_m, depth_m}}")
    if any(float(fade.get(k, 1.0)) <= 0 for k in ("start_m", "depth_m")):
        raise LookError(f"world.fade {fade!r}: start_m and depth_m must be > 0")
    if world.get("show_hdri"):
        raise LookError("world.fade blends into the flat background, so it needs show_hdri false")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


@dataclass
class Look:
    """A validated look file."""

    raw: dict
    render: dict
    frames: dict
    cameras: dict
    world: dict
    lights: list
    materials: dict
    assign: list = field(default_factory=list)
    hide: list = field(default_factory=list)

    def section(self, name: str) -> dict:
        return self.raw.get(name) or {}

    def is_hidden(self, labels: list[str]) -> bool:
        return any(rx.search(lb) for rx in self.hide for lb in labels if lb)

    def assigned(self, labels: list[str]) -> str | None:
        """First assign rule whose regex matches any label (shape, then body, then entity name)."""
        for rx, name in self.assign:
            if any(lb and rx.search(lb) for lb in labels):
                return name
        return None

    def material_spec(self, name: str) -> dict:
        """Preset values merged with the look's overrides for material (or preset) `name`."""
        entry = self.materials.get(name)
        if entry is None:
            if name not in presets.PRESETS:
                raise LookError(f"material {name!r} is neither in look.materials nor a preset")
            entry = {"preset": name}
        preset = entry.get("preset", presets.FALLBACK_MATERIAL)
        if preset not in presets.PRESETS:
            raise LookError(f"materials.{name}.preset {preset!r} unknown; presets: {sorted(presets.PRESETS)}")
        spec = {**presets.DEFAULTS, **presets.PRESETS[preset]}
        overrides = {k: v for k, v in entry.items() if k != "preset"}
        spec.update(overrides)
        spec["_name"] = name
        spec["_overrides"] = sorted(overrides)
        return spec


def _check_camera(name: str, cam: dict) -> None:
    where = f"cameras.{name}"
    keys = cam.get("keys")
    if not isinstance(keys, list) or not keys:
        raise LookError(f"{where}.keys must be a non-empty list, got {keys!r}")
    times = []
    for i, k in enumerate(keys):
        times.append(float(k.get("t", 0.0)))
        _vec(k.get("eye"), 3, f"{where}.keys[{i}].eye")
        _vec(k.get("target"), 3, f"{where}.keys[{i}].target")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise LookError(f"{where}.keys times {times} must increase strictly")
    if float(cam.get("lens_mm", 50)) <= 0 or float(cam.get("sensor_mm", 36)) <= 0:
        raise LookError(f"{where}: lens_mm and sensor_mm must be > 0")
    if float(cam.get("fstop", 0)) < 0:
        raise LookError(f"{where}.fstop is {cam.get('fstop')!r}, must be >= 0")
    focus = cam.get("focus")
    if focus is not None and not ({"label", "point", "distance"} & set(focus)):
        raise LookError(f"{where}.focus is {focus!r}, expected one of label / point / distance")
    if focus and "point" in focus:
        _vec(focus["point"], 3, f"{where}.focus.point")
    if "resolution" in cam:
        _vec(cam["resolution"], 2, f"{where}.resolution")


def _check_lights(lights: list) -> None:
    if not isinstance(lights, list):
        raise LookError(f"lights is {type(lights).__name__}, expected a list")
    for i, lt in enumerate(lights):
        if lt.get("type", "area") not in LIGHT_TYPES:
            raise LookError(f"lights[{i}].type is {lt.get('type')!r}, expected one of {LIGHT_TYPES}")
        _vec(lt.get("eye"), 3, f"lights[{i}].eye")
        _vec(lt.get("target", [0, 0, 0]), 3, f"lights[{i}].target")
        if float(lt.get("kelvin", 5500)) < 1000:
            raise LookError(f"lights[{i}].kelvin is {lt.get('kelvin')!r}, must be >= 1000")


def _check_materials(look: Look) -> None:
    allowed = set(presets.OVERRIDE_KEYS) | set(presets.EXTRA_KEYS)
    for name, entry in look.materials.items():
        if not isinstance(entry, dict):
            raise LookError(f"materials.{name} is {entry!r}, expected an object")
        bad = sorted(set(entry) - allowed)
        if bad:
            raise LookError(f"materials.{name}: unknown keys {bad}; allowed: {sorted(allowed)}")
        look.material_spec(name)
    for kind in ("liquids", "points", "instances", "curves"):
        for ent, cfg in look.section(kind).items():
            if cfg.get("material"):
                look.material_spec(cfg["material"])
            inst = cfg.get("instance")
            if inst is not None and inst not in presets.GRAINS:
                raise LookError(f"{kind}.{ent}.instance {inst!r} unknown; grains: {sorted(presets.GRAINS)}")
    for ent, cfg in look.section("surfaces").items():
        for m in cfg.get("materials", []):
            look.material_spec(m)


def _check_bevels_skins(look: Look) -> None:
    """look.bevels: [{match, bevel_m, segments}]; look.skins: {name: {mesh, weights, material, ...}}."""
    bevels = look.raw.get("bevels") or []
    if not isinstance(bevels, list):
        raise LookError(f"bevels is {type(bevels).__name__}, expected a list")
    for i, b in enumerate(bevels):
        _regex(b.get("match"), f"bevels[{i}].match")
        if float(b.get("bevel_m", 0.0003)) <= 0 or int(b.get("segments", 3)) < 1:
            raise LookError(f"bevels[{i}]: bevel_m must be > 0 and segments >= 1, got {b!r}")
    for name, cfg in look.section("skins").items():
        if cfg.get("local") and not cfg.get("body"):
            raise LookError(f"skins.{name}.local needs 'body' (the frame the mesh is given in)")
        keys = ("mesh",) if cfg.get("body") else ("mesh", "weights")
        for key in keys:
            if not isinstance(cfg.get(key), str):
                raise LookError(f"skins.{name}.{key} is {cfg.get(key)!r}, expected a path under the bundle root")
        for key in ("materials", "material_map"):
            for src, mat in (cfg.get(key) or {}).items():
                look.material_spec(mat)
        if cfg.get("blend", "dqs") not in ("dqs", "lbs", "dual_quat", "linear"):
            raise LookError(f"skins.{name}.blend is {cfg.get('blend')!r}, expected dqs or lbs")
        if cfg.get("material"):
            look.material_spec(cfg["material"])


def _check_contact(look: Look) -> None:
    """liquids.<name>.contact: [{match, distance_m (> 0, optional), overlap_m (>= 0, optional)}]."""
    for name, cfg in look.section("liquids").items():
        rules = cfg.get("contact") or []
        if not isinstance(rules, list):
            raise LookError(f"liquids.{name}.contact is {type(rules).__name__}, expected a list of rules")
        for i, rule in enumerate(rules):
            where = f"liquids.{name}.contact[{i}]"
            if not isinstance(rule, dict):
                raise LookError(f"{where} is {rule!r}, expected {{'match': regex, ...}}")
            _regex(rule.get("match"), f"{where}.match")
            if "distance_m" in rule and not float(rule["distance_m"]) > 0:
                raise LookError(f"{where}.distance_m is {rule['distance_m']!r}, must be > 0")
            if float(rule.get("overlap_m", 0.0)) < 0:
                raise LookError(f"{where}.overlap_m is {rule['overlap_m']!r}, must be >= 0")


def _check_props_wb(look: Look) -> None:
    """look.props entries and render.white_balance."""
    props = look.raw.get("props") or []
    if not isinstance(props, list):
        raise LookError(f"props is {type(props).__name__}, expected a list")
    for i, p in enumerate(props):
        model = p.get("model")
        if not isinstance(model, str) or not model.startswith(("polyhaven:", "file:")):
            raise LookError(f"props[{i}].model is {model!r}, expected 'polyhaven:<id>' or 'file:<path>'")
        _vec(p.get("pose", [0, 0, 0, 0, 0, 0, 1]), 7, f"props[{i}].pose")
        scale = p.get("scale", 1.0)
        if not isinstance(scale, (int, float)):
            _vec(scale, 3, f"props[{i}].scale")
        if p.get("material"):
            look.material_spec(p["material"])
    ground = look.raw.get("ground")
    if ground is not None:
        if not isinstance(ground, dict) or "z" not in ground or "material" not in ground:
            raise LookError(f"ground is {ground!r}, expected {{z, material, size_m?, center?}}")
        look.material_spec(ground["material"])
        if ground.get("center") is not None:
            _vec(ground["center"], 2, "ground.center")
    wb = look.render.get("white_balance", "auto")
    if isinstance(wb, dict):
        if float(wb.get("kelvin", 6500)) < 1000:
            raise LookError(f"render.white_balance.kelvin is {wb.get('kelvin')!r}, must be >= 1000")
    elif wb not in ("auto", "off"):
        raise LookError(f"render.white_balance is {wb!r}, expected 'auto', 'off' or {{'kelvin': K, 'tint': t}}")


def parse_look(raw: dict) -> Look:
    """Validate a look dict and fill defaults; raises LookError naming the bad field."""
    if raw.get("look") != 1:
        raise LookError(f"look is {raw.get('look')!r}, this renderer reads look 1")
    render = _merge(RENDER_DEFAULTS, raw.get("render", {}))
    _vec(render["resolution"], 2, "render.resolution")
    if int(render["samples"]) < 1:
        raise LookError(f"render.samples is {render['samples']!r}, must be >= 1")
    cameras = raw.get("cameras") or {}
    if not cameras:
        raise LookError("cameras: at least one camera is required")
    for name, cam in cameras.items():
        _check_camera(name, cam)
    world = _merge(WORLD_DEFAULTS, raw.get("world", {}))
    _vec(world["background_color"], 3, "world.background_color")
    if not isinstance(world["background_from_hdri"], bool):
        raise LookError(f"world.background_from_hdri is {world['background_from_hdri']!r}, expected true or false")
    _check_fade(world)
    lights = raw.get("lights", [])
    _check_lights(lights)
    assign = []
    for i, rule in enumerate(raw.get("assign", [])):
        if "match" not in rule or "material" not in rule:
            raise LookError(f"assign[{i}] is {rule!r}, expected match and material")
        assign.append((_regex(rule["match"], f"assign[{i}].match"), rule["material"]))
    hide = [_regex(p, f"hide[{i}]") for i, p in enumerate(raw.get("hide", []))]
    look = Look(raw=raw, render=render, frames=raw.get("frames", {}), cameras=cameras, world=world, lights=lights,
                materials=raw.get("materials", {}), assign=assign, hide=hide)
    for _, name in assign:
        look.material_spec(name)
    _check_materials(look)
    _check_bevels_skins(look)
    _check_contact(look)
    _check_props_wb(look)
    return look


def load_look(path: str | Path) -> Look:
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LookError(f"cannot read look file {path}: {exc}") from exc
    return parse_look(raw)


def frame_list(frames: dict, count: int, window: tuple[int, int] | None = None) -> list[int]:
    """Bundle frame indices to render.

    look.frames start/end/step: start inclusive, end inclusive, negative values count from the end (-1 = last).
    window = (A, B) from --frames A:B, A inclusive and B exclusive, intersects the list.
    """
    start, end, step = int(frames.get("start", 0)), int(frames.get("end", -1)), int(frames.get("step", 1))
    if step < 1:
        raise LookError(f"frames.step is {step}, must be >= 1")
    start = start + count if start < 0 else start
    end = end + count if end < 0 else end
    out = list(range(max(start, 0), min(end, count - 1) + 1, step))
    if window is not None:
        out = [f for f in out if window[0] <= f < window[1]]
    return out
