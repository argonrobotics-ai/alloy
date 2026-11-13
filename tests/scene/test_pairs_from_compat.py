# SPDX-License-Identifier: Apache-2.0
"""``MaterialCard.pairs_from`` (AUDIT S2-010) keeps every scene's friction (physics' two conditions, 2026-11-13):

1. same friction: a multi-object scene of per-object materials (mass, restitution, own friction, set_friction, two
   base materials with sourced cards, a static plywood table) builds and steps bit for bit as with 0.1.2's copied pair
   cards (checked against main e65d95254 itself on a 12-box mix: 78 MuJoCo rows, shape mu and 20-step positions
   equal);
2. round trips: Scene.save then alloy.load, and Scene.to_json then Scene.from_dict, keep pairs_from and give the
   same pair mu; cards saved by 0.1.2 (derived materials without pairs_from, the copied pair files beside them) load
   to the same model.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import alloy
from alloy.cards.library import default_root
from alloy.cards.loader import load_material
from alloy.cards.units_yaml import yaml_dump, yaml_load


def _mix() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.8, 0.8, 0.1), pos=(0.0, 0.0, 0.05), static=True, material="plywood_marine")
    for i in range(10):
        kw = {"restitution": 0.2} if i % 5 == 4 else {"mass": 0.05 + 0.01 * i} if i % 3 else {}
        s.add_box(f"b{i}", size=(0.03, 0.03, 0.03), pos=((i % 5) * 0.06 - 0.12, (i // 5) * 0.06 - 0.03, 0.12),
                  material="abs" if i % 2 else "fr4", **kw)
    s.add_box("own", size=(0.03, 0.03, 0.03), pos=(0.2, 0.2, 0.12), friction=0.9, mass=0.04)
    s.add_sphere("ball", 0.015, pos=(-0.2, 0.2, 0.12), material="abs", mass=0.02)
    s.set_friction("b1", "b2", 0.9)
    return s


def _as_0_1_2(root: Path) -> int:
    """Rewrite cards saved now as 0.1.2 saved them: no pairs_from, and the base's sourced library pair cards copied
    onto every pair of its family (0.1.2's alloy.api._materialize._copy_pairs). Returns the files written."""
    family: dict[str, list[str]] = {}
    for p in sorted((root / "materials").glob("*.yaml")):
        data = yaml_load(p.read_text(encoding="utf-8"), p.stem)
        base = data.pop("pairs_from", None)
        if base is not None:
            family.setdefault(base, [base]).append(data["name"])
            p.write_text(yaml_dump(data), encoding="utf-8")
    written = 0
    for path in sorted((default_root() / "pairs").glob("*.yaml")):
        a, _, b = path.stem.partition("__")
        if a not in family and b not in family:
            continue
        data = yaml_load(path.read_text(encoding="utf-8"), path.stem)
        for x, y in itertools.product(family.get(a, [a]), family.get(b, [b])):
            out = root / "pairs" / f"{'__'.join(sorted((x, y)))}.yaml"
            if (x, y) == (a, b) or out.exists():
                continue
            out.write_text(yaml_dump({**data, "a": x, "b": y}), encoding="utf-8")
            written += 1
    return written


def _fingerprint(sim, steps: int = 20) -> dict[str, np.ndarray]:
    m = sim.engine.model
    out = {"shape_mu": m.shape_material_mu.numpy()}
    for k in ("pair_geom1", "pair_geom2", "pair_friction", "pair_solref", "pair_solimp"):
        out[k] = getattr(m.mujoco, k).numpy()
    sim.step(steps)
    out["pos"] = np.concatenate([v.ravel() for k, v in sorted(sim.state().items()) if k.endswith(".pos")])
    return out


def _pair_mu(sim) -> dict[frozenset, float]:
    """{objects of a MuJoCo row: its mu} and {object: its shape mu}, by name (independent of shape order)."""
    m = sim.engine.model
    label = [m.body_label[b] if b >= 0 else f"static{s}" for s, b in enumerate(m.shape_body.numpy())]
    rows = {frozenset((label[int(g1)], label[int(g2)])): float(f[0]) for g1, g2, f in
            zip(m.mujoco.pair_geom1.numpy(), m.mujoco.pair_geom2.numpy(), m.mujoco.pair_friction.numpy())}
    rows.update({frozenset((n,)): float(mu) for n, mu in zip(label, m.shape_material_mu.numpy())})
    return rows


def test_cards_saved_by_0_1_2_load_and_step_bit_identically(tmp_path):
    """Condition 1 and the 0.1.2 half of condition 2. ~15 s on a CPU."""
    s = _mix()
    s.save(tmp_path / "now")
    s.save(tmp_path / "old")
    assert _as_0_1_2(tmp_path / "old") > 20 and not list((tmp_path / "now" / "pairs").glob("*.yaml"))
    now = _fingerprint(alloy.load(str(tmp_path / "now" / "scenes" / "scene.yaml"), device="cpu"))
    old = _fingerprint(alloy.load(str(tmp_path / "old" / "scenes" / "scene.yaml"), device="cpu"))
    assert len(now["pair_geom1"]) > 20
    for k in now:
        np.testing.assert_array_equal(now[k], old[k], err_msg=k)


def test_save_load_and_json_round_trips_keep_pairs_from_and_the_mu(tmp_path):
    """Condition 2: the built scene, its saved cards loaded back, and its JSON round trip resolve the same mu for
    every row and shape; the saved derived cards name their base. ~20 s on a CPU."""
    s = _mix()
    ref = _pair_mu(alloy.build(s, device="cpu"))
    assert ref[frozenset(("b1", "b3"))] == float(np.float32(0.575))            # abs/abs card, both given a mass
    s.save(tmp_path)
    assert load_material(tmp_path / "materials" / "abs_for_b1.yaml").pairs_from == "abs"
    assert load_material(tmp_path / "materials" / "abs_for_own.yaml").pairs_from is None     # its own friction
    loaded = alloy.Scene.load(str(tmp_path / "scenes" / "scene.yaml"))
    assert _pair_mu(alloy.build(loaded, device="cpu")) == ref
    assert _pair_mu(alloy.build(alloy.Scene.from_dict(json.loads(s.to_json())), device="cpu")) == ref
    assert _pair_mu(alloy.build(alloy.Scene.from_dict(json.loads(loaded.to_json())), device="cpu")) == ref
