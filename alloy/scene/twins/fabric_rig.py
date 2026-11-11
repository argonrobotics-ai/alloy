# SPDX-License-Identifier: Apache-2.0
"""The measured-fabric Cusick drape rig (SCR-WP2-14): published KES-F fabrics on Alloy's production route.

D03 (deform_cloth.py) drapes one synthetic cotton-like card on raw solvers. D03M drapes real fabrics: each pack entry
(``bench/scenarios/packs/fabric_drape_*.json``, schema ``alloy.bench.fabric_drape/1``) carries a paper's KES-F inputs
and the drape coefficient it measured on the same fabric. The card is built from those inputs alone, the drape runs
through ``alloy.Scene`` + ``alloy.build`` (the route a user's card gets), and the converged DC is compared with the
measured one. Users: the D03M bench scenario (``python -m bench.scenarios.fabric_drape_run``, rows in
bench/results/deform/D03M_*.jsonl) and the R6 law's production-route twin (``alloy.scene.twins.drape_route``).

Card mapping, pre-registered ([internal note] scr/WP2.md SCR-WP2-14; no number is fitted to the measured drape):
- density = W / h, h the sheet's contact thickness max(T0, H_MIN), so the areal mass is W exactly;
- elastic.bending = the warp / weft mean of KES B (KES's convention); ``bend="geo"``: sqrt(B_warp B_weft);
- arm L, the HEADLINE (ruling 2026-11-08 10:56Z, stated before the first run): the shipped KES cards' convention
  (library cloth_cotton_plain, cloth_polyester_woven), youngs back-calculated from B at the KES thickness,
  E = 12 (1 - nu^2) B / T0^3, nu 0.3, no elastic.cloth, so D16 routes it to CLOTH_VBD. Its super row has the bar
  |DC - DC_measured| <= 0.05;
- diagnostics, no bar: arm T, E h = the KES tensile secant (load / strain, geometric mean of warp and weft); arm S,
  E h = 2 (1 + nu) G, so the VBD membrane's shear modulus is KES G; arm A, elastic.cloth (warp / weft stretch, G,
  warp / weft B), so D16 routes it to CLOTH_STYLE3D, alone in its scene; arm O, the same card with a route override to
  CLOTH_VBD, which gives it the orthotropic membrane (``alloy.scene.ortho_membrane``, gap #3).

Rig: the specimen disc (the R6 twin's jittered-hex mesh, ``alloy.scene.twins.drape.disc_mesh``, rescaled to the
pack's diameter) clamped between two static discs of the support diameter, as in the drapemeter (the upper disc's
bottom at the sheet thickness). Every fabric x arm gets its own rig, RIG_SPACING apart, so one build steps them all.
Each world starts each specimen off-centre by up to JITTER, so the fold pattern is not imposed. Gravity eases in over
G_RAMP_S (smoothstep); the rig is scored every FRAME_S over the last SCORE_S (``fabric_score.Scorer``).
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np

NU = 0.3
H_MIN = 2.0e-3             # m: VBD sheets thinner than 2 mm need a smaller dt (AGENTS.md section 7)
ARMS = ("L", "T", "S", "A", "O")
HEADLINE = "L"
BAR_DC = 0.05              # |DC - DC_measured| on the headline arm's super row (ruling 10:56Z)
JITTER = 2.0e-3            # m: per-world start offset (uniform in a square)
G_RAMP_S = 0.5
SCORE_S = 1.0              # s: scored window at the end of the run
FRAME_S = 0.1
GROUND_Z = -0.5
RIG_SPACING = 0.5          # m between rig axes
R6_SPECIMEN = 0.15         # m: the R6 mesh's radius (rescaled to the pack's)


# --- card mapping ----------------------------------------------------------------------------------------------------

def thickness(fab: dict) -> float:
    """The sheet's contact thickness [m]: the measured T0, at least H_MIN."""
    return max(float(fab["thickness_T0_m"]), H_MIN)


def bending(fab: dict, bend: str = "mean") -> float:
    """Isotropic bending rigidity [N m] from KES B warp / weft."""
    bw, bf = float(fab["bending_warp_N_m"]), float(fab["bending_weft_N_m"])
    return 0.5 * (bw + bf) if bend == "mean" else math.sqrt(bw * bf)


def tensile(fab: dict, direction: str) -> float:
    """KES-FB1 secant stretch stiffness [N/m]: the test load over the strain it reached."""
    return float(fab["tensile_load_N_m"]) / float(fab[f"tensile_strain_{direction}"])


def youngs_L(fab: dict, bend: str = "mean") -> float:
    """Arm L's E [Pa]: the plate formula D = E T0^3 / (12 (1 - nu^2)) solved for E at the KES thickness."""
    return 12.0 * (1.0 - NU * NU) * bending(fab, bend) / float(fab["thickness_T0_m"]) ** 3


def membrane_Eh(fab: dict, arm: str, bend: str = "mean") -> float:
    """E h [N/m] at the sheet's thickness for the isotropic arms (A reports T's)."""
    if arm == "L":
        return youngs_L(fab, bend) * thickness(fab)
    if arm in ("T", "A", "O"):
        return math.sqrt(tensile(fab, "warp") * tensile(fab, "weft"))
    if arm == "S":
        return 2.0 * (1.0 + NU) * float(fab["shear_G_N_m"])
    raise ValueError(f"unknown arm {arm!r} (arms {ARMS})")


def _q(value: float, note: str, source: str = "measured") -> dict:
    return {"value": float(value), "source": source, "note": note}


def _aniso(fab: dict, src: str) -> dict:
    return {"stretch_warp": _q(tensile(fab, "warp"), f"{src}; KES EM warp secant"),
            "stretch_weft": _q(tensile(fab, "weft"), f"{src}; KES EM weft secant"),
            "shear": _q(float(fab["shear_G_N_m"]), f"{src}; KES G"),
            "bend_warp": _q(float(fab["bending_warp_N_m"]), f"{src}; KES B warp"),
            "bend_weft": _q(float(fab["bending_weft_N_m"]), f"{src}; KES B weft")}


def card_name(fab: dict, arm: str, bend: str = "mean") -> str:
    return f"fabric_{fab['id']}_{arm}" + ("" if bend == "mean" else f"_{bend}")


def material_card(fab: dict, arm: str, bend: str = "mean", source: str = "") -> dict[str, Any]:
    """The arm's alloy.material/1 card (a dict for yaml.safe_dump)."""
    h, src = thickness(fab), f"{fab['id']} ({source or 'pack'})"
    eh = membrane_Eh(fab, arm, bend)
    elastic: dict[str, Any] = {
        "youngs": _q(eh / h, f"{src}; arm {arm}: E h {eh:.4g} N/m at the sheet thickness {h * 1e3:g} mm", "model"),
        "poisson": _q(NU, "not measured by KES-F; 0.3 (SCR-WP2-14, pre-registered)", "model"),
        "bending": _q(bending(fab, bend), f"{src}; KES B, {bend} of warp and weft"),
    }
    if arm in ("A", "O"):
        elastic["cloth"] = _aniso(fab, src)
    surface = _q(0.5, "clamped specimen: friction does not enter the drape", "model")
    return {"schema": "alloy.material/1", "name": card_name(fab, arm, bend),
            "density": _q(float(fab["areal_mass_kg_m2"]) / h, f"{src}; W over the sheet thickness {h * 1e3:g} mm"),
            "surface": {"mu_static": surface, "mu_dynamic": surface,
                        "restitution": _q(0.0, "fabric impacts fully damped", "model")},
            "elastic": elastic}


# --- rig -------------------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Rig:
    fab: dict
    arm: str
    bend: str
    xy: tuple[float, float]

    @property
    def name(self) -> str:
        return card_name(self.fab, self.arm, self.bend)


def specimen_mesh(radius: float, edge: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The R6 twin's jittered-hex disc mesh rescaled to ``radius`` (edge kept): vertices [V, 3] at z = 0, triangles
    [T, 3]. (D03M's square-grid disc lives in bench/scenarios/fabric_drape.py.)"""
    from alloy.scene.twins import drape as R6

    s = radius / R6_SPECIMEN
    v2, tris = R6.disc_mesh(edge / s, seed)
    return np.column_stack([v2 * s, np.zeros(len(v2))]), np.asarray(tris, np.int64)


def min_edge(mesh: tuple[np.ndarray, np.ndarray]) -> float:
    v, t = mesh
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).min())


def check_mesh(mesh: tuple[np.ndarray, np.ndarray], rigs: list["Rig"]) -> float:
    """The mesh's shortest edge [m]; ValueError when it is shorter than a rig's sheet thickness. With self-contact
    on, such a mesh starts in self-contact (the R6 mesher's ring edges reach 1.1 mm at a 5 mm cell): it blows up on
    the route (2026-11-08: strain to 49,000 %, sim.check() ok), so a run on it measures nothing."""
    shortest = min_edge(mesh)
    h = max(thickness(r.fab) for r in rigs)
    if shortest < h * (1.0 - 1e-6):                    # an edge at the distance is fine (WP11's warning agrees)
        raise ValueError(f"specimen mesh: shortest edge {shortest * 1e3:.2f} mm < the sheet thickness {h * 1e3:g} mm "
                         "(self-contact would start in contact); use another seed, a coarser cell or mesh='grid'")
    return shortest


def rigs_for(pack: dict, rigs: str, fabrics: str = "") -> list[Rig]:
    """Rigs for the fabrics named in ``fabrics`` (ids or their last part; all when empty) times the ``rigs`` tokens
    ``arm[:bend]``, laid out on a square grid."""
    want = [f for f in fabrics.split(",") if f]
    fabs = [f for f in pack["fabrics"] if not want or f["id"].split("_")[-1] in want or f["id"] in want]
    tokens = [t.split(":") for t in rigs.split(",") if t]
    combos = [(f, t[0], t[1] if len(t) > 1 else "mean") for f in fabs for t in tokens]
    if any(arm == "A" for _, arm, _ in combos) and len(combos) > 1:
        raise ValueError("arm A (Style3D) runs alone: D16 routes anisotropic cloth to Style3D only without other cloth")
    n = int(np.ceil(np.sqrt(len(combos))))
    return [Rig(f, arm, bend, (RIG_SPACING * (i % n), RIG_SPACING * (i // n)))
            for i, (f, arm, bend) in enumerate(combos)]


def write_cards(out: Path, rigs: list[Rig], source: str) -> Path:
    import yaml

    root = out / "cards"
    (root / "materials").mkdir(parents=True, exist_ok=True)
    for r in rigs:
        card = material_card(r.fab, r.arm, r.bend, source)
        (root / "materials" / f"{card['name']}.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
    return root


def add_rig(scene, rig: Rig, mesh: tuple[np.ndarray, np.ndarray], r_sup: float, starts: np.ndarray,
            clamp: bool = True) -> None:
    """Support, clamp and specimen of one rig. The specimen is an add_cloth item whose shape is swapped for the disc
    mesh (alloy.Scene has no disc sheet), as XB4's drape harness does. ``clamp=False`` leaves the specimen resting on
    the support (the card's friction), as XB4's L5 drape does: a diagnostic, not the drapemeter (Style3D's contact
    fails the clamp at its first step, 2026-11-08)."""
    v, tris = mesh
    h, (x, y) = thickness(rig.fab), rig.xy
    scene.add_cylinder(f"{rig.name}_support", r_sup, 0.1, pos=(x, y, -0.05), static=True)
    if clamp:
        scene.add_cylinder(f"{rig.name}_clamp", r_sup, 0.01, pos=(x, y, h + 0.005), static=True)
    pos = np.column_stack([starts + np.array([x, y]), np.full(len(starts), 0.5 * h)])
    d = 2.0 * float(np.abs(v[:, :2]).max())
    scene.add_cloth(rig.name, size=(d, d), pos=pos, thickness=h, material=rig.name, self_contact=True)
    item = scene.items[rig.name]
    item["shape"] = {"mesh": {"path": "", "sha256": "", "scale": [1.0, 1.0, 1.0], "collision": "convex"}}
    item["mesh"] = {"vertices": v.tolist(), "faces": tris.tolist()}
    item["volume"] = _area(v, tris) * h
    item["card_extra"].setdefault("budget", {})["particles"] = len(v)
    if rig.arm == "O":
        item["card_extra"]["route"] = {"route": "cloth_vbd", "reason": "arm O: the orthotropic VBD membrane"}


def make_scene(rigs: list[Rig], worlds: int, accuracy: str, root: Path, mesh, r_sup: float, seed: int = 0,
               clamp: bool = True):
    import alloy

    scene = alloy.Scene(worlds=worlds, dt=0.01, roots=[str(root)], cloth_accuracy=accuracy)
    scene.add_ground(z=GROUND_Z)
    starts = np.random.default_rng(seed).uniform(-JITTER, JITTER, (worlds, 2))
    for rig in rigs:
        add_rig(scene, rig, mesh, r_sup, starts, clamp)
    return scene


def _area(v: np.ndarray, t: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1).sum())


__all__ = ["ARMS", "BAR_DC", "FRAME_S", "G_RAMP_S", "HEADLINE", "H_MIN", "JITTER", "NU", "RIG_SPACING", "Rig",
           "SCORE_S", "add_rig", "bending", "card_name", "check_mesh", "make_scene", "material_card", "membrane_Eh",
           "min_edge", "rigs_for", "specimen_mesh", "tensile", "thickness", "write_cards", "youngs_L"]
