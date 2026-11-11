# SPDX-License-Identifier: Apache-2.0
"""Claimed end-to-end settings of the render gates, one per scene and size: the fastest setting that passes every
quality gate (no speed claim without a passing quality gate).

``bench/render_gates/matrix.json`` is the source when it names a claim for the scene and size
(``scenes.<scene>.claims.<res>``); the table here is the fallback and agrees with it. A claim names the samples per
pixel, the denoiser ("learned" or "oidn") and the learned denoiser's test-time augmentation.

Example::

    from alloy.camera.gate_claims import claim_for
    claim_for("c3", "128x128")     # Claim(spp=5, kind='learned', tta=2)
    claim_for("t12", "128x128")    # Claim(spp=40, kind='learned', tta=1, weights='t12_128')
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Claim:
    """One claimed setting: ``spp`` samples per pixel, denoised by ``kind`` ("learned" or "oidn") at ``tta``, with the
    learned denoiser's ``weights`` (a manifest name; None = the default model for the size)."""

    spp: int
    kind: str = "learned"
    tta: int = 1
    weights: str | None = None

    def label(self) -> str:
        """Short text for rows and docs, e.g. "5 spp + learned (tta 2)" or "40 spp + learned t12_128 (tta 1)"."""
        if self.kind != "learned":
            return f"{self.spp} spp + {self.kind}"
        return f"{self.spp} spp + learned" + (f" {self.weights}" if self.weights else "") + f" (tta {self.tta})"


# Per size, for scenes without their own claim (C3-class scenes: the learned denoiser passes at low spp)
BY_SIZE = {"128x128": Claim(5, "learned", 2), "640x480": Claim(12, "learned", 1), "1280x720": Claim(12, "learned", 1)}
# Per scene and size, where the size's claim fails a gate (G1 on the RTX PRO 6000; OIDN passes from 64 spp). T12 at
# 128 px: T12-tuned weights pass G1 from 40 spp on every seed tried (1,273 img/s vs 692 at 64 spp + OIDN). C3 at
# 1280x720: C3-tuned weights pass at 16 spp (G1 0.79; 4x the speed of 64 spp + OIDN on the same GPU and backend).
# C2 (water and glass) and C4 (frosted tub, clear lid): the fastest setting with G1 5/5 on the RTX PRO 6000 (OptiX);
# lower spp failed (C2 16 spp 3/5 at every size; C4 128 px 4 spp 4/5). C4 640x480 at 16 spp passed at worst ratio
# 1.00 exactly, so the claim keeps 32 (0.84). At 1280x720 the per-scene c4_1280 weights pass G1 from 16 spp at worst
# 0.96 (thin), so the claim takes 32 (0.71, parity and equal-spp PASS; the 640 weights failed even at 64 spp).
BY_SCENE = {
    ("c3", "1280x720"): Claim(16, "learned", 1, "c3_1280"),
    ("t12", "128x128"): Claim(40, "learned", 1, "t12_128"),
    ("t12", "640x480"): Claim(64, "oidn"),
    ("t12", "1280x720"): Claim(64, "oidn"),
    ("c2", "128x128"): Claim(32, "learned", 2),
    ("c2", "640x480"): Claim(32, "learned", 1, "c2_640"),
    ("c2", "1280x720"): Claim(32, "learned", 1, "c2_1280"),
    ("c4", "128x128"): Claim(8, "learned", 2, "c4_128"),
    ("c4", "640x480"): Claim(32, "learned", 1, "c4_640"),
    ("c4", "1280x720"): Claim(32, "learned", 1, "c4_1280"),
    # Deformables (SCR-OSS4-DEFORM, L4, OptiX): the towel fails G1 at 64 spp + OIDN (1.26 / 1.03) and at the size
    # claims (12 spp + learned: 4.17 at 640x480), and passes at 128 spp + OIDN (0.71 / 0.85); the sand heap (sub-pixel
    # grains at 128 px) passes at 128 spp + OIDN (0.59 / 0.84; a flatter heap failed 64 spp at 128 px, 3.47).
    ("towel", "128x128"): Claim(128, "oidn"),
    ("towel", "640x480"): Claim(128, "oidn"),
    ("heap", "128x128"): Claim(128, "oidn"),
    ("heap", "640x480"): Claim(128, "oidn"),
    # Live SPH liquid (RENDER-GA-1, L4, OptiX): the pour (water stream into a cup) passes at 128 spp + OIDN, G1 0.80 /
    # 0.54, equal spp 10/10 at both sizes.
    ("pour", "128x128"): Claim(128, "oidn"),
    ("pour", "640x480"): Claim(128, "oidn"),
}


def parse(entry: dict) -> Claim:
    """A Claim from a matrix.json entry: ``spp`` plus ``kind``/``tta``/``weights`` or a ``denoiser`` text such as
    "learned, tta 2" or "OIDN".

    Example: ``parse({"spp": 64, "den": "on", "denoiser": "OIDN"})`` -> Claim(64, "oidn", 1).
    """
    text = str(entry.get("denoiser", "")).lower()
    kind = entry.get("kind") or ("oidn" if "oidn" in text else "learned")
    m = re.search(r"tta\s*(\d+)", text)
    tta = int(entry.get("tta") or (m.group(1) if m else 1))
    return Claim(int(entry["spp"]), str(kind), tta, entry.get("weights"))


def parse_claim(text: str) -> Claim:
    """A Claim from "SPP,KIND[,TTA[,WEIGHTS]]" (a candidate on the command line).

    Example: ``parse_claim("16,learned,1,c3_1280")`` -> Claim(16, "learned", 1, "c3_1280").
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) < 2 or not parts[0].isdigit() or parts[1] not in ("learned", "oidn"):
        from .errors import CameraError  # noqa: PLC0415
        raise CameraError(f"claim {text!r}", 'write SPP,KIND[,TTA[,WEIGHTS]] with KIND "learned" or "oidn", '
                          "e.g. 16,learned,1,c3_1280")
    tta = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    return Claim(int(parts[0]), parts[1], tta, parts[3] if len(parts) > 3 and parts[3] else None)


def _spec() -> dict | None:
    try:
        from bench.render_gates.matrix import load_spec  # noqa: PLC0415  (gate tool, not shipped with the camera)
        return load_spec()
    except (ImportError, OSError, ValueError) as exc:
        log.info("render-gate spec unavailable (%s); using the built-in claims", exc)
        return None


def claim_for(scene: str, res: str, spec: dict | None = None) -> Claim:
    """The claim for ``scene`` at ``res``: matrix.json's per-scene claim, else the table's per-scene claim, else the
    size's claim (matrix.json, then the table). ``spec``: a loaded matrix.json (default: read it when available)."""
    spec = _spec() if spec is None else spec
    own = (((spec or {}).get("scenes") or {}).get(scene) or {}).get("claims", {}).get(res)
    if own:
        return parse(own)
    if (scene, res) in BY_SCENE:
        return BY_SCENE[(scene, res)]
    per_size = ((spec or {}).get("claims") or {}).get(res)
    return parse(per_size) if per_size else BY_SIZE.get(res, Claim(12))
