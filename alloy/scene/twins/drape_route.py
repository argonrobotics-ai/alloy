# SPDX-License-Identifier: Apache-2.0
"""R6 Cusick drape on the production route (SCR-WP2-14, coordinator ruling 2026-11-08 14:59Z (3)).

R6 (``alloy.scene.twins.drape.LAW``: DC within 0.05, node count within 1) is judged on measured data: Petrak et al.
2021's woven fabrics, the D03M pack (``bench/scenarios/packs/fabric_drape_petrak2021.json``) written as the rig dataset
``bench/rigs/R6.cloth_drape/petrak2021-v1`` (``write_measured``). The twin drapes them the way a user's KES card is
built and routed today (``fabric_rig``'s headline arm L, CLOTH_VBD), through ``alloy.Scene`` and ``alloy.build`` at
``cloth_accuracy="super"``, with D03M's rig (``fabric_run.run_rigs``). So R6 tests what users get, not a hand-built
solver. Each world's window-mean DC is one sample; ``alloy.bench.rigjudge.judge_rig`` judges them against each fabric's
measured DC. The paper reports no node counts, so those rows are ``not_run``.

The DC uncertainty (ruling 14:59Z (2)): Petrak gives each DC as a mean of 5 specimens with no spread, so the
instrument's u95 comes from a sourced Cusick repeatability, ``U95`` (DC5's search, notes/wp2-fabric/sources.md): the
largest specimen SD in Toshikj & Mladenovikj 2025's Cusick table, as the u95 of a 5-specimen mean. It is 0.012
against the 0.05 bar (TUR 4.1, above the judge's 3). A remount of one specimen can move DC more on stiff fabrics
(``U95_NOTE``). Without a source the instrument carries ``U95_LABEL`` and no u95 (no TUR, never "inconclusive").

The synthetic strip-elastica set and the raw-Newton ``DrapeTwin`` (``alloy.scene.twins.drape``) stay as a CPU
regression fixture with no bar (ruling 10:56Z (3)).
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.types import Condition, RigCard, RigDataset, RigTrial
from alloy.core.errors import LawError
from alloy.scene.twins import fabric_rig as FR
from alloy.scene.twins.drape import LAW

log = logging.getLogger("alloy.scene")

REPO = Path(__file__).resolve().parents[3]
PACK = REPO / "bench" / "scenarios" / "packs" / "fabric_drape_petrak2021.json"
DATASET_ID = "petrak2021-v1"
DATASET = REPO / "bench" / "rigs" / LAW.id / DATASET_ID
U95_LABEL = "u95 not reported by the source; bar 0.05 assumed from repeat-test scatter"
#: largest 5-specimen SD of DC on a Cusick meter, 18 cm disc, BS 5058 (Table 9: 0.55 / 0.50 / 0.98 %), as the u95 of
#: a 5-specimen mean: t95(4) x 0.0098 / sqrt(5) = 0.0122
U95 = 2.776 * 0.0098 / 5 ** 0.5
U95_SOURCE = ("Toshikj & Mladenovikj 2025, Textiles 5(4):63, doi:10.3390/textiles5040063 (CC BY), Table 9: Cusick "
              "drape meter, 18 cm disc, BS 5058:1974, 5 replicate specimens, DC SD 0.50-0.98 %; u95 of a 5-specimen "
              "mean = t95(4) x 0.0098 / sqrt(5). Table read from web.archive.org snapshot 20251205035025")
U95_NOTE = ("contrary: Matsudaira & Yang 1997, doi:10.4188/transjtmsj.50.9_T242 (FRL-type tester, not Cusick): "
            "remounting one specimen moved DC 0.034-0.136 on stiff fabrics as the fold count changed")
FABRIC_KEYS = ("areal_mass_kg_m2", "thickness_T0_m", "bending_warp_N_m", "bending_weft_N_m", "shear_G_N_m",
               "tensile_strain_warp", "tensile_strain_weft", "tensile_load_N_m")
#: (worlds, mesh edge [m], seconds, mesh seed): the full run is D03M's headline protocol on the R6 mesher's seed 7 (8 mm
#: seed 0's shortest edge, 2.418 mm, is under F9's 2.42 mm sheet, so it would start in self-contact; seed 7: 2.87
#: mm); quick is a CPU plumbing smoke (3 cm cells, 0.3 s), not a verdict
PROTOCOL = {False: (8, 0.008, 4.5, 7), True: (1, 0.03, 0.3, 0)}


def measured_dataset(pack: dict, u95: float | None = U95, u95_source: str = U95_SOURCE,
                     note: str = U95_NOTE) -> RigDataset:
    """The R6 rig dataset from a D03M pack: one condition per fabric (its SI inputs as params), one valid trial per
    fabric (the measured DC; the source's mean of its specimens). ``u95`` only with a cited source; None writes
    ``U95_LABEL``."""
    inst: dict[str, Any] = {"what": "Cusick drape meter: " + "; ".join(pack.get("source_notes", [])[2:3])}
    if u95 is None:
        inst["note"] = U95_LABEL
    else:
        inst.update(u95=float(u95), source=u95_source, note=note)
    rig = pack["rig"]
    card = RigCard(law_id=LAW.id, dataset_id=DATASET_ID, protocol_version="petrak2021/1",
                   objects={"specimen": {"radius_m": 0.5 * rig["specimen_diameter_m"],
                                         "support_radius_m": 0.5 * rig["support_diameter_m"]}},
                   instruments={"drape_coefficient": inst}, holder={}, environment={})
    conds = tuple(Condition(f["id"], {k: float(f[k]) for k in FABRIC_KEYS}) for f in pack["fabrics"])
    trials = tuple(RigTrial(f["id"], 0, "valid", {"t": np.zeros(1)}, {"drape_coefficient": float(f["dc_measured"])})
                   for f in pack["fabrics"])
    sha = hashlib.sha256(json.dumps({"pack": pack["id"], "u95": u95, "source": u95_source}, sort_keys=True)
                         .encode()).hexdigest()
    return RigDataset(card, conds, trials, sha)


def write_measured(root: Path = REPO / "bench" / "rigs", **kw: Any) -> Path:
    """Write ``measured_dataset(PACK)`` under ``root`` (immutable: refuses an existing dataset id)."""
    from alloy.bench.rigs import write_rig  # noqa: PLC0415

    return write_rig(measured_dataset(json.loads(PACK.read_text()), **kw), root)


def dataset(path: Path = DATASET) -> RigDataset:
    """The measured R6 dataset (sha256-checked by ``load_rig``)."""
    from alloy.bench.rigs import load_rig  # noqa: PLC0415

    if not path.exists():
        raise LawError(f"R6: no measured dataset at {path}; write it with alloy.scene.twins.drape_route.write_measured")
    return load_rig(path)


def _pack_of(data: RigDataset) -> dict:
    """The rig's pack (fabric_rig's input) from the dataset: the fabrics' inputs and measured DCs, the geometry."""
    dc = {t.condition_id: t.scalars["drape_coefficient"] for t in data.split("valid")}
    obj = data.card.objects["specimen"]
    return {"id": f"{LAW.id}/{data.card.dataset_id}",
            "rig": {"specimen_diameter_m": 2.0 * obj["radius_m"], "support_diameter_m": 2.0 * obj["support_radius_m"]},
            "fabrics": [{"id": c.id, **dict(c.params), "dc_measured": dc.get(c.id)} for c in data.conditions]}


class DrapeRouteTwin(BaseTwin):
    """R6's twin: the measured fabrics on the production route (module docstring)."""

    law_id = LAW.id
    units = {"drape_coefficient": "1", "node_count": "1"}

    def __init__(self, data: RigDataset | None = None, arm: str = FR.HEADLINE, accuracy: str = "super") -> None:
        super().__init__()
        self.data, self.arm, self.accuracy = data, arm, accuracy

    def settings_record(self, solver: str) -> dict[str, Any]:
        """The solve the route runs, for the bench report (BaseTwin's default names the raw twin's solver)."""
        from alloy.scene import cloth_accuracy as CA  # noqa: PLC0415

        return {"solver": "vbd", "route": "cloth_vbd", "arm": self.arm, "cloth_accuracy": self.accuracy,
                "kwargs": dict(CA.KWARGS[self.accuracy]), "protocol": dict(zip(("worlds", "edge_m", "seconds", "seed"),
                                                                             PROTOCOL[False]))}

    def evaluate(self, ctx) -> tuple:
        from alloy.bench.rigjudge import judge_rig  # noqa: PLC0415
        from alloy.scene.twins.fabric_run import RunOpts, run_rigs  # noqa: PLC0415

        data = self.data or dataset()
        pack = _pack_of(data)
        worlds, edge, seconds, seed = PROTOCOL[bool(ctx.quick)]
        rigs = FR.rigs_for(pack, self.arm)
        with tempfile.TemporaryDirectory(prefix="alloy-r6-") as tmp:
            opts = RunOpts(out=Path(tmp), worlds=worlds, device=ctx.device, accuracy=self.accuracy, edge=edge,
                           seconds=seconds, seed=seed)
            rows, per_world = run_rigs(pack, rigs, opts)
        values = {"drape_coefficient": np.concatenate([per_world[r.name]["dc"] for r in rigs]),
                  "node_count": np.concatenate([per_world[r.name]["nodes"] for r in rigs])}
        conds = [r.fab["id"] for r in rigs for _ in range(worlds)]
        verdicts, calib = judge_rig(LAW, self, values, data, conds, None)
        self.monitors.update({"rows": rows, "worlds": worlds * len(rigs), "arm": self.arm, "accuracy": self.accuracy,
                              "protocol": {"worlds": worlds, "edge_m": edge, "seconds": seconds, "seed": seed},
                              "finite": all(r["finite"] for r in rows), "check_ok": all(r["check_ok"] for r in rows),
                              "calib_error": calib, "dataset": data.card.dataset_id})
        log.info("R6 route twin: %s", {r["fabric"]: round(r["dc_mean"], 3) for r in rows})
        return verdicts


__all__ = ["DATASET", "DrapeRouteTwin", "LAW", "PACK", "U95", "U95_LABEL", "U95_NOTE", "U95_SOURCE", "dataset",
           "measured_dataset", "write_measured"]
