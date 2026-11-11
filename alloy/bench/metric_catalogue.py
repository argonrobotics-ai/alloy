# SPDX-License-Identifier: Apache-2.0
"""The SPEC §5 metric catalogue as data: id, unit, target, producer. Deferred ids never enter the release scoreboard.

Targets are copied from SPEC v1.1 §5.1-5.2 (proposals until the first bench numbers, then frozen by SCR).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricDef:
    id: str
    unit: str
    target: str
    producer: str


_ROWS = """
SB-1|1|r >= 0.8, MMRV <= 0.10|policy suite (phase 2)
SB-2|1|forces <= 10 %, precision geometry <= 0.1 mm; R1-R3 100 %, R1-R9 >= 80 %|rig suite
SB-3|m|p99 <= max_penetration, max <= 2 max_penetration|CT-1 (WP3)
SB-4|1|0 pass-through events|CT-2, LIQ-8
SB-5|1|T12 >= 30x RT on one L4; T09 >= 5x|speed ladder (WP8)
SB-6|1|>= 1x RT (T12)|speed ladder PAR-8 (WP8)
SB-7|1|upper bound <= 1e-5|bully corpus (WP7)
SB-8|1|<= 1 %/sim-min, never > 5 % in 1 s|CPL-4 (WP4)
SB-9|1|>= 80 %|CARD-1 (WP2)
SB-10|USD|<= $3 per million sim-s|speed ladder PAR-15 (WP8)
SB-11|1|100 %|PRC-4 (WP5)
SB-12|1|100 % at release (merge gate: no regression vs baseline)|alloy.bench run --suite analytic (WP6)
CARD-1|1|>= 80 %|WP2
CARD-2|1|>= 95 % agreement|WP2
CARD-3|1|0|WP2
CARD-4|1|<= 1e-6|WP2
CARD-5|1|100 %|WP1
CARD-6|1|20/20|WP2
CARD-7|1|<= 5 % at 30 links|WP2
CARD-8|1|<= 15 %|WP2
CARD-9|1|<= 10 %|WP2
CARD-10|1|<= 2 %|WP1
CARD-11|s|<= 60 s; <= 10 ms per object|WP2
CARD-12|1|>= 0.9x|WP2
CT-1|m|<= max_pen / 2 max_pen|WP3
CT-2|1|0|WP3
CT-3|1|<= 10 %; +/- 0.5 mm|R1 (WP3)
CT-4|1|1.5 mm +/- 1 %; <= 10 %|A9 + R2 (WP3)
CT-5|m|0.100 +/- 0.005 mm; <= 5 %|WP3
CT-6|1|<= 1.05; <= 0.05 mm|WP3
CT-7|m|<= 1 um; <= max(2 um, 0.05 voxel)|WP3
CT-8|1|within 20 %|WP3
CT-9|1|<= 2x|WP3
CT-10|MB|<= 64 MB / <= 1 GB|WP3
CT-11|1|bitwise|WP3
CT-12|1|0 dropped|WP3
CT-13|1|0|WP3
CPL-1|1|median <= 1 %, p99 <= 5 %|WP4
CPL-2|m|<= 1 mm/s, 0.1 mm (precision); <= 1 cm/s, 1 mm|WP4
CPL-3|1|<= 1 %|WP4
CPL-4|1|<= 1 %; <= 5 %|WP4
CPL-5|1|0; 100 %|WP4
CPL-6|1|>= 5x where f <= 5 % at ratio 8|WP4
CPL-7|1|<= 2.2; <= 10 %|WP4
CPL-8|1|>= 3x; <= 0.5 mm|WP4
CPL-9|1|bitwise|WP4
CPL-10|1/s|<= 1|WP4
PRC-1|m|<= 0.01 mm / 10 min|WP5
PRC-2|m|<= 0.05 mm|WP5
PRC-3|1|<= 2x, permutation p >= 0.05|WP5
PRC-4|1|100 %|WP5
PRC-5|1|100 %|WP5
PRC-6|1|equal|WP5
PRC-7|1|<= 1.3x time, <= 1.5x memory|WP5
PRC-8|m|<= 1 um|WP5
PRC-9|1|100 %|WP5
PRC-10|1|0|WP5
PRC-11|1|<= 1.5x (A), <= 2x (B)|WP5
PRC-12|1|0; flagged|WP5
PRC-13|1|100 %; 0|WP5
PRC-14|1|reported; spread <= 1-ulp-nudge spread|WP5
PAR-1|1|<= 5 %|WP8
PAR-2|1/s|>= 1.6 M world-steps/s|WP8
PAR-3|1|>= 30x RT|WP8
PAR-4|1|>= 5x RT|WP8
PAR-5|1/s|>= 25 k world-steps/s; <= CREEP_GENERAL|WP8
PAR-6|1|>= 100 cable-s/s|WP8
PAR-8|1|>= 1x|WP8
PAR-9|1|<= 130|WP8
PAR-10|1|>= 90 %|WP8
PAR-11|1|+/- 15 %; 0|WP8
PAR-12|1|>= 70 %|WP8
PAR-13|1|>= 95 %|WP8
PAR-14|1|0|WP8
PAR-15|USD|<= $3 / M sim-s|WP8
LIQ-1|1|<= 1e-6|WP9
LIQ-2|1|<= 0.5 %|WP9
LIQ-3|1|<= 0.1 % / <= 1 %|WP9
LIQ-4|1|<= 3 %; <= 5 %|A13 (WP9)
LIQ-5|1|NRMSE <= 0.1|A14 (WP9)
LIQ-6|1|<= 5 %|WP9
LIQ-7|1|<= 5 %; rig +/- 2 g|R7 (WP9)
LIQ-8|1|0|WP7/WP9
LIQ-9|1|>= 0.5x RT; >= 50 world-s/wall-s|WP9
LIQ-10|1|<= 3 mm/s; no growth|A13.resting_jitter (WP9)
TOP-1|1|<= 1e-6|WP10
TOP-2|1|<= 15 %|WP10
TOP-3|1|0, 0; counted; unchanged|WP10
TOP-4|1|<= 10 %|WP10
TOP-5|1|<= 1e-6 / <= 2 %|WP10
ROD-1|1|<= 5 %|WP2
ROD-2|1|<= 2 %; <= 5 %|WP2
ROD-3|s|<= 1 ms; <= 2 ms|upstream (deferred)
ROD-4|1|>= 1,000 cable-s/s|WP8
ROD-5|1|0 link crossings|WP2
BEN-1|1|100 %|A16 (WP6)
BEN-2|1|(RE(valid) - RE(calib)) / bound <= 0.5|rig suite (WP6)
BEN-3|m|<= 0.01 mm / 10 s precision, <= 0.1 mm / 10 s else|A5 / R3 (WP6)
BUL-1|1|0; none|WP7
BUL-2|1|0|WP7
FAC-1|1|reported|WP7
FAC-2|1|reported|WP7
FIT-1|1|within the law's tolerance|WP11
FIT-2|s|<= 1 h|WP11
FIT-3|1|mu +/- 2 %, k +/- 10 %|WP11
"""

# SPEC §5 "Deferred metrics"; SB-2 is deferred while no real rig dataset exists (decided at scoreboard time).
DEFERRED: frozenset[str] = frozenset({"SB-1", "PAR-2", "PAR-13", "PAR-5", "PAR-6", "ROD-3", "PAR-9", "PRC-2",
                                      "PRC-8", "TOP-2"})

METRICS: tuple[MetricDef, ...] = tuple(MetricDef(*line.split("|")) for line in _ROWS.strip().splitlines())
BY_ID: dict[str, MetricDef] = {m.id: m for m in METRICS}

__all__ = ["MetricDef", "METRICS", "BY_ID", "DEFERRED"]
