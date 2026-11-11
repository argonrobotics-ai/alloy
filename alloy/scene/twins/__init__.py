# SPDX-License-Identifier: Apache-2.0
"""WP2 rig-law twins for soft media (bench §4.4): R5 squeeze, R6 drape, R8 granular; R9 rope lives in `rope.py`.

Each module exports `LAW` (a `LawSpec`), a twin class run by `alloy.bench.runner.run_law`, and `dataset(seed)`, the
synthetic `RigDataset` it is tested against until real rig data exist (SPEC §6.4, protocol "synthetic/1").
"""
