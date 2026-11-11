# SPDX-License-Identifier: Apache-2.0
"""The launch scorecard: one command per row, `python -m alloy.bench.scorecard <row>` (docs/benchmarks/).

Every row writes the same files into its output folder: `rows.jsonl` (one `alloy.scorecard.row/1` record per
number), `headline.json` (the one line the scorecard page shows) and, for numbers that may not be published,
`rows.internal.jsonl` with a restricted-numbers header line. `page` builds the one-page scorecard from those folders.
"""

from alloy.bench.scorecard.types import Headline, Row, RowSpec

__all__ = ["Headline", "Row", "RowSpec"]
