# SPDX-License-Identifier: Apache-2.0
"""Material fitting (WP11, SPEC §3.12): fit card values (friction, restitution, stiffness, ...) from rig recordings.

Use: build a `FitProblem(law, twin, data, params)` with a fresh fit twin (`alloy.fit.twins`: R3 `FitInclineTwin`,
R4 `FitBounceTwin`, the stiffness `SqueezeTwin`; any twin with `fit_fields`, `bind_values`, `fix_values`), call
`fit(problem, FitConfig(...), device)`, then `write_back(result, CardLibrary(), reason=...)`. GPU scale:
`FitConfig(worlds=1024)` (candidate-conditions per batch); CPU tests: `quick=True` and small `worlds`.

How it works:
- `problem.prepare` reads the `calib` split once into targets; the search never holds the dataset. `validate` is
  the only reader of `valid` trials, through WP6's `run_law` with the fitted values fixed (split enforced in code).
- `batch.TwinBatch` runs one candidate x condition per world group (virtual conditions `"<cond>~<k>"`), one Newton
  model per chunk of `FitConfig.worlds` candidate-conditions; `objective.score` sums squared normalised errors.
- `search.run_cma` drives `cma.CMAES` (unit box, IPOP restarts, wall budget, `tol_x` stop). `method="hybrid"` adds
  `refine` (Gauss-Newton with batched finite-difference Jacobians and a Jacobian 95 % interval in the report).
  `method="grad"` (Newton autodiff) runs CMA-ES with the reason recorded: at this pin Newton's rigid-contact tape
  gradients fail the finite-difference check (`grad.py`, [internal note]).
- `writeback.write_back` refuses without held-out evidence or with a gap RE(valid) - RE(calib) > 0.5 x bound, and
  writes `Q(source="fit", unc, note=<fit id ...>)` through `CardLibrary.put`.
Errors: FitError, SplitError, HeldOutGapError from alloy.core.errors (SCR-WP11-1; re-exported by alloy.fit.errors).
"""

from alloy.fit.fitter import fit
from alloy.fit.types import FitConfig, FitParam, FitProblem, FitResult
from alloy.fit.writeback import write_back

__all__ = ["fit", "write_back", "FitConfig", "FitParam", "FitProblem", "FitResult"]
