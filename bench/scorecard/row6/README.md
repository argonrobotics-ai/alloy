# Row 6 data: coverage

`python -m alloy.bench.scorecard 6` reads the two files here and counts them. Nothing in this row is a simulation run.

- `capabilities.jsonl`: one record per (capability, engine) cell, 26 physics capabilities for Alloy and seven
  other engines. Fields: `status` (`yes`, `partial` (includes opt-in and experimental), `no`, `unknown`), `raw_value`
  (the survey's own wording), `evidence` (public source locations; for Alloy a repo path such as a test, a guide page
  or a known-gap page, with `:line` or `::test_name` where known), `evidence_commit` (the Alloy commit each Alloy
  location was read at), `evidence_note` (why `evidence` is short or empty), `caveat` (limits, known gaps).
- `ga_tasks.jsonl`: one record per Gym-Anything (GA) task, 200 in all. `verdict_class` is whether we read
  the physics classes the task needs as supported in Alloy: a reading of commits and tests, not a run of the task.
  No task here has been run and scored by Gym-Anything's own verifier. `alloy_env` names the scripted Alloy port
  where there is one (28 tasks); a port is judged by our own checks, k of n of the task's success gates.
  `verdict_public_package` is the same reading for the public package; `verdict_strict` is an older reading that
  is no longer quoted; `verdict_before_rescore_2026_11_06` is the reading before the re-score of 2026-11-06. `evidence_tier` runs from
  `A`, a port whose own checks judge all n gates (each with 1 to 6 by another measure than GA's), down to `E`, a
  class reading with no port. The
  `gates_*` fields count GA's success gates by how the port judges them (`gates_known: false` where the review
  has no gate table). `trials_note` says whether worlds are copies of one start. `not_public` names task modules
  that are not in the public package.

**Wording rule.** Every GA verdict is *Alloy's re-implementation of GA's criteria on Alloy's port (scripted
privileged actor); not run through GA's verifier.* A pass is not a pass of GA's benchmark and not a robot doing the
task. Where there is no port, the verdict is a class judgment, and says so.

**Rival cells are read, not tested.** A rival's `yes` means the feature exists in the engine's public code, docs or
papers. Alloy's own `yes` needs a passing test, so the grading is stricter for Alloy than for the others. No cell is
a run of another engine.

**Evidence.** Where a cell lists no evidence, the `evidence_note` says why; a rival cell with no recorded public
source reads `unknown`.

**Date.** The review of the 200 tasks is Alloy's, 2026-11-10; the capability cells were read 2026-11-09 (cells marked
"not re-read since 2026-11-05" are older).
