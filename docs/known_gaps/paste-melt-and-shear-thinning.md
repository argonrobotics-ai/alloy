# A material cannot both melt and shear-thin

**What is wrong.** The thermal plug-in writes a melting material's MPM viscosity (its melt law, from
`alloy/plugins/thermal/phase_materials.py`) before each solver step. The Herschel-Bulkley update rewrites the
viscosity of every particle with `fluid.consistency` inside the step. On a material with both, the melt law is lost
without a warning.

**The number.** No library material has both today (dark_chocolate and porridge_oat melt, none of them has K and n);
`tests/solvers/mpm_rheology/test_herschel_bulkley.py::test_no_library_material_both_shear_thins_and_melts` keeps it
that way. A custom card with both would run with the HB viscosity only.

**Why.** Both features write the same per-particle array, `model.mpm.viscosity`, and there is no rule yet for
combining a temperature factor with a shear-rate law.

**Workaround.** Give a melting material its melt law only (no `consistency` / `flow_index`), or a shear-thinning
paste no melt law.

**The fix and when.** When the first material needs both (melted chocolate is shear-thinning): scale K by the thermal
factor inside the Herschel-Bulkley kernel, or refuse the combination at build. Not scheduled.
