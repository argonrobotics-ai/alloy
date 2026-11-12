# A surface film is one-way with what it does not declare

**What.** A surface film (the `SurfaceFlow` plug-in) couples only to what its config names:
- the bodies its charts ride;
- its tools (`Applicator`): the underside ploughs the film and the pad presses back;
- its lumped films' bodies;
- with `sph=True`, the SPH liquid its drops join.

Anything else passes through the film as if it were not there, and feels nothing from it. That covers every other
colliding body (a dropped block, a robot's links, a pusher) and other particles (cloth, paper, paste, soft bodies,
SPH liquid without `sph=True`). The film also keeps its depth under them: a block resting in a puddle neither
displaces the water nor floats on it, and a cloth laid on a wet plate does not get wet.

**Number.** The material sweep (the material-combination matrix, finding F3, not shipped), 1 mm of water on a plate:
- a block, cloth, paper, sponge, paste, water or the pusher comes down onto the plate as if the water were not
  there: the block's bottom lands at the plate top to 0.1 mm, and the cloth at its bare-plate height;
- every cell keeps its depth;
- the film's ledger stays exact.

**Why.** The film is a height field on charts. It reads the geometry of declared tools (`tool_geometry`), applies
their pad forces (`pad_implicit`) and books what they carry. It has no contact query against the rest of the scene,
and other solvers do not see it.

**Now.** Setup warns once, naming the undeclared colliding bodies and the particle count
(`alloy/plugins/surface_flow/oneway.py`; `flow.one_way` holds the text, None when nothing passes through).

**Workaround.**
- To couple a rigid body (a sponge's holder, a scraper, a robot's tool), declare it as a tool: `Applicator(body=...,
  shape="rect" | "disc" | "roller")`, with a `Store` if it should pick up or lay film. The film then sees its
  underside and pushes back on it.
- SPH liquid landing on a film joins it with `sph=True`.
- For a porous sheet that should soak up a film, model the sheet as the chart, with `absorption`.
- Cloth, paper, MPM paste and soft bodies have no workaround.

**Planned fix.** Not in v1, which warns instead. The two-way fix, estimated:
1. **Rigid bodies and robot links, about one week.** Every colliding shape of an
   undeclared body becomes an implicit ceiling over the charts it overlaps (boxes, spheres, capsules and cylinders
   exactly; meshes sampled by SDF), with the pad's pressure and drag back on the body. Tools already work this way,
   so the step is: shape queries for 2-3 days, the force and its stability bound for 1-2 days, and laws (a block
   displacing a puddle, a floating board, lubricated sliding) for 1-2 days.
2. **Particles (cloth, paper, MPM, soft, SPH without `sph=True`), 2-3 weeks across owners (VBD, MPM, SPH).** Each
   solver would need:
   - the film displaced by its particles' footprints;
   - the film's forces on the particles (drag, capillary adhesion);
   - the water a cloth or paper takes up;
   - a stability bound per solver pair.

   Not in v1.
