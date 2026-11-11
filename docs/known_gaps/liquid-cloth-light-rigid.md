# Cloth, liquid and a light free rigid body do not build in one scene

## What is wrong
A scene with a liquid, cloth (or a soft body) and free rigid bodies builds only when the rigid bodies are heavy
enough against the cloth. When the free rigid bodies and robot links together weigh less than twice the cloth and
soft bodies together, `alloy.build` raises `BuildError`, wherever the objects are, even far apart. The same scene
without the liquid builds, and so does the same scene without the cloth.

## Number
The rule is a mass ratio: cloth and soft mass over rigid mass above 0.5 is refused. Example: a 20 g sheet, water in
a static tub and a free ball of 34.9 g (radius 2 cm, default material) 1.4 m away: ratio 0.57, refused; the error
gives 40.1 g as the lightest ball that builds. With a 100 g box and a 10 g sheet (ratio 0.1) the scene builds and
steps.

## Why
A rigid body that is light against the cloth it could touch needs the strongly coupled solver pair: the cloth's
load on the body is too large for the lagged exchange that heavier bodies use. That pair cannot step a liquid in
the same scene yet. The test is by mass only, not by distance, so a light body is refused even where it could never
reach the cloth.

## Workaround
The error's `fix` names each of these with its number:
- make the free rigid objects heavier, to at least twice the cloth and soft bodies in all (`mass=`);
- make the light objects `static=True` when they do not need to move, or `kinematic=True` to move them yourself
  (`sim.drive`);
- make the cloth lighter;
- build the liquid in a scene of its own.

## Planned fix
The strongly coupled pair beside a liquid, or the lagged pair for light bodies that cannot reach the cloth. No date
yet.
