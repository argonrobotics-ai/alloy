# SPDX-License-Identifier: Apache-2.0
"""What ``alloy.il`` needs of an environment, checked once with an error that says what to pass instead.

Example::

    from alloy.il.support import require_vector_env
    require_vector_env(env, "Recorder")          # raises EnvError for a PortEnv or a Gymnasium wrapper
"""

from __future__ import annotations

from typing import Any

from alloy.envs.errors import EnvError

NEEDS = ("xp", "num_envs", "autoreset", "task")
ROD_ROUTE = "rod_vbd"


def require_vector_env(env: Any, who: str) -> None:
    """Raise ``EnvError`` unless ``env`` is an ``alloy.envs.VectorEnv`` (or steps like one).

    Example: ``require_vector_env(env, "generate")``."""
    missing = [name for name in NEEDS if not hasattr(env, name)]
    if not missing:
        return
    kind = type(env).__name__
    inner = getattr(env, "env", None)
    if inner is not None and all(hasattr(inner, name) for name in NEEDS):
        fix = (f"pass the VectorEnv inside the wrapper ({kind}.env) to {who}, and apply the wrapper's effect in "
               "your policy (a Recorder stores the actions it is given)")
    else:
        fix = (f"use an env built from a Task (alloy.envs.VectorEnv, e.g. alloy.envs.make('GripperLift-v0')); "
               f"for a {kind}, step it in your own loop and write the arrays yourself")
    raise EnvError(f"{who} needs an alloy.envs.VectorEnv, and got a {kind} (no {', '.join(missing)})", fix,
                   field="env")


def labels(env: Any, group: str | None) -> list[str]:
    """Column labels of one vector observation as the policy gets it: the env's labels, then every layer's
    (``ObsHistory`` repeats them per step). Example: ``labels(env, None)[:2]``."""
    names = list(env.obs_names(group) if group is not None else env.obs_names())
    for layer in getattr(env, "layers", ()):
        names = list(layer.names(group or "policy", names))
    return names


def unrestorable(env: Any, has_particles: bool) -> str | None:
    """Why a recorded start state cannot put ``env`` back (then ``replay`` refuses and ``Episode.states`` lacks
    that part), or None. Example: ``unrestorable(env, False)``."""
    if has_particles:
        return "this scene has cloth, liquid or other particles, whose start state a replay cannot restore"
    routes = getattr(env.sim, "routes", None)
    rods = [name for name, r in (routes() if callable(routes) else {}).items() if r.get("route") == ROD_ROUTE]
    if rods:
        return (f"{rods} step on the rod solver ({ROD_ROUTE}), whose segment poses are not in the joint state a "
                "recording keeps")
    writer = getattr(env, "writer", None)
    if writer is not None and int(writer.size) and not getattr(writer, "_world", None) and hasattr(writer, "_hist"):
        return ("this env's actions go through an actuator model (ActuatorSpec), whose held targets and delays a "
                "recording does not keep")
    return None


__all__ = ["labels", "require_vector_env", "unrestorable"]
