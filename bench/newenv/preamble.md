# New-environment task for {engine_name}

You are building a new simulation environment, from the engine's public documentation only, as a cold start.

**Engine.** Use **{engine_name}** and no other physics engine. {engine_install}
Documentation: {engine_docs}.

**Time.** {agent_minutes} minutes in total. The engine is already installed (above), so the time is for building and checking the environment. The harness then re-runs `~/task/run.sh` once
from a clean state, capped at {rerun_minutes} minutes; only that re-run is scored.
